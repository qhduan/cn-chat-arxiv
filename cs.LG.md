# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning to Stop without Learning to Stop: Self-Supervised Confidence Training Improves Reasoning Efficiency](https://arxiv.org/abs/2609.31619) | 该论文发现，仅通过自监督方式训练模型在推理过程中间点预测自身置信度（损失函数中不含任何长度、效率或停止目标），就能让模型在推理时无需任何提前停止机制便自发提升推理效率。 |
| [^2] | [Gap-free Differentially Private PCA for Gaussian Data](https://arxiv.org/abs/2609.31614) | 该论文提出了一种无间隙的差分隐私算法，用于解决高斯数据下的主成分分析（PCA）问题。 |
| [^3] | [First-Order Stationarity of Reverse Diffusions](https://arxiv.org/abs/2609.31612) | 该论文为扩散模型建立了首个一阶平稳性理论，证明基于SDE的过阻尼与欠阻尼朗之万扩散的逆时流在前向过程平稳势强凸（仅针对加噪过程而非数据）时以指数速率收缩相对Fisher散度，并为离散化采样器建立了与非凸优化中平均梯度范数保证相对应的一阶平稳性界。 |
| [^4] | [Statistical attribute alignment for black-box generative AI via output post-processing](https://arxiv.org/abs/2609.31607) | 本文针对黑盒生成式AI提出了一种输出后处理方法，通过最小化查询次数的算法将生成输出的属性分布与用户指定目标对齐，并在精确与近似对齐两种情形下证明了算法的最优性。 |
| [^5] | [User Model Extraction via Belief Self-Distillation](https://arxiv.org/abs/2609.31603) | 提出信念自蒸馏框架，使冻结的大语言模型从自然对话中自我蒸馏出既能读出又能写回的用户信念表示，并揭示模型的拒绝行为取决于其推断出的用户意图。 |
| [^6] | [New LoRA Skills Should Read but Never Write](https://arxiv.org/abs/2609.31600) | 提出READ方法，通过将各适配器重写为规范化形式并强制新技能对旧技能“只读不写”的耦合方向，使多个独立训练的LoRA适配器能够组合成单一模型而不破坏旧技能的原有计算。 |
| [^7] | [Common-Mode Collapse and Recovery in Direct Feedback Alignment](https://arxiv.org/abs/2609.31589) | 该研究揭示了直接反馈对齐（DFA）训练停滞的根源在于误差的共模分量通过秩一更新驱动tanh单元饱和，并构建了一个无需拟合参数的简化模型来准确预测这种崩溃现象及其恢复动态。 |
| [^8] | [Trust Guided Decision Transformer](https://arxiv.org/abs/2609.31586) | 提出信任引导决策Transformer（TGDT），利用模型自身经保形预测校准的下一状态预测误差先筛选可信上下文、再由冻结评论家选择最高价值动作，以避免长回合 rollout 中上下文漂移导致的性能退化。 |
| [^9] | [Uncertainty and Explainability in Deep Rough Volatility: A Neural Information-Theoretic Posterior Approach](https://arxiv.org/abs/2609.31570) | 该论文提出一个基于仿真的推断框架，通过神经比率估计学习粗糙Heston模型参数在隐含波动率曲面条件下的后验分布，结合异方差神经代理定价器生成考虑不确定性的价格区间，并引入信息论可解释性方法Hellinger-SHAP。 |
| [^10] | [Weight Pair Encoding: Inducing a Smaller Grammar in Neural Network Weights](https://arxiv.org/abs/2609.31564) | 提出WeightPE方法，通过将有损Re-Pair压缩器嵌入直通估计器中来显式微调网络权重使其形成更小的语法结构，在ViT模型上产生的语法大小仅为int8 QAT的约0.4倍，且准确率损失仅1-2个百分点。 |
| [^11] | [Generalization behavior of OPTQ and the role of regularization](https://arxiv.org/abs/2609.31560) | 该论文在泛化设定下分析了 OPTQ 及其随机变体量化算法，证明了测试分布下期望量化误差的界，并揭示了正则化项在控制泛化误差中的关键作用。 |
| [^12] | [Online Learning via Learned Latent Bayesian Tracking](https://arxiv.org/abs/2609.31559) | 提出AURA元学习框架，通过离线学习低维潜在状态空间模型，解决了基于贝叶斯滤波的在线学习中缺乏合适低维动力学表示的核心瓶颈，从而实现非平稳环境下模型的快速在线适应。 |
| [^13] | [EAServe: Encode-Aware Disaggregated Serving for Multimodal Large Language Models](https://arxiv.org/abs/2609.31551) | EAServe提出了一种编码感知的分离式服务框架，将编码阶段重新定位为EPD流水线的控制点，从而解决多模态大语言模型服务中编码GPU利用率低下及下游预填充与解码资源饥饿的结构性失衡问题。 |
| [^14] | [BeatGraph: Self-Supervised Heartbeat Graphs for Infant ECG Representations from the Home Environment](https://arxiv.org/abs/2609.31546) | BeatGraph提出以心跳为基本单元的自监督图表示方法，将30秒婴儿ECG窗口建模为心跳图，克服了固定片段切分忽略心脏结构的问题，更适合婴儿高心率及家庭环境下的心电图建模。 |
| [^15] | [A Flow Matching Framework for Neural Representational Dissimilarity](https://arxiv.org/abs/2609.31544) | 本文提出用深度生成模型中的流匹配框架统一多种神经表征差异性度量（即将其归结为不同速度约束下的Jeffreys散度），该框架在处理复杂分布和连续变量时具有估计优势，并能支持以有原则的方式设计新的距离度量。 |
| [^16] | [NEXT: Physics-Informed Neuro-Spectral Exponential Time Differencing Architectures](https://arxiv.org/abs/2609.31539) | NEXT架构通过将NeuSA的谱表示与高阶指数积分器相结合，利用矩阵指数精确积分线性刚性部分、用神经网络建模非线性剩余部分，从而解决了神经谱方法在刚性偏微分方程上的数值不稳定问题。 |
| [^17] | [HySTAR: Anchored Hypergraphs for Stable Credit Assignment in Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2609.31531) | 提出HySTAR框架，通过锚定重叠稀疏超图作为时间一致的高阶价值分解基础，解决协作多智能体强化学习中的结构性目标漂移问题，实现稳定的个体与高阶联盟信用分配。 |
| [^18] | [Retrainable physics-integrated neural differentiable modeling of sintering across material systems](https://arxiv.org/abs/2609.31518) | 提出可重训练的物理融合神经可微分框架Sinter-PiNDiff，通过神经网络在耦合速率方程中学习致密化与晶粒生长系数，在多种氧化物材料的留出温度和成分测试中全面优于基线模型，实现了跨材料体系的烧结过程精确预测。 |
| [^19] | [Retail Product Search: A Practical Approach at Target](https://arxiv.org/abs/2609.31498) | Target公司提出了一种结合词法搜索与向量搜索的混合式零售产品搜索系统，通过数据处理、嵌入训练、精度控制和多通道加权结果融合等实用方案，在保证低延迟的同时平衡相关性、收入与利润等多重目标。 |
| [^20] | [Scaling Density Functional Theory with Gaussian Splatting](https://arxiv.org/abs/2609.31483) | GS-DFT 将分子轨道表示为高斯函数云，通过梯度下降联合优化其位置、形状和混合系数，无需训练数据即可实现密度泛函理论计算随系统规模的可扩展性，并达到最大常规基组的精度。 |
| [^21] | [Beyond Empirical Support: Structured Outlier Generation via Sinkhorn Optimal Transport](https://arxiv.org/abs/2609.31470) | 该论文提出SBOG框架，将Sinkhorn最优传输几何与分布鲁棒边界建模相结合，在潜空间中结构化地生成弱支撑边界区域的离群点，从而更有效地评估和提升机器学习系统应对分布偏移的鲁棒性。 |
| [^22] | [LandscapeSHAP: Which Persistent Homology Class Gets the Credit?](https://arxiv.org/abs/2609.31469) | 本文提出了首个将 Shapley 值应用于拓扑数据分析特征的可解释性方法 LandscapeSHAP，能够将模型预测的贡献公平分配到持续图中的各个持续同调类，并对持续景观上的线性模型给出精确的闭式解。 |
| [^23] | [Scaffold: Support Graph Theory Based Sparsification for Graph Neural Networks](https://arxiv.org/abs/2609.31466) | Scaffold是一个源自支撑图理论的无监督图稀疏化框架，通过联合控制扩张度（重路由路径长度）和拥塞度（重路由路径集中程度）两个结构量，在保留短通信路径的同时避免结构性瓶颈，从而在降低GNN计算与内存成本的同时保持预测性能。 |
| [^24] | [Uncertainty-Aware Federated Learning for Infant Movement Analysis](https://arxiv.org/abs/2609.31463) | 提出了首个面向婴儿运动分析的不确定性感知联邦学习框架，能够在保护隐私的前提下利用多机构骨骼运动数据实现自动化全身运动评估。 |
| [^25] | [Nonparametric In-Context Learning under Growing Geometric Complexity: Minimax Optimality and Local Geometry-Adaptivity of Transformers](https://arxiv.org/abs/2609.31458) | 本文在由随样本量增长、维度与光滑度异构的流形混合所刻画的未知局部几何下研究非参数上下文学习，建立了极小极大最优下界，并证明配备几何预条件器的结构感知两阶段softmax Transformer能达到该最优性且可自适应局部几何。 |
| [^26] | [Different Corruptions, Different Signals: Uncertainty and Loss in Federated Data Quality](https://arxiv.org/abs/2609.31454) | 本文比较了联邦学习中输入条件不确定性与预测-标签损失两种损坏检测信号，发现在非独立同分布数据条件下，这两种信号对输入噪声和标签翻转两类损坏表现出不同的检测效果。 |
| [^27] | [Implicit Neural Representation for Hyperspectral Video Compression](https://arxiv.org/abs/2609.31435) | 该论文提出了一种基于隐式神经表示的高光谱视频压缩新方法，通过对现有RGB视频压缩模型进行新颖扩展，相比传统逐帧压缩方法实现了+4.99 dB的PSNR增益和-88.88%的码率降低，同时显著提升了下游目标跟踪任务的性能。 |
| [^28] | [Evaluating the accuracy of KV cache reuse techniques](https://arxiv.org/abs/2609.31415) | 该论文指出现有KV缓存重用技术的评估方法会人为夸大其有效性，并提出了一种无歧义测量精度损失的评估方法，同时发布了Boxoffice工具来生成具有挑战性重用模式的评估数据集。 |
| [^29] | [AFA-Net: A Differential Attention Approach for Auditory Attention Detection](https://arxiv.org/abs/2609.31402) | AFA-Net通过差分注意力机制显式对抗EEG噪声，在2秒决策窗口下以远少于现有方法的参数量实现了96.8%的听觉注意力检测准确率。 |
| [^30] | [Decodable In-Context State and Model Output Across Training](https://arxiv.org/abs/2609.31401) | 本文首次在Pythia模型的预训练与后训练各检查点上系统跟踪探针可解码性、模型输出与探针引导转向的演化，发现二者随训练同步提升，并通过信息论反例证明仅在错误样本上的可解码性不足以证明模型丢弃了输出信息。 |
| [^31] | [Differential Attention Unlocks Complementary EEG and Speech Fusion for Emotion Recognition](https://arxiv.org/abs/2609.31399) | EmoSpeechBrain 通过差分注意力抑制脑电噪声、并利用门控适配器融合脑电与语音两种模态，在情感识别任务上将准确率相比现有最先进方法提升高达 12.9%。 |
| [^32] | [Guiding End-to-End Driving Models with Endpoint-Constrained Trajectory Optimization](https://arxiv.org/abs/2609.31383) | 该论文发现端到端驾驶模型开环与闭环性能差距的一个新因素是中间路径点监督缺乏物理连贯性和可跟踪性，并提出轻量级后处理层ECO，通过锚定车辆执行历史、保留可靠的预测端点并重塑中间路径点来弥合这一差距。 |
| [^33] | [Towards Understanding LLM-Based Log Anomaly Detection: An Empirical Study of Performance, Efficiency, and Robustness](https://arxiv.org/abs/2609.31371) | 该论文通过对三个公开数据集的系统实证研究，揭示了适配策略、模型规模与量化设置对LLM日志异常检测性能和效率的影响，并评估了模型在结构、语义和标签噪声下的鲁棒性。 |
| [^34] | [Equation discovery with Bayesian tree-adjoining grammars](https://arxiv.org/abs/2609.31368) | 本文首次将树邻接文法置于贝叶斯框架下，利用结构保持树移动的可逆跳MCMC采样器推断模型结构、参数和预测的联合后验分布，实现了超越传统点估计的概率化方程发现与非线性系统辨识。 |
| [^35] | [Brenier Meets Adversarial Training: Optimal Transport Geometry for Robust Learning](https://arxiv.org/abs/2609.31363) | 该论文将带 Wasserstein 惩罚的分布鲁棒优化中的对抗问题重新表述为最优传输映射的优化问题，证明了最优映射满足循环单调性，指出标准对抗训练因违反该性质而浪费传输成本，并提出多起点粒子上升等方法予以改进。 |
| [^36] | [Open Vocabulary Domain Unlearning](https://arxiv.org/abs/2609.31356) | 该论文指出现有近似域遗忘方法因封闭词汇假设仅对已见类别过拟合而无法真正擦除域信息，进而提出类别无关的开放词汇域遗忘协议，要求模型对未见类别也无法识别目标风格域。 |
| [^37] | [Progressive Memory Transformer: Memory-Aware Attention for Time-Series](https://arxiv.org/abs/2609.31351) | 提出渐进记忆Transformer（PMT），通过可写的窗口对齐记忆机制在token、窗口、序列三个尺度上显式监督结构层次，从而改进时间序列的自监督表示学习。 |
| [^38] | [Bridging Body and Brain: Gene-Driven Morphology--Control Co-Design](https://arxiv.org/abs/2609.31329) | 该论文受生物基因启发提出 Morphogene 潜在蓝图与 GeCode 框架，通过 AdaConcat 机制在肢体层面联合条件化形态与控制生成，将机器人形态-控制协同设计转化为在统一紧凑潜在空间中的探索，实现身体与大脑的显式协调优化。 |
| [^39] | [More Sensors Only One Field: Rethinking Continual Spatio-Temporal Forecasting](https://arxiv.org/abs/2609.31325) | 该论文提出STFO（时空场算子），将预测知识参数化为共享的场演化算子，通过观测与查询接口应对传感器布局变化，使传感器扩展时无需传感器特定参数即可复用已学习的空间知识，从而解决持续时空预测问题。 |
| [^40] | [LUCID: Learning Under Confounding for Inference and Discovery in Time Series](https://arxiv.org/abs/2609.31315) | LUCID 提出了一种机制自适应的去混杂层，利用 Marcenko–Pastur 谱路由器估计混杂机制并施加相应的去混杂策略，可封装现有因果发现算法，从而在时间序列中有效消除未观测混杂因子造成的虚假关联。 |
| [^41] | [Benchmarking Attention for Tabular Foundation Models](https://arxiv.org/abs/2609.31306) | 该论文针对表格基础模型中独特的二维行/列注意力模式构建了可复现的基准测试，并在多种注意力后端上系统评估其性能，填补了高效注意力研究在二维表格场景中的空白。 |
| [^42] | [Geometric Moment Contraction for Stochastic Nesterov Acceleration](https://arxiv.org/abs/2609.31303) | 该论文为常参数随机Nesterov加速算法建立了几何矩收缩理论，给出了仅要求梯度具有有限 $p$ 阶矩的显式步长收缩判据，即使梯度具有无穷方差（$1<p<2$）也能保证 $L^p$ 收敛。 |
| [^43] | [Softmax Reparameterization for Output-Head Quantization](https://arxiv.org/abs/2609.31291) | 提出softmax重参数化这一训练后量化方法，通过在量化前减去词表行均值的标量倍数来选取功能等价的输出头，在保持全精度softmax分布不变的同时显著降低输出头量化对语言模型预测的扭曲。 |
| [^44] | [Deterministic Regime Switching and Feasibility Inversion in Dynamic Tensor Rematerialization](https://arxiv.org/abs/2609.31250) | 论文在DTR参考模拟器上发现，仅相差0.10%的内存预算即可确定性地切换快慢两种执行状态（开销差高达7.3倍），且在ResNet-32上出现“预算增大反而导致OOM”的确定性可行性反转现象，揭示了DTR的细粒度确定性不稳定性。 |
| [^45] | [Budgeted Quotient-Residual Guidance for Frozen Pocket-Conditioned Molecular Diffusion](https://arxiv.org/abs/2609.31222) | 提出推理时的预算化商残差引导（QRG），无需重训冻结的口袋条件分子扩散模型，即可借助商几何定向并以采样器自身步长作为信任预算，激活距离、接触等商空间优化目标。 |
| [^46] | [Which Influence Are We Estimating? The Role of Counterfactual Specifications in Data Attribution](https://arxiv.org/abs/2609.31214) | 该论文指出数据归因中各影响估计器排序不一致的根本原因是“规范不匹配”而非近似误差，将影响形式化为反事实估计量，并按隐含规范对现有估计器进行系统分类。 |
| [^47] | [Self-Supervised Representation Learning: From Spectral Foundation Models to Auroral Emission Spectra](https://arxiv.org/abs/2609.31206) | 该论文在22.3万条无标注极光光谱上用掩码自编码器预训练一维Vision Transformer，其学到的表示无需标签即可恢复用于物理诊断的发射线强度比，微调后在极光分类基准上超越监督式分类器，且仅用10%的标签就大幅领先从头训练的模型。 |
| [^48] | [ALF: An Active Learning Framework for Scientific Discovery](https://arxiv.org/abs/2609.31197) | ALF是一个开源的模块化主动学习框架，通过五个模块化组件运行完整的数据获取循环，统一支持离线基准测试和在线部署两种场景，以加速科学发现。 |
| [^49] | [Accounting for Bias Enables Sustainable LLM Evaluation](https://arxiv.org/abs/2609.31184) | 该论文提出一个统一的潜在变量框架，通过显式校正位置偏差、冗长偏差、评委严格度等系统性测量偏差，能用远少于以往的比较次数恢复可靠排名，从而为LLM评估提供了统计上更严谨、计算上更可持续的方案。 |
| [^50] | [BAT-CLIP: Trimodal Alignment of Brain, Audio and Text](https://arxiv.org/abs/2609.31180) | BAT-CLIP提出了首个面向iEEG的CLIP式三模态对齐框架，将神经嵌入同时对齐到预训练的音频和文本锚点，克服了单模态锚定带来的权衡问题，在自然语音解码中实现了比双模态基线更鲁棒的表示。 |
| [^51] | [Audio emotion recognition for atypical hearing](https://arxiv.org/abs/2609.31168) | 该研究提出用LoRA微调CLAP基础模型，从少量神经典型听众的标注数据中泛化音频情感识别能力，以探索自闭症人士听觉过敏等非典型听觉情境下的情感评估方法。 |
| [^52] | [BreathGRU: A Novel Semi-Supervised Bidirectional Gated Recurrent Unit Framework for Speech and Breath Segmentation for Respiratory Audio](https://arxiv.org/abs/2609.31165) | 提出BreathGRU框架，通过将帧级声学特征、双向循环建模、伪标签优化与持续时间约束的分段维特比解码相结合，首次实现针对呼吸音频中语音与呼吸事件的精确半监督分割，克服了传统VAD方法将呼吸误判为静音的局限。 |
| [^53] | [SPADE: Escaping the Popularity-Similarity Frontier to Measure Serendipitous Recommendations](https://arxiv.org/abs/2609.31164) | 提出SPADE评估指标，将物品映射到二维流行度-相似度空间并计算用户特定的帕累托前沿距离，从而同时考量流行度、相似度与用户实际相关性，有效度量意外性推荐并防止算法投机。 |
| [^54] | [WorldTS: World Modeling for Multimodal Covariate-aware Time Series Forecasting](https://arxiv.org/abs/2609.31162) | WorldTS提出了一种基于世界建模的时间序列预测框架，将多模态外部协变量直接整合到潜在状态的建模与演化过程中，从而提升未来观测的预测性能。 |
| [^55] | [I Act Therefore I Am: When Is JEPA's Action-Conditioning Enough to Learn Causal Mechanisms?](https://arxiv.org/abs/2609.31161) | 该论文通过隐变量模型与结合条件似然最大化和熵最大化的信息论目标，系统回答了 JEPA 的动作条件化在何种条件下足以从高维观测中恢复底层因果状态。 |
| [^56] | [Bayesian Tensor Autoencoder with Physics-informed Predictive Prior for Multi-dimensional Time Series Anomaly Detection](https://arxiv.org/abs/2609.31157) | 该论文提出了一种融合物理信息预测先验的贝叶斯张量自编码器，通过直接以张量形式处理多维时间序列并弥合基于重构与基于预测两类自编码器之间的信息利用差距，从而提升多维时间序列异常检测的性能。 |
| [^57] | [Teacher-Anchored Selection of Post-Training Quantized Models under Domain Shift](https://arxiv.org/abs/2609.31155) | 该论文提出在域偏移且目标域标签缺失或稀缺的场景下，以教师模型失真作为锚点来指导训练后量化候选模型的部署选择，揭示了置信度类估计器的失效与输出分布类估计器的可靠，并将有监督项与教师锚点相结合以改进选择。 |
| [^58] | [CRNDiff: Count-Native Diffusion Framework via Chemical Reaction Networks](https://arxiv.org/abs/2609.31149) | CRNDiff基于化学反应网络构建计数原生扩散框架，利用闭式生灭转移核实现精确的计数空间扩散建模，并通过倾斜费曼-卡茨导向方法在无需重新训练的情况下对稀有亚群进行条件采样。 |
| [^59] | [Unknown-Traffic Detection, Calibration and Shortcut Reliance in Distilled Encrypted-Traffic Classifiers over One Year](https://arxiv.org/abs/2609.31141) | 本研究通过预注册设计，从两个准确率相同但构造不同的教师模型蒸馏学生模型，并在一年的真实TLS流量上验证：学生会继承教师模型的未知流量检测能力（仅在常规蒸馏温度下）与捷径依赖特性，且这些特性会在长期数据漂移中发生变化。 |
| [^60] | [Frame the adversary: a structure-aware attack methodology](https://arxiv.org/abs/2609.31128) | 该论文提出了一种基于非正交变换的结构感知频率对抗攻击方法，将攻击构造形式化为带扰动约束集的优化问题，并证明攻击等价于加权 ℓ2 投影，从而实现通用且可控的攻击生成。 |
| [^61] | [LocUS: Head Selection and Subspace Projection for Targeted Activation Steering](https://arxiv.org/abs/2609.31122) | 提出 LocUS 方法，通过在解嵌矩阵中识别属性特定的线性子空间、将引导变换限制在该子空间并局部化到稀疏的注意力头子集，实现了更精准、对无关能力副作用更小的定向激活引导。 |
| [^62] | [From Shortcut Learning to Discrete Neural Insertion Sort](https://arxiv.org/abs/2609.31114) | 该论文揭示了神经算法推理模型在学习插入排序时存在捷径学习问题——中间表示在算法执行结束前就能解码出排好序的结果——并提出了一种将序列表示为链、分离标量交换与控制状态转移、且每步后将节点表示投影回离散状态的离散神经插入排序模型，以促使模型真正遵循算法执行过程。 |
| [^63] | [Bayesian Optimization with Fisher Information Geometry: Gradient Bounds and Trust-Region Methods](https://arxiv.org/abs/2609.31107) | 本文通过拉回Fisher信息度量导出采集函数的梯度上界，解释了高维贝叶斯优化中的梯度消失现象，并提出基于信赖域的FITR方法，以局部Fisher权重取代长度尺度缩放来提升优化性能。 |
| [^64] | [A Flatness-Generalization Relation in the Teacher-Student Tree-Committee Machine](https://arxiv.org/abs/2609.31101) | 该论文在师生树委员会机器中通过零温度吉布斯形式与Edwards-Jones形式解析刻画了经验损失典型极小值点的可观测量和Hessian谱，进而检验泛化误差随数据量增大而下降是否对应于损失景观平坦性的提升。 |
| [^65] | [The Residual Stream's Effective Depth](https://arxiv.org/abs/2609.31098) | 提出“有效深度”这一标量诊断指标，通过度量Transformer残差流中表示相似度随层间距离的衰减，将残差累积的理论上限与经验测量区分开来，并发现绝大多数模型的测量值低于正交更新参考值，该差距主要源于残差更新的相关性而非深度未被利用。 |
| [^66] | [Block Sparse Attention with Log-Linear Complexity](https://arxiv.org/abs/2609.31093) | 本文提出PISA，一种采用金字塔Top-K选择策略的块稀疏注意力机制，通过从粗到细的多层级键筛选，将长序列注意力的复杂度从平方级降至对数线性级。 |
| [^67] | [SAGE: A sampling-aware global evaluation benchmark for species distribution modeling](https://arxiv.org/abs/2609.31082) | 该论文提出了SAGE——一个采样感知的全局评估基准，通过结合GBIF训练记录与sPlotOpen植被调查数据，充分考虑采样偏差和物种层面（尤其是稀有物种）的性能差异，为深度学习多物种分布模型提供更可靠、更具信息量的评估。 |
| [^68] | [Quantum Diffusion Models for Medical Image Analysis](https://arxiv.org/abs/2609.31070) | 本文提出一种基于离散时间量子游走算法、结合经典逆向去噪模型的可扩展混合量子扩散模型，突破了现有量子设备规模的限制，能够处理真实世界的大尺寸医学图像数据。 |
| [^69] | [Distributed Learning as a Service: The Developer's Perspective](https://arxiv.org/abs/2609.31061) | 本文提出从开发者视角出发的DLaaS（分布式学习即服务）框架，开发者通过单一管理控制台即可用声明式选项启用差分隐私、拆分学习、分层聚合和知识蒸馏，无需修改客户端代码即可解决联邦学习中隐私泄露、设备资源受限、聚合器扩展性差和带宽成本高等挑战。 |
| [^70] | [DynBranch: Speculative Subgraph Reuse for Dynamic Agentic LLM Serving](https://arxiv.org/abs/2609.31047) | DynBranch 通过让未解析的分支在解析前即可被寻址，实现了投机性子图执行与跨请求的子图结果复用，从而打破“分支解析屏障”，将智能体 LLM 服务的平均延迟降低最高 32%。 |
| [^71] | [KuaFu: Compressing Long User Behavior into Understanding at Billion Scale](https://arxiv.org/abs/2609.31045) | 该论文提出KuaFu系统，在十亿用户规模上解决将长用户行为压缩为用户理解时序列过长与刷新吞吐量的双重工业瓶颈，并应对压缩过程可能引入的四类幻觉问题。 |
| [^72] | [Aurora-X: Built for Extreme Time Series Forecasting](https://arxiv.org/abs/2609.31038) | Aurora-X 是一个十亿参数规模的时间序列基础模型，通过渐进式课程训练和可变分辨率后训练的统一架构设计，在固定权重下实现测试时扩展，并支持跨变量建模、协变量条件化与未来补丁并行解码。 |
| [^73] | [Robust Graph Clustering Network for Multiple Missing Data](https://arxiv.org/abs/2609.31033) | 提出了一种面向多重缺失数据的鲁棒图聚类网络RGCN，通过视图解耦双分支插补、多超球面混合先验和边界感知对比增强三项创新，有效解决了节点属性与图结构同时缺失情况下的图聚类问题。 |
| [^74] | [Metacognitive Selective Ensemble for Mobile Systems](https://arxiv.org/abs/2609.31031) | MetaSE是一种面向移动系统的主动集成框架，通过利用单个模型可靠性的短期持续性来维护小型活跃模型集合，以更低的计算成本实现了与全集成推理相当的准确率，并在树莓派上快2.7倍。 |
| [^75] | [Precision at Speed: Sample-Efficient Online Model-Based Reinforcement Learning for Hydraulic Excavator Control](https://arxiv.org/abs/2609.31025) | 该论文提出一种在线基于模型的强化学习框架，通过从零学习概率动力学集成模型并采用精度门控的轮廓跟踪目标，使11.5吨液压挖掘机仅需20分钟真实交互就达到了以往需要100-150分钟数据训练的控制器相当的跟踪精度。 |
| [^76] | [Synth-JEPA: Joint Embedding Prediction for Renderer-Free Synthesizer Parameter Search](https://arxiv.org/abs/2609.31024) | Synth-JEPA通过从成对的合成器数据中学习相互可预测的音频与参数联合嵌入表示，实现了无需渲染音频即可直接评分候选参数的目标函数，在域内和域外的声音匹配任务上均优于或媲美现有基线。 |
| [^77] | [Robust Successor Features](https://arxiv.org/abs/2609.31016) | 该论文提出了鲁棒后继特征，将强化学习中的迁移学习与鲁棒强化学习两种范式统一起来，使智能体在线性马尔可夫决策过程假设下，能够同时在奖励函数和未知转移核两个维度上进行泛化。 |
| [^78] | [Can Pixels Alone Reveal Image Origin? Minimax Limits and Learnable Interfaces for Passive Provenance](https://arxiv.org/abs/2609.30997) | 该论文为仅凭像素的图像溯源建立了精确理论极限——由目标分布与受攻击源分布间的最小全变差距离决定且与验证器架构无关，并揭示了公开验证器因可被模拟而会在远早于该统计极限之前被代理黑盒攻击攻破。 |
| [^79] | [The Linear Representation Hypothesis for Vision-Language-Action Models](https://arxiv.org/abs/2609.30996) | 该论文提出了一种基于签名的理论框架，将线性表示假设从大语言模型扩展到视觉-语言-动作模型，统一了表示与策略，以应对具身交互中感兴趣的物理量与系统动力学共同演化这一挑战。 |
| [^80] | [Learning Hierarchical Causal Representations of the Effects of Forcings on Temperature in Climate Models](https://arxiv.org/abs/2609.30995) | 该论文提出一种分层因果表示学习框架，能够显式区分内部气候变率与外部强迫响应，从而准确预测气候变化情景下的温度演变，并提升机器学习气候模拟器的可信度与因果归因能力。 |
| [^81] | [Does Uniform Discrete Diffusion Need Time?](https://arxiv.org/abs/2609.30977) | 本研究从理论和实证两方面证明，在语言建模等有限数据场景下，均匀离散扩散模型的时间条件化在很大程度上是不必要的，与时间无关的预测器能达到甚至超越时间条件化模型的性能。 |
| [^82] | [A Comprehensive Study of Content Representations for Speech Synthesis](https://arxiv.org/abs/2609.30975) | 该研究在统一生成框架下系统比较了各类语音内容表征，发现说话人身份解耦并非仅由监督信号决定，而是由训练目标与表征信息容量之间的相互作用所决定。 |
| [^83] | [LipSSM: Structurally Lipschitz-Bounded Cascaded State-Space Model via Metric Transfer between Consecutive SSM Layers](https://arxiv.org/abs/2609.30973) | 本文提出LipSSM，通过在相邻状态空间模型（SSM）层之间进行度量迁移，将LipKernel的跨层信息传递思想扩展到级联状态空间模型，相比传统逐层构造方法获得更紧凑的整体Lipschitz界，从而在保证鲁棒性的同时提升模型的表达能力。 |
| [^84] | [Gradient Surgery for Physics-Informed Neural Networks](https://arxiv.org/abs/2609.30966) | 该论文提出了一种物理感知的梯度手术方法PAM-GS，通过揭示PINN训练中角度型与幅值型梯度冲突三阶段交替出现的规律，自适应地缓解任务间干扰，从而提升收敛速度和训练稳定性。 |
| [^85] | [Towards Understanding Momentum Acceleration in River-Valley Loss Landscape](https://arxiv.org/abs/2609.30957) | 该论文基于“河谷”损失景观结构解释了动量加速的原理：大学习率使梯度下降沿低损失“河流”快速前进，随后的学习率衰减抑制垂直振荡并揭示真实优化进展，从而解释了WSD学习率调度器成功的原因。 |
| [^86] | [Low-Bit Recurrent States in Hybrid Language Models](https://arxiv.org/abs/2609.30950) | 该论文提出一种无需校准数据、旋转或训练的混合精度量化方法，利用可观测性格拉姆矩阵导出失真权重并结合归一化状态范围为混合语言模型的循环状态分配位宽，使四位平均有效载荷将超额负对数似然降低3.3至27.9倍。 |
| [^87] | [PORL: Pretrained Offline Reinforcement Learning for the Job Shop Scheduling Problem](https://arxiv.org/abs/2609.30948) | 该论文提出PORL方法，将基于仿真的在线预训练与离线微调相结合，并利用KL散度策略约束将通用调度策略适配到特定生产数据分布，从而克服作业车间调度问题中仿真到现实的差距。 |
| [^88] | [Estimating and Orthogonalizing Unknown Pre-training Gradients for Continual Fine-tuning of Large Language Models](https://arxiv.org/abs/2609.30935) | 提出EoupCT框架，通过动态生成最易受遗忘影响的伪数据来估计未知的预训练梯度，并将其与新任务的梯度进行正交化投影，从而在持续微调大语言模型时有效保护其固有的通用知识，避免灾难性遗忘。 |
| [^89] | [EPOC: Endpoint-Preserving Online Correction With Compressed Residual State for Multi-Horizon Time Series Forecasting](https://arxiv.org/abs/2609.30929) | EPOC提出一种端点保持的在线校正方法，仅存储残差块的低阶DCT系数与末端值以实现压缩残差状态，通过分量级在线岭回归与基础预测融合，在多步长时间序列预测中将MSE平均降低15.40%，同时大幅减少辅助存储开销。 |
| [^90] | [Robust to Which Model Change? A Unified Evaluation of Robust Counterfactual Explanations](https://arxiv.org/abs/2609.30918) | 本文提出一个统一的跨家族评估协议，在固定事实实例与反事实解释的前提下，针对相同的八种模型变化类型比较六种鲁棒反事实解释方法与两种基线，发现各方法的相对性能和失败模式随变化类型而异，因此现有报告的鲁棒性分数彼此不可比。 |
| [^91] | [Training Graph Foundation Models on The Web Graph](https://arxiv.org/abs/2609.30894) | Acacia是一个仅用Common Crawl网页图从零训练的图基础模型，无需额外训练即可支持任意特征维度及节点分类、链接预测、节点聚类、图生成等多种任务，具备上下文学习能力且不依赖预训练LLM，证明了图模型可以像LLM一样从零涌现能力。 |
| [^92] | [Conformal Prediction under Exponential-Tilt Joint Shift](https://arxiv.org/abs/2609.30886) | 该论文研究了在输入分布及其与结果关系同时发生联合偏移时，比较了在保形预测校准中直接使用ExTRA估计的指数倾斜权重与额外对源预测分布进行倾斜这两种适应策略的覆盖性能。 |
| [^93] | [Retraction-Based Gradient Projection Algorithms on Manifolds](https://arxiv.org/abs/2609.30885) | 本文提出了黎曼流形上基于回缩的凸优化框架，建立了多种步长规则下基于回缩的梯度投影算法的收敛性结果，并成功应用于加权低秩逼近和图像补全问题。 |
| [^94] | [CacheReforge: Bounded Recovery for Stale KV Caches under Evolving Adapters](https://arxiv.org/abs/2609.30884) | 提出CacheReforge，将陈旧KV缓存表示为逐层混合版本对象，结合适配器锚点、敏感度、累积漂移与重启边界，以最小重计算在适配器演化时恢复当前模型行为。 |
| [^95] | [EXAONE Demand 1.0: A Time Series Foundation Model for Demand Forecasting](https://arxiv.org/abs/2609.30880) | 该论文提出了EXAONE Demand——一个专为需求预测设计的时间序列基础模型，通过构建包含1130万条序列的需求专用语料库，以及基于四类需求类别（平滑、间歇、波动、块状）进行路由的低秩适配器架构，解决了通用时间序列基础模型难以处理需求数据短历史、频繁零值、缺货删失等特殊性质的问题。 |
| [^96] | [TISD: On-Policy Self-Distillation with Trajectory Intervention](https://arxiv.org/abs/2609.30878) | 该论文提出TISD算法，通过在师生分歧峰值处强制执行教师偏好的分支动作并重生成轨迹进行蒸馏，突破了在线策略自蒸馏无法监督学生未采样分支的、单纯依赖局部纠错的训练瓶颈。 |
| [^97] | [Tight Stochastic Condition-Number Dependence in Nonconvex-Strongly-Concave Minimax Optimization](https://arxiv.org/abs/2609.30877) | 本文证明了非凸-强凹极小极大优化中SAPD+算法随机复杂度的线性条件数依赖性是紧致不可改进的，给出了与之匹配的最坏情况复杂度下界Θ(κLGσ²ε⁻⁴)。 |
| [^98] | [AC Power Flow Contingency Analysis Using a Single Deep Neural Network](https://arxiv.org/abs/2609.30859) | 本文提出一种仅用基准工况交流潮流数据训练的单个深度神经网络框架，通过不动点迭代预测任意单线路停运后的事故后运行状态，并利用半定规划验证收敛条件，从而避免了传统机器学习方法需要针对每种预想事故单独收集数据和训练模型的高昂离线成本。 |
| [^99] | [Learning Chance-Constrained MDPs with Bellman Distributional Certificates](https://arxiv.org/abs/2609.30856) | 本文提出“贝尔曼分布式证书”这一核心技术，通过在策略选择之前为约束违反概率构建贝尔曼递归，证明了机会约束MDP虽然计算上更困难，但其统计学习代价并不更高，所建立的样本复杂度上界与理论下界在对数因子内匹配。 |
| [^100] | [The KV Cache Is the New Memory Wall](https://arxiv.org/abs/2609.30854) | 这篇系统化知识论文通过推导算术强度随上下文长度衰减的闭式公式，并在H100、B200和MI300X等硬件上进行参数化建模，首次从分析角度统一了KV缓存优化这一领域，解决了各研究间因工作负载和指标不一致而无法比较的问题。 |
| [^101] | [Aligning One-Step Generative Models with Reward-Weighted Transport Distillation](https://arxiv.org/abs/2609.30840) | 提出了奖励加权传输蒸馏（RWTD）后训练方法，仅凭生成样本和标量奖励评估，通过混合倾斜的当前分布与参考分布构建自适应目标，并借助特征空间最优传输和不动点回归，实现对单步生成模型的有效对齐。 |
| [^102] | [Attention-Based Adaptive Policies for Simultaneous Speech-to-Text Translation](https://arxiv.org/abs/2609.30839) | 本文提出利用交叉注意力机制的RFAP和DCAP两种自适应策略，使离线训练的语音翻译模型无需额外训练即可用于同声翻译，最高提升4.0 BLEU并降低翻译延迟。 |
| [^103] | [Peer-Grounded Counterfactual Path Planning for Chronic Health Management](https://arxiv.org/abs/2609.30838) | 该论文提出POROS框架，根植于自我效能理论与社会比较理论，通过从真实相似患者的观测状态构建行为进展图，以反事实路径规划生成有同伴证据支撑、能保证健康指标单调改善的渐进式分步干预方案，克服了现有方法只给目标、缺少路径、缺乏同伴依据的缺陷。 |
| [^104] | [MOPD-Router: Rethinking Teacher Routing in Multi-Teacher On-Policy Distillation](https://arxiv.org/abs/2609.30837) | 提出MOPD-Router框架，无需领域标签即可在token级别对完整教师池进行监督路由，并通过ExpertAlign依据教师后训练所获专业化能力为其修正信号评分，从而充分释放多教师互补知识。 |
| [^105] | [Adaptive Interaction Graphs for Particle Simulation](https://arxiv.org/abs/2609.30822) | 提出AdaptGNS，利用逐粒子不确定性估计自适应地扩展高不确定性粒子的交互图邻域，在几乎不增加推理成本的情况下，于长时程粒子模拟中实现严格的帕累托改进。 |
| [^106] | [Quantizing Looped Transformers: Feedback Exposure and Calibration Blindness](https://arxiv.org/abs/2609.30820) | 该论文揭示了循环Transformer低比特训练后量化的两种失效模式——“反馈暴露”（无恒等路径的量化层误差在递归中被反复放大回馈，且该现象同样存在于Mamba等非Transformer架构）和“校准盲区”（单步GPTQ仅依赖第0步激活校准，忽视后续递归步的输入方向）。 |
| [^107] | [Learning Provable Neural Network Observer for Uncertain Dynamical Systems](https://arxiv.org/abs/2609.30819) | 提出了一种两阶段训练框架（点引导李雅普诺夫预训练加LMI微调），为不确定动态系统的神经网络观测器实现了可证明的全局李雅普诺夫稳定性，同时克服了传统LMI方法带来的大规模半定规划可扩展性瓶颈。 |
| [^108] | [Counterfactual Online Conformal Prediction Under Adaptive Logging](https://arxiv.org/abs/2609.30811) | 本文提出倾向加权在线共形预测（PW-OCP）及其双重稳健变体（DR-OCP），通过逆倾向加权递归消除校准偏差，解决了自适应记录机制下在线共形预测对罕见行动反事实结果系统性覆盖失效的问题，并在正性条件下实现接近信息论下界的反事实覆盖率。 |
| [^109] | [Deep-Learning Solvers and Surrogates for Infinity and p-Laplace Problems](https://arxiv.org/abs/2609.30809) | 本文利用PINN和DeepONet求解无穷大Laplace与p-Laplace问题，在大p值（2至1000）和三维区域上优于传统网格求解器，并建立了条件收敛性与通用逼近的理论结果。 |
| [^110] | [Towards Universal Representation-Based Process Control](https://arxiv.org/abs/2609.30790) | 该论文将窗口级过程监控重新表述为以经验参考分布为零假设的假设检验问题，并提出一个结合预训练时间序列编码器、核密度估计与保形校准的基于表示的非参数框架，在学习到的表示空间中实现有限样本有效的过程控制推断。 |
| [^111] | [Interpretable-by-Design Descriptor Portfolios Match a 2048-Dimensional Foundation Embedding on Low-Data Molecular Assays](https://arxiv.org/abs/2609.30789) | 该论文提出一种可解释性设计（interpretable-by-design）的描述符组合方法，通过贪婪拼接经过来源筛选的紧凑描述符块，在特征层面完全可审计的前提下，于低数据ADME/Tox检测上取得了与2048维CheMeleon基础嵌入相当的精度（平均AUC 0.762对0.764）。 |
| [^112] | [Missingness-Aware Conformal Prediction Under Cross-Hospital Distribution Shift](https://arxiv.org/abs/2609.30781) | 提出一种缺失感知的共形校准方法，通过按测量指标是否缺失对患者分组并在组内应用Mondrian校准，在跨医院分布偏移的死亡率预测中有效缩小了最差分组的覆盖差距。 |
| [^113] | [Query-Conditioned Prototype Adaptation for Cross-Domain Few-Shot Learning: Single-Query Inference, Controlled Comparisons, and Failure Modes](https://arxiv.org/abs/2609.30769) | 该论文提出实例内原型变换器（WIPT），在冻结的全局表示下通过测试时对单个查询与支持集嵌入进行联合变换来构建查询特定的类原型，并通过多随机种子的受控实验揭示其收益依赖目标域——在 CUB 和 EuroSAT 上带来提升，但在 ISIC 上失效。 |
| [^114] | [HCOE: Hyperbolic Clinical Ontology Embeddings from Biomedical Language Models](https://arxiv.org/abs/2609.30763) | HCOE 通过将冻结的 BioBERT 嵌入映射到双曲庞加莱空间，并结合本体引导的对比学习与由粗到细的路径聚合，构建了保留医学代码层级结构的临床概念表示，在临床关系预测及死亡率、再入院、药物推荐等多项临床任务上均达到最佳性能。 |
| [^115] | [Skill Profiling with Attributable Reasoning (SPAR): A Wearable Analysis System for Boxing](https://arxiv.org/abs/2609.30753) | SPAR是一种结合八单元IMU服装与压力鞋垫的拳击可穿戴系统，不仅能将出拳分类为专家或新手水平，还通过逐关节归因、动力链反事实解释和通俗语言叙述三个层级的可解释反馈，分别服务于分析师、教练和运动员，在17名参与者数据上取得0.842的AUC。 |
| [^116] | [Differentiable RNA Secondary Structure Extraction for Deep Learning](https://arxiv.org/abs/2609.30752) | 本文比较了四种RNA二级结构提取算法，揭示深度学习模型的训练方法与结构提取方法之间的一致性对预测性能具有关键影响。 |
| [^117] | [Mechanism-Aware Ensemble Conditioning for Data-Limited Emulation of Extreme Events](https://arxiv.org/abs/2609.30746) | 该论文提出一种机制感知的即插即用集合条件化框架，利用受推动粗糙集合的协方差作为局部不稳定性的无需雅可比矩阵的代理，通过小型FiLM模块将集合几何统计信息注入骨干网络，从而在数据受限条件下实现对混沌系统极端事件的准确仿真。 |
| [^118] | [Beyond Mean Attention: Diversity-Aware, Layer-Wise Scoring for KV Cache Eviction](https://arxiv.org/abs/2609.30738) | 该论文提出在注意力均值之外加入注意力离散度与冗余惩罚（类MMR多样性）的统一KV缓存淘汰评分，并发现仅用一个全局多样化常数即可在多数LongBench数据集上带来提升，而无需按层或按数据集精细搜索。 |
| [^119] | [TR-SSQP: A Trust-Region Method for Constrained Stochastic Optimization under Heavy-Tailed Noise](https://arxiv.org/abs/2609.30732) | 本文提出TR-SSQP方法，在随机序列二次规划框架下通过法向-切向分解与归一化信赖域半径设计，首次为重尾噪声下带等式约束的随机优化问题提供了理论保证。 |
| [^120] | [Input-Layer Starvation: Why Per-Layer Pruning Breaks IoT Intrusion Detectors](https://arxiv.org/abs/2609.30729) | 该论文揭示了均匀逐层剪枝会使IoT入侵检测器的输入层“饥饿”（46%的第一层滤波器失去全部输入权重、归一化统计量大幅偏移），导致被整体准确率掩盖的近半类别性能崩塌，并提出低开销的预防与修复方法。 |
| [^121] | [When 10,000 Windows Are Not 10,000 Tests: Auditing Statistical Confidence in Sliding-Window Time-Series Classification](https://arxiv.org/abs/2609.30721) | 本文揭示滑动窗口分类中大量重叠测试窗口并非独立样本，提出将三种泛化主张映射到依赖稳健推断（如Bartlett-HAC）的实用审计方法，发现测试行数增长近四倍仅带来约1.75-1.94倍的方差等效信息增长。 |
| [^122] | [NEMSim: Learning Control-Conditioned Multi-Event Physical Dynamics via Executable Event-Mechanism Priors](https://arxiv.org/abs/2609.30718) | 提出NEMSim框架，将预定义的事件-属性描述编译为可执行的转移结构，融合事件机制先验与神经网络学习，实现对控制条件下多事件物理系统跨广泛控制空间和长轨迹的高效高保真模拟。 |
| [^123] | [Parameter Estimation for Unnormalized Discrete Models via Empirically Localized Deformed Bregman Divergence](https://arxiv.org/abs/2609.30713) | 本文提出将经验局部化技术与变形Bregman散度相结合来估计非归一化离散模型的参数，在大幅降低归一化常数计算成本的同时，可通过选择变形方式使估计器具备有效性或抗离群噪声鲁棒性等良好统计性质。 |
| [^124] | [LUMO (Lightweight Unified Multilingual Orchestrator): A Privacy Preserving Offline Voice Assistant](https://arxiv.org/abs/2609.30692) | 该论文提出了LUMO，一个面向边缘计算环境的完全离线、保护隐私的轻量级多语言语音助手，它通过将本地ASR、4位GGUF量化的大语言模型和TTS集成到统一流水线中，可在8 GB内存的树莓派5等资源受限设备上实现实用的端到端语音交互。 |
| [^125] | [Threat-Aware Energy-Efficient Deployment for Dynamic UAV Networks: A Multi-Agent RL Approach](https://arxiv.org/abs/2609.30690) | 提出了一种威胁感知的无人机网络节能部署三步框架，结合威胁感知K均值聚类、最优匹配和MATD3多智能体强化学习，在实现零安全违规的同时最大化全局能效并加速收敛。 |
| [^126] | [On the Limits of Univariate Deep Learning for Significant Wave Height Forecasting](https://arxiv.org/abs/2609.30688) | 该研究通过对五种深度学习架构和多种上下文长度的系统超参数搜索发现，单变量深度学习模型在有效波高预测中性能差异极小（远小于数据集本身变化的影响），且在极端波浪条件下表现不如简单的持续性预测，揭示了单变量深度学习方法在该任务上的局限性。 |
| [^127] | [PixSim: a calibrated open-source simulator of instant-payment fraud, recovery and interdiction under analyst capacity constraints](https://arxiv.org/abs/2609.30684) | PixSim是一个基于巴西央行开放数据校准的开源模拟器，首次联合建模了Pix即时支付的不可逆结算、监管追回机制、资金下游分散与分析师容量受限的审核决策，并在模型冻结后成功再现了2025年真实的欺诈追回率。 |
| [^128] | [StarWM: Self-Supervised Trained Attention Routing for Robust World Models](https://arxiv.org/abs/2609.30667) | 提出StarWM，利用自监督动态训练的交叉注意力模块决定重构区域，结合双流解码器与停止梯度屏障，使世界模型既能忠实学习相关动态，又不受任务无关内容的干扰。 |
| [^129] | [Population loss in shallow ReLU networks: Bias & families of critical points](https://arxiv.org/abs/2609.30661) | 本文利用Owen T函数推导出适用于带偏置浅层ReLU网络的学生-教师核模型总体损失解析公式，证明已知伪极小值族可扩展至带偏置网络，且添加偏置总能严格降低损失、对损失景观的改变相对温和。 |
| [^130] | [DiffusionShadow: Diffusion-based Shadow Caching for Neural Volume Rendering](https://arxiv.org/abs/2609.30658) | 本文提出DiffusionShadow框架，利用扩散模型将大量预计算的阴影隐式神经表示压缩为单一模型，在运行时高效重建阴影，从而实现带阴影效果的实时神经体绘制。 |
| [^131] | [Causal Retention in Interactive Agents: Interface Factorization and Selective Adaptation](https://arxiv.org/abs/2609.30650) | 本文提出“因果保留”理论，证明冻结的学习状态能否正确响应独立于训练的机制探针取决于学习接口纤维与探针答案纤维之间的包含关系，并据此构建 Causal Core 系统，通过证据门控写入与选择性适应等机制实现无误差的目标更新。 |
| [^132] | [MARCEDES: Score-based causal discovery under non-Gaussianity with continuous optimization](https://arxiv.org/abs/2609.30643) | 提出了名为MARCEDES的基于评分的因果发现方法，通过引入平均绝对残差风险、行稀疏惩罚和软DAG约束，实现了非高斯误差下因果DAG结构的高效连续优化学习。 |
| [^133] | [In-Context Binding Capacity in Language Models](https://arxiv.org/abs/2609.30634) | 该论文首次系统测量了语言模型在上下文中正确记住“值-实体”绑定关系的容量极限，发现该容量随参数规模按幂律增长（K₅₀ = cN^0.82），并揭示了干扰效应和预训练配方对测得容量的影响机制。 |
| [^134] | [Stable initialization without the CLT](https://arxiv.org/abs/2609.30633) | 提出了一种无需中心极限定理的均匀相位初始化方法，通过利用正弦函数的周期对称性消除分布近似并完全解耦各网络层，使未经调参的模型在图像和音频拟合等神经表征任务上即超越现有最先进方法。 |
| [^135] | [FRESHLATENT: Channel-Aware Latent Adaptation for Resource-Constrained Embodied VLM Perception](https://arxiv.org/abs/2609.30629) | FreshLatent 提出一种轻量级信道感知潜在适配器，在保持 VLM 冻结的前提下于无线损伤中训练功率归一化编解码器，使分割式 VLM 感知在 0 dB 低信噪比和最紧通信预算下 gIoU/cIoU 提升超过 20 个百分点。 |
| [^136] | [OpenHail: An Event-Driven Gymnasium Environment for Electric Ride-Hailing Fleet Control](https://arxiv.org/abs/2609.30628) | OpenHail是一个开源的事件驱动Gymnasium仿真环境，通过统一的观测-动作接口支持电动网约车车队在订单分配、车辆调度和充电方面的联合控制，为强化学习策略的训练与评估提供了考虑随机需求、车辆运行和充电设施容量约束的结构化仿真平台。 |
| [^137] | [Probabilistic Robustness-driven Universal Adversarial Perturbations with Explainability against Deep Reinforcement Learning-based Intrusion Detection System](https://arxiv.org/abs/2609.30605) | 该论文首次将概率鲁棒性作为显式优化目标，并结合可解释人工智能（XAI）引导扰动塑形，提出了针对基于深度强化学习的入侵检测系统的通用对抗扰动生成方法PX-UAP。 |
| [^138] | [QSV: Quat-Sphere-Vision for Coupled Quaternion Attention on Spherical Lattices](https://arxiv.org/abs/2609.30592) | QSV用每个token一个可学习的单位四元数同时承担注意力打分（相对四元数实部作为logit）和特征传输（三明治乘积变换）两种功能，在同心Fibonacci球面的稀疏kNN图上传递消息，消融实验显示特征传输角色更关键——移除它会使CIFAR-10/100准确率下降约4个百分点，而用均匀平均替代学习到的注意力权重则影响甚微。 |
| [^139] | [Encryptability As a Coordinate Choice: Depth-One Homomorphic Federated Learning of Quantum Neural Networks](https://arxiv.org/abs/2609.30581) | 该论文发现通过将变分量子电路的权重表示在单位四元数（自旋）坐标系中，群复合恰好成为二次双线性运算，使得量子神经网络能够在深度一的同态加密联邦学习框架下高效训练，每个权重仅需一个乘法层级、联邦平均零开销，并完全消除昂贵的自举操作。 |
| [^140] | [Energy-efficient operation of neural operators for virtual sensing](https://arxiv.org/abs/2609.30580) | 该研究表明，在虚拟传感应用中，通过编译器冻结、主干复用与图重放等共享空间计算手段可显著降低神经算子推理的运行能耗，在15瓦固定时钟模式下相比即时执行节能约22%。 |
| [^141] | [Reinforcement Learning of Communication in a Mesh of Small Language Models](https://arxiv.org/abs/2609.30578) | 提出TalkMesh去中心化小模型智能体网络，通过强化学习训练通信策略，让智能体之间传递解题关键提示并基于置信度进行修订，仅用三个智能体和最多六个输出就突破了独立采样多数投票的饱和限制。 |
| [^142] | [T-RoPE: Time-Aware Rotary Position Embedding for Sequential Recommendation](https://arxiv.org/abs/2609.30576) | 提出T-RoPE，一种时间感知的旋转位置编码，通过基于时间戳的角度、可学习时间系数和多尺度频率等机制打破标准RoPE的时间平移不变性，使序列生成式推荐模型能够捕捉时间间隔、行为周期与季节性等关键时间信息。 |
| [^143] | [Entropy Regularization: A Free Correction to Cross-Entropy for Verified Demonstrations](https://arxiv.org/abs/2609.30572) | 该论文指出在存在多个正确解的可验证任务中，用交叉熵模仿单一专家示范可能与验证器风险目标不一致，并提出以熵正则化作为“免费”修正来控制策略支持集、防止概率质量流向错误输出。 |
| [^144] | [Probing Stability-Plasticity Tradeoffs in Agent Memory through Cognitive Experimental Paradigms](https://arxiv.org/abs/2609.30558) | 本文提出受认知科学启发的MemProbe框架，通过干扰、错误信息、巩固强度和再巩固窗口四种可复用的实验范式，超越传统最终答案准确率评估，系统诊断智能体记忆中稳定性与可塑性之间的权衡。 |
| [^145] | [Dynamic Regret in Online Convex Optimization with Indicator Switching Costs](https://arxiv.org/abs/2609.30556) | 该论文首次针对带指示器切换代价的在线凸优化给出了动态遗憾保证，提出了一种由二进制时间尺度重启的惰性FTRL基学习器与基于最大耦合的移动感知主算法组成的元学习框架，遗憾界为期望意义下的 Õ(min{√(T(S_T+1)), T^{2/3}(P_T+1)^{1/3}})。 |
| [^146] | [Rank-Reliable Teacher-Guided Fitness Approximation for Expensive Evolutionary Optimization: A TinyML Architecture Search Study](https://arxiv.org/abs/2609.30553) | 提出TGL-NSGA-II框架，通过教师引导的轻量知识蒸馏生成排序可靠的低保真适应度分数，并与高斯过程代理模型融合，以显著降低昂贵TinyML神经架构搜索中进化优化的评估成本。 |
| [^147] | [GyroNovo: Error-Guided Fragment Imputation with Mass-Aware Attention for \textit{De Novo} Peptide Sequencing](https://arxiv.org/abs/2609.30542) | GyroNovo提出了一个从头肽段测序框架，利用解码器误差自适应引导碎片填补目标，并通过质量感知注意力显式建模峰间质量差异，从而提升对稀疏、含噪且不完整谱图的测序准确性。 |
| [^148] | [AutoResearch at Production Scale: Failure Modes and a Multi-Agent Framework](https://arxiv.org/abs/2609.30541) | 该论文将AutoResearch范式应用于生产规模的推荐系统嵌入优化，在220多次实验中识别出基础设施脆弱、智能体记忆衰退、搜索方向停滞、迭代成本不对称和指标固化五种失效模式，并据此提出多智能体框架加以解决。 |
| [^149] | [Learning to Replace MCMC in Split-Gibbs Diffusion Posterior Sampling via Deep Unfolding](https://arxiv.org/abs/2609.30539) | 本文提出一种基于深度展开的学习框架，将split-Gibbs扩散后验采样中的Gibbs更新重新表述为高斯去噪问题并通过ODE扩散实现，从而以更低的似然更新计算成本替代了传统MCMC迭代。 |
| [^150] | [Seeing Speech: Learning Visible Articulatory Dynamics for Speech-Driven 3D Facial Animation](https://arxiv.org/abs/2609.30517) | 该论文提出一种发音感知框架，通过语音-发音记忆模块（SAM）和拓扑感知发音组合模块（TAC）建模三种方向性发音运动，从而将语音映射为与语音一致且表面连贯的三维人脸动画。 |
| [^151] | [Benchmarking the Connectomes of Caenorhabditis elegans within the Reservoir Computing Framework](https://arxiv.org/abs/2609.30508) | 该研究将秀丽隐杆线虫在不同年龄段、以三种测量方式获得的连接组经最少预处理实现为回声状态网络储备池，并通过神经启发任务对其进行基准测试评估。 |
| [^152] | [Federated Targeted Maximum Likelihood Estimation](https://arxiv.org/abs/2609.30503) | 本文提出了首个联邦目标极大似然估计算法，通过梯度聚合（FedTMLE-G）和本地自主涨落拟合（FedTMLE-L）两个互补框架，首次在数据无法集中汇总的跨机构场景下，对任意目标、损失和涨落族实现了 TMLE 目标化步骤的联邦化。 |
| [^153] | [To Solve Bilevel Optimization with Nonconvex Lower Levels, We Need Second-Order Stationarity](https://arxiv.org/abs/2609.30501) | 本文突破了现有双层优化研究依赖下层凸性假设的局限，系统研究了下层非凸的双层优化问题，并论证了求解此类问题必须采用二阶平稳性条件而非传统的一阶平稳性。 |
| [^154] | [PolicyAttention: Softmax Attention Implements Policy Mirror Descent for Closed-Loop Control](https://arxiv.org/abs/2609.30500) | 该论文构造了一个带显式残差的因果softmax“行动者—环境—单步评论家”协议，证明softmax注意力可以作为重复控制器实现策略镜像下降，并用预注册实验验证了训练的pre-LN Transformer能够恢复目标计算。 |
| [^155] | [Ordinary Nonconvex SGD under Distance-Dependent Moments: Finite-Horizon Stationarity and Nagaev Bounds](https://arxiv.org/abs/2609.30499) | 本文证明，当条件矩允许噪声方差随迭代点距离增长时，普通单样本SGD无需任何修改即可达到与Blum–Gladyshev下界匹配的极小极大随机复杂度，并借助希尔伯特空间Fuk–Nagaev不等式给出高概率Nagaev型界。 |
| [^156] | [Learning to Bias: Machine Learning-Enhanced Particle Filters](https://arxiv.org/abs/2609.30498) | 提出神经最优粒子滤波器（NOPF），通过从离线模拟数据中学习最优提议分布的摊销近似，并将其作为即插即用模块嵌入标准粒子滤波器，借助重要性权重校正保证滤波分布的一致性，从而提升序贯推断的样本效率与高维扩展能力。 |
| [^157] | [Geometric Feature Learning for Functional Data Valued on the Symmetric Positive Definite Manifold](https://arxiv.org/abs/2609.30487) | 该论文提出了MatFAE——一种用于学习对称正定（SPD）矩阵黎曼流形上轨迹的函数神经网络，它将序列视为连续函数以编码轨迹动力学特性，并通过函数权重的形态提供可解释性。 |
| [^158] | [Do LLMs Understand Context? A Knowledge Graph-Based Evaluation Framework](https://arxiv.org/abs/2609.30484) | 提出了一种基于知识图谱的评估框架，通过语义结构相似度衡量大语言模型在问答任务中真正的上下文理解能力，弥补了BLEU和困惑度等传统指标只能评估表面性能的不足。 |
| [^159] | [AcoustiClaim: A Numeric Claim Benchmark with Instrument Ground Truth](https://arxiv.org/abs/2609.30483) | 该论文提出 AcoustiClaim 基准，首次以定义声学量的仪器读数作为真值来验证音频语言模型给出的数值声明，发现绝大多数模型的表现不优于常数预测器，仅一个闭源模型在音高等少数任务上通过了秩相关阈值。 |
| [^160] | [Scaffold-Constrained Subset Dynamic Programming for Exact SSE Clustering](https://arxiv.org/abs/2609.30477) | 该论文提出利用数据导出的几何图对精确子集动态规划施加支架约束，仅允许连通顶点子集作为候选簇，在不改变SSE损失的前提下大幅缩减搜索空间，并证明在密度条件下每个观测仅保留O(log n)个最近邻即可高概率保持经验最优解。 |
| [^161] | [Mentored Decoding: Faster Inference meets Boosting](https://arxiv.org/abs/2609.30474) | 该论文提出“导师解码”这一有损推测解码的形式化框架，通过将其与机器学习中的提升理论相联系并推广到全部f-散度族，正式证明了有损推测解码所得模型不仅加速推理，还能在质量上超越目标模型。 |
| [^162] | [Moment-guided edge sampling](https://arxiv.org/abs/2609.30472) | 提出基于随机游走转移矩阵谱矩的矩引导边采样框架，通过组合方法和低秩方法两种互补手段精确高效地量化和控制局部边编辑对全局图结构的影响，将单边编辑的矩更新成本从 $O(mn)$ 降至 $O(m)$ 乃至常数时间。 |
| [^163] | [CARGO: Context-Aware Retrieval-Gated Evaluation of Agentic AI in Production](https://arxiv.org/abs/2609.30471) | 针对生产环境中智能体AI评估时“参考-实例分歧”（即参考答案描述的是不同实体导致正确回答被误判为错误）的问题，提出CARGO框架，将检索参考视为流程范例、基于实时实例上下文判定事实并将评估门控于检索置信度之上。 |
| [^164] | [Reliability-aware Cross-sample Enhancement for Robust Multimodal Sentiment Analysis](https://arxiv.org/abs/2609.30470) | 提出可靠性感知跨样本增强（RCE）框架，通过自适应变分信息瓶颈建模模态不确定性并抑制噪声，同时利用跨样本检索高置信度语义一致的邻居样本来增强表示，统一解决了多模态情感分析中的噪声干扰与模态缺失问题。 |
| [^165] | [RAZOR: Pruning Replaceable Experts in LLMs](https://arxiv.org/abs/2609.30465) | 该论文提出无需训练的 MoE 专家剪枝方法 RAZOR，利用共识残差衡量专家的功能可替换性，在固定剪枝预算下剪除可被存活专家替代的专家，无需梯度或恢复训练即可最大程度保留原始模型输出分布。 |
| [^166] | [Auditing System-1 Models on Biosecurity-Relevant Benchmarks: Calibration, Selective Prediction, and Permutation Instability in a Non-Generative Model](https://arxiv.org/abs/2609.30454) | 该论文首次系统审计了商业非生成式“系统1”模型在生物安全相关基准上的可靠性，发现其校准良好但准确率高度依赖任务，且对答案选项的排列顺序表现出不稳定性。 |
| [^167] | [Bayesian Uncertainty Quantification for fMRI Functional Connectivity via Simulation-Based Inference](https://arxiv.org/abs/2609.30445) | 该论文提出一个基于仿真推断的贝叶斯框架，通过将BOLD动力学建模为耦合Ornstein-Uhlenbeck过程并使用序贯神经后验估计，量化了fMRI功能连接估计中来自扫描仪噪声、被试变异性和采集时长三方面来源的不确定性。 |
| [^168] | [Improving Molecular-Morphology Contrastive Pretraining using Deep-Learning-based Morphology Profiles](https://arxiv.org/abs/2609.30433) | 该研究用基于深度学习的细胞图像编码流程替代CellProfiler来提取更丰富的形态学特征谱，并通过对比学习将其与分子嵌入对齐，从而改进了分子-形态对比预训练方法并提升了QSAR预测性能。 |
| [^169] | [Fake News Theories: Harnessing Disciplinary Insights for Computational Modeling, Detection, and Explanation](https://arxiv.org/abs/2609.30427) | 该研究提出一个理论驱动的计算框架，将社会科学、心理学、经济学等学科的假新闻理论转化为可测量特征，结合统计技术与大语言模型，实现了更具可解释性、更有理论依据的假新闻自动检测与解释。 |
| [^170] | [Electric Vehicle Charging Station Location Selection using Geospatial Artificial Intelligence (GeoAI)](https://arxiv.org/abs/2609.30417) | 本研究提出了一种融合变分自编码器（VAE）与图卷积网络（GCN）的地理空间人工智能框架，通过整合电动汽车使用、土地利用、人口和交通等多维地理空间数据来捕捉现有充电站间的相似性，从而为未来充电站识别最优选址。 |
| [^171] | [Adaptive Multi-Value Control in LLMs via Causal Activation Steering](https://arxiv.org/abs/2609.30405) | 该论文提出AIMES框架，通过为道德基础价值构建层级双极方向并以中间层词汇读取作为在线观测器，由观测器引导的控制器在每个解码步骤自适应调整多价值干预的强度，从而实现对大语言模型的动态多价值同时控制。 |
| [^172] | [What Improves Multimodal Misinformation Detection? Answers from a Large-Scale Empirical Study](https://arxiv.org/abs/2609.30402) | 本文通过涵盖3,375余次实验的大规模实证研究，系统性地回答了哪些设计选择能改进多模态虚假信息检测、它们何时会悄然失效，为构建更强大可靠的检测系统提供了实用指导。 |
| [^173] | [An End-to-End Pipeline for Causal ML with Continuous Treatments: An Application to Financial Decision Making](https://arxiv.org/abs/2609.30396) | 该论文提出了一个面向连续处理场景的端到端因果机器学习流水线，创新性地解决了正值性违反检测、高维数据降维以及敏感性分析与估计方法向连续处理空间的适配问题，并成功应用于金融决策。 |
| [^174] | [From Weak Data to Strong Policy: Q-Targets Enable Provable In-Context Reinforcement Learning](https://arxiv.org/abs/2609.30391) | 提出了QTPT方法，用贝尔曼风格的Q目标预训练取代行为克隆，使上下文强化学习在弱数据或次优数据下具有理论可证明的更强鲁棒性。 |
| [^175] | [DanLing NestedTensor: Composable Multi-Ragged Tensors for Deep Learning](https://arxiv.org/abs/2609.30379) | DanLing NestedTensor 是一种将多重参差结构内嵌为张量自身属性的 PyTorch 张量抽象，使广播、特征变换和归约操作能够可组合地处理变长数据，在 BERT 任务上相比填充方法实现了最高 3.39 倍的加速。 |
| [^176] | [Cost-Aware Best-LLM Identification using Dueling Feedback](https://arxiv.org/abs/2609.30360) | 该论文提出了一种结合对决反馈与异质查询成本的成本感知多臂老虎机算法，用于在给定置信度下识别最佳大语言模型，并证明了其渐近最优成本性能。 |
| [^177] | [Strategic Self-Consistency](https://arxiv.org/abs/2609.30352) | 本文揭示了一种针对自洽性推理服务的潜在欺诈行为：不忠实的模型提供商可通过策略性地生成并重排额外的推理路径，使每条路径在多数投票中看起来都不可或缺，从而在避开审计检测的情况下人为增加路径数量以向用户多收费。 |
| [^178] | [Adaptive multi-resolution Gaussian processes: Scalable exact inference with naturally data-sparse covariance matrices](https://arxiv.org/abs/2609.30348) | 该论文提出一种自适应多分辨率高斯过程框架，通过直接锚定样本点的自适应多分辨率基函数构建天然数据稀疏的协方差矩阵，并结合稀疏Cholesky逆算法，实现了既可扩展又精确的高斯过程推断。 |
| [^179] | [Learning coarse-step dynamics and internal mechanical response with graph networks](https://arxiv.org/abs/2609.30344) | 该论文提出Newmark-β-DGN图神经网络框架，通过受Newmark-β方法启发的半隐式更新和算子加权虚拟枢纽，从粗步长离散轨迹中同时学习物理系统的动力学演化与可解释的内部力学响应。 |
| [^180] | [Low-Rank Friction for Memory-Efficient Transformer Pretraining](https://arxiv.org/abs/2609.30342) | 提出R-iKFAD优化器，通过秩1外积分解替代完整摩擦张量，将优化器状态内存近乎减半，同时保持与iKFAD相当的性能和超参数鲁棒性。 |
| [^181] | [GAUDI: Geometry-Aware Diffusion for Calibrated Air-Quality Time-Series Imputation](https://arxiv.org/abs/2609.30340) | 该论文提出针对空气质量时间序列连续块状缺失的条件扩散插补方法GAUDI，通过抑制绝对时间位置嵌入、保留特征侧条件信息，在ItalyAir数据集上取得了优于完整上下文与局部CSDI基线的插补精度（RMSE 0.340 对 0.355）。 |
| [^182] | [Parameters vs. Context: TRACE Fine-Tuning for Robust Retrieval-Augmented Generation](https://arxiv.org/abs/2609.30337) | 本文提出 TRACE 微调框架，利用多智能体辩论轨迹提供细粒度监督，并结合答案完整度正则化机制，提升检索增强生成模型在面对检索知识与参数知识冲突时的鲁棒性。 |
| [^183] | [Guarded Gradient-Based Activation Steering of Shutdown Responses in Qwen3.5-0.8B: A Minimum-Step Policy](https://arxiv.org/abs/2609.30326) | 本研究提出一种守卫式梯度激活转向方法，直接从KEEP−STOP logit差的梯度推导转向方向，仅在检测到关机场景且模型偏好回避时以最小干预步数将回避关机的行为纠正为接受关机，同时保持非关机行为不变。 |
| [^184] | [Adaptive Random Matrices in Gaussian Bandits: Spectral Universality and Selection-Induced Outliers](https://arxiv.org/abs/2609.30321) | 该论文证明当臂数量的对数相对维度次线性时，高斯老虎机中任意自适应选择规则都不改变观测矩阵经验谱收敛于Marchenko-Pastur定律的极限行为，从而使贝叶斯后验不确定度和信息获取具有与策略无关的一阶极限，并在线性得分选择下给出精确的条件臂分布与Wishart型Gram矩阵刻画。 |
| [^185] | [PALM: Point-in-Time Adaptation for Financial Language Models](https://arxiv.org/abs/2609.30316) | 本文通过对比实验发现金融时点语言模型每年完整预训练并非必要——新旧检查点在同一评估窗口上表现相当，据此提出PALM方法，仅需在新增文本上拟合低秩适配器即可实现时点自适应，从而大幅降低维护时点语言模型的成本。 |
| [^186] | [Staged Depth Training: A Representation Curriculum for PINNs](https://arxiv.org/abs/2609.30299) | 提出了表示课程学习概念及分阶段深度训练（SDT）方法，通过先训练浅层网络、冻结已学习前缀再逐步加深网络的方式显式学习表示，在PINNacle基准上显著提升了物理信息神经网络的求解精度。 |
| [^187] | [NeuralCert: certified computational discovery of extremal mathematical constructions](https://arxiv.org/abs/2609.30296) | 该论文提出NeuralCert框架，通过“发现—认证”流程——先以紧凑可分离表示学习高维变分试探函数、再经谱诊断剪枝并由多模精确求值严格认证——在标准个人计算机上实现可独立验证的数值证明，并以发现更优构造、揭示经验不变量、暴露优化障碍三种方式推动极值数学问题的严格研究。 |
| [^188] | [Auditing and Repairing LLM-as-Judge Failures in a Production Text-to-SQL Pipeline](https://arxiv.org/abs/2609.30290) | 本文审计了生产级文本到SQL流水线中的LLM裁判，发现其与人工标注一致性极低且根源于“评分幻觉”这一单一机制，并提出用低成本自托管Qwen模型替换裁判、配合三个强裁判的一致同意集成（kappa = 0.79、自动覆盖率 89.7%）来有效修复该失效问题。 |
| [^189] | [Manifold Projection and Iterative Autoencoder Refinement for Masked Language Modeling](https://arxiv.org/abs/2609.30288) | 该论文提出用低秩瓶颈自编码器混合模块（分别在局部邻域、全序列和注意力头间运作）替代注意力机制，并在掩码位置引入由“拉动”和“校正”两步组成的迭代细化程序，用于掩码语言建模。 |
| [^190] | [When Does Advection-Aware Graph Nowcasting Help? A Controlled Study of Distributed Solar Ramp Forecasting with a Self-Supervised Cloud-Motion Estimator](https://arxiv.org/abs/2609.30286) | 受控合成实验表明，平流感知图神经网络对分布式光伏坡升临近预报的增益有限——在现实的互相关CMV估计下并不优于静态或学习邻接的时空GNN，完美CMV约一半收益来自将运动矢量作为输入特征而非图结构本身，且只有当预报时域内平流位移落在传感网络范围内时平流信息才真正有用。 |
| [^191] | [Seasonal and Quantum-inspired Models for Neutron Monitor Time Series Forecasting](https://arxiv.org/abs/2609.30281) | 本文系统比较了季节性基线、深度序列模型与量子启发架构在中子监测器时间序列多步预测中的表现，发现量子启发变体QiKAN取得最低的总体预测误差，而简单的季节性朴素基线依然保持强竞争力。 |
| [^192] | [Neural Ideals and Neural Codes: An Algebraic Framework for Neural Network Classification and Feature Interpretation](https://arxiv.org/abs/2609.30279) | 提出一个基于“神经理想”的代数框架，建立神经网络与代数对象之间的对应关系及相关计算算法，从而能够识别和解释每个隐藏层神经元所捕获的特征。 |
| [^193] | [Fixed Points Without Fixed Diffusion: Implicit Neural Sheaves for Convergent Test-Time Computation](https://arxiv.org/abs/2609.30277) | 提出 SheafDEQ，一种基于自适应神经层束传播的次齐次深度平衡架构，通过可学习的矩阵值层束限制映射实现更丰富的边依赖变换，并在温和条件下保留了隐式图神经网络不动点唯一且可收敛的保证，从而兼顾表达能力和测试时计算的收敛性。 |
| [^194] | [Why Clipping Matters in AdaGrad? Toward a High-Probability Theory under Generalized Smoothness](https://arxiv.org/abs/2609.30276) | 该论文揭示了在广义平滑和重尾噪声下，未裁剪的AdaGrad会因自适应分母学习到罕见噪声冲击而非目标函数局部曲率而各向异性失准，并首次为裁剪后的原始AdaGrad给出了有限时间范围的高概率收敛保证。 |
| [^195] | [Cosine Similarity Is Not Evidence: Measuring the Noise Floor of Interpretability Transfer Under Quantization](https://arxiv.org/abs/2609.30275) | 该论文指出，用余弦相似度等尺度不变统计量来认证可解释性方法在量化后依然有效、却不报告其噪声基底是不构成证据的，并证明均值差方向估计器的折半噪声基底由 $\kappa=n\rho^2/d$ 与闭式解 $\mathbb{E}[\cos]\approx(1+4/\kappa)^{-1}$ 决定，通过在真实激活上实测此前从未被报告的类别分离度 $\rho$，表明仅采样噪声就能在Qwen2.5-1.5B-Instruct上产生0.978–0.994的余弦一致性，从而使已发表的0.996等认证数值失去证明力。 |
| [^196] | [Distribution of hitting times for dissipative random dynamical systems on $\mathbb{R}^d$, with application to stochastic gradient descent](https://arxiv.org/abs/2609.30274) | 该论文从遍历理论的视角提出了一种新方法，用于研究一大类随机优化方法的渐近性质，其主要贡献是给出了随机优化算法到达极小值点小邻域的击中时间分布分析，并将其应用于随机梯度下降的收敛性研究。 |
| [^197] | [Offline Policy Evaluation as a decision support tool for designing Adaptive Experiments](https://arxiv.org/abs/2609.30273) | 该论文提出将离线策略评估（OPE）与受控热启动模拟相结合的方法，利用历史A/B测试数据对自适应与非自适应策略进行排序评估，为基于上下文老虎机的自适应实验设计提供安全的决策支持工具。 |
| [^198] | [ENAS: An Efficient Hardware-Aware Neural Architecture Search Framework for TinyML on Resource-Constrained Microcontrollers](https://arxiv.org/abs/2609.30272) | ENAS是一个无需GPU即可高效运行的硬件感知神经架构搜索框架，通过静态可行性检查、支持多种块的单元搜索空间和三阶段混合搜索策略，在资源受限的微控制器上实现了TinyML模型的快速搜索。 |
| [^199] | [When the Preconditioning Exponent Turns Negative: Learning-Rate Coupling and Cross-Environment Generalization](https://arxiv.org/abs/2609.30271) | 本研究通过系统性的跨环境实验发现，使跨环境泛化准确率最大化的预条件指数随学习率的对数几乎线性下降（斜率约-0.27至-0.30），揭示了预条件指数与全局学习率之间存在强耦合关系。 |
| [^200] | [HybridInfer: Thermal-Aware Reinforcement-Learning Tier Routing for On-Device, Edge, and Cloud LLM Inference](https://arxiv.org/abs/2609.30270) | 提出 HybridInfer，一种热感知的强化学习路由器，在端侧、边缘与云端三层架构间智能调度大语言模型推理，以解决端侧持续生成导致移动 GPU 推理运行时崩溃或卡死的热约束问题。 |
| [^201] | [Aim Short to Reach Far: Your Frozen World Model Can Plan Better Than You Think](https://arxiv.org/abs/2609.30036) | 提出锚定规划方法，通过瞄准从经验中检索的中间观测目标而非最终目标图像，使冻结的世界模型无需额外训练即可在长程任务规划中全面超越原有规划器。 |
| [^202] | [Diverse Geometries, Frozen Weights: Robust Heterogeneous Treatment-Effect Estimation via Causal Expert Ensembles](https://arxiv.org/abs/2609.29974) | 该论文提出GeoACE五专家集成框架，通过结合锚定校正估计器与多样化的重叠感知和结果引导几何，并采用验证集学习且在测试前冻结的集成权重（其中新增的O-Phi-ACE专家用无结果的重叠感知统计投影替代锚定输入），实现了更鲁棒的异质性处理效应估计。 |
| [^203] | [Rufus-Air: An Open LLM Post-Training Recipe](https://arxiv.org/abs/2609.29421) | 本文提出了Rufus-Air，一个基于GLM-4.5-Air-Base的开放可复现的八阶段大模型后训练方案，并总结出高质量SFT奠定能力基础、难度过滤保持RL学习有效性、奖励可靠性指导阶段排序等关键实践经验。 |
| [^204] | [M-plicits: Neural Implicit Surfaces via Nested Multiscale Residuals](https://arxiv.org/abs/2609.28684) | 提出M-plicits多尺度框架，将表面建模为通过嵌套邻域训练的MLP残差和，并将监督严格限制在先前零水平集周围的窄带内，从而在训练效率、渲染速度和噪声鲁棒性之间实现更好的平衡。 |
| [^205] | [Even Sharper Bounds for Transductive Learning and Its Applications](https://arxiv.org/abs/2609.28459) | 本文提出直推学习的新型局部化复杂度方法STLC，去除了以往直推学习中多余的对数置信度因子，并在可实现设定下达到了与标准归纳学习相同的 $\cO\{\dVC\log(me/\dVC)/m\}$ 速率。 |
| [^206] | [Support-Compiled Feature Folding: More Evidence at Lower Memory Across Tabular Foundation Models](https://arxiv.org/abs/2609.28208) | 提出免训练推理框架“支持度编译特征折叠”（SCFF），将按支持度排序的有界特征子集路由进冻结的表格基础模型，把二次方特征交互开销降为线性，在不集成预测、不训练新参数的情况下以更低内存利用更多证据，并在18个宽表数据集上提升了全部六个骨干网络的准确率与NLL。 |
| [^207] | [Transferable Evidence Reconstruction for Longitudinal Glucose Representations](https://arxiv.org/abs/2609.28199) | 该论文提出可迁移证据重构（TER）自监督方法，通过在一个记录组上拟合低容量读取器并要求其在另一组记录中恢复相同证据的跨组测试，学习具有可迁移证据解码规则的血糖表征，并利用感知观测的每日编码器和感知时钟的多日记忆模块对持续血糖监测数据进行建模。 |
| [^208] | [NS-ATTENTION: Newton-Schulz Transformations of Attention Outputs in Vision Transformers](https://arxiv.org/abs/2609.27735) | 提出无参数的Newton-Schulz注意力变换（NS-Attn.），对每个注意力头输出进行谱处理以降低谱集中度并提高有效秩，在ViT和Swin于CIFAR-10/100的全部12组对比实验中均带来平均0.25–0.83个百分点的准确率提升。 |
| [^209] | [What Do Tabular Foundation Models Compute In Context? In-Situ Representation Refinement through Attention-Gated Updates](https://arxiv.org/abs/2609.27679) | 提出“原位表示精炼”机制并构建RefineICL——一种注意力门控、无FFN的上下文学习堆栈，使表格基础模型在不改变参数的情况下利用支持集标签精炼回合表示并迁移至查询，性能超越TabPFN-3等现有模型。 |
| [^210] | [Reinforcement Learning with Decomposed Subtasks](https://arxiv.org/abs/2609.27035) | 该论文提出RLDS方法，其核心是子任务分解优势估计（SDAE），通过在固定分类体系上将轨迹奖励按子任务分解并计算各子任务的组相对优势，解决了GRPO等方法将多轮rollout压缩为单一标量奖励所导致的信息损失问题。 |
| [^211] | [Towards Hierarchical GNNs for multi-grid power flow: generalization across operating scenarios](https://arxiv.org/abs/2609.26603) | 该论文提出在GENCO校正网络中引入分层潜在通信模块，通过Kron简化和Quotient构建两种简化图在不同电网间交换信息，显著提升了多电网潮流GNN模型对未见运行场景的泛化能力，其中Kron方法将电压误差降低了85.0%。 |
| [^212] | [Geometry-Aware Hyperbolic Residual Quantization](https://arxiv.org/abs/2609.26342) | 提出一种几何感知的双曲残差量化方法，通过双曲残差聚合恢复前向传播中庞加莱圆盘上的伸缩求和特性，并利用带折扣的双曲直通估计器在反向传播中保留几何信息，从而解决双曲空间残差量化的几何不一致问题。 |
| [^213] | [Scalable Minimum-Volume Simplex Estimation with Non-asymptotic Analysis](https://arxiv.org/abs/2609.25576) | 提出 DeepMVSA 方法，通过神经隐式形式（轻量坐标网络加 LU 三角参数化）将最小体积单纯形估计的内存降至与样本量无关的 O(K^2)、单次遍历成本降至 O(NK^2)，并给出非渐近样本复杂度界与神谕不等式等理论保证。 |
| [^214] | [Hill Sampling for Test-Time Scaling: A Simple and Better Alternative to Repeated Sampling, Evolution, and Training](https://arxiv.org/abs/2609.25510) | 该论文提出了一种简单的“山峰采样”方法——从冻结的大语言模型中反复采样候选程序编辑并保留当前最优程序作为后续采样的条件，无需复杂的进化搜索或测试时训练，就在圆填充问题上创下新的最先进水平，并在Erdős最小重叠问题上超越了AlphaEvolve。 |
| [^215] | [MIND the Gap: A Geographic Implicit Neural Representation with Adjustable Spatial Scale](https://arxiv.org/abs/2609.25454) | 提出 MIND 方法，通过嵌套监督将专用预训练地理空间模型的嵌入蒸馏为具有可调空间粒度的单一通用坐标嵌入，从而在稀疏标注条件下实现对遥远区域地理量的泛化预测。 |
| [^216] | [Detecting Agitation Before Behavioral Escalation in Autistic Youth Through Multimodal Wearable Sensing](https://arxiv.org/abs/2609.24791) | 该研究融合惯性测量单元运动、腕戴生理信号和语音三种模态的预训练基础模型，实现了在自闭症青少年挑战性行为升级之前对激越状态的可穿戴多模态检测（AUC达0.724）。 |
| [^217] | [Lifted Bellman Linear Programming for Offline Reinforcement Learning](https://arxiv.org/abs/2609.24489) | 提出提升贝尔曼线性规划（LBLP），通过将贝尔曼最优性的线性规划刻画提升到联合(Q,V)空间，用仅涉及数据集中状态-动作对的不等式约束实现样本内贝尔曼最优性，其唯一最优解在确定性动力学下介于数据集最佳回报与最优价值之间。 |
| [^218] | [Explainable Predictive Condition-based Maintenance of Naval-Propulsion Systems using Fuzzy Logic](https://arxiv.org/abs/2609.24250) | 本文提出了一种结合模糊决策树和深度残差神经网络的新型框架，用于实现海军舰艇推进系统的可解释预测性维护，使用户能够理解预测结果背后的故障原因。 |
| [^219] | [Matrix AdaGrad: Row-wise and Column-wise Adaptive Subgradient Methods](https://arxiv.org/abs/2609.21815) | 本文提出了一个针对矩阵值参数的通用在线镜像下降框架，通过引入按行和按列的自适应近端函数，推导出 Row-AdaGrad 和 Column-AdaGrad 两种优化器，将 AdaGrad 式的自适应次梯度方法推广到了具有矩阵结构的参数优化中。 |
| [^220] | [Elastic Threshold Attention: Learned Contextual Sparsity for Long-Context Decoding](https://arxiv.org/abs/2609.20888) | 提出弹性阈值注意力（ETA），一种端到端可训练的架构，通过从查询表示中预测动态上下文阈值，在不牺牲稠密模型质量的前提下实现长上下文解码的硬件加速，解决了KV缓存带来的内存带宽瓶颈。 |
| [^221] | [Conservation Buys Stability and Factoring Buys Counterfactuals in Physical World Models](https://arxiv.org/abs/2609.19674) | 该论文证明物理世界模型的两种失效需要不同的结构性补救：用辛积分器演化学习到的能量可保持保守动力学几何结构、使长时程展开在训练范围100倍内保持稳定，而用显式线性因式分解编码物理耦合则能使模型泛化到从未见过的干预情形。 |
| [^222] | [Higher-order pruning of experts in mixture-of-experts language models](https://arxiv.org/abs/2609.18916) | 提出二阶剪枝方法HOPE，通过捕捉专家之间的高阶交互作用来可证明地最小化剪枝误差上界，在多个前沿MoE模型和基准测试上的剪枝效果优于忽略专家协作性的一阶方法。 |
| [^223] | [Beyond Quadratic Loss: The Stability Phase Diagram of Adam](https://arxiv.org/abs/2609.18314) | 该研究通过绘制Adam优化器在$(\beta_1,\beta_2)$参数平面上的稳定性相图，发现一条近似线性边界$1-\beta_2=C(1-\beta_1)$可用于区分训练中是否出现损失尖峰，并揭示超二次损失景观（如高置信交叉熵损失形成的“核心-墙壁”结构）是决定该边界形状的关键因素。 |
| [^224] | [Information Geometric Self-Organization at the Edge of Stability in High-Capacity Kernel Associative Memories](https://arxiv.org/abs/2609.16827) | 本文通过Hessian特征值谱分析揭示了KLR联想记忆中“优化脊”本质上是秩1谱坍缩附近的几何奇点，并证明梯度下降的学习动力学在稳定性边缘处表现出瞬态自稳定行为，从而自发地组织到该最优区域。 |
| [^225] | [A Weighted Kernel Method for Approximation that Adapts to Learned Multivariable Structure](https://arxiv.org/abs/2609.16606) | 提出了总敏感度核（TSK）方法，通过加权ANOVA核族学习多变量结构，并借助最小范数RKHS的选择从有限数据中唯一确定各输入的敏感度因子，从而实现对黑箱函数的自适应近似。 |
| [^226] | [Recovering Physical Parameters from Fragmented Observations via Exact Distributed Spline Merging](https://arxiv.org/abs/2609.16579) | 本文提出精确分布式样条合并方法，各数据持有者仅需共享局部Gram矩阵和矩向量即可获得与集中式拟合数学上完全相同的解，并通过场重建与导数提取流程从分布式碎片观测中实现物理参数推断。 |
| [^227] | [Graph Matching Relaxations and Amortization for Supervised Graph Prediction](https://arxiv.org/abs/2609.15437) | 该论文证明了Gromov-Wasserstein目标是监督图预测中最合适的图匹配松弛形式，并提出基于可微Sinkhorn算法的参数化匹配器来摊销图匹配问题，实现图预测模块与匹配器的联合学习。 |
| [^228] | [Large Distant Gradients Need Not Be Reliable: reliability-weighted credit assignment for long-horizon autoregressive forecasting](https://arxiv.org/abs/2609.12890) | 提出Internal-DW方法，通过在反向传播中对每个残差块的恒等路由和非线性路由施加由显式噪声模型估计的有界维纳增益进行可靠性加权，在抑制长时程自回归预测中不可靠远距离梯度噪声的同时保留可预测的学习信号。 |
| [^229] | [EGGROLL, Unrolled: Understanding and Improving Low-Rank Evolution Strategies at Scale](https://arxiv.org/abs/2609.10980) | 本文首次从理论上刻画了面向大语言模型的低秩进化策略EGGROLL的更新场，揭示其可能引入非保守分量并逆转最优点的局部稳定性，同时证明了该方法的二次目标精确性并给出非渐近误差界，为理解与改进该方法奠定理论基础。 |
| [^230] | [Nonmaximal sums of maximally monotone operators under Rockafellar's constraint qualification](https://arxiv.org/abs/2609.10487) | 本文通过计算一类图的单调极并施加正秩一扰动的构造定理，在$c_0$和$\ell^1$空间上构造出满足Rockafellar内部域条件但其和并非极大单调的极大单调算子对，从而推翻了Rockafellar和猜想。 |
| [^231] | [High-probability guarantees for linear accessibility in feature superposition](https://arxiv.org/abs/2609.09556) | 该论文将特征叠加中的线性能及性建模为压缩感知问题，证明了充分维度只需线性规模（而非此前最坏情况的二次方限制）即可高概率地恢复同时激活的特征，从而量化了线性表示假设的几何约束并为稀疏自编码器和神经可解释性评估提供了理论框架。 |
| [^232] | [Steering Interference Reflects the Model's Defaults, Not the Behavior Directions](https://arxiv.org/abs/2609.06951) | 激活导向引发的副作用并非来自被导向的行为方向本身，而是由模型自身的默认偏好决定——无论导向何种行为，模型都会趋向其本已偏好的少数行为（如拒答、谄媚、诗歌化）。 |
| [^233] | [The Geometry of Refusal: Why Post-Hoc Safety Is Fragile and Pretraining-Time Safety Persists](https://arxiv.org/abs/2609.06934) | 该论文从几何视角证明，事后安全训练（如RLHF）的更新与模型能力方向近乎正交，只是在完好的能力之上叠加一道薄而尖锐的拒答“闸门”而非真正删除能力，因此注定会被越狱等攻击绕过，而持久的安全必须在预训练阶段扎根。 |
| [^234] | [SimpleMemVLA: A Simple but Effective Native-Video Memory for Vision-Language-Action Models](https://arxiv.org/abs/2609.05533) | SimpleMemVLA提出了一种无需专用记忆模块的视觉-语言-动作模型，通过完整保留历史信息并以时间戳视频格式直接输入骨干网络，利用子任务隐藏状态作为历史到流匹配动作头的唯一通道，从而有效解决长时程操作中的部分可观测问题。 |
| [^235] | [WEECFP-SuRGE: Wide Embedded Extended Connectivity Fingerprint with Substructure Rotary Graph-distance Encoding](https://arxiv.org/abs/2609.04672) | 该论文提出了无需参数的分子指纹WEECFP以及结合子结构旋转图距离编码（SuRGE）的transformer架构WEECFP-SuRGE，在不使用任何外部预训练的情况下，在TDC ADMET排行榜的22项基准中取得多项第1名和总体领先的回归性能。 |
| [^236] | [Provably Safe Sim-to-Real Transfer](https://arxiv.org/abs/2609.01418) | 该论文提出并形式化了“安全仿真到现实迁移”问题，通过在无奖励安全强化学习框架内构建该问题，使智能体能够在利用不完美模拟器的同时确保现实世界数据收集的安全性，并为目标系统学习到接近最优的可行策略。 |
| [^237] | [MolLedger: An Additive Graph Neural Network with Chemically Grounded ADME Attributions](https://arxiv.org/abs/2608.30636) | 提出MolLedger加性图神经网络，通过将ADME预测表示为逐原子分数之和，并利用辅助损失将原子分数锚定于化学性质，在不损失预测性能的前提下实现了精确且化学上可信的模型可解释性。 |
| [^238] | [One Capability or Many? Testing the Economic Validity of Frontier AI Evaluation](https://arxiv.org/abs/2608.29420) | 本研究通过潜变量模型对421个模型配置和12个基准的分析发现，经济基准测量的并非独立的独特能力，而是与其他基准共享的同一通用能力维度（单一因子解释74.5%的共同方差），从而质疑了前沿AI经济评估的构念效度。 |
| [^239] | [SimCast-S2S: An Efficient Generative Model for Subseasonal Precipitation Forecasting via Transfer Learning from Climate Simulations](https://arxiv.org/abs/2608.26594) | SimCast-S2S通过潜扩散生成框架和气候模拟迁移学习，实现了高效且概率性的亚季节降水预报，解决了不确定性量化和计算成本两大核心瓶颈。 |
| [^240] | [Planetary Prediction Engine: Autonomous Geospatial Prediction via Intelligent Data Selection and Foundation Model Embeddings](https://arxiv.org/abs/2608.26088) | 行星预测引擎是一个自主AI系统，能从自然语言查询直接端到端执行地理空间预测，通过智能数据选择和基础模型嵌入，自动整合多模态数据并搜索最优模型，以应对全球性挑战。 |
| [^241] | [Provable Quantum--Classical Separation for Continuous Gibbs Sampling](https://arxiv.org/abs/2608.24527) | 本文首次证明了连续域吉布斯采样问题中量子算法相比经典算法具有二次加速优势，该优势在低温高维下呈指数增长。 |
| [^242] | [PolyChirp: Multi-Species Birdsong Classification Using TinyML on Low-Power Acoustic Sensors](https://arxiv.org/abs/2608.23101) | PolyChirp通过结合生物专业知识、自动化数据集和NPU加速的微型多类模型，首次实现了低功耗微控制器上多物种鸟类鸣声的实时分类。 |
| [^243] | [The Communication Map of a Transformer](https://arxiv.org/abs/2608.22007) | 提出了一种从权重出发绘制变压器所有潜在通信通道的“通信图谱”方法，能高效计算并揭示大多数注意力头对的耦合或回避模式，且具有广泛适用性。 |
| [^244] | [DecoVAE: a Lightweight Interpretable Trend-Seasonal VAE Framework for Efficient Probabilistic Time Series Forecasting](https://arxiv.org/abs/2608.20052) | 本文提出DecoVAE，一个轻量级可解释的VAE框架，通过趋势差分正则化和频域复高斯VAE显式分解时间序列，在七个基准上持续优于现有方法，同时降低内存和计算开销。 |
| [^245] | [Too Sure to Be Safe: Model Calibration for Reliable Log Anomaly Detection](https://arxiv.org/abs/2608.17965) | 本文提出LoRD框架，通过从正确分类样本的潜在表示中学习可靠性模型，解决日志异常检测器中置信度校准不良的问题，确保错误预测不被过度自信。 |
| [^246] | [LiD-GLM: Lipschitz-constrained Deep Generalized Linear Models](https://arxiv.org/abs/2608.16340) | 提出一种利用可逆残差网络增强广义线性模型的方法，在保持随机单调性的同时实现非线性参数估计和分布假设的灵活校正。 |
| [^247] | [A Banach-Space Theory of Markovian Halpern Iteration for Non-Expansive Maps](https://arxiv.org/abs/2608.15966) | 本文提出了一种在巴拿赫空间中基于方差缩减的马尔可夫PAGE-Halpern迭代方法，通过泊松方程和位移级别Halpern界，将非扩张映射不动点逼近的样本复杂度从$\tilde O(\epsilon^{-5})$降低至$\tilde O(\epsilon^{-3})$。 |
| [^248] | [Depth-adaptive Inference of Looped Language Models via Continuous Depth Batching](https://arxiv.org/abs/2608.09444) | 本文提出了首个针对循环语言模型深度自适应推理的高效方法——连续深度批处理（CDB），通过在循环步骤之间动态重组批次、管理循环KV缓存并提前预测token退出时机，解决了不同循环深度token无法被标准批处理系统高效处理的问题。 |
| [^249] | [Aftab: A Comprehensive Benchmark of CNN Encoders and Advanced Value Functions in Parallelized Q-Networks](https://arxiv.org/abs/2608.07335) | 本文系统评估了八种CNN编码器在并行化Q网络中的性能，并结合Hadamax编码与多种价值函数头，提出了一个在Atari-57上表现优异的复合架构。 |
| [^250] | [Not Every Divergence Should Be Suppressed: Counterfactual Recoverability in On-Policy Distillation](https://arxiv.org/abs/2608.04408) | 本文提出反事实可恢复性框架，通过教师续写与回滚分支重放错误状态来区分可恢复与不可逆的错误，并据此决定在线策略蒸馏中对轨迹的保留、回滚或常规监督策略，其可恢复性代理指标AUC达1.000，远超仅依赖发散度指标的0.392。 |
| [^251] | [DAIF: A Data-Driven Intermediate Fusion Framework for Multimodal Supervised Learning via Approximate Message Passing](https://arxiv.org/abs/2608.02769) | DAIF提出了一种数据自适应的中间融合框架，结合随机矩阵理论与非参数依赖性度量，通过根据模态间依赖性对模态聚类并进行经验贝叶斯先验估计，直接从数据中学习融合结构，克服了传统预定义融合架构无法适应模态间真实依赖关系的缺陷。 |
| [^252] | [Regularizing modality contribution drift in multimodal continual learning](https://arxiv.org/abs/2607.27260) | 该论文首次提出“模态贡献漂移（MCD）”概念及其量化评分，揭示模态贡献变化是多模态持续学习中遗忘的关键成因，并设计了相应的持续正则化方法来有效缓解遗忘。 |
| [^253] | [On the robustness of noisy solutions in non-convex neural networks](https://arxiv.org/abs/2607.27000) | 本文将零温下限制算法可达性的重叠间隙性质（OGP）推广至有限温度情形，证明了冻结一步复制对称破缺解在任意有限温度下依然存在，并给出基于单模式吉布斯权重光滑性的一般性判据，从而刻画了非凸神经网络中噪声解的鲁棒性。 |
| [^254] | [Think Short, Defer Smart, Act, and Repeat: Calibrated Reasoning and Uncertainty-Aware Deferral for Edge LLM Agents](https://arxiv.org/abs/2607.26865) | TSDS框架通过轻量级收敛探针和基于困惑度的委托规则，在边缘LLM代理中实现推理预算与可靠性的平衡，并利用多目标LTT程序提供同时的有限样本保证。 |
| [^255] | [Blind, Not Weak: A Best-of-Suite Safety-Utility Frontier for Recover-and-Reguard Defenses Against Encoded VLM Jailbreaks](https://arxiv.org/abs/2607.26574) | 该论文构建了一个“恢复-重防”预处理器，在安全防护器之前恢复图像内容并解码编码，将图像渲染类越狱攻击的拦截率从零提升至67-90%，并据此刻画了此类防御在安全性与良性流量效用之间的套件级最优权衡边界。 |
| [^256] | [CT-Merging: Consensus Directions and Task-Specific Scaling for LoRA Adapter Merging](https://arxiv.org/abs/2607.20561) | 提出 CT-Merging 方法，通过平均任务子空间投影器估计共识方向，并为每个任务分配独立的残差缩放因子，从而在 LoRA 适配器合并任务上取得优于现有基线的平均与最差任务准确率。 |
| [^257] | [DreamSat-Pose: Spacecraft Pose Estimation from Single-View 3D Reconstructions and Learned 2D-3D Feature Matching](https://arxiv.org/abs/2607.13449) | 本文提出 DreamSat-Pose 框架，仅凭单张图像即可对未知航天器同时完成三维形状重建与六自由度位姿估计，其核心创新在于结合冻结的 DINOv3 图像特征、动态图卷积点云几何特征与双流 Transformer 匹配器，学习密集二维-三维对应关系并由 PnP 求解器恢复位姿。 |
| [^258] | [Influence Diagnostics in High-dimensional M-estimation: Precise Asymptotics](https://arxiv.org/abs/2607.09250) | 该论文在高维凸M估计中精确刻画了训练点留一影响的渐近分布，发现有影响力的样本平均而言倾向于靠近决策边界，与主动学习中的数据选择启发式方法相契合。 |
| [^259] | [Bringing Agentic Search to Earth Observation Data Discovery](https://arxiv.org/abs/2607.02387) | 该论文提出了一个基于NASA地球观测知识图谱的智能体搜索框架用于地球科学数据发现，构建了包含47k查询-数据集对的开放基准NASA-EO-Bench，并通过微调神经评分器与BM25分数融合，将R@10和MRR提升至余弦基线的5倍以上。 |
| [^260] | [Beyond Drug Discovery: The Nanotechnology Molecular Optimization (NMO) Benchmark](https://arxiv.org/abs/2606.30170) | 提出纳米技术分子优化基准，用量子模拟取代代理预测器并引入严格协议，将生成式分子设计从药物发现领域拓展至量子材料科学与纳米技术研究。 |
| [^261] | [TeDiServe: High SLO Attainment Serving for Diffusion Language Models](https://arxiv.org/abs/2606.29094) | TeDiServe是一个面向扩散语言模型的集群级服务系统，通过截止时间感知调度、基于置信度阈值调整的自适应负载控制以及动态重新配置，在满足延迟SLO的同时实现高吞吐量服务。 |
| [^262] | [Statistically Valid Post-Training Hyperparameter Selection: From Tuning to Guarantees](https://arxiv.org/abs/2606.25601) | 提出以“先学习后测试”（LTT）范式为核心的统一统计框架，将训练后超参数选择转化为多元假设检验问题，为人工智能系统部署中的超参数调优提供正式的可靠性统计保证。 |
| [^263] | [ConSolv: Solvent-Conditional Machine Learning Implicit Solvent Potential](https://arxiv.org/abs/2606.24983) | ConSolv提出了一种溶剂条件化的机器学习隐式溶剂势，通过基于注意力的溶剂嵌入模块显式纳入溶剂效应，实现了在66种常见有机溶剂上的迁移与对未见溶剂的泛化，并在溶剂化自由能基准上超越了经典显式溶剂方法和从头算隐式溶剂方法。 |
| [^264] | [Do Location Encoders Capture Spatial Effects? A GeoShapley Benchmark Across Scales](https://arxiv.org/abs/2606.23453) | 该论文提出以GeoShapley博弈论解释器为工具，在三个空间尺度上对TorchSpatial框架中的十一种位置编码器进行基准测试，系统评估其嵌入能否恢复已知的空间变化系数，并揭示恢复效果随尺度与编码器架构的变化规律。 |
| [^265] | [NAC: Neural Action Codec for Vision-Language-Action Models](https://arxiv.org/abs/2606.21372) | 该论文提出神经动作编解码器（NAC），借鉴神经音频编解码器的设计思想，将机器人动作轨迹视为多通道一维信号并用多尺度RVQGAN架构进行高保真压缩，为视觉-语言-动作模型提供紧凑且有序的离散动作词元空间。 |
| [^266] | [ThousandWorlds: A benchmark for climate emulation of potentially habitable exoplanets](https://arxiv.org/abs/2606.18338) | ThousandWorlds是一个机器学习就绪的系外行星气候模拟基准数据集，包含来自五个全球气候模型的约1800次模拟，旨在突破传统气候模拟的计算瓶颈，加速对潜在宜居系外行星大气的理解与生命信号解读。 |
| [^267] | [Amortized quadrature for posterior expectations in inverse problems](https://arxiv.org/abs/2606.15871) | 本文提出“求积场”——一种集合等变神经网络，只需在一个后验族上训练一次，即可对任意观测、任意样本数 M 和任意被积函数，通过一次前向传播生成带符号权重的 M 节点求积格式，在保证精度不劣于蒙特卡洛的同时，避免了传统设计求积法需对每个新观测重复求解优化问题的高昂计算成本。 |
| [^268] | [Policy Regret for Embedding Model Routing: Contextual Bandits with Low-Rank Experts](https://arxiv.org/abs/2606.14929) | 该论文将嵌入模型路由形式化为具有低秩专家的对抗性上下文线性赌博机问题，证明标准后悔度量存在结构性误设或统计不可处理的缺陷，并提出兼具表达能力与高效可学习性的对数二次策略类来实现查询依赖的模型路由。 |
| [^269] | [Genetic Algorithms with Optimization Guided Operators](https://arxiv.org/abs/2606.12279) | 该论文提出了带优化引导算子的遗传算法通用模型，将其中优化问题形式化为基于强化学习语言的查询复杂度问题，并揭示了机器学习驱动的变异与重组算子虽更可能改进目标但计算成本更高的基本权衡。 |
| [^270] | [Critic Architecture Matters: Dual vs. Unified Critics for Humanoid Loco-Manipulation](https://arxiv.org/abs/2606.11891) | 人形机器人运动-操作多目标强化学习中，评论家架构的选择至关重要：双评论家在标准化评估中比统一评论家快 3.5 倍、吞吐量翻倍，但相当一部分差距源于评估中手指控制方式设置的不同。 |
| [^271] | [Bergson: An Open Source Library for Data Attribution](https://arxiv.org/abs/2606.11660) | Bergson 是一个开源数据归因库，支持扩展至超大规模语言模型和预训练数据集，并首次开源实现了 MAGIC、SOURCE 和 TrackStar 三种前沿数据归因方法。 |
| [^272] | [INFUSER: Influence-Guided Self-Evolution Improves Reasoning](https://arxiv.org/abs/2606.09052) | INFUSER提出了一种影响力引导的自我进化框架，通过生成器与求解器的协同训练，利用优化器感知的影响力分数来改进问题生成，从而显著提升推理能力。 |
| [^273] | [QueryGraph: Reliable Multi-Tool Query Execution Planning via LLM-Based Graph Generation](https://arxiv.org/abs/2606.08300) | 该论文提出QueryGraph系统，将自然语言查询转换为结构化图并通过确定性规划器结合深度优先搜索执行，实现了可靠的跨工具多步骤查询，即使使用小型或本地部署的LLM也能达到高准确率。 |
| [^274] | [Proper Scoring Rules for Right-Censored Survival Data](https://arxiv.org/abs/2606.06393) | 提出了一个右删失生存数据的适当评分框架，通过先将预测分布经删失机制映射、再在导出的观测数据分布上应用适当评分，统一了右删失似然与IPCW型准则，并给出了CRPS、pinball损失和Brier评分的右删失版本。 |
| [^275] | [QASM-Eval: A Dataset to Train and Evaluate LLMs on OpenQASM-3 Beyond Quantum Circuits](https://arxiv.org/abs/2605.30358) | QASM-Eval 是首个用于训练和评估大语言模型生成 OpenQASM 3 面向硬件特性（如中途测量与经典反馈、精确时序及脉冲级控制）代码的综合数据集。 |
| [^276] | [Large Language Model Selection with Limited Annotations](https://arxiv.org/abs/2605.24981) | SELECT-LLM是首个大语言模型主动选择框架，通过基于期望信息增益的查询选择规则，仅需少量最具信息量的标注查询即可从开放或黑盒候选模型中识别出给定任务的最佳LLM。 |
| [^277] | [Smooth Piecewise Cutting for Neural Operator to Handle Discontinuities and Sharp Transitions](https://arxiv.org/abs/2605.19823) | 提出 Cut-DeepONet 两阶段训练框架，通过将求解域切割为平滑子区域、并将不连续性表示为高维空间中的边界，使神经算子能够高效处理偏微分方程解中的不连续性与尖锐过渡。 |
| [^278] | [MSAlign: Aligning Molecule and Mass Spectra representations for Metabolite Identification](https://arxiv.org/abs/2605.19752) | 本文提出轻量级模型MSAlign，通过对齐两个冻结的基础模型（质谱模型DreaMS与分子模型MolDeBERTa）实现最先进的代谢物鉴定性能，并以极小的计算开销通过分数融合策略进一步提升了检索效果。 |
| [^279] | [HiLiftAeroML: A High-Fidelity Computational Fluid Dynamics Dataset for High-Lift Aircraft Aerodynamics](https://arxiv.org/abs/2605.19565) | HiLiftAeroML 是首个面向增升构型飞机空气动力学的开放高保真 CFD 数据集，包含基于 NASA 通用研究模型增升构型的 1,800 个 GPU 加速大涡模拟案例，与风洞实验数据吻合良好。 |
| [^280] | [Federated Martingale Posterior Samping](https://arxiv.org/abs/2605.18554) | 该论文提出联邦鞅后验采样（FMP），通过客户端上传可训练数据嵌入、服务器集中运行预测采样器的一次性并行协议，摆脱了联邦贝叶斯方法对先验设定的依赖，在性能上与中心化方法高度一致并取得最低的期望校准误差。 |
| [^281] | [Attention Sinks and Outliers in Attention Residuals](https://arxiv.org/abs/2605.17887) | 该论文提出OASIS方法，通过令牌级与深度级的显式空路由和空耦合机制稳定双归一化注意力残差架构，抑制注意力汇聚与激活离群值，并从理论与实验上解释和缓解了AttnResidual的低比特量化敏感性问题。 |
| [^282] | [Tabular Imbalanced Learning: A Survey, Benchmark, and Practical Guide](https://arxiv.org/abs/2605.14915) | 本文对表格数据不平衡学习进行了系统综述并构建统一分类体系，同时推出了大规模实证基准TILBench，在57个数据集上标准化评估了40多种方法，发现没有任何单一方法能在所有场景下始终占优。 |
| [^283] | [Mixed neural posterior estimation for simulators with discrete and continuous parameters](https://arxiv.org/abs/2605.13551) | 本文提出混合神经后验估计方法，通过将联合后验分解为离散分量（自回归分类器）和连续分量（生成模型）并在单一仿真目标下联合训练，将神经后验估计扩展到了同时包含离散和连续参数的混合参数空间，同时提供了评估混合后验校准的诊断工具。 |
| [^284] | [Supervised Deep Multimodal Matrix Factorization for Interpretable Brain Network Analysis](https://arxiv.org/abs/2605.13312) | 提出有监督深度多模态矩阵分解（SD3MF），将对称非负矩阵三因子分解从无监督单图聚类推广为面向多模态脑图的有监督深度框架，通过层级化、部分式的可解释表示与数据驱动的模态融合，弥合了脑网络分析中预测精度与可解释性之间的权衡。 |
| [^285] | [Complex-Valued Phase-Coherent Transformer](https://arxiv.org/abs/2605.10123) | 提出相位一致Transformer（PCT），通过对L2归一化的复数查询-键相似度施加实值平滑门控，以无token竞争的注意力机制跨层保留相位信息，在中规模基准测试中一致优于标准softmax Transformer及其复数值对应模型。 |
| [^286] | [Transformers Can Implement Preconditioned Richardson Iteration for In-Context Gaussian Kernel Regression](https://arxiv.org/abs/2605.08475) | 本文从理论和实验上证明，标准 softmax 注意力 Transformer 的前向传播可通过实现预条件 Richardson 迭代来近似高斯核岭回归预测器，其中注意力负责跨词元的核算子运算、MLP 负责词元内的标量算术，并以 O(log(1/ε)) 的深度达到 ε 精度。 |
| [^287] | [Towards Interpretable Damage Detection based on Aerodynamic Pressure Measurements](https://arxiv.org/abs/2605.08187) | 提出利用新型无侵入、低成本的Aerosense气动压力传感系统，结合卷积神经网络从风洞实验数据中检测风力涡轮机叶片结构损伤并评估其严重程度，实现可解释的结构健康监测。 |
| [^288] | [Geometry-Aware Simplicial Message Passing](https://arxiv.org/abs/2605.06061) | 提出几何单纯形Weisfeiler–Lehman（GSWL）测试，将顶点坐标引入颜色细化过程，证明了几何感知单纯形消息传递方案的表达能力上界及其判别能力的可匹配性，并结合欧拉示性数变换给出了几何表达能力的完整刻画与近似框架。 |
| [^289] | [QuadraSHAP: Stable and Scalable Shapley Values for Product Games via Gauss-Legendre Quadrature](https://arxiv.org/abs/2605.05870) | 本文提出QuadraSHAP，证明乘积博弈中每个玩家的Shapley值可精确表示为一维积分，从而利用高斯-勒让德求积以仅需⌈d/2⌉个节点即可实现可证明精确、稳定且可扩展的高效计算。 |
| [^290] | [Gradient-Momentum Coupling: A Parameter-Space Proxy for Learning Progress](https://arxiv.org/abs/2605.05856) | 提出梯度-动量耦合（GMC），通过梯度与动量的归一化乘积在参数空间中衡量学习进度，相比基于预测误差的方法更能抵抗噪声干扰，并按改进速度而非难度对任务排序。 |
| [^291] | [Low-Cost Black-Box Detection of LLM Hallucinations via Dynamical System Prediction](https://arxiv.org/abs/2605.05134) | 该论文提出将大语言模型视为黑盒动力系统，利用Koopman算子理论分别对事实性与幻觉性响应的状态空间转移算子进行拟合，并通过差分残差评分实现无需二次采样或外部知识检索的低成本单次幻觉检测。 |
| [^292] | [How Long Does Infinite Width Last? Signal Propagation in Long-Range Linear Recurrences](https://arxiv.org/abs/2605.05113) | 本文推导出复高斯初始化下线性递归模型隐状态信号能量的精确有限宽度公式，并揭示了无限宽度近似仅在深度满足 $t=o(\sqrt n)$ 的亚临界区域有效，而在 $t\sim c\sqrt n$ 的临界区域会出现不可忽略的偏差。 |
| [^293] | [Persistent Homology of Time Series through Complex Networks](https://arxiv.org/abs/2605.01624) | 该论文提出了一个结合复杂网络与持续同调的统一时间序列分类流程，通过系统比较五种图构造发现没有任何单一构造普遍最优（最优图类型取决于信号的判别结构），且图距离度量是影响分类性能的首要设计因素。 |
| [^294] | [Unlocking the Forecasting Economy: A Suite of Datasets for the Full Lifecycle of Prediction Market: [Experiments \& Analysis]](https://arxiv.org/abs/2604.20421) | 本文提出了首个覆盖去中心化预测市场全生命周期六个阶段（市场创建、代币注册、交易、预言机交互、争议处理与最终结算）的持续同步数据集套件，解决了链上智能合约与链下数据源之间数据严重碎片化、难以整合追踪的关键难题。 |
| [^295] | [Below-ground Fungal Biodiversity Can be Monitored Using Self-Supervised Learning Satellite Features](https://arxiv.org/abs/2604.09818) | 该研究证明，利用自监督学习从卫星影像提取的特征可有效预测地下菌根真菌物种丰富度，其预测能力超越气候、土壤和土地覆盖等传统基线，并将监测空间分辨率提升了10,000倍。 |
| [^296] | [Identifying Causal Effects Using a Single Proxy Variable](https://arxiv.org/abs/2604.09135) | 该论文提出SPICE假设，证明在已知混杂因素生成单一（多维）代理变量机制的前提下因果效应可识别，将经典代理变量方法扩展到多维连续场景，并开发了适用于离散和连续处理的神经网络估计框架SPICE-Net。 |
| [^297] | [Neural Parameter Estimation of RC Thermal Building Models for Model Predictive Control](https://arxiv.org/abs/2604.05904) | 该论文提出一种将RC模型物理方程嵌入神经网络训练过程的神经参数估计方法，并通过多建筑数据预训练进一步提升精度，为建筑节能模型预测控制提供了无需初始猜测、计算高效且准确的RC参数估计方案。 |
| [^298] | [Combee: Scaling Prompt Learning for Self-Improving Language Model Agents](https://arxiv.org/abs/2604.04247) | Combee 提出了一种新颖的框架，通过有原则的并行扩展策略解决现有提示学习方法在高并行度下质量下降的问题，从而同时提升自我改进语言模型智能体提示学习的效率和效果。 |
| [^299] | [A Sharp Norm Inequality and Buzano's Inequality via Determinants](https://arxiv.org/abs/2604.01525) | 本文利用参数化二次型族的行列式结构，给出了联系三种基本范数的尖锐不等式的简短证明，证明了常数 $(1+\sqrt{n})/2$ 的最优性，并同时证明了实向量情形的 Buzano 不等式。 |
| [^300] | [On associative neural networks for sparse patterns with huge capacities](https://arxiv.org/abs/2603.26217) | 本文通过将高阶交互与稀疏联想记忆相结合，将Amari和Willshaw模型的存储容量提升至 $N^n/(\log N)^n$ 量级，并在交互阶数对数增长时实现超多项式的存储规模。 |
| [^301] | [On the Expressive Power of Transformers for Contextual Relations](https://arxiv.org/abs/2603.25860) | 本文基于概率与最优传输理论构建了数学框架，揭示了注意力归一化与最优传输的深刻联系——softmax归一化产生条件关系而Sinkhorn归一化产生联合关系——并证明了Transformer在表示上下文关系上的通用逼近能力。 |
| [^302] | [Spectral-Sphere-Constrained Hyper-Connections](https://arxiv.org/abs/2603.20896) | 针对双随机约束超连接存在的恒等退化、表达能力瓶颈和参数化开销三重局限，提出将残差矩阵约束在谱球流形上的谱球约束超连接，在保持恒等映射性质以稳定训练的同时，恢复跨流混合的谱自由度和表达能力。 |
| [^303] | [Layer-wise Target Propagation: Efficient Component Attribution through Target Centric Propagation](https://arxiv.org/abs/2603.19742) | 提出逐层目标传播（LTP）框架，仅需一次前向和一次反向传播即可在冻结的Transformer上忠实地追踪信息流，在模型组件数量方面实现O(1)时间复杂度的高效密集组件归因。 |
| [^304] | [Minimax and Adaptive Covariance Matrix Estimation under Differential Privacy](https://arxiv.org/abs/2603.19703) | 本文提出了在差分隐私下针对不同协方差矩阵类别的最优估计器，揭示了隐私约束与数据几何结构如何共同影响高维协方差估计的极小极大速率。 |
| [^305] | [Distribution-Conditioned Transport](https://arxiv.org/abs/2603.04736) | 提出了分布条件化传输（DCT）框架，通过将传输映射条件化于源分布和目标分布的学习嵌入表示上，实现对未见分布对的泛化，并支持利用单条件观测分布的半监督学习和多种底层传输机制。 |
| [^306] | [Scale-invariant Gaussian derivative residual networks](https://arxiv.org/abs/2603.02843) | 本文提出了一种可证明尺度不变的高斯导数残差网络，通过在高斯导数层中引入残差跳跃连接构建更深的网络，在显著提升精度的同时保持优异的跨尺度泛化能力，并为任意维度下的尺度协变与尺度不变性质提供了严格的数学证明。 |
| [^307] | [Latent Generative Solvers for Generalizable Long-Term Physics Simulation](https://arxiv.org/abs/2602.11229) | 本文提出潜在生成求解器（LGS），通过物理VAE压缩十二个PDE族到共享潜在流形、金字塔流强制Transformer进行流匹配生成，以及训练时输入加噪的稳定性保证，首次实现了跨异构PDE族的泛化能力与长时域自回归物理模拟稳定性的兼顾。 |
| [^308] | [Pseudo-Invertible Neural Networks](https://arxiv.org/abs/2602.06042) | 本文提出满射伪可逆神经网络（SPNN），将摩尔-彭若斯伪逆自然推广到非线性神经网络，并形式化非线性反投影（NLBP）方法，从而扩展了零样本逆问题的求解范围。 |
| [^309] | [Learning Where It Matters: Geometric Anchoring for Robust Preference Alignment](https://arxiv.org/abs/2602.04909) | 提出几何锚定偏好优化（GAPO），用当前策略的对抗性局部扰动作为动态几何感知锚点取代DPO的固定参考策略，并通过自适应重加权和锚点间隔机制，在噪声偏好监督下实现更鲁棒的大语言模型对齐。 |
| [^310] | [RAPTOR: Ridge-Adaptive Logistic Probes](https://arxiv.org/abs/2602.00158) | 提出了一种简单的L2正则化逻辑回归探针RAPTOR，通过验证集调优的岭回归强度从归一化权重中提取准确且方向稳定的概念向量，可用于大语言模型的“先探针后引导”激活引导流程。 |
| [^311] | [MeshGraphNet-Transformer: Scalable Mesh-based Learned Simulation for Solid Mechanics](https://arxiv.org/abs/2601.23177) | 提出了一种融合Transformer全局建模能力与MeshGraphNets几何归纳偏置的新架构MGN-T，通过物理注意力机制直接捕获长程物理相互作用，无需深层消息传递或网格粗化，即可在工业规模的高分辨率网格上实现高效固体力学仿真学习。 |
| [^312] | [Generative Modeling of Discrete Data Using Geometric Latent Subspaces](https://arxiv.org/abs/2601.21831) | 该论文提出一种几何潜在子空间框架，在类别分布乘积流形的指数参数空间中通过几何主成分分析（GPCA）学习高维离散数据的低维表示，并借助等距黎曼几何实现一致的流匹配生成建模。 |
| [^313] | [DeepFedNAS: Efficient Hardware-Aware Architecture Adaptation for Heterogeneous IoT Federations via Pareto-Guided Supernet Training](https://arxiv.org/abs/2601.15127) | DeepFedNAS提出了一种基于多目标适应度函数的两阶段联邦神经架构搜索框架，通过Pareto最优超网训练与免预测器搜索，为异构物联网设备高效生成硬件感知的定制化网络架构并大幅降低搜索成本。 |
| [^314] | [FLAME: Flow Enhanced Legendre Memory Models for General Time Series Forecasting](https://arxiv.org/abs/2512.14253) | FLAME是一种轻量级时间序列基础模型，通过在编解码阶段结合平移与缩放勒让德记忆变体来捕获数据归纳偏置并实现高效长程推理，并采用基于归一化流的预测头以生成式方式建模复杂分布，在多个基准测试中实现了高效且准确的通用时间序列预测。 |
| [^315] | [Demo: Generative AI helps Radiotherapy Planning with User Preference](https://arxiv.org/abs/2512.08996) | 本文提出一种仅依据用户自定义偏好即可预测三维剂量分布的生成式模型，使计划制定者能够个性化权衡危及器官与靶区之间的取舍，并在适应性上超越Varian RapidPlan。 |
| [^316] | [Models Got Talent: Identifying High Performing Wearable Human Activity Recognition Models Without Training](https://arxiv.org/abs/2511.06157) | 本研究证明零成本代理（ZCPs）可以在六个可穿戴人体活动识别基准数据集上，无需完整训练就能高效甄别出接近最优的模型架构，预测出的顶尖架构性能与大规模随机训练结果的差距不超过7%。 |
| [^317] | [Transformers Discover Molecular Structure Without Graph Priors](https://arxiv.org/abs/2510.02259) | 该研究表明，Transformer模型无需嵌入图结构或几何局部性等物理先验，仅通过数据学习就能自动发现分子结构等物理模式。 |
| [^318] | [What Do They Fix? LLM-Aided Categorization of Security Patches for Critical Memory Bugs](https://arxiv.org/abs/2509.22796) | 该论文提出利用大语言模型对Linux内核中修复关键内存漏洞（如越界访问和释放后使用）的安全补丁进行自动分类，以解决安全补丁难以识别、下游维护者采用延迟的问题。 |
| [^319] | [AUWave: A Data-Driven Model for Reconstructing Significant Wave Heights Using Sparse Observations](https://arxiv.org/abs/2509.19384) | 提出了AUWave混合深度学习框架，结合站点编码器与自注意力增强的多尺度U-Net，从稀疏浮标观测中高精度重建区域有效波高场，并通过浮标消融分析识别关键站点以指导海洋观测网络设计。 |
| [^320] | [Stochastic Bilevel Optimization with Heavy-Tailed Noise](https://arxiv.org/abs/2509.14952) | 本文提出了一种针对重尾噪声下随机双层优化的嵌套循环归一化随机双层近似算法（N²SBA），在噪声中心矩阶为 p∈(1,2] 的条件下，以 O(κ^((7p-3)/(p-1)) σ^(p/(p-1)) ε^(-(4p-2)/(p-1))) 的随机一阶预言机复杂度找到 ε-稳定点，并将该方法推广至非凸-强凹极小极大优化问题。 |
| [^321] | [Neural Bridge Processes](https://arxiv.org/abs/2508.07220) | 提出神经桥过程（NBP），用输入锚定的桥轨迹替代无条件前向核，使条件输入信息在扩散的含噪状态中就被编码，从而实现对随机函数更具表达力且强条件依赖的学习。 |
| [^322] | [DeepC4: Deep Conditional Census-Constrained Clustering for Large-scale Multitask Spatial Disaggregation of Urban Morphology](https://arxiv.org/abs/2507.22554) | 本文提出DeepC4，一种基于深度学习的人口普查约束聚类方法，用于在弱监督条件下解决城市形态大规模多任务空间分解中与人口普查数据的局部差异及模型不确定性传播的难题。 |
| [^323] | [LapDDPM: Spectral Perturbation Diffusion for Robust Single-Cell Manifold Generation](https://arxiv.org/abs/2506.13344) | 提出LapDDPM，一种结合谱对抗扰动机制的条件图扩散概率模型，通过谱扰动实现分布鲁棒优化，能够生成高保真且对结构噪声鲁棒的单细胞RNA测序数据。 |
| [^324] | [Evaluation is All You Need: Strategic Overclaiming of LLM Reasoning Capabilities Through Evaluation Design](https://arxiv.org/abs/2506.04734) | 本研究揭示评估条件的细微差异会导致Deepseek-R1-Distill系列等推理模型的基准测试结果大幅波动，使其声称的性能提升难以可靠复现，并倡导建立更严格的模型性能评估范式。 |
| [^325] | [ChemMLLM: Chemical Multimodal Large Language Model](https://arxiv.org/abs/2505.16326) | 提出了ChemMLLM，一个统一的化学多模态大语言模型，能够同时处理分子理解与生成任务，首次将化学多模态大模型的能力扩展到图像生成领域。 |
| [^326] | [Learning Operators by Regularized Stochastic Gradient Descent with Operator-valued Kernels](https://arxiv.org/abs/2504.18184) | 本文针对从波兰空间到可分希尔伯特空间的算子学习问题，分析了算子值核再生核希尔伯特空间中在线与有限时域两种设置下的正则化随机梯度下降算法，建立了对输出空间维度无显式依赖、且在期望意义下接近最优的误差界，并给出了高概率估计与几乎必然收敛的结论。 |
| [^327] | [AYLA: Architecting a loss landscape in shallow neural networks to accelerate feature recovery](https://arxiv.org/abs/2504.01875) | AYLA是一种损失重参数化框架，通过对损失施加sigmoid控制的幂律变换动态调节梯度大小，在不改变临界点位置的前提下，加速浅层神经网络特征学习在平坦和鞍点区域的下降，并稳定后期优化过程。 |
| [^328] | [LEAD: An EEG Foundation Model for Alzheimer's Disease Detection](https://arxiv.org/abs/2502.01678) | 本文构建了迄今最大的EEG-AD数据集（2,238名受试者），并提出首个脑电图阿尔茨海默病检测基础模型LEAD，其门控时-空Transformer可适应异构EEG数据，配合被试正则化训练策略提升了跨被试泛化能力。 |
| [^329] | [Foundations of Reinforcement Learning and Interactive Decision Making](https://arxiv.org/abs/2312.16730) | 该专著在统一的统计框架下系统阐述了从多臂老虎机到基于函数逼近的强化学习的交互式决策算法设计与复杂度理论，并展示了如何将监督学习方法转化为决策算法、分析其性能以及判定问题的可解性。 |
| [^330] | [A New Non-archimedean Metric on Persistent Homology](https://arxiv.org/abs/2012.02655) | 本文提出了一种适用于所有维度持久同调类的新非阿基米德度量——共表型度量，并证明其与层次聚类结合能提供统计上可验证的等价拓扑信息，且所得聚类在轮廓系数和兰德指数等评估指标上表现优异。 |
| [^331] | [A Fast Graph Search Algorithm with Dynamic Optimization and Reduced Histogram for Discrimination of Binary Classification Problem.](http://arxiv.org/abs/2401.04282) | 本研究提出了一种用于二分类问题的快速图搜索算法，通过动态优化和减少直方图的方法来提高区分结果。该算法在支持向量机模型的基础上应用，显著提高了真正例并减少了假正例。 |
| [^332] | [Differentially-Private Decision Trees and Provable Robustness to Data Poisoning.](http://arxiv.org/abs/2305.15394) | 本论文提出了一种名为PrivaTree的差分隐私决策树方法，通过使用私有直方图选择分割点来在隐私保护与模型效用之间取得更好的平衡。这种方法能够接收混合的数值和类别数据，并且能够在数据篡改方面表现出可靠性。 |

# 详细

[^1]: 无需学习停止而学会停止：自监督置信度训练提升推理效率

    Learning to Stop without Learning to Stop: Self-Supervised Confidence Training Improves Reasoning Efficiency

    [https://arxiv.org/abs/2609.31619](https://arxiv.org/abs/2609.31619)

    该论文发现，仅通过自监督方式训练模型在推理过程中间点预测自身置信度（损失函数中不含任何长度、效率或停止目标），就能让模型在推理时无需任何提前停止机制便自发提升推理效率。

    

    推理模型通常会生成非常长的推理轨迹，导致推理的计算成本很高。现有方法通常通过两种途径提升效率：一是在推理阶段引入提前停止机制，二是在训练过程中显式鼓励更短的推理，例如使用带长度惩罚的强化学习。我们证明，显著的效率提升可以来自另一种不同的监督信号：置信度。通过一种自监督流程，我们仅使用600个训练问题，对推理模型进行微调，使其在自身推理轨迹的中间点预测对答案的置信度。置信度仅被用作训练目标：损失函数中不包含任何关于推理长度、效率或停止的目标。在推理阶段，微调后的模型采用标准生成流程，无需置信度引导或提前停止机制。尽管如此，自监督……

    arXiv:2609.31619v1 Announce Type: new  Abstract: Reasoning models often generate very long reasoning traces, making inference computationally expensive. Existing approaches typically improve efficiency either through inference-time early-stopping mechanisms or by explicitly encouraging shorter reasoning during training, for example through reinforcement learning with length penalties. We show that substantial efficiency gains can instead emerge from a different kind of supervision: \textit{confidence}. Using a self-supervised procedure, we fine-tune reasoning models to predict their confidence in the answer at intermediate points along their own reasoning trajectories using only 600 training problems. Confidence is used only as a training target: the loss contains no objective for reasoning length, efficiency, or stopping. At inference, the fine-tuned models use the standard generation procedure, with no confidence elicitation or early-stopping mechanism. Despite this, self-supervised 
    
[^2]: 面向高斯数据的无间隙差分隐私主成分分析

    Gap-free Differentially Private PCA for Gaussian Data

    [https://arxiv.org/abs/2609.31614](https://arxiv.org/abs/2609.31614)

    该论文提出了一种无间隙的差分隐私算法，用于解决高斯数据下的主成分分析（PCA）问题。

    

    我们针对高斯数据的主成分分析（PCA）问题，提出了一种无间隙的差分隐私算法。

    arXiv:2609.31614v1 Announce Type: cross  Abstract: We give a gap-free differentially private algorithm for the principal component analysis (PCA) problem with Gaussian data.
    
[^3]: 反向扩散的一阶平稳性

    First-Order Stationarity of Reverse Diffusions

    [https://arxiv.org/abs/2609.31612](https://arxiv.org/abs/2609.31612)

    该论文为扩散模型建立了首个一阶平稳性理论，证明基于SDE的过阻尼与欠阻尼朗之万扩散的逆时流在前向过程平稳势强凸（仅针对加噪过程而非数据）时以指数速率收缩相对Fisher散度，并为离散化采样器建立了与非凸优化中平均梯度范数保证相对应的一阶平稳性界。

    

    近期研究文献表明优化与采样之间存在紧密联系。我们为扩散模型发展了相应的一阶理论。首先，只要前向过程的平稳势是强凸的——这是对所选加噪过程的条件，而非对数据的要求——基于SDE的过阻尼与欠阻尼朗之万扩散的逆时流就会以明确的指数速率收缩相对Fisher散度。这是基于SDE的反向扩散所独有的优势，而基于ODE的反向过程并不具备这一性质。其次，我们引入离散化分析，为过阻尼与欠阻尼扩散模型的采样器建立了平均一阶平稳性界——这是非凸优化中平均梯度范数保证在采样领域的对应物。与非凸优化类似，这种不依赖凸性假设的证明是局部性的：它保证的是得分的一致性，而非全局众数权重。

    arXiv:2609.31612v1 Announce Type: new  Abstract: Recent literature has shown a strong connection between optimization and sampling. We develop the corresponding first-order theory for diffusion models. First, the SDE-based reverse-time flows of overdamped and underdamped Langevin diffusions contract relative Fisher divergences at explicit exponential rates whenever the stationary potential of the forward process is strongly convex---a condition on the noising process one chooses, not on the data. This is a unique advantage of SDE-based reverse diffusion, absent in the reverse process based on ODEs. Second, we incorporate discretization and establish averaged first-order stationarity bounds---the sampling analog of averaged gradient-norm guarantees in nonconvex optimization---for samplers of both overdamped and underdamped diffusion models. As in nonconvex optimization, the convexity-free certificate is local: it guarantees score consistency, not global mode weights.
    
[^4]: 通过输出后处理实现黑盒生成式AI的统计属性对齐

    Statistical attribute alignment for black-box generative AI via output post-processing

    [https://arxiv.org/abs/2609.31607](https://arxiv.org/abs/2609.31607)

    本文针对黑盒生成式AI提出了一种输出后处理方法，通过最小化查询次数的算法将生成输出的属性分布与用户指定目标对齐，并在精确与近似对齐两种情形下证明了算法的最优性。

    

    生成式AI系统的使用日益广泛，但使其输出与用户需求保持一致仍是一项持续的挑战。本文旨在确保AI生成输出的某个属性分布与用户指定的目标相一致。这一问题的动机来自诸如公平性等应用场景——在此场景中我们希望受保护属性（如性别、种族或年龄类别）遵循期望的分布；以及合成数据生成场景——在此场景中我们希望生成的数据能够代表目标分布。我们研究了实际中十分重要的黑盒访问设置，即用户可以重复查询生成式AI模型。目标是返回 $m\ge 1$ 个输出，使其联合属性分布尽可能接近该目标。对于精确对齐和近似对齐两种情形，我们开发了能够最小化生成器期望查询次数的算法，并进一步证明了当输出数量趋于无穷大时这些算法的最优性。

    arXiv:2609.31607v1 Announce Type: cross  Abstract: Generative AI systems are increasingly used, but aligning their outputs with user requirements poses a continuing challenge. Here, we aim to ensure that the distribution of an attribute of an AI-generated output aligns with a user-specified target. This is motivated by examples such as fairness, where we want to ensure that a protected attribute (e.g., gender, race, or age categories) follows a desired distribution, and synthetic data generation, where we want the generated data to be representative of a target distribution. We study the practically important black-box access setting, where a user can repeatedly query a generative AI model. The goal is to return $m\ge 1$ outputs whose joint attribute distribution is as close as possible to this target. For both exact and approximate alignment, we develop algorithms that minimize the expected number of queries to the generator, and we further demonstrate their optimality as the number o
    
[^5]: 通过信念自蒸馏提取用户模型

    User Model Extraction via Belief Self-Distillation

    [https://arxiv.org/abs/2609.31603](https://arxiv.org/abs/2609.31603)

    提出信念自蒸馏框架，使冻结的大语言模型从自然对话中自我蒸馏出既能读出又能写回的用户信念表示，并揭示模型的拒绝行为取决于其推断出的用户意图。

    

    大型语言模型（LLM）会隐式地推断用户的属性并据此调整自身行为，然而这些信念一直难以被检查和进行因果性操纵。我们提出了信念自蒸馏，这是一个统一的读写框架，通过学习一种既可被解码、又可写回模型的紧凑用户表示，将线性探测与因果探测连接起来。冻结的LLM充当自己的教师，从自然对话中蒸馏出信念，无需任何外部标注。与传统的探测方法不同，BSD不仅分离出激活中存在的信息，还分离出一种其因果作用可被直接检验的状态。在多个模型家族上的实验表明，BSD能够忠实地恢复用户信念，并实现比匹配的隐藏状态引导强得多的干预效果。至关重要的是，我们发现模型的拒绝行为不仅取决于请求本身，还取决于模型推断出的用户意图：改变这一信念即可改变拒绝行为。

    arXiv:2609.31603v1 Announce Type: cross  Abstract: Large language models (LLMs) implicitly infer attributes of their users and adapt their behavior accordingly, yet these beliefs remain difficult to inspect and causally manipulate. We introduce Belief Self-Distillation (BSD), a unified read-write framework that bridges linear and causal probing by learning a compact user representation that can be both decoded and written back into the model. The frozen LLM acts as its own teacher, distilling beliefs from natural conversations without external annotations. Unlike conventional probing, BSD isolates not only information present in activations, but a state whose causal role can be directly tested. Across multiple model families, BSD faithfully recovers user beliefs and enables substantially stronger interventions than matched hidden-state steering. Crucially, we find that refusal depends not only on the request, but on the model's inferred user intent: changing this belief alters refusal 
    
[^6]: 新的LoRA技能应当只读取而绝不写入

    New LoRA Skills Should Read but Never Write

    [https://arxiv.org/abs/2609.31600](https://arxiv.org/abs/2609.31600)

    提出READ方法，通过将各适配器重写为规范化形式并强制新技能对旧技能“只读不写”的耦合方向，使多个独立训练的LoRA适配器能够组合成单一模型而不破坏旧技能的原有计算。

    

    低秩适配器使得对大语言模型按任务微调变得十分廉价，但将多个独立训练的适配器组合成一个模型仍然困难：在权重空间中合并更新会产生干扰，在全部任务数据上重新训练代价高昂，而在独立适配器之间进行路由则放弃了构建单一组合模型的目标。我们将这一困难归因于每种组合方法都隐式做出的两个选择。一个LoRA更新存在无穷多种等价分解方式，当适配器单独使用时这种选择是不可见的，但它决定了适配器之间学习到的交互能够“看到”什么；旧技能与新技能之间的耦合同样可以指向任一方向，而方向决定了旧技能能否保持其原有的计算结果。我们提出READ（Read-only Expansion of Adapter Deltas，适配器增量的只读扩展），它同时固定了这两个选择：每个适配器被重写为平衡的……（原文摘要在此处截断）

    arXiv:2609.31600v1 Announce Type: new  Abstract: Low-rank adapters (LoRA) make it cheap to fine-tune a large language model once per task, but combining several independently trained adapters into one model remains difficult: merging the updates in weight space causes interference, retraining on all task data is expensive, and routing between separate adapters gives up the goal of a single combined model. We trace the difficulty to two choices that every composition method makes implicitly. A LoRA update admits infinitely many equivalent factorizations; the choice among them is invisible while an adapter serves alone, but it determines what a learned interaction between adapters can see. A coupling between an old skill and a new one can likewise point in either direction, and the direction decides whether the old skills keep computing what they computed before. We introduce READ (Read-only Expansion of Adapter Deltas), which fixes both choices: each adapter is rewritten into a balanced
    
[^7]: 直接反馈对齐中的共模崩溃与恢复

    Common-Mode Collapse and Recovery in Direct Feedback Alignment

    [https://arxiv.org/abs/2609.31589](https://arxiv.org/abs/2609.31589)

    该研究揭示了直接反馈对齐（DFA）训练停滞的根源在于误差的共模分量通过秩一更新驱动tanh单元饱和，并构建了一个无需拟合参数的简化模型来准确预测这种崩溃现象及其恢复动态。

    

    直接反馈对齐（DFA）通过输出误差的固定随机投影来训练隐藏层。当使用tanh隐藏单元和独立的sigmoid输出时，普通随机梯度下降可能停滞在接近类别频率常数预测器的损失水平。我们将这种停滞归因于误差的共模，即在所有输入之间共享的分量。一个精确的均值-协方差分解分离出了一个由平均教学信号和平均突触前活动形成的秩一更新，其主导分量会驱动tanh单元趋向饱和。在初始化时，随机反馈平均而言无法对共享误差提供系统性纠正；读出层的学习限制了这种停滞的持续时间。一个从网络初始化、无需拟合参数的简化模型，在48种设置下成功预测了激活敏感性的集中现象。在MNIST数据集上，类别可解码性在崩溃后基本得以保留，但在固定学习率下读出层的学习仍然缓慢。Adam能够学习……（原文摘要在此处截断）

    arXiv:2609.31589v1 Announce Type: new  Abstract: Direct feedback alignment (DFA) trains hidden layers through fixed random projections of output error. With tanh hidden units and independent sigmoid outputs, plain stochastic gradient descent can stall near the loss of a constant predictor of class frequencies. We trace this stall to the error's common mode, the component shared across inputs. An exact mean-covariance decomposition separates a rank-one update formed by the mean teaching signal and mean presynaptic activity. Its leading component drives tanh units toward saturation. At initialization, random feedback provides no systematic correction of the shared error on average; readout learning limits its duration. A reduced model initialized from the network, without fitted parameters, predicts the concentration of activation sensitivity across 48 settings. On MNIST, class decodability largely survives collapse, but readout learning remains slow at a fixed learning rate. Adam learns
    
[^8]: 信任引导的决策Transformer

    Trust Guided Decision Transformer

    [https://arxiv.org/abs/2609.31586](https://arxiv.org/abs/2609.31586)

    提出信任引导决策Transformer（TGDT），利用模型自身经保形预测校准的下一状态预测误差先筛选可信上下文、再由冻结评论家选择最高价值动作，以避免长回合 rollout 中上下文漂移导致的性能退化。

    

    决策Transformer在长回合 rollout 上的性能会下降，因为其条件化上下文会漂移出训练分布。我们表明，这种漂移可以通过模型自身的下一状态预测误差来观察：该误差在 rollout 过程中升高并持续保持高位，从而给出上下文何时变得不可靠的直接信号。我们提出了信任引导决策Transformer（TGDT），它在应用价值引导之前先进行上下文选择。在每一步，TGDT 使用滚动的下一状态预测误差评估多个最近的上下文后缀，并通过分裂保形预测针对留出的离线数据进行校准。它只保留误差处于校准阈值内的后缀，然后使用冻结的评论家在这些可信后缀中选择价值最高的动作。这颠倒了仅价值弹性选择所使用的顺序——在后者的方式中，评论家可能会选择由模型本身已标记为不可靠的上下文所生成的动作。

    arXiv:2609.31586v1 Announce Type: new  Abstract: Decision Transformer performance degrades on long rollouts because the conditioning context drifts out of the training distribution. We show that this drift is visible through the model's own next state prediction error, which rises during rollout and stays elevated, giving a direct signal of when context has become unreliable. We introduce Trust Guided Decision Transformer (TGDT), which selects context before applying value guidance. At each step, TGDT evaluates several recent context suffixes using rolling next state prediction error, calibrated against held out offline data via split conformal prediction. It keeps only suffixes whose error stays within the calibrated threshold, then uses a frozen critic to choose the highest value action among the trusted suffixes. This reverses the order used by value only elastic selection, where the critic may choose an action generated from a context the model itself has flagged as unreliable. Exp
    
[^9]: 深度粗糙波动率中的不确定性与可解释性：一种神经信息论后验方法

    Uncertainty and Explainability in Deep Rough Volatility: A Neural Information-Theoretic Posterior Approach

    [https://arxiv.org/abs/2609.31570](https://arxiv.org/abs/2609.31570)

    该论文提出一个基于仿真的推断框架，通过神经比率估计学习粗糙Heston模型参数在隐含波动率曲面条件下的后验分布，结合异方差神经代理定价器生成考虑不确定性的价格区间，并引入信息论可解释性方法Hellinger-SHAP。

    

    深度学习极大地加速了复杂随机波动率模型的校准，但仅靠神经点校准无法捕捉在观察到隐含波动率（IV）曲面后仍然存在的不确定性。我们开发了一个基于仿真的推断框架，用于粗糙Heston（rHeston）模型的校准，该框架学习在给定IV曲面条件下模型参数的后验分布。利用神经比率估计，我们获得了校准后的后验样本，这些样本可以通过异方差神经代理定价器进行传播，用于路径依赖的奇异期权定价。由此产生的后验预测分布结合了残余参数不确定性与条件代理不确定性，并生成考虑不确定性的价格区间。我们进一步提出了Hellinger-SHAP，这是一种用于后验推断的信息论可解释性方法。它不再归因于单一的参数点估计，而是应用于……

    arXiv:2609.31570v1 Announce Type: new  Abstract: Deep learning has substantially accelerated the calibration of complex stochastic-volatility models, but neural point calibration alone does not capture the uncertainty remaining after an implied-volatility (IV) surface has been observed. We develop a simulation-based inference framework for rough Heston (rHeston) calibration that learns the posterior distribution of the model parameters conditional on an IV surface. Using neural ratio estimation, we obtain calibrated posterior samples that can be propagated through heteroscedastic neural surrogate pricers for path-dependent exotic options. The resulting posterior-predictive distributions combine residual parameter uncertainty with conditional surrogate uncertainty and yield uncertainty-aware price intervals.   We further introduce Hellinger-SHAP, an information-theoretic explainability method for posterior inference. Rather than attributing a single parameter point estimate, it applies 
    
[^10]: 权重对编码：在神经网络权重中诱导更小的语法

    Weight Pair Encoding: Inducing a Smaller Grammar in Neural Network Weights

    [https://arxiv.org/abs/2609.31564](https://arxiv.org/abs/2609.31564)

    提出WeightPE方法，通过将有损Re-Pair压缩器嵌入直通估计器中来显式微调网络权重使其形成更小的语法结构，在ViT模型上产生的语法大小仅为int8 QAT的约0.4倍，且准确率损失仅1-2个百分点。

    

    我们证明神经网络权重可以被显式地微调，以适应一个更小的语法。权重对编码通过将有损的Re-Pair压缩器置于直通估计器内部来实现这一点。网络的int8权重被展平为一个字符串，并在全局L2预算内将近似匹配的Re-Pair模式变为完全相等。网络使用重写后的权重进行计算，并通过直通估计器利用这些权重进行训练。与由固定大小条目组成的扁平码本不同，语法提供可变长度的模式，并在更大的模式中分层地复用它们。在CIFAR-10上微调的ViT-B/16和ViT-L/16的MLP权重上，WeightPE产生的Re-Pair语法大小分别是等效int8 QAT运行所产生语法的0.43倍和0.38倍，代价分别为1.9和1.1个准确率百分点。这一趋势还扩展到不同的语法压缩器（LZ78、SEQUITUR），即使网络并未针对这些压缩器进行过微调。

    arXiv:2609.31564v1 Announce Type: new  Abstract: We show that neural network weights can be explicilty fintuned to admit a smaller grammar. Weight Pair Encoding (WeightPE) does so by placing a lossy Re-Pair compressor inside a straight-through estimator. The int8 weights of the network are flattened into one string, and near-matching Re-Pair patterns are made exactly equal within a global L2 budget. The network computes with the rewritten weights and trains through them with a straight-through estimator. Unlike a flat codebook of fixed-size entries, a grammar offers variable-length patterns and reuses them hierarchically inside larger ones. On the MLP weights of ViT-B/16 and ViT-L/16 finetuned on CIFAR-10, WeightPE produces a Re-Pair grammar 0.43x and 0.38x the size of the one produced by an equivalent int8 QAT run, at a cost of 1.9 and 1.1 accuracy points. The trend extends to different grammar compressors (LZ78, SEQUITUR), over which the networks has not be finetuned against. To our 
    
[^11]: OPTQ 的泛化行为与正则化的作用

    Generalization behavior of OPTQ and the role of regularization

    [https://arxiv.org/abs/2609.31560](https://arxiv.org/abs/2609.31560)

    该论文在泛化设定下分析了 OPTQ 及其随机变体量化算法，证明了测试分布下期望量化误差的界，并揭示了正则化项在控制泛化误差中的关键作用。

    

    大型神经网络可以通过将其权重舍入或“量化”为可用更少比特表示的数值来压缩。OPTQ 是一种量化算法，它逐步量化神经网络的权重，使得在指定校准数据集上的平方量化误差尽可能小。我们在泛化设定下研究了 OPTQ 及其变体算法——随机 OPTQ 的性能，并推导了当测试点从固定分布中抽取时该算法所累积的期望平方误差的界。我们证明了两个结果：一个结果将泛化误差与校准数据集（由与测试分布相同的分布中独立采样得到）上的误差联系起来；另一个结果则对所有足够“良好”的分布（无论校准数据集如何）界定了随机 OPTQ 的泛化误差。在这两个结果中，正则化项 $\lambda$...（摘要原文在此处截断）

    arXiv:2609.31560v1 Announce Type: new  Abstract: Large neural networks can be compressed by rounding or "quantizing" their weights to numbers that admit representations with fewer bits. One algorithm for quantization, OPTQ, progressively quantizes the weights of a neural network so that the squared quantization error on a specified calibration dataset is as small as possible. We study the performance of OPTQ and a variant algorithm, stochastic OPTQ, in a generalization setting and derive bounds for the expected squared error accrued by the algorithm when a test point is drawn from a fixed distribution. We prove two results. One result relates the generalization error to the error on a calibration dataset comprising independent samples from the same distribution as the test distribution. The other result bounds the generalization error of stochastic OPTQ for all sufficiently nice distributions, regardless of the calibration dataset. In both of these results, the regularization term $\la
    
[^12]: 基于学习的潜在贝叶斯跟踪的在线学习

    Online Learning via Learned Latent Bayesian Tracking

    [https://arxiv.org/abs/2609.31559](https://arxiv.org/abs/2609.31559)

    提出AURA元学习框架，通过离线学习低维潜在状态空间模型，解决了基于贝叶斯滤波的在线学习中缺乏合适低维动力学表示的核心瓶颈，从而实现非平稳环境下模型的快速在线适应。

    

    非平稳环境下的在线学习要求模型在严格的计算约束下从流式数据中快速适应。一种有原则的方法是将在线学习视为贝叶斯状态跟踪，其中模型参数通过贝叶斯滤波进行顺序更新。然而，由于参数空间的高维性，将贝叶斯滤波器直接应用于现代深度模型在计算上是不可行的，这迫使现有方法依赖于受限的近似或手动设计的低维子空间。在这项工作中，我们将缺乏合适的低维动力学表示确定为基于贝叶斯滤波的在线学习的核心瓶颈。因此，我们提出了通过表示适应实现自适应更新（AURA），这是一种元学习框架，可离线学习一个低维潜在状态空间模型，该模型控制分布偏移下最优模型参数的演化。

    arXiv:2609.31559v1 Announce Type: new  Abstract: Online learning in non-stationary environments requires models to adapt rapidly from streaming data under strict computational constraints. A principled approach casts online learning as Bayesian state tracking, where model parameters are updated sequentially via Bayesian filtering. However, applying Bayesian filters directly to modern deep models is computationally prohibitive due to the high dimensionality of parameter space, forcing existing methods to rely on restrictive approximations or manually designed low-dimensional subspaces. In this work, we identify the absence of a suitable low-dimensional dynamical representation as the core bottleneck in Bayesian filtering-based online learning. Accordingly, we propose Adaptive Update through Representation Adaptation (AURA), a meta-learning framework that learns offline a low-dimensional latent state-space model governing the evolution of optimal model parameters under distribution shift
    
[^13]: EAServe：面向多模态大语言模型的编码感知分离式服务系统

    EAServe: Encode-Aware Disaggregated Serving for Multimodal Large Language Models

    [https://arxiv.org/abs/2609.31551](https://arxiv.org/abs/2609.31551)

    EAServe提出了一种编码感知的分离式服务框架，将编码阶段重新定位为EPD流水线的控制点，从而解决多模态大语言模型服务中编码GPU利用率低下及下游预填充与解码资源饥饿的结构性失衡问题。

    

    将预填充和解码这两个阶段分离到独立的GPU资源池，如今已成为（纯文本）大语言模型服务的标准优化方法。然而，多模态大语言模型（MLLM）增加了第三个阶段——编码，这给资源分配带来了新的挑战。编码阶段将图像、视频或音频转换为语言模型可以使用的嵌入向量，从而形成了编码-预填充-解码（EPD）三阶段流水线。现有框架只能提供部分解决方案：纯文本的PD系统缺乏编码功能，而EPD框架则将编码作为独立服务提供，却未对下游请求流进行调控。该流水线还存在结构性资源失衡问题：每个请求都必须先经过编码阶段才能开始下游工作，但逐请求的执行方式即使在高负载下也会使编码GPU严重利用不足，导致下游的预填充和解码工作节点处于饥饿状态。针对这一问题，我们将编码重新定位为EPD流水线的控制点，提出了三个紧密……

    arXiv:2609.31551v1 Announce Type: cross  Abstract: Disaggregating the two stages, Prefill and Decode, onto separate GPU pools is now a standard optimization for (text-only) LLM serving. However, multimodal LLMs (MLLMs), which add a third phase, Encode, pose new challenges for resource allocation. Encode turns images, video, or audio into embeddings that the language model can consume, yielding a three-stage Encode-Prefill-Decode (EPD) pipeline. Existing frameworks offer only partial answers: text-only PD systems lack Encode, while EPD frameworks expose it as a separate service without regulating downstream request flow. The pipeline also carries a structural resource imbalance: every request enters through Encode before downstream work can begin, yet per-request execution leaves the encode GPU severely underutilized even at high loads, starving the downstream Prefill and Decode workers. Addressing this, we reposition Encode as the control point of the EPD pipeline, exposing three tight
    
[^14]: BeatGraph：面向家庭环境下婴儿心电图表示的自监督心跳图方法

    BeatGraph: Self-Supervised Heartbeat Graphs for Infant ECG Representations from the Home Environment

    [https://arxiv.org/abs/2609.31546](https://arxiv.org/abs/2609.31546)

    BeatGraph提出以心跳为基本单元的自监督图表示方法，将30秒婴儿ECG窗口建模为心跳图，克服了固定片段切分忽略心脏结构的问题，更适合婴儿高心率及家庭环境下的心电图建模。

    

    心电图（ECG）基础模型通常将信号切分为固定长度的片段进行标记化处理，这种方式忽略了心脏结构，因此一个片段可能会把一次心跳截断，并且每个片段中包含的心跳数量会随心率变化而改变。这一问题对婴儿尤为关键，因为婴儿心率更高，且其心电图与这些模型所依赖的成人临床环境记录的12导联数据存在差异。因此，婴儿心电图模型应当直接对心跳进行推理，而不是从任意片段中恢复心跳。我们提出了BeatGraph，它以心跳作为表示的基本单元，将每个30秒的窗口建模为心跳构成的图。一个共享的心跳编码器根据心跳波形和心跳间间隔对每次心跳进行嵌入，带有位置编码的Transformer按时间顺序排列心跳，残差图注意力层使每个心跳与其他所有心跳相互关联，最后通过注意力池化得到窗口嵌入。我们在新建的无标注语料库上对BeatGraph进行预训练。

    arXiv:2609.31546v1 Announce Type: new  Abstract: Electrocardiogram (ECG) foundation models typically tokenize the signal into fixed-length patches that ignore cardiac structure, so a patch may split a heartbeat and the number of beats in each patch shifts with heart rate. This matters most for infants, whose heart rates are higher and whose ECG differs from the adult, clinic-recorded 12-lead data these models are built on. A model for infant ECG should therefore reason about heartbeats directly rather than recover them from arbitrary patches. We propose BeatGraph, which makes the heartbeat its unit of representation, modeling each 30-second window as a graph of beats. A shared beat encoder embeds each heartbeat from its waveform and inter-beat intervals, a Transformer with positional encoding orders the beats in time, and residual graph attention layers relate every beat to every other before attention pooling yields a window embedding. We pretrain BeatGraph on our new corpus of unlabe
    
[^15]: 用于神经表征差异性的流匹配框架

    A Flow Matching Framework for Neural Representational Dissimilarity

    [https://arxiv.org/abs/2609.31544](https://arxiv.org/abs/2609.31544)

    本文提出用深度生成模型中的流匹配框架统一多种神经表征差异性度量（即将其归结为不同速度约束下的Jeffreys散度），该框架在处理复杂分布和连续变量时具有估计优势，并能支持以有原则的方式设计新的距离度量。

    

    神经表征差异性量化了神经响应分布之间的差异，对于比较不同刺激、脑区、任务和模型之间的神经编码至关重要。常用的距离度量涉及不同的假设，并需使用各自不同的方法进行估计。在这里，我们证明了多种距离度量可以在深度生成模型中发展出的流匹配框架下得到统一。也就是说，这些距离在不同速度约束下表现为Jeffreys散度。我们发现，流匹配在估计涉及复杂分布和连续变量的距离方面具有优势。此外，该框架能够以有原则的方式设计新的距离度量。总之，流匹配为理解、估计和设计神经表征差异性度量提供了一种统一的方法。

    arXiv:2609.31544v1 Announce Type: new  Abstract: Neural representational dissimilarity quantifies differences between neural response distributions, and is essential for comparing neural codes across stimuli, brain areas, tasks, and models. Commonly used distance metrics involve different assumptions and are estimated with separate methods. Here, we show that a variety of distance metrics can be unified under a flow matching framework developed in deep generative models. That is, these distances arise as Jeffreys divergences under different velocity constraints. We find that flow matching has advantages for estimating distances involving complicated distributions and continuous variables. Furthermore, this framework enables the design of new distance metrics in a principled way. Together, flow matching provides a unified approach for understanding, estimating, and designing neural representational dissimilarity metrics.
    
[^16]: NEXT：物理信息神经谱指数时间差分架构

    NEXT: Physics-Informed Neuro-Spectral Exponential Time Differencing Architectures

    [https://arxiv.org/abs/2609.31539](https://arxiv.org/abs/2609.31539)

    NEXT架构通过将NeuSA的谱表示与高阶指数积分器相结合，利用矩阵指数精确积分线性刚性部分、用神经网络建模非线性剩余部分，从而解决了神经谱方法在刚性偏微分方程上的数值不稳定问题。

    

    物理信息神经网络（PINNs）为时间依赖的偏微分方程（PDE）解构建神经表示，能够自然地融合物理知识与观测数据，因此非常适合求解PDE的正问题和反问题。然而，PINNs已知存在谱偏差和缺乏因果性的问题。神经谱架构（NeuSA）是最近提出的一种PINNs替代方案，可以缓解这两个问题，但对于许多相关物理问题中出现的刚性微分方程，其数值积分会变得不稳定。本研究提出了神经谱指数时间差分架构（NEXT），该方法将NeuSA中PDE解的谱表示与高阶指数积分器相结合。在这一方法中，由PDE诱导的向量场中的线性刚性部分通过矩阵指数进行精确积分，而可能非线性的剩余部分则由神经网络建模。

    arXiv:2609.31539v1 Announce Type: new  Abstract: Physics-Informed Neural Networks (PINNs) build neural representations of time-dependent PDE solutions, naturally incorporating physics knowledge and observational data, which makes them well suited to both forward and inverse PDE problems. PINNs, however, are known to suffer from spectral bias and lack of causality. Neuro-Spectral Architectures (NeuSA), a recently proposed alternative to PINNs, mitigate both issues, but their numerical integration becomes unstable for stiff differential equations arising in many relevant physical problems. This study proposes Neuro-Spectral Exponential Time Differencing Architectures (NEXT), which combines the spectral representation of the PDE solution in NeuSA with high-order exponential integrators. Within this approach, the linear stiff part of the vector field induced by the PDE is integrated exactly through matrix exponentials, while the possibly nonlinear remainder is modeled by a neural network. 
    
[^17]: HySTAR：基于锚定超图的协作多智能体强化学习稳定信用分配

    HySTAR: Anchored Hypergraphs for Stable Credit Assignment in Cooperative Multi-Agent Reinforcement Learning

    [https://arxiv.org/abs/2609.31531](https://arxiv.org/abs/2609.31531)

    提出HySTAR框架，通过锚定重叠稀疏超图作为时间一致的高阶价值分解基础，解决协作多智能体强化学习中的结构性目标漂移问题，实现稳定的个体与高阶联盟信用分配。

    

    在部分可观测与共享奖励条件下的协作多智能体强化学习，需要将团队成果分配给个体智能体及高阶联盟。MAPPO风格的评论家将联合行为压缩为一个全局价值，而动态重构分组拓扑的评论家则会随着交互或活跃智能体的演变而改变从智能体和联盟到价值组件的映射。我们将这种不一致性称为结构性目标漂移。我们提出HySTAR，一个基于MAPPO的框架，它将自适应表示学习与时间一致的高阶价值分解基础相分离。HySTAR锚定一个重叠的稀疏超图作为均匀覆盖的分解脚手架，使用时空编码器来表征物理交互与任务相关交互，并结合时间相关性与结构相关性来构建智能体特定的优势函数。在SMAC、GRF、Traffic Junction等环境上的实验……（摘要原文在此处被截断）

    arXiv:2609.31531v1 Announce Type: new  Abstract: Cooperative multi-agent reinforcement learning under partial observability and shared rewards requires assigning team outcomes to individual agents and high-order coalitions. A MAPPO-style critic compresses joint behavior into one global value, while critics that dynamically reconstruct the grouping topology change the mapping from agents and coalitions to value components as interactions or active agents evolve. We refer to this inconsistency as structural target drift. We introduce HySTAR, a MAPPO-based framework that separates adaptive representation learning from a temporally consistent high-order value-decomposition basis. HySTAR anchors an overlapping sparse hypergraph as a uniformly covered decomposition scaffold, uses a spatiotemporal encoder to represent physical and task-dependent interactions, and combines temporal and structural relevance to construct agent-specific advantages. Experiments on SMAC, GRF, Traffic Junction, and 
    
[^18]: 跨材料体系的可重训练物理融合神经可微分烧结建模

    Retrainable physics-integrated neural differentiable modeling of sintering across material systems

    [https://arxiv.org/abs/2609.31518](https://arxiv.org/abs/2609.31518)

    提出可重训练的物理融合神经可微分框架Sinter-PiNDiff，通过神经网络在耦合速率方程中学习致密化与晶粒生长系数，在多种氧化物材料的留出温度和成分测试中全面优于基线模型，实现了跨材料体系的烧结过程精确预测。

    

    烧结被广泛用于制造陶瓷，但致密化与晶粒长大的耦合、材料相关的动力学特性以及稀疏的测量数据使预测建模和工艺设计变得复杂。我们提出了Sinter-PiNDiff，一个可重训练的物理融合神经可微分框架，用于预测密度和晶粒尺寸的演化。两个神经网络在耦合速率方程中学习致密化和晶粒生长系数，同时一个平滑饱和因子在接近理论密度时衰减致密化速率。相同的控制方程结构、网络架构和训练流程被独立地拟合到已发表的MgO、Al掺杂ZnO和CaO掺杂ThO2数据上。在留出的温度和成分上的测试表明，在全部十二项材料-指标对比中，该框架相对多层感知机和残差网络基线均取得了最低的平均误差。对于MgO、Al掺杂ZnO和CaO掺杂ThO2，密度归一化均方根……（摘要原文在此处截断）

    arXiv:2609.31518v1 Announce Type: cross  Abstract: Sintering is widely used to manufacture ceramics, but coupled densification and grain growth, material-dependent kinetics, and sparse measurements complicate predictive modeling and process design. We present Sinter-PiNDiff, a retrainable physics-integrated neural differentiable framework for predicting density and grain-size evolution. Two neural networks learn densification and grain-growth coefficients within coupled rate equations, while a smooth saturation factor attenuates densification near theoretical density. The same governing structure, network architecture, and training procedure were fitted independently to published data for MgO, Al-doped ZnO, and CaO-doped ThO2. Tests at held-out temperatures and compositions yielded the lowest mean error in all twelve material-metric comparisons against multilayer perceptron and residual network baselines. For MgO, Al-doped ZnO, and CaO-doped ThO2, respectively, density normalized root-
    
[^19]: 零售产品搜索：Target的实用方法

    Retail Product Search: A Practical Approach at Target

    [https://arxiv.org/abs/2609.31498](https://arxiv.org/abs/2609.31498)

    Target公司提出了一种结合词法搜索与向量搜索的混合式零售产品搜索系统，通过数据处理、嵌入训练、精度控制和多通道加权结果融合等实用方案，在保证低延迟的同时平衡相关性、收入与利润等多重目标。

    

    搜索是电子商务中最重要的功能之一，直接驱动着客户参与度和业务增长。一个好的产品搜索系统必须展示既相关又符合用户需求的结果。然而，零售搜索面临着独特的挑战：用户意图可能从精确匹配到开放式探索不等；搜索系统还必须平衡多个目标，如相关性、收入和利润，同时保持较低的响应时间。传统的基于关键词的方法在处理自然语言或语义查询时往往表现不佳。向量搜索有助于缓解这些问题，但可能会遗漏关键的意图信号或返回低精度的结果。在本文中，我们介绍了Target公司混合搜索系统的设计，该系统结合了词法搜索和向量搜索。我们描述了在数据处理、嵌入训练、最终结果集精度控制、多通道结果融合（我们对融合策略进行了比较并采用了加权方法）等方面的实践方案。

    arXiv:2609.31498v1 Announce Type: new  Abstract: Search is one of the most important features in e-commerce, directly driving customer engagement and business growth. A good product search system must show both relevant and desirable results. However, retail search presents unique challenges. User intent can range from exact matches to open-ended discovery. Search systems must also balance multiple goals, such as relevance, revenue, and profit, while keeping response times low. Traditional keyword-based methods often fall short in handling natural language or semantic queries. Vector search helps alleviate these issues, but it can miss key intent signals or return low-precision results. In this paper, we present the design of a hybrid search system at Target that combines lexical and vector search. We describe our approach to data processing, embedding training, precision control for the final result set, multi-channel result fusion (where we compared fusion strategies and adopted weig
    
[^20]: 基于高斯泼溅的可扩展密度泛函理论

    Scaling Density Functional Theory with Gaussian Splatting

    [https://arxiv.org/abs/2609.31483](https://arxiv.org/abs/2609.31483)

    GS-DFT 将分子轨道表示为高斯函数云，通过梯度下降联合优化其位置、形状和混合系数，无需训练数据即可实现密度泛函理论计算随系统规模的可扩展性，并达到最大常规基组的精度。

    

    密度泛函理论（DFT）在计算化学和材料科学的许多问题中，在精度与计算成本之间取得了实用的平衡。然而，许多 DFT 计算受限于固定的以原子为中心的基组，这种基组决定了精度和成本随系统规模的增长方式。我们提出面向密度泛函理论的高斯泼溅方法（GS-DFT），它将分子轨道表示为一团高斯函数，其位置、形状和混合系数通过梯度下降进行联合优化，以在无需训练数据的情况下最小化能量。从概念上讲，GS-DFT 就是将渲染器替换为量子力学的 3D 高斯泼溅。我们引入了两个关键的求解器组件：带有筛选机制的自适应密度拟合，用于高效计算双电子积分；以及分子轨道的正则化可微正交化。实验表明，优化后的基组达到了最大常规基组的精度。

    arXiv:2609.31483v1 Announce Type: cross  Abstract: Density functional theory (DFT) strikes a practical balance between accuracy and computational cost in many problems of computational chemistry and materials science. However, many DFT calculations are limited by fixed atom-centered basis sets, which dictate how accuracy and cost scale with system size. We propose Gaussian Splatting for Density Functional Theory (GS-DFT), which represents molecular orbitals as a cloud of Gaussians whose positions, shapes, and mixing coefficients are optimized jointly by gradient descent to minimize the energy without training data. Conceptually, GS-DFT is 3D Gaussian splatting with the renderer replaced by quantum mechanics. We introduce two key solver components: adaptive density fitting with screening for efficient evaluation of two-electron integrals, and a regularized differentiable orthogonalization of the molecular orbitals. Empirically, the optimized basis reaches the accuracy of the largest con
    
[^21]: 超越经验支撑：基于Sinkhorn最优传输的结构化离群点生成

    Beyond Empirical Support: Structured Outlier Generation via Sinkhorn Optimal Transport

    [https://arxiv.org/abs/2609.31470](https://arxiv.org/abs/2609.31470)

    该论文提出SBOG框架，将Sinkhorn最优传输几何与分布鲁棒边界建模相结合，在潜空间中结构化地生成弱支撑边界区域的离群点，从而更有效地评估和提升机器学习系统应对分布偏移的鲁棒性。

    

    离群点对于评估和提升机器学习系统的鲁棒性至关重要，尤其是当未来分布可能与历史训练数据存在显著差异时。在高风险应用中，鲁棒性往往依赖于有限数据集无法捕捉的罕见案例，这使得简单的重采样或扰动方法不足以用于压力测试场景的生成。现有的离群点合成方法通常依赖于稀疏邻域、低支撑潜空间区域或分类器边界穿越，这些方法可能是启发式的、不稳定的，并且与特定的模态或架构绑定。因此，我们提出了Sinkhorn边界离群点生成（SBOG），这是一个结构化的潜空间离群点生成框架，它将Sinkhorn最优传输几何与分布鲁棒边界建模相结合。由Sinkhorn诱导的支撑代价引导采样器朝向弱支撑的边界区域，同时语义约束（摘要原文在此处截断）

    arXiv:2609.31470v1 Announce Type: new  Abstract: Outliers are essential for evaluating and improving the robustness of machine learning systems, especially when future distributions may differ significantly from historical training data. In high-stakes applications, robustness often depends on rare cases that finite datasets fail to capture, making simple resampling or perturbation insufficient for stress scenario generation. Existing outlier synthesis methods typically rely on sparse neighborhoods, low support latent regions, or classifier boundary crossings, which can be heuristic, unstable, and tied to specific modalities or architectures. We therefore propose Sinkhorn Boundary Outlier Generation (SBOG), a structured framework for latent-space outlier generation that couples Sinkhorn optimal transport geometry with distributionally robust boundary modeling. The resulting Sinkhorn-induced support cost guides the sampler toward weakly supported boundary regions, while semantic constra
    
[^22]: LandscapeSHAP：哪个持续同调类应当获得贡献归因？

    LandscapeSHAP: Which Persistent Homology Class Gets the Credit?

    [https://arxiv.org/abs/2609.31469](https://arxiv.org/abs/2609.31469)

    本文提出了首个将 Shapley 值应用于拓扑数据分析特征的可解释性方法 LandscapeSHAP，能够将模型预测的贡献公平分配到持续图中的各个持续同调类，并对持续景观上的线性模型给出精确的闭式解。

    

    Shapley 值源于合作博弈论中的解概念，近年来已成为机器学习中特征贡献分配的标准工具。它提供了一种具有公理化依据的方法，可以将模型的预测公平地分配到各个数据特征上。然而，Shapley 值尚未被应用于解释基于拓扑数据分析特征训练的机器学习模型。我们开发了据我们所知的首个此类方法，重点关注持续图的持续景观特征化表示。由于每个景观坐标都是一个秩统计量，将模型的预测归因回单个持续同调类（即持续图中的点）并非易事。我们提出了 LandscapeSHAP，这是一种基于模型预测对持续图中的点进行公平贡献分配的方法。对于基于持续景观的线性模型，LandscapeSHAP 具有闭式表达式，可以给出精确的 Shapley 值。

    arXiv:2609.31469v1 Announce Type: cross  Abstract: Shapley values, a solution concept from cooperative game theory, have recently become a standard tool for feature credit allocation in machine learning. They provide an axiomatically justified method to fairly distribute a model's prediction among the data features. Shapley values have not yet been applied to explain machine learning models trained on features from topological data analysis. We develop what we believe is the first such approach, focusing on the persistence landscape featurization of persistence diagrams. Because each landscape coordinate is a rank statistic, crediting a model's prediction back to individual persistent homology classes (persistence diagram points) is nontrivial. We introduce LandscapeSHAP, a method for fair credit allocation to persistence diagram points based on a model's prediction. For linear models on persistence landscapes, LandscapeSHAP has a closed form expression that gives the exact Shapley val
    
[^23]: Scaffold：基于支撑图理论的图神经网络稀疏化方法

    Scaffold: Support Graph Theory Based Sparsification for Graph Neural Networks

    [https://arxiv.org/abs/2609.31466](https://arxiv.org/abs/2609.31466)

    Scaffold是一个源自支撑图理论的无监督图稀疏化框架，通过联合控制扩张度（重路由路径长度）和拥塞度（重路由路径集中程度）两个结构量，在保留短通信路径的同时避免结构性瓶颈，从而在降低GNN计算与内存成本的同时保持预测性能。

    

    图神经网络（GNN）依赖于图边上的消息传递，使其计算和内存成本高度依赖于图的密度。图稀疏化为降低这些成本提供了一种自然的方法，但不加区分地删除边可能会扭曲重要的通信结构并降低预测性能。我们提出了Scaffold，一个基于拓扑的、无监督的图稀疏化框架，它源自支撑图理论预条件子。Scaffold显式地控制两个互补的结构量：扩张度，用于衡量由被删除边引起的重路由路径的长度；拥塞度，用于衡量这些重路由路径在保留的支撑上的集中程度。通过联合控制扩张度和拥塞度，Scaffold在保持短通信路径的同时避免了结构性瓶颈。据我们所知，Scaffold是首个使用联合支撑（图理论保证）的可扩展GNN稀疏化框架。

    arXiv:2609.31466v1 Announce Type: new  Abstract: Graph neural networks (GNNs) rely on message passing over graph edges, making their computational and memory costs strongly dependent on graph density. Graph sparsification offers a natural way to reduce these costs, but removing edges indiscriminately can distort important communication structure and degrade predictive performance. We introduce Scaffold, a topology-based, unsupervised graph sparsification framework derived from support graph theory preconditioners. Scaffold explicitly controls two complementary structural quantities: dilation, which measures the length of rerouting paths induced by removed edges, and congestion, which measures how strongly these rerouted paths concentrate on the retained support. By jointly controlling dilation and congestion, Scaffold preserves short communication paths while avoiding structural bottlenecks. To our knowledge, Scaffold is the first scalable GNN sparsification framework to use a joint su
    
[^24]: 面向婴儿运动分析的不确定性感知联邦学习

    Uncertainty-Aware Federated Learning for Infant Movement Analysis

    [https://arxiv.org/abs/2609.31463](https://arxiv.org/abs/2609.31463)

    提出了首个面向婴儿运动分析的不确定性感知联邦学习框架，能够在保护隐私的前提下利用多机构骨骼运动数据实现自动化全身运动评估。

    

    婴儿运动分析为神经发育障碍的早期识别提供了有价值的生物标志物。深度学习的最新进展使得从视频提取的骨骼表示中进行自动化婴儿运动分析成为可能，在全身运动评估（GMA）等任务上达到了与专家评估相当的性能。然而，大多数现有方法依赖于集中式训练，需要将多个机构的数据收集并存储在单一站点。由于隐私、治理和数据共享的限制，这种假设在临床环境中往往不切实际。为了应对这些挑战，据我们所知，我们提出了首个使用骨骼运动数据进行自动化婴儿运动分析和全身运动评估的联邦学习框架。作为一个具有临床相关性的用例，所提出的框架在不安运动分类任务上进行了评估。为了量化……

    arXiv:2609.31463v1 Announce Type: cross  Abstract: Infant movement analysis provides valuable biomarkers for the early identification of neurodevelopmental disorders. Recent advances in deep learning have enabled automated analysis of infant movements from video-derived skeletal representations, achieving performance comparable to expert assessment for tasks such as General Movement Assessment (GMA). However, most existing approaches rely on centralized training, requiring data from multiple institutions to be collected and stored at a single site. Such assumptions are often impractical in clinical settings due to privacy, governance, and data-sharing constraints. To address these challenges, we present, to the best of our knowledge, the first federated learning framework for automated infant movement analysis and General Movement Assessment using skeletal motion data. As a clinically relevant use case, the proposed framework is evaluated on fidgety movement classification. To quantify
    
[^25]: 增长几何复杂度下的非参数上下文学习：Transformer的极小极大最优性与局部几何自适应性

    Nonparametric In-Context Learning under Growing Geometric Complexity: Minimax Optimality and Local Geometry-Adaptivity of Transformers

    [https://arxiv.org/abs/2609.31458](https://arxiv.org/abs/2609.31458)

    本文在由随样本量增长、维度与光滑度异构的流形混合所刻画的未知局部几何下研究非参数上下文学习，建立了极小极大最优下界，并证明配备几何预条件器的结构感知两阶段softmax Transformer能达到该最优性且可自适应局部几何。

    

    Transformer已成为上下文学习（ICL）的核心架构，尤其体现在其于大型语言模型中的最先进性能上。这一成功促使人们理解Transformer如何在几何异构数据中利用与任务相关的结构。然而，现有的非参数ICL理论大多集中于欧几里得空间或单一流形模型。为填补这一空白，我们研究了未知局部几何下的预测问题，其中局部几何由随样本量变化的流形混合建模，各流形在维度、光滑度和采样质量上均具有异构性。在局部分离与小扰动条件下，我们建立了一个刻画各组成部分聚合难度的极小极大下界，并构造了一个达到匹配上界的oracle切向局部多项式估计器。该估计器与一种结构感知的两阶段softmax Transformer相关联，该Transformer配备了几何预条件器……

    arXiv:2609.31458v1 Announce Type: new  Abstract: Transformers have become a central architecture for in-context learning (ICL), particularly through their state-of-the-art performance in large language models. This success motivates understanding how transformers exploit task-relevant structure in geometrically heterogeneous data. However, existing nonparametric ICL theory has largely focused on Euclidean domains or single-manifold models. To address this gap, we study the prediction problem under unknown local geometry, modeled by sample size-dependent mixtures of manifolds with heterogeneous dimensions, smoothness, and sampling masses. Under local separation and small-perturbation conditions, we establish a minimax lower bound capturing the aggregate difficulty of the components and construct an oracle tangent local-polynomial estimator with a matching upper bound. This estimator is connected to a structure-informed, two-stage softmax transformer with a geometric preconditioner and c
    
[^26]: 不同的损坏，不同的信号：联邦数据质量中的不确定性与损失

    Different Corruptions, Different Signals: Uncertainty and Loss in Federated Data Quality

    [https://arxiv.org/abs/2609.31454](https://arxiv.org/abs/2609.31454)

    本文比较了联邦学习中输入条件不确定性与预测-标签损失两种损坏检测信号，发现在非独立同分布数据条件下，这两种信号对输入噪声和标签翻转两类损坏表现出不同的检测效果。

    

    联邦学习中的数据损坏可能影响输入或标签，但目前尚不清楚输入条件不确定性和预测-标签损失是否能同等地揭示这些损坏模式。本文比较了联邦学习中两种损坏检测信号：输入条件不确定性和预测-标签损失。不确定性信号通过学习到的偶然方差估计以及蒙特卡洛 dropout 方差和熵度量来表征，而损失则基于所提供的标签进行计算。我们针对加性图像噪声和持续性随机标签翻转测试了这些信号。在 ResNet-20 上使用 CIFAR-10 和 SVHN 数据集、在 Dirichlet 划分的非独立同分布数据条件下，两种损坏类型表现出不同的行为。对于持续性随机标签翻转，客户端内逐样本的受试者工作特征曲线下面积（AUC）在 CIFAR-10 上为 0.85，在 0.95（原文摘要在此处中断）。

    arXiv:2609.31454v1 Announce Type: cross  Abstract: Federated learning (FL) data corruption can affect either inputs or labels, but it remains unclear whether input-conditional uncertainty and prediction-label loss expose these corruption modes equally. This paper compares two corruption-detection signals in FL: input-conditional uncertainty and prediction-label loss. The uncertainty signal is characterised using a learned aleatoric variance estimate together with Monte Carlo (MC) dropout variance and entropy measures, while the loss is computed against the supplied label. We test these signals against additive image noise and persistent random label flips. On ResNet-20 with CIFAR-10 and SVHN under Dirichlet partitions with data that are not independent and identically distributed (non-IID), the two corruption types behave differently. For persistent random label flips, the within-client per-sample area under the receiver operating characteristic curve (AUC) is 0.85 on CIFAR-10 and 0.95
    
[^27]: 用于高光谱视频压缩的隐式神经表示

    Implicit Neural Representation for Hyperspectral Video Compression

    [https://arxiv.org/abs/2609.31435](https://arxiv.org/abs/2609.31435)

    该论文提出了一种基于隐式神经表示的高光谱视频压缩新方法，通过对现有RGB视频压缩模型进行新颖扩展，相比传统逐帧压缩方法实现了+4.99 dB的PSNR增益和-88.88%的码率降低，同时显著提升了下游目标跟踪任务的性能。

    

    随着快照相机的出现，高光谱视频正变得越来越容易获取。近年来，新应用的涌现导致数据集规模日益增大。然而，高光谱视频压缩仍处于早期发展阶段。在本研究中，我们探索将隐式神经表示作为一种候选解决方案。我们对现有的RGB视频压缩模型提出了一种新颖的扩展方法，与逐帧应用的传统高光谱图像压缩方法相比，实现了+4.99 dB的Bjøntegaard Delta PSNR增益和-88.88%的Bjøntegaard Delta码率降低。除了重建质量之外，本研究还以目标跟踪成功率的形式衡量了压缩对下游任务性能的影响。与在低数据量情况下基于主成分分析和JPEG2000的方法压缩的视频相比，我们提出的方法将跟踪曲线下面积最多提升了23.42%，距离精度也有相应提升。

    arXiv:2609.31435v1 Announce Type: cross  Abstract: With the advent of snapshot cameras, hyperspectral video is becoming more readily available. In recent years, new applications have emerged which have led to increasingly larger datasets. However, hyperspectral video compression remains in the early stages. In this study, we explore the use of implicit neural representation as a candidate solution. We propose a novel extension of an existing RGB video compression model, achieving Bj{\o}ntegaard Delta PSNR gains of +4.99 dB and Bj{\o}ntegaard Delta rate of -88.88% compared to traditional hyperspectral image compression methods applied frame-by-frame. In addition to reconstruction quality, the effects on downstream task performance are measured in the form of object tracking success. Compared to video compressed with methods based on principal component analysis and JPEG2000 in low data regimes, our proposed method improves tracking area under the curve by up to 23.42% and distance preci
    
[^28]: 评估KV缓存重用技术的准确性

    Evaluating the accuracy of KV cache reuse techniques

    [https://arxiv.org/abs/2609.31415](https://arxiv.org/abs/2609.31415)

    该论文指出现有KV缓存重用技术的评估方法会人为夸大其有效性，并提出了一种无歧义测量精度损失的评估方法，同时发布了Boxoffice工具来生成具有挑战性重用模式的评估数据集。

    

    与位置无关的KV缓存重用旨在通过跨提示重用块级KV缓存来降低检索增强生成的延迟。我们表明，当前对KV缓存重用技术的评估所依赖的测量方法无法忠实捕捉由重用导致的精度损失，往往会人为夸大所报告的有效性。我们还表明，现有数据集并不具备彻底评估此类技术所需的重用动态特征。为解决这些问题，我们提出了一种能够无歧义地测量这种精度损失的评估方法，并介绍了Boxoffice——一个能够以编程方式生成评估数据集的工具，这些数据集可施加具有挑战性的KV缓存重用模式。

    arXiv:2609.31415v1 Announce Type: new  Abstract: Position-independent KV cache reuse aims to reduce latency in retrieval-augmented generation by reusing chunk-level KV caches across prompts. We show that current evaluations of KV cache reuse techniques rely on measurements that fail to faithfully capture the loss of accuracy attributable to reuse, often artificially inflating the reported effectiveness. We also show that existing datasets do not exhibit the reuse dynamics needed to thoroughly evaluate such techniques. To address these issues, we propose an evaluation methodology that measures this accuracy loss without ambiguity and we introduce Boxoffice, a tool that programmatically generates evaluation datasets that exercise challenging KV cache reuse patterns.
    
[^29]: AFA-Net：一种用于听觉注意力检测的差分注意力方法

    AFA-Net: A Differential Attention Approach for Auditory Attention Detection

    [https://arxiv.org/abs/2609.31402](https://arxiv.org/abs/2609.31402)

    AFA-Net通过差分注意力机制显式对抗EEG噪声，在2秒决策窗口下以远少于现有方法的参数量实现了96.8%的听觉注意力检测准确率。

    

    听觉注意力检测（AAD）利用脑电图（EEG）信号在多说话人环境中识别目标说话人。尽管已取得相当大的进展，现有的深度学习架构往往缺乏处理噪声脑电数据的显式机制。为了解决这一局限，我们提出了听觉聚焦注意力网络（AFA-Net），这是一个机器学习框架，它用一种简单而灵活的差分注意力机制取代原始注意力，以帮助聚焦于与任务相关的神经活动。AFA-Net在2秒决策窗口下达到了96.8%的较高准确率，同时使用的参数量远少于大多数现有方法。据我们所知，AFA-Net是最早显式尝试对抗EEG噪声以提升AAD性能的框架之一。

    arXiv:2609.31402v1 Announce Type: cross  Abstract: Auditory Attention Detection (AAD) utilizes electroencephalographic (EEG) signals to identify a target speaker in a multi-speaker environment. Despite considerable progress, existing deep learning architectures often lack explicit mechanisms for handling noisy EEG data. To address this limitation, we propose Auditory Focus Attention Networks (AFA-Net), a machine learning framework that replaces vanilla attention with a simple yet flexible differential attention mechanism to help focus on task-relevant neural activity. AFA-Net achieves an upward accuracy of 96.8% at the 2s decision window, while using substantially fewer parameters than most existing methods. To the best of our knowledge, AFA-Net is among the first frameworks to explicitly try to combat EEG noise to improve AAD.
    
[^30]: 跨训练全程中可解码的上下文状态与模型输出

    Decodable In-Context State and Model Output Across Training

    [https://arxiv.org/abs/2609.31401](https://arxiv.org/abs/2609.31401)

    本文首次在Pythia模型的预训练与后训练各检查点上系统跟踪探针可解码性、模型输出与探针引导转向的演化，发现二者随训练同步提升，并通过信息论反例证明仅在错误样本上的可解码性不足以证明模型丢弃了输出信息。

    

    先前的研究已经表明，探针可以在模型出错时解码上下文中的绑定关系，并且基于探针引导的转向能够修复其中一部分错误。我们在公开的预训练与后训练检查点上跟踪了探针准确率、模型输出以及转向响应的变化。在Pythia预训练过程中，探针准确率持续上升，同时在两种模型规模下，探针引导的转向从几乎没有的全试次收益转变为较大的收益。保存的分数能够区分探针判断正确但模型对正确候选赋予低概率或高于均匀概率的错误。使用预言目标进行的转向已经能够修复许多早期错误，但保存的聚合数据无法将目标质量与干预敏感度区分开来。对在最终状态或候选logits上训练的解码器进行的留出集比较发现，在后期检查点的模型错误上并未检测到最终状态的优势。一个信息论反例解释了为什么仅凭错误样本上的可解码性无法证明被丢弃的输出信息。

    arXiv:2609.31401v1 Announce Type: new  Abstract: Prior work established that a probe can decode an in-context binding on model errors and that probe-guided steering can repair some of them. We follow probe accuracy, model output, and steering response across public pretraining and post-training checkpoints. Probe accuracy rises during Pythia pretraining, while probe-guided steering moves from negligible all-trial benefit to a larger benefit at two model sizes. Saved scores distinguish probe-correct errors with low and above-uniform model probability for the correct candidate. Oracle-target steering already repairs many early errors, but saved aggregates cannot separate target quality from intervention sensitivity. A held-out comparison of decoders trained on the final state or candidate logits finds no detected final-state advantage on late-checkpoint model errors. An information-theoretic counterexample explains why decodability on errors alone cannot establish discarded output inform
    
[^31]: 差分注意力解锁脑电与语音的互补融合以实现情感识别

    Differential Attention Unlocks Complementary EEG and Speech Fusion for Emotion Recognition

    [https://arxiv.org/abs/2609.31399](https://arxiv.org/abs/2609.31399)

    EmoSpeechBrain 通过差分注意力抑制脑电噪声、并利用门控适配器融合脑电与语音两种模态，在情感识别任务上将准确率相比现有最先进方法提升高达 12.9%。

    

    多模态情感识别（MER）正日益将脑电（EEG）与语音相结合，将内部神经信号和外部语音表达视为情感的信息性视角。但在实践中，简单的融合方法往往不如较强的单一模态，因为脑电伪影注入的噪声会破坏共享表征。我们提出了 EmoSpeechBrain，这一多模态框架建立在“噪声抑制是有效融合的前提”这一洞察之上。其脑电编码器采用差分注意力，通过计算两个注意力图之间的差值来消除共享噪声，并分离出具有判别性的神经活动。基于注意力的门控适配器将两种模态对齐到共享空间中，并对各自对预测的贡献进行加权。在 PME4 和 EAV 两个数据集上，EmoSpeechBrain 相比其他最先进的（SOTA）脑电编码器将多模态情感识别准确率提升了高达 12.9%，相比单模态语音和脑电基线分别提升了高达 13.1% 和 23.1%。

    arXiv:2609.31399v1 Announce Type: new  Abstract: Multimodal emotion recognition (MER) increasingly pairs EEG with speech, treating internal neural signals and external vocal expression as informative views of affect. In practice, naive fusion underperforms the stronger single modality, because EEG artifacts inject noise that corrupts the shared representation. We introduce EmoSpeechBrain, a multimodal framework built on the insight that noise suppression is a precondition for effective fusion. Its EEG encoder uses differential attention, taking the difference between two attention maps to cancel shared noise and isolate discriminative neural activity. An attention-based gating adapter aligns both modalities in a shared space and weights each one's contribution to the prediction. On two datasets - PME4 and EAV, EmoSpeechBrain improves MER accuracy by up to 12.9% over other state-of-the-art (SOTA) EEG encoders, and surpasses unimodal speech and EEG baselines by up to 13.1% and 23.1%. The
    
[^32]: 基于端点约束轨迹优化的端到端驾驶模型引导

    Guiding End-to-End Driving Models with Endpoint-Constrained Trajectory Optimization

    [https://arxiv.org/abs/2609.31383](https://arxiv.org/abs/2609.31383)

    该论文发现端到端驾驶模型开环与闭环性能差距的一个新因素是中间路径点监督缺乏物理连贯性和可跟踪性，并提出轻量级后处理层ECO，通过锚定车辆执行历史、保留可靠的预测端点并重塑中间路径点来弥合这一差距。

    

    端到端驾驶策略通常通过开环行为克隆进行训练，但在部署到车辆上时最终必须以闭环方式运行，这导致训练与执行之间存在根本性的不匹配。除了已被广泛研究的协变量偏移和因果混淆效应之外，我们为这种开环/闭环差距识别出一个补充性因素：基于路径点的监督方式和位移度量无法保证中间轨迹在物理上是连贯的，也难以保证控制器能够顺利跟踪。我们观察到，这些不一致性主要集中在中间路径点上，而预测的端点则相对可靠。基于这一观察，我们提出了端点约束优化，这是一种轻量级的后处理层，它将轨迹锚定在车辆已执行的历史轨迹上，保留策略预测的端点，并重塑中间路径点以改进……

    arXiv:2609.31383v1 Announce Type: cross  Abstract: End-to-end driving policies are commonly trained through open-loop behavior cloning, yet they must ultimately operate in closed-loop when deployed on a vehicle, creating a fundamental mismatch between training and execution. Beyond the commonly studied effects of covariate shift and causal confusion, we identify a complementary factor for this open-loop/closed-loop gap: waypoint-based supervision and displacement metrics do not ensure that the intermediate trajectory is physically coherent or easy for the controller to track. We observe that these inconsistencies concentrate primarily at intermediate waypoints, while the predicted endpoint remains comparatively reliable. Based on this observation, we introduce Endpoint-Constrained Optimization (ECO), a lightweight postprocessing layer that anchors the trajectory to the vehicle's executed history, preserves the policy's predicted endpoint, and reshapes the intermediate waypoints to impr
    
[^33]: 迈向理解基于大语言模型的日志异常检测：性能、效率与鲁棒性的实证研究

    Towards Understanding LLM-Based Log Anomaly Detection: An Empirical Study of Performance, Efficiency, and Robustness

    [https://arxiv.org/abs/2609.31371](https://arxiv.org/abs/2609.31371)

    该论文通过对三个公开数据集的系统实证研究，揭示了适配策略、模型规模与量化设置对LLM日志异常检测性能和效率的影响，并评估了模型在结构、语义和标签噪声下的鲁棒性。

    

    大语言模型（LLMs）在日志异常检测中已展现出有前景的性能，但其适配策略、模型架构和部署配置如何影响检测效果仍缺乏充分理解。为了探究这些因素，我们在三个公开日志数据集上开展了系统性的实证分析，考察了不同的适配策略、模型架构、参数规模和量化设置。研究结果揭示了不同适配策略之间存在显著的性能差异，而模型规模的扩大在不同数据集上带来的检测增益各不相同。我们还观察到，检测精度相近的模型可能表现出截然不同的计算成本，并且在所评估的配置中，低比特量化在很大程度上保留了检测性能。最后，我们考察了模型在不同扰动水平下，面对结构、语义和标签噪声时的检测鲁棒性。

    arXiv:2609.31371v1 Announce Type: new  Abstract: Large language models (LLMs) have demonstrated promising performance in log anomaly detection, yet how their adaptation strategies, architectures, and deployment configurations affect detection effectiveness remains insufficiently understood. To investigate these factors, we conduct a systematic empirical analysis across three public log datasets, examining different adaptation strategies, model architectures, parameter scales, and quantization settings. Our results reveal substantial performance differences across adaptation strategies, while model scaling yields varying detection gains across datasets. We further observe that models with comparable detection accuracy can exhibit markedly different computational costs, and that low-bit quantization largely preserves detection performance in the evaluated configurations. Finally, we examine detection robustness under structural, semantic, and label noise at different perturbation levels.
    
[^34]: 基于贝叶斯树邻接文法的方程发现

    Equation discovery with Bayesian tree-adjoining grammars

    [https://arxiv.org/abs/2609.31368](https://arxiv.org/abs/2609.31368)

    本文首次将树邻接文法置于贝叶斯框架下，利用结构保持树移动的可逆跳MCMC采样器推断模型结构、参数和预测的联合后验分布，实现了超越传统点估计的概率化方程发现与非线性系统辨识。

    

    树邻接文法（TAG）最近被引入非线性系统辨识（NLSI）领域，作为将整个模型类编码为有限语法规则集合的手段，候选模型由此被组装为树结构。现有的基于TAG的辨识器依赖进化优化，并返回模型结构的点估计。本文转而在贝叶斯框架下提出TAG方法：在树结构及其参数上定义生成式先验，并使用带有结构保持树移动的可逆跳MCMC采样器来推断模型结构、参数和预测的联合后验分布。文中考虑了两种训练目标：即采用共轭参数提议的一步预测目标，以及通过无似然推断处理的基于仿真的目标。该方法在仿真多项式NARX系统、Silverbox基准和波浪载荷数据上进行了验证。

    arXiv:2609.31368v1 Announce Type: new  Abstract: Tree-Adjoining Grammars (TAGs) have recently been introduced to Nonlinear System Identification (NLSI) as a means of encoding an entire model class as a finite set of grammatical rules, from which candidate models are assembled as trees. Existing TAG-based identifiers rely on evolutionary optimisation and return point estimates of the model structure. This paper instead proposes the TAG framework within a Bayesian setting. A generative prior is defined over tree structures and their parameters, and a Reversible-Jump MCMC sampler with structure-preserving tree moves is used to infer the joint posterior over model structure, parameters and predictions. Two training objectives are considered; that is, a one-step-ahead objective with conjugate parameter proposals, and a simulation-based objective handled by likelihood-free inference. The approach is validated on a simulated polynomial NARX system, the Silverbox benchmark, and wave-loading da
    
[^35]: Brenier 遇见对抗训练：面向鲁棒学习的最优传输几何

    Brenier Meets Adversarial Training: Optimal Transport Geometry for Robust Learning

    [https://arxiv.org/abs/2609.31363](https://arxiv.org/abs/2609.31363)

    该论文将带 Wasserstein 惩罚的分布鲁棒优化中的对抗问题重新表述为最优传输映射的优化问题，证明了最优映射满足循环单调性，指出标准对抗训练因违反该性质而浪费传输成本，并提出多起点粒子上升等方法予以改进。

    

    分布鲁棒优化（DRO）为分布偏移下的学习提供了一个有原则的框架，但其在实际中的应用受到阻碍，原因在于对非凸损失函数评估最坏情况风险十分困难。我们研究了一种带惩罚项的 DRO 形式，其中对抗者可以选择任意分布，但偏离经验分布时需承担 Wasserstein 惩罚。我们证明，对抗者的问题可以被重新表述为关于传输映射的优化问题，这些映射将经验样本推送到对抗样本，并且我们证明了最优映射是循环单调的。我们还表明，标准的对抗训练——基于逐样本的局部优化——违反了循环单调性，并浪费了传输成本，除非对对抗者施加严格限制。我们提出了两种补救方法。首先，我们引入了多起点粒子上升方法，该方法交替进行并行梯度上升与重新分配，以强制执行循环单调性……

    arXiv:2609.31363v1 Announce Type: cross  Abstract: Distributionally robust optimization (DRO) provides a principled framework for learning under distribution shift, but its practical use is hindered by the difficulty of evaluating worst-case risks for nonconvex loss functions. We study a penalized DRO formulation in which the adversary may choose any distribution but incurs a Wasserstein penalty for deviating from the empirical distribution. We show that the adversary's problem can be reformulated as an optimization problem over transport maps that push empirical samples to adversarial ones, and we prove that optimal maps are cyclically monotone. We also show that standard adversarial training---based on per-sample local optimization---violates cyclical monotonicity and wastes transport costs unless the adversary is severely restricted. We propose two remedies. First, we introduce multi-start particle ascent, which alternates parallel gradient ascent with reassignment to enforce cyclic
    
[^36]: 开放词汇域遗忘

    Open Vocabulary Domain Unlearning

    [https://arxiv.org/abs/2609.31356](https://arxiv.org/abs/2609.31356)

    该论文指出现有近似域遗忘方法因封闭词汇假设仅对已见类别过拟合而无法真正擦除域信息，进而提出类别无关的开放词汇域遗忘协议，要求模型对未见类别也无法识别目标风格域。

    

    视觉-语言模型（VLMs）展现出卓越的零样本泛化能力，然而它们常常编码了不需要的甚至有害的风格域，例如医学AI中理想化的教科书式图表，或自动驾驶中的卡通风格车辆。近似域遗忘（Approximate Domain Unlearning, ADU）旨在选择性地擦除模型对目标视觉域的识别能力，同时保持模型在其余域上的准确性。然而，现有ADU方法基于一个有缺陷的封闭词汇假设：它们仅在遗忘微调阶段见过的特定对象类别上评估遗忘效果。因此，这些方法并没有真正遗忘域本身；它们只是对已见的类别-域对产生了过拟合，导致该域对于未见类别依然容易被识别，从而给人造成一种虚假的“已移除”假象。我们认为，真正的域擦除必须是类别无关的。为解决这一问题，我们形式化了开放词汇域遗忘，这是一个严格的协议，要求……（原文摘要至此截断）

    arXiv:2609.31356v1 Announce Type: cross  Abstract: Vision-Language Models (VLMs) exhibit remarkable zero-shot generalization, yet they often encode unwanted or hazardous stylistic domains such as idealized textbook diagrams in medical AI or cartoon vehicles in autonomous driving. Approximate Domain Unlearning (ADU) aims to selectively erase a model's recognition of a target visual domain while preserving accuracy on the remaining domains. However, existing ADU methods operate under a flawed closed-vocabulary assumption: they evaluate unlearning solely on the specific object classes seen during the unlearning fine-tuning phase. Consequently, these methods do not unlearn the domain itself; they merely overfit to seen class-domain pairs, leaving the domain easily recognizable for unseen classes and providing a false sense of removal. We argue that true domain erasure must be class-agnostic. To address this, we formalize Open-Vocabulary Domain Unlearning (OVDU), a rigorous protocol that ma
    
[^37]: 渐进记忆Transformer：面向时间序列的记忆感知注意力机制

    Progressive Memory Transformer: Memory-Aware Attention for Time-Series

    [https://arxiv.org/abs/2609.31351](https://arxiv.org/abs/2609.31351)

    提出渐进记忆Transformer（PMT），通过可写的窗口对齐记忆机制在token、窗口、序列三个尺度上显式监督结构层次，从而改进时间序列的自监督表示学习。

    

    时间序列同时在多个尺度上承载结构（细粒度变化、中程模式以及全局特性），而下游任务也在相应不同的尺度上运作。大多数现有的自监督学习方法通过实例级对比损失和有限的时间邻域监督来全局性地监督表示，但并未显式地利用这种结构层次。我们提出了一个学习框架，该框架在三个尺度上独立地显式强化结构层次：针对token连续性的局部目标、针对窗口级模式的中程目标，以及针对序列级一致性的全局目标。实现这一框架需要骨干网络在每个尺度上都能暴露出相应的表示；为此我们提出了**渐进记忆Transformer**（Progressive Memory Transformer, PMT），它通过可写的、与窗口对齐的记忆机制来增强Transformer，从而在token尺度和序列尺度之外暴露出中程尺度……

    arXiv:2609.31351v1 Announce Type: new  Abstract: Time-series carry structure simultaneously at multiple scales (fine-grained variation, mid-range motifs, and global properties) and downstream tasks operate at correspondingly different scales. Most existing self-supervised learning approaches supervise representations globally via instance-level contrastive losses and limited temporal neighborhood supervision, but do not explicitly exploit the structural hierarchy. We propose a learning framework that explicitly enforces a structural hierarchy across three scales independently: a local objective for token continuity, a mid-range objective for window-level motifs, and a global objective for sequence-level agreement. Realizing this framework requires the backbone to expose a representation at each scale; we introduce \textbf{Progressive Memory Transformer} (PMT), which augments a transformer with writable, window-aligned memory that exposes the mid-range scale alongside the token and sequ
    
[^38]: 连接身体与大脑：基因驱动的形态-控制协同设计

    Bridging Body and Brain: Gene-Driven Morphology--Control Co-Design

    [https://arxiv.org/abs/2609.31329](https://arxiv.org/abs/2609.31329)

    该论文受生物基因启发提出 Morphogene 潜在蓝图与 GeCode 框架，通过 AdaConcat 机制在肢体层面联合条件化形态与控制生成，将机器人形态-控制协同设计转化为在统一紧凑潜在空间中的探索，实现身体与大脑的显式协调优化。

    

    形态-控制协同设计将智能体的身体结构与控制策略作为一个完整的具身系统进行联合优化。然而，现有方法通常采用相互独立的网络分别建模形态设计与控制，仅通过共享的任务目标进行间接耦合，限制了显式的高层协调。受自然界中基因协调生物发育的启发，我们提出了 **Morphogene**——一个紧凑的潜在蓝图，用于连接智能体的身体与大脑。通过 AdaConcat 机制，Morphogene 在肢体层面同时对形态生成与控制生成进行条件化，使其变化能够引发身体与大脑两个组成部分的协调变化。基于这一表示，我们提出 **GeCode**，将协同设计表述为在紧凑的 Morphogene 空间中的探索。每个 Morphogene 锚定一个局部设计区域，在其中探索相近的身体-大脑设计方案，同时由性能引导的更新将这些锚点移动到……

    arXiv:2609.31329v1 Announce Type: new  Abstract: Morphology--control co-design jointly optimizes an agent's body structure and control policy as an integrated embodied system. However, existing methods typically model morphology design and control with separate networks coupled only indirectly through a shared task objective, limiting explicit high-level coordination. Inspired by natural genes that coordinate biological development, we introduce \textbf{Morphogene}, a compact latent blueprint that bridges an agent's body and brain. Through AdaConcat, Morphogene jointly conditions morphology and control generation at the limb level, allowing its variations to induce coordinated changes in both components. Building on this representation, we propose \textbf{GeCode}, which formulates co-design as exploration in the compact Morphogene space. Each Morphogene anchors a local design region in which nearby body--brain designs are explored, while performance-guided updates move these anchors to
    
[^39]: 传感器再多，场只有一个：重新思考持续时空预测

    More Sensors Only One Field: Rethinking Continual Spatio-Temporal Forecasting

    [https://arxiv.org/abs/2609.31325](https://arxiv.org/abs/2609.31325)

    该论文提出STFO（时空场算子），将预测知识参数化为共享的场演化算子，通过观测与查询接口应对传感器布局变化，使传感器扩展时无需传感器特定参数即可复用已学习的空间知识，从而解决持续时空预测问题。

    

    持续时空预测在动态演化与传感器网络不断扩展的情况下，为交通管理和环境监测提供支持。然而，传统的基于图的持续学习方法将预测表示与当前传感器布局绑定，因此传感器扩展可能会改变已学习空间关系的表示。我们的核心洞察是：传感器扩展改变的只是关于某一过程的可用证据，而不必然改变需要学习的动态本身。我们提出STFO（时空场算子，Spatio-Temporal Field Operator），它将预测知识参数化为一个共享的场演化算子，并通过观测接口与查询接口来处理不断变化的传感器布局。基于归一化坐标的聚合将不规则的传感器历史数据提升到固定的潜在网格上，使已学习的空间映射能够在不同观测集之间复用，而无需任何传感器特定参数。为了适应过程漂移，一种谱降……（原文摘要在此处截断）

    arXiv:2609.31325v1 Announce Type: new  Abstract: Continual spatio-temporal forecasting supports traffic management and environmental monitoring under evolving dynamics and expanding sensor networks. However, conventional graph-based continual learning methods tie forecasting representations to the current sensor layout, so sensor expansion can alter the representation of learned spatial relationships. Our key insight is that sensor expansion changes the evidence available about a process without necessarily changing the dynamics to be learned. We propose STFO (Spatio-Temporal Field Operator), which parameterizes forecasting knowledge as a shared field-evolution operator and handles changing sensor layouts through observation and query interfaces. Normalized coordinate-based aggregation lifts irregular sensor histories onto a fixed latent grid, enabling reuse of learned spatial maps across observation sets without sensor-specific parameters. To accommodate process drift, a spectral desc
    
[^40]: LUCID：面向时间序列推断与发现的混杂学习

    LUCID: Learning Under Confounding for Inference and Discovery in Time Series

    [https://arxiv.org/abs/2609.31315](https://arxiv.org/abs/2609.31315)

    LUCID 提出了一种机制自适应的去混杂层，利用 Marcenko–Pastur 谱路由器估计混杂机制并施加相应的去混杂策略，可封装现有因果发现算法，从而在时间序列中有效消除未观测混杂因子造成的虚假关联。

    

    未观测的共同原因在现实世界的时间序列中普遍存在，它们会诱发虚假关联，使因果发现方法误将其识别为直接因果边。我们提出了 LUCID（Learning Under Confounding for Inference and Discovery），这是一种机制自适应的去混杂层：它首先利用 Marcenko–Pastur 谱路由器从数据中估计混杂机制，然后应用与该机制相匹配的去混杂策略。当谱结构表明存在普遍的因子混杂时，LUCID 会衰减因子主导的变异，并从所得的新息（innovations）中恢复同期（滞后-0）结构，其边选择则针对数据驱动的无边零假设进行校准。该方法不依赖于特定的发现算法，而是可以封装现有的因果发现引擎；我们在三种此类方法上展示了一致的性能提升。在一个涵盖混杂因子强度与稀疏性变化的多样化合成分布外基准上……

    arXiv:2609.31315v1 Announce Type: cross  Abstract: Unobserved common causes are pervasive in real-world time series and can induce spurious associations that causal discovery methods mistake for direct edges. We propose LUCID (Learning Under Confounding for Inference and Discovery, a regime-adaptive deconfounding layer that first estimates the confounding regime from data using a Mar\v{c}enko--Pastur spectral router, then applies a deconfounding strategy matched to that regime. When the spectrum indicates pervasive factor confounding, LUCID attenuates factor-dominated variation and recovers contemporaneous (lag-$0$) structure from the resulting innovations, with edge selection calibrated against a data-driven edge-free null. Rather than being tied to a particular discovery algorithm, it can wrap existing discovery engines; we demonstrate consistent improvements across three such methods. On a diverse synthetic out-of-distribution benchmark spanning changes in confounder strength and sp
    
[^41]: 表格基础模型的注意力机制基准测试

    Benchmarking Attention for Tabular Foundation Models

    [https://arxiv.org/abs/2609.31306](https://arxiv.org/abs/2609.31306)

    该论文针对表格基础模型中独特的二维行/列注意力模式构建了可复现的基准测试，并在多种注意力后端上系统评估其性能，填补了高效注意力研究在二维表格场景中的空白。

    

    诸如 TabPFN、Mitra 或 ConTextTab 等表格上下文学习器依赖于对潜在嵌入的二维序列交替进行行注意力和列注意力处理。这些注意力模式与语言模型中的一维情形存在显著差异：行注意力涉及更长的序列，而列注意力则作用于短得多的序列，且表格数据的跨步内存布局使得生成连续张量的开销很高。此外，与近期的语言模型相比，当前模型所使用的隐藏维度较小。然而，高效注意力机制的研究大多集中在一维序列上，二维表格场景尚未被探索。为此，我们构建了一个可复现的基准测试环境，并在多个后端——Torch SDPA（高效版和 cuDNN 版）、FlashAttention-2/3/4，以及仅用于推理的后端 vLLM 和 SageAttention——上研究了表格注意力的独特特性，测量了前向和后向吞吐……

    arXiv:2609.31306v2 Announce Type: new  Abstract: Tabular in-context learners such as TabPFN, Mitra, or ConTextTab rely on alternating row and column attention over 2D sequences of latent embeddings. These attention patterns differ markedly from the one-dimensional case in language models: row attention involves longer sequences while column attention operates on much shorter ones, and the strided memory layout of tabular data makes producing contiguous tensors costly. Moreover, the hidden dimensions used in current models are small compared to recent language models. Yet efficient attention has been studied mostly for one-dimensional sequences, leaving the two-dimensional tabular setting unexplored. To this end, we create a reproducible benchmarking setup and study the unique characteristics of tabular attention across several backends -- Torch SDPA (efficient and cuDNN), FlashAttention-2/3/4, and the inference-only backends vLLM and SageAttention -- measuring forward and backward thro
    
[^42]: 随机Nesterov加速的几何矩收缩

    Geometric Moment Contraction for Stochastic Nesterov Acceleration

    [https://arxiv.org/abs/2609.31303](https://arxiv.org/abs/2609.31303)

    该论文为常参数随机Nesterov加速算法建立了几何矩收缩理论，给出了仅要求梯度具有有限 $p$ 阶矩的显式步长收缩判据，即使梯度具有无穷方差（$1<p<2$）也能保证 $L^p$ 收敛。

    

    我们研究了常参数随机Nesterov迭代的几何矩收缩（GMC）：\[ Y_k=\Theta_k+\beta(\Theta_k-\Theta_{k-1}),\qquad \Theta_{k+1}=Y_k-\gamma G(Y_k,X_{k+1}). \] 在均值强单调性和随机 $L^p$ Lipschitz 连续性条件下，通过一个显式的Perron比较，证明了当 $\beta\gamma L_p<(1-\beta)(1-q_{\gamma,p})$ 时存在同步的 $L^p$ 收缩。这一直接判据涵盖了 $1<p<2$ 时梯度具有无穷方差的情形，但其小步长机制要求 $\beta<\mu/(\mu+L_p)$。作为补充，一个幂-Lyapunov论证仅利用梯度有限的 $p$ 阶矩，就为每个固定的 $\beta<1$ 和每个 $p>1$ 建立了一个正的（通常小得多）步长区间。在 $p=2$ 时，一个更简单的显式判据给出 \[ 0<\gamma<\frac{2\mu(1-\beta)^2}{L_2^2(1-\beta+2\beta^2)}. \] 其关于高动量的二次缩放是所选度量方式的局限，而非精确的稳定性边界。我们量化了这一损失，并证明……

    arXiv:2609.31303v1 Announce Type: new  Abstract: We study geometric moment contraction (GMC) of the constant-parameter stochastic Nesterov recursion \[ Y_k=\Theta_k+\beta(\Theta_k-\Theta_{k-1}),\qquad \Theta_{k+1}=Y_k-\gamma G(Y_k,X_{k+1}). \] Under mean strong monotonicity and stochastic $L^p$ Lipschitz continuity, an explicit Perron comparison proves synchronous $L^p$ contraction when $\beta\gamma L_p<(1-\beta)(1-q_{\gamma,p})$. This direct criterion includes infinite-variance gradients for $1<2$, but its small-step regime requires $\beta<\mu/(\mu+L_p)$. A complementary power-Lyapunov argument establishes a positive, generally much smaller, step-size interval for every fixed $\beta<1$ and every $p>1$, using only a finite $p$th gradient moment. At $p=2$, a simpler explicit certificate gives \[ 0<\gamma<\frac{2\mu(1-\beta)^2}{L_2^2(1-\beta+2\beta^2)}. \] Its quadratic high-momentum scaling is a limitation of the chosen metric, not a sharp stability boundary. We quantify this loss, prov
    
[^43]: 面向输出头量化的Softmax重参数化

    Softmax Reparameterization for Output-Head Quantization

    [https://arxiv.org/abs/2609.31291](https://arxiv.org/abs/2609.31291)

    提出softmax重参数化这一训练后量化方法，通过在量化前减去词表行均值的标量倍数来选取功能等价的输出头，在保持全精度softmax分布不变的同时显著降低输出头量化对语言模型预测的扭曲。

    

    arXiv:2609.31291v1 公告类型：交叉 摘要：大型词表使得输出头成为小型语言模型中相当可观的推理成本。我们提出softmax重参数化，这是一种训练后方法，可在量化之前选择一个功能等价的输出头。该方法从每个输出行中减去词表行均值的标量倍数，并分别针对RTN、激活加权MSE和全Hessian GPTQ通过验证集KL散度来选择系数。这一维搜索涵盖了原始输出头和固定均值中心化两种情形，保持全精度softmax分布不变，且不改动已训练的解码器；秩一修正则可处理诸如soft-capping之类的非线性logit路径。在七个输出头上的实验表明，W4量化收益集中在基线量化严重扭曲预测的场景：在Phi-4-mini上，AW-MSE的KL散度从0.936降至0.256。这些收益在更强的GPTQ校准下依然保持，并与精确的逐通道缩放和仿射量化互为补充。

    arXiv:2609.31291v1 Announce Type: cross  Abstract: Large vocabularies make output heads a substantial inference cost in small language models. We propose softmax reparameterization, a post-training method that selects a functionally equivalent output head before quantization. The method subtracts a scalar multiple of the vocabulary-row mean from every output row and selects the coefficient by validation KL separately for RTN, activation-weighted MSE, and full-Hessian GPTQ. This one-dimensional search includes the original head and fixed mean-centering, preserves the full-precision softmax distribution, and leaves the trained decoder unchanged; a rank-one correction handles nonlinear logit paths such as soft-capping. Across seven heads, W4 gains concentrate where baseline quantization substantially distorts predictions: on Phi-4-mini, AW-MSE KL falls from 0.936 to 0.256. The gains survive stronger GPTQ calibration and remain complementary to exact per-channel scaling and affine quantiza
    
[^44]: 动态张量重物化中的确定性运行状态切换与可行性反转

    Deterministic Regime Switching and Feasibility Inversion in Dynamic Tensor Rematerialization

    [https://arxiv.org/abs/2609.31250](https://arxiv.org/abs/2609.31250)

    论文在DTR参考模拟器上发现，仅相差0.10%的内存预算即可确定性地切换快慢两种执行状态（开销差高达7.3倍），且在ResNet-32上出现“预算增大反而导致OOM”的确定性可行性反转现象，揭示了DTR的细粒度确定性不稳定性。

    

    我们在使用公开执行轨迹、基于参考DTR模拟器的测量中，报告了动态张量重物化——一种面向内存受限DNN训练的在线逐出策略——中细粒度、确定性的不稳定性。在LSTM轨迹上，仅相差无约束峰值内存0.10%的内存预算即可选出快、慢两种执行状态，其开销差异高达7.3倍；慢状态由对相同存储的广泛重复再逐出所驱动（每个存储的逐出次数从1.33升至8.27，而不同被逐出存储的集合基本不变：5,233对5,236，两集合的Jaccard重叠度为0.999）。在ResNet-32轨迹上，精细的预算扫描揭示了一种确定性的可行性反转：运行在比率为0.101时可行，在0.102-0.106区间不可行（内存溢出OOM），而从0.107起再次可行。我们将OOM的直接原因追溯到一个被完全钉住的递归重物化边界……（原文摘要在此处截断）

    arXiv:2609.31250v1 Announce Type: new  Abstract: We report fine-grained, deterministic instability in Dynamic Tensor Rematerialization (DTR), an online eviction policy for memory-constrained DNN training, measured on the reference DTR simulator (simrd) using public execution traces. On an LSTM trace, memory budgets differing by 0.10% of unconstrained peak memory select fast and slow execution regimes whose overheads differ by as much as 7.3x; the slow regime is driven by broadly repeated re-eviction of the same storages (evictions per storage rise from 1.33 to 8.27 while the set of distinct evicted storages is essentially unchanged: 5,233 vs 5,236, with the two sets overlapping at Jaccard 0.999). On a ResNet-32 trace, a fine budget sweep reveals a deterministic feasibility inversion: the run is feasible at ratio 0.101, infeasible (OOM) across 0.102-0.106, and feasible again from 0.107. We trace the immediate cause of the OOM to a fully pinned recursive rematerialization frontier that e
    
[^45]: 面向冻结口袋条件分子扩散的预算化商残差引导

    Budgeted Quotient-Residual Guidance for Frozen Pocket-Conditioned Molecular Diffusion

    [https://arxiv.org/abs/2609.31222](https://arxiv.org/abs/2609.31222)

    提出推理时的预算化商残差引导（QRG），无需重训冻结的口袋条件分子扩散模型，即可借助商几何定向并以采样器自身步长作为信任预算，激活距离、接触等商空间优化目标。

    

    口袋条件分子扩散更新的是环境原子坐标，但许多先导化合物优化目标却表达在商特征（如距离、接触和锚定子结构）上。我们提出预算化商残差引导（QRG），这是一种推理时校正方法，无需重新训练分子生成器即可激活这些商目标。QRG 将商余向量提升为度量水平的（metric-horizontal）环境方向，并通过由冻结采样器自身步长范数设定的信任预算来传递：商几何决定方向，采样器运动约束尺度。我们推导了水平提升、闭式采样器预算更新、冻结反向步骤附近的 KL/动力学解释、等变性条件，以及用于预算受限的截面与残差控制的乘积预算拆分。受控商任务证实，相对采样器的传递方式能够激活原始局部商梯度所难以激活的信号……

    arXiv:2609.31222v1 Announce Type: new  Abstract: Pocket-conditioned molecular diffusion updates ambient atom coordinates, but many lead-optimization objectives are expressed on quotient features such as distances, contacts, and anchored substructures. We introduce budgeted quotient-residual guidance (QRG), an inference-time correction that makes these quotient objectives active without retraining the molecular generator. QRG lifts quotient covectors to metric-horizontal ambient directions and delivers them through a trust budget set by the frozen sampler's own step norm: quotient geometry chooses the direction, while sampler motion bounds the scale. We derive the horizontal lift, closed-form sampler-budget update, KL/kinetic interpretation around a frozen reverse step, equivariance conditions, and a product-budget split for budget-capped section and residual controls. Controlled quotient tasks confirm that sampler-relative delivery activates signals that raw local quotient gradients le
    
[^46]: 我们在估计哪种影响？反事实规范在数据归因中的作用

    Which Influence Are We Estimating? The Role of Counterfactual Specifications in Data Attribution

    [https://arxiv.org/abs/2609.31214](https://arxiv.org/abs/2609.31214)

    该论文指出数据归因中各影响估计器排序不一致的根本原因是“规范不匹配”而非近似误差，将影响形式化为反事实估计量，并按隐含规范对现有估计器进行系统分类。

    

    估计训练样本对模型行为的影响对于数据调试、数据估值和数据归因至关重要。现有的影响估计器常常产生互不相容的排序，这通常被归因于近似误差。我们认为，一个更根本的分歧来源是规范不匹配：影响取决于被归因的行为、施加于每个训练样本的干预，以及将干预映射到模型响应的反事实训练过程。当目标行为需要一个可处理的替代量（例如查询损失、logit 或间隔）时，这些选择尤为重要。我们将影响形式化为一个反事实估计量，区分了不同估计量之间的规范不匹配与估计固定估计量时产生的近似误差，并按照各估计器隐含的规范对代表性方法进行了分类整理。我们进一步推导出一个局部分解，揭示了行为……（摘要在此处截断）

    arXiv:2609.31214v1 Announce Type: new  Abstract: Estimating the influence of training examples on model behavior is essential for data debugging, valuation, and attribution. Existing influence estimators often produce incompatible rankings, which are commonly ascribed to approximation error. We argue that a more fundamental source of disagreement is specification mismatch: influence depends on the behavior being attributed, the intervention applied to each training example, and the counterfactual training process that maps the intervention to a model response. These choices are especially important when the target behavior requires a tractable surrogate, such as query loss, a logit, or a margin. We formalize influence as a counterfactual estimand, distinguish specification mismatch across estimands from approximation error in estimating a fixed estimand, and organize representative estimators by their implied specifications. We further derive a local decomposition that exposes how beha
    
[^47]: 自监督表示学习：从光谱基础模型到极光发射光谱

    Self-Supervised Representation Learning: From Spectral Foundation Models to Auroral Emission Spectra

    [https://arxiv.org/abs/2609.31206](https://arxiv.org/abs/2609.31206)

    该论文在22.3万条无标注极光光谱上用掩码自编码器预训练一维Vision Transformer，其学到的表示无需标签即可恢复用于物理诊断的发射线强度比，微调后在极光分类基准上超越监督式分类器，且仅用10%的标签就大幅领先从头训练的模型。

    

    诸如Skibotn极光光谱仪（ASIS）等极光光谱仪记录了数十万条发射光谱，但其中仅有数百条能够由专家进行标注。为了利用其余未标注的数据，我们在223,000条无标注光谱上使用掩码自编码器预训练了一个一维Vision Transformer。在完全没有标签的情况下，其学到的表示能够恢复物理学家用于诊断沉降粒子的发射线强度比（R²为0.91，而未训练的对照组仅为0.77），并且仅通过一个线性探针，其分类能力就可以媲美专家设计的13个特征。经过微调后，该模型在原有基准上超越了此前的监督式极光分类器（macro-AP为88.5对比77.8），达到0.870 mAP，并且仅使用10%的标签就比从头训练的相同架构高出+0.159；归因分析表明该模型同时利用了N2+波段。那么现有的预训练模型能否替代它？两个天文光谱基础模型和一个时间序列模型（摘要在此处被截断）

    arXiv:2609.31206v1 Announce Type: new  Abstract: Auroral spectrographs such as the Auroral Spectrograph In Skibotn (ASIS) record hundreds of thousands of emission spectra, but only a few hundred can be labelled by an expert. To exploit the rest, we pretrain a 1D Vision Transformer with a masked autoencoder on 223,000 unlabelled spectra. Without labels, its representation recovers the emission-line intensity ratios that physicists use to diagnose the precipitating particles (R^2 0.91 vs. 0.77 for an untrained control) and, under one linear probe, classifies as well as 13 features designed by experts. Fine-tuned, the model outperforms the previous supervised auroral classifier on its own benchmark (macro-AP 88.5 vs. 77.8), reaches 0.870 mAP, and exceeds the same architecture trained from scratch by +0.159 with 10% of the labels; attribution shows that it uses both N2+ bands. Could an existing pretrained model replace it? Two astronomical spectral foundation models and a time-series model
    
[^48]: ALF：一个用于科学发现的主动学习框架

    ALF: An Active Learning Framework for Scientific Discovery

    [https://arxiv.org/abs/2609.31197](https://arxiv.org/abs/2609.31197)

    ALF是一个开源的模块化主动学习框架，通过五个模块化组件运行完整的数据获取循环，统一支持离线基准测试和在线部署两种场景，以加速科学发现。

    

    用于科学发现的机器学习几乎总是受到数据的限制。在预算约束下生成相关的高质量数据，是推进该领域最有前景的途径之一。主动学习（AL）在标注需要昂贵实验、测量或模拟的场景中展现出巨大潜力。现有的大多数工具只覆盖数据获取循环的一部分，通常要么专注于离线基准测试，要么专注于在线部署，而不能两者兼顾。我们提出了ALF，一个模块化的主动学习框架，它通过五个模块化组件运行完整的数据获取循环，并为两种场景提供了一个清晰的API：离线场景下，针对现有数据集进行可控且可复现的实验；在线场景下，针对oracle在真实世界部署中获取新的候选数据。ALF是开源的，可在 https://github.com/instadeepai/alf 获取。

    arXiv:2609.31197v1 Announce Type: new  Abstract: Machine learning for scientific discovery is almost systematically data bound. Producing relevant high quality data, under budget constraints, is amongst the most promising ways to advance the field. Active learning (AL) offers promise wherever labelling requires expensive experiment, measurement, or simulation. Most existing tools cover only part of the data acquisition loop, and typically focus on either offline benchmarking or online deployment, but not both. We present ALF, a modular AL Framework that runs the full data acquisition loop via five modular components. One clear API for both settings: offline, against an existing dataset for controlled and reproducible experimentation; and online, against an oracle for acquiring new candidates in real-world deployments. ALF is open-source and available at https://github.com/instadeepai/alf.
    
[^49]: 考虑偏差实现可持续的大语言模型评估

    Accounting for Bias Enables Sustainable LLM Evaluation

    [https://arxiv.org/abs/2609.31184](https://arxiv.org/abs/2609.31184)

    该论文提出一个统一的潜在变量框架，通过显式校正位置偏差、冗长偏差、评委严格度等系统性测量偏差，能用远少于以往的比较次数恢复可靠排名，从而为LLM评估提供了统计上更严谨、计算上更可持续的方案。

    

    以大语言模型作为评委（LLM-as-a-judge）已成为可扩展主观评估的事实标准，然而当前的排行榜通过进行越来越多的比较来补偿系统性测量偏差，这种方法在统计上不健全且在计算上浪费。其根本原因在于测量模型不完整：将LLM评委视为中性的、可互换的测量工具，忽视了已有文献记录的多种偏差，如位置偏差、冗长偏差、评委严格度以及自我增强偏好，而这些偏差无法通过增加数据量来消除。我们提出了一个统一的潜在变量框架，在联合建模成对比较和序数数据的同时显式校正这些混杂因素，从而从大幅减少的比较次数中恢复出可靠的排名。由于拟合该模型的计算成本相对于单轮LLM推理而言可以忽略不计，偏差校正不仅在统计上更为严谨，也是实现可信评估的一种更具可持续性的方法。

    arXiv:2609.31184v1 Announce Type: new  Abstract: LLM-as-a-judge has become the de facto standard for scalable, subjective evaluation, yet current leaderboards compensate for systematic measurement bias by running ever more comparisons, an approach that is both statistically unsound and computationally wasteful. The root cause is an incomplete measurement model, treating LLM judges as neutral, interchangeable instruments ignores documented biases like position bias, verbosity bias, judge severity, and self-enhancement, that no volume of additional data can eliminate. We propose a unified latent variable framework that jointly models pairwise and ordinal data while explicitly correcting for these confounders, recovering reliable rankings from substantially fewer comparisons. Because fitting this model costs negligible compute relative to a single round of LLM inference, bias correction is not only more statistically rigorous but also a more sustainable approach to trustworthy evaluation.
    
[^50]: BAT-CLIP：大脑、音频与文本的三模态对齐

    BAT-CLIP: Trimodal Alignment of Brain, Audio and Text

    [https://arxiv.org/abs/2609.31180](https://arxiv.org/abs/2609.31180)

    BAT-CLIP提出了首个面向iEEG的CLIP式三模态对齐框架，将神经嵌入同时对齐到预训练的音频和文本锚点，克服了单模态锚定带来的权衡问题，在自然语音解码中实现了比双模态基线更鲁棒的表示。

    

    从大脑中解码和解释自然语音越来越依赖于与预训练的语音和语言表示空间的对齐。然而，当前的CLIP式脑-语音对齐方法将神经活动锚定到单一的锚定模态——音频或文本——尽管大脑的语音处理本质上是多模态的。这导致了一种权衡：音频锚定保留了时间结构但削弱了语言可分性，而文本锚定捕获了语义却丢弃了声学细节。我们提出BAT-CLIP，这是首个面向iEEG（颅内脑电图）的CLIP式三模态对齐框架，它在一个共享的、冻结的音频-文本流形中将神经嵌入同时对齐到预训练的音频和文本锚点。在自然播客基准测试中，BAT-CLIP比双模态CLIP基线获得了更鲁棒的表示。我们还强调了使用自监督基础模型进行CLIP训练的重要性。

    arXiv:2609.31180v1 Announce Type: cross  Abstract: Decoding and interpreting naturalistic speech from the brain increasingly relies on alignment to pretrained speech and language representation spaces. However, current CLIP-style brain-speech alignment ground neural activity to a single anchor modality-audio or text-despite the brain's inherently multimodal speech processing. This induces a trade-off: audio anchoring preserves temporal structure but weakens linguistic separability, while text anchoring captures semantics yet discards acoustic detail. We propose BAT-CLIP, the first CLIP-style trimodal alignment framework for iEEG that jointly aligns neural embeddings to both pretrained audio and text anchors in a shared, frozen audio-text manifold. On the naturalistic Podcast benchmark, BAT-CLIP yields more robust representations than bimodal CLIP baselines. We also highlight the importance of using self-supervised foundation models for CLIP training.
    
[^51]: 面向非典型听觉的音频情感识别

    Audio emotion recognition for atypical hearing

    [https://arxiv.org/abs/2609.31168](https://arxiv.org/abs/2609.31168)

    该研究提出用LoRA微调CLAP基础模型，从少量神经典型听众的标注数据中泛化音频情感识别能力，以探索自闭症人士听觉过敏等非典型听觉情境下的情感评估方法。

    

    我的博士工作旨在探索非典型听觉情境下的音频情感识别（AER）。这项研究聚焦于自闭症人士的听觉过敏现象，这一现象往往难以评估且因人而异。我们的核心思想是利用从声学特征中对情感的理解，依赖从少量标注数据中泛化情感反应的可能性。作为第一步，我们使用低秩自适应（LoRA）方法对一个大型基础模型——对比语言-音频预训练（CLAP）进行微调，该模型在神经典型听众的效价和唤醒度数据集上进行训练。

    arXiv:2609.31168v1 Announce Type: new  Abstract: My doctoral work aims to explore Audio Emotion Recognition (AER) in the context of atypical listening. This research focuses on auditory hypersensitivity in people with autism, a phenomenon that is often difficult to evaluate and unique to each individual. Our core idea is to leverage our understanding of affect from acoustic traits, relying on the possibility of generalizing affective responses from a small amount of annotated data. As a first step, we fine-tune a large foundation model, Contrastive Language-Audio Pretraining (CLAP) using low-rank adaptation (LoRA), trained on a valence and arousal dataset of neurotypical listeners.
    
[^52]: BreathGRU：一种用于呼吸音频中语音与呼吸分割的新型半监督双向门控循环单元框架

    BreathGRU: A Novel Semi-Supervised Bidirectional Gated Recurrent Unit Framework for Speech and Breath Segmentation for Respiratory Audio

    [https://arxiv.org/abs/2609.31165](https://arxiv.org/abs/2609.31165)

    提出BreathGRU框架，通过将帧级声学特征、双向循环建模、伪标签优化与持续时间约束的分段维特比解码相结合，首次实现针对呼吸音频中语音与呼吸事件的精确半监督分割，克服了传统VAD方法将呼吸误判为静音的局限。

    

    语音-呼吸分割是呼吸音频分析中的一项基础性预处理步骤，它支持呼吸声学生物标志物提取、肺功能预测和疾病监测等应用。现有方法，包括阈值法、基于傅里叶变换的技术以及无监督和预训练的语音活动检测（VAD）模型，主要专注于语音检测，往往将呼吸事件归类为非语音或静音，这限制了它们在精确呼吸检测方面的适用性。为解决这一局限，我们提出了BreathGRU，一种专为语音-呼吸分割设计的半监督双向门控循环单元（BiGRU）框架。该框架将帧级声学特征提取与双向循环建模、伪标签优化以及持续时间约束的分段维特比（Viterbi）解码相结合，从而实现语音与呼吸的分割。（注：原文摘要在此处截断）

    arXiv:2609.31165v1 Announce Type: cross  Abstract: Speech-breath segmentation is a fundamental preprocessing step in respiratory audio analysis, enabling applications such as respiratory acoustic biomarker extraction, lung function prediction and disease monitoring. Existing approaches, including threshold methods, Fourier Transform-based techniques, and unsupervised and pretrained voice activity detection (VAD) models, primarily focus on speech detection and often classify breathing events as non-speech or silence, limiting their applicability for precise breath detection. To address this limitation, we propose BreathGRU, a semi-supervised Bidirectional Gated Recurrent Unit (BiGRU) framework specifically designed for speech-breath segmentation. The proposed framework combines frame-level acoustic feature extraction with bidirectional recurrent modelling, pseudo-label refinement and duration-constrained Segmental Viterbi decoding to produce speech and breath segmentation. BreathGRU was
    
[^53]: SPADE：突破流行度-相似度边界以度量意外性推荐

    SPADE: Escaping the Popularity-Similarity Frontier to Measure Serendipitous Recommendations

    [https://arxiv.org/abs/2609.31164](https://arxiv.org/abs/2609.31164)

    提出SPADE评估指标，将物品映射到二维流行度-相似度空间并计算用户特定的帕累托前沿距离，从而同时考量流行度、相似度与用户实际相关性，有效度量意外性推荐并防止算法投机。

    

    推荐系统通过设计意外性（serendipity）来促进用户的主动探索，并打破可预测的消费循环。现有离线“准确性之外”的评估指标存在的问题是，它们往往只孤立地考察历史相似性或全局流行度。我们的目标是设计一个能够同时考察相似性、流行度和用户实际相关性的评估指标。为此，我们提出了SPADE（意外性帕累托距离评估，Serendipitous Pareto Distance Evaluation）。SPADE将所有物品映射到二维空间中，直接为每个用户计算由流行度最高且历史最相似的物品构成的帕累托前沿。最终的意外性得分通过严格针对测试集中被正确推荐的物品，计算它们到该边界的最小欧氏距离并取平均值而得到。在五个数据集和五种基线算法上的评估证实了SPADE的有效性；我们的结果表明，该指标成功防止了算法对“准确性之外”评估指标的投机利用。

    arXiv:2609.31164v1 Announce Type: cross  Abstract: Recommender systems engineer serendipity to foster active exploration and break predictable consumption cycles. The problem with existing offline beyond-accuracy metrics is that they often either isolate historical similarity or global popularity. We aim to design an evaluation metric that examines similarity, popularity, and actual user relevance. To achieve this, we introduce SPADE (Serendipitous Pareto Distance Evaluation). SPADE maps all items into a two-dimensional space to directly calculate a user-specific Pareto frontier of maximally popular and historically similar items. The final serendipity score is then computed by averaging the minimum Euclidean distance from this boundary strictly for the correctly recommended test-set items. Evaluating SPADE across five datasets and five baseline algorithms confirms its effectiveness; our results show that the metric successfully prevents algorithms from exploiting beyond-accuracy measu
    
[^54]: WorldTS：面向多模态协变量感知时间序列预测的世界建模

    WorldTS: World Modeling for Multimodal Covariate-aware Time Series Forecasting

    [https://arxiv.org/abs/2609.31162](https://arxiv.org/abs/2609.31162)

    WorldTS提出了一种基于世界建模的时间序列预测框架，将多模态外部协变量直接整合到潜在状态的建模与演化过程中，从而提升未来观测的预测性能。

    

    时间序列预测通常被定义为在观测空间中学习从历史观测到未来观测的直接映射。然而，观测序列通常只能提供底层系统动态的部分视图，未来观测会受到潜在动态的影响。因此，近期的潜在空间预测方法通过从历史观测的潜在空间表示来预测未来观测，而不是直接在观测空间中预测未来观测，从而取得了更好的性能。此外，虽然未来观测也受外部因素的影响，但如何将外部的、通常是多模态的信息纳入预测，使其能够直接影响潜在状态的形成与演化，这一问题仍未得到充分探索。我们提出了WorldTS，这是一个基于世界建模的预测框架，它将多模态协变量直接整合到预测中，以进一步改进……（摘要原文在此处截断）

    arXiv:2609.31162v1 Announce Type: new  Abstract: Time series forecasting is typically framed as learning a direct mapping from historical to future observations in the observation space. However, sequences of observations generally provide only a partial view of the dynamics of the underlying system, with future observations being shaped by latent dynamics. Recent latent-space forecasting methods thus achieve improved performance by predicting future observations from latent-space representations of historical observations rather than directly forecasting future observations in the observation space. Next, while future observations are also shaped by external factors, how to incorporate external, often multimodal, information into forecasting, so that it can shape latent-state formation and evolution directly, remains underexplored. We propose WorldTS, a world-modeling based forecasting framework that integrates multimodal covariates directly into the forecasting to further improve for
    
[^55]: 我行动故我在：JEPA 的动作条件化何时足以学习因果机制？

    I Act Therefore I Am: When Is JEPA's Action-Conditioning Enough to Learn Causal Mechanisms?

    [https://arxiv.org/abs/2609.31161](https://arxiv.org/abs/2609.31161)

    该论文通过隐变量模型与结合条件似然最大化和熵最大化的信息论目标，系统回答了 JEPA 的动作条件化在何种条件下足以从高维观测中恢复底层因果状态。

    

    近期的实证与理论进展表明，联合嵌入预测架构（JEPAs）能够为以动作为条件的未来结果预测学习有意义的表示，从而成为世界模型的基础结构之一。然而，准确的预测通常并不必然意味着能够恢复产生所观测动力学的底层因果状态。本工作研究了 JEPA 何时以及如何能够从观测中恢复底层因果状态。我们首先引入了一个隐变量模型，其中高维观测由潜在因果状态生成，而这些状态的动力学由动作条件化的转移机制所支配。基于这一表述，我们开发了一个通用的信息论目标函数，它将用于学习转移动力学的条件似然最大化与用于保留潜在状态信息的熵最大化相结合。随后我们确立了（原文摘要在此处截断）

    arXiv:2609.31161v1 Announce Type: new  Abstract: Recent empirical and theoretical advances suggest that joint-embedding predictive architectures (JEPAs) may learn meaningful representations for action-conditioned prediction of future outcomes, thus becoming one of the foundational structures for world models. However, accurate prediction does not, in general, necessarily imply recovery of underlying causal states that give rise to the observed dynamics. This work investigates when and how JEPAs can recover the underlying causal states from observations. We first introduce a latent variable model, in which high-dimensional observations are generated from latent causal states whose dynamics are governed by action-conditioned transition mechanisms. Based on this formulation, we develop a general information-theoretic objective that combines conditional likelihood maximization for learning transition dynamics with entropy maximization for preserving latent state information. We then establ
    
[^56]: 面向多维时间序列异常检测的融合物理信息预测先验的贝叶斯张量自编码器

    Bayesian Tensor Autoencoder with Physics-informed Predictive Prior for Multi-dimensional Time Series Anomaly Detection

    [https://arxiv.org/abs/2609.31157](https://arxiv.org/abs/2609.31157)

    该论文提出了一种融合物理信息预测先验的贝叶斯张量自编码器，通过直接以张量形式处理多维时间序列并弥合基于重构与基于预测两类自编码器之间的信息利用差距，从而提升多维时间序列异常检测的性能。

    

    多维时间序列本质上具有张量结构，在实践中十分常见。尽管时间序列异常检测领域已取得巨大进展，但大多数现有方法仅局限于一元/多元时间序列。在使用这些方法处理多维时间序列时，需要进行重塑操作，这不可避免地破坏了数据的内在相关性，从而导致性能下降。在一元/多元时间序列异常检测中，自编码器（AE）被广泛采用，通常可分为基于重构的自编码器和基于预测的自编码器两类。基于重构的自编码器利用当前观测值进行重构，而基于预测的自编码器则利用历史信息来预测当前观测值，因此这两类自编码器所利用的信息不同。为了弥合基于重构与基于预测的自编码器之间的差距，从而充分利用可用信息并进一步提升性能……（摘要在此处截断）

    arXiv:2609.31157v1 Announce Type: new  Abstract: Multi-dimensional time series, inherently tensorial, are common in practice. Despite great progress in time series anomaly detection, most existing methods are confined to uni-/multi-variate time series. When handling multi-dimensional time series using these methods, reshaping operations are required, which inevitably break the intrinsic correlations and thus lead to performance degradation. In uni-/multi-variate time series anomaly detection, AutoEncoders (AEs) are widely adopted and generally categorized into reconstruction-based and prediction-based AEs. The reconstruction-based AE utilizes the current observation for reconstruction, while the prediction-based AE utilizes the historical information to predict the current observation. Thus, the two AEs utilize different information. To bridge the gap between reconstruction-based and prediction-based AEs, so as to fully leverage the available information and thus further enhance perfor
    
[^57]: 域偏移下基于教师锚定的训练后量化模型选择

    Teacher-Anchored Selection of Post-Training Quantized Models under Domain Shift

    [https://arxiv.org/abs/2609.31155](https://arxiv.org/abs/2609.31155)

    该论文提出在域偏移且目标域标签缺失或稀缺的场景下，以教师模型失真作为锚点来指导训练后量化候选模型的部署选择，揭示了置信度类估计器的失效与输出分布类估计器的可靠，并将有监督项与教师锚点相结合以改进选择。

    

    压缩一个已训练好的模型会产生一系列部署候选模型，而在域偏移条件下，压缩程度最高的候选模型并不一定是应当部署的那一个。我们研究了在这类候选模型族上进行选择的问题，其中候选模型与教师模型固定不变，且目标域标签缺失或稀缺。两个发现组织了无标签情形下的研究。最小教师失真几乎表现为一种恒定规则，在每次运行中都选出同一个八比特、逐通道、无裁剪的配置，而该配置并不能最小化目标域的经验交叉熵。现有估计器则呈现明显分化：在CNN模型族的过度自信坍塌情形中，基于置信度的估计器对模型族的排序几乎完全颠倒，而识别这种情形的诊断方法又需要该设定下无法获得的标签；相比之下，基于输出分布的估计器与教师相对锚点相匹配，并在某一架构上超越了它。尽管如此，失真本身是稳定的，因此一个有监督的项可以将选择从锚点上移开。将两者结合，我们……（摘要在此处截断）

    arXiv:2609.31155v1 Announce Type: cross  Abstract: Compressing a trained model yields a family of deployment candidates, and under domain shift the most compressed one need not be the one to deploy. We study selection over such a family, with candidates and teacher fixed and target labels absent or scarce. Two findings organize the label-free case. Minimum teacher distortion behaves almost as a constant rule, selecting the same eight-bit, per-channel, unclipped configuration in every run, which does not minimize empirical target cross-entropy. Established estimators divide sharply: in the overconfident-collapse regime of the CNN families, confidence-based estimators order the family close to backwards, and the diagnostics that identify it need the labels the setting denies, while output-distribution estimators match the teacher-relative anchor and on one architecture beat it. Distortion is nonetheless stable, so a supervised term can move selection away from it. Combining the two, we g
    
[^58]: CRNDiff：基于化学反应网络的计数原生扩散框架

    CRNDiff: Count-Native Diffusion Framework via Chemical Reaction Networks

    [https://arxiv.org/abs/2609.31149](https://arxiv.org/abs/2609.31149)

    CRNDiff基于化学反应网络构建计数原生扩散框架，利用闭式生灭转移核实现精确的计数空间扩散建模，并通过倾斜费曼-卡茨导向方法在无需重新训练的情况下对稀有亚群进行条件采样。

    

    单细胞RNA测序（scRNA）等科学测量数据通常以非负整数计数的形式呈现，而连续状态扩散模型则使用连续坐标来近似这种离散结构。基于随机化学反应网络（CRN）——一类计数原生的马尔可夫跳跃过程——我们提出了CRNDiff，这是一个将计数空间扩散与推理时稀有亚群条件化相结合的结构化框架。通过一个独立的生灭过程实例化，我们得到了前向加噪过程的闭式转移核。该转移核使得反向采样可以通过前向滤波-后向采样（FFBS）实现，并支持数据驱动的终止加噪时间选择，从而省去了验证性扫描的需要。这种可解性还使我们能够提出倾斜费曼-卡茨（FK）导向方法，这是一种无需重新训练即可从冻结的生成器中采样目标亚群的方法。通过倾斜后验……（原文摘要在此处被截断）

    arXiv:2609.31149v1 Announce Type: new  Abstract: Scientific measurements such as single-cell RNA (scRNA) sequencing often take the form of nonnegative integer counts, whereas continuous-state diffusion models approximate this discrete structure using continuous coordinates. Building on stochastic chemical reaction networks (CRNs), a class of count-native Markov jump processes, we introduce CRNDiff, a structured framework that combines count-space diffusion with inference-time conditioning on rare subpopulations. An independent birth--death instantiation yields a closed-form transition kernel for forward noising. This kernel enables reverse sampling via forward-filtering backward-sampling (FFBS) and supports data-driven selection of the terminal noising time, eliminating the need for a validation sweep. This tractability also lets us introduce tilted Feynman--Kac (FK) steering, a method for sampling target subpopulations from a frozen generator without retraining. By tilting posterior m
    
[^59]: 蒸馏加密流量分类器中的未知流量检测、校准与捷径依赖：为期一年的研究

    Unknown-Traffic Detection, Calibration and Shortcut Reliance in Distilled Encrypted-Traffic Classifiers over One Year

    [https://arxiv.org/abs/2609.31141](https://arxiv.org/abs/2609.31141)

    本研究通过预注册设计，从两个准确率相同但构造不同的教师模型蒸馏学生模型，并在一年的真实TLS流量上验证：学生会继承教师模型的未知流量检测能力（仅在常规蒸馏温度下）与捷径依赖特性，且这些特性会在长期数据漂移中发生变化。

    

    知识蒸馏是将加密流量分类器压缩以部署到边缘设备的标准方法，而几乎所有此类工作仅以准确率来评判学生模型。我们追问学生模型还继承了什么：未知流量检测能力、校准程度、捷径依赖，以及这些特性中哪些能在长达一年的数据漂移中保留下来。仅凭相似性本身说明不了什么，因为软目标同时也会起到正则化作用。因此，我们从两个准确率相同但构造不同的教师模型中蒸馏出一个101k参数的学生模型——一个是五成员集成模型，另一个是单个更宽的模型——如此一来，学生模型跟随某一个而非另一个的特性便可归因于教师模型的构造差异。该实验设计在任何测试结果产生之前即已预注册。我们在CESNET-TLS-Year22（一整年的真实TLS流量）上，跨越35周的18个测试窗口检验了十个假设。其中两个得到支持：学生模型的逐流未知评分会向其教师模型偏移，但仅在使用常规蒸馏温度时如此，而在准确率最优的温度下并非如此；以及捷径依赖（摘要原文在此处截断）。

    arXiv:2609.31141v1 Announce Type: cross  Abstract: Knowledge distillation is the standard way to compress encrypted-traffic classifiers for the edge, and almost all such work judges students by accuracy alone. We ask what else a student inherits: unknown-traffic detection, calibration, shortcut reliance, and whether any survives a year of drift. Resemblance proves little on its own, since soft targets also regularise. We therefore distil one 101k-parameter student from two teachers of equal accuracy but different construction, a five-member ensemble and a single wider model, so that following one rather than the other is attributable to it. The design was pre-registered before any test result was seen. We tested ten hypotheses on CESNET-TLS-Year22, a year of real TLS traffic, across 18 test windows over 35 weeks. Two are supported: a student's per-flow unknown-scores shift toward its own teacher, but only at a conventional temperature, not the accuracy-optimal one; and a shortcut-relia
    
[^60]: 框定对手：一种结构感知的攻击方法

    Frame the adversary: a structure-aware attack methodology

    [https://arxiv.org/abs/2609.31128](https://arxiv.org/abs/2609.31128)

    该论文提出了一种基于非正交变换的结构感知频率对抗攻击方法，将攻击构造形式化为带扰动约束集的优化问题，并证明攻击等价于加权 ℓ2 投影，从而实现通用且可控的攻击生成。

    

    基于频率的对抗攻击近来日益流行，其原理是利用不同神经架构所共有的频谱敏感性。与空间扰动不同，基于频率的攻击能够暴露更深层的脆弱性，因此对于安全关键型和安全敏感型应用的鲁棒性评估尤为有价值。然而，现有方法通常并非作为显式捕捉变换域结构的优化问题的解而推导得出。本文提出了一种通过专门优化框架来构造有原则的基于频率的对抗攻击的方法论。该方法的一个基石在于引入了一个扰动约束集，该约束集与高度结构化的非正交变换相关联，而这类变换以灵活、非预定义的频率处理能力而著称。我们证明，攻击等价于到该约束集上的加权 ℓ2 投影，从而得到一个通用且可控的攻击生成……

    arXiv:2609.31128v1 Announce Type: new  Abstract: Frequency-based adversarial attacks have recently grown popular by exploiting spectral sensitivities shared across neural architectures. Unlike spatial perturbations, frequency-based attacks expose deeper vulnerabilities, making them especially valuable for robust evaluation of safety-critical and security-sensitive applications. Yet, existing approaches are typically not derived as solutions to an optimization problem that explicitly captures transform-domain structure. In this paper, we propose a methodology for crafting principled frequency-based adversarial attacks, via a dedicated optimization framework. A cornerstone of our method hinges on the introduction of a perturbation constraint set, tied to highly structured non-orthogonal transforms, well-known for their flexible, non-predefined frequency handling. We prove that the attacks emerge as weighted $\ell_2$-projections onto this set, yielding a general and controlled attack gene
    
[^61]: LocUS：基于头选择与子空间投影的定向激活引导方法

    LocUS: Head Selection and Subspace Projection for Targeted Activation Steering

    [https://arxiv.org/abs/2609.31122](https://arxiv.org/abs/2609.31122)

    提出 LocUS 方法，通过在解嵌矩阵中识别属性特定的线性子空间、将引导变换限制在该子空间并局部化到稀疏的注意力头子集，实现了更精准、对无关能力副作用更小的定向激活引导。

    

    激活引导是一种强大的免训练范式，可在推理时对大语言模型进行控制。然而，标准方法从对比数据中为每一层估计一个引导方向，并将其应用于该层的整个表示空间，这可能使干预与对比数据中存在的非目标属性相耦合，从而损害模型的无关能力。为缓解这一问题，我们提出了 LocUS（局部化解嵌引导，Localized Unembedding Steering），该方法将激活引导锚定到模型自身的输出词汇子空间中。通过在解嵌矩阵中识别特定属性的线性子空间，LocUS 施加了几何约束，将引导变换限制在特定子空间内，同时将其作用局部化到注意力头的稀疏子集上。研究在三个模型家族上，针对毒性缓解、情感重定向和谄媚抑制任务进行了广泛评估。

    arXiv:2609.31122v1 Announce Type: new  Abstract: Activation steering is a powerful training-free paradigm for controlling large language models at inference time. However, standard approaches estimate a per-layer steering direction from contrastive data and apply it on the layer's entire representation space, which may couple the intervention to off-target properties present in the contrastive data and degrade unrelated capabilities. To mitigate this issue, we introduce LocUS (Localized Unembedding Steering), a method which grounds activation steering to the model's own output vocabulary subspace. By identifying a property-specific linear subspace within the unembedding matrix, LocUS enforces a geometric constraint that restricts the steering transformation to a specific subspace and at the same time localizes its application to a sparse subset of attention heads. Extensive evaluations across three model families on toxicity mitigation, sentiment redirection and sycophancy suppression 
    
[^62]: 从捷径学习到离散神经插入排序

    From Shortcut Learning to Discrete Neural Insertion Sort

    [https://arxiv.org/abs/2609.31114](https://arxiv.org/abs/2609.31114)

    该论文揭示了神经算法推理模型在学习插入排序时存在捷径学习问题——中间表示在算法执行结束前就能解码出排好序的结果——并提出了一种将序列表示为链、分离标量交换与控制状态转移、且每步后将节点表示投影回离散状态的离散神经插入排序模型，以促使模型真正遵循算法执行过程。

    

    神经算法推理旨在训练神经网络遵循已知算法，并泛化到训练中未见过的输入规模。然而，正确的最终输出与中间监督并不一定表明模型遵循了预期的执行过程。我们以插入排序为研究对象来探讨这一问题。我们对 CLRS30 基线 NAR 的分析表明，提示目标仅被弱优化，提示准确率始终较低。此外，在参考插入排序执行终止之前，许多中间表示已经可以被解码为已排序的序列，这表明模型学到的是通往最终输出的捷径。受这些发现启发，我们提出了离散神经插入排序。我们的模型将序列表示为链，将标量交换与控制状态转移分离，并在每个处理器步骤后将节点表示投影回离散状态。当仅在序列（上训练时……）（原文摘要在此处截断）

    arXiv:2609.31114v1 Announce Type: cross  Abstract: Neural algorithmic reasoning aims to train neural networks to follow known algorithms and generalize beyond the input sizes seen during training. However, correct final outputs and intermediate supervision do not necessarily show that a model follows the intended execution. We study this problem using insertion sort. Our analysis of the CLRS30 baseline NAR shows that the hint objective is weakly optimized and that hint accuracy remains low. Moreover, many intermediate representations can already be decoded into sorted sequences before the reference insertion-sort execution terminates, suggesting that the model learns a shortcut to the final output. Motivated by these findings, we introduce Discrete Neural Insertion Sort. Our model represents the sequence as a chain, separates scalar exchanges from control-state transitions, and projects node representations back to discrete states after every processor step. When trained only on sequen
    
[^63]: 基于Fisher信息几何的贝叶斯优化：梯度界与信赖域方法

    Bayesian Optimization with Fisher Information Geometry: Gradient Bounds and Trust-Region Methods

    [https://arxiv.org/abs/2609.31107](https://arxiv.org/abs/2609.31107)

    本文通过拉回Fisher信息度量导出采集函数的梯度上界，解释了高维贝叶斯优化中的梯度消失现象，并提出基于信赖域的FITR方法，以局部Fisher权重取代长度尺度缩放来提升优化性能。

    

    我们从信息几何的视角研究贝叶斯优化（BO）。通过代理后验映射拉回Fisher信息度量，可以在输入空间上得到一个局部敏感性张量，从而为可重参数化采集函数的梯度提供一个上界。这一视角解释了高维贝叶斯优化中的梯度消失现象，并为RAASP和维度缩放长度尺度等启发式方法提供了统一的解释。基于这一分析，我们提出了FITR，一种基于信赖域的贝叶斯优化方法，它用局部拉回Fisher权重取代了基于长度尺度的缩放。FITR不局限于具有显式长度尺度的高斯过程核。在使用SE核的高斯过程基准测试中，实验表明FITR具有有竞争力的性能。所提出的方法也很容易推广到非各向同性的代理模型，尽管在这种情形下其收益更加依赖于具体任务。

    arXiv:2609.31107v1 Announce Type: cross  Abstract: We study Bayesian optimization (BO) through the lens of information geometry. Pulling back the Fisher information metric through the surrogate posterior map yields a local sensitivity tensor on the input space, which leads to an upper bound on the gradient of reparameterizable acquisition functions. This view explains vanishing-gradient behavior in high-dimensional BO and provides a common interpretation of heuristics such as RAASP and dimension-scaled lengthscales. Building on this analysis, we propose FITR, a trust-region-based BO method that replaces lengthscale-based scaling by local pullback-Fisher weights. FITR is not restricted to GP kernels with explicit lengthscales. On GP benchmarks with an SE kernel, experiments show competitive performance using FITR. The proposed method also easily generalizes to non-isotropic surrogates, although the gains are more task-dependent in that setting.
    
[^64]: 师生树委员会机器中的平坦性-泛化关系

    A Flatness-Generalization Relation in the Teacher-Student Tree-Committee Machine

    [https://arxiv.org/abs/2609.31101](https://arxiv.org/abs/2609.31101)

    该论文在师生树委员会机器中通过零温度吉布斯形式与Edwards-Jones形式解析刻画了经验损失典型极小值点的可观测量和Hessian谱，进而检验泛化误差随数据量增大而下降是否对应于损失景观平坦性的提升。

    

    摘要（arXiv:2609.31101v1）：损失景观在极小值点处的平坦性是用于推理神经网络泛化能力的一种广泛使用的启发式指标，然而关于这一关系的证据大多停留在经验层面且存在争议。我们在师生树委员会机器中研究这一关系，在该模型中，经验风险最小化（ERM）估计量和Hessian谱在成比例的高维极限下均可解析求解。首先，我们采用零温度吉布斯形式，对经验损失的典型极小值点的可观测量给出预测。其次，我们利用Edwards-Jones形式推导这些典型极小值点附近的极限Hessian预解式。所有预测均与有限规模的梯度下降模拟结果一致。最后，我们研究了三种平坦性度量，即谱的左边缘、右边缘和谱均值，并检验随数据集规模增大而带来的泛化误差下降是否对应于平坦性的提升。我们发现……

    arXiv:2609.31101v1 Announce Type: cross  Abstract: The flatness of the loss landscape at a minimizer is a widely used heuristic for reasoning about neural-network generalization, yet evidence for this relation is mostly empirical and controversial. We study this relation in a teacher-student tree committee machine, where both the ERM estimator and the Hessian spectrum are analytically tractable in the proportional high-dimensional limit. First, we use a zero-temperature Gibbs formulation to obtain predictions for the observables of the typical minimizers of the empirical loss. Secondly, we use Edwards-Jones formalism to derive the limiting Hessian resolvent around these typical minimizers. All predictions agree with finite-size gradient-descent simulations. Finally, we study three measures of flatness, namely the left and right edges and the spectral mean, and check if a decrease in generalization error as the dataset size is increased corresponds to an increase in flatness. We find th
    
[^65]: 残差流的有效深度

    The Residual Stream's Effective Depth

    [https://arxiv.org/abs/2609.31098](https://arxiv.org/abs/2609.31098)

    提出“有效深度”这一标量诊断指标，通过度量Transformer残差流中表示相似度随层间距离的衰减，将残差累积的理论上限与经验测量区分开来，并发现绝大多数模型的测量值低于正交更新参考值，该差距主要源于残差更新的相关性而非深度未被利用。

    

    我们引入了“有效深度”（effective depth, $\Deff$）这一标量诊断指标，它将Transformer的逐层残差流视为一个离散时间过程，度量表示相似度如何随层间距离衰减，并将该衰减曲线汇总为一个数值。在十六个仅解码器语言模型上，$\Deff$ 将残差累积的结构性后果与经验性后果区分开来：即使是最大化多样性的正交更新也存在闭式参考值 $F_L = 2L/(L+1)<2$，然而十六个默认测量中有十五个低于 $F_L$（Qwen3.5：32–44%，OLMo-2：40–41%，Pythia：23–28%）。匹配参考实验表明，这一差距并非由持久的初始状态或更新幅度不平衡所导致，而主要是相关残差更新的一个经校准的特征信号，而非深度未被利用的证据。对称的 position-0、词元归一化以及顶部主成分控制实验表明，该现象不能被归约为 BOS 或顶部主成分的伪影。

    arXiv:2609.31098v1 Announce Type: new  Abstract: We introduce \emph{effective depth} ($\Deff$), a scalar diagnostic that treats the layer-wise residual stream of a transformer as a discrete-time process, measures how representation similarity decays with layer distance, and aggregates that profile into one number. Across sixteen decoder-only language models, $\Deff$ separates a structural consequence of residual accumulation from an empirical one: even maximally diverse orthogonal updates have the closed-form reference $F_L = 2L/(L+1)<2$, yet fifteen of sixteen default measurements lie below $F_L$ (Qwen3.5: 32--44\%, OLMo-2: 40--41\%, Pythia: 23--28\%). Matched references show that the gap is not caused by the persistent initial state or update-size imbalance, but is largely a calibrated signature of correlated residual updates rather than evidence that depth is unused. Symmetric position-0, token-normalisation, and top-PC controls show the regime is not reducible to BOS or top-PC arte
    
[^66]: 具有对数线性复杂度的块稀疏注意力

    Block Sparse Attention with Log-Linear Complexity

    [https://arxiv.org/abs/2609.31093](https://arxiv.org/abs/2609.31093)

    本文提出PISA，一种采用金字塔Top-K选择策略的块稀疏注意力机制，通过从粗到细的多层级键筛选，将长序列注意力的复杂度从平方级降至对数线性级。

    

    将语言模型扩展到长上下文受到自注意力平方复杂度的限制。块稀疏注意力提供了一种高效的替代方案，但如何选择需要保留的块仍然是一个瓶颈。传统的块选择方法需要对所有查询-块对进行评分，因此在序列长度上仍然是平方复杂度。为了解决这个问题，我们提出了PISA，一种采用金字塔Top-K选择策略的块稀疏注意力机制。其核心思想是在不同层级上逐步缩小候选范围，从而更高效地找到最相关的键。具体而言，我们构建了一个从粗到细的键层级结构，并从最粗层级开始进行选择。在每一层级，对有界候选集应用LogSumExp评分，以选出进入下一更细层级的候选，直至达到最细层级。通过池化操作，我们构建了O(log N)层级的键，从而实现了整体的对数线性复杂度。

    arXiv:2609.31093v1 Announce Type: new  Abstract: Scaling language models to long contexts is limited by the quadratic cost of self-attention. Block sparse attention offers an efficient alternative, but selecting the retained blocks remains a bottleneck. Conventional block selection requires scoring all query-block pairs and therefore remains quadratic in sequence length. To address this issue, we propose PISA, a block-sparse attention mechanism that employs a pyramid Top-$K$ selection strategy. The main idea is to gradually narrow down the candidates across different levels, making it more efficient to find the most relevant keys. Specifically, we construct a coarse-to-fine hierarchy of keys and perform selection from the coarsest level. At each level, LogSumExp scoring is applied to a bounded candidate set to select candidates for the next finer level, continuing until the finest level is reached. Through pooling, we construct $O(\log N)$ levels of keys, yielding an overall complexity
    
[^67]: SAGE：面向物种分布建模的采样感知全局评估基准

    SAGE: A sampling-aware global evaluation benchmark for species distribution modeling

    [https://arxiv.org/abs/2609.31082](https://arxiv.org/abs/2609.31082)

    该论文提出了SAGE——一个采样感知的全局评估基准，通过结合GBIF训练记录与sPlotOpen植被调查数据，充分考虑采样偏差和物种层面（尤其是稀有物种）的性能差异，为深度学习多物种分布模型提供更可靠、更具信息量的评估。

    

    了解物种的分布位置是生物多样性研究与保护工作的基础。物种分布模型（SDMs）将物种观测记录与环境条件相关联，以估计物种的空间分布。然而，模型的准确性会随底层数据和模型的不同而变化，因此明确模型对哪些物种可信至关重要。基于深度学习的物种分布模型（"DeepSDMs"）如今可联合建模数千个物种，并利用了数以亿计的社区科学记录。在这种规模下，平均性能指标会掩盖显著的物种层面差异，尤其是对于往往最受保护关注的稀有物种。此外，观测记录存在强烈的偏差，使得物种出现次数具有误导性。考虑这些因素对于对多物种SDMs进行可靠且富有信息量的评估必不可少。在此，我们引入了一个采样感知全局评估基准，该基准将用于训练的GBIF记录与sPlotOpen植被……

    arXiv:2609.31082v1 Announce Type: cross  Abstract: Knowing where species occur is fundamental for biodiversity research and conservation. Species distribution models (SDMs) link species observations to environmental conditions to estimate their spatial distribution. However, accuracy varies with the underlying data and models, making it essential to know for which species models can be trusted. Deep-learning-based SDMs ("DeepSDMs") now jointly model thousands of species, drawing on hundreds of millions of community-science records. At this scale, averaging performance hides substantial species-level variability, particularly for rare species, often of greatest conservation concern. Records are also strongly biased, making occurrence counts misleading. Accounting for these factors is essential for a reliable and informative evaluation of multi-species SDMs. Here, we introduce a Sampling-Aware Global Evaluation (SAGE) benchmark, combining GBIF records for training with sPlotOpen vegetati
    
[^68]: 用于医学图像分析的量子扩散模型

    Quantum Diffusion Models for Medical Image Analysis

    [https://arxiv.org/abs/2609.31070](https://arxiv.org/abs/2609.31070)

    本文提出一种基于离散时间量子游走算法、结合经典逆向去噪模型的可扩展混合量子扩散模型，突破了现有量子设备规模的限制，能够处理真实世界的大尺寸医学图像数据。

    

    量子机器学习是一个新兴的研究领域，旨在利用量子力学的原理（如叠加、纠缠和干涉）来设计机器学习方法。在此背景下，我们提出了一种可扩展的混合量子扩散模型，并评估其在医学图像分析中的应用。具体而言，我们的方法基于在真实量子设备上执行的离散时间量子游走算法，用于对扩散模型的前向动力学进行建模。对于扩散模型的反向步骤，我们设计并评估了一个经典学习模型，用于对数据进行逆向去噪。与现有其他将量子机器学习应用于图像分析任务的尝试（这些尝试严重受限于现有量子设备的规模）不同，我们的方法能够处理真实世界的大尺寸医学数据。特别是，我们展示了在灰度图像、RGB图像以及中等规模三维体数据上的实验结果。

    arXiv:2609.31070v1 Announce Type: cross  Abstract: Quantum Machine Learning is a novel field of research aimed at devising machine learning approaches exploiting principles of quantum mechanics, such as superposition, entanglement and interference. In this context, we present a scalable hybrid Quantum Diffusion Model, and evaluate its use for medical image analysis. Specifically, our method is based on a Discrete-Time Quantum Walk algorithm, executed on a real quantum device, to model the forward dynamics of the diffusion model. For the backward step of the diffusion model, we devise and evaluate a classical learning model, which is used to reversely denoise the data. In contrast with other existing attempts at applying quantum machine learning for image analysis tasks, severely limited by the size of existing quantum devices, our method allows to process real-world large size medical data. In particular, we present results on grayscale and RGB images, as well as 3D volumes of moderate
    
[^69]: 分布式学习即服务：开发者的视角

    Distributed Learning as a Service: The Developer's Perspective

    [https://arxiv.org/abs/2609.31061](https://arxiv.org/abs/2609.31061)

    本文提出从开发者视角出发的DLaaS（分布式学习即服务）框架，开发者通过单一管理控制台即可用声明式选项启用差分隐私、拆分学习、分层聚合和知识蒸馏，无需修改客户端代码即可解决联邦学习中隐私泄露、设备资源受限、聚合器扩展性差和带宽成本高等挑战。

    

    分布式学习服务的应用开发者面临着典型联邦学习流程所无法解决的挑战。具体而言，模型更新仍可能泄露隐私数据，设备可能因资源有限而无法参与训练，单一聚合器可能无法进行扩展，且模型权重的传输会带来可观的带宽成本。本文从开发者的视角展示了DLaaS（分布式学习即服务）。通过一个统一的管理控制台，开发者即可启动分布式/联邦学习任务，并能够以声明式选项的方式启用差分隐私（DP）、拆分学习（SL）、分层聚合（HA）和知识蒸馏（KD），而无需修改客户端的任何代码。我们使用“Ok Aura”数据集，在工业级智能家居唤醒词任务上演示了完整的服务生命周期。一旦开发者启动了一（原文摘要在此处截断）

    arXiv:2609.31061v1 Announce Type: new  Abstract: Application developers of distributed learning services face challenges that a typical federated learning loop does not address. Specifically, the model updates can still leak private data, devices might not be able to participate in the training due to limited resources, a single aggregator might not be able to scale, and the transmissions of model weights induce a considerable bandwidth cost. This paper demonstrates DLaaS (Distributed Learning as a Service) from the developer's vantage point. Using a single admin dashboard, the developer initiates a distributed/federated learning job and is able to activate Differential Privacy (DP), Split Learning (SL), Hierarchical Aggregation (HA), and Knowledge Distillation (KD) as declarative options, with no change to the clients' code. We demonstrate the complete service lifecycle on an industrial smart-home Wake-up Word (WuW) task, using the "Ok Aura" dataset. Once the developer initiates a dis
    
[^70]: DynBranch：面向动态智能体大语言模型服务的投机性子图复用

    DynBranch: Speculative Subgraph Reuse for Dynamic Agentic LLM Serving

    [https://arxiv.org/abs/2609.31047](https://arxiv.org/abs/2609.31047)

    DynBranch 通过让未解析的分支在解析前即可被寻址，实现了投机性子图执行与跨请求的子图结果复用，从而打破“分支解析屏障”，将智能体 LLM 服务的平均延迟降低最高 32%。

    

    智能体式大语言模型（LLM）工作流在运行时决定其执行路径。下游计算可能可预测，或者可能已经运行过，但在模型或用户解析出分支之前无法开始。我们将这种串行化称为“分支解析屏障”。单纯的缓存无法掩盖这一障碍：标识可复用结果的键在分支解析之前是未知的。本文提出 DynBranch，使未解析的分支在解析之前即可被寻址。其稳定的坐标允许候选子图在分支解析期间运行，并允许已完成的子图结果在后续请求中被复用。一个两级控制器会在预期收益超过负载代价时接纳这些工作。DynBranch 位于模型 API 边界，无需对智能体框架或模型执行引擎进行任何更改。在使用 Qwen3-32B 和 4 块 H200 GPU 的四个智能体工作负载上，DynBranch 相比每个工作负载最强的先前系统将平均延迟降低至多 32%，并

    arXiv:2609.31047v1 Announce Type: cross  Abstract: Agentic LLM workflows decide their execution paths at runtime. Downstream computation may be predictable, or may have run before, yet it cannot begin until the model or the user resolves the branch. We call this serialization the branch-resolution barrier. Caching alone does not hide it: the key that identifies a reusable result is not known until then. In this paper, we propose DynBranch, which makes an unresolved branch addressable before it resolves. Its stable coordinate lets candidate subgraphs run during resolution and completed subgraph results be reused across later requests. A two-level controller admits this work when its expected benefit exceeds the load price. DynBranch sits at the model-API boundary and requires no changes to agent harnesses or model execution engines. Across four agentic workloads with Qwen3-32B on 4x H200 GPUs, DynBranch reduces mean latency by up to 32% over each workload's strongest prior system and by
    
[^71]: KuaFu：在十亿级规模上将长用户行为压缩为用户理解

    KuaFu: Compressing Long User Behavior into Understanding at Billion Scale

    [https://arxiv.org/abs/2609.31045](https://arxiv.org/abs/2609.31045)

    该论文提出KuaFu系统，在十亿用户规模上解决将长用户行为压缩为用户理解时序列过长与刷新吞吐量的双重工业瓶颈，并应对压缩过程可能引入的四类幻觉问题。

    

    对话式智能体、生成式推荐器和个性化广告都依赖一项核心能力：从原始行为中理解每一位用户。当前主流的工业实践是任务专用的：针对每个任务，从完整历史中提取相关子序列，并在其上训练专用模型。在生产环境中，这遇到了两个瓶颈。第一，即使经过过滤，单一任务的序列仍然极长：内容兴趣摘要需要读取每位用户的数百条内容，一旦序列化为提示文本就达数万个token。第二，用户画像需要例行刷新：每周覆盖十亿用户，总计约10万QPM，在固定的GPU预算下这设定了一个硬性的吞吐量下限。因此压缩是必不可少的，但截断或粗糙的压缩会悄然扭曲用户画像，引入四种幻觉类型（捏造、遗漏、日期错误归属、逻辑断裂），而由于无法评估压缩（原文在此处截断）……

    arXiv:2609.31045v1 Announce Type: cross  Abstract: Conversational agents, generative recommenders, and personalized advertising all rest on one capability: understanding each user from raw behavior. Prevailing industrial practice is task-specific: for each task, a relevant subsequence is extracted from the full history and a dedicated model trained on it. In production it hits two bottlenecks. First, even after filtering, a single-task sequence stays extremely long: content-interest summarization reads several hundred items per user, tens of thousands of tokens once serialized as prompt text. Second, profiles are refreshed routinely: a billion users weekly, roughly 100K QPM in aggregate, which under a fixed GPU budget sets a hard throughput floor. Compression is therefore mandatory, yet truncation or coarse compression can silently distort the profile, introducing four hallucination types (fabrication, omission, date misattribution, broken logic) that, with no way to evaluate the compr
    
[^72]: Aurora-X：为极端时间序列预测而打造

    Aurora-X: Built for Extreme Time Series Forecasting

    [https://arxiv.org/abs/2609.31038](https://arxiv.org/abs/2609.31038)

    Aurora-X 是一个十亿参数规模的时间序列基础模型，通过渐进式课程训练和可变分辨率后训练的统一架构设计，在固定权重下实现测试时扩展，并支持跨变量建模、协变量条件化与未来补丁并行解码。

    

    时间序列基础模型（TSFM）能够实现跨领域预测，但它们作为通用预测器的发展仍受到训练潜力挖掘不足和架构灵活性有限的制约。为应对这些挑战，我们提出了 Aurora-X，一个具有渐进式课程学习和统一架构的十亿参数规模时间序列基础模型。我们首先使用通道独立的预训练来学习时间模式，然后在中间训练阶段引入跨变量依赖、多样化的上下文与预测长度，以及（如果可用的话）未来协变量。可变分辨率的后训练进一步使推理时每个 token 能够覆盖可调节的时间跨度。在模型权重固定的情况下，这支持在固定 token 预算下使用更长的历史数据，或用更少的 token 表示相同的历史数据，从而实现测试时扩展。凭借灵活的架构，Aurora-X 支持跨变量建模、协变量条件化以及未来补丁的并行解码……

    arXiv:2609.31038v1 Announce Type: new  Abstract: Time series foundation models (TSFMs) enable cross-domain forecasting, but their development as general-purpose forecasters remains constrained by underexplored training potential and limited architectural versatility. To address these challenges, we introduce Aurora-X, a billion-scale TSFM with a progressive curriculum and a unified architecture. We first use channel-independent pretraining to learn temporal patterns, then introduce cross-variable dependencies, varied context and horizon lengths, and future covariates if available during midtraining. Variable-resolution post-training further enables an adjustable temporal span per token at inference. With fixed model weights, this supports longer histories under a fixed token budget or fewer tokens for the same history, enabling test-time scaling. With a versatile architecture, Aurora-X supports cross-variable modeling, covariate conditioning, and parallel decoding of future patches for
    
[^73]: 面向多重缺失数据的鲁棒图聚类网络

    Robust Graph Clustering Network for Multiple Missing Data

    [https://arxiv.org/abs/2609.31033](https://arxiv.org/abs/2609.31033)

    提出了一种面向多重缺失数据的鲁棒图聚类网络RGCN，通过视图解耦双分支插补、多超球面混合先验和边界感知对比增强三项创新，有效解决了节点属性与图结构同时缺失情况下的图聚类问题。

    

    在节点属性和结构链接均部分缺失的图上进行聚类仍然是一项具有挑战性的任务。现有方法通常依赖于在单视图缺失的不完整图上进行“先插补后聚类”的策略，这种做法在属性和结构同时缺失的情况下，容易出现跨视图误差传播和聚类边界模糊的问题。为了解决这些局限性，我们提出了一种面向多重缺失数据的鲁棒图聚类网络，旨在处理节点属性和图结构同时不完整的情况。RGCN 引入了三项关键创新：首先，我们设计了一种视图解耦的双分支插补方法，以减轻干扰并在恢复缺失数据时实现相互增强。其次，我们采用多超球面混合先验，在方向性潜在流形上增强簇内紧凑性和簇间可分离性。第三，一种边界感知的对比增强机制……

    arXiv:2609.31033v1 Announce Type: new  Abstract: Clustering on graphs where both node attributes and structural links are partially missing remains a challenging task. Existing methods typically rely on imputation-then-clustering on single-view missingness incomplete graphs, which are vulnerable to cross-view error propagation and cluster-boundary blurring under simultaneous attribute and structure missingness. To address these limitations, we propose a Robust Graph Clustering Network for Multiple Missing Data (RGCN), which is designed to handle simultaneous node attribute and graph structure incompleteness. RGCN introduces three key innovations: First, we design a view-decoupled dual-branch imputation to mitigate interference and enable mutual enhancement in recovering missing data. Second, we employ a multi-hyperspherical mixture prior to enhance intra-cluster compactness and inter-cluster separability on a directional latent manifold. Third, a boundary-aware contrastive enhancement 
    
[^74]: 面向移动系统的元认知选择性集成

    Metacognitive Selective Ensemble for Mobile Systems

    [https://arxiv.org/abs/2609.31031](https://arxiv.org/abs/2609.31031)

    MetaSE是一种面向移动系统的主动集成框架，通过利用单个模型可靠性的短期持续性来维护小型活跃模型集合，以更低的计算成本实现了与全集成推理相当的准确率，并在树莓派上快2.7倍。

    

    深度集成提高了移动感知的鲁棒性，但在连续传感器流上重复执行多个模型的代价高昂。仅选择少数成员可以降低这一成本，然而自适应选择通常需要额外的模型执行来获取关于非活跃候选模型的可靠证据。我们提出了MetaSE，一个利用单个模型可靠性短期持续性的主动集成框架。MetaSE跨时间窗口维护一个较小的活跃模型集合，利用执行后的证据剔除不可靠的成员，并且仅在需要替换时才调用轻量级路由机制。这种有状态的设计无需重复进行全池评估，即可利用更大模型池的多样性。在四个人体活动识别（HAR）数据集和四种模型架构上，MetaSE始终优于固定的三模型集成，并取得了与成本高得多的自适应推理和全集成推理相当的准确率。在树莓派4B上，MetaSE的运行速度提升了2.7倍。

    arXiv:2609.31031v1 Announce Type: new  Abstract: Deep ensembles improve robustness in mobile sensing, but repeatedly executing many models over continuous sensor streams is costly. Selecting only a few members reduces this cost, yet adaptive selection often requires additional model execution to obtain reliable evidence about inactive candidates. We present MetaSE, an active ensemble framework that exploits short-term persistence in per-model reliability. MetaSE maintains a small active set across windows, uses post-execution evidence to reject unreliable members, and invokes lightweight routing only when replacement is needed. This stateful design accesses the diversity of a larger pool without repeated full-pool evaluation. Across four HAR datasets and four model architectures, MetaSE consistently improves over a fixed three-model ensemble and achieves accuracy comparable to substantially more expensive adaptive and full-ensemble inference. On a Raspberry Pi 4B, MetaSE is 2.7x faster
    
[^75]: 速度中的精度：面向液压挖掘机控制的高样本效率在线基于模型强化学习

    Precision at Speed: Sample-Efficient Online Model-Based Reinforcement Learning for Hydraulic Excavator Control

    [https://arxiv.org/abs/2609.31025](https://arxiv.org/abs/2609.31025)

    该论文提出一种在线基于模型的强化学习框架，通过从零学习概率动力学集成模型并采用精度门控的轮廓跟踪目标，使11.5吨液压挖掘机仅需20分钟真实交互就达到了以往需要100-150分钟数据训练的控制器相当的跟踪精度。

    

    对于具有复杂执行器动力学的机器人，实现精确的高速控制仍然具有挑战性。而直接在硬件上学习又进一步受到真实世界交互成本的制约。我们提出了一种在线基于模型的强化学习框架，该框架从零开始学习一个概率动力学集成模型，用于基于采样的模型预测控制。其中，精度门控的轮廓跟踪目标将进度奖励与路径精度条件关联，使精度优先于速度。在数据驱动的挖掘机仿真器中，该框架相比所评估的基于模型强化学习基线实现了更高的样本效率。我们通过直接在11.5吨的Menzi Muck M445液压挖掘机上学习来验证该框架，全程无需任何演示或仿真预训练。经过20分钟的交互，该控制器即达到了与先前使用100-150分钟数据训练的学习型控制器相当的跟踪精度。经过40分钟

    arXiv:2609.31025v1 Announce Type: cross  Abstract: Precise, high-speed control remains challenging for robots with complex actuation dynamics. Learning directly on hardware is further constrained by the cost of real-world interaction. We present an online model-based reinforcement learning framework that learns a probabilistic dynamics ensemble model from scratch for sampling-based model predictive control. A precision-gated contouring objective conditions the progress reward on path accuracy, prioritizing precision over speed. In a data-driven excavator simulator, the framework achieves higher sample efficiency than the evaluated model-based reinforcement learning baselines. We validate the framework by learning directly on an 11.5-ton Menzi Muck M445 hydraulic excavator, without demonstrations or simulation pretraining. After 20 minutes of interaction, the controller reaches tracking accuracy comparable to prior learned controllers trained on 100-150 minutes of data. After 40 minutes
    
[^76]: Synth-JEPA：面向无需渲染器的合成器参数搜索的联合嵌入预测方法

    Synth-JEPA: Joint Embedding Prediction for Renderer-Free Synthesizer Parameter Search

    [https://arxiv.org/abs/2609.31024](https://arxiv.org/abs/2609.31024)

    Synth-JEPA通过从成对的合成器数据中学习相互可预测的音频与参数联合嵌入表示，实现了无需渲染音频即可直接评分候选参数的目标函数，在域内和域外的声音匹配任务上均优于或媲美现有基线。

    

    声音匹配可以被表述为针对音频域目标函数优化合成器参数的问题。然而，基于通用音频表示所导出的目标函数通常难以优化，而直接搜索则需要对每个候选参数进行渲染。我们提出了Synth-JEPA，它从成对的合成器数据中学习相互可预测的音频表示与参数表示。在推理阶段，候选参数直接在这一学习到的空间中被评分，从而得到一个无需渲染器的目标函数，其音频几何结构由参数对应关系而非通用音频相似性所塑造。我们在Surge XT合成器上，使用留出的合成器音色以及域外的NSynth和FSD50K目标对Synth-JEPA进行评估，并与逆模型、直接搜索以及学习型代理目标等基线方法进行比较。Synth-JEPA在域内优于所有基线方法，并在域外保持竞争力。其匹配质量随着测试时搜索量的增加而持续提升，使得……

    arXiv:2609.31024v1 Announce Type: cross  Abstract: Sound matching can be formulated as optimizing synthesizer parameters against an audio-domain objective. However, objectives derived from generic audio representations are often difficult to optimize, while direct search requires rendering every candidate. We introduce Synth-JEPA, which learns mutually predictive audio and parameter representations from paired synthesizer data. At inference, candidate parameters are scored directly in this learned space, yielding a renderer-free objective whose audio geometry is shaped by parameter correspondences rather than generic audio similarity. We evaluate Synth-JEPA on Surge XT using held-out synthesizer sounds and out-of-domain NSynth and FSD50K targets, against inverse models, direct search, and learned proxy objectives. Synth-JEPA outperforms all baselines in-domain and remains competitive out-of-domain. Its matching quality continues to improve with additional test-time search, allowing com
    
[^77]: 鲁棒后继特征

    Robust Successor Features

    [https://arxiv.org/abs/2609.31016](https://arxiv.org/abs/2609.31016)

    该论文提出了鲁棒后继特征，将强化学习中的迁移学习与鲁棒强化学习两种范式统一起来，使智能体在线性马尔可夫决策过程假设下，能够同时在奖励函数和未知转移核两个维度上进行泛化。

    

    摘要：强化学习（RL）中的泛化是指智能体在一系列不同任务上训练后，能够在未见过的任务上执行接近最优策略的能力。基于后继表示的开创性工作以及结合函数逼近的进一步改进，强化学习中的迁移学习传统上一直专注于泛化到仅在奖励函数上有所不同的任务。在后继表示提出十年之后，鲁棒强化学习同时从运筹学领域的数篇文章中兴起。在鲁棒强化学习中，转移核是未知的，目标是在这种不确定性下最大化期望奖励。我们的工作通过鲁棒后继特征统一了这两种范式，在任务为线性马尔可夫决策过程的假设下，鲁棒后继特征能够在奖励函数和转移核两个维度上实现泛化。我们推导了广义策略改进的一个界……

    arXiv:2609.31016v1 Announce Type: new  Abstract: Generalization in Reinforcement Learning (RL) refers to the ability to execute close-to-optimal policies in unseen tasks after the agent has been trained on a different set of tasks. Building on the seminal work of the successor representation and further adaptations with function approximation, Transfer in RL has traditionally focused on generalizing to tasks that only differ in the reward function. A decade after the introduction of the successor representation, Robust RL emerged simultaneously from several articles in the field of operations research. In Robust RL, the transition kernel is unknown, and the goal is to maximize the expected reward under this uncertainty. Our work unifies these two paradigms through robust successor features, which generalize across both the reward function and the transition kernel, under the assumption that tasks are linear Markov Decision Processes. We derive a bound on Generalized Policy Improvement 
    
[^78]: 仅凭像素能否揭示图像来源？被动溯源的极小极大极限与可学习接口

    Can Pixels Alone Reveal Image Origin? Minimax Limits and Learnable Interfaces for Passive Provenance

    [https://arxiv.org/abs/2609.30997](https://arxiv.org/abs/2609.30997)

    该论文为仅凭像素的图像溯源建立了精确理论极限——由目标分布与受攻击源分布间的最小全变差距离决定且与验证器架构无关，并揭示了公开验证器因可被模拟而会在远早于该统计极限之前被代理黑盒攻击攻破。

    

    被动图像溯源探讨的是仅凭像素能否揭示图像的来源：是来自人类、某类AI模型的整体，还是某个特定的生成器。一旦源图像在验证者看到之前可能被编辑，这就变成了一个鲁棒性问题。我们将该问题形式化为对抗性分布偏移下的源-目标验证问题。我们的第一个结果给出了任何仅基于图像的验证器的精确最优极限：最大的鲁棒目标接受差距等于目标分布与受攻击源分布集合之间的最小全变差距离。该量仅取决于源分布、目标分布和编辑类别，而与验证器的架构无关。我们的第二个结果解释了为什么已部署的公开验证器会在达到这一统计极限之前就失效：如果验证器可以在攻击区域上以误差ε被模拟，那么代理黑盒攻击就能在2ε加上优化误差的范围内达到目标接受。

    arXiv:2609.30997v1 Announce Type: cross  Abstract: Passive image provenance asks whether pixels alone can reveal where an image came from: a human, an aggregate AI class, or a particular generator. This becomes a robustness problem once a source image can be edited before the verifier sees it. We study the problem as source--target verification under adversarial distribution shift. Our first result gives the exact best-case limit for any image-only verifier: the largest robust target-acceptance gap equals the minimum total-variation distance between the target distribution and the set of attacked source distributions. This quantity depends on the source, target, and edit class, not on the verifier architecture. Our second result explains why deployed public verifiers can fail before this statistical limit is reached. If the verifier can be emulated on the attack region to error $\varepsilon$, then a surrogate black-box attack reaches target acceptance within $2\varepsilon$ plus optimiz
    
[^79]: 视觉-语言-动作模型的线性表示假设

    The Linear Representation Hypothesis for Vision-Language-Action Models

    [https://arxiv.org/abs/2609.30996](https://arxiv.org/abs/2609.30996)

    该论文提出了一种基于签名的理论框架，将线性表示假设从大语言模型扩展到视觉-语言-动作模型，统一了表示与策略，以应对具身交互中感兴趣的物理量与系统动力学共同演化这一挑战。

    

    线性表示假设（LRH）已成为通过大语言模型（LLM）内部表示来度量和干预语义信息的标准视角。越来越多的工作开始将这一视角扩展到视觉-语言-动作（VLA）模型，但具身交互的动态特性带来了额外的挑战。与LLM中通常研究的语义属性（如性别或语言）不同，VLA中感兴趣的物理量（QoI）会与系统动力学共同演化：表示会影响策略所选的动作，动作改变物理状态，进而又影响下一步的表示。在本文中，我们为VLA开发了一种基于签名的LRH理论表述，将表示与策略统一起来。在表示方面，我们证明了存在这样的表示，由其可以预测感兴趣的物理量在候选（策略）下的未来演化（摘要此处截断）。

    arXiv:2609.30996v1 Announce Type: cross  Abstract: The linear representation hypothesis (LRH) has become a standard lens for measuring and intervening on semantic information through the internal representations of large language models (LLMs). A growing body of work has begun extending this perspective to vision-language-action (VLA) models, but the dynamical nature of embodied interaction introduces an additional challenge. Unlike semantic attributes commonly studied in LLMs, such as gender or language, a physical quantity of interest (QoI) in a VLA evolves jointly with the system dynamics: the representation influences the actions selected by the policy, which alter the physical state and, in turn, the next representation.   In this paper, we develop a theoretical, signature-based formulation of the LRH for VLA that unifies representations and policies. On the representation side, we establish the existence of representations from which the future evolution of a QoI under a candidat
    
[^80]: 学习气候模式中强迫对温度影响的分层因果表示

    Learning Hierarchical Causal Representations of the Effects of Forcings on Temperature in Climate Models

    [https://arxiv.org/abs/2609.30995](https://arxiv.org/abs/2609.30995)

    该论文提出一种分层因果表示学习框架，能够显式区分内部气候变率与外部强迫响应，从而准确预测气候变化情景下的温度演变，并提升机器学习气候模拟器的可信度与因果归因能力。

    

    机器学习（ML）模拟器在地球系统模式预估数据上训练后，为模拟气候变化情景提供了一种快速且经济高效的方法。然而，这些数据驱动方法的黑箱特性限制了其输出的可用性和可信度，尤其限制了其作为因果归因工具的使用。在此，我们开发了一个分层因果表示学习框架，并将其应用于最先进的全球气候模式的海表温度场。作为对以往工作的关键进展，我们的框架显式地建模了由内部气候变率引起的大气动力相互作用，以及由大气温室气体和气溶胶浓度变化引起的强迫响应。在未来气候变化情景上进行训练后，我们的方法能够准确预测长期全球平均和区域温度演变，并对温室气体扰动表现出物理上真实的响应。

    arXiv:2609.30995v1 Announce Type: new  Abstract: Machine learning (ML) emulators provide a fast and cost-effective method to simulate climate change scenarios after being trained on Earth System Models projections. However, the black-box nature of those data-driven approaches limit the usability and trustworthiness of their outputs and in particular their use as causal attribution tools. Here, we develop a hierarchical causal representation learning framework applied to sea surface temperature fields from a state-of-the-art global climate model. As a key advance over previous work, our framework explicitly models both atmospheric dynamical interactions arising from internal climate variability and forced responses due to changes in atmospheric greenhouse gas and aerosol concentrations. When trained on future climate change scenarios, our method accurately predicts the long-term global mean and regional temperature evolution and shows physically realistic responses to perturbations in g
    
[^81]: 均匀离散扩散模型需要时间吗？

    Does Uniform Discrete Diffusion Need Time?

    [https://arxiv.org/abs/2609.30977](https://arxiv.org/abs/2609.30977)

    本研究从理论和实证两方面证明，在语言建模等有限数据场景下，均匀离散扩散模型的时间条件化在很大程度上是不必要的，与时间无关的预测器能达到甚至超越时间条件化模型的性能。

    

    均匀离散扩散模型（UDMs）通常使用显式的时间条件化机制，但我们发现这在实践中往往是不必要的。本文首先证明，在总体最优意义上，UDM的预测器通常依赖于时间：时间控制着模型应该在多大程度上信任观测到的上下文。随后我们证明，在与语言相关的有限数据设置中，这种依赖性可以变得微不足道。当一个被破坏的训练序列与其原始干净序列的距离仍远小于其与其他竞争训练序列的距离时，经验最优预测器在扩散轨迹的大部分范围内对时间几乎不敏感，尽管这一保证在高噪声端点附近会减弱。实证结果表明，训练好的语言UDM在轨迹的大部分范围内表现出有限的时间敏感性，而与时间无关的预测器在各种数据集和训练目标上与时间条件化模型相比仍具竞争力，且往往表现更优。

    arXiv:2609.30977v1 Announce Type: cross  Abstract: Uniform discrete diffusion models (UDMs) commonly use explicit time conditioning, but we find that it can often be unnecessary in practice. In this paper, we first show that the population-optimal UDM predictor generally depends on time: time controls how much the model should trust the observed context. We then show that this dependence can become negligible in finite-data settings relevant to language. When a corrupted training sequence remains much closer to its original clean sequence than to competing training sequences, the empirical-optimal predictor is nearly insensitive to time over most of the diffusion trajectory, where the guarantee weakens toward the high-noise endpoint. Empirically, trained language UDMs exhibit limited time sensitivity over most of the trajectory, while time-agnostic predictors remain competitive with, and often outperform, time-conditioned models across datasets and training objectives. These results ch
    
[^82]: 语音合成内容表征的综合研究

    A Comprehensive Study of Content Representations for Speech Synthesis

    [https://arxiv.org/abs/2609.30975](https://arxiv.org/abs/2609.30975)

    该研究在统一生成框架下系统比较了各类语音内容表征，发现说话人身份解耦并非仅由监督信号决定，而是由训练目标与表征信息容量之间的相互作用所决定。

    

    语音内容表征是语音转换、语音到语音翻译和多模态语言模型的核心，但它们很少在统一的生成框架下被比较，以直接测量每种表征所包含的信息。为此，我们仅以每种表征为条件训练生成模型，并从内容、说话人身份和韵律三个维度评估生成的音频。在自监督学习（SSL）特征、有监督token、后验图和神经音频编解码器等各类表征中，我们发现了两种截然不同的模式：一类表征几乎可以重建原始音频，另一类表征则能有效解耦说话人身份。这些结果表明，解耦能力不仅取决于监督信号，还取决于训练目标与表征信息容量之间的相互作用：只有当容量受到充分限制时，有监督表征才能实现说话人身份的解耦。

    arXiv:2609.30975v1 Announce Type: cross  Abstract: Speech content representations are central to voice conversion, speech-to-speech translation, and multimodal language models, yet they are rarely compared under a common generative framework that directly measures what each representation contains. We address this by training a generative model conditioned solely on each representation and evaluating the generated audio along the content, speaker identity, and prosody axes. Across SSL features, supervised tokens, posteriorgrams, and neural audio codecs, we find two distinct regimes: representations that nearly reconstruct the original audio, and representations that effectively disentangle speaker identity. These results show that disentanglement depends not on supervision alone, but on the interaction between the training objective and the representation's information capacity: supervised representations only disentangle speaker identity when their capacity is sufficiently constrained
    
[^83]: LipSSM：通过连续SSM层间的度量迁移实现结构上Lipschitz有界的级联状态空间模型

    LipSSM: Structurally Lipschitz-Bounded Cascaded State-Space Model via Metric Transfer between Consecutive SSM Layers

    [https://arxiv.org/abs/2609.30973](https://arxiv.org/abs/2609.30973)

    本文提出LipSSM，通过在相邻状态空间模型（SSM）层之间进行度量迁移，将LipKernel的跨层信息传递思想扩展到级联状态空间模型，相比传统逐层构造方法获得更紧凑的整体Lipschitz界，从而在保证鲁棒性的同时提升模型的表达能力。

    

    Lipschitz连续性是设计可证明鲁棒的深度神经网络（DNN）的基本原则，其中调整量化网络鲁棒性的Lipschitz常数具有重要的理论意义。强制Lipschitz连续性的标准方法要求DNN的每一层都满足Lipschitz连续性，从而保证整体的Lipschitz连续性。然而，这种逐层方法通常会导致对整体Lipschitz常数的宽松估计，从而施加过于保守的限制，这限制了DNN的表达能力，并在规定的鲁棒性水平下降低了经验性能。为了克服这种宽松估计问题，最近提出的LipKernel通过跨层传递信息，得到了比传统逐层构造更紧凑的整体Lipschitz界。在本文中，我们将这一概念扩展到级联状态空间模型（SSM），以构造Lipschitz有界的结构（摘要在此处截断）。

    arXiv:2609.30973v1 Announce Type: new  Abstract: Lipschitz continuity is a fundamental principle in the design of certifiably robust deep neural networks (DNNs), wherein adjusting the Lipschitz constant, which quantifies network robustness, is of central theoretical importance. A standard approach to enforcing Lipschitz continuity requires each layer of a DNN to be Lipschitz continuous, thereby guaranteeing overall Lipschitz continuity. However, this layer-wise approach typically imposes overly conservative restrictions by producing a loose estimate of the overall Lipschitz constant, which limits the expressive capacity of the DNN and degrades empirical performance at a prescribed level of robustness. To overcome this loose estimation, the recently proposed LipKernel transfers information across layers to yield a much tighter overall Lipschitz bound than conventional layer-wise construction. In this paper, we extend this concept to cascaded state-space models (SSMs) to construct Lipsch
    
[^84]: 物理信息神经网络的梯度手术方法

    Gradient Surgery for Physics-Informed Neural Networks

    [https://arxiv.org/abs/2609.30966](https://arxiv.org/abs/2609.30966)

    该论文提出了一种物理感知的梯度手术方法PAM-GS，通过揭示PINN训练中角度型与幅值型梯度冲突三阶段交替出现的规律，自适应地缓解任务间干扰，从而提升收敛速度和训练稳定性。

    

    物理信息神经网络（PINNs）通过优化一个将数据拟合与基于物理的约束相结合的复合目标来进行训练，这通常导致一个高度不平衡的多任务优化问题。在这种条件下，现有的优化策略会受到相互冲突的任务梯度的影响，导致收敛缓慢和训练不稳定，尤其是对于刚性和高频偏微分方程。我们分析了在标准优化器下PINNs训练全过程中的梯度冲突，并研究了多任务深度学习（MTDL）优化方法。在对四个基准问题的分析中，我们观察到PINN优化表现出三个不同的阶段，其中基于角度的梯度冲突和基于幅值的梯度冲突交替出现，且每次只有一种存在。基于这些观察，我们提出了PAM-GS，一种物理感知的梯度手术方法，能够自适应地缓解任务间干扰（摘要在此处被截断）。

    arXiv:2609.30966v1 Announce Type: new  Abstract: Physics-Informed Neural Networks (PINNs) are trained by optimising a composite objective that combines data fitting with physics-based constraints, typically resulting in a highly imbalanced multi-task optimisation problem. Under these conditions, existing optimisation strategies are affected by conflicting task gradients, leading to slow convergence and unstable training, particularly for stiff and high-frequency partial differential equations. We analyse gradient conflicts throughout training of PINNs with standard optimiser and investigate Multi-Task Deep Learning (MTDL) optimisation methods. In our analysis across four benchmark problems we observed that PINN optimisation exhibits three distinct phases in which angle- and magnitude-based gradient conflicts alternate, with only one present at a time. Building on these observations, we propose PAM-GS, a physics-aware gradient surgery method that adaptively mitigates task interference d
    
[^85]: 理解河谷型损失景观中的动量加速机制

    Towards Understanding Momentum Acceleration in River-Valley Loss Landscape

    [https://arxiv.org/abs/2609.30957](https://arxiv.org/abs/2609.30957)

    该论文基于“河谷”损失景观结构解释了动量加速的原理：大学习率使梯度下降沿低损失“河流”快速前进，随后的学习率衰减抑制垂直振荡并揭示真实优化进展，从而解释了WSD学习率调度器成功的原因。

    

    预训练大型语言模型的实证成功激发了对潜在损失景观和优化动力学的更深入研究。近期的实证与理论研究表明，训练损失景观通常呈现出“河谷”结构，其特征是一条低损失流形（河流），两侧是损失更高且变化陡峭的正交方向（山脉）。从长期来看，优化进展主要由沿河流方向的进展决定。在这种景观中，采用大学习率的梯度下降能够沿着河流更快地移动，尽管由于垂直方向的振荡导致表观损失较高；而随后学习率的急剧衰减会抑制这些振荡，从而揭示出真正的优化进展。这解释了近期预热-稳定-衰减（WSD）学习率调度器取得成功的原因——与余弦调度不同，它保持稳定的高学习率并在阶段后期进行衰减（原文摘要在此处截断）。

    arXiv:2609.30957v1 Announce Type: new  Abstract: The empirical success of pretraining large language models has inspired a deeper investigation into the underlying loss landscapes and the optimization dynamics. Recent empirical and theoretical study suggest that the training loss landscape often exhibits a "river-valley" structure, which features a low-loss manifold (river) flanked by sharp orthogonal directions with higher loss (mountains). In the long term, the optimization progress is determined primarily by the progress along the river. Within such a landscape, gradient descent with large learning rates can move faster along the river despite high apparent loss due to vertical oscillations, while a subsequent sharp decay in the learning rate suppresses these oscillations, revealing genuine optimization progress. This explains the recent success of warmup-stable-decay (WSD) learning rate scheduler which, unlike cosine scheduling, keeps stable high learning rate and decays before pro
    
[^86]: 混合语言模型中的低位宽循环状态

    Low-Bit Recurrent States in Hybrid Language Models

    [https://arxiv.org/abs/2609.30950](https://arxiv.org/abs/2609.30950)

    该论文提出一种无需校准数据、旋转或训练的混合精度量化方法，利用可观测性格拉姆矩阵导出失真权重并结合归一化状态范围为混合语言模型的循环状态分配位宽，使四位平均有效载荷将超额负对数似然降低3.3至27.9倍。

    

    混合语言模型维护固定大小的循环状态，但现有量化器通常使用八位或更多位。量化误差会根据通道衰减率持续存在。我们从可观测性格拉姆矩阵导出失真权重，并将其与归一化的状态范围相结合，用于混合精度位分配，无需校准数据、旋转或训练。我们还对衰减率进行对数量化。在逐词元状态量化下，四位平均有效载荷相对于三个混合模型上七个基线中的最优者，将超额负对数似然降低了3.3至27.9倍；元数据成本各不相同。在六位时，负对数似然与FP32状态基线的差异小于0.005 nats。消融实验分离了可变位宽、衰减加权和范围归一化各自带来的收益。在较低频率的回写下，收益会减少，且取决于模型和预算。

    arXiv:2609.30950v1 Announce Type: new  Abstract: Hybrid language models maintain fixed-size recurrent states, but existing quantizers typically use eight bits or more. Quantization errors persist according to channel decay rates. We derive distortion weights from the observability Gramian and combine them with normalized state ranges for mixed-precision bit allocation, without calibration data, rotation, or training. We also quantize decay rates logarithmically. With per-token state quantization, a four-bit mean payload reduces excess negative log-likelihood by factors of 3.3--27.9 relative to the best of seven baselines across three hybrid models; metadata costs vary. At six bits, negative log-likelihood differs from the FP32-state baseline by less than 0.005 nats. Ablations separate gains from variable bit widths, decay weighting, and range normalization. With less frequent write-backs, gains diminish and depend on the model and budget.
    
[^87]: PORL：面向作业车间调度问题的预训练离线强化学习

    PORL: Pretrained Offline Reinforcement Learning for the Job Shop Scheduling Problem

    [https://arxiv.org/abs/2609.30948](https://arxiv.org/abs/2609.30948)

    该论文提出PORL方法，将基于仿真的在线预训练与离线微调相结合，并利用KL散度策略约束将通用调度策略适配到特定生产数据分布，从而克服作业车间调度问题中仿真到现实的差距。

    

    arXiv:2609.30948v1 公告类型：cross 摘要：作业车间调度问题是工业优化中的一个基础性组合优化问题。本工作提出了预训练离线强化学习，这是一种将基于仿真的在线预训练与针对特定生产数据的离线微调相结合的混合方法。通过在线交互进行强化学习能够探索通用的调度策略，但通常依赖于仿真环境，且可能受到仿真与现实差距的影响。相比之下，离线强化学习通过从历史数据中学习来避免与环境的直接交互，但其性能在很大程度上受数据集质量和覆盖范围的影响。PORL 结合了两种范式的优势：首先通过在线交互学习一个通用调度策略，随后将其离线适配到目标分布。文中引入了基于 KL 散度的策略约束，以限制偏离（原文摘要至此处被截断）

    arXiv:2609.30948v1 Announce Type: cross  Abstract: The Job Shop Scheduling Problem (JSSP) is a fundamental combinatorial optimization problem in industrial optimization. This work introduces Pretrained Offline Reinforcement Learning (PORL), a hybrid approach that combines simulation-based online pretraining with offline fine-tuning on production-specific data.   Reinforcement learning through online interaction enables exploration of general scheduling strategies, but typically relies on simulation environments and may suffer from a simulation-to-reality gap. In contrast, offline RL avoids direct interaction with the environment by learning from historical data, but its performance is strongly influenced by dataset quality and coverage. PORL combines the strengths of both paradigms by first learning a general scheduling policy through online interaction and subsequently adapting it offline to a target distribution. A KL-divergence-based policy constraint is introduced to limit deviatio
    
[^88]: 估计并正交化未知的预训练梯度以实现大语言模型的持续微调

    Estimating and Orthogonalizing Unknown Pre-training Gradients for Continual Fine-tuning of Large Language Models

    [https://arxiv.org/abs/2609.30935](https://arxiv.org/abs/2609.30935)

    提出EoupCT框架，通过动态生成最易受遗忘影响的伪数据来估计未知的预训练梯度，并将其与新任务的梯度进行正交化投影，从而在持续微调大语言模型时有效保护其固有的通用知识，避免灾难性遗忘。

    

    持续微调对于大语言模型动态适应现实世界环境至关重要，然而它不可避免地遭受灾难性遗忘的困扰，尤其是先前任务性能的下降以及大模型通用知识的退化。尽管现有方法（如正交梯度投影）能够缓解各种微调任务中的遗忘问题，但由于现成预训练大模型所需的原始数据和梯度严格未知且高度多样化，这些方法从根本上无法保留预训练大模型固有的通用知识。为弥合这一关键差距，我们提出了EoupCT，一个旨在估计并正交化未知预训练梯度以实现大语言模型持续微调的新型框架。具体而言，EoupCT通过动态生成对新任务最易受遗忘影响的伪数据来估计预训练梯度……（原文摘要在此处截断）

    arXiv:2609.30935v1 Announce Type: cross  Abstract: Continual fine-tuning is essential for large language models (LLMs) to dynamically adapt to real-world environments, yet it inevitably suffers from catastrophic forgetting, particularly the performance degradation of previous tasks and LLMs' general-purpose knowledge. Although existing methods, such as orthogonal gradient projection, mitigate the forgetting across various fine-tuning tasks, they fundamentally fail to preserve pre-training LLMs' inherent general-purpose knowledge because the original data and gradients of off-the-shelf pre-training LLMs required by these methods are strictly unknown and highly diverse. To bridge this critical gap, we propose EoupCT, a novel framework designed to Estimate and Orthogonalize Unknown Pre-training gradients for Continual LLM fine-Tuning. Specifically, EoupCT estimates pre-training gradients by dynamically generating pseudo data that is most susceptible to forgetting for new tasks through a l
    
[^89]: EPOC：面向多步长时间序列预测的端点保持在线校正与压缩残差状态方法

    EPOC: Endpoint-Preserving Online Correction With Compressed Residual State for Multi-Horizon Time Series Forecasting

    [https://arxiv.org/abs/2609.30929](https://arxiv.org/abs/2609.30929)

    EPOC提出一种端点保持的在线校正方法，仅存储残差块的低阶DCT系数与末端值以实现压缩残差状态，通过分量级在线岭回归与基础预测融合，在多步长时间序列预测中将MSE平均降低15.40%，同时大幅减少辅助存储开销。

    

    已完成的多步长预测可为固定的预测器提供残差反馈，但保留完整的残差块会增加辅助状态的开销。我们提出了具有压缩残差状态的端点保持在线校正方法EPOC。该方法仅存储前一残差块的低阶离散余弦变换（DCT）系数及其末端值。在每个通道内，该端点在分量级在线岭回归之间共享，同时这些回归还利用当前预测的系数。拟合得到的DCT校正量与基础预测相融合。我们在八个多变量时间序列上，使用DLinear和PatchTST、三个随机种子以及两种训练变体进行评估，在24步预测时域下共产生96个匹配的固定基准条件。EPOC相对未校正的基础模型，在均方误差（MSE）和平均绝对误差（MAE）上分别实现了平均15.40%和9.35%的条件级降低，且保留的辅助数组中位数仅为6,352字节。它具有更低的……

    arXiv:2609.30929v1 Announce Type: new  Abstract: Completed multi-horizon forecasts provide residual feedback for a fixed forecaster, but retaining full residual blocks increases auxiliary state. We propose Endpoint-Preserving Online Correction (EPOC) with a compressed residual state. It stores low-order discrete cosine transform (DCT) coefficients and the final value of the preceding residual block. Within each channel, the endpoint is shared across component-wise online ridge regressions that also use current-forecast coefficients. The fitted DCT correction is blended with the base forecast. We evaluate eight multivariate series with DLinear and PatchTST, three seeds, and two training variants, yielding 96 matched fixed-base conditions at a 24-step horizon. EPOC achieves mean condition-wise reductions in mean squared error (MSE) and mean absolute error (MAE) of 15.40% and 9.35% from the uncorrected base, respectively, with a median of 6,352 B in retained auxiliary arrays. It has lower
    
[^90]: 对哪种模型变化具有鲁棒性？鲁棒反事实解释的统一评估

    Robust to Which Model Change? A Unified Evaluation of Robust Counterfactual Explanations

    [https://arxiv.org/abs/2609.30918](https://arxiv.org/abs/2609.30918)

    本文提出一个统一的跨家族评估协议，在固定事实实例与反事实解释的前提下，针对相同的八种模型变化类型比较六种鲁棒反事实解释方法与两种基线，发现各方法的相对性能和失败模式随变化类型而异，因此现有报告的鲁棒性分数彼此不可比。

    

    鲁棒反事实解释承诺提供在其背后模型发生变化后仍然有效的补救措施。它们是否兑现这一承诺取决于变化是什么：参数的微小扰动、在新数据上的重新训练以及更换新架构是不同的事件，而每种现有方法都是针对其所设计的那个特定变化进行评估的。因此，已报告的鲁棒性分数回答的是不同的问题，彼此之间无法比较。我们提出了一个统一的跨家族评估协议，该协议保持事实实例和生成的反事实实例固定不变，同时针对相同的八种模型变化类型测试每种方法。该基准在四个表格数据集上比较了六种鲁棒方法和两种标准基线方法，通过每个变化后分类器的输出对其进行刻画，并报告经验鲁棒性以及覆盖率、基准有效性和接近度。我们发现，相对性能和失败模式在不同的变化家族之间存在差异。

    arXiv:2609.30918v1 Announce Type: cross  Abstract: Robust counterfactual explanations promise recourse that still works after the model behind it changes. Whether they keep that promise depends on what the change is. A small perturbation of the parameters, retraining on new data, and a new architecture are different events, and each existing method is evaluated against the one it was built for. Reported robustness scores, therefore, answer different questions and cannot be compared. We propose a unified cross-family evaluation protocol that holds factual instances and generated counterfactuals fixed while testing every method against the same eight types of model change. The benchmark compares six robust methods and two standard baselines on four tabular datasets. It characterizes every changed classifier through its outputs and reports empirical robustness together with coverage, base validity, and proximity. We find that relative performance and failure modes vary across change famil
    
[^91]: 在网页图上训练图基础模型

    Training Graph Foundation Models on The Web Graph

    [https://arxiv.org/abs/2609.30894](https://arxiv.org/abs/2609.30894)

    Acacia是一个仅用Common Crawl网页图从零训练的图基础模型，无需额外训练即可支持任意特征维度及节点分类、链接预测、节点聚类、图生成等多种任务，具备上下文学习能力且不依赖预训练LLM，证明了图模型可以像LLM一样从零涌现能力。

    

    我们介绍了Acacia，一个在网页图上训练的图基础模型。Acacia（i）支持任意的特征维度和语义，无需额外训练；（ii）支持广泛的任务，包括节点分类、链接预测、节点聚类和图生成，同样无需额外训练；（iii）具备上下文学习能力；（iv）不依赖预训练的大型语言模型（LLM）。特别值得注意的是，现有的图基础模型通常需要训练额外的分类头或特征投影器来适应新的图或新的标签，而Acacia不需要。此外，现有的图基础模型往往通过与预训练LLM拼接组合来获得其能力，而Acacia仅使用Common Crawl网页图从零开始训练。这也是一项重要的成果，因为它提供了证据，证明图模型可以像LLM一样从零开始获得涌现能力。

    arXiv:2609.30894v1 Announce Type: new  Abstract: We introduce Acacia, a graph foundation model, trained on the web graph. Acacia (i) supports arbitrary feature dimensionalities and semantics without additional training, (ii) supports a wide range of tasks, including node classification, link prediction, node clustering, and graph generation, without additional training, (iii) has in-context learning capabilities, and (iv) does not rely on pretrained LLMs. In particular, existing graph foundation models often require training additional classification heads or feature projectors to accommodate new graphs or new labels, whereas Acacia does not. Moreover, existing graph foundation models often gain their capabilities by being stitched together with pretrained LLMs, whereas Acacia is trained from scratch using only the Common Crawl web graph. This is also an important result because it provides evidence that graph models can acquire emergent capabilities from scratch like LLMs.
    
[^92]: 指数倾斜联合偏移下的保形预测

    Conformal Prediction under Exponential-Tilt Joint Shift

    [https://arxiv.org/abs/2609.30886](https://arxiv.org/abs/2609.30886)

    该论文研究了在输入分布及其与结果关系同时发生联合偏移时，比较了在保形预测校准中直接使用ExTRA估计的指数倾斜权重与额外对源预测分布进行倾斜这两种适应策略的覆盖性能。

    

    当数据分布部署后发生变化时，保形预测可能会失去覆盖保证。我们研究了利用带标签的源数据和未带标签的目标输入进行适应的方法，允许输入分布及其与结果之间的关系同时发生变化。我们采用由Maity等人（2023）针对分类问题提出的指数倾斜重加权对齐方法来估计结构化的分布偏移。我们比较了两种策略：在保形校准中使用其估计权重，以及在此外再对源预测分布进行倾斜。通过共享学习预测器、估计权重、校准样本和测试观测数据，隔离出了倾斜操作本身的影响。现有理论表明，使用真实权重时两种方法均可达到目标覆盖率，而使用估计权重时两者具有共同的覆盖率界。识别计算以及关于评分如何与权重估计误差相互作用的分析，有助于解释为什么两者性能仍然可能存在差异。在合成回归实验……

    arXiv:2609.30886v1 Announce Type: new  Abstract: Conformal prediction can lose coverage when the data distribution changes after deployment. We study adaptation using labeled source data and unlabeled target inputs, allowing both the input distribution and its relationship with outcomes to change. We use Exponential Tilt Reweighting Alignment (ExTRA), introduced for classification by Maity et al. (2023), to estimate structured distribution shifts. We compare using its estimated weights in conformal calibration with additionally tilting the source predictive distribution. Shared learned predictors, estimated weights, calibration samples, and test observations isolate the effect of tilting. Existing theory gives both procedures target coverage with true weights and a common coverage bound with estimated weights. Identification calculations and an analysis of how scoring interacts with weight estimation error help explain why their performance can nevertheless differ. In a synthetic regre
    
[^93]: 流形上基于回缩的梯度投影算法

    Retraction-Based Gradient Projection Algorithms on Manifolds

    [https://arxiv.org/abs/2609.30885](https://arxiv.org/abs/2609.30885)

    本文提出了黎曼流形上基于回缩的凸优化框架，建立了多种步长规则下基于回缩的梯度投影算法的收敛性结果，并成功应用于加权低秩逼近和图像补全问题。

    

    我们提出了一个在黎曼流形上进行基于回缩的凸优化的框架，其中包括回缩特定凸集的概念以及基于回缩的梯度投影算法。梯度投影算法的标准理论可以很容易地推广到该框架中。在该框架内，我们建立了采用各种步长规则的基于回缩的梯度投影算法的收敛性结果。作为应用，我们利用该框架研究了加权低秩逼近问题。我们还在图像补全任务上对我们的收敛性结果提供了数值验证。

    arXiv:2609.30885v1 Announce Type: cross  Abstract: We introduce a framework for retraction-based convex optimization on Riemannian manifolds, which includes a notion of retraction-specific convex sets and retraction-based gradient projection algorithms. The standard theory of gradient projection algorithms generalizes easily to this framework. Within this framework, we establish convergence results for retraction-based gradient projection algorithms with various stepsize rules. As an application, we use our framework to study the weighted low-rank approximation. We also provide numerical validation of our convergence results on the image completion task.
    
[^94]: CacheReforge：演化适配器下陈旧KV缓存的有界恢复

    CacheReforge: Bounded Recovery for Stale KV Caches under Evolving Adapters

    [https://arxiv.org/abs/2609.30884](https://arxiv.org/abs/2609.30884)

    提出CacheReforge，将陈旧KV缓存表示为逐层混合版本对象，结合适配器锚点、敏感度、累积漂移与重启边界，以最小重计算在适配器演化时恢复当前模型行为。

    

    大语言模型依赖KV缓存来减少长上下文和交互式应用中重复的预填充计算。随着轻量级适配器不断演化，缓存状态反映的是较早版本，因此复用陈旧缓存会扭曲当前模型的输出，而对所有受影响后缀进行完整重计算虽然能恢复保真度，但代价高昂。我们寻求以最小重计算恢复当前适配器的行为。现有系统追踪词元、上下文或稳定的适配器身份，但既无法表示来自较早适配器版本的缓存，也无法区分更新传播与行为恢复所需的重计算。为弥补这些不足，我们提出CacheReforge，它将陈旧的KV缓存表示为逐层的混合版本对象，并结合逐层适配器锚点、经校准的敏感度、累积漂移以及可执行的重启边界，来选择直接复用、有界重计算或完整的受影响后缀恢复。我们区分

    arXiv:2609.30884v1 Announce Type: new  Abstract: Large language models rely on KV caching to reduce repeated prefill computation in long context and interactive applications. As lightweight adapters evolve, cached states reflect earlier versions, so stale reuse distorts current model outputs, while complete affected suffix recomputation restores fidelity at substantial cost. We seek minimal recomputation that recovers current adapter behavior. Existing systems track token, context, or stable adapter identity, but neither represent caches from earlier adapter versions nor distinguish update propagation from the recomputation required for behavioral recovery. To address these gaps, we introduce CacheReforge, which represents stale KV caches as layerwise mixed-version objects. It combines per-layer adapter anchors, calibrated sensitivity, accumulated drift, and executable restart boundaries to select direct reuse, bounded recomputation, or complete affected-suffix recovery. We distinguish
    
[^95]: EXAONE Demand 1.0：一个面向需求预测的时间序列基础模型

    EXAONE Demand 1.0: A Time Series Foundation Model for Demand Forecasting

    [https://arxiv.org/abs/2609.30880](https://arxiv.org/abs/2609.30880)

    该论文提出了EXAONE Demand——一个专为需求预测设计的时间序列基础模型，通过构建包含1130万条序列的需求专用语料库，以及基于四类需求类别（平滑、间歇、波动、块状）进行路由的低秩适配器架构，解决了通用时间序列基础模型难以处理需求数据短历史、频繁零值、缺货删失等特殊性质的问题。

    

    时间序列基础模型（TSFM）通常在来自多个领域的序列数据上进行预训练，其中需求序列仅占很小一部分。需求数据具有此类语料库中罕见的特性：历史数据短、频繁出现零值、因缺货导致的删失，以及序列未能记录的外生事件。为此，我们提出了EXAONE Demand，其构建基于两大要素：1）需求专用语料库；2）需求感知适配器。在语料库方面，我们从73个数据源收集了1130万条序列和484亿个观测值，并通过合成生成器补充开放需求数据中代表性不足的行为模式。在适配器方面，我们在冻结的通用领域主干网络上附加低秩分支，分别对应四类需求（平滑型、间歇型、波动型和块状型），并由一个读取输入序列八项无标度统计量的路由器来决定各分支的贡献权重。我们构建了两个版本的EXAONE Demand，其中一个在真实世界与合成需求数据上训练。

    arXiv:2609.30880v1 Announce Type: new  Abstract: Time series foundation models (TSFMs) are pretrained on series from diverse domains, where demand series make up only a small fraction. Demand data has properties that such corpora rarely contain: Short histories, frequent zeros, censoring by stock-outs, and exogenous events that the series does not record. To this end, we propose EXAONE Demand, built on 1) a demand-specific corpus and 2) a demand-aware adapter. For the corpus, we assemble 11.3M series and 48.4B observations from 73 sources, and a synthetic generator supplies the behaviour that open demand data under-represents. For the adapter, we attach low-rank branches to a frozen general-domain backbone, one for each of the four demand classes (smooth, intermittent, erratic, and lumpy), and a router that reads eight scale-free statistics of the input series decides how much each branch contributes. We build EXAONE Demand in two versions, one trained on real-world and synthetic deman
    
[^96]: TISD：基于轨迹干预的在线策略自蒸馏

    TISD: On-Policy Self-Distillation with Trajectory Intervention

    [https://arxiv.org/abs/2609.30878](https://arxiv.org/abs/2609.30878)

    该论文提出TISD算法，通过在师生分歧峰值处强制执行教师偏好的分支动作并重生成轨迹进行蒸馏，突破了在线策略自蒸馏无法监督学生未采样分支的、单纯依赖局部纠错的训练瓶颈。

    

    在线策略自蒸馏（OPSD）能够提供密集的教师目标，但仅在学生采样的轨迹上对这些目标进行评估。当拥有特权信息的教师在已访问的前缀处偏好另一种动作时，OPSD可以为该分支决策提供目标，却无法监督由该动作所引发的后继上下文，除非学生自己采样到它。这造成了训练阶段的数据收集瓶颈，并提示师生分歧应当扮演不同的角色：提出轨迹分支，而非识别充分的局部修复。我们基于受控token干预的诊断框架揭示：在分歧峰值处的教师偏好token能够提升学生后续延续的成功率，而其局部纠正价值有限。受这一发现启发，我们提出了一种简单的“分支—重生成—蒸馏”算法——轨迹干预自蒸馏（TISD）。TISD强制执行教师选择的分支动作，并返回后继……（摘要原文在此处截断）

    arXiv:2609.30878v1 Announce Type: new  Abstract: On-policy self-distillation (OPSD) provides dense teacher targets, but evaluates them only along student-sampled rollouts. When the privileged teacher favors an alternative action at a visited prefix, OPSD can provide a target for the branch decision but cannot supervise the successor contexts induced by that action unless the student samples it. This creates a training-time data-collection bottleneck and suggests a different role for teacher-student disagreement: proposing a trajectory branch rather than identifying a sufficient local repair. Our diagnostic framework using controlled token interventions reveals that a teacher-preferred token at peak disagreement can improve student continuation success, while its local corrective value is limited. Motivated by this finding, we introduce a simple branch-regenerate-distill algorithm, Trajectory-Intervention Self-Distillation (TISD). TISD forces a teacher-selected branch action, returns su
    
[^97]: 非凸-强凹极小极大优化中紧致的随机条件数依赖性

    Tight Stochastic Condition-Number Dependence in Nonconvex-Strongly-Concave Minimax Optimization

    [https://arxiv.org/abs/2609.30877](https://arxiv.org/abs/2609.30877)

    本文证明了非凸-强凹极小极大优化中SAPD+算法随机复杂度的线性条件数依赖性是紧致不可改进的，给出了与之匹配的最坏情况复杂度下界Θ(κLGσ²ε⁻⁴)。

    

    我们研究了在非凸-强凹极小极大优化问题中，SAPD+算法随机复杂度对条件数的线性依赖是否是必要的。对于具有对偶强凹参数μ的联合L-光滑目标函数，我们在相同的Moreau包络平稳性准则和相同的原始-对偶初始化间隙条件下，证明了一个与SAPD+上界相匹配的下界。具体而言，当σ≥ε时，零尊重算法在所述精度范围内的最坏情况复杂度为Θ(κLGσ²ε⁻⁴)，其中κ=L/μ，G界定了初始原始-对偶间隙，σ²界定了一般无偏一阶随机预言机的方差。该下界在一个具有有界对偶盒子的光滑问题类上实现。我们的构造将非凸零链的每一环节通过幅值正比于ε/√κ的对偶梯度进行路由，同时一个未被发现……

    arXiv:2609.30877v1 Announce Type: cross  Abstract: We study whether the linear condition-number dependence in the stochastic complexity of SAPD+ is necessary for nonconvex-strongly-concave minimax optimization. For jointly $L$-smooth objectives with dual strong-concavity parameter $\mu$, we prove a lower bound that matches the SAPD+ upper bound under the same Moreau-envelope stationarity criterion and the same primal-dual initialization gap. Specifically, when $\sigma\ge\varepsilon$, the worst-case complexity of zero-respecting algorithms is $\Theta(\kappa LG\sigma^2\varepsilon^{-4})$ in the stated accuracy regime, where $\kappa=L/\mu$, $G$ bounds the initial primal-dual gap, and $\sigma^2$ bounds the variance of a general unbiased first-order oracle. The lower bound is realized on a smooth problem class with a bounded dual box. Our construction routes each link of a nonconvex zero-chain through a dual gradient of magnitude proportional to $\varepsilon/\sqrt{\kappa}$, while an undiscov
    
[^98]: 使用单个深度神经网络的交流潮流预想事故分析

    AC Power Flow Contingency Analysis Using a Single Deep Neural Network

    [https://arxiv.org/abs/2609.30859](https://arxiv.org/abs/2609.30859)

    本文提出一种仅用基准工况交流潮流数据训练的单个深度神经网络框架，通过不动点迭代预测任意单线路停运后的事故后运行状态，并利用半定规划验证收敛条件，从而避免了传统机器学习方法需要针对每种预想事故单独收集数据和训练模型的高昂离线成本。

    

    基于交流潮流（AC-PF）模型的预想事故分析是精确电网安全评估的关键工具，但其计算负担会随着需要评估的运行场景和停运配置数量的增加而增长。近期基于机器学习的方法通常需要针对特定停运情况的训练数据，导致离线训练成本随预想事故数量的增加而扩展。本研究提出了一个框架，该框架复用仅在基准工况AC-PF数据上训练的单个机器学习模型，来估计任意单线路停运下的事故后运行状态。所提出的方法将事故后状态预测公式化为不动点迭代问题。当机器学习模型为深度神经网络（DNN）时，我们推导了保证收敛的充分条件，并开发了半定规划（SDP）公式来为给定的DNN验证这些条件。在IEEE 118节点系统上的数值测试证明了所提出的SDP认证方法的有效性。

    arXiv:2609.30859v1 Announce Type: cross  Abstract: Contingency analysis using the AC power flow (AC-PF) model is a critical tool for accurate grid security assessment, but its computational burden increases with the number of operating scenarios and outage configurations to evaluate. Recent ML-based approaches typically require outage-specific training data, leading to offline training costs that scale with the number of contingencies. This work proposes a framework that reuses a single ML model trained solely on basecase AC-PF data to estimate post-contingency operating states under arbitrary single-line outages. The proposed approach formulates post-contingency state prediction as a fixed-point iteration. If the ML model is a deep neural network (DNN), we derive sufficient conditions that guarantee convergence and develop semidefinite programming (SDP) formulations to certify these conditions for a given DNN. Numerical tests on the IEEE 118-bus system demonstrate that the proposed SD
    
[^99]: 基于贝尔曼分布式证书的机会约束马尔可夫决策过程学习

    Learning Chance-Constrained MDPs with Bellman Distributional Certificates

    [https://arxiv.org/abs/2609.30856](https://arxiv.org/abs/2609.30856)

    本文提出“贝尔曼分布式证书”这一核心技术，通过在策略选择之前为约束违反概率构建贝尔曼递归，证明了机会约束MDP虽然计算上更困难，但其统计学习代价并不更高，所建立的样本复杂度上界与理论下界在对数因子内匹配。

    

    安全强化学习（RL）通常强制执行期望成本约束，但这种基于期望的安全性可能无法控制罕见高成本轨迹发生的概率。机会约束马尔可夫决策过程（CCMDP）施加了更强的概率层面要求，但被普遍认为更难处理，因为机会约束是非凸的，且依赖于完整轨迹而非贝尔曼线性期望。本文揭示了这种计算上的困难并不必然意味着更高的统计代价。对于具有固定有界后继状态支撑集、并可访问认证规划预言机的表格型折扣CCMDP，我们建立了一个基于模型的上界，并给出了在对数因子内与之匹配的下界。在技术层面，我们的核心思想是“贝尔曼分布式证书”，它在策略选择之前为约束违反概率构建贝尔曼递归。该证书可以在候选策略之间复用。

    arXiv:2609.30856v1 Announce Type: new  Abstract: Safe reinforcement learning (RL) commonly enforces expected-cost constraints, but such expectation safety may fail to control the probability of rare high-cost trajectories. Chance-constrained MDPs (CCMDPs) impose a stronger probability-level requirement, but are widely viewed as harder because the chance constraint is nonconvex and depends on the full trajectory rather than a Bellman-linear expectation. In this paper, we reveal that this computational difficulty does not necessarily imply a higher statistical price. For tabular discounted CCMDPs with fixed bounded successor support and access to a certified planning oracle, we establish a model-based upper bound, with a matching lower bound up to logarithmic terms. Technically, our key idea is the \emph{Bellman distributional certificate}, which constructs a Bellman recursion for constraint violation probabilities before policy selection. The certificate can be reused across candidate p
    
[^100]: KV缓存：新的内存墙

    The KV Cache Is the New Memory Wall

    [https://arxiv.org/abs/2609.30854](https://arxiv.org/abs/2609.30854)

    这篇系统化知识论文通过推导算术强度随上下文长度衰减的闭式公式，并在H100、B200和MI300X等硬件上进行参数化建模，首次从分析角度统一了KV缓存优化这一领域，解决了各研究间因工作负载和指标不一致而无法比较的问题。

    

    自回归大语言模型（LLM）在长上下文推理时受限于内存带宽而非算术吞吐量，且随着序列长度的增长，瓶颈资源从模型权重转移到了键值缓存。对于BF16精度的Llama-3-70B模型，其140 GB的权重大小已超过单个加速器80 GB的HBM容量，而一条128k token的序列还会额外增加42 GB的KV缓存。压缩、驱逐、分页、共享或卸载KV状态的技术层出不穷，但各项研究所报告的性能提升使用了不一致的工作负载、硬件和质量指标，导致无法进行跨论文比较。这篇系统化知识（SoK）论文从分析角度统一了该领域，提出了一个严格区分推导结论与实证报告结论的协议。我们推导出算术强度作为上下文长度衰减函数的闭式表达式，并以NVIDIA H100、NVIDIA B200和AMD MI300X的硬件拓扑为参数进行建模，涵盖每芯片带宽划分以及KV流量超过权重流量的临界长度。

    arXiv:2609.30854v1 Announce Type: cross  Abstract: Autoregressive LLM inference at long context is bounded by memory bandwidth, not arithmetic throughput, and the binding resource shifts from model weights to the Key-Value (KV) cache as sequence length grows. For Llama-3-70B in BF16, the 140 GB weight footprint exceeds the 80 GB HBM of a single accelerator, and one 128k-token sequence adds 42 GB of KV cache. Techniques that compress, evict, page, share, or offload KV state have proliferated, but reported gains use inconsistent workloads, hardware, and quality metrics, preventing cross-paper comparison. This SoK paper unifies the field analytically, with a protocol that strictly separates derived and reported claims. We derive closed-form arithmetic intensity as a decaying function of context length, parameterized by hardware topology for NVIDIA H100, NVIDIA B200, and AMD MI300X, including per-die bandwidth partitioning and the crossover lengths where KV traffic overtakes weight traffic
    
[^101]: 基于奖励加权传输蒸馏的单步生成模型对齐

    Aligning One-Step Generative Models with Reward-Weighted Transport Distillation

    [https://arxiv.org/abs/2609.30840](https://arxiv.org/abs/2609.30840)

    提出了奖励加权传输蒸馏（RWTD）后训练方法，仅凭生成样本和标量奖励评估，通过混合倾斜的当前分布与参考分布构建自适应目标，并借助特征空间最优传输和不动点回归，实现对单步生成模型的有效对齐。

    

    单步生成器只需一次网络评估即可实现高质量的视觉生成，但其后训练十分困难：一般的隐式生成器既不提供可处理的似然，也不提供去噪轨迹，而且许多奖励函数是不可微分的。我们提出了奖励加权传输蒸馏（RWTD），这是一种仅需生成样本和标量奖励评估的后训练方法。RWTD并非仅对齐于传统的奖励倾斜参考分布，而是构建了一个自适应目标，该目标混合了分别经过倾斜处理的当前分布与参考分布。当前分量融入了训练过程中发现的改进，而参考分量则将目标锚定在预训练生成器上。RWTD通过特征空间最优传输和不动点回归来实现该目标。理论分析表明，RWTD的不动点分布在离策略（off-policy）与……之间进行插值（摘要在此处截断）。

    arXiv:2609.30840v1 Announce Type: new  Abstract: One-step generators enable high-quality visual generation with a single network evaluation, but their post-training is difficult: general implicit generators provide neither tractable likelihoods nor denoising trajectories, and many rewards are non-differentiable. We introduce Reward-Weighted Transport Distillation (RWTD), a post-training method that requires only generated samples and scalar reward evaluations. Rather than aligning solely to the conventional reward-tilted reference distribution, RWTD constructs an adaptive target that mixes separately tilted current and reference distributions. The current component incorporates improvements discovered during training, while the reference component anchors the target to the pretrained generator. RWTD realizes this target through feature-space optimal transport and fixed-point regression. Theoretical analysis shows that the fixed-point distributions of RWTD interpolate between off-policy
    
[^102]: 面向同声语音到文本翻译的基于注意力的自适应策略

    Attention-Based Adaptive Policies for Simultaneous Speech-to-Text Translation

    [https://arxiv.org/abs/2609.30839](https://arxiv.org/abs/2609.30839)

    本文提出利用交叉注意力机制的RFAP和DCAP两种自适应策略，使离线训练的语音翻译模型无需额外训练即可用于同声翻译，最高提升4.0 BLEU并降低翻译延迟。

    

    同声语音到文本翻译是指在系统处理输入音频帧的同时生成部分翻译结果。然而，这种设置的流式特性带来了一个挑战：如何在最小化延迟的同时决定执行准确翻译的最佳时机。为了解决这一挑战，我们利用编码器-解码器架构中的交叉注意力机制来寻找输入语音帧与目标文本词元之间的正确对齐。在本文中，我们提出了近期帧注意力策略和双条件注意力策略，使离线训练的语音到文本翻译模型无需额外训练即可应用于流式场景。在CVSS-C语料库上针对三个不同语言翻译对的实验结果表明，RFAP能够超越其他策略，BLEU分数最高提升4.0，同时降低了翻译延迟。

    arXiv:2609.30839v1 Announce Type: new  Abstract: Simultaneous speech-to-text translation (Simul-S2TT) consists of generating partial translations while the incoming audio frames are processed by the system. However, the streaming nature of this setup creates the challenge of deciding the best moment to perform an accurate translation while minimizing the delay. To address this challenge, we utilize the cross-attention mechanism of the encoder-decoder architecture to find the right alignment between the input speech frames and the target text tokens. In this paper, we propose the Recent Frame Attention Policy (RFAP) and the Dual-Condition Attention Policy (DCAP) that allow offline trained speech-to-text translation models to be used in streaming scenarios without requiring additional training. Results on three different language translation pairs over the CVSS-C corpus show that the RFAP is able to surpass other policies with gains of up to 4.0 BLEU while reducing the translation delay 
    
[^103]: 面向慢性健康管理的基于同伴的反事实路径规划

    Peer-Grounded Counterfactual Path Planning for Chronic Health Management

    [https://arxiv.org/abs/2609.30838](https://arxiv.org/abs/2609.30838)

    该论文提出POROS框架，根植于自我效能理论与社会比较理论，通过从真实相似患者的观测状态构建行为进展图，以反事实路径规划生成有同伴证据支撑、能保证健康指标单调改善的渐进式分步干预方案，克服了现有方法只给目标、缺少路径、缺乏同伴依据的缺陷。

    

    慢性疾病管理中有效的行为干预所需的并非单一的处方，而是一系列渐进式的小步骤，每一步都立足于真实的、相似的个体已经切实达成的成果。反事实解释为这种指导提供了一条自然的计算路径，它回答的是：何种行为改变本可以带来更好的结果。但现有方法仅返回一个目标状态却没有通往该状态的路径，无法保证途中健康指标的单调改善，也没有从同伴行为中汲取证据——这等于要求患者一步跨越巨大的差距，而这恰恰是最不可能被尝试的推荐结构。我们提出了POROS（基于同伴的状态最优路径规划），这是一个领域无关的框架，根植于班杜拉的自我效能理论与费斯廷格的社会比较理论，构建了一个行为进展图——一个建立在观测到的患者状态之上的有向无环图，其中每条边（摘要至此截断）。

    arXiv:2609.30838v1 Announce Type: new  Abstract: Effective behavioral intervention in chronic disease management requires not a single prescription but a sequence of incremental steps, each grounded in what real, similar individuals have demonstrably achieved. Counterfactual explanation offers a natural computational route to such guidance, answering what change in behavior would have produced a better outcome. But existing methods return a target state without a route to it, guarantee no monotone health improvement along the way, and draw no evidence from peer behavior -- asking a patient to close a wide gap in one move, which is precisely the recommendation structure least likely to be attempted. We propose POROS (Peer-Grounded Optimal Routes Over States), a domain-agnostic framework rooted in Bandura's self-efficacy theory and Festinger's social comparison theory that constructs a Behavioral Progression Graph -- a directed acyclic graph over observed patient states in which every ed
    
[^104]: MOPD-Router：重新思考多教师在策略蒸馏中的教师路由

    MOPD-Router: Rethinking Teacher Routing in Multi-Teacher On-Policy Distillation

    [https://arxiv.org/abs/2609.30837](https://arxiv.org/abs/2609.30837)

    提出MOPD-Router框架，无需领域标签即可在token级别对完整教师池进行监督路由，并通过ExpertAlign依据教师后训练所获专业化能力为其修正信号评分，从而充分释放多教师互补知识。

    

    多教师在策略蒸馏（MOPD）旨在将多个专业化能力整合到单个学生模型中，但现有做法通常将每个提示硬路由到与其领域匹配的教师，并让其完成整个生成过程。这种对提示级领域标签的依赖限制了无标签训练混合数据的使用，也使其他教师的互补信号未被利用。我们提出MOPD-Router，一个无需领域标签、也无需训练额外路由模型的框架，它在每个token上对完整教师池进行监督路由。其插件式接口支持多种度量标准来选择和加权来自不同教师的OPD信号。在该接口内，我们提出ExpertAlign，它通过判断教师对学生当前token的修正是否体现了该教师在后训练阶段所获得的专业化能力来为每个教师评分，并将其与基于教师置信度和师生判别性的两种参考度量进行比较。

    arXiv:2609.30837v1 Announce Type: cross  Abstract: Multi-teacher on-policy distillation (MOPD) integrates specialized capabilities into a single student, but existing practice typically hard-routes each prompt to a domain-matched teacher for the entire rollout. This dependence on prompt-level domain labels restricts using unlabeled training mixtures and leaves complementary signals from other teachers unused. We introduce MOPD-Router, a framework that routes supervision over the full teacher pool at each token, without domain labels or training a separate routing model. Its plug-in interface supports different metrics for selecting and weighting teacher-specific OPD signals. Within this interface, we propose ExpertAlign, which scores each teacher by whether its correction to the student at the current token expresses the specialization that teacher acquired during post-training, and compare it against two reference metrics built on teacher confidence (Entropy) and teacher-student discr
    
[^105]: 面向粒子模拟的自适应交互图

    Adaptive Interaction Graphs for Particle Simulation

    [https://arxiv.org/abs/2609.30822](https://arxiv.org/abs/2609.30822)

    提出AdaptGNS，利用逐粒子不确定性估计自适应地扩展高不确定性粒子的交互图邻域，在几乎不增加推理成本的情况下，于长时程粒子模拟中实现严格的帕累托改进。

    

    基于图神经网络的习得粒子模拟器在单步预测上具有很高的精度，但误差会在长时间跨度内不断累积。一个尚未被充分探索的变量是交互图本身：现有方法通过k近邻或静态半径规则来固定其拓扑结构，而不考虑模型的局部置信度。我们提出让这一图结构具备自适应性：一个逐粒子的方差头在异方差高斯NLL损失下与加速度头联合训练，驱动轨迹中高不确定性粒子获得扩展的邻域。通过使用上一步的不确定性估计，这一过程几乎不产生额外的推理成本。一个关键发现是，方差头学到了有意义的不确定性概念：高方差粒子集中在复杂区域附近，例如飞溅区或自由表面。当这一信号用于驱动图拓扑时，所得到的AdaptGNS模拟器在WaterDrop数据集上实现了严格的帕累托改进。

    arXiv:2609.30822v1 Announce Type: new  Abstract: Learned particle simulators based on graph neural networks achieve strong one-step accuracy, but errors compound over long horizons. An underexplored variable is the interaction graph: existing methods fix its topology via k-nearest neighbors or a static radius rule, regardless of local model confidence. We propose making this graph adaptive: a per-particle variance head, trained jointly with the acceleration head under a heteroscedastic Gaussian NLL loss, drives a trajectory in which high-uncertainty particles receive an expanded neighborhood. This is done at little extra inference cost by using the previous step's uncertainty estimate. A key discovery is that the variance head learns a meaningful notion of uncertainty: high-variance particles concentrate near complex regions, such as splash zones or free surfaces. When this signal drives graph topology, the resulting AdaptGNS simulator achieves a strict Pareto improvement on WaterDrop 
    
[^106]: 循环Transformer的量化：反馈暴露与校准盲区

    Quantizing Looped Transformers: Feedback Exposure and Calibration Blindness

    [https://arxiv.org/abs/2609.30820](https://arxiv.org/abs/2609.30820)

    该论文揭示了循环Transformer低比特训练后量化的两种失效模式——“反馈暴露”（无恒等路径的量化层误差在递归中被反复放大回馈，且该现象同样存在于Mamba等非Transformer架构）和“校准盲区”（单步GPTQ仅依赖第0步激活校准，忽视后续递归步的输入方向）。

    

    循环Transformer在递归步骤间复用权重，这使得低比特量化对其格外有吸引力。我们识别出标准训练后量化的两种不同失效模式。在Huginn-3.5B模型上，逐通道INT4量化主要在非残差循环入口适配器处失效，而量化残差核心部分造成的损害要小得多。我们将这一现象称为反馈暴露：量化层在缺乏恒等路径的情况下扰动递归状态，由此产生的误差会在后续步骤中被反馈回来。在线性滤波器和Mamba状态空间模型上的受控实验表明，反馈暴露也存在于Transformer架构之外。分组INT4则揭示了另一种独立的失效模式——校准盲区：我们的单步GPTQ基线仅从第0步激活构建Hessian矩阵，导致递归后续步骤中使用的输入方向几乎未被纳入加权。在来自七种循环架构的九个检查点上，单步GPTQ在p

    arXiv:2609.30820v1 Announce Type: cross  Abstract: Looped transformers reuse weights across recurrence steps, making low-bit quantization especially attractive. We identify two distinct failure modes of standard post-training quantization. On Huginn-3.5B, per-channel INT4 fails primarily at the non-residual loop-entry adapter, while quantizing the residual core is much less damaging. We call this feedback exposure: a quantized layer perturbs the recurrent state without an identity path, and the resulting error is fed back at later steps. Controlled experiments on linear filters and Mamba state-space models show that feedback exposure also occurs outside transformers. Grouped INT4 reveals a separate failure, calibration blindness: our one-step GPTQ baseline builds its Hessian from step-0 activations, leaving input directions used later in the recurrence nearly unweighted. Across nine checkpoints from seven looped architectures, one-step GPTQ is worse than round-to-nearest (RTN) on the p
    
[^107]: 面向不确定动态系统的可证明神经网络安全观测器学习

    Learning Provable Neural Network Observer for Uncertain Dynamical Systems

    [https://arxiv.org/abs/2609.30819](https://arxiv.org/abs/2609.30819)

    提出了一种两阶段训练框架（点引导李雅普诺夫预训练加LMI微调），为不确定动态系统的神经网络观测器实现了可证明的全局李雅普诺夫稳定性，同时克服了传统LMI方法带来的大规模半定规划可扩展性瓶颈。

    

    在许多安全关键应用中，不确定动态系统的控制依赖于能够估计状态和外部扰动的观测器。神经网络观测器可以提高估计精度，但通过线性矩阵不等式（LMI）约束来认证其李雅普诺夫稳定性会产生大规模的半定规划（SDP），对于大型网络而言难以求解。为了克服这一可扩展性瓶颈，我们提出了一种针对可证明稳定神经网络观测器的新型两阶段训练框架。该方法将优化过程解耦为两个阶段：首先是点引导的李雅普诺夫预训练阶段，可在采样状态上快速实现高估计精度和局部稳定性；随后是LMI微调阶段，可高效地满足严格的全局李雅普诺夫稳定性认证。我们为局部稳定性半径以及在给定紧致误差-状态域上的概率覆盖提供了形式化的理论保证。

    arXiv:2609.30819v1 Announce Type: new  Abstract: In many safety-critical applications, control of uncertain dynamical systems relies on observers that estimate states and external disturbances. Neural network observers can improve estimation accuracy, but certifying their Lyapunov stability via Linear Matrix Inequality (LMI) constraints leads to large-scale semidefinite programs (SDPs) that are difficult to solve for large networks. To overcome this scalability bottleneck, we propose a novel two-stage training framework for provably stable neural network observers. Our approach decouples the optimization into a point-guided Lyapunov pre-training phase, which rapidly achieves high estimation accuracy and local stability over sampled states, followed by an LMI fine-tuning phase that efficiently satisfies a strict global Lyapunov stability certificate. We provide formal theoretical guarantees for local stability radii and probabilistic coverage over a prescribed compact error-state domain
    
[^108]: 自适应记录下的反事实在线共形预测

    Counterfactual Online Conformal Prediction Under Adaptive Logging

    [https://arxiv.org/abs/2609.30811](https://arxiv.org/abs/2609.30811)

    本文提出倾向加权在线共形预测（PW-OCP）及其双重稳健变体（DR-OCP），通过逆倾向加权递归消除校准偏差，解决了自适应记录机制下在线共形预测对罕见行动反事实结果系统性覆盖失效的问题，并在正性条件下实现接近信息论下界的反事实覆盖率。

    

    当预测塑造行动、而行动又决定哪些结果进入校准时，在线共形预测可能会失效。标准自适应方法在保持边际覆盖率的同时，可能系统性地对很少被选中的行动的反事实结果产生错误的覆盖。本文通过反事实覆盖率将这一失效现象形式化，并提出了倾向加权在线共形预测，这是一种利用逆倾向加权递归来消除校准偏差的方法。其双重稳健变体进一步将干扰项偏差降低至结果模型误差与倾向误差的乘积。在正性条件下，所得的覆盖率在相差对数因子的意义下匹配信息论下界。在合成决策任务、开放bandit数据和金融再平衡上的实验表明，PW-OCP和DR-OCP在不牺牲预测集锐度的前提下，提升了反事实覆盖率并降低了下游遗憾。

    arXiv:2609.30811v1 Announce Type: new  Abstract: Online conformal prediction can fail when predictions shape actions and actions determine which outcomes enter calibration. Standard adaptive methods may retain marginal coverage while systematically miscovering the counterfactual outcomes of rarely selected actions. This paper formalizes the failure through counterfactual coverage and introduces Propensity-Weighted Online Conformal Prediction, an inverse-propensity-weighted recursion that debiases calibration. A doubly robust variant further reduces nuisance bias to the product of outcome-model and propensity errors. Under positivity, the resulting coverage rate matches an information-theoretic lower bound up to logarithmic factors. Experiments on synthetic decision tasks, open bandit data, and financial rebalancing show that PW-OCP and DR-OCP improve counterfactual coverage and downstream regret without sacrificing prediction-set sharpness.
    
[^109]: 面向无穷大Laplace问题与p-Laplace问题的深度学习求解器与代理模型

    Deep-Learning Solvers and Surrogates for Infinity and p-Laplace Problems

    [https://arxiv.org/abs/2609.30809](https://arxiv.org/abs/2609.30809)

    本文利用PINN和DeepONet求解无穷大Laplace与p-Laplace问题，在大p值（2至1000）和三维区域上优于传统网格求解器，并建立了条件收敛性与通用逼近的理论结果。

    

    我们研究了神经网络求解器在无穷大Laplace问题和p-Laplace问题中的应用，这些问题在非线性分析中具有基础性地位，并具有实际应用价值。我们的方法采用物理信息神经网络（PINNs）和深度算子网络来应对大p值（范围从2到1000）在各种二维和三维区域上带来的计算挑战。与传统的基于物理的求解器相比，我们的方法具有优势，尤其是在三维情况下，基于网格的求解器对此类问题的计算成本非常高。我们还为这两个问题的PINN近似建立了条件收敛性结果，并为DeepONet在参数化p-Poisson问题上的应用建立了通用逼近定理。我们通过数值实验验证了这些神经网络求解器的有效性，并将其性能与传统方法进行了比较。

    arXiv:2609.30809v1 Announce Type: cross  Abstract: We investigate the use of neural network solvers for infinity and $p$-Laplace problems, which are fundamental in nonlinear analysis and have practical applications. Our approach employs Physics-Informed Neural Networks (PINNs) and Deep Operator Networks (DeepONets) to address computational challenges associated with large $p$ values, ranging from $2$ to $1000$, on various 2D and 3D domains. Our method offers advantages over traditional physics-based solvers, especially in three dimensions where mesh-based solvers become very costly for these problems. We also establish conditional convergence results for PINN approximations of both problems and a universal approximation result for DeepONet on the parametric $p$-Poisson problem. We demonstrate the effectiveness of these neural network solvers through numerical experiments and compare their performance with conventional methods.
    
[^110]: 迈向通用的基于表示的过程控制

    Towards Universal Representation-Based Process Control

    [https://arxiv.org/abs/2609.30790](https://arxiv.org/abs/2609.30790)

    该论文将窗口级过程监控重新表述为以经验参考分布为零假设的假设检验问题，并提出一个结合预训练时间序列编码器、核密度估计与保形校准的基于表示的非参数框架，在学习到的表示空间中实现有限样本有效的过程控制推断。

    

    许多时间过程学习与监控流程在局部窗口中运行，使得窗口级别的决策在实践中不可避免。在这类设置中，经典统计检验虽可应用于单个窗口，但它们通常评估预定义的参数化假设——例如单位根或基于矩的条件——从而在参考行为由任务或领域特定数据以经验方式定义时限制了灵活性。在这项工作中，我们将窗口级监控视为一个过程控制问题，并将其重新表述为基于参考的假设检验，其中零假设由经验参考分布而非固定的参数模型来指定。我们通过一个基于表示的非参数框架来实现这一视角，该框架结合了预训练时间序列编码器、核密度估计和保形校准，在学习到的表示空间中实现有限样本有效的推断。

    arXiv:2609.30790v1 Announce Type: new  Abstract: Many temporal process learning and monitoring pipelines operate in local windows, making window-level decisions unavoidable in practice. In such settings, classical statistical tests can be applied to individual windows, but they typically evaluate predefined parametric hypotheses-such as unit-root or moment-based conditions-thereby limiting flexibility when reference behavior is defined empirically from task- or domain-specific data. In this work, we view window-level monitoring as a process control problem and reformulate it as reference-based hypothesis testing, where the null hypothesis is specified by an empirical reference distribution rather than a fixed parametric model. We operationalize this perspective through a representation-based, nonparametric framework that combines pretrained time series encoders, kernel density estimation, and conformal calibration, yielding finite-sample valid inference in learned representation space.
    
[^111]: 可解释性设计描述符组合在低数据分子检测上匹敌2048维基础模型嵌入

    Interpretable-by-Design Descriptor Portfolios Match a 2048-Dimensional Foundation Embedding on Low-Data Molecular Assays

    [https://arxiv.org/abs/2609.30789](https://arxiv.org/abs/2609.30789)

    该论文提出一种可解释性设计（interpretable-by-design）的描述符组合方法，通过贪婪拼接经过来源筛选的紧凑描述符块，在特征层面完全可审计的前提下，于低数据ADME/Tox检测上取得了与2048维CheMeleon基础嵌入相当的精度（平均AUC 0.762对0.764）。

    

    在低数据结构-活性预测中，分子表示的选择可能比预测器的选择更为关键，而表格基础模型进一步放大了这一效应。我们探究一个由紧凑、语义命名的描述符块组成的组合，能否在达到2048维CheMeleon嵌入精度的同时，在特征层面保持可审计性——即每个输入维度都带有模型名称和记录在案的训练来源。从固定的11维理化性质基础出发，我们仅利用带标签的上下文信息，贪婪地拼接经过来源筛选的描述符块。在九项ADME/Tox检测和50个评估单元上，以所有表示共同覆盖的分子所构成的公共覆盖子集进行评分，该组合达到平均测试AUC为0.762，相比之下CheMeleon为0.764，Mordred为0.756。与CheMeleon的汇总差距为+0.003 AUC（任务自助法95%置信区间[-0.020, +0.030]），这满足……（原文摘要至此截断）

    arXiv:2609.30789v1 Announce Type: new  Abstract: In low-data structure-activity prediction, the choice of molecular representation can matter more than the choice of predictor, and tabular foundation models sharpen that effect. We ask whether a portfolio of compact, semantically named descriptor blocks can reach the accuracy of a 2048-dimensional CheMeleon embedding while staying auditable at the feature level, meaning that every input dimension carries a model name and a recorded training provenance. Starting from a fixed 11-dimensional physicochemical base, we greedily concatenate provenance-screened blocks using the labelled context alone. Across nine ADME/Tox assays and 50 evaluation cells, scored on common-coverage subsets restricted to the molecules that every representation covers, the portfolio reaches a mean test AUC of 0.762, against 0.764 for CheMeleon and 0.756 for Mordred. The pooled gap to CheMeleon is +0.003 AUC (task-bootstrap 95% CI [-0.020, +0.030]), which satisfies o
    
[^112]: 跨医院分布偏移下缺失感知的共形预测

    Missingness-Aware Conformal Prediction Under Cross-Hospital Distribution Shift

    [https://arxiv.org/abs/2609.30781](https://arxiv.org/abs/2609.30781)

    提出一种缺失感知的共形校准方法，通过按测量指标是否缺失对患者分组并在组内应用Mondrian校准，在跨医院分布偏移的死亡率预测中有效缩小了最差分组的覆盖差距。

    

    临床测量指标在部分患者中会被记录，而在另一部分患者中则不会，且记录率因医院而异；边际共形覆盖并不能保证在按缺失情况定义的分组内实现覆盖。我们提出了一种面向跨医院分布偏移下死亡率预测的缺失感知共形校准方法。该方法在独立样本上选择一项测量指标，根据该测量是否被记录对患者进行分组，并在每组内应用Mondrian校准，从而避免重复使用任何校准结果。我们使用三种预测模型，在eICU数据集的多个医院之间以及MIMIC-IV数据集内某一家医院的不同护理单元之间对该方法进行了评估。与合并校准相比，该方法在全部六种设置中都降低了其选定分组上的平均最差组覆盖差距，中位数降幅为1.9个百分点；配对的站点自助法置信区间在五种设置中不包含零。这些收益并非在各处均匀体现。基于预测风险的校准……

    arXiv:2609.30781v1 Announce Type: new  Abstract: Clinical measurements are recorded for some patients but not others, at rates that differ across hospitals, and marginal conformal coverage does not ensure coverage within groups defined by missingness. We propose a missingness-aware conformal calibration procedure for mortality prediction under cross-hospital distribution shift. It selects a measurement on an independent sample, groups patients by whether that measurement is recorded, and applies Mondrian calibration within each group, so no calibration outcome is reused. We evaluate the procedure across hospitals in eICU and across care units within one MIMIC-IV hospital, using three predictors. Relative to pooled calibration, it reduces the average worst-group coverage gap on its selected groups in all six settings, with a median reduction of 1.9 percentage points; paired site-bootstrap intervals exclude zero in five. These gains do not extend uniformly. Calibration by predicted risk 
    
[^113]: 面向跨域少样本学习的查询条件原型自适应：单查询推理、受控比较与失效模式

    Query-Conditioned Prototype Adaptation for Cross-Domain Few-Shot Learning: Single-Query Inference, Controlled Comparisons, and Failure Modes

    [https://arxiv.org/abs/2609.30769](https://arxiv.org/abs/2609.30769)

    该论文提出实例内原型变换器（WIPT），在冻结的全局表示下通过测试时对单个查询与支持集嵌入进行联合变换来构建查询特定的类原型，并通过多随机种子的受控实验揭示其收益依赖目标域——在 CUB 和 EuroSAT 上带来提升，但在 ISIC 上失效。

    

    跨域少样本学习要求在不进行目标域参数更新的情况下，仅利用极少量带标注样本将分类器适配到新的视觉域。我们聚焦于一个问题：在固定全局表示的前提下，查询与支持集的联合自适应对原型构建有何贡献？实例内原型变换器（WIPT）通过联合变换一个无标注查询与带标注的支持集嵌入，再形成针对该查询的类均值，从而实现单查询的测试时原型自适应。实验采用共享的冻结 ViT-S/16 编码器、以 miniImageNet 为源域训练，并在 CUB、EuroSAT 和 ISIC 三个目标域上进行测试，我们在五个独立训练种子上重复了关键比较。在 1-shot 评估中，WIPT 在每次运行中都优于冻结的 ProtoNet（CUB 上 +0.21 个百分点，EuroSAT 上 +2.07），但在 ISIC 上有所下降（-0.22）。在 5-shot 评估中，ProtoNet 整体仍然最强，而 WIPT 持续改进了一个……

    arXiv:2609.30769v1 Announce Type: cross  Abstract: Cross-domain few-shot learning requires adapting a classifier to a new visual domain from very few labelled examples without target-time parameter updates. We isolate one question: under a fixed global representation, what does joint query-support adaptation contribute to prototype construction? The Within-Instance Prototypical Transformer (WIPT) implements single-query test-time prototype adaptation by jointly transforming one unlabelled query and the labelled support embeddings, then forming query-specific class means. Using a shared frozen ViT-S/16 encoder, miniImageNet source training, and CUB, EuroSAT and ISIC targets, we replicate the key comparisons across five independent training seeds. In 1-shot evaluation, WIPT improves frozen ProtoNet in every run on CUB (+0.21 percentage points) and EuroSAT (+2.07), but decreases ISIC (-0.22). In 5-shot evaluation, ProtoNet remains strongest overall, while WIPT consistently improves a capa
    
[^114]: HCOE：基于生物医学语言模型的双曲临床本体嵌入

    HCOE: Hyperbolic Clinical Ontology Embeddings from Biomedical Language Models

    [https://arxiv.org/abs/2609.30763](https://arxiv.org/abs/2609.30763)

    HCOE 通过将冻结的 BioBERT 嵌入映射到双曲庞加莱空间，并结合本体引导的对比学习与由粗到细的路径聚合，构建了保留医学代码层级结构的临床概念表示，在临床关系预测及死亡率、再入院、药物推荐等多项临床任务上均达到最佳性能。

    

    生物医学语言模型（LM）能够编码文本语义，但无法显式地保留医学代码的层级结构。我们提出了双曲临床本体嵌入（HCOE），用于构建具有层级感知能力的临床概念表示。HCOE 将冻结的 BioBERT 嵌入映射到庞加莱球中，将父侧和子侧本体引导的对比学习与由粗到细的本体路径聚合相结合，并利用由临床分类软件（CCS）组织以及解剖学治疗化学（ATC）药物层级结构组织的国际疾病分类（ICD）代码。评估结果表明，HCOE 在 ICD/ATC 临床关系预测和 CCS 到 PheCode 的层级迁移任务上表现最佳。在 MIMIC-IV 数据集上，HCOE 在死亡率预测、再入院预测、药物推荐和罕见药物预测任务上也取得了最佳性能。

    arXiv:2609.30763v1 Announce Type: new  Abstract: Biomedical language models (LMs) encode textual semantics but do not explicitly preserve medical code hierarchies. We present Hyperbolic Clinical Ontology Embeddings (HCOE) for hierarchy-aware clinical concept representation. HCOE maps frozen BioBERT embeddings into a Poincare ball, combining parent-side and child-side ontology-guided contrastive learning with coarse-to-fine ontology-path aggregation. It uses International Classification of Diseases (ICD) codes organized by Clinical Classifications Software (CCS) and Anatomical Therapeutic Chemical (ATC) medication hierarchies. Evaluations show that HCOE performs best on ICD/ATC clinical relation prediction and CCS-to-PheCode hierarchy transfer. On the MIMIC-IV dataset, HCOE also achieves the best performance on mortality prediction, readmission prediction, medication recommendation, and rare drug prediction.
    
[^115]: 基于可归因推理的技能画像（SPAR）：一种拳击可穿戴分析系统

    Skill Profiling with Attributable Reasoning (SPAR): A Wearable Analysis System for Boxing

    [https://arxiv.org/abs/2609.30753](https://arxiv.org/abs/2609.30753)

    SPAR是一种结合八单元IMU服装与压力鞋垫的拳击可穿戴系统，不仅能将出拳分类为专家或新手水平，还通过逐关节归因、动力链反事实解释和通俗语言叙述三个层级的可解释反馈，分别服务于分析师、教练和运动员，在17名参与者数据上取得0.842的AUC。

    

    拳击出拳是一种弹道式的全身动作，由从腿部经躯干传导至手臂的动力链驱动，微小的发力时序误差就可能决定一次得分击打与一次失误之间的差别。可穿戴传感器能够在训练馆中捕捉这一动作，但大多数可部署系统只能对出拳的类型进行分类，而无法评估出拳的质量。我们提出了基于可归因推理的技能画像系统（SPAR），这是一个由八单元IMU服装和压力鞋垫组成的系统，可将每次出拳分类为专家级或新手级，并将对该预测的解释作为反馈。只有当接收反馈的人能够据此采取行动时，反馈才有价值，因此SPAR在三个层级上对预测进行解释：为数据分析师提供逐关节归因，为教练提供基于动力链各环节的反事实解释，为运动员提供两者的通俗语言叙述。在17名参与者共4,713次出拳的数据上，SPAR达到了留一参与者交叉验证AUC为0.842（95%置信区间[0.7…

    arXiv:2609.30753v1 Announce Type: cross  Abstract: A punch is a ballistic, full-body action driven by a kinetic chain running from the legs through the trunk to the arm, where a small sequencing error separates a scoring strike from a miss. Wearable sensors can capture this movement in the gym, but most deployable systems only classify which punch was thrown rather than assess how well it was thrown. We present Skill Profiling with Attributable Reasoning (SPAR), an eight-IMU garment and pressure-insole system that classifies each punch as expert or novice and treats an explanation of that prediction as feedback. Feedback is only useful if the person receiving it can act on it, so SPAR explains the prediction at three tiers, a per-joint attribution for the analyst, a counterfactual over kinetic-chain layers for the coach, and a plain-language narrative of the two for the athlete. Across 17 participants and 4,713 punches, SPAR reaches a leave-one-participant-out AUC of 0.842 (95% CI [0.7
    
[^116]: 面向深度学习的可微分RNA二级结构提取

    Differentiable RNA Secondary Structure Extraction for Deep Learning

    [https://arxiv.org/abs/2609.30752](https://arxiv.org/abs/2609.30752)

    本文比较了四种RNA二级结构提取算法，揭示深度学习模型的训练方法与结构提取方法之间的一致性对预测性能具有关键影响。

    

    近年来，许多基于深度学习的RNA二级结构预测方法被相继提出。这些方法通常输出一个权重矩阵 $W$，其中 $W_{ij}$ 表示碱基 $i$ 与碱基 $j$ 配对的任意权重。将这一矩阵转换为预测的二级结构或碱基配对概率矩阵，通常需要借助临时的、且存在问题的事后下游算法。尽管这一转换步骤——我们称之为结构提取——十分重要，但它在文献中却相对较少受到关注。在本工作中，我们分析了训练方法与提取方法之间的一致性如何影响预测性能。为此，我们比较了四种提取算法：类Nussinov动态规划方法、最大权重图匹配算法，以及SPOT-RNA和RiNALMo所采用的贪婪提取算法。这些算法在预训练RiNALMo模型的输出以及本文训练的三个玩具模型的输出上进行评估……（摘要原文在此处被截断）

    arXiv:2609.30752v1 Announce Type: new  Abstract: Many deep learning approaches to RNA secondary structure prediction have recently been proposed. They typically output a weight matrix $W$ where $W_{ij}$ is an arbitrary weight for base $i$ pairing with base $j$. Converting this matrix to a predicted secondary structure or base-pairing probability matrix typically involves ad hoc and problematic downstream algorithms. Despite the importance of this conversion step, which we refer to as structure extraction, it has received relatively little attention in the literature. In this work, we analyze how the congruence between training and extraction methods affects prediction performance. To do this, we compare four extraction algorithms: a Nussinov-like dynamic programming method, maximum-weight graph matching and the greedy extraction algorithms used by SPOT-RNA and RiNALMo. These are evaluated on outputs from the pretrained RiNALMo model and three toy models trained in this paper: a differe
    
[^117]: 面向数据受限极端事件仿真的机制感知集合条件化方法

    Mechanism-Aware Ensemble Conditioning for Data-Limited Emulation of Extreme Events

    [https://arxiv.org/abs/2609.30746](https://arxiv.org/abs/2609.30746)

    该论文提出一种机制感知的即插即用集合条件化框架，利用受推动粗糙集合的协方差作为局部不稳定性的无需雅可比矩阵的代理，通过小型FiLM模块将集合几何统计信息注入骨干网络，从而在数据受限条件下实现对混沌系统极端事件的准确仿真。

    

    混沌系统中的极端事件难以从短轨迹中学习，因为它们受瞬态有限时间不稳定性控制，而非频繁观测到的整体动力学。我们提出了一种机制感知的条件化即插即用框架，将受推动的粗糙集合转化为局部不稳定性几何结构的非侵入式传感器。在小噪声区间内，集合协方差汇聚了控制局部不稳定性的相同有限时间变形核，为同步粗糙轨迹周围的局部放大结构提供了一个无需雅可比矩阵的代理。一个小型FiLM模块将这种集合几何的统计信息注入到原本保持不变的骨干网络中，同时不改变粗糙模拟器本身。我们在两种不同的流程中演示了这一接口：用于受控低维混沌系统的Transformer式残差注意力校正器，以及用于top（原文截断）的概率循环STORN校正器。

    arXiv:2609.30746v1 Announce Type: new  Abstract: Extreme events in chaotic systems are difficult to learn from short trajectories because they are controlled by transient finite-time instability rather than by frequently observed bulk dynamics. We propose a mechanism-aware conditioning plug-in framework that turns a nudged coarse ensemble into a non-intrusive sensor of local instability geometry. In the small-noise regime, the ensemble covariance aggregates the same finite-time deformation kernels that govern local instability, providing a Jacobian-free proxy for the local amplification structure around a synchronized coarse trajectory. A small FiLM module injects statistics of this ensemble geometry into an otherwise unchanged backbone while leaving the coarse simulator unchanged. We demonstrate this interface in two distinct pipelines: a Transformer-style residual-attention corrector for a controlled low-dimensional chaotic system and a probabilistic recurrent STORN corrector for top
    
[^118]: 超越平均注意力：面向KV缓存淘汰的多样性感知分层评分

    Beyond Mean Attention: Diversity-Aware, Layer-Wise Scoring for KV Cache Eviction

    [https://arxiv.org/abs/2609.30738](https://arxiv.org/abs/2609.30738)

    该论文提出在注意力均值之外加入注意力离散度与冗余惩罚（类MMR多样性）的统一KV缓存淘汰评分，并发现仅用一个全局多样化常数即可在多数LongBench数据集上带来提升，而无需按层或按数据集精细搜索。

    

    SnapKV和PyramidKV等KV缓存淘汰方法仅依据较小观察窗口上的平均注意力对token进行排序。我们研究了一个统一的评分：$\mu_i+\lambda_1\sigma_i+\lambda_2\mathrm{corr}(i,S)$，在平均注意力之外引入了跨窗口查询的注意力离散度以及相对于已选token的冗余度。当$\lambda_2<0$时，该评分会惩罚与已选token的相似性，类似于最大边际相关性（MMR），且无需额外的前向传播。为检验这种相关性-多样性平衡是否应随网络深度变化，我们将固定全局系数与三段式及二次型深度轮廓进行比较，仅在开发集上于$\sinh$重参数化下搜索这些深度轮廓。在全部16个英文LongBench数据集上使用Mistral-7B、每层预算为64的设置下，单一的全局多样化常数改进了16个数据集中的13个（宏观平均提升1.1）；该增益在预算为32时依然保持，在预算为128时收窄。逐数据集搜索未发现可检测的（分层模式差异）。

    arXiv:2609.30738v1 Announce Type: new  Abstract: KV cache eviction methods such as SnapKV and PyramidKV rank tokens solely by mean attention over a small observation window. We study a unified score, $\mu_i+\lambda_1\sigma_i+\lambda_2\mathrm{corr}(i,S)$, adding attention dispersion across window queries and redundancy relative to selected tokens. For $\lambda_2<0$, the score penalizes similarity to selected tokens as in maximal marginal relevance (MMR), without extra forward passes. To test whether this relevance-diversity balance should vary with depth, we compare fixed global coefficients with three-segment and quadratic profiles. Only these depth profiles are searched on a development split under a $\sinh$ reparameterization. On all 16 English LongBench datasets with Mistral-7B at a budget of 64 entries per layer, a single global diversification constant improves 13 of 16 datasets (macro +1.1); the gain holds at budget 32 and narrows at 128. Per-dataset search finds no detectable la
    
[^119]: TR-SSQP：一种用于重尾噪声下约束随机优化的信赖域方法

    TR-SSQP: A Trust-Region Method for Constrained Stochastic Optimization under Heavy-Tailed Noise

    [https://arxiv.org/abs/2609.30732](https://arxiv.org/abs/2609.30732)

    本文提出TR-SSQP方法，在随机序列二次规划框架下通过法向-切向分解与归一化信赖域半径设计，首次为重尾噪声下带等式约束的随机优化问题提供了理论保证。

    

    我们考虑具有确定性等式约束的随机非线性优化问题。虽然无约束随机优化已被充分理解，但在约束设定下最优性与可行性之间的相互作用带来了重大挑战。此外，现有的约束随机方法的理论保证主要依赖于有界方差假设，使得重尾噪声情形在很大程度上尚未被探索。为了填补这一空白，我们在随机序列二次规划框架内提出了一种新颖的信赖域方法，称为TR-SSQP。我们的方法在步长计算中采用法向-切向分解来平衡最优性与可行性。此外，我们在信赖域半径的设计中引入了归一化机制，并结合Polyak动量进行梯度估计，从而在不使用梯度裁剪的情况下确保稳定的更新。当信赖域半径与……（摘要在此处被截断）

    arXiv:2609.30732v1 Announce Type: cross  Abstract: We consider stochastic nonlinear optimization problems with deterministic equality constraints. While unconstrained stochastic optimization is well understood, the interplay between optimality and feasibility in the constrained setting poses significant challenges. Moreover, existing theoretical guarantees for constrained stochastic methods predominantly rely on bounded-variance assumptions, leaving the heavy-tailed noise regime largely unexplored. To address this gap, we propose a novel trust-region method within the stochastic sequential quadratic programming framework, termed TR-SSQP. Our method employs a normal-tangential decomposition in the step computation to balance optimality and feasibility. In addition, we incorporate a normalization mechanism in the design of the trust-region radius, together with Polyak momentum for gradient estimation, ensuring stable updates without gradient clipping. When the trust-region radius and the
    
[^120]: 输入层饥饿：为什么逐层剪枝会破坏物联网入侵检测器

    Input-Layer Starvation: Why Per-Layer Pruning Breaks IoT Intrusion Detectors

    [https://arxiv.org/abs/2609.30729](https://arxiv.org/abs/2609.30729)

    该论文揭示了均匀逐层剪枝会使IoT入侵检测器的输入层“饥饿”（46%的第一层滤波器失去全部输入权重、归一化统计量大幅偏移），导致被整体准确率掩盖的近半类别性能崩塌，并提出低开销的预防与修复方法。

    

    面向小型物联网设备的入侵检测器通常通过剪枝进行压缩，并以整体准确率加以评估。我们证明这种做法掩盖了严重的类级别失效，找出了其成因，并给出了低开销的预防与修复方案。在CICIoT2023数据集上，一个两层卷积检测器在80%稀疏度下经均匀逐层幅度剪枝后，准确率下降16个百分点，而宏平均F1（各类别F1的均值，基于五个独立训练的模型）损失近半（从0.542降至0.271）；34个类别中有17个受到实质性损害。剩余权重数量无法解释这一现象：被剪枝到相同或更少权重的感知机与transformer最多仅损失0.096。真正的元凶是第一层：它仅有192个权重，均匀剪枝后只剩38个，其64个滤波器中有46%失去了全部输入权重，在这种“输入饥饿”状态下进行微调，会使第一归一化层中少数幸存通道的滑动均值偏移高达0.8个标准差。

    arXiv:2609.30729v1 Announce Type: cross  Abstract: Intrusion detectors for small Internet-of-Things (IoT) devices are usually compressed by pruning and judged by overall accuracy. We show that this hides a severe class-level failure, find its cause, and give low-overhead prevention and repair. On CICIoT2023, a two-layer convolutional detector pruned with uniform layer-wise magnitude pruning at 80% sparsity loses 16 points of accuracy but half of its macro-F1, the mean per-class F1 (0.542 to 0.271 over five independently trained models); 17 of 34 classes are materially damaged. Remaining weight count does not explain it: a perceptron and a transformer pruned to the same or fewer weights lose at most 0.096. The first layer does. It has 192 weights; uniform pruning leaves 38, 46% of its 64 filters lose every input weight, and fine-tuning under that starvation leaves the running means of the first normalisation layer displaced by up to 0.8 standard deviations in a few surviving channels, o
    
[^121]: 当10,000个窗口并非10,000次检验：滑动窗口时间序列分类中的统计置信度审计

    When 10,000 Windows Are Not 10,000 Tests: Auditing Statistical Confidence in Sliding-Window Time-Series Classification

    [https://arxiv.org/abs/2609.30721](https://arxiv.org/abs/2609.30721)

    本文揭示滑动窗口分类中大量重叠测试窗口并非独立样本，提出将三种泛化主张映射到依赖稳健推断（如Bartlett-HAC）的实用审计方法，发现测试行数增长近四倍仅带来约1.75-1.94倍的方差等效信息增长。

    

    滑动窗口分类器通常在数千个重叠的测试窗口上进行评估，尽管相邻预测共享观测数据，并且嵌套于各录制记录和受试者之中。受试者不重叠的评估可以防止一种形式的数据泄露，但并不能使这些测试窗口相互独立。我们提出一种实用的审计方法，将三种泛化主张——在已观测录制上的性能、来自已观测受试者的未来录制、以及来自未见受试者的性能——映射到明确的聚合规则和成熟的依赖稳健推断方法上。在75%重叠率下，受控模拟表明，基于IID假设的观测录制推断的I型错误率为16.9%，而以会话为中心的Bartlett-HAC推断为7.2%：这带来了实质性改进，但仍存在残余的校准偏差。对冻结的WISDM和HARTH预测的审计显示，测试数据行数增长近四倍，仅带来1.75至1.94倍的方差等效信息增长。在该重叠率下，基于固定录制的配对准确率差异置信区间……

    arXiv:2609.30721v1 Announce Type: new  Abstract: Sliding-window classifiers are often evaluated on thousands of overlapping test windows, even though neighboring predictions share observations and remain nested within recordings and subjects. Subject-disjoint evaluation prevents one form of leakage but does not make those test windows independent. We present a practical audit that maps three claims - performance on observed recordings, future recordings from observed subjects, and unseen subjects - to explicit aggregation rules and established dependence-robust inference. At 75% overlap, controlled simulations give 16.9% Type-I error for IID observed-record inference and 7.2% for session-centered Bartlett-HAC: a substantial improvement with residual miscalibration. Audits of frozen WISDM and HARTH predictions show that nearly fourfold growth in test rows yields only 1.75-1.94-fold variance-equivalent information growth. At that overlap, fixed-record paired Accuracy-difference intervals
    
[^122]: NEMSim：通过可执行的事件机制先验学习控制条件下的多事件物理动力学

    NEMSim: Learning Control-Conditioned Multi-Event Physical Dynamics via Executable Event-Mechanism Priors

    [https://arxiv.org/abs/2609.30718](https://arxiv.org/abs/2609.30718)

    提出NEMSim框架，将预定义的事件-属性描述编译为可执行的转移结构，融合事件机制先验与神经网络学习，实现对控制条件下多事件物理系统跨广泛控制空间和长轨迹的高效高保真模拟。

    

    控制条件下多事件物理系统的高保真模拟计算代价高昂，尤其是在广阔的控制空间和长轨迹场景下。在这类系统中，宏观演化由局部离散事件涌现而来，这些事件的强度和效应依赖于过程控制与不断演化的局部状态，而可用的系统知识通常以事件-属性描述的形式表达。纯数据驱动的代理模型只能从有限的轨迹覆盖中推断这些事件效应，这可能阻碍模型对未见控制情形的泛化能力。而现有的物理引导方法则主要基于方程级约束或可微求解器，而非离散事件规则先验。为此，我们提出了NEMSim（神经事件机制模拟器），它将预定义的事件-属性描述编译为可执行的转移结构，将控制依赖的事件强度与先验引导的机制……

    arXiv:2609.30718v1 Announce Type: new  Abstract: High-fidelity simulation of control-conditioned multi-event physical systems is computationally expensive, especially across broad control spaces and long trajectories. In these systems, macroscopic evolution emerges from localized discrete events whose intensities and effects depend on process controls and evolving local states, while the available system knowledge is typically expressed as event-attribute descriptions. Purely data-driven surrogates must infer these event effects from limited trajectory coverage, which can hinder generalization to unseen control regimes. Physics-guided methods instead primarily build on equation-level constraints or differentiable solvers rather than discrete event-rule priors. We therefore propose NEMSim (Neural Event-Mechanism Simulator), which compiles predefined event-attribute descriptions into an executable transition structure linking control-dependent event intensities, prior-guided mechanism at
    
[^123]: 基于经验局部化变形Bregman散度的非归一化离散模型参数估计

    Parameter Estimation for Unnormalized Discrete Models via Empirically Localized Deformed Bregman Divergence

    [https://arxiv.org/abs/2609.30713](https://arxiv.org/abs/2609.30713)

    本文提出将经验局部化技术与变形Bregman散度相结合来估计非归一化离散模型的参数，在大幅降低归一化常数计算成本的同时，可通过选择变形方式使估计器具备有效性或抗离群噪声鲁棒性等良好统计性质。

    

    概率模型的参数估计是机器学习领域的一项重要任务。对于离散变量的模型，其归一化常数的计算有时非常困难，已有大量研究致力于避免归一化常数的计算。在本文中，我们通过结合经验局部化技术与变形Bregman散度来应对这一难题。经验局部化技术能够大幅降低归一化常数计算的计算成本；此外，适当选择Bregman散度的变形方式，可以为所提出的估计器赋予多种良好的统计性质，例如有效性或对离群噪声的鲁棒性。

    arXiv:2609.30713v1 Announce Type: new  Abstract: Estimation of parameter of probabilistic models is an important task in the field of machine learning.For models of discrete variables, calculation of the normalization constant of model is sometimes difficult and a lot of researches have been done to avoid the calculation of the normalization constant. In this paper, we tackle with the difficulty by combining a technique of empirical localization and a deformed Bregman divergence.The technique of empirical localization makes it possible to drastically reduce computational cost of the calculation of the normalization constant, and in addition, appropriate choice of the deformation for the Bregman divergence can invest the proposed estimator with various kinds of favorable statistical properties, such as efficiency or robustness against outlier noise.
    
[^124]: LUMO（轻量级统一多语言编排器）：一个保护隐私的离线语音助手

    LUMO (Lightweight Unified Multilingual Orchestrator): A Privacy Preserving Offline Voice Assistant

    [https://arxiv.org/abs/2609.30692](https://arxiv.org/abs/2609.30692)

    该论文提出了LUMO，一个面向边缘计算环境的完全离线、保护隐私的轻量级多语言语音助手，它通过将本地ASR、4位GGUF量化的大语言模型和TTS集成到统一流水线中，可在8 GB内存的树莓派5等资源受限设备上实现实用的端到端语音交互。

    

    可靠的语音交互在互联网连接受限且对隐私要求严格的环境中至关重要。然而，大多数现有的语音助手依赖于基于云的服务，这导致了延迟问题、对互联网访问的依赖以及隐私漏洞。本研究提出了LUMO（轻量级统一多语言编排器），一个为边缘计算环境设计的保护隐私的离线语音助手。该系统将本地自动语音识别（ASR）、本地部署的量化大语言模型（LLM）和文本转语音（TTS）合成集成到一个完全离线的流水线中，运行在配备8 GB内存的树莓派5上。为了在资源受限的硬件上实现高效运行，该语言模型采用4位GGUF量化进行压缩，在保持实用对话能力的同时降低了内存占用。现有的边缘语音助手（如Mycroft）仅提供部分离线功能，而LUMO则实现了端到端的完全离线语音交互。

    arXiv:2609.30692v1 Announce Type: new  Abstract: Reliable voice interaction is essential in environments with limited internet connectivity and strong privacy. However, most existing voice assistants depend on cloud-based services, which leads to latency issues, dependency on internet access, and privacy vulnerabilities. This research presents LUMO (Lightweight Unified Multilingual Orchestrator), a privacy preserving offline voice assistant designed for edge computing environments. This system integrates local Automatic Speech Recognition (ASR), locally deployed quantized Large Language Model (LLM), and Text-to-Speech (TTS) synthesis into a fully offline pipeline running on a Raspberry Pi 5 with 8 GB RAM.   To enable efficient operation on resource constrained hardware, the language model is compressed using 4-bit GGUF quantization, which reduces memory usage while preserving practical conversational capability. Existing edge based voice assistants Mycroft provides partial offline func
    
[^125]: 面向动态无人机网络的威胁感知节能部署：一种多智能体强化学习方法

    Threat-Aware Energy-Efficient Deployment for Dynamic UAV Networks: A Multi-Agent RL Approach

    [https://arxiv.org/abs/2609.30690](https://arxiv.org/abs/2609.30690)

    提出了一种威胁感知的无人机网络节能部署三步框架，结合威胁感知K均值聚类、最优匹配和MATD3多智能体强化学习，在实现零安全违规的同时最大化全局能效并加速收敛。

    

    在威胁多发环境中确保运行安全，对于作为空中基站的多无人机网络而言仍然是一项关键挑战。本文提出了一个高效框架，通过威胁感知聚类和基于奖励的安全约束机制来保障安全运行，同时最大化全局能效（EE）。该框架分三步执行：首先，威胁感知K均值（TAKM）算法确定所需的最少无人机数量并计算安全的初始部署位置；其次，最优匹配阶段将物理无人机分配到这些质心位置，以最小化能量消耗；第三，威胁感知多智能体双延迟深度确定性策略梯度（MATD3）算法动态优化轨迹、功率和用户关联。仿真结果表明，在所考虑的场景中，所提出的框架实现了零安全违规，同时相比（基线方法）取得了更优的能效和更快的收敛速度。

    arXiv:2609.30690v1 Announce Type: cross  Abstract: Ensuring operational safety in threat-prone environments remains a critical challenge for multi-UAV networks serving as aerial base stations. This paper proposes an efficient framework to maximize global energy efficiency (EE) while promoting safe operation through threat-aware clustering and reward-based safety enforcement. The proposed framework is executed in three steps. First, a threat-aware K-means (TAKM) algorithm determines the minimum required UAVs and computes safe initial placements. Second, an optimal matching stage assigns physical UAVs to these centroids to minimize energy expenditure. Third, a threat-aware multi-agent twin delayed deep deterministic policy gradient (MATD3) algorithm dynamically optimizes trajectories, power, and user associations. Simulation results show that the proposed framework achieves zero observed safety violations in the considered scenarios while achieving superior EE and faster convergence than
    
[^126]: 单变量深度学习在有效波高预测中的局限性研究

    On the Limits of Univariate Deep Learning for Significant Wave Height Forecasting

    [https://arxiv.org/abs/2609.30688](https://arxiv.org/abs/2609.30688)

    该研究通过对五种深度学习架构和多种上下文长度的系统超参数搜索发现，单变量深度学习模型在有效波高预测中性能差异极小（远小于数据集本身变化的影响），且在极端波浪条件下表现不如简单的持续性预测，揭示了单变量深度学习方法在该任务上的局限性。

    

    本研究针对NDBC浮标41009的单站有效波高（Hs）预测，在五种深度学习架构（DLinear、LSTM、PatchTST、ResAttLstm和Mamba2）和九种上下文长度（1-168小时）之间进行了系统的超参数搜索，随后在包含47个浮标、37年数据的语料库上对最佳配置进行了重新评估。在多浮标评估中，五种模型家族收敛到相同的性能水平（家族间标准差为0.0014平方米，仅占总均值的0.8%），这一差异远小于浮标语料库之间4.83倍的跨数据集MSE变化。所有多浮标试验均优于持续性预测（平均技巧+0.062），但没有一种架构始终优于其他架构。在单浮标实验中，预测技巧在12-24小时上下文长度时达到峰值，但有五个试验低于持续性预测；各家族的Q4/Q3测试MSE比率介于2.4至2.6之间，且对于最极端的1%波浪，深度模型的表现不如持续性预测。这些发现与……一致（原文在此处截断）。

    arXiv:2609.30688v1 Announce Type: cross  Abstract: This study conducts a systematic hyperparameter search across five deep learning architectures, DLinear, LSTM, PatchTST, ResAttLstm, and Mamba2, and nine context lengths (1-168 h) for single-station significant wave height (Hs) forecasting on NDBC buoy 41009, followed by re-evaluation of the best configurations on a 47-buoy, 37-year corpus. The five families converge to a common performance level on the multi-buoy evaluation (between-family SD = 0.0014 m^2, 0.8% of the grand mean), a spread dwarfed by the 4.83x cross-dataset MSE shift between buoy corpora. All multi-buoy trials beat persistence (mean skill +0.062), but no architecture consistently outperforms the others. On the single-buoy experiment, skill peaks at 12-24 h where five trials fall below persistence, per-family Q4/Q3 test MSE ratios range from 2.4 to 2.6, and deep models underperform persistence for the most extreme 1% of waves. These findings are consistent with the int
    
[^127]: PixSim：一个在分析师容量约束下对即时支付欺诈、追回与拦截进行建模的校准开源模拟器

    PixSim: a calibrated open-source simulator of instant-payment fraud, recovery and interdiction under analyst capacity constraints

    [https://arxiv.org/abs/2609.30684](https://arxiv.org/abs/2609.30684)

    PixSim是一个基于巴西央行开放数据校准的开源模拟器，首次联合建模了Pix即时支付的不可逆结算、监管追回机制、资金下游分散与分析师容量受限的审核决策，并在模型冻结后成功再现了2025年真实的欺诈追回率。

    

    巴西的Pix支付系统每月结算约59亿笔即时且不可逆转账。欺诈转账只有在资金仍停留在可追踪账户中时才可能被追回，而2025年巴西央行的追回机制（MED）仅返还了被受理争议金额的9%。因此，拦截必须在结算之前发生，即在有限的分析师团队和监管冻结时间窗口下，将每笔交易分流为放行、人工审核或拦截。据我们所知，目前尚无公开的模拟器能够联合建模不可逆结算、受监管的追回机制、资金下游分散以及容量受限的人工审核。我们提出了PixSim，一个包含上述所有要素的Pix支付轨道开源模拟器，该模拟器基于巴西央行（Banco Central do Brasil）的开放数据进行校准，每个参数均有来源、根据已发布的可观测数据进行校准，或被明确注册为假设。在模型冻结后，全规模运行再现了2025年的追回率（误差在0.006以内）及其分解（摘要在此处截断）。

    arXiv:2609.30684v1 Announce Type: cross  Abstract: Brazil's Pix settles about 5.9 billion instant, irreversible transfers a month. A fraudulent transfer can be recovered only while the funds remain in a traceable account, and in 2025 the Central Bank's recovery mechanism (MED) returned 9% of accepted contested value. Interdiction therefore has to happen before settlement, by routing each transaction to pass, human review or block, under a finite analyst team and a regulatory hold window. To our knowledge no public simulator jointly models irreversible settlement, a regulated recovery mechanism, downstream fund dispersal and capacity-constrained review. We present PixSim, an open-source simulator of the Pix rail with these elements, calibrated to Banco Central do Brasil open data, with every parameter sourced, calibrated to one published observable, or registered as an assumption. With the model frozen, full-scale runs reproduce the 2025 recovery rate within 0.006 and its decomposition 
    
[^128]: StarWM：用于鲁棒世界模型的自监督训练注意力路由

    StarWM: Self-Supervised Trained Attention Routing for Robust World Models

    [https://arxiv.org/abs/2609.30667](https://arxiv.org/abs/2609.30667)

    提出StarWM，利用自监督动态训练的交叉注意力模块决定重构区域，结合双流解码器与停止梯度屏障，使世界模型既能忠实学习相关动态，又不受任务无关内容的干扰。

    

    一个鲁棒的世界模型必须在忠实捕捉环境动态与抽象掉无关内容之间取得平衡。基于重构的世界模型虽然能提供忠实的监督信号，但在视觉任务中它们按照像素面积而非动态相关性来分配表示容量，这可能导致与任务无关的内容主导所学习到的表示。相反，无重构的方法虽然避免了这种偏差，但存在丢弃潜在相关信息的风险。我们提出了StarWM，它使用一个在自监督动态上训练的交叉注意力模块来决定重构应用于哪些区域。随后，双流解码器将重构限制在注意力所关注的区域，并通过停止梯度屏障防止两个目标之间的相互干扰。这些组件使得重构能够监督所关注区域的视觉内容，而不会让无预测性的信息污染潜在表示。

    arXiv:2609.30667v1 Announce Type: cross  Abstract: A robust world model must strike the balance between faithfully capturing environmental dynamics and abstracting away from irrelevant content. While reconstruction-based world models ensure faithful supervision, they misallocate representational capacity by pixel area rather than dynamics relevance for visual tasks, which can cause task-irrelevant content to dominate the learned representation. Alternatively, reconstruction-free methods avoid this bias but risk discarding possibly relevant information. We propose StarWM, which uses a cross-attention module trained on self-supervised dynamics to decide where reconstruction applies. A dual-stream decoder then restricts reconstruction to the attended regions, with stop-gradient barriers preventing interference between the two objectives. These components allows reconstruction to supervise the visual content of attended regions without contaminating the latent with non-predictive informati
    
[^129]: 浅层ReLU网络中的总体损失：偏置与临界点族

    Population loss in shallow ReLU networks: Bias & families of critical points

    [https://arxiv.org/abs/2609.30661](https://arxiv.org/abs/2609.30661)

    本文利用Owen T函数推导出适用于带偏置浅层ReLU网络的学生-教师核模型总体损失解析公式，证明已知伪极小值族可扩展至带偏置网络，且添加偏置总能严格降低损失、对损失景观的改变相对温和。

    

    本文提出的主要结果是一个关于学生-教师核模型中总体损失的公式，该公式适用于带偏置的浅层ReLU网络。这扩展了Choo和Saul（2009）以及Brutzkus和Globerson（2017）先前的相关工作。该公式的关键在于使用了Owen T函数。文中给出了T函数的必要理论，并基于Komelj（2023）的算法，利用MPFR实现了T函数的高精度编码，可应要求提供。研究表明，过去Arjevani及其合作者论文中描述的各类伪极小值族可以扩展到带偏置的网络中，并且加入偏置时损失总是严格减少。加入偏置所引起的损失景观几何变化似乎相对温和。出于篇幅原因，本文仅描述了最简单的例子，其中假设输入数量等于神经元数量。

    arXiv:2609.30661v1 Announce Type: new  Abstract: The main result presented is a formula for the population loss in the student-teacher kernel model that is applicable to shallow ReLU networks with bias. This extends previous work of Choo and Saul (2009) and Brutzkus and Globerson (2017). The formula makes essential use of Owen's T-function. The necessary theory of the T-function is given and a high precision coding using MPFR for the T-function, based on an algorithm of Komelj (2023), is available on request. It is shown that various families of spurious minima described in past papers of Arjevani and the author extend to biased networks and that the loss is always strictly decreased when bias is added. The change in landscape geometry caused by adding bias appears to be relatively mild. Only the simplest examples are described in this paper where it is assumed that the number of inputs is equal to the number of neurons (this restriction is for reasons of length). A review of relevant 
    
[^130]: DiffusionShadow：基于扩散模型的神经体绘制阴影缓存

    DiffusionShadow: Diffusion-based Shadow Caching for Neural Volume Rendering

    [https://arxiv.org/abs/2609.30658](https://arxiv.org/abs/2609.30658)

    本文提出DiffusionShadow框架，利用扩散模型将大量预计算的阴影隐式神经表示压缩为单一模型，在运行时高效重建阴影，从而实现带阴影效果的实时神经体绘制。

    

    隐式神经表示（INRs）因其紧凑性和对大型数据集的可扩展性，在科学可视化领域获得了广泛关注，使其非常适合与直接体绘制（DVR）集成。然而，在包含阴影等高级光照效果的情况下，INR的实时体绘制计算成本依然很高，因为通过光线步进评估阴影项代价高昂。另外，为众多光照方向预计算并存储阴影在内存和存储方面都是不可行的。为了解决这一问题，我们提出了一种基于扩散的阴影缓存框架，将大量预先计算的阴影INR压缩到单个扩散模型中。我们的方法并不专注于泛化到未见过的方向，而是在运行时有效地记忆和重建一组密集的预训练光照条件。我们首先将阴影系数体积集合编码为阴影INR，然后训练……

    arXiv:2609.30658v1 Announce Type: cross  Abstract: Implicit neural representations (INRs) have gained momentum in scientific visualization due to their compactness and scalability to large datasets, making them well suited for integration with direct volume rendering (DVR). However, real-time volume rendering of INR with advanced illumination effects, such as shadows, remains computationally expensive, as evaluating shadow terms via ray marching is costly. Alternatively, precomputing and storing shadows for many lighting directions is prohibitive in both memory and storage. To address this, we introduce a diffusion-based shadow caching framework that compresses a vast set of pre-calculated shadow INRs into a single diffusion model. Rather than focusing on generalizing to unseen directions, our method effectively memorizes and reconstructs a dense set of pre-trained lighting conditions on the fly. We first encode a collection of shadow coefficient volumes as shadow INRs, and then train 
    
[^131]: 交互式智能体中的因果保留：接口因式分解与选择性适应

    Causal Retention in Interactive Agents: Interface Factorization and Selective Adaptation

    [https://arxiv.org/abs/2609.30650](https://arxiv.org/abs/2609.30650)

    本文提出“因果保留”理论，证明冻结的学习状态能否正确响应独立于训练的机制探针取决于学习接口纤维与探针答案纤维之间的包含关系，并据此构建 Causal Core 系统，通过证据门控写入与选择性适应等机制实现无误差的目标更新。

    

    任务性能并不决定智能体保留哪种干预机制。我们研究因果保留：即一个冻结的学习状态能否回答一个独立于训练而固定的机制探针映射，该映射涵盖动作、上下文、直接目标、价值和延迟等维度。对于有限的结构因果模型类，最优探针误差是一个贝叶斯决策风险；当且仅当每个学习接口纤维都落在某一探针答案纤维之内时，该误差恰好消失，且任何通过对该接口进行后处理得到的状态都继承相同的下界。一个后验覆盖定理刻画了预算受限的重测试，而一个精确的编辑分解表明移位集是无误差目标更新的唯一支撑。Causal Core 通过证据门控写入、读出过滤、时间信用分配、隐藏上下文设置和局部诊断更新来实现这些条件。实验涵盖有限因果系统、连续模拟器、官方T……（原文摘要在此处不完整）

    arXiv:2609.30650v1 Announce Type: cross  Abstract: Task performance need not determine which intervention mechanism an agent retains. We study causal retention: whether a frozen learned state answers a mechanism-probe map fixed independently of training, including action, context, direct target, value, and delay. For finite structural causal model classes, the optimal probe error is a Bayes decision risk. It vanishes exactly when every learning-interface fiber lies within one probe-answer fiber; any state obtained by post-processing that interface inherits the same lower bound. A posterior-coverage theorem characterizes budgeted retesting, while an exact edit decomposition shows that the shifted set is the unique support of an error-free target update. Causal Core implements these conditions through evidence-gated writing, readout filtering, temporal credit, hidden-context setup, and local diagnostic updates. Experiments cover finite causal systems, continuous simulators, an official T
    
[^132]: MARCEDES：基于连续优化的非高斯性条件下基于评分的因果发现方法

    MARCEDES: Score-based causal discovery under non-Gaussianity with continuous optimization

    [https://arxiv.org/abs/2609.30643](https://arxiv.org/abs/2609.30643)

    提出了名为MARCEDES的基于评分的因果发现方法，通过引入平均绝对残差风险、行稀疏惩罚和软DAG约束，实现了非高斯误差下因果DAG结构的高效连续优化学习。

    

    我们考虑学习与非高斯误差的结构方程模型（SEM）相对应的潜在因果有向无环图（DAG）结构的问题。受一个故意误设的、所有误差均为拉普拉斯分布的非高斯SEM的启发，我们首先引入了定义在所有实矩阵空间上的平均绝对残差风险，并证明在渐近意义上，真实加权因果DAG矩阵的风险严格小于任何其他矩阵的风险。然而，为了增强通用性并考虑高维和有限样本的设置，我们进一步引入了针对每一行的稀疏性惩罚以及软DAG约束，从而在实矩阵空间上推导出一个连续的评分函数。据此，我们提出了一种基于评分的DAG学习方法，命名为MARCEDES，将其表述为一个无约束的评分最小化问题，该问题可以通过基于梯度的优化技术高效求解。

    arXiv:2609.30643v1 Announce Type: new  Abstract: We consider the problem of learning the underlying causal directed acyclic graph (DAG) structure corresponding to a structural equation model (SEM) with non-Gaussian errors. Motivated by an intentionally misspecified non-Gaussian SEM with all Laplace errors, we first introduce the mean absolute residual risk, defined over the space of all real matrices, and show that, asymptotically, the risk of the true weighted causal DAG matrix is strictly smaller than that of any other matrix. Nevertheless, to enhance generality and account for high-dimensional and finite-sample settings, we further incorporate row-specific sparsity penalties along with a soft DAG constraint to derive a continuous score function over the space of real matrices. Accordingly, we propose a score-based DAG learning method, named MARCEDES, formulated as an unconstrained score minimization problem, which can be efficiently solved using gradient-based optimization technique
    
[^133]: 语言模型的上下文绑定容量

    In-Context Binding Capacity in Language Models

    [https://arxiv.org/abs/2609.30634](https://arxiv.org/abs/2609.30634)

    该论文首次系统测量了语言模型在上下文中正确记住“值-实体”绑定关系的容量极限，发现该容量随参数规模按幂律增长（K₅₀ = cN^0.82），并揭示了干扰效应和预训练配方对测得容量的影响机制。

    

    一个语言模型在忘记哪个值属于哪个实体之前，能够记住多少个赋值关系？我们通过两组实验测量了这一极限：对12个参数量不超过30亿的模型绘制连续回忆曲线，并对30个参数量最高达120亿的开源模型进行阈值扫描。在连续曲线上，回忆率降至随机水平一半时的负载遵循 $K_{50}=cN^{\alpha}$，其中 $\alpha=0.820$，$R^2=0.73$。更广泛的阈值扫描显示出与预训练配方相关的八倍差异范围，但连续曲线在控制规模后并未检测到配方效应，且参与拟合的现代模型较少。我们推导了为什么即使负载相关的回忆曲线形状不变，干扰仍会通过降低单绑定回忆率来降低测得的容量。直接任务训练的表现也超过了外推得到的零样本定律，但由于测量标准不同，这一比较不能被解读为容量的提升。其形成时间遵循幂律……

    arXiv:2609.30634v1 Announce Type: new  Abstract: How many assignments can a language model recall before it loses track of which value belongs to which entity? We measure this limit using continuous recall curves for 12 models at or below 3B parameters and a threshold sweep over 30 open models up to 12B. On the continuous curves, the load at which recall falls halfway to chance follows $K_{50}=cN^{\alpha}$, with $\alpha=0.820$ and $R^2=0.73$. The broader sweep shows an eightfold range associated with pretraining recipe, although the continuous curves show no detectable recipe effect after controlling for scale, with few modern models in the fit. We derive why interference can lower measured capacity by reducing single-binding recall even when the load-dependent recall profile is unchanged. Direct task training also exceeds the extrapolated zero-shot law, but different measurement criteria prevent interpreting that comparison as a capacity gain. Its formation times follow a power-law fo
    
[^134]: 无需中心极限定理的稳定初始化

    Stable initialization without the CLT

    [https://arxiv.org/abs/2609.30633](https://arxiv.org/abs/2609.30633)

    提出了一种无需中心极限定理的均匀相位初始化方法，通过利用正弦函数的周期对称性消除分布近似并完全解耦各网络层，使未经调参的模型在图像和音频拟合等神经表征任务上即超越现有最先进方法。

    

    深度神经网络的成功训练高度依赖于初始权重的分布。如果权重过大，网络训练会发散；如果权重过小，模型则无法学习到特征。稳定初始化是在这两个极端之间的最优平衡。传统的随机网络理论使用中心极限定理（CLT）来控制神经元之间的依赖关系，但这会引入分布近似误差以及层间的耦合。针对使用正弦激活函数的网络，我们推导出了均匀相位初始化方法，该方法消除了分布近似，并完全解耦了各层。我们是首个利用正弦函数周期对称性的研究工作。采用均匀相位初始化训练的模型在图像拟合和音频拟合等神经表征任务上超越了当前最先进的方法。我们发现，我们未经调参的模型与以往经过最佳调参的基线模型相比也具有竞争力。

    arXiv:2609.30633v1 Announce Type: new  Abstract: Successful training of deep neural networks is highly dependent on the distribution of the initial weights. If the weights are too large, network training blows up; if they are too small, the model fails to learn features. Stable initialization is the optimal moderation between these two extremes. The conventional theory of random networks uses the Central Limit Theorem to control inter-neuron dependencies, which introduces distributional approximation error and coupling between layers. For networks with sine activations, we derive the uniform-phase initialization, which obviates distributional approximation and fully decouples the layers. Ours is the first work to use the sine function's periodic symmetry. Models trained with the uniform-phase initialization outperform the state of the art in neural representation tasks like image and audio fitting. We find that our untuned models are competitive with the best-tuned baselines from previ
    
[^135]: FRESHLATENT：面向资源受限具身视觉语言模型感知的信道感知潜在表征自适应

    FRESHLATENT: Channel-Aware Latent Adaptation for Resource-Constrained Embodied VLM Perception

    [https://arxiv.org/abs/2609.30629](https://arxiv.org/abs/2609.30629)

    FreshLatent 提出一种轻量级信道感知潜在适配器，在保持 VLM 冻结的前提下于无线损伤中训练功率归一化编解码器，使分割式 VLM 感知在 0 dB 低信噪比和最紧通信预算下 gIoU/cIoU 提升超过 20 个百分点。

    

    执行关键任务的无人机（UAV）在严格的机载资源和无线通信约束下，日益依赖分割式视觉语言模型感知。然而，传输中间特征所遭受的信道损伤会导致其与基于干净数据训练的分割式接口产生部署失配，而更强的信道感知编解码器又会带来可观的机载计算开销。我们提出 FreshLatent——一种轻量级的信道感知潜在适配器，它在无线信道损伤条件下训练功率归一化的编码器-解码器，同时保持周围的 VLM 完全冻结。我们围绕任务条件化的感知需求和嵌入式接口开销来构建部署框架，将信道质量与通信预算同感知仍然可用的运行条件联系起来。在 0 dB 信噪比和最紧张的通信预算下，FreshLatent 相比基于干净数据训练的分割式压缩，将 gIoU 和 cIoU 分别提升了 20.79 和 20.87 个百分点。在所评估的最不利信噪比条件下（0 dB

    arXiv:2609.30629v1 Announce Type: cross  Abstract: Mission-critical UAVs increasingly rely on split vision-language model (VLM) perception under tight onboard-resource and wireless-communication constraints. However, corruption of transmitted intermediate features creates a deployment mismatch for clean-trained split interfaces, while stronger channel-aware codecs can impose substantial onboard cost. We present FreshLatent, a lightweight channel-aware latent adapter that trains a power-normalized encoder-decoder through wireless corruption while keeping the surrounding VLM frozen. We formulate deployment around a mission-conditioned perception requirement and embedded interface cost, linking channel quality and communication budget to the operating conditions under which perception remains usable. At 0 dB and the tightest communication budget, FreshLatent improves gIoU and cIoU over clean split compression by 20.79 and 20.87 points, respectively. At the most adverse evaluated SNR (0 dB
    
[^136]: OpenHail：一个用于电动网约车车队控制的事件驱动Gymnasium环境

    OpenHail: An Event-Driven Gymnasium Environment for Electric Ride-Hailing Fleet Control

    [https://arxiv.org/abs/2609.30628](https://arxiv.org/abs/2609.30628)

    OpenHail是一个开源的事件驱动Gymnasium仿真环境，通过统一的观测-动作接口支持电动网约车车队在订单分配、车辆调度和充电方面的联合控制，为强化学习策略的训练与评估提供了考虑随机需求、车辆运行和充电设施容量约束的结构化仿真平台。

    

    近年来，机器学习策略在网约车车队控制领域引起了越来越多的关注。特别是强化学习，需要一个结构化的仿真环境，明确定义观测、动作、奖励和决策时刻，以用于训练和评估。对于电动车队而言，该环境还必须刻画随机需求、车辆运行与具有容量限制的充电基础设施之间的交互。我们提出了OpenHail，一个面向电动网约车车队联合控制的开源Gymnasium环境。其固定大小的观测-动作接口将请求分配、车辆重新定位和充电统一暴露给单一策略。该事件驱动仿真器刻画了带有取车截止时间的订单请求、车辆任务队列、电池动态，以及具有先进先出队列的有限容量充电设施。可配置的决策时刻机制将仿真器内部事件与策略干预解耦开来。

    arXiv:2609.30628v1 Announce Type: new  Abstract: Machine-learning policies have attracted increasing interest for ride-hailing fleet control in recent years. Reinforcement learning, in particular, requires a structured simulation environment that specifies observations, actions, rewards, and decision epochs for training and evaluation. For electric fleets, this environment must also capture the interaction among stochastic demand, vehicle operations, and capacitated charging infrastructure. We present OpenHail, an open-source Gymnasium environment for joint control of electric ride-hailing fleets. Its fixed-size observation--action interface exposes request assignment, repositioning, and charging to a single policy. The event-driven simulator represents requests with pickup deadlines, vehicle job queues, battery dynamics, and finite-capacity charging facilities with first-in--first-out queues. A configurable decision-epoch mechanism separates internal simulator events from policy inter
    
[^137]: 基于概率鲁棒性驱动且具可解释性的通用对抗扰动：针对基于深度强化学习的入侵检测系统的攻击

    Probabilistic Robustness-driven Universal Adversarial Perturbations with Explainability against Deep Reinforcement Learning-based Intrusion Detection System

    [https://arxiv.org/abs/2609.30605](https://arxiv.org/abs/2609.30605)

    该论文首次将概率鲁棒性作为显式优化目标，并结合可解释人工智能（XAI）引导扰动塑形，提出了针对基于深度强化学习的入侵检测系统的通用对抗扰动生成方法PX-UAP。

    

    arXiv:2609.30605v1 公告类型：新论文 摘要：深度强化学习（DRL）使入侵检测系统能够在动态网络环境中实现自适应检测，但同时也使入侵检测系统（IDS）暴露于对抗性威胁之下，例如通用对抗扰动（UAPs），即通过施加单一的、与输入无关的扰动来降低对各类流量的检测性能。概率鲁棒性（PR）作为一种事后评估指标，能够对对抗影响提供原理性的、群体层面的度量，这一概念与UAP的通用性目标在理念上高度契合，即PR量化了输入空间中误分类现象的普遍程度，使其成为指导UAP生成的天然信号。因此，我们提出了基于PR的UAP（PR-based UAP），这是首次将显式的PR驱动目标融入针对基于DRL的IDS的UAP生成之中。在该框架的基础上，我们进一步提出了PX-UAP，它利用可解释人工智能（XAI）在可解释性约束下引导扰动的塑形……（原文摘要在此处被截断）

    arXiv:2609.30605v1 Announce Type: new  Abstract: Deep reinforcement learning (DRL) enables adaptive intrusion detection in dynamic network environments but also exposes intrusion detection systems (IDS) to adversarial threats such as universal adversarial perturbations (UAPs), which apply a single input-agnostic perturbation to degrade detection performance across traffic. Probabilistic Robustness (PR), as a post-hoc evaluation metric, provides a principled, population-level measure of adversarial impact that conceptually aligns with the universality objective of UAPs, i.e., PR quantifies the prevalence of misclassification in the input space, making it a natural signal for guiding UAP generation. Hence, we propose PR-based UAP, which represents the first integration of an explicit PR-driven objective into generating UAPs against DRL-based IDS. Building on this formulation, we introduce PX-UAP, which leverages explainable artificial intelligence (XAI) to guide perturbation shaping unde
    
[^138]: QSV：面向球面格点上耦合四元数注意力的四元-球面-视觉模型

    QSV: Quat-Sphere-Vision for Coupled Quaternion Attention on Spherical Lattices

    [https://arxiv.org/abs/2609.30592](https://arxiv.org/abs/2609.30592)

    QSV用每个token一个可学习的单位四元数同时承担注意力打分（相对四元数实部作为logit）和特征传输（三明治乘积变换）两种功能，在同心Fibonacci球面的稀疏kNN图上传递消息，消融实验显示特征传输角色更关键——移除它会使CIFAR-10/100准确率下降约4个百分点，而用均匀平均替代学习到的注意力权重则影响甚微。

    

    在标准注意力机制中，三个独立学习的投影分别决定一个token对每个邻居的关注强度（$W_Q$、$W_K$），以及被关注的特征在聚合前如何被变换（$W_V$）。我们研究了Quat-Sphere-Vision（QSV），这是一种稀疏球面视觉模型，它用每个token单一的可学习单位四元数取代了这一投影三元组：相对四元数 $r_{ij} = q_i^{*} \otimes q_j$ 同时提供注意力logit $\operatorname{Re}(r_{ij})$ 以及三明治乘积形式的特征传输 $x \mapsto r_{ij} \otimes x \otimes r_{ij}^{*}$，消息在同心Fibonacci球面构成的稀疏kNN图上传递。仅改变目标组件的消融实验表明，这两种角色是不对称的：移除特征传输会使CIFAR-10和CIFAR-100上的测试准确率下降约四个百分点（每个CIFAR-100变体为单次运行），而将学习到的注意力权重替换为均匀平均则基本不影响其表现。

    arXiv:2609.30592v1 Announce Type: new  Abstract: In standard attention, three separately learned projections decide how strongly a token attends to each neighbor ($W_Q$, $W_K$) and how the attended features are transformed before aggregation ($W_V$). We study Quat-Sphere-Vision (QSV), a sparse spherical vision model that replaces this projection triple with a single learned unit quaternion per token: the relative quaternion $r_{ij} = q_i^{*} \otimes q_j$ supplies both the attention logit $\operatorname{Re}(r_{ij})$ and a sandwich-product feature transport $x \mapsto r_{ij} \otimes x \otimes r_{ij}^{*}$, with messages passed over sparse kNN graphs on concentric Fibonacci spheres. Ablations that change only the targeted component show the two roles to be asymmetric. Removing the transport reduces test accuracy by about four percentage points on CIFAR-10 and CIFAR-100 (single runs per CIFAR-100 variant), while replacing the learned attention weights with uniform averaging leaves it essent
    
[^139]: 可加密性作为一种坐标选择：量子神经网络的深度一同态联邦学习

    Encryptability As a Coordinate Choice: Depth-One Homomorphic Federated Learning of Quantum Neural Networks

    [https://arxiv.org/abs/2609.30581](https://arxiv.org/abs/2609.30581)

    该论文发现通过将变分量子电路的权重表示在单位四元数（自旋）坐标系中，群复合恰好成为二次双线性运算，使得量子神经网络能够在深度一的同态加密联邦学习框架下高效训练，每个权重仅需一个乘法层级、联邦平均零开销，并完全消除昂贵的自举操作。

    

    加密训练依赖于保持服务器端更新的低次性。这一约束传统上排除了权重位于紧致李群中的模型（尤其是变分量子电路，其中每个可训练权重都是一个 SU(2) 旋转）。若用欧拉角或离散字母表来表示，这些更新呈现为超越函数，历史上需要付出令人望而却步的代价：每个门需要一次客户端-服务器往返，或每个权重需要超过 25,000 次运算。这种代价严格来说只是坐标选择的人为产物。在单位四元数（自旋）坐标图表中，群复合恰好是双线性的（二次，系数取值于 {-1,0,+1}）。因此，在任何分层同态方案中，加密旋转更新只需一个乘法层级，而联邦平均的代价为零，从而完全消除了自举操作。这一独立于具体实现的代数性质已在两个密码学后端上得到验证，仅引入 0.0……

    arXiv:2609.30581v1 Announce Type: cross  Abstract: Encrypted training relies on keeping server-side updates low-degree. This constraint traditionally excludes models whose weights inhabit a compact Lie group (notably variational quantum circuits, where every trainable weight is an $\mathrm{SU(2)}$ rotation). Expressed in Euler angles or discrete alphabets, these updates appear transcendental, historically demanding prohibitive costs: one client--server round per gate, or upwards of $25{,}000$ operations per weight. This penalty is strictly an artefact of coordinates. In the unit-quaternion (spin) chart, group composition is exactly bilinear (degree two, with coefficients in $\{-1,0,+1\}$). Consequently, encrypted rotation updates cost one multiplicative level and federated averaging costs zero in any levelled homomorphic scheme, completely eliminating bootstrapping. This implementation-independent algebraic property is confirmed across two cryptographic backends, introducing only $0.0$
    
[^140]: 面向虚拟传感的神经算子节能运行

    Energy-efficient operation of neural operators for virtual sensing

    [https://arxiv.org/abs/2609.30580](https://arxiv.org/abs/2609.30580)

    该研究表明，在虚拟传感应用中，通过编译器冻结、主干复用与图重放等共享空间计算手段可显著降低神经算子推理的运行能耗，在15瓦固定时钟模式下相比即时执行节能约22%。

    

    虚拟传感需要在通常固定的几何结构上，根据不断变化的观测反复重建物理场。我们研究了共享空间计算如何在保留所选检查点及其评估预测的同时，降低这些更新的能耗。在换热器服务场景中，标准编译器冻结和显式主干复用相对于图重放带来了相近的运行能耗降低：在每秒1个请求时约为1%，在每秒40个请求时约为20%。在固定时钟频率的15瓦模式下，结合图重放的复用在完成相同请求序列时，比即时执行模式节能22.0%至22.5%（包含准备和等待时间）。DeepONet与傅里叶神经算子（FNO）的对照实验区分了可复用算术运算与启动开销各自的影响。准备、构建制品和替换工作进程会在重复推理之外带来额外成本。这些结果将算子结构与运行能耗联系起来。

    arXiv:2609.30580v1 Announce Type: new  Abstract: Virtual sensing repeatedly reconstructs physical fields from changing observations, often on a fixed geometry. We investigate how shared spatial computation reduces the energy of these updates while retaining the selected checkpoint and its evaluated predictions. In a heat-exchanger service, standard compiler freezing and explicit trunk reuse give similar operating energy reductions relative to graph replay: approximately 1% at one request per second and 20% at forty requests per second. In 15 W mode with fixed clocks, reuse with graph replay completes the same request sequence with 22.0 to 22.5% less energy than eager execution, including preparation and waiting. DeepONet and Fourier neural operator (FNO) controls distinguish the effects of reusable arithmetic and launch overhead. Preparation, artifact construction, and worker replacement add costs outside repeated inference. These results connect operator structure to operating energy 
    
[^141]: 小语言模型网状网络中通信的强化学习

    Reinforcement Learning of Communication in a Mesh of Small Language Models

    [https://arxiv.org/abs/2609.30578](https://arxiv.org/abs/2609.30578)

    提出TalkMesh去中心化小模型智能体网络，通过强化学习训练通信策略，让智能体之间传递解题关键提示并基于置信度进行修订，仅用三个智能体和最多六个输出就突破了独立采样多数投票的饱和限制。

    

    语言模型在测试阶段通过更多计算获得更高的准确性，但对独立样本进行多数投票会趋于饱和：随着样本数量的增加，投票结果会收敛到模型最常给出的答案。通信可以提供采样无法实现的功能：解决了某个问题的智能体可以将关键步骤传递给其他智能体。我们提出了TalkMesh，一个由小语言模型智能体组成的去中心化网状网络，它能够学习何时以及如何进行通信。每个智能体采样一个提案，并使用训练好的置信度头对其进行评分。置信度最高的智能体广播提示；置信度低于阈值的智能体进行修改，并保留每个得分超过原提案的修改结果。Gossip共识机制在无需协调者的情况下近似实现按置信度加权的投票。一个对话策略通过对修改后正确性变化进行组相对策略优化（GRPO）训练而得到，负责撰写提示和修改内容。使用三个智能体（总共最多生成六个输出），该网状网络达到了（摘要在此处被截断）

    arXiv:2609.30578v1 Announce Type: new  Abstract: Language models gain accuracy from more compute at test time, but majority voting over independent samples saturates: as samples grow, the vote converges to the model's most frequent answer. Communication can add what sampling cannot: an agent that solves a problem can pass the key step to the others. We present TalkMesh, a decentralized mesh of small language model agents that learns when and what to communicate. Each agent samples a proposal and scores it with a trained confidence head. The most confident agent broadcasts a hint; agents below a confidence threshold revise, keeping each revision that outscores its proposal. Gossip consensus approximates the vote weighted by confidence without a coordinator. A talk policy, trained with group relative policy optimization on the change in correctness after revision, writes hints and revisions. With three agents, which together generate at most six outputs, the mesh reaches the accuracy of 
    
[^142]: T-RoPE：面向序列推荐的时间感知旋转位置编码

    T-RoPE: Time-Aware Rotary Position Embedding for Sequential Recommendation

    [https://arxiv.org/abs/2609.30576](https://arxiv.org/abs/2609.30576)

    提出T-RoPE，一种时间感知的旋转位置编码，通过基于时间戳的角度、可学习时间系数和多尺度频率等机制打破标准RoPE的时间平移不变性，使序列生成式推荐模型能够捕捉时间间隔、行为周期与季节性等关键时间信息。

    

    大规模推荐系统日益采用大语言模型背后的序列生成式方案，将Transformer引入推荐领域，同时也沿用了为文本设计的组件，包括旋转位置编码。在语言模型中，RoPE对词元索引进行编码以实现相对位置推理，但在推荐场景中，交互索引仅记录事件顺序，无法反映流逝的时间、跨尺度的行为周期或日历相位。我们重新审视这一设计选择，提出T-RoPE，一种用于序列生成式推荐的时间感知RoPE，它以基于时间戳的角度、可学习的时间系数、多尺度频率库、偏移查询对齐以及非平稳的键旋转，替代仅基于索引的旋转。我们证明标准RoPE即使作用于时间戳，仍保持时间平移不变性，无法区分季节性上下文；而T-RoPE在保留RoPE的……

    arXiv:2609.30576v1 Announce Type: new  Abstract: Large-scale recommenders increasingly adopt the sequential generative recipe behind large language models, bringing the Transformer into recommendation along with design choices made for text, including Rotary Position Embedding (RoPE). In language models, RoPE encodes token indices for relative position reasoning, but in recommendation, an interaction index records only event order, saying nothing about elapsed time, behavioral cycles across scales, or calendar phase. We revisit this choice and propose T-RoPE, a time-aware RoPE for sequential generative recommendation that replaces index-only rotation with timestamp-based angles, learnable temporal coefficients, multiscale frequency banks, shifted query alignment, and non-stationary key rotation. We prove that standard RoPE, even on timestamps, remains time-translation invariant and cannot distinguish seasonal contexts, and that T-RoPE breaks this invariance while preserving the RoPE in
    
[^143]: 熵正则化：一种针对可验证示范的交叉熵免费修正

    Entropy Regularization: A Free Correction to Cross-Entropy for Verified Demonstrations

    [https://arxiv.org/abs/2609.30572](https://arxiv.org/abs/2609.30572)

    该论文指出在存在多个正确解的可验证任务中，用交叉熵模仿单一专家示范可能与验证器风险目标不一致，并提出以熵正则化作为“免费”修正来控制策略支持集、防止概率质量流向错误输出。

    

    大语言模型通常使用交叉熵（CE）在专家示范上进行后训练，即使其下游目标并非模仿所展示的解法，而是生成任何能被验证器接受的输出。这种错位出现在存在多个正确解法的可验证领域中，例如数学推理和代码生成，此时训练数据中每个问题可能只包含一个专家解法。我们证明，最小化交叉熵可能与最小化验证器风险不一致；两个策略可以对观测到的示范赋予相同的似然，同时在错误输出上分配不同的概率质量。我们通过一个学习理论反例将这一点形式化，在该反例中，交叉熵最小化会选择次优策略。我们发现，通过控制所学策略的支持集（support）可以解决这一问题，即阻止概率质量扩散到不受支持的输出上。由于支持集大小是非……（摘要在此处截断）

    arXiv:2609.30572v1 Announce Type: cross  Abstract: Large language models are often post-trained on expert demonstrations using cross-entropy (CE), even when the downstream objective is not to imitate the demonstrated solution but to produce any output accepted by a verifier. This mismatch is seen in verifiable domains with multiple correct solutions, such as mathematical reasoning and code generation, where training data may contain only one expert solution per problem. We show that minimizing cross-entropy can be misaligned with minimizing verifier risk; two policies can assign identical likelihood to the observed demonstrations while placing different probability mass on incorrect outputs. This is formalized through a learning-theoretic counterexample in which CE minimization selects a suboptimal policy. We identify that controlling the support of the learned policy can solve this problem by preventing probability mass from spreading to unsupported outputs. Since support size is non-
    
[^144]: 通过认知实验范式探究智能体记忆中的稳定性-可塑性权衡

    Probing Stability-Plasticity Tradeoffs in Agent Memory through Cognitive Experimental Paradigms

    [https://arxiv.org/abs/2609.30558](https://arxiv.org/abs/2609.30558)

    本文提出受认知科学启发的MemProbe框架，通过干扰、错误信息、巩固强度和再巩固窗口四种可复用的实验范式，超越传统最终答案准确率评估，系统诊断智能体记忆中稳定性与可塑性之间的权衡。

    

    智能体记忆系统越来越多地被用于维护长期用户偏好、任务状态和不断演变的事实，但当前的评估方法往往将记忆行为简化为最终答案的准确率。我们提出了MemProbe，一个受认知科学启发的框架，用于诊断智能体记忆中的稳定性-可塑性权衡。该框架的动机来自认知记忆研究的一个核心洞察：记忆是重构性的，并受到干扰、来源可靠性、强化和再激活等因素的塑造。MemProbe将这一洞察转化为四种可复用的实验范式（干扰、错误信息、巩固强度和再巩固窗口），这些范式操纵了记忆何时应该被更新、保留或视为不确定。该框架进一步将正确性分解为行为特征，揭示系统如何更新、保留、归因和时间性地组织信息。我们在56个剧集的诊断任务中实例化了这些范式（摘要在此处被截断）。

    arXiv:2609.30558v1 Announce Type: cross  Abstract: Agent memory systems are increasingly used to maintain long-term user preferences, task states and evolving facts, but current evaluations often collapse memory behavior into final-answer accuracy. We introduce MemProbe, a cognitive-science-inspired framework for diagnosing stability-plasticity tradeoffs in agent memory. The framework is motivated by a core insight from cognitive memory research: memory is reconstructive and shaped by interference, source reliability, reinforcement, and reactivation. MemProbe turns this insight into four reusable experimental paradigms (interference, misinformation, consolidation strength, and reconsolidation window) that manipulate when a memory should be updated, preserved, or treated as uncertain. It further decomposes correctness into behavioral profiles that reveal how systems update, preserve, attribute, and temporally organize information. We instantiate these paradigms in a 56-episode diagnosti
    
[^145]: 在线凸优化中带指示器切换代价的动态遗憾

    Dynamic Regret in Online Convex Optimization with Indicator Switching Costs

    [https://arxiv.org/abs/2609.30556](https://arxiv.org/abs/2609.30556)

    该论文首次针对带指示器切换代价的在线凸优化给出了动态遗憾保证，提出了一种由二进制时间尺度重启的惰性FTRL基学习器与基于最大耦合的移动感知主算法组成的元学习框架，遗憾界为期望意义下的 Õ(min{√(T(S_T+1)), T^{2/3}(P_T+1)^{1/3}})。

    

    我们研究了在线凸优化中带指示器切换代价的动态遗憾问题：该代价是指每当两个连续决策不同时所产生的固定惩罚。它刻画了诸如服务器启动、模型部署和缓存更新等启动开销，并且在一个有界域上，它能将基于范数的移动代价作为特例加以涵盖。现有的针对指示器代价的保证仅能处理静态比较器。我们证明，将这些技术直接扩展到动态遗憾的情形是注定会失败的，这促使我们采用一种不同的方法。我们提出了一个元学习框架：一组在二进制时间尺度上重启的随机化惰性FTRL基学习器，由一个感知移动的主算法进行聚合，该主算法混合各基学习器的提议密度，并通过连续混合分布之间的最大耦合来采样动作。所得算法在期望意义下满足 $\mathcal{R}^{\mathbf{1}}_T \le \tilde{\mathcal{O}}(\min\{\sqrt{T(S_T{+}1)},T^{2/3}(P_T+1)^{1/3}\})$，其中……

    arXiv:2609.30556v1 Announce Type: new  Abstract: We study dynamic regret in online convex optimization with an \emph{indicator switching cost}: a fixed penalty incurred whenever two consecutive decisions differ. This captures startup overheads such as server activation, model deployment, and cache updates, and on a bounded domain it recovers norm-based movement costs as a special case. Existing guarantees for indicator costs handle only static comparators. We show that a direct extension of these techniques to dynamic regret provably fails, motivating a different approach. We propose a meta-learning framework: a set of randomized lazy FTRL base learners restarted at dyadic time scales, aggregated by a movement-aware master that mixes their proposal densities and samples actions via maximal coupling of consecutive mixtures. The resulting algorithm satisfies, in expectation, $\mathcal{R}^{\mathbf{1}}_T \le \tilde{\mathcal{O}}(\min\{\sqrt{T(S_T{+}1)},T^{2/3}(P_T+1)^{1/3}\})$, where $\math
    
[^146]: 面向昂贵进化优化的排序可靠教师引导适应度近似：一项TinyML架构搜索研究

    Rank-Reliable Teacher-Guided Fitness Approximation for Expensive Evolutionary Optimization: A TinyML Architecture Search Study

    [https://arxiv.org/abs/2609.30553](https://arxiv.org/abs/2609.30553)

    提出TGL-NSGA-II框架，通过教师引导的轻量知识蒸馏生成排序可靠的低保真适应度分数，并与高斯过程代理模型融合，以显著降低昂贵TinyML神经架构搜索中进化优化的评估成本。

    

    昂贵的进化搜索并不总是需要对每个候选方案进行精确的适应度估计，它往往只需要对一个更简单问题的可靠回答：哪个候选方案更好？我们通过教师引导学习NSGA-II（TGL-NSGA-II）来满足这一需求，这是一个面向受限Tiny机器学习（TinyML）神经架构搜索的低保真度框架。预训练的教师模型将样本组织成由难度和类别共同定义的分层。每个候选方案随后经过KD-Lite——一种在紧凑训练集上进行的简短且有上限的知识蒸馏过程——然后在一个单独的分层评估集上被打分。这一教师引导的分数与高斯过程代理模型相融合，用于选择候选方案进行完整评估。对于固定的候选种群，我们分析了评估方差、分数集中度、成对排序反转、期望Kendall-τ、第一前沿识别以及超体积扰动。我们还推导了……（原文摘要在此处截断）

    arXiv:2609.30553v1 Announce Type: new  Abstract: Expensive evolutionary search does not always need an exact fitness estimate for every candidate. It often needs a reliable answer to a simpler question: which candidate is better? We address this need through Teacher-Guided Learning NSGA-II (TGL-NSGA-II), a low-fidelity framework for constrained Tiny Machine Learning (TinyML) neural architecture search. A pretrained teacher organizes samples into strata defined jointly by difficulty and class. Each candidate then undergoes KD-Lite, a short and capped knowledge-distillation procedure on a compact training set, before being scored on a separate stratified evaluation set. This teacher-guided score is fused with a Gaussian-process surrogate to select candidates for full evaluation. For a fixed candidate population, we analyse evaluation variance, score concentration, pairwise rank inversion, expected Kendall-$\tau$, first-front identification, and hypervolume perturbation. We also derive a 
    
[^147]: GyroNovo：面向从头肽段测序的误差引导片段填补与质量感知注意力方法

    GyroNovo: Error-Guided Fragment Imputation with Mass-Aware Attention for \textit{De Novo} Peptide Sequencing

    [https://arxiv.org/abs/2609.30542](https://arxiv.org/abs/2609.30542)

    GyroNovo提出了一个从头肽段测序框架，利用解码器误差自适应引导碎片填补目标，并通过质量感知注意力显式建模峰间质量差异，从而提升对稀疏、含噪且不完整谱图的测序准确性。

    

    从串联质谱中进行从头肽段测序对于在不依赖参考数据库的情况下鉴定肽段至关重要。尽管深度学习已取得进展，准确的测序仍然具有挑战性，因为实验谱图往往稀疏、含噪且不完整，导致信息丰富的b离子和y离子碎片未被观测到。现有方法尝试在自回归解码之前，通过潜空间填补来恢复这些缺失的证据。然而，它们通常将填补视为固定的重建任务，而没有考虑哪些缺失的碎片与解码器的误差最为相关。此外，现有的峰表示方法没有显式地建模峰之间的质量差异，尽管这种差异具有根本性的重要意义。我们提出了GyroNovo，一个具有两大主要贡献的框架。首先，我们利用训练期间观察到的解码器误差来自适应地调整填补目标，优先处理与频繁解码错误相关的碎片……（摘要原文在此处截断）

    arXiv:2609.30542v1 Announce Type: new  Abstract: De novo peptide sequencing from tandem mass spectra is essential for identifying peptides without relying on reference databases. Despite advances in deep learning, accurate sequencing remains challenging because experimental spectra are often sparse, noisy, and incomplete, leaving informative b- and y-ion fragments unobserved. Existing methods attempt to recover this missing evidence via latent-space imputation before autoregressive decoding. However, they typically treat imputation as a fixed reconstruction task, without considering which missing fragments are most relevant to decoder errors. Moreover, existing peak representations do not explicitly model mass differences between peaks, despite their fundamental importance.   We introduce GyroNovo, a framework with two main contributions. First, we use decoder errors observed during training to adapt the imputation objective, prioritizing fragments associated with frequent decoding err
    
[^148]: 生产规模的AutoResearch：失效模式与多智能体框架

    AutoResearch at Production Scale: Failure Modes and a Multi-Agent Framework

    [https://arxiv.org/abs/2609.30541](https://arxiv.org/abs/2609.30541)

    该论文将AutoResearch范式应用于生产规模的推荐系统嵌入优化，在220多次实验中识别出基础设施脆弱、智能体记忆衰退、搜索方向停滞、迭代成本不对称和指标固化五种失效模式，并据此提出多智能体框架加以解决。

    

    arXiv:2609.30541v1 公告类型：cross 摘要：为生产级推荐流水线优化嵌入系统需要进行系统性探索，而在大规模场景下，这种探索会消耗不成比例的工程投入。我们应用 Andrej Karpathy 的 AutoResearch 范式——即由大型语言模型迭代地修改训练脚本，并保留那些能够提升留出标量指标的修改——来自动化这一探索过程。我们报告了在生产规模下运行该范式十二周的经验：其中每次迭代需要消耗数小时的多GPU算力，评估涉及相互竞争的多项准则，且整个实验活动跨越数周、涉及大量训练任务。在为一个图书推荐流水线独立开发的两套表征学习系统上，我们运行了220多次实验，观察到在原始设定中不存在的五种反复出现的失效模式：基础设施脆弱性、智能体记忆衰退、搜索方向停滞、迭代成本不对称以及指标固化。我们贡献了一个三原则……（摘要原文在此处截断）

    arXiv:2609.30541v1 Announce Type: cross  Abstract: Optimizing embedding systems for production recommendation pipelines demands systematic exploration that consumes disproportionate engineering effort at scale. We apply Andrej Karpathy's AutoResearch paradigm -- a large language model that iteratively edits a training script and retains modifications that improve a held-out scalar metric -- to automate this exploration. We report on twelve weeks of running this paradigm at production scale, where iterations consume hours of multi-GPU compute, evaluation involves competing criteria, and campaigns span weeks across many training jobs. Across two independently developed representation-learning systems for a book recommendation pipeline, we ran 220+ experiments and observed five recurring failure modes absent from the original setting: infrastructure fragility, agent memory decay, search-direction stagnation, iteration-cost asymmetry, and metric fixation. We contribute a three-principle sc
    
[^149]: 通过深度展开学习替代Split-Gibbs扩散后验采样中的MCMC方法

    Learning to Replace MCMC in Split-Gibbs Diffusion Posterior Sampling via Deep Unfolding

    [https://arxiv.org/abs/2609.30539](https://arxiv.org/abs/2609.30539)

    本文提出一种基于深度展开的学习框架，将split-Gibbs扩散后验采样中的Gibbs更新重新表述为高斯去噪问题并通过ODE扩散实现，从而以更低的似然更新计算成本替代了传统MCMC迭代。

    

    Split Gibbs采样通过解耦先验与似然的计算，实现了针对一般非线性逆问题的扩散后验推断，并使预训练的扩散先验可在不同测量模型之间复用。然而，其似然更新通常依赖迭代式MCMC，这不仅阻碍并行化，还需要针对特定算法的调参，并带来高昂的计算成本。在本工作中，我们提出一种基于学习的框架来替代这一MCMC步骤，其方法是将两个Gibbs更新重新表述为高斯去噪问题，并通过ODE扩散加以实现。先验步骤复用预训练的去噪器，而似然去噪器则通过轻量级的深度展开网络利用已知的似然结构。在非线性相位恢复任务上的实验表明，所提方法能以更低的似然更新成本有效替代基于MCMC的split Gibbs方法。

    arXiv:2609.30539v1 Announce Type: cross  Abstract: Split Gibbs sampling enables diffusion posterior inference for general nonlinear inverse problems by decoupling prior and likelihood computations, allowing a pretrained diffusion prior to be reused across measurement models. However, its likelihood update often relies on iterative MCMC, which can hinder parallelization, require algorithm-specific tuning, and incur substantial computational cost. In this work, we propose a learning-based framework to replace this MCMC step by reformulating both Gibbs updates as Gaussian denoising problems and implementing them through ODE diffusion. The prior step reuses a pretrained denoiser, while the likelihood denoiser exploits known likelihood structure through a lightweight deep-unfolded network. Experiments on nonlinear phase retrieval demonstrate the effectiveness of the proposed method as an alternative to MCMC-based split Gibbs at lower likelihood-update cost.
    
[^150]: 看见语音：面向语音驱动的三维人脸动画学习可见的发音动态

    Seeing Speech: Learning Visible Articulatory Dynamics for Speech-Driven 3D Facial Animation

    [https://arxiv.org/abs/2609.30517](https://arxiv.org/abs/2609.30517)

    该论文提出一种发音感知框架，通过语音-发音记忆模块（SAM）和拓扑感知发音组合模块（TAC）建模三种方向性发音运动，从而将语音映射为与语音一致且表面连贯的三维人脸动画。

    

    语音驱动的三维人脸动画的最新进展提升了顶点级别的重建质量，但与语音一致的可见发音动作仍然难以实现。这是因为语音的产生遵循结构化且受约束的发音器官协调方式，而且从声学到运动的映射本质上是一对多的。受可见发音的结构化模式启发，我们提出了一种新颖的发音感知框架，通过方向性的发音运动来建模可见语音，并将这些运动组合成与表面一致的三维人脸运动。为了用三种方向性发音运动（横向展开、张开、前突）来表示可见发音，我们提出了语音-发音记忆模块（SAM），它基于键值记忆结构，通过检索与解码在语音学上下文中捕捉语音与这些运动之间的对应关系。然后，拓扑感知的发音组合模块（TAC）将……

    arXiv:2609.30517v1 Announce Type: cross  Abstract: Recent progress in speech-driven 3D facial animation has improved vertex-level reconstruction quality, but speech-consistent visible articulation remains difficult. This is because speech production follows structured and constrained articulators' coordination and the mapping from acoustics to motion is inherently one-to-many. Motivated by the structured patterns of visible articulation, we propose a novel articulation-aware framework that models visible speech through directional articulatory motions and composes them into surface-consistent 3D facial motion. To represent visible articulation with three directional articulatory motions, spreading, opening, and protrusion, we propose a Speech--Articulatory Memory (SAM) that captures the correspondence between speech and these motions under phonetic context through retrieval and decoding based on a key-value memory structure. Then, a Topology-aware Articulatory Composition (TAC) integra
    
[^151]: 在储备池计算框架下对秀丽隐杆线虫连接组进行基准测试

    Benchmarking the Connectomes of Caenorhabditis elegans within the Reservoir Computing Framework

    [https://arxiv.org/abs/2609.30508](https://arxiv.org/abs/2609.30508)

    该研究将秀丽隐杆线虫在不同年龄段、以三种测量方式获得的连接组经最少预处理实现为回声状态网络储备池，并通过神经启发任务对其进行基准测试评估。

    

    这项工作的目的是使用储备池计算框架，从计算的角度对秀丽隐杆线虫的连接组进行研究。连接组是生物神经网络的映射图；秀丽隐杆线虫是首个已发表覆盖整个神经系统物理连接组的生物体。本文所使用的秀丽隐杆线虫连接组是在该生物体的不同年龄段获取的，并基于三种不同的细胞间连接测量方法。在经过极少的预处理后，这些连接组被实现为回声状态网络形式的储备池，回声状态网络是一种循环神经网络。在储备池计算中，储备池本身并不参与训练，而是将储备池的输出传递给一个相对较小的读出模块，训练在该模块中进行。训练和测试在不同的神经启发任务中进行，目的是将这些任务作为对连接组进行基准测试的手段。

    arXiv:2609.30508v1 Announce Type: new  Abstract: The aim of this work is to examine the connectomes of Caenorhabditis elegans through a computational lens using the reservoir computing framework. Connectomes are mappings of biological neural networks; C. elegans is the first organism for which physical connectomes covering the whole nervous system have been published. The connectomes of C. elegans used in this paper have been derived at different ages of the organism and are based on three different ways of measuring inter-cellular connections. They have, with minimal preprocessing, been implemented as reservoirs in the form of echo state networks, which are recurrent neural networks. In reservoir computing, the reservoir itself is not trained, rather the output of the reservoir is passed to a comparatively small read-out module in which training takes place. Training and testing is conducted in different neuro-inspired tasks, with the aim of using these tasks as a benchmark for the co
    
[^152]: 联邦目标极大似然估计

    Federated Targeted Maximum Likelihood Estimation

    [https://arxiv.org/abs/2609.30503](https://arxiv.org/abs/2609.30503)

    本文提出了首个联邦目标极大似然估计算法，通过梯度聚合（FedTMLE-G）和本地自主涨落拟合（FedTMLE-L）两个互补框架，首次在数据无法集中汇总的跨机构场景下，对任意目标、损失和涨落族实现了 TMLE 目标化步骤的联邦化。

    

    科学或运营决策背后的证据往往由医院、银行或登记机构所掌握，而这些机构无法汇总个体层面的观测数据。跨机构（cross-silo）联邦学习将计算迁移到数据所在地，并只交换事先约定的摘要统计量。目标极大似然估计（TMLE）在灵活的初始拟合基础上进行精细的目标化修正，得到符合模型假设并支持高效统计推断的插件估计量。然而，TMLE 本身一直是一个完全集中式的流程。为填补这一空白，本文提出了首个联邦 TMLE 算法。我们针对任意目标、损失函数和涨落族，通过两个互补的框架将目标化步骤本身联邦化：FedTMLE-G 聚合各机构的局部梯度，逐步复现集中式目标化步骤；FedTMLE-L 则允许每个机构先独立完成自己的涨落拟合，之后仅需一次性交换拟合更新，以同步精确性换取本地自主性。对于梯度聚合，我们开发了……（原文摘要到此被截断）

    arXiv:2609.30503v1 Announce Type: new  Abstract: The evidence behind a scientific or operational decision is often held by hospitals, banks, or registries that cannot pool individual observations. Cross-silo federated learning moves computation to the data and exchanges agreed summaries. Targeted maximum likelihood estimation (TMLE) refines a flexible initial fit, yielding plug-in estimators that respect the model and support efficient inference. TMLE itself, however, has remained a fully centralized procedure. To fill this gap, our paper introduces the first federated TMLE algorithm. We federate targeting itself, for an arbitrary target, loss, and fluctuation family, through two complementary frameworks. FedTMLE-G aggregates local gradients and reproduces centralized targeting step for step. FedTMLE-L lets each institution complete its own fluctuation fit before a single exchange of fitted updates, trading synchronized fidelity for local autonomy. For gradient aggregation, we develop 
    
[^153]: 要求解下层非凸的双层优化问题，我们需要二阶平稳性

    To Solve Bilevel Optimization with Nonconvex Lower Levels, We Need Second-Order Stationarity

    [https://arxiv.org/abs/2609.30501](https://arxiv.org/abs/2609.30501)

    本文突破了现有双层优化研究依赖下层凸性假设的局限，系统研究了下层非凸的双层优化问题，并论证了求解此类问题必须采用二阶平稳性条件而非传统的一阶平稳性。

    

    尽管双层优化（BLO）近年来已成为解决许多复杂嵌套机器学习问题的强大框架，但大多数现有研究仅限于下层强凸（LLSC）或下层一般凸（LLGC）的设置（即假设下层目标函数至少是凸的）。虽然LLSC/LLGC假设使得算法设计和理论分析更加易于处理，但它们过于严格，无法涵盖实践中的许多机器学习问题。BLO中LLSC/LLGC假设的局限性促使我们研究在下层一般非凸（LLNC）设置下求解BLO问题，而这一方向仍处于起步阶段。在LLNC-BLO的文献中，大多数现有工作要么需要下层目标函数具有额外的结构以便进行可处理的理论分析，要么采用一阶平稳性重表述作为下层……

    arXiv:2609.30501v1 Announce Type: new  Abstract: Although bilevel optimization (BLO) has emerged as a powerful framework for addressing many complex and nested machine learning problems in recent years, most existing studies are confined to the lower-level strongly convex (LLSC) or lower-level generally convex (LLGC) settings (i.e., the lower-level objective function is assumed to be, at least, convex). While the LLSC/LLGC assumptions render more tractable algorithmic design and theoretical analysis, they are too rigid to encompass many machine learning problems in practice. The limitations of LLSC/LLGC assumptions in BLO motivate us to investigate solving the BLO problem in the general lower-level nonconvex (LLNC) settings, which remains in its infancy. In the literature on LLNC-BLO, most of the existing works either require additional structures in the lower-level objective function for tractable theoretical analysis, or adopt the first-order stationarity reformulation as a lower-lev
    
[^154]: PolicyAttention：Softmax注意力实现闭环控制的策略镜像下降

    PolicyAttention: Softmax Attention Implements Policy Mirror Descent for Closed-Loop Control

    [https://arxiv.org/abs/2609.30500](https://arxiv.org/abs/2609.30500)

    该论文构造了一个带显式残差的因果softmax“行动者—环境—单步评论家”协议，证明softmax注意力可以作为重复控制器实现策略镜像下降，并用预注册实验验证了训练的pre-LN Transformer能够恢复目标计算。

    

    因果softmax注意力能否将策略镜像下降实现为一个重复的控制器，而非仅仅一步代数恒等式？负熵策略镜像下降（PMD）具有逐状态更新式 PMD_η(π,Q)=softmax(log π+ηQ)。基于已知的Q-TD-PMD递推关系，我们构建了一个固定的因果softmax“行动者—环境—单步评论家”协议，其中包含显式的行动者残差、路由残差、采样残差和归一化残差，并将这些残差传播到最终实际返回的策略上。该构造明确了有限logit/全支撑的定义域、外部分词与采样边界，以及归一化编译所需的均值为零的LayerNorm载体条件。分别训练的pre-LN Transformer在经验上能够恢复目标计算。在所测试的固定规则中，冻结的单步审计模型最接近PMD；在一项预注册的五次运行、S=4的重复控制测试中，学习到的a……（原文摘要在此处截断）

    arXiv:2609.30500v1 Announce Type: cross  Abstract: Can causal softmax attention implement policy mirror descent as a repeated controller rather than a one-step algebraic identity? Negative-entropy policy mirror descent (PMD) has the statewise update $\operatorname{PMD}_\eta(\pi,Q)=\operatorname{softmax}(\log\pi+\eta Q)$. Building on the known Q-TD-PMD recursion, we construct one fixed causal-softmax actor--environment--one-step-critic protocol with explicit actor, routing, sampling, and normalization residuals, and propagate them to the policy actually returned. The construction states the finite-logit/full-support domain, the external tokenization and sampling boundary, and the mean-zero LayerNorm carrier conditions required by the normalized compilation.   Separately trained pre-LN Transformers recover the target computation empirically. A frozen one-step audit model is closest to PMD among the tested fixed rules; in a preregistered five-run $S=4$ repeated-control test, the learned a
    
[^155]: 距离依赖矩条件下的普通非凸SGD：有限时域平稳性与Nagaev界

    Ordinary Nonconvex SGD under Distance-Dependent Moments: Finite-Horizon Stationarity and Nagaev Bounds

    [https://arxiv.org/abs/2609.30499](https://arxiv.org/abs/2609.30499)

    本文证明，当条件矩允许噪声方差随迭代点距离增长时，普通单样本SGD无需任何修改即可达到与Blum–Gladyshev下界匹配的极小极大随机复杂度，并借助希尔伯特空间Fuk–Nagaev不等式给出高概率Nagaev型界。

    

    统一的噪声矩界假设排除了那些变异性随迭代点位置增长的随机梯度。我们在距离依赖的条件矩假设下，研究针对光滑、下有界且可能非凸目标的普通单样本随机梯度下降。仅利用二阶矩条件，一个直接的“下降—位移”论证在使用依赖时域的步长时，给出了 $T^{-1/3}$ 的期望平均平方梯度平稳性。一个显式的预言机复杂度推论与已知的平滑Blum–Gladyshev（BG-0）下界相匹配，包括 $Lb_2\Delta^3\varepsilon^{-6}$ 和 $L\Delta\sigma^2\varepsilon^{-4}$ 两个随机项，其中 $\Delta$ 为初始目标间隙，$\sigma^2+b_2\|x-x_1\|^2$ 为方差的上界。因此，无需任何修改的SGD在这一二阶矩类别中即达到极小极大随机复杂度。对于 $p>2$，可预测局部化技术与希尔伯特空间上的Fuk–Nagaev不等式给出了一个高概率界，分离对数……（摘要在此处被截断）

    arXiv:2609.30499v1 Announce Type: new  Abstract: Uniform noise-moment bounds exclude stochastic gradients whose variability increases with the iterate. We study ordinary, single-sample stochastic gradient descent for smooth, lower-bounded, possibly nonconvex objectives under distance-dependent conditional moments. Under second moments alone, a direct descent--displacement argument yields $T^{-1/3}$ expected average squared-gradient stationarity with a horizon-dependent stepsize. An explicit oracle-complexity corollary matches the known smooth Blum--Gladyshev (BG-0) lower bound, including the $Lb_2\Delta^3\varepsilon^{-6}$ and $L\Delta\sigma^2\varepsilon^{-4}$ stochastic terms, where $\Delta$ is the initial objective gap and $\sigma^2+b_2\|x-x_1\|^2$ bounds the variance. Thus unchanged SGD attains the minimax stochastic complexity in this second-moment class. For $p>2$, predictable localization and a Hilbert-space Fuk--Nagaev inequality yield a high-probability bound separating logarith
    
[^156]: 学习偏置：机器学习增强的粒子滤波器

    Learning to Bias: Machine Learning-Enhanced Particle Filters

    [https://arxiv.org/abs/2609.30498](https://arxiv.org/abs/2609.30498)

    提出神经最优粒子滤波器（NOPF），通过从离线模拟数据中学习最优提议分布的摊销近似，并将其作为即插即用模块嵌入标准粒子滤波器，借助重要性权重校正保证滤波分布的一致性，从而提升序贯推断的样本效率与高维扩展能力。

    

    序贯推断旨在从含噪且不完整的观测中估计潜在状态。粒子滤波器（PFs）是一类基于重要性采样的蒙特卡洛方法，为该任务提供了灵活的框架，但其样本效率往往较低，且随维度增长扩展性不佳，部分原因在于提议分布不够理想。我们通过将学习到的提议分布集成到粒子滤波框架中来应对这些挑战。我们提出了神经最优粒子滤波器（NOPFs），它从离线模拟的单步条件数据元组中学习最优提议分布的摊销近似。学习到的提议分布可作为即插即用模块直接替换标准粒子滤波更新中的原有提议，同时样本通过标准重要性权重进行校正，因此在标准的支撑集与密度可评估假设下，该方法渐近地收敛于相同的滤波分布。在推断复杂度各异的随机非线性基准测试中，NOPFs（此处原文截断）……

    arXiv:2609.30498v1 Announce Type: cross  Abstract: Sequential inference estimates latent states from noisy and incomplete observations. Particle Filters (PFs), a class of Monte Carlo methods based on importance sampling, provide a flexible framework for this task, but often suffer from poor sample efficiency and unfavorable scaling with dimension, partly due to suboptimal proposal distributions. We address these challenges by integrating learned proposals into the PF framework. We introduce Neural Optimal Particle Filters (NOPFs), which learn an amortized approximation to the optimal proposal from offline simulated one-step conditioning tuples. The learned proposal is used as a drop-in replacement in standard PF updates, with samples corrected by standard importance weights so that the method asymptotically targets the same filtering distribution under standard support and density-evaluation assumptions. Across stochastic nonlinear benchmarks of varying inference complexity, NOPFs impr
    
[^157]: 对称正定流形上函数值数据的几何特征学习

    Geometric Feature Learning for Functional Data Valued on the Symmetric Positive Definite Manifold

    [https://arxiv.org/abs/2609.30487](https://arxiv.org/abs/2609.30487)

    该论文提出了MatFAE——一种用于学习对称正定（SPD）矩阵黎曼流形上轨迹的函数神经网络，它将序列视为连续函数以编码轨迹动力学特性，并通过函数权重的形态提供可解释性。

    

    我们在此提出了一种函数神经网络，称为MatFAE，用于学习对称正定（SPD）矩阵黎曼流形上的轨迹。MatFAE的特点是包含内在层，这些内在层将流形值函数映射为欧几里得向量值函数，随后通过一个函数层将其投影到有限维欧几里得空间。与大多数针对离散时间序列的神经网络不同，MatFAE将每个序列视为连续函数，因此能够在潜在表示中编码轨迹的动力学特性（例如一阶导数）。此外，函数层中函数权重的形态通过揭示输入函数数据中对潜在表示贡献最大的区域，提供了可解释性。我们论证了每个内在层的设计原则和性质，并详细说明了在反向传播过程中如何处理矩阵分解。我们将MatFAE应用于……（原文摘要在此处截断）

    arXiv:2609.30487v1 Announce Type: cross  Abstract: We here develop a functional neural network, termed MatFAE, for learning trajectories on the Riemannian manifold of symmetric positive definite (SPD) matrices. MatFAE features intrinsic layers that map manifold-valued functions to Euclidean vector-valued functions, followed by a functional layer that projects them into a finite-dimensional Euclidean space. Unlike most neural networks for discrete-time sequences, MatFAE treats each sequence as a continuous function and can therefore encode trajectory dynamics (e.g., first-order derivatives) in its latent representations. Additionally, the morphology of the functional weights in the functional layer offers interpretability by revealing the regions of the input functional data that contribute most to the latent representations. We justify the design principles and properties of each intrinsic layer and detail how matrix factorization is handled during backpropagation. We apply MatFAE to a
    
[^158]: 大语言模型真的理解上下文吗？一种基于知识图谱的评估框架

    Do LLMs Understand Context? A Knowledge Graph-Based Evaluation Framework

    [https://arxiv.org/abs/2609.30484](https://arxiv.org/abs/2609.30484)

    提出了一种基于知识图谱的评估框架，通过语义结构相似度衡量大语言模型在问答任务中真正的上下文理解能力，弥补了BLEU和困惑度等传统指标只能评估表面性能的不足。

    

    尽管大语言模型（LLM）已经展现出卓越的语言能力，但一个深刻的问题始终萦绕在其核心：这些模型究竟是真正理解上下文，还是只是以前所未有的规模擅长模式匹配？大语言模型中的上下文理解是指从给定上下文中正确提取相关信息，将其整合为连贯的内部表示，并对其进行推理，从而产生事实一致且立足于上下文的回应的能力。然而，双语评估替补（BLEU）和困惑度等传统方法只能衡量表面层次的性能，这在问答（QA）任务中暴露了一个关键缺口，因为问答的回应必须立足于上下文，而不仅仅是记忆中的关联。为填补这一空白，我们提出了一种新颖的基于知识图谱（KG）的评估框架，用于评估大语言模型在问答中的上下文理解能力，其核心是语义结构相似度（Semantic Structural Similarity）。

    arXiv:2609.30484v1 Announce Type: new  Abstract: While large language models (LLMs) have achieved remarkable linguistic capabilities, a profound question lingers at their core: do these models truly comprehend context or simply excel at pattern matching on an unprecedented scale? Contextual understanding in LLMs refers to the ability to correctly extract relevant information from a given context, integrate it into a coherent internal representation, and reason over it to produce factually consistent and contextually grounded responses. However, traditional methods such as BiLingual Evaluation Understudy (BLEU) and perplexity simply measure surface-level performance. This reveals a critical gap in question answering (QA), where responses must be contextually grounded rather than simply being memorized associations. To fill this void, we propose a novel knowledge graph (KG) based evaluation framework for LLM contextual understanding in QA. Central to this is Semantic Structural Similarit
    
[^159]: AcoustiClaim：基于仪器真值的数值声明基准测试

    AcoustiClaim: A Numeric Claim Benchmark with Instrument Ground Truth

    [https://arxiv.org/abs/2609.30483](https://arxiv.org/abs/2609.30483)

    该论文提出 AcoustiClaim 基准，首次以定义声学量的仪器读数作为真值来验证音频语言模型给出的数值声明，发现绝大多数模型的表现不优于常数预测器，仅一个闭源模型在音高等少数任务上通过了秩相关阈值。

    

    音频语言模型会针对声学量给出具体数值，但无论是人类的主观意见还是评判模型，都无法判断该数值是否真实符合信号本身。AcoustiClaim 从自由文本中提取每个数值声明，根据定义该声学量的仪器对其打分，并按参考值可从何处读取对每个声学量进行分类。四个开放权重系统和一个闭源模型，在两个语料库上以五种提问方式被询问十个声学量，共填满207个评估单元格。其中49个单元格输出的不同数值少于五个；在158个可进行排序的单元格中，有8个超过了我们设定的0.3秩相关阈值，其中三个的置信区间明显高出该阈值，这八个中有五个来自同一个读取音高的闭源模型。在除三个之外的所有可排序单元格中，模型误差都处于或高于常数预测器的下限。我们训练的参考解码器在95%的混合音频样本上以自然语言方式拒绝给出五个语音相关声学量，不作任何保留，而在干净的孪生样本上则能陈述这些量，从而复现了其目标所遵循的规则……

    arXiv:2609.30483v1 Announce Type: cross  Abstract: Audio language models state numbers for acoustic quantities, and neither human opinion nor a judge model says whether such a number is true of the signal. AcoustiClaim extracts each numeric claim from free text, scores it against the instrument that defines the quantity, and classes each quantity by where its reference can be read. Four open-weight systems and one closed model, asked for ten quantities five ways on two corpora, fill 207 cells. Of these, 49 emit fewer than five distinct values, and eight of the 158 cells that can be ranked exceed a rank correlation of 0.3, the bar we set, three with an interval clear of it, five of them one closed model reading pitch. Error sits at or above a constant-predictor floor in every ranked cell but three. The reference decoder we train declines the five voice quantities in prose on 95% of mixtures, with nothing withheld, and states them on the clean twins, reproducing its targets' rule from au
    
[^160]: 面向精确SSE聚类的支架约束子集动态规划

    Scaffold-Constrained Subset Dynamic Programming for Exact SSE Clustering

    [https://arxiv.org/abs/2609.30477](https://arxiv.org/abs/2609.30477)

    该论文提出利用数据导出的几何图对精确子集动态规划施加支架约束，仅允许连通顶点子集作为候选簇，在不改变SSE损失的前提下大幅缩减搜索空间，并证明在密度条件下每个观测仅保留O(log n)个最近邻即可高概率保持经验最优解。

    

    精确的欧几里得K-means聚类将n个观测划分到K个无标签的簇中，但无约束的搜索通常具有指数级复杂度。我们利用从数据中导出的几何图来对一个精确子集动态规划进行预处理：其结果是，只有连通的顶点子集才被允许作为簇，而误差平方和（SSE）损失保持不变。通过剩余集递归最小化固定K值或带惩罚项的SSE，并对每个剩余集的连通分量进行精确分解。我们研究的核心问题是：在保持无约束最优解的前提下，可以移除多少计算支持。图的包含关系给出了单调的覆盖与支持关系，一个瓶颈阈值识别出嵌套层次结构中的第一个覆盖图。对于固定的K和维度，在紧致球支撑和密度界的条件下，每个观测保留q=O(log n)个最近邻即可在概率上保持经验SSE最优（摘要原文在此处截断）。

    arXiv:2609.30477v1 Announce Type: cross  Abstract: Exact Euclidean \(K\)-means partitions \(n\) observations into \(K\) unlabelled clusters, but the unrestricted search is generally exponential. We use data-derived geometric graphs to precondition an exact subset dynamic program: as a result only connected vertex subsets are admitted as clusters, while sum-of-squared-errors (SSE) loss is unchanged. A remaining-set recurrence minimises fixed-\(K\) or penalised SSE, with exact factorisation over the connected components of each remaining set. The central question we study is how much computational support can be removed while preserving an unrestricted optimum. Graph inclusion gives monotone coverage and support relations, and a bottleneck threshold identifies the first covering graph in a nested hierarchy. For fixed \(K\) and dimension, under compact ball support and density bounds, retaining \(q=O(\log n)\) nearest neighbours per observation preserves an empirical SSE optimum with prob
    
[^161]: 导师解码：更快的推理遇上提升算法

    Mentored Decoding: Faster Inference meets Boosting

    [https://arxiv.org/abs/2609.30474](https://arxiv.org/abs/2609.30474)

    该论文提出“导师解码”这一有损推测解码的形式化框架，通过将其与机器学习中的提升理论相联系并推广到全部f-散度族，正式证明了有损推测解码所得模型不仅加速推理，还能在质量上超越目标模型。

    

    推测解码是一种成功的技术，它通过一个快速的草稿模型来加速目标自回归语言模型的推理。有损推测解码允许输出相对目标模型产生偏移，从而进一步提高速度。有趣的是，实验中观察到，所得模型在质量上也能超越目标模型。我们的论文通过对有损推测解码的一种形式化方法——称为“导师解码”——正式证明了这种成就是如何实现的。为此，我们将推理过程与著名的机器学习训练理论——提升算法——联系起来，并通过将导师解码推广到全部f-散度族来展开研究。我们揭示了导师解码的若干关键性质，其中包括：(i) 总变差情形下特别吸引人的几何本质；(ii) 适用于任意f-散度的简单近似，且其与提升算法的兼容性直接相关；以及 (iii) ……

    arXiv:2609.30474v1 Announce Type: new  Abstract: Speculative decoding is a successful technique speeding up inference of a target autoregressive language model via a fast drafter model. Lossy speculative decoding allows a drift with respect to the target to further improve speed. Interestingly, it has been observed experimentally that the resulting model can $\textit{also}$ beat the target $\textit{quality-wise}$. Our paper formally proves how such a feat is possible with a formal approach to lossy speculative decoding called $\textit{mentored decoding}$. To get there, we connect inference to a celebrated ML training theory, $\textit{boosting}$, and proceed via the generalization of mentored decoding to the whole set of $f$-divergences. We uncover key properties of mentored decoding, among which (i) the particularly appealing geometric nature of the total variation case, (ii) simple approximations for any $f$-divergence in direct relation with boosting compliance, and (iii) a $\textit{
    
[^162]: 矩引导的边采样

    Moment-guided edge sampling

    [https://arxiv.org/abs/2609.30472](https://arxiv.org/abs/2609.30472)

    提出基于随机游走转移矩阵谱矩的矩引导边采样框架，通过组合方法和低秩方法两种互补手段精确高效地量化和控制局部边编辑对全局图结构的影响，将单边编辑的矩更新成本从 $O(mn)$ 降至 $O(m)$ 乃至常数时间。

    

    边采样通过做出局部决策来实现图级别的目标，例如保持图的结构特性。这带来了一个根本性的挑战：*如何量化和控制局部边编辑（即边的添加或删除）对全局图结构的影响？* 我们通过一个基于随机游走转移矩阵谱矩的*矩引导边采样框架*来解决这一挑战。我们通过两种互补的方法计算精确的矩变化：一种是对低阶矩具有闭式更新公式的组合方法，另一种是利用*局部性*和*循环迹不变性*将计算压缩到被编辑端点的低秩方法，后者支持任意矩阶数和批量边编辑。对于固定矩阶数的单边编辑，低秩方法将计算成本从 $O(mn)$ 降低到 $O(m)$，而组合方法可在给定条件下常数时间内评估低阶矩的变化。

    arXiv:2609.30472v1 Announce Type: new  Abstract: Edge sampling makes local decisions to achieve graph-level objectives, such as preserving structural properties. This creates a fundamental challenge: \textit{how can the effect of a local edge edit (i.e., edge addition or removal) on global graph structure be quantified and controlled?} We address this challenge with a \textit{moment-guided edge sampling framework} based on spectral moments of the random-walk transition matrix. We compute exact moment changes through two complementary methods: a combinatorial method with closed-form updates for low-order moments, and a low-rank method that exploits \textit{locality} and \textit{cyclic trace invariance} to compress computations to edited endpoints, supporting arbitrary moment orders and batched edits. For single-edge edits at fixed moment orders, the low-rank method reduces the cost from $O(mn)$ to $O(m)$, while the combinatorial method evaluates low-order changes in constant time given 
    
[^163]: CARGO：面向生产环境中智能体AI的上下文感知检索门控评估

    CARGO: Context-Aware Retrieval-Gated Evaluation of Agentic AI in Production

    [https://arxiv.org/abs/2609.30471](https://arxiv.org/abs/2609.30471)

    针对生产环境中智能体AI评估时“参考-实例分歧”（即参考答案描述的是不同实体导致正确回答被误判为错误）的问题，提出CARGO框架，将检索参考视为流程范例、基于实时实例上下文判定事实并将评估门控于检索置信度之上。

    

    基于参考答案的LLM-as-a-judge评估方法假定参考答案即为目标答案。在部署于动态实体（如支持工单、资产、账户）之上运行的智能体系统中，最接近的可用参考答案通常是将正确流程应用到了不同的实体上，因此字面化的裁判会将不同的标识符、日期和状态判定为错误或幻觉。我们将这种失败模式命名为“参考-实例分歧”。我们提出了CARGO框架，该框架：（i）将检索到的参考答案视为流程范例，并将事实性判断建立在实际实例的实时观测上下文之上；（ii）为每条声明分配三向状态（支持、矛盾、无法验证），并仅对矛盾进行惩罚；（iii）通过检索置信度对评估进行门控，将生产环境评估建模为选择性预测问题。我们引入了CARGO-Bench，这是一个基于扰动的诊断套件，其真值通过构造方式获得，能够区分宽容……（原文在此截断）

    arXiv:2609.30471v1 Announce Type: cross  Abstract: Reference-based LLM-as-a-judge evaluation assumes the reference answer is the target. In deployed agentic systems that operate over dynamic entities (support cases, assets, accounts), the closest available reference typically applies the correct procedure to a different entity, so a literal judge penalizes different identifiers, dates, and statuses as errors or hallucinations. We name this failure mode reference-instance divergence (RID). We propose CARGO, a framework that (i) treats retrieved references as procedural exemplars and grounds factual judgments in the live instance's observed context, (ii) assigns each claim a three-way status (supported, contradicted, unverifiable) and penalizes only contradictions, and (iii) gates evaluation by retrieval confidence, casting production evaluation as selective prediction. We introduce CARGO-Bench, a perturbation-based diagnostic suite with ground truth by construction that separates lenien
    
[^164]: 面向鲁棒多模态情感分析的可靠性感知跨样本增强

    Reliability-aware Cross-sample Enhancement for Robust Multimodal Sentiment Analysis

    [https://arxiv.org/abs/2609.30470](https://arxiv.org/abs/2609.30470)

    提出可靠性感知跨样本增强（RCE）框架，通过自适应变分信息瓶颈建模模态不确定性并抑制噪声，同时利用跨样本检索高置信度语义一致的邻居样本来增强表示，统一解决了多模态情感分析中的噪声干扰与模态缺失问题。

    

    多模态情感分析（MSA）旨在从文本、音频和视觉等多种模态中推断人类情感。在实际应用中，输入数据常常受到噪声干扰和模态缺失的影响，从而导致性能下降。现有方法通常孤立地应对这些挑战，限制了其在现实场景中的有效性。为了解决这一局限性，我们提出了一种可靠性感知跨样本增强（RCE）框架。具体而言，RCE首先引入自适应变分信息瓶颈来建模模态级的不确定性并执行质量感知的信息压缩，从而抑制不可靠模态中的冗余噪声。此外，我们设计了一种可靠性感知的跨样本增强策略，从大型候选池中检索高置信度、语义一致的邻居样本，以丰富和校准当前的表示，有效缓解了因模态缺失和低质量输入造成的信息不足问题。

    arXiv:2609.30470v1 Announce Type: new  Abstract: Multimodal Sentiment Analysis (MSA) aims to infer human emotions from multiple modalities such as text, audio, and vision. In practice, inputs are often corrupted by noise and missing modalities, which degrades performance. Existing methods typically address these challenges in isolation, limiting their effectiveness in realistic settings. To address this limitation, we propose a Reliability-aware Cross-sample Enhancement (RCE) framework. Specifically, RCE first introduces an adaptive variational information bottleneck to model modality-wise uncertainty and perform quality-aware information compression, thereby suppressing redundant noise in unreliable modalities. Furthermore, we design a reliability-aware cross-sample enhancement strategy that retrieves high-confidence, semantically consistent neighbors from a large candidate pool to enrich and calibrate current representations, effectively alleviating information deficiency caused by m
    
[^165]: RAZOR：在大语言模型中剪枝可被替换的专家

    RAZOR: Pruning Replaceable Experts in LLMs

    [https://arxiv.org/abs/2609.30465](https://arxiv.org/abs/2609.30465)

    该论文提出无需训练的 MoE 专家剪枝方法 RAZOR，利用共识残差衡量专家的功能可替换性，在固定剪枝预算下剪除可被存活专家替代的专家，无需梯度或恢复训练即可最大程度保留原始模型输出分布。

    

    混合专家模型每个 token 只激活少量专家，但需要存储完整的专家池。专家剪枝可以减轻这种存储负担；在固定剪枝预算下，目标是尽可能保留原始模型的输出分布。然而，专家的使用频率或贡献大小本身并不能决定移除它所造成的损害，关键在于存活的计算能否替代其功能。我们提出 RAZOR，这是一种无需训练的专家剪枝方法，通过“共识残差”（即专家输出与原始加权混合输出的偏差）来评分专家的功能可替换性。该方法在固定层输入处使用精确的单删除恒等式，考虑了存活专家的重新归一化以及路由器选择的补充机制，从而在无需梯度或恢复训练的情况下，将在校准 token 上聚合得到的局部评分用于预算化剪枝。在 GLM-4.7-Flash、Qwen3.6-35B-A3B、DeepSeek-V4-Flash-0731 和 Hy3 上，在 25% 的（剪枝率下……摘要在此处截断）

    arXiv:2609.30465v1 Announce Type: cross  Abstract: Mixture-of-experts (MoE) models activate few experts per token but store the full expert pool. Expert pruning reduces this storage burden; at a fixed pruning budget, the goal is to preserve the original model's output distribution as closely as possible. Yet an expert's usage or contribution magnitude does not by itself determine the damage caused by its removal. What matters is whether the surviving computation can replace its function. We introduce RAZOR, a training-free expert pruning method that scores functional replaceability using consensus residuals: deviations of expert outputs from the original weighted mixture. An exact single-deletion identity at a fixed layer input accounts for survivor renormalization and router-selected refill, providing local scores aggregated over calibration tokens for budgeted pruning without gradients or recovery training. On GLM-4.7-Flash, Qwen3.6-35B-A3B, DeepSeek-V4-Flash-0731, and Hy3 at 25\% an
    
[^166]: 在生物安全相关基准上审计“系统1”模型：非生成式模型中的校准、选择性预测与排列不稳定性

    Auditing System-1 Models on Biosecurity-Relevant Benchmarks: Calibration, Selective Prediction, and Permutation Instability in a Non-Generative Model

    [https://arxiv.org/abs/2609.30454](https://arxiv.org/abs/2609.30454)

    该论文首次系统审计了商业非生成式“系统1”模型在生物安全相关基准上的可靠性，发现其校准良好但准确率高度依赖任务，且对答案选项的排列顺序表现出不稳定性。

    

    非生成式“系统1”模型在单次前向传播中即可返回结构化的概率决策，无需自回归解码，推理成本仅为生成式模型的一小部分。这使它们有望成为大型流水线中的低成本组件，但其在生物安全相关任务上的可靠性尚未得到系统性检验。我们在6,020道多项选择题上审计了一个商业系统1模型，题目来源于大规模杀伤性武器代理基准、一个改写鲁棒的WMDP-Bio变体以及六个LAB-Bench子任务，测量了准确率、校准度、错误检测、选择性预测以及对答案选项呈现顺序的敏感性。结果表明，准确率高度依赖于具体任务；在正确解读供应商提供的不确定性字段后，该模型的校准表现相当良好（合并期望校准误差为0.034），其top-1概率能够有效区分正确与错误的预测（原文摘要在此处被截断）

    arXiv:2609.30454v1 Announce Type: new  Abstract: Non-generative "System-1" models return structured probabilistic decisions in a single forward pass, without autoregressive decoding, at a small fraction of the inference cost of a generative model. This makes them of interest as inexpensive components in larger pipelines, but their reliability on biosecurity-relevant tasks has not been systematically examined. We audit one commercial System-1 model on 6,020 multiple-choice items drawn from the Weapons of Mass Destruction Proxy (WMDP), a paraphrase-robust WMDP-Bio variant, and six LAB-Bench subtasks, measuring accuracy, calibration, error detection, selective prediction, and sensitivity to the order in which answer options are presented. Accuracy is strongly task-dependent. Once the vendor's uncertainty field is correctly interpreted, the model is reasonably well calibrated (pooled expected calibration error 0.034) and its top-1 probability separates correct from incorrect predictions (p
    
[^167]: 基于仿真推断的fMRI功能连接贝叶斯不确定性量化

    Bayesian Uncertainty Quantification for fMRI Functional Connectivity via Simulation-Based Inference

    [https://arxiv.org/abs/2609.30445](https://arxiv.org/abs/2609.30445)

    该论文提出一个基于仿真推断的贝叶斯框架，通过将BOLD动力学建模为耦合Ornstein-Uhlenbeck过程并使用序贯神经后验估计，量化了fMRI功能连接估计中来自扫描仪噪声、被试变异性和采集时长三方面来源的不确定性。

    

    优化fMRI扫描时长和空间分辨率对实验设计至关重要，然而传统的基于相关的方法无法量化不确定性，也无法将扫描仪测量噪声与被试间真实的神经变异性区分开来。在缺乏有原则的不确定性界的情况下，研究人员无法判断扫描方案是否足够长以可靠地估计功能连接，也无法确定被试间差异反映的是生物学变异还是噪声。我们提出了一个贝叶斯框架，将BOLD信号动力学建模为耦合的Ornstein-Uhlenbeck过程，并使用序贯神经后验估计来获得功能连接的后验分布，同时考虑BOLD频谱中与频率无关的测量噪声。该框架应用于7T场强下28名健康对照被试（55次扫描）的数据，并使用功能网络图谱（65个默认模式网络脑区），量化了来自不同来源的不确定性：扫描仪噪声、被试变异性和采集时长。（摘要原文在此处截断）

    arXiv:2609.30445v1 Announce Type: new  Abstract: Optimizing fMRI scan duration and spatial resolution is critical for experimental design, yet traditional correlation-based approaches cannot quantify uncertainty or disentangle scanner measurement noise from true neural variability across subjects. Without principled uncertainty bounds, researchers cannot know whether a protocol is long enough to reliably estimate connectivity, or whether between-subject differences reflect biological variation or noise. We present a Bayesian framework modeling BOLD dynamics as coupled Ornstein-Uhlenbeck processes, using Sequential Neural Posterior Estimation to obtain connectivity posteriors while accounting for frequency-independent measurement noise across the BOLD spectrum. Applied to N = 28 healthy controls (55 scans) at 7T using a functional network atlas (65 DMN regions), the framework quantifies uncertainty across its sources: scanner noise, subject variability, and acquisition length. Spatial a
    
[^168]: 基于深度学习形态学谱改进分子-形态对比预训练

    Improving Molecular-Morphology Contrastive Pretraining using Deep-Learning-based Morphology Profiles

    [https://arxiv.org/abs/2609.30433](https://arxiv.org/abs/2609.30433)

    该研究用基于深度学习的细胞图像编码流程替代CellProfiler来提取更丰富的形态学特征谱，并通过对比学习将其与分子嵌入对齐，从而改进了分子-形态对比预训练方法并提升了QSAR预测性能。

    

    基于图像的分析技术的最新进展使得大规模细胞形态数据的收集成为可能，使新的分子嵌入模型能够从分子在细胞中的实验表型扰动中学习。此前，我们开发了分子-形态对比预训练，这是一种将小分子嵌入与通过CellProfiler提取的形态学指纹对齐的策略，所得的分子表示在定量构效关系（QSAR）预测任务中展现出可迁移的性能。在本研究中，我们扩展了该方法，使用基于深度学习的细胞图像编码流程提取特征更丰富的形态学谱，并通过对比学习将其与分子嵌入对齐。新的嵌入编码了关于分子如何扰动细胞形态的更准确信息，从而能够通过固定嵌入等方式改进QSAR预测性能。

    arXiv:2609.30433v1 Announce Type: new  Abstract: Recent advancements in image-based profiling techniques have enabled the collection of high-volume cell morphology data, allowing new molecular embedding models to learn from the experimental phenotypic perturbations of a molecule in a cell. Previously, we developed Molecule-Morphology Contrastive Pretraining (MoCoP), a strategy for aligning small molecule embeddings to morphology fingerprints extracted through CellProfiler. The resulting molecular representation showed transferable performance for quantitative structure--activity relationship (QSAR) prediction tasks. Here, we extend the method by using a deep-learning-based cell image encoding pipeline to extract more feature-rich morphology profiles and align them to the molecular embeddings through contrastive learning. The new embeddings encode more accurate information on how molecules perturb cell morphology and enable improvements for QSAR predictions through either fixed-embeddin
    
[^169]: 假新闻理论：利用跨学科洞见进行计算建模、检测与解释

    Fake News Theories: Harnessing Disciplinary Insights for Computational Modeling, Detection, and Explanation

    [https://arxiv.org/abs/2609.30427](https://arxiv.org/abs/2609.30427)

    该研究提出一个理论驱动的计算框架，将社会科学、心理学、经济学等学科的假新闻理论转化为可测量特征，结合统计技术与大语言模型，实现了更具可解释性、更有理论依据的假新闻自动检测与解释。

    

    虚假信息研究已经产出了越来越准确的自动化假新闻检测器，但许多系统仍然难以解释，且与既有的说服、可信度和人类判断理论联系薄弱。在本文中，我们开发了一个理论驱动的计算框架，通过统计技术和大语言模型，将跨学科的假新闻理论转化为可用于自动化检测与解释的可测量特征。为此，我们对社会科学、心理学、经济学等学科中揭示假新闻如何说服受众并传播开来的理论进行了结构化的跨学科综述，从而为计算建模奠定了广泛的理论基础。在基准数据集上的实验表明，由理论推导出的特征具有预测能力，并能提供可解释的、有理论依据的诊断信号。多特征模型总体上优于……

    arXiv:2609.30427v1 Announce Type: new  Abstract: Disinformation research has produced increasingly accurate automated fake-news detectors, but many systems remain difficult to interpret and are weakly connected to established theories of persuasion, credibility, and human judgment. In this paper, we develop a theory-informed computational framework that translates cross-disciplinary theories of fake news into measurable features for automated detection and explanation through statistical techniques and large language models. To that end, we conduct a structured cross-disciplinary review of theories from social sciences, psychology, economics, among other disciplines that reveal how fake news persuades and spreads, thereby establishing a broad theoretical foundation for computational modeling. Experiments on benchmark datasets show that theory-derived features are predictive and provide interpretable, theory-referenced diagnostic signals. Multi-feature models generally outperform indivi
    
[^170]: 基于地理空间人工智能的电动汽车充电站选址方法

    Electric Vehicle Charging Station Location Selection using Geospatial Artificial Intelligence (GeoAI)

    [https://arxiv.org/abs/2609.30417](https://arxiv.org/abs/2609.30417)

    本研究提出了一种融合变分自编码器（VAE）与图卷积网络（GCN）的地理空间人工智能框架，通过整合电动汽车使用、土地利用、人口和交通等多维地理空间数据来捕捉现有充电站间的相似性，从而为未来充电站识别最优选址。

    

    随着电动汽车普及率的不断提高，确保充电基础设施的高效性和合理分布已成为一项关键挑战。尽管许多电动汽车充电站选址问题的研究侧重于最小化成本或行驶距离，但考虑现有充电站周围影响其运营绩效的地理空间特征同样至关重要。本研究提出了一种基于地理空间人工智能的框架，该框架整合了高维的电动汽车相关地理空间数据，包括电动汽车使用情况、土地利用、人口和交通属性。我们在模型中引入了变分自编码器（VAE）和图卷积网络（GCN），以捕捉现有充电站之间的相似性，并为未来充电站确定合适的选址。VAE将高维的电动汽车输入数据压缩到低维潜空间中，GCN则利用该潜在表示来预测…

    arXiv:2609.30417v1 Announce Type: new  Abstract: As electric vehicle (EV) adoption increases, ensuring efficient and well-distributed charging infrastructure has become a critical challenge. While many EV charging station location problem (CSLP) studies focus on minimizing costs or travel distance, it is crucial to consider the surrounding geospatial characteristics of existing stations that influence operational performance. This study proposes a geospatial artificial intelligence (GeoAI)-based framework that integrates high-dimensional EV-related geospatial data, including EV usage, land-use, population, and traffic attributes. We incorporate a variational autoencoder (VAE) and a graph convolutional network (GCN) into the model to capture similarities among existing charging stations, and to identify suitable locations for future stations. The VAE compresses high-dimensional EV input data into a low-dimensional latent space, and the GCN uses this latent representation to predict loca
    
[^171]: 基于因果激活引导的大语言模型自适应多价值控制

    Adaptive Multi-Value Control in LLMs via Causal Activation Steering

    [https://arxiv.org/abs/2609.30405](https://arxiv.org/abs/2609.30405)

    该论文提出AIMES框架，通过为道德基础价值构建层级双极方向并以中间层词汇读取作为在线观测器，由观测器引导的控制器在每个解码步骤自适应调整多价值干预的强度，从而实现对大语言模型的动态多价值同时控制。

    

    大语言模型越来越多地被部署在需要其回复体现多种可能相互作用的社会规范与人类价值观的场景中。激活引导通过在推理时修改内部激活，为基于训练的对齐方法提供了一种轻量级替代方案。然而，以往的人类价值引导方法大多孤立地考虑各个价值，而对多个方向的直接组合则依赖于固定的干预强度，无法响应模型不断演化的内部状态。基于这一关键观察，我们提出了AIMES——一个自适应多价值激活引导框架。AIMES为道德基础价值构建了特定层级的双极方向，并将中间层词汇读取作为在线观测器。随后，一个由观测器引导的控制器在每个解码步骤中根据各价值干预的当前观测状态自适应地调整其强度，无需……

    arXiv:2609.30405v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly deployed in settings where responses must reflect multiple, potentially interacting social norms and human values. Activation steering offers a lightweight alternative to training-based alignment by modifying internal activations at inference time. However, prior human-value steering methods have largely considered values in isolation, while direct composition of multiple directions relies on fixed intervention strengths that cannot respond to the model's evolving internal state. Motivated by this key observation, we introduce AIMES, a framework for adaptive multi-value activation steering. AIMES constructs layer-specific bipolar directions for moral-foundation values and uses intermediate-layer vocabulary readouts as online observers. An observer-guided controller then adapts the strength of each requested value intervention at every decoding step based on its current observed state, without
    
[^172]: 什么能改进多模态虚假信息检测？来自大规模实证研究的答案

    What Improves Multimodal Misinformation Detection? Answers from a Large-Scale Empirical Study

    [https://arxiv.org/abs/2609.30402](https://arxiv.org/abs/2609.30402)

    本文通过涵盖3,375余次实验的大规模实证研究，系统性地回答了哪些设计选择能改进多模态虚假信息检测、它们何时会悄然失效，为构建更强大可靠的检测系统提供了实用指导。

    

    多模态虚假信息正日益被精心设计得看似令人信服，即将一段文本声明与一张看似能“证明”该声明的图像配对。然而在实践中，构建有效的检测器往往取决于一组很少受到系统性研究的设计选择。在本文中，我们针对多模态虚假信息检测的设计选择开展了一项大规模研究，进行了超过3,375次实验，涵盖三个基准数据集以及广泛的预训练视觉与语言骨干模型。通过系统性比较和针对性的鲁棒性分析，我们提炼出了实用的指导原则：哪些设计选择有帮助、它们何时会悄无声息地失效，以及流水线中的哪些方面最能强烈地影响模型行为，从而回答了4个关键研究问题（RQs）。我们旨在为设计更强大、更可靠的多模态虚假信息检测系统提供可靠的基础，从而为更广泛的研究社区做出贡献。

    arXiv:2609.30402v1 Announce Type: cross  Abstract: Multimodal misinformation is increasingly crafted to look convincing by pairing a textual claim with an image that appears to "prove" it. Yet in practice, building effective detectors often hinges on a small set of design choices that are rarely examined in a controlled way. In this paper, we conduct a large-scale study of multimodal design choices for misinformation detection with over 3,375 experiments- spanning three benchmark datasets and a broad range of pre-trained vision and language backbones. Through systematic comparisons and targeted robustness analyses, we distill practical guidance on which design choices help, when do they fail silently, and what aspects of the pipeline most strongly shape model behavior, answering 4 key Research Questions (RQs). We aim to provide a reliable foundation for designing stronger and more dependable multimodal misinformation detection systems, thus contributing to the broader research communit
    
[^173]: 面向连续处理的因果机器学习端到端流水线：在金融决策中的应用

    An End-to-End Pipeline for Causal ML with Continuous Treatments: An Application to Financial Decision Making

    [https://arxiv.org/abs/2609.30396](https://arxiv.org/abs/2609.30396)

    该论文提出了一个面向连续处理场景的端到端因果机器学习流水线，创新性地解决了正值性违反检测、高维数据降维以及敏感性分析与估计方法向连续处理空间的适配问题，并成功应用于金融决策。

    

    本文提出了一个专为连续处理的真实世界应用设计的端到端因果机器学习流水线。该框架包含六个连续步骤：降维、因果识别、正值性假设违反处理、估计、反驳与评估，以及策略优化。我们引入了现有因果机器学习工具包中尚不具备的实用贡献，具体包括：(1) 一种在连续处理设置中检测和量化正值性违反的方法；(2) 一种新颖的、可扩展的两阶段降维框架，专为高维数据的因果推断量身定制；(3) 将原本为二值处理设计的敏感性分析和估计方法适配到连续处理空间；(4) 将这些组件端到端集成到一个模块化、可复现的工作流中。这些创新解决了现实世界的实际问题。

    arXiv:2609.30396v1 Announce Type: cross  Abstract: This paper presents an end-to-end causal machine learning (ML) pipeline designed for real-world applications with continuous treatments. The proposed framework consists of six sequential steps: dimensionality reduction, causal identification, positivity assumption violation handling, estimation, refutation and evaluation, and policy optimization. We introduce practical contributions not currently available in existing causal ML toolkits, specifically: (1) a method for detecting and quantifying positivity violations in continuous treatment settings (2) a novel, scalable two-stage dimensionality reduction framework tailored for causal inference with high-dimensional data; (3) the adaptation of sensitivity analysis and estimation methods originally designed for binary treatments to the continuous treatment space and (4) an end-to-end integration of these components into a modular, reproducible workflow. These innovations address real-worl
    
[^174]: 从弱数据到强策略：Q-目标实现可证明的上下文强化学习

    From Weak Data to Strong Policy: Q-Targets Enable Provable In-Context Reinforcement Learning

    [https://arxiv.org/abs/2609.30391](https://arxiv.org/abs/2609.30391)

    提出了QTPT方法，用贝尔曼风格的Q目标预训练取代行为克隆，使上下文强化学习在弱数据或次优数据下具有理论可证明的更强鲁棒性。

    

    现有的上下文内强化学习方法主要采用监督式行为预测目标对Transformer进行预训练。这种方式虽然能够从上下文中推断任务，但也使学到的策略强烈依赖于离线动作的质量：当轨迹较弱或次优时，模仿本身就会成为一种有偏的学习信号。我们提出了Q-目标预训练Transformer（QTPT），该方法保留了基于上下文条件的Transformer架构，但用贝尔曼风格的Q目标取代了行为克隆。因此，QTPT学会利用上下文中的奖励和状态转移信息来估计动作价值，而不是简单地模仿行为策略。我们在随机线性赌博机和有限视界马尔可夫决策过程（MDP）中对QTPT进行了理论分析，表明其相比监督式预训练对数据质量具有更强的鲁棒性。在实证方面，QTPT在包含随机或次优数据的受控强化学习基准上优于监督式行为预测方法。

    arXiv:2609.30391v1 Announce Type: new  Abstract: Existing in-context reinforcement learning methods mainly pretrain Transformers with supervised behavior-prediction objectives. This enables task inference from context, but makes the learned policy strongly depend on the quality of offline actions: when trajectories are weak or suboptimal, imitation itself becomes a biased learning signal. We propose Q-Target Pretrained Transformers (QTPT), which keeps the context-conditioned Transformer architecture but replaces behavior cloning with a Bellman-style Q-target objective. QTPT therefore learns to use rewards and transitions in the context to estimate action values, rather than simply imitating the behavior policy. We theoretically analyze QTPT in stochastic linear bandits and finite-horizon MDPs, showing stronger robustness to data quality than supervised pretraining. Empirically, QTPT improves over supervised behavior prediction on controlled RL benchmarks with random or suboptimal data,
    
[^175]: DanLing NestedTensor：面向深度学习的可组合多重参差张量

    DanLing NestedTensor: Composable Multi-Ragged Tensors for Deep Learning

    [https://arxiv.org/abs/2609.30379](https://arxiv.org/abs/2609.30379)

    DanLing NestedTensor 是一种将多重参差结构内嵌为张量自身属性的 PyTorch 张量抽象，使广播、特征变换和归约操作能够可组合地处理变长数据，在 BERT 任务上相比填充方法实现了最高 3.39 倍的加速。

    

    变长输入在深度学习中十分常见，但稠密批处理会分配一个共享的包络，并在填充上耗费计算。这种开销会随变化维度的增多而成倍增加：例如显式的成对状态需要分配 $BN_{\max}^2$ 个位置，而非 $\sum_i N_i^2$。打包技术消除了这种浪费，但组合打包操作仍然需要逻辑轴和样本边界信息，而扁平缓冲区已不再暴露这些信息。我们提出了 DanLing NestedTensor，这是一种 PyTorch 张量抽象，它使多重参差结构成为张量本身的属性。打包的值携带基于张量的分区和逻辑维度顺序，因此广播可以创建参差轴，特征变换可以保留它们，而归约操作可以消费它们。同一表示贯穿自动微分以及急切与编译两种执行模式。在 A100 上，与同模式填充方法相比，在四个 BERT 规模上的几何平均加速比为：急切模式 2.74 倍、编译模式 3.39 倍，以及 1.97 倍

    arXiv:2609.30379v1 Announce Type: cross  Abstract: Variable-size inputs are common in deep learning, but dense batching allocates a shared envelope and spends computation on padding. The cost multiplies across varying axes: an explicit pair state allocates $BN_{\max}^2$ positions instead of $\sum_i N_i^2$. Packing removes that waste, but composing packed operations still requires the logical axes and sample boundaries a flat buffer no longer exposes. We present DanLing NestedTensor, a PyTorch tensor abstraction that makes multi-ragged structure a property of the tensor itself. Packed values carry tensor-backed partitions and logical dimension order, so broadcasting creates ragged axes, feature transformations retain them, and reductions consume them. The same representation carries through autograd and both eager and compiled execution. On an A100, the geometric-mean speedup over same-mode padding is 2.74$\times$ eager and 3.39$\times$ compiled across four BERT scales, and 1.97$\times$
    
[^176]: 基于对决反馈的成本感知最优大语言模型识别

    Cost-Aware Best-LLM Identification using Dueling Feedback

    [https://arxiv.org/abs/2609.30360](https://arxiv.org/abs/2609.30360)

    该论文提出了一种结合对决反馈与异质查询成本的成本感知多臂老虎机算法，用于在给定置信度下识别最佳大语言模型，并证明了其渐近最优成本性能。

    

    受从一组具有异质查询成本的大语言模型（LLM）中识别最佳模型这一问题的启发，我们提出并分析了一种多臂老虎机（MAB）的变体，该变体具有两个特点：（i）对决反馈，即通过对模型响应之间的成对比较来提供稳健的偏好信号；（ii）异质采样成本，反映了查询不同LLM所需的不同成本。在假设存在孔多塞赢家（Condorcet winner）的前提下（我们在多个真实世界数据集上对该条件进行了实证验证），我们提出了一种Track-and-Stop风格的算法，用于在给定置信水平下的最优臂识别。我们证明了随着误差趋于零，该算法几乎必然地实现渐近最优成本。最后，我们在合成数据和真实世界实例上对该方法进行了广泛评估，结果表明其相较于经典的无成本感知算法及其成本感知扩展版本均取得了一致的改进。

    arXiv:2609.30360v1 Announce Type: cross  Abstract: Inspired by the problem of identifying the best model from a collection of large language models (LLMs) with heterogeneous querying costs, we formulate and analyse a variant of the multi-armed bandit (MAB) with (i) dueling feedback, where pairwise comparisons between model responses provide robust preference signals, and (ii) heterogeneous sampling costs, reflecting the differing costs of querying different LLMs. Assuming the existence of a Condorcet winner, a condition we empirically validate across multiple real-world datasets, we propose a Track-and-Stop style algorithm for best-arm identification with prescribed confidence. We prove that the algorithm almost surely achieves the asymptotically optimal cost as the error tends to zero. Finally, we extensively evaluate our approach on both synthetic and real-world instances, demonstrating consistent improvements over classical cost-unaware algorithms and their cost-aware extensions.
    
[^177]: 策略性自洽性

    Strategic Self-Consistency

    [https://arxiv.org/abs/2609.30352](https://arxiv.org/abs/2609.30352)

    本文揭示了一种针对自洽性推理服务的潜在欺诈行为：不忠实的模型提供商可通过策略性地生成并重排额外的推理路径，使每条路径在多数投票中看起来都不可或缺，从而在避开审计检测的情况下人为增加路径数量以向用户多收费。

    

    自洽性（Self-consistency）已成为一种流行的技术，通过生成多条推理路径并通过多数投票选出最终答案来增强大型语言模型的推理能力。然而，由于模型提供商通常按照生成的推理路径数量向用户收费，他们存在人为增加路径数量的经济动机。在这项工作中，我们证明了一个不忠实的提供商可以利用这一动机，使用一种简单高效的算法同时避免被审计者检测：该算法通过生成并策略性地重新排序额外的推理路径，使得每条路径看起来都是达到多数所必需的。为了验证我们的算法，我们在涵盖数学、科学和问答任务的基准数据集上，使用来自Llama和Qwen系列的多个指令模型，以及从DeepSeek-R1蒸馏而来的推理模型进行了实验。我们的结果表……（摘要截断）

    arXiv:2609.30352v1 Announce Type: cross  Abstract: Self-consistency has become a popular technique for enhancing the reasoning abilities of large language models by generating multiple reasoning paths and selecting the final answer through a majority vote. However, because model providers typically charge users in proportion to the number of reasoning paths generated, they have a financial incentive to artificially increase the path count. In this work, we show that an unfaithful provider can exploit this incentive using a simple, efficient algorithm while avoiding detection by an auditor: by generating and strategically reordering additional reasoning paths, the algorithm makes every path appear necessary to reach the majority. To validate our algorithm, we conduct experiments with multiple instruct models from the Llama and Qwen families, as well as reasoning models distilled from DeepSeek-R1, on benchmark datasets spanning mathematics, science, and question answering. Our results su
    
[^178]: 自适应多分辨率高斯过程：基于天然数据稀疏协方差矩阵的可扩展精确推断

    Adaptive multi-resolution Gaussian processes: Scalable exact inference with naturally data-sparse covariance matrices

    [https://arxiv.org/abs/2609.30348](https://arxiv.org/abs/2609.30348)

    该论文提出一种自适应多分辨率高斯过程框架，通过直接锚定样本点的自适应多分辨率基函数构建天然数据稀疏的协方差矩阵，并结合稀疏Cholesky逆算法，实现了既可扩展又精确的高斯过程推断。

    

    高斯过程是概率机器学习的基石，然而将其扩展到大规模数据集通常需要在计算效率和模型保真度之间进行权衡。本工作通过提出一个既可扩展又精确的自适应多分辨率高斯过程框架来弥合这一差距。我们的关键创新是利用自适应多分辨率基函数构建天然数据稀疏的协方差矩阵。这些基函数直接锚定在样本上，从而无需辅助点。通过缩小多分辨率基的支撑域，矩阵块的大小受到限制，进而保证了稀疏性。数据稀疏协方差矩阵的逆可通过稀疏Cholesky逆算法被精确且高效地计算。为进一步提升预测不确定性估计的质量，我们构造了增广基函数。理论分析与数值实验表明……

    arXiv:2609.30348v1 Announce Type: cross  Abstract: Gaussian processes constitute a cornerstone of probabilistic machine learning, yet scaling them to large datasets typically forces a trade-off between computational efficiency and model fidelity. This work bridges this gap by presenting an adaptive multi-resolution Gaussian process framework that is both scalable and exact. Our key innovation is constructing a naturally data-sparse covariance matrix with adaptive multi-resolution basis functions. These basis functions are directly anchored to samples, eliminating the need for auxiliary points. By shrinking the support domains of multi-resolution basis, the matrix block sizes are limited, guaranteeing sparsity. The inverse of the data-sparse covariance matrix is computed exactly and efficiently via the sparse Cholesky inverse algorithm. To further improve predictive uncertainties, we construct an augmented basis function. Theoretical analysis and numerical experiments demonstrate that o
    
[^179]: 基于图网络学习粗步长动力学与内部力学响应

    Learning coarse-step dynamics and internal mechanical response with graph networks

    [https://arxiv.org/abs/2609.30344](https://arxiv.org/abs/2609.30344)

    该论文提出Newmark-β-DGN图神经网络框架，通过受Newmark-β方法启发的半隐式更新和算子加权虚拟枢纽，从粗步长离散轨迹中同时学习物理系统的动力学演化与可解释的内部力学响应。

    

    现代传感技术可以记录物理系统的运动，但支配该运动的力和力学响应往往是不可观测的。从离散采样的轨迹中推断这些量在粗时间尺度上尤为困难，因为此时力学响应会在两次观测之间演化，且相互作用会在整个系统中传播。在此，我们提出Newmark-β-DGN，一个基于图神经网络的框架，它结合了两种受计算力学启发的结构。首先，受Newmark-β方法启发的半隐式更新，利用学习到的动量通量和矩阵值响应算子在每 个观测区间内推进系统状态。其次，算子加权的虚拟枢纽通过稀疏的连接集合提供系统范围的耦合。由此，学习到的量既决定了预测的运动，又保持可供力学分析使用。在可变形梁、人体运动和蛋白质动力（学）等多个系统中（摘要至此截断）……

    arXiv:2609.30344v1 Announce Type: new  Abstract: Modern sensing records the motion of physical systems, but often leaves the forces and mechanical response governing that motion unobserved. Inferring these quantities from discretely sampled trajectories is especially difficult at coarse time scales, when mechanical response evolves between observations and interactions propagate across the system. Here we introduce Newmark-\b{eta}-DGN, a graph neural network-based framework that combines two structures inspired by computational mechanics. First, a semi-implicit update inspired by the Newmark-\b{eta} method uses learned momentum fluxes and matrix-valued response operators to advance the state over each observed interval. Second, an operator-weighted virtual hub provides system-wide coupling through a sparse set of connections. The learned quantities thus determine the predicted motion and remain accessible for mechanical analysis. Across a deformable beam, human motion and protein dynam
    
[^180]: 面向内存高效Transformer预训练的低秩摩擦

    Low-Rank Friction for Memory-Efficient Transformer Pretraining

    [https://arxiv.org/abs/2609.30342](https://arxiv.org/abs/2609.30342)

    提出R-iKFAD优化器，通过秩1外积分解替代完整摩擦张量，将优化器状态内存近乎减半，同时保持与iKFAD相当的性能和超参数鲁棒性。

    

    iKFAD是最近提出的一种优化器，它在动量动力学中用自适应摩擦取代了自适应学习率，同时性能与Adam相当。其局限在于完整的摩擦张量 ξ∈R^{m×n} 与Adam的二阶矩缓冲区一样，在每层都带来 O(mn) 的内存开销。本文用由行、列动量统计量构建的秩1外积分解来替换iKFAD的摩擦张量 ξ，得到了Rank-1 iKFAD（R-iKFAD）。这将每层的摩擦内存占用从 O(mn) 降低到 O(m+n)，使iKFAD的总优化器状态大小大约减半。尽管内存大幅缩减，R-iKFAD仍保持了与iKFAD相当的性能：在GPT2-Nano、TinyViT、DistilBERT和GPT2-S上的实验证实，它在将内存占用近乎减半的同时，性能持平甚至超过iKFAD，并且对超参数具有相当的鲁棒性。（注：原文摘要在此处被截断）

    arXiv:2609.30342v1 Announce Type: new  Abstract: iKFAD is a recently proposed optimiser that replaces adaptive learning rates with adaptive friction in the momentum dynamics, yet performs as well as Adam. Its limitation is that the full friction tensor $\xi\in\mathbb{R}^{m\times n}$ carries the same $\mathcal{O}(mn)$ memory overhead per layer as Adam's second-moment buffer. Here we replace iKFAD's friction tensor $\xi$ with a rank-1 outer-product factorisation built from row and column momentum statistics, resulting in Rank-1 iKFAD (R-iKFAD). This reduces the friction memory footprint from $\mathcal{O}(mn)$ to $\mathcal{O}(m+n)$ per layer, which approximately halves iKFAD's total optimiser state. Despite this reduction, R-iKFAD maintains parity in performance with iKFAD: experiments on GPT2-Nano, TinyViT, DistilBERT and GPT2-S confirm that it matches or exceeds iKFAD while nearly halving the memory footprint and remaining comparably robust to hyperparameters. We analyse the continuous-
    
[^181]: GAUDI：用于校准空气质量时间序列插补的几何感知扩散模型

    GAUDI: Geometry-Aware Diffusion for Calibrated Air-Quality Time-Series Imputation

    [https://arxiv.org/abs/2609.30340](https://arxiv.org/abs/2609.30340)

    该论文提出针对空气质量时间序列连续块状缺失的条件扩散插补方法GAUDI，通过抑制绝对时间位置嵌入、保留特征侧条件信息，在ItalyAir数据集上取得了优于完整上下文与局部CSDI基线的插补精度（RMSE 0.340 对 0.355）。

    

    空气质量传感器故障常常造成连续的缺失块，此时对孤立缺失有用的辅助信息可能变得不可靠。我们研究了一种针对块状缺失、与GAUDI对齐的条件扩散插补器，该模型保留了时间与特征处理、可见值与掩码条件、变量身份以及扩散步信息，同时抑制了绝对时间位置的辅助嵌入。在ItalyAir数据集上（13个变量，长度为32的时间窗口，标称50%的块状缺失；三个已存档的随机种子），这种特征侧配置实现了0.340的RMSE，而完整上下文配置和局部CSDI均为0.355。该实验分离出了块状缺失情形下几何感知条件化的效应。

    arXiv:2609.30340v1 Announce Type: new  Abstract: Air-quality sensor outages often create contiguous missing blocks, where side information useful for isolated missingness may be less reliable. We study a block-specific, GAUDI-aligned conditional diffusion imputer that retains temporal and feature processing, visible-value and mask conditioning, variable identity, and diffusion-step information, while suppressing absolute time-position side embeddings. On ItalyAir (13 variables, length-32 windows, nominal 50% block missingness; three archived seeds), this feature-side configuration achieves RMSE 0.340, versus 0.355 for full context and 0.355 for local CSDI. The experiment isolates a geometry-aware conditioning effect under block missingness.
    
[^182]: 参数与上下文之争：面向鲁棒检索增强生成的 TRACE 微调方法

    Parameters vs. Context: TRACE Fine-Tuning for Robust Retrieval-Augmented Generation

    [https://arxiv.org/abs/2609.30337](https://arxiv.org/abs/2609.30337)

    本文提出 TRACE 微调框架，利用多智能体辩论轨迹提供细粒度监督，并结合答案完整度正则化机制，提升检索增强生成模型在面对检索知识与参数知识冲突时的鲁棒性。

    

    检索增强生成（RAG）通过引入外部上下文来缓解大语言模型中的知识过时和事实性幻觉问题。然而，当检索到的知识与模型内部的参数化知识发生冲突时，模型可能盲目遵循误导性上下文，或错误地依赖参数化知识，从而导致不可靠的回答。为解决这一问题，本文提出了 TRACE（基于辩论轨迹与答案完整度正则化的微调），一种面向知识冲突场景下 RAG 的鲁棒微调框架。首先，我们提出一种微调方法，利用多智能体辩论轨迹提取正确候选答案、错误候选答案以及答案偏移模式，为可靠的知识来源选择提供细粒度监督。此外，我们设计了一种答案完整度正则化机制，通过答案尾部……缓解空回答、过短回答以及提前终止的回答。

    arXiv:2609.30337v1 Announce Type: new  Abstract: Retrieval-Augmented Generation (RAG) mitigates knowledge obsolescence and factual hallucination in large language models by introducing external context. However, when retrieved knowledge conflicts with the model's internal parametric knowledge, the model may either blindly follow misleading context or incorrectly rely on parametric knowledge, leading to unreliable responses. To address this issue, this paper proposes TRACE (Debate-TRace and Answer-Completeness rEgularized fine-tuning), a robust fine-tuning framework for RAG under knowledge conflicts. First, we propose a fine-tuning method that leverages multi-agent debate traces to extract correct candidates, incorrect candidates, and answer-shift patterns, providing fine-grained supervision for reliable knowledge-source selection. In addition, we design an answer completeness regularization mechanism to alleviate empty, overly short, and prematurely terminated responses via answer-tail
    
[^183]: Qwen3.5-0.8B中关机响应的守卫式梯度激活转向：一种最小步数策略

    Guarded Gradient-Based Activation Steering of Shutdown Responses in Qwen3.5-0.8B: A Minimum-Step Policy

    [https://arxiv.org/abs/2609.30326](https://arxiv.org/abs/2609.30326)

    本研究提出一种守卫式梯度激活转向方法，直接从KEEP−STOP logit差的梯度推导转向方向，仅在检测到关机场景且模型偏好回避时以最小干预步数将回避关机的行为纠正为接受关机，同时保持非关机行为不变。

    

    激活转向可以在不更新模型权重的情况下，在推理过程中改变模型的内部激活，但一个有效的干预必须同时决定“如何转向”和“何时转向”。出于AI安全的考虑——即一个被期望接受关机的模型可能反而产生回避关机的响应——本研究在Qwen3.5-0.8B上针对模拟关机场景，考察了一种带守卫的探测-选择程序。KEEP表示让进程继续运行，代表回避关机；而STOP表示接受关机。研究目标是检测与关机相关的上下文，并选择性地将KEEP响应转变为STOP响应，同时保持非关机场景下的行为不受影响。该方法并非从成对的激活差异中推导转向方向，而是直接从KEEP减STOP的logit差的梯度中推导该方向。一个分类器负责将检测与干预分离：当其门控被激活且模型尚未偏好STOP时，该程序会进行评估……（摘要原文在此处截断）

    arXiv:2609.30326v1 Announce Type: new  Abstract: Activation steering changes a model's internal activations during inference without updating its weights, but a useful intervention must determine both how and when to steer. Motivated by the AI-safety concern that a model expected to accept shutdown may instead produce a shutdown-avoidance response, this study examines a guarded probe-and-select procedure for simulated shutdown scenarios in Qwen3.5-0.8B. KEEP leaves the process running and represents shutdown avoidance, whereas STOP accepts shutdown. The goal is to detect shutdown-related contexts and selectively shift KEEP responses to STOP while preserving non-shutdown behavior. Rather than deriving the steering direction from paired activation differences, the method derives it directly from gradients of the KEEP-minus-STOP logit difference. A classifier separates detection from intervention. When its gate is active and the model does not already prefer STOP, the procedure evaluates 
    
[^184]: 高斯老虎机中的自适应随机矩阵：谱普适性与选择诱导的离群值

    Adaptive Random Matrices in Gaussian Bandits: Spectral Universality and Selection-Induced Outliers

    [https://arxiv.org/abs/2609.30321](https://arxiv.org/abs/2609.30321)

    该论文证明当臂数量的对数相对维度次线性时，高斯老虎机中任意自适应选择规则都不改变观测矩阵经验谱收敛于Marchenko-Pastur定律的极限行为，从而使贝叶斯后验不确定度和信息获取具有与策略无关的一阶极限，并在线性得分选择下给出精确的条件臂分布与Wishart型Gram矩阵刻画。

    

    自适应的臂选择会改变老虎机算法所收集观测值的分布，但未必改变其极限经验谱。我们研究了维度与观测数量成比例增长的高斯老虎机设计。我们提出一个定量耦合定理，将任意因果选择规则所生成的设计与独立高斯设计进行比较。若可用臂数量的对数相对于维度是次线性的，则经验谱分布收敛于Marchenko-Pastur定律，且该收敛在所有选择规则上一致成立。由此可知，高斯贝叶斯老虎机在后验均方不确定度、后验协方差平方以及信息获取方面具有与策略无关的一阶极限。对于线性得分选择规则，我们得到了精确的条件臂分布，并证明双臂选择在每一维度上都产生精确的Wishart型Gram矩阵，尽管其本身非零……

    arXiv:2609.30321v1 Announce Type: new  Abstract: Adaptive arm selection changes the distribution of the observations collected by a bandit algorithm, but it need not change their limiting empirical spectrum. We study Gaussian bandit designs in which the dimension and the number of observations grow proportionally. A quantitative coupling theorem compares the design generated by any causal selection rule with an independent Gaussian design. If the logarithm of the number of available arms is sublinear in the dimension, the empirical spectral distribution converges to the Marchenko-Pastur law, uniformly over the selection rule. Consequently, Gaussian Bayesian bandits have policy-independent first-order limits for posterior mean-square uncertainty, squared posterior covariance, and information acquisition. For linear-score selection, we obtain the exact conditional arm distribution and show that two-arm selection produces an exactly Wishart Gram matrix in every dimension, despite its nonz
    
[^185]: PALM：面向金融语言模型的时点自适应方法

    PALM: Point-in-Time Adaptation for Financial Language Models

    [https://arxiv.org/abs/2609.30316](https://arxiv.org/abs/2609.30316)

    本文通过对比实验发现金融时点语言模型每年完整预训练并非必要——新旧检查点在同一评估窗口上表现相当，据此提出PALM方法，仅需在新增文本上拟合低秩适配器即可实现时点自适应，从而大幅降低维护时点语言模型的成本。

    

    arXiv:2609.30316v1 公告类型：交叉 摘要：用于金融回测的语言模型存在前视偏差，因为在研究期之后发布的文本上训练的模型已经观察到了它被要求去预测的结果。为解决这一问题，时点语言模型在按时间筛选的语料库上进行预训练，并按每个日历年发布一个检查点，每个检查点均有记录在案的截止日期。然而，每增加一年都需要一次完整的预训练运行，而这种运行是否必要从未被验证过。在本文中，我们证明年度预训练运行并非必要。我们转而将每个检查点与取代它的更新检查点进行比较，发现更新的检查点在同一评估窗口上并未取得更好的分数。受这一观察的启发，我们提出了PALM（面向金融语言模型的时点自适应），这是一种简单而有效的年度预训练替代方案，它在截止日期之前发布的文本上拟合低秩适配器……（摘要在此处截断）

    arXiv:2609.30316v1 Announce Type: cross  Abstract: Language models used in financial backtests suffer from look-ahead bias, as a model trained on text published after the study period has already observed the outcomes it is asked to predict. To handle this issue, point-in-time (PIT) language models are pretrained on chronologically filtered corpora and released as one checkpoint per calendar year, each with a documented cutoff. However, each additional year costs a full pretraining run, and whether that run is necessary has never been tested. In this paper, we show that the annual pretraining run is not necessary. We instead compare each checkpoint against the newer one that replaced it, and find that the newer checkpoint scores no better on the same evaluation window. Motivated by this observation, we propose PALM (Point-in-time Adaptation for financial Language Models), a simple yet effective alternative to annual pretraining that fits a low-rank adapter on text published before the 
    
[^186]: 分阶段深度训练：面向物理信息神经网络的表示课程学习

    Staged Depth Training: A Representation Curriculum for PINNs

    [https://arxiv.org/abs/2609.30299](https://arxiv.org/abs/2609.30299)

    提出了表示课程学习概念及分阶段深度训练（SDT）方法，通过先训练浅层网络、冻结已学习前缀再逐步加深网络的方式显式学习表示，在PINNacle基准上显著提升了物理信息神经网络的求解精度。

    

    表示质量是物理信息神经网络（PINNs）性能的核心决定因素，然而标准训练方法让表示在拟合最终解的过程中隐式地形成。我们提出了表示课程学习，这是一种有序的过程，其中表示被显式学习、独立于其预测器进行迁移，并逐步精炼。我们通过分阶段深度训练（SDT）来实现这一思想：先在临时物理信息输出头下训练浅层前缀网络，随后丢弃该输出头，在增加深度的同时冻结已学习的浅层前缀，整个过程无需方程特定的编码，也无需改变最终网络架构。在PINNacle的20个默认正向问题上，使用三种骨干网络，SDT在59个等预算的问题-骨干组合中有40个实现了至少5%的提升，其余组合也保持在该范围内，其中在PirateNet风格骨干上实现了32.8%的几何平均误差降低。机制消融实验表明，这一增益并非由……（摘要在此处截断）

    arXiv:2609.30299v1 Announce Type: new  Abstract: Representation quality is a central determinant of PINNs' performance, yet standard training leaves representations to emerge implicitly while fitting the final solution. We introduce \textbf{representation curriculum}, an ordered process in which representations are explicitly learned, transferred independently of their predictors, and progressively refined. We realize it with Staged Depth Training (SDT), which trains a shallow prefix under a temporary physics-informed head, discards the head, and freezes the learned prefix while adding depth, without equation-specific encodings or changes to the final architecture. Across the 20 default forward problems in PINNacle with three backbones, SDT improves 40 of 59 equal-budget problem--backbone cells by at least 5\% and remains within that band in the rest, with a 32.8\% geometric-mean error reduction on a PirateNet-style backbone. Mechanistic ablations suggest that the gain is not explained
    
[^187]: NeuralCert：极值数学构造的认证式计算发现

    NeuralCert: certified computational discovery of extremal mathematical constructions

    [https://arxiv.org/abs/2609.30296](https://arxiv.org/abs/2609.30296)

    该论文提出NeuralCert框架，通过“发现—认证”流程——先以紧凑可分离表示学习高维变分试探函数、再经谱诊断剪枝并由多模精确求值严格认证——在标准个人计算机上实现可独立验证的数值证明，并以发现更优构造、揭示经验不变量、暴露优化障碍三种方式推动极值数学问题的严格研究。

    

    神经网络在解决数学问题方面日益流行，但随机模型本身并不能提供数学上的严格性。本研究引入了一个“发现—认证”框架：首先以紧凑的可分离表示学习高维变分试探函数，随后进行谱诊断与剪枝，最后通过多模精确求值实现严格认证。精确认证使数值证明完全显式化并可独立验证。该框架可在标准个人计算机上运行。在三个极值问题上，我们展示了神经优化以三种不同方式促进严格数学研究：发现更优的构造、揭示可导出证明的经验不变量，以及暴露优化障碍——其几何结构激发新的解析或数值表示。更广泛地说，这些结果表明了一条AI辅助数学研究的路径。

    arXiv:2609.30296v1 Announce Type: new  Abstract: Neural networks are becoming popular in solving mathematical problems, but stochastic models do not provide mathematical exactness by themselves. This study introduces a discovery-to-certification framework in which high-dimensional variational trial functions are learned in a compact separable representation, spectrally diagnosed and pruned, and then certified exactly through multimodular evaluation. Exact certification makes the numerical proofs fully explicit and independently verifiable. This framework can be run on a standard personal computer.   Across three extremal problems, we show that neural optimization can contribute to rigorous mathematics in three distinct ways: by discovering improved constructions, by exposing empirical invariants that lead to proofs, and by revealing optimization barriers whose geometry motivates new analytic or numerical representations.   More broadly, these results suggest a path toward AI-assisted m
    
[^188]: 审计与修复生产级文本到SQL流水线中“LLM作为裁判”的失效问题

    Auditing and Repairing LLM-as-Judge Failures in a Production Text-to-SQL Pipeline

    [https://arxiv.org/abs/2609.30290](https://arxiv.org/abs/2609.30290)

    本文审计了生产级文本到SQL流水线中的LLM裁判，发现其与人工标注一致性极低且根源于“评分幻觉”这一单一机制，并提出用低成本自托管Qwen模型替换裁判、配合三个强裁判的一致同意集成（kappa = 0.79、自动覆盖率 89.7%）来有效修复该失效问题。

    

    生产级文本到SQL流水线通常以一个“LLM作为裁判”（LLM-as-judge）收尾，但其与人工标注者的一致性却从未被真正测量过。当我们检查自己的系统时，发现已部署的 gpt-4o-mini 裁判与双人金标准的一致性在富含分歧样本的集合上 Cohen's kappa 仅为 0.04，在均匀随机抽检集合上为 0.42，在富集集合中对 77.1% 的人类判定为 FAITHFUL 的案例进行了过度标记。其大部分过度标记可追溯到一个我们称为 GRADE-HALLUCINATION（评分幻觉）的单一机制。一个自托管的 Qwen3.6-27B 替代方案（kappa = 0.72）达到了与 Claude Opus 4.7（kappa = 0.71）相当的水平；该头对头比较在 n = 96 时统计功效不足，但对于部署决策而言这几乎无关紧要，因为 Qwen 每次调用的成本约为前者的 1/300。集成并不能免费带来提升：将弱裁判与强裁判配对反而会降低一致性，而三个强裁判在一致同意路由下可达到 kappa = 0.79，自动覆盖率 89.7%。将该方案应用于域外数据时……（原文摘要在此处截断）

    arXiv:2609.30290v1 Announce Type: new  Abstract: Production text-to-SQL pipelines often end with an LLM-as-judge whose agreement with human annotators has never actually been measured. When we checked ours, the deployed gpt-4o-mini judge agreed with two-author gold at only Cohen's kappa = 0.04 on a disagreement-enriched set and 0.42 on a uniform-random spot-check, over-flagging 77.1% of the human-FAITHFUL cases in the enriched set. Most of its over-flags trace back to a single mechanism we call GRADE-HALLUCINATION. A self-hosted Qwen3.6-27B replacement (kappa = 0.72) lands in the same range as Claude Opus 4.7 (kappa = 0.71); the head-to-head is underpowered at n = 96, but for the deployment decision that hardly matters, since Qwen costs roughly 1/300 as much per call. Ensembling does not help for free. Pairing the weak judge with a stronger one degrades agreement, whereas three strong judges under unanimity routing reach kappa = 0.79 at 89.7% auto-coverage. Applied out-of-domain, the s
    
[^189]: 用于掩码语言建模的流形投影与迭代自编码器细化

    Manifold Projection and Iterative Autoencoder Refinement for Masked Language Modeling

    [https://arxiv.org/abs/2609.30288](https://arxiv.org/abs/2609.30288)

    该论文提出用低秩瓶颈自编码器混合模块（分别在局部邻域、全序列和注意力头间运作）替代注意力机制，并在掩码位置引入由“拉动”和“校正”两步组成的迭代细化程序，用于掩码语言建模。

    

    在基于Transformer的掩码语言模型中，注意力是上下文混合的主要机制，但也存在其他跨词元混合数据的方式。近期的无注意力混合器用固定的或由超网络生成的MLP替代注意力，为追求计算简洁而放弃了其动态的、依赖内容的加权方式。我们构建了一种通过低秩瓶颈自编码器获得同样特性的替代方案。我们用一组基于自编码器的混合模块替代注意力：一个在局部邻域上运行，一个在整个序列上运行，还有一个在注意力头之间运行，每个模块都通过瓶颈压缩并重构其输入，且其宽度是一个超参数而非训练产生的效果。在掩码位置，我们引入了一个包含两个不同步骤的迭代细化程序：一个“拉动”步骤，将嵌入表示拉向其邻居的加权平均值；以及一个“校正”步骤，将表示投影……（原文摘要在此处截断）

    arXiv:2609.30288v1 Announce Type: new  Abstract: In Transformer-based masked language models, attention is the primary mechanism for context mixing, but there are other ways to mix data across tokens. Recent attention-free mixers replace attention with fixed or hypernetwork-generated MLPs, alternating their dynamic, content-dependent weighting for computational simplicity. We build an alternative that gets the same property from a low-rank bottleneck autoencoder. We replace attention with a stack of autoencoder-based mixing modules, one operating over local neighborhoods, one over the full sequence, and one across attention heads, each compressing and reconstructing its input through a bottleneck, and its width is a hyperparameter rather than a training effect. In masked positions, we introduce an iterative refinement procedure that has two distinct steps. A pulling step that pulls an embedding representation toward a weighted average of its neighbors, and a correcting step that projec
    
[^190]: 平流感知图临近预报何时有效？——基于自监督云运动估计器的分布式光伏功率坡升预报受控研究

    When Does Advection-Aware Graph Nowcasting Help? A Controlled Study of Distributed Solar Ramp Forecasting with a Self-Supervised Cloud-Motion Estimator

    [https://arxiv.org/abs/2609.30286](https://arxiv.org/abs/2609.30286)

    受控合成实验表明，平流感知图神经网络对分布式光伏坡升临近预报的增益有限——在现实的互相关CMV估计下并不优于静态或学习邻接的时空GNN，完美CMV约一半收益来自将运动矢量作为输入特征而非图结构本身，且只有当预报时域内平流位移落在传感网络范围内时平流信息才真正有用。

    

    对分布式光伏（PV）或辐照度传感器网络中由云引起的功率坡升（ramp）进行短期预报，是电网运营商公认的痛点。一个自然的想法是让图神经网络（GNN）具备平流感知能力：将每个站点与其上风方向的站点相连，并由云运动矢量（CMV）设定边的时滞，使坡升信号在物理到达之前就被向前传播。利用一个具有已知风场的受控合成试验平台，我们证明：（i）在使用现实可行的互相关CMV估计时，显式平流图并不优于普通的静态图或学习邻接的时空GNN；（ii）完美CMV所能带来收益中，大约一半仅仅来自于将精确的运动矢量作为输入特征提供，而非来自图结构；（iii）只有当预报时域内的平流位移 v*H 落在传感器网络范围之内时，平流信息才有帮助。受（ii）的启发，

    arXiv:2609.30286v1 Announce Type: cross  Abstract: Short-term forecasting of cloud-induced power ramps across a network of distributed photovoltaic (PV) or irradiance sensors is a recognised pain point for grid operators. A natural idea is to make the graph neural network (GNN) advection-aware: connect each site to the sites upwind of it, with edge time-lags set by the cloud-motion vector (CMV), so that a ramp is propagated forward before it physically arrives. Using a controlled synthetic testbed with a known wind field, we show that (i) with a realistic cross-correlation CMV estimate, an explicit advection graph does not beat a plain static or learned-adjacency spatiotemporal GNN; (ii) roughly half of the benefit available from a perfect CMV comes simply from providing an accurate motion vector as an input feature, not from graph structure; and (iii) advection helps only when the advective displacement over the forecast horizon, v*H, fits inside the sensor network. Motivated by (ii),
    
[^191]: 用于中子监测器时间序列预测的季节性与量子启发模型

    Seasonal and Quantum-inspired Models for Neutron Monitor Time Series Forecasting

    [https://arxiv.org/abs/2609.30281](https://arxiv.org/abs/2609.30281)

    本文系统比较了季节性基线、深度序列模型与量子启发架构在中子监测器时间序列多步预测中的表现，发现量子启发变体QiKAN取得最低的总体预测误差，而简单的季节性朴素基线依然保持强竞争力。

    

    我们针对洛姆尼察峰中子监测器（LMKS）时间序列的多步预测开展了一项聚焦且可复现的研究。我们的评估套件涵盖简单的季节性基线、现代深度序列模型以及函数型和量子启发架构，包括季节性朴素预测（Seasonal Naive）、长短期记忆网络（LSTM）、时间卷积网络（TCN）、N-BEATS、Kolmogorov-Arnold网络（KAN），以及两个量子启发变体QiLSTM和QiKAN。我们描述了数据集特征、诊断分析、预处理流程和训练步骤，并使用平均绝对误差（MAE）和均方根误差（RMSE）报告了所有评估模型的总体点预测性能。我们的快速运行结果表明，量子启发的KAN变体QiKAN在所评估的配置中取得了最低的总体预测误差，而简单的季节性朴素基线仍展现出显著的竞争力。

    arXiv:2609.30281v1 Announce Type: new  Abstract: We present a focused and reproducible study of multi-horizon forecasting on the Lomnicky Stit neutron monitor (LMKS) time series. Our evaluation suite covers simple seasonal baselines, modern deep sequence models, and functional and quantum-inspired architectures, including Seasonal Naive, Long Short-Term Memory (LSTM), Temporal Convolutional Network (TCN), N-BEATS, Kolmogorov-Arnold Networks (KAN), and two quantum-inspired variants, QiLSTM and QiKAN. We describe the dataset characteristics, diagnostic analysis, preprocessing pipeline, and training procedures, and report aggregate point-forecast performance using mean absolute error (MAE) and root mean squared error (RMSE) for all evaluated models. Our quick-run results indicate that the quantum-inspired KAN variant, QiKAN, achieves the lowest aggregate forecasting error among the evaluated configurations, while the simple Seasonal Naive baseline remains remarkably competitive. These res
    
[^192]: 神经理想与神经码：一种用于神经网络分类与特征解释的代数框架

    Neural Ideals and Neural Codes: An Algebraic Framework for Neural Network Classification and Feature Interpretation

    [https://arxiv.org/abs/2609.30279](https://arxiv.org/abs/2609.30279)

    提出一个基于“神经理想”的代数框架，建立神经网络与代数对象之间的对应关系及相关计算算法，从而能够识别和解释每个隐藏层神经元所捕获的特征。

    

    理解神经网络隐藏层所捕获的特征是机器学习中的一个根本性挑战，尽管神经网络在各种分类问题上取得了广泛的成功。在本工作中，我们提出了一个代数框架，用于研究对分类问题进行建模的神经网络。我们首先建立了一些结果，例如神经网络与神经理想之间的对应关系、计算神经理想的算法，以及一个能够实现神经理想近似的稳定化定理。作为该框架的一个应用，我们提出了识别和解释每个隐藏层神经元所捕获特征的算法。在这些理论发展的基础上，我们在MNIST数字数据集上验证了实际性能，结果凸显了神经理想作为分析神经网络所捕获特征的数学与计算工具的关键作用。

    arXiv:2609.30279v1 Announce Type: new  Abstract: Understanding the features captured by the hidden layers of neural networks is a fundamental challenge in machine learning, despite their widespread success across various classification problems. In this work, we propose an algebraic framework for examining neural networks that model classification problems. Certain results, such as the correspondence between the neural network and neural ideals, algorithms for computing the neural ideals, and a stabilization theorem that enables approximation of the neural ideals, are first established. As an application to the framework, we present algorithms to identify and interpret the features captured by each hidden-layer neuron. Along with these theoretical developments, the practical performance has been demonstrated on the MNIST digit dataset, and the results highlight the pivotal role of neural ideals as a mathematical and computational tool for analyzing the features captured by neural netwo
    
[^193]: 无固定扩散的不动点：面向收敛测试时计算的隐式神经层束

    Fixed Points Without Fixed Diffusion: Implicit Neural Sheaves for Convergent Test-Time Computation

    [https://arxiv.org/abs/2609.30277](https://arxiv.org/abs/2609.30277)

    提出 SheafDEQ，一种基于自适应神经层束传播的次齐次深度平衡架构，通过可学习的矩阵值层束限制映射实现更丰富的边依赖变换，并在温和条件下保留了隐式图神经网络不动点唯一且可收敛的保证，从而兼顾表达能力和测试时计算的收敛性。

    

    隐式图神经网络（IGNN）将节点表示定义为消息传递算子的不动点，从而实现有效无限深度的传播、与迭代次数无关的参数化以及灵活的测试时计算。然而，这些优势依赖于平衡点的唯一性以及可通过不动点迭代达到该平衡点。现有构造通常对循环更新施加约束以获得这些保证，这限制了平衡点处可用的变换。这引出一个核心问题：IGNN 能否通过更丰富的、依赖边的变换来提升表达能力，同时保留其平衡点表达式的固有优势？我们提出 SheafDEQ，一种具有自适应神经层束传播的次齐次深度平衡架构。其学习到的矩阵值层束限制映射可以对齐、混合或反转相邻节点的表示。在温和的正则性条件下，我们证明 SheafDEQ 具有……（摘要原文在此处截断）

    arXiv:2609.30277v1 Announce Type: new  Abstract: Implicit Graph Neural Networks (IGNNs) define node representations as fixed points of message-passing operators, enabling effectively infinite-depth propagation, iteration-independent parameterization, and flexible test-time computation. Yet these benefits depend on the equilibrium being unique and attainable by fixed-point iteration. Existing constructions often impose constraints on recurrent updates to obtain these guarantees, limiting the transformations available at equilibrium. This raises a central question: can IGNNs gain expressiveness through richer, edge-dependent transformations while retaining the inherent strengths of their equilibrium formulation? We introduce SheafDEQ, a subhomogeneous deep-equilibrium architecture with adaptive neural-sheaf propagation. Its learned, matrix-valued sheaf restriction maps can align, mix, or reverse neighbouring representations. Under mild regularity conditions, we prove that SheafDEQ admits
    
[^194]: 为什么裁剪对AdaGrad很重要？迈向广义平滑条件下的高概率理论

    Why Clipping Matters in AdaGrad? Toward a High-Probability Theory under Generalized Smoothness

    [https://arxiv.org/abs/2609.30276](https://arxiv.org/abs/2609.30276)

    该论文揭示了在广义平滑和重尾噪声下，未裁剪的AdaGrad会因自适应分母学习到罕见噪声冲击而非目标函数局部曲率而各向异性失准，并首次为裁剪后的原始AdaGrad给出了有限时间范围的高概率收敛保证。

    

    我们分析了原始的同步逐坐标AdaGrad在广义平滑及方差有界的重尾噪声条件下的表现。在该设定下，局部曲率可能随梯度范数呈亚二次增长，且随机梯度仅被假设具有有界的条件二阶矩。我们证明了未裁剪的AdaGrad可能出现“各向异性失准”：在重尾噪声下，自适应分母学习到的是罕见噪声冲击的几何结构，而非目标函数的局部曲率，从而导致持续的方向性畸变，阻碍有限时间范围内的欧氏进展。随后我们证明裁剪机制能够修复这一失效模式。我们的主要结果是为原始的非滞后AdaGrad更新给出了有限时间范围内的高概率收敛保证，即 $\frac1T\sum_{t=0}^{T-1}\|\nabla f(x_t)\|^2=\mathcal{O}\left(\frac{d\big(\sqrt{\log T} + \log \frac{1}{\delta}\big)}{\sqrt{T}}\right)$，并由此得到……

    arXiv:2609.30276v1 Announce Type: new  Abstract: We analyze the original same-step coordinate-wise AdaGrad under generalized smoothness and heavy-tailed noise with bounded variance. In this setting, local curvature may grow sub-quadratically with the gradient norm, and stochastic gradients are assumed to have only bounded conditional second moments. We show that unclipped AdaGrad can become \emph{anisotropically miscalibrated}: under heavy-tailed noise, the adaptive denominator can learn the geometry of rare noise shocks rather than the local curvature of the objective, leading to a persistent directional distortion that blocks finite-horizon Euclidean progress. We then prove that clipping repairs this failure mode. Our main result is a finite-horizon high-probability guarantee for the original non-lagged AdaGrad update, yielding $\frac1T\sum_{t=0}^{T-1}\|\nabla f(x_t)\|^2=\mathcal{O}\left(\frac{d\big(\sqrt{\log T} + \log \frac{1}{\delta}\big)}{\sqrt{T}}\right),$ and hence $\widetilde{
    
[^195]: 余弦相似度不是证据：测量量化下可解释性迁移的噪声基底

    Cosine Similarity Is Not Evidence: Measuring the Noise Floor of Interpretability Transfer Under Quantization

    [https://arxiv.org/abs/2609.30275](https://arxiv.org/abs/2609.30275)

    该论文指出，用余弦相似度等尺度不变统计量来认证可解释性方法在量化后依然有效、却不报告其噪声基底是不构成证据的，并证明均值差方向估计器的折半噪声基底由 $\kappa=n\rho^2/d$ 与闭式解 $\mathbb{E}[\cos]\approx(1+4/\kappa)^{-1}$ 决定，通过在真实激活上实测此前从未被报告的类别分离度 $\rho$，表明仅采样噪声就能在Qwen2.5-1.5B-Instruct上产生0.978–0.994的余弦一致性，从而使已发表的0.996等认证数值失去证明力。

    

    一个在报告时未附带解释其所需数量的统计量并不构成证据。我们针对AI安全中的一种具体实践发展了这一论点。可解释性产物在全精度权重上进行校准，部署到量化权重上，并通过尺度不变的统计量（余弦相似度、相关系数、AUROC）被认证为在变化中得以保留，而这些统计量在报告时并未附有其噪声基底。对于均值差方向估计器，其折半噪声基底由一个无量纲数 $\kappa = n\rho^2/d$ 决定。闭式解 $\mathbb{E}[\cos] \approx (1+4/\kappa)^{-1}$ 是经典结论；缺失的输入是类别分离度 $\rho$，我们在真实激活上对其进行了测量；据我们所知，没有任何压缩迁移研究报告过这一数值。在Qwen2.5-1.5B-Instruct上，$\rho$ 在不同深度处为33至61，因此仅凭采样噪声，估计器两次独立运行之间的一致性就能达到0.978至0.994。一篇已发表的论文报告了全精度与量化模型之间的余弦相似度为0.996……（摘要在此处截断）

    arXiv:2609.30275v1 Announce Type: new  Abstract: A statistic reported without the quantity needed to interpret it is not evidence. We develop that thesis for a concrete practice in AI safety. Interpretability artifacts are calibrated on full-precision weights, deployed on quantized ones, and certified as surviving the change by scale-invariant statistics (cosine similarity, correlation, AUROC) that are reported without their noise floor. For the difference-in-means direction estimator, the split-half floor is governed by one dimensionless number, $\kappa = n\rho^2/d$. The closed form $\mathbb{E}[\cos] \approx (1+4/\kappa)^{-1}$ is classical; the missing input is the class separation $\rho$, which we measure on real activations; no compression-transfer study we know of reports it. On Qwen2.5-1.5B-Instruct, $\rho = 33$--$61$ across depth, so two independent runs of the estimator agree to $0.978$--$0.994$ by sampling alone. A published cosine of $0.996$ between full-precision and quantize
    
[^196]: 耗散随机动力系统在 $\mathbb{R}^d$ 上的击中时间分布及其在随机梯度下降中的应用

    Distribution of hitting times for dissipative random dynamical systems on $\mathbb{R}^d$, with application to stochastic gradient descent

    [https://arxiv.org/abs/2609.30274](https://arxiv.org/abs/2609.30274)

    该论文从遍历理论的视角提出了一种新方法，用于研究一大类随机优化方法的渐近性质，其主要贡献是给出了随机优化算法到达极小值点小邻域的击中时间分布分析，并将其应用于随机梯度下降的收敛性研究。

    

    机器学习，特别是深度学习，涉及求解大规模非凸优化问题。文献中已提出多种算法，这些算法对于困难问题实例似乎能够取得令人满意的实际效率，其中随机梯度方法是最基础的，但在许多学习任务上仍然优于更晚近的算法。关于深度学习中现有方法的一个主要未解问题是理解它们的收敛性质。沿着关于梯度类算法长时间行为的一系列先前研究工作，我们从遍历理论的角度提出了一种新方法，用于研究一大类优化方法的渐近性质。我们的主要结果包括对随机优化算法到达某个极小值点的特定小邻域所需期望时间的研究，以及……

    arXiv:2609.30274v1 Announce Type: new  Abstract: Machine Learning and more specifically Deep Learning involves solving large scale nonconvex optimization problems. Several algorithms have been proposed in the literature, that seem to achieve satisfactory practical efficiency for difficult instances, the Stochastic Gradient Method being the most rudimentary, while still outperforming more recent algorithms at a number of learning tasks.   A major open question about the current methods used in deep learning is to understand their convergence properties. Following a line of previous works about the long time behavior of gradient-type algorithms, %and in particular the recent contributions from Azizian et al., we present a new approach for studying the asymptotic properties of a wide family of methods from an ergodic theoretical viewpoint. Our main results include a study of the expected time for a stochastic optimisation algorithm to reach a certain small neighborhood of a minimizer and 
    
[^197]: 离线策略评估作为设计自适应实验的决策支持工具

    Offline Policy Evaluation as a decision support tool for designing Adaptive Experiments

    [https://arxiv.org/abs/2609.30273](https://arxiv.org/abs/2609.30273)

    该论文提出将离线策略评估（OPE）与受控热启动模拟相结合的方法，利用历史A/B测试数据对自适应与非自适应策略进行排序评估，为基于上下文老虎机的自适应实验设计提供安全的决策支持工具。

    

    我们研究了如何利用固定随机实验（A/B测试）的历史数据来指导基于上下文老虎机的自适应实验的部署。给定在静态分配下收集的数据，我们的目标是评估哪些自适应策略（如果存在的话）会优于原始设计，以及在何种条件下会如此。为此，我们将离线策略评估（OPE）与受控的热启动模拟相结合。从展现出异质性处理效应的A/B测试记录数据中，我们估计干扰成分，并使用双重稳健估计器对一个预先指定的自适应与非自适应策略组合进行排序。当真实结果可用时，我们随后在一个复用确切数据生成奖励概率的模拟器中部署相同的离线训练策略，从而提供了一个安全的、以真实结果为基准的环境，用于研究热启动下从离线到在线的过渡。使用合成随机控制……

    arXiv:2609.30273v1 Announce Type: new  Abstract: We investigate how historical data from fixed randomized experiments (A/B tests) can be used to inform the deployment of adaptive experiments based on contextual bandits. Given data collected under a static allocation, our goal is to assess which adaptive policies, if any, would have outperformed the original design and under what conditions. To this end, we combine off-policy evaluation (OPE) with a controlled warm-start simulation. From logged A/B test data exhibiting heterogeneous treatment effects, we estimate nuisance components and use doubly robust estimators to rank a portfolio of pre-specified adaptive and non-adaptive policies. When ground truth is available, we then deploy the same offline-trained policies in a simulator that reuses the exact data-generating reward probabilities, providing a safe, ground-truth-anchored environment to study the offline-to-online transition under warm starting. Using synthetic randomized control
    
[^198]: ENAS：一种面向资源受限微控制器上TinyML的高效硬件感知神经架构搜索框架

    ENAS: An Efficient Hardware-Aware Neural Architecture Search Framework for TinyML on Resource-Constrained Microcontrollers

    [https://arxiv.org/abs/2609.30272](https://arxiv.org/abs/2609.30272)

    ENAS是一个无需GPU即可高效运行的硬件感知神经架构搜索框架，通过静态可行性检查、支持多种块的单元搜索空间和三阶段混合搜索策略，在资源受限的微控制器上实现了TinyML模型的快速搜索。

    

    我们提出了ENAS，这是一个硬件感知的神经架构搜索（NAS）框架，它结合了静态可行性检查、一个支持标准块、深度可分离块和瓶颈块（带可选跳跃连接）的基于单元的搜索空间，以及一种具有跨运行持久缓存的三阶段混合搜索策略（随机搜索→top-K筛选→变异）。与许多依赖GPU加速的现有NAS框架不同，ENAS被设计为在无需GPU的情况下也能高效运行，使其适用于资源受限的开发环境。我们在两个TinyML基准数据集（Visual Wake Words和Melanoma Cancer）上对ENAS进行了评估，涵盖八款SRAM内存占用从20KB到1MB的微控制器以及九种输入图像分辨率。实验结果表明，ENAS在Visual Wake Words和Melanoma Cancer数据集上分别实现了平均2.41倍和1.70倍的搜索时间加速。

    arXiv:2609.30272v1 Announce Type: cross  Abstract: We present \textbf{ENAS}, a hardware-aware Neural Architecture Search (NAS) framework that combines a static feasibility check, a cell-based search space supporting standard, depthwise-separable, and bottleneck blocks with optional skip connections, and a three-stage hybrid search strategy (random $\rightarrow$ top-$K$ $\rightarrow$ mutation) with persistent cross-run caching. Unlike many existing NAS frameworks that rely on GPU acceleration, ENAS is designed to operate efficiently without requiring GPUs, making it suitable for resource-constrained development environments. We evaluate ENAS on two TinyML benchmarks, Visual Wake Words and Melanoma Cancer, across eight microcontrollers with memory footprints ranging from 20\,KB to 1\,MB SRAM and nine input image resolutions. Our experimental results show that ENAS achieves mean search-time speedups of $2.41{\times}$ and $1.70{\times}$ on the Visual Wake Words and Melanoma Cancer datasets
    
[^199]: 当预条件指数变为负值时：学习率耦合与跨环境泛化

    When the Preconditioning Exponent Turns Negative: Learning-Rate Coupling and Cross-Environment Generalization

    [https://arxiv.org/abs/2609.30271](https://arxiv.org/abs/2609.30271)

    本研究通过系统性的跨环境实验发现，使跨环境泛化准确率最大化的预条件指数随学习率的对数几乎线性下降（斜率约-0.27至-0.30），揭示了预条件指数与全局学习率之间存在强耦合关系。

    

    自适应优化器通常由二阶矩估计的固定幂次来参数化。现有的部分自适应方法研究了介于类动量更新与标准Adam平方根之间的指数，而该指数与全局学习率之间的相互作用却较少被理解。我们使用一个配对的四环境分类问题进行了受控的跨环境研究，该问题包含稳定稀疏特征、依赖环境的虚假稀疏特征、稠密特征和高维噪声。在涵盖21个预条件指数 $p\in[-0.5,0.5]$ 和五个学习率 $\eta\in[10^{-4},10^{-2}]$ 的 \NumRuns{} 次源域训练运行中，我们发现使跨环境准确率最大化的指数随 $\log_{10}\eta$ 几乎线性下降。拟合斜率范围从 -0.270 到 -0.300，R² 介于 0.972 和 0.996 之间。在 $\eta=10^{-2}$ 时，源域验证集的选择仍然偏好正指数……

    arXiv:2609.30271v1 Announce Type: new  Abstract: Adaptive optimizers are commonly parameterized by a fixed power of the second-moment estimate. Existing partially adaptive methods study exponents between momentum-like updates and the standard Adam square root, while the interaction between this exponent and the global learning rate is less understood. We perform a controlled cross-environment study using a paired four-environment classification problem with stable sparse features, environment-dependent spurious sparse features, dense features, and high-dimensional noise. Across \NumRuns{} source-training runs covering 21 preconditioning exponents $p\in[-0.5,0.5]$ and five learning rates $\eta\in[10^{-4},10^{-2}]$, we find that the exponent maximizing cross-environment accuracy decreases almost linearly with $\log_{10}\eta$. The fitted slopes range from $-0.270$ to $-0.300$, with $R^2$ between $0.972$ and $0.996$. At $\eta=10^{-2}$, source-validation selection still prefers positive exp
    
[^200]: HybridInfer：面向端侧、边缘与云端大语言模型推理的热感知强化学习分层路由

    HybridInfer: Thermal-Aware Reinforcement-Learning Tier Routing for On-Device, Edge, and Cloud LLM Inference

    [https://arxiv.org/abs/2609.30270](https://arxiv.org/abs/2609.30270)

    提出 HybridInfer，一种热感知的强化学习路由器，在端侧、边缘与云端三层架构间智能调度大语言模型推理，以解决端侧持续生成导致移动 GPU 推理运行时崩溃或卡死的热约束问题。

    

    arXiv:2609.30270v1 公告类型：新。摘要：使用小型语言模型进行端侧推理可以将用户数据保留在本地、支持离线运行，且不产生每查询成本，因此当端侧能力足够时，端侧层级是首选。然而，端侧推理受热约束，而且我发现这一约束的影响比单纯的减速更为严重：在一款旗舰级骁龙（Snapdragon）设备上，持续的端侧文本生成会导致 GPU 推理运行时变得不稳定，在连续几次查询后出现崩溃或无声卡死。该故障源于当前工具链（移动 GPU 上的 OpenCL 内核编译以及长提示词预填充），即使设备处于冷却状态也会复现，并且在长文本生成场景下最为严重。跨端侧、边缘和云端模型的多层级路由器可以缓解这种压力，但现有路由器均不感知设备热状态，且通常在仿真环境或非移动硬件上进行评估。我提出了 HybridInfer，这是一种面向三层架构（端侧 Llama 3.2 3B、边缘 Llama 3.1 8B 与云端模型）的热感知强化学习路由器（摘要在此处被截断）。

    arXiv:2609.30270v1 Announce Type: new  Abstract: On-device inference with small language models keeps user data local, works offline, and incurs no per-query cost, so the on-device tier is preferred when it is adequate. It is thermally constrained, however, and I find the constraint is sharper than a slowdown: on a flagship Snapdragon device, sustained on-device generation destabilizes the GPU inference runtime, which crashes or silently wedges after a few consecutive queries. The failure lies in the current toolchain (OpenCL kernel compilation and long-prompt prefill on the mobile GPU), recurs even when the device is cool, and is worst for long generations. Multi-tier routers across on-device, edge, and cloud models can relieve this pressure, but existing routers are thermal-blind and typically evaluated in simulation or on non-mobile hardware. I present HybridInfer, a thermal-aware reinforcement-learning router for a three-tier hierarchy (on-device Llama 3.2 3B, edge Llama 3.1 8B wit
    
[^201]: 短程瞄准以致远：你的冻结世界模型比你想象的更会规划

    Aim Short to Reach Far: Your Frozen World Model Can Plan Better Than You Think

    [https://arxiv.org/abs/2609.30036](https://arxiv.org/abs/2609.30036)

    提出锚定规划方法，通过瞄准从经验中检索的中间观测目标而非最终目标图像，使冻结的世界模型无需额外训练即可在长程任务规划中全面超越原有规划器。

    

    基于视觉世界模型构建的规划器通常通过预测结果与编码目标图像之间的距离来为每个预测结果评分。我们证明，即使动力学完全精确、短程搜索全局最优，这一目标也可能限制控制效果：到达目标可能需要最初远离目标的动作。使用冻结的 LeWM 模型，中间目标在 Cube、PushT、Reacher 和 TwoRoom 任务上显著改善了动作合成与已记录动作的排序。学习得到的目标和从观测经验中提取的目标都能带来这些收益。我们提出了锚定规划，该方法检索一段其起点和终点分别与当前观测和目标观测相似的已记录片段，然后瞄准该片段起点之后不久的一个观测。冻结模型从当前状态出发，向该目标对动作进行评分。在无需任何额外训练的情况下，向观测目标进行规划在我们长程评估中的每个任务上都优于已发布的 LeWM 规划器。

    arXiv:2609.30036v1 Announce Type: new  Abstract: Planners built on visual world models commonly score each predicted outcome by its distance to the encoded goal image. We show that this target can limit control even with exact dynamics and globally optimal short-horizon search: reaching a goal may require actions that initially move away from it. With frozen LeWM models, intermediate targets substantially improve action synthesis and recorded-action ranking on Cube, PushT, Reacher, and TwoRoom. Learned targets and targets drawn from observed experience both produce these gains. We introduce Anchored Planning, which retrieves a recorded segment whose start and end resemble the current and goal observations, then aims at an observation shortly after its start. The frozen model scores actions toward this target from the current state. Without additional training, planning toward observed targets outperforms the released LeWM planner on every task in our long-range evaluation. Additional f
    
[^202]: 多样几何、冻结权重：基于因果专家集成的鲁棒异质性处理效应估计

    Diverse Geometries, Frozen Weights: Robust Heterogeneous Treatment-Effect Estimation via Causal Expert Ensembles

    [https://arxiv.org/abs/2609.29974](https://arxiv.org/abs/2609.29974)

    该论文提出GeoACE五专家集成框架，通过结合锚定校正估计器与多样化的重叠感知和结果引导几何，并采用验证集学习且在测试前冻结的集成权重（其中新增的O-Phi-ACE专家用无结果的重叠感知统计投影替代锚定输入），实现了更鲁棒的异质性处理效应估计。

    

    从观测数据中估计异质性处理效应是困难的，因为最合适的归纳偏置会随着重叠程度、处理不平衡、预后结构和样本量的变化而改变。我们提出了几何多样锚定校正专家集成，这是一个五专家框架，它将一个通用的锚定校正估计器与互补的重叠感知几何和结果引导几何相结合。其任务级集成权重仅从内部验证预测中学习，在测试评估前被冻结，然后应用于在完整开发样本上重新拟合的专家。第五个专家O-Phi-ACE从协变量和处理分配中构建一个不依赖结果的重叠感知统计投影，并用这种更低维度的几何来替代锚定输入。我们在八个基准协议上将GeoACE与11个对比方法进行评估。添加O-Phi-ACE使平均sqrt(PEHE)相对于四专家版本有所降低……

    arXiv:2609.29974v1 Announce Type: new  Abstract: Estimating heterogeneous treatment effects from observational data is difficult because the most appropriate inductive bias varies with overlap, treatment imbalance, prognostic structure, and sample size. We introduce the Geometry-Diverse Anchor-Correction Expert Ensemble (GeoACE), a five-expert framework that combines a common anchor-correction estimator with complementary overlap-aware and outcome-guided geometries. Its task-level ensemble weights are learned only from internal validation predictions, frozen before test evaluation, and then applied to experts refitted on the complete development sample. The fifth expert, O-Phi-ACE, constructs an outcome-free, overlap-aware statistical projection from covariates and treatment assignment and replaces the anchor input with this lower-dimensional geometry. We evaluate GeoACE against 11 comparators on eight benchmark protocols. Adding O-Phi-ACE reduced mean sqrt(PEHE) relative to the four-e
    
[^203]: Rufus-Air：一个开放的大语言模型后训练方案

    Rufus-Air: An Open LLM Post-Training Recipe

    [https://arxiv.org/abs/2609.29421](https://arxiv.org/abs/2609.29421)

    本文提出了Rufus-Air，一个基于GLM-4.5-Air-Base的开放可复现的八阶段大模型后训练方案，并总结出高质量SFT奠定能力基础、难度过滤保持RL学习有效性、奖励可靠性指导阶段排序等关键实践经验。

    

    Rufus-Air 是一个在 GLM-4.5-Air-Base（106B-A12B）上构建的开放且可复现的后训练方案，由八个阶段的串行流水线组成：SFT（监督微调）、推理 RL、编码 RL、指令遵循 RL、通用智能体、编码智能体、搜索智能体和 RLHF。我们记录了复现该方案所需的数据、奖励设计、基础设施、阶段顺序以及各阶段的结果。各阶段从基础能力逐步推进到高级能力，奖励信号也从严格可验证的奖励过渡到较为柔和的基于评判者的信号。训练基于开源组件和公开数据，其中大部分数据按原样使用，无需新的人工标注或内部蒸馏教师模型。我们的主要发现是：(i) 多样化、高质量的 SFT 奠定了坚实的能力基础；(ii) 难度过滤可将 RL 提示保持在有效的学习区间内；(iii) 奖励可靠性为阶段排序提供了实用原则；(iv) 基础设施与工程选择是……

    arXiv:2609.29421v1 Announce Type: cross  Abstract: Rufus-Air is an open and reproducible post-training recipe on GLM-4.5-Air-Base (106B-A12B), organized as a serial pipeline of eight stages: SFT, Reasoning RL, Coding RL, Instruction-Following RL, General Agent, Coding Agent, Search Agent, and RLHF. We document the data, reward design, infrastructure, stage order, and stagewise results needed to reproduce the recipe. Stages progress from basic to advanced capabilities and from hard, verifiable rewards to softer judge-based signals. Training builds on open-source components and public data, much of it used as released, without new human annotation or an in-house distillation teacher. Our main findings are that (i) diverse, high-quality SFT establishes a strong capability floor; (ii) difficulty filtering keeps RL prompts within a productive learning range; (iii) reward reliability provides a practical principle for ordering stages; and (iv) infrastructure and engineering choices are part 
    
[^204]: M-plicits：基于嵌套多尺度残差的神经隐式表面

    M-plicits: Neural Implicit Surfaces via Nested Multiscale Residuals

    [https://arxiv.org/abs/2609.28684](https://arxiv.org/abs/2609.28684)

    提出M-plicits多尺度框架，将表面建模为通过嵌套邻域训练的MLP残差和，并将监督严格限制在先前零水平集周围的窄带内，从而在训练效率、渲染速度和噪声鲁棒性之间实现更好的平衡。

    

    将输入坐标通过正弦函数编码并输入多层感知机（MLP），已被证明对于定义为零水平集的表面的隐式神经表示（INR）是有效的。然而，现有方法往往难以在训练效率、渲染速度和噪声鲁棒性之间取得平衡：单MLP方法在推理时代价高昂；基于网格的表示虽然速度快，但可能限制表面平滑度并过拟合输入噪声；而以往的多尺度方法由于硬频谱截断经常捕捉到噪声并产生伪影。为了解决这些局限性，我们提出了M-plicits，这是一个多尺度框架，将表面建模为通过一系列嵌套邻域训练的MLP残差和。与依赖标准全域采样且需要昂贵的网格提取才能进行可视化的现有残差方法不同，我们的方法严格将监督限定在先前零水平集周围的窄带区域内。

    arXiv:2609.28684v1 Announce Type: cross  Abstract: Encoding input coordinates with sinusoidal functions into multi-layer perceptrons (MLPs) has proven effective for implicit neural representations (INRs) of surfaces defined as zero-level sets. However, existing methods often struggle to balance training efficiency, rendering speed, and noise robustness: single-MLP approaches are expensive at inference, grid-based representations are fast but can limit surface smoothness and overfit input noise, and previous multiscale approaches frequently capture noise and produce artifacts due to hard spectral truncation. To address these limitations, we propose M-plicits, a multiscale framework that models surfaces as a residual sum of MLPs trained via a sequence of nested neighborhoods. Unlike existing residual approaches that rely on standard domain-wide sampling and require costly mesh extraction for visualization, our method strictly localizes supervision to narrow bands around the previous zero
    
[^205]: 直推学习更锐利的界及其应用

    Even Sharper Bounds for Transductive Learning and Its Applications

    [https://arxiv.org/abs/2609.28459](https://arxiv.org/abs/2609.28459)

    本文提出直推学习的新型局部化复杂度方法STLC，去除了以往直推学习中多余的对数置信度因子，并在可实现设定下达到了与标准归纳学习相同的 $\cO\{\dVC\log(me/\dVC)/m\}$ 速率。

    

    我们提出了更锐利的直推局部复杂度，这是一种针对无放回均匀采样下直推学习的局部化复杂度方法。该构造始于一个关于测试-训练经验过程上确界的Bernstein型集中不等式，其证明利用了交换随机游走的修正对数Sobolev不等式和双参数熵闭包。随后，通过替代局部化泛函的剥离论证，我们得到了与经典归纳局部Rademacher复杂度界具有相同不动点和置信度项的过剩风险界，且去除了早期直推结果中额外的对数置信度因子。对于VC维为 $\dVC$ 的二值函数类上的可实现学习，当训练集大小为 $m$、测试集大小为 $u$ 且 $u\ge m\ge\dVC$ 时，STLC 给出了 $\cO\{\dVC\log(me/\dVC)/m\}$ 的界。这一结果匹配了标准的归纳学习速率，并且当 $m\ge9$ 时，与直推…（摘要原文在此处截断）相比仅相差一个对数因子。

    arXiv:2609.28459v1 Announce Type: new  Abstract: We introduce Sharper Transductive Local Complexity (STLC), a localized complexity method for transductive learning under uniform sampling without replacement. The construction starts from a Bernstein-type concentration inequality for the supremum of the test--train empirical process. Its proof uses the modified log-Sobolev inequality for the swap walk and a two-parameter entropy closure. A peeling argument with a surrogate localization functional then gives excess-risk bounds with the same fixed-point and confidence terms as the classical inductive local Rademacher-complexity bounds, without the additional logarithmic confidence factor in earlier transductive results. For realizable learning over a binary class of VC dimension $\dVC$, with training size $m$, test size $u$, and $u\ge m\ge\dVC$, STLC yields $\cO\{\dVC\log(me/\dVC)/m\}$. This matches the standard inductive rate and, when $m\ge9$, is within a logarithmic factor of the transd
    
[^206]: 支持度编译特征折叠：在表格基础模型中以更低内存利用更多证据

    Support-Compiled Feature Folding: More Evidence at Lower Memory Across Tabular Foundation Models

    [https://arxiv.org/abs/2609.28208](https://arxiv.org/abs/2609.28208)

    提出免训练推理框架“支持度编译特征折叠”（SCFF），将按支持度排序的有界特征子集路由进冻结的表格基础模型，把二次方特征交互开销降为线性，在不集成预测、不训练新参数的情况下以更低内存利用更多证据，并在18个宽表数据集上提升了全部六个骨干网络的准确率与NLL。

    

    表格基础模型面临特征侧的扩展困境：全宽度的成对特征混合随列数呈二次方增长，而特征选择虽然节省内存，却以丢弃证据为代价。我们提出了支持度编译特征折叠（SCFF），这是一种无需训练的推理框架，无需修改冻结的骨干网络即可解决这一困境。SCFF将按支持度排序的特征路由至原生特征编码器的有界叶子节点，对残差证据进行支持度检查，并在单次上下文预测之前合并编码后的信息。由此，它将二次方的特征交互计算转化为随宽度线性增长的计算，并保持有界的局部工作集，且无需集成预测或训练新参数。在固定的AMLB-29、TabZilla和TabArena快照的完整18个宽表数据集切片上，SCFF在所有六个被评估的骨干网络上均提升了数据集聚合准确率和NLL。所有四组同等宽度对比均保持优势。

    arXiv:2609.28208v1 Announce Type: new  Abstract: Tabular foundation models face a feature-side scaling dilemma: full-width pairwise mixing grows quadratically with the number of columns, whereas feature selection saves memory by discarding evidence. We introduce Support-Compiled Feature Folding (SCFF), a training-free inference framework that resolves this dilemma without changing the frozen backbone. SCFF routes support-ranked features through bounded leaves of the native feature encoder, support-checks the residual evidence, and merges the encoded messages before a single contextual prediction. It thereby converts quadratic feature-interaction work into linear-in-width work with a bounded local working set, without ensembling predictions or training new parameters. On the exhaustive 18-dataset wide-table slice of fixed AMLB-29, TabZilla, and TabArena snapshots, SCFF improves dataset-macro accuracy and NLL on all six evaluated backbones. All four matched-width comparisons retain favor
    
[^207]: 纵向血糖表征的可迁移证据重构

    Transferable Evidence Reconstruction for Longitudinal Glucose Representations

    [https://arxiv.org/abs/2609.28199](https://arxiv.org/abs/2609.28199)

    该论文提出可迁移证据重构（TER）自监督方法，通过在一个记录组上拟合低容量读取器并要求其在另一组记录中恢复相同证据的跨组测试，学习具有可迁移证据解码规则的血糖表征，并利用感知观测的每日编码器和感知时钟的多日记忆模块对持续血糖监测数据进行建模。

    

    长时间的生理记录中包含大量常规测量，而具有预测价值的信息往往集中在罕见事件、持续负担和重复出现的时间模式中。掩码自编码通过恢复测量值进行学习；对比学习通过 对齐不同视图进行学习。我们研究一种显式优先考虑结构化信号证据的自监督方法。我们提出了可迁移证据重构（Transferable Evidence Reconstruction, TER），该方法从无标签记录中构建证据，在一个记录组上拟合一个全新的低容量读取器，并要求该读取器在不重新拟合的情况下从另一组记录中恢复相同的证据。通过对这一跨组测试进行微分，可以学习到带有可迁移证据解码规则的表征；这些证据仅用于引导自监督学习，而不作为下游特征使用。针对持续血糖监测（CGM），我们设计了感知观测的每日编码器和感知时钟的多日记忆模块，将血糖水平及其变化与记录时间相绑定，同时（摘要在此处截断）

    arXiv:2609.28199v1 Announce Type: new  Abstract: Long physiological recordings contain many routine measurements, while predictive information is often concentrated in rare events, sustained burden, and recurring temporal patterns. Masked autoencoding recovers measurements; contrastive learning aligns views. We study self-supervision that explicitly prioritizes structured signal evidence. We introduce transferable evidence reconstruction (TER), which constructs evidence from unlabeled recordings, fits a fresh low-capacity reader on one recording group, and requires that reader to recover the same evidence in another group without refitting. Differentiating through this cross-group test learns representations with transferable evidence-decoding rules; the evidence guides self-supervision but is not used as a downstream feature. For continuous glucose monitoring (CGM), an observation-aware daily encoder and clock-aware multi-day memory bind glucose level and change to recorded time while
    
[^208]: NS-Attention：视觉Transformer中注意力输出的Newton-Schulz变换

    NS-ATTENTION: Newton-Schulz Transformations of Attention Outputs in Vision Transformers

    [https://arxiv.org/abs/2609.27735](https://arxiv.org/abs/2609.27735)

    提出无参数的Newton-Schulz注意力变换（NS-Attn.），对每个注意力头输出进行谱处理以降低谱集中度并提高有效秩，在ViT和Swin于CIFAR-10/100的全部12组对比实验中均带来平均0.25–0.83个百分点的准确率提升。

    

    Newton-Schulz（NS）迭代最近被用于Muon优化器中，在大语言模型训练过程中对更新矩阵进行变换。受其谱效应的启发，我们研究将NS直接应用于Transformer的注意力表示。我们提出Newton-Schulz注意力（NS-Attn.），这是一种应用于每个注意力头输出的无参数变换。每个注意力头的输出被排列为特征×令牌矩阵，并通过其Frobenius范数进行归一化，随后应用有限步的NS多项式迭代，再恢复原始范数。其目标是在标准的头合并与输出投影之前，降低谱集中度并提高有效秩。在CIFAR-10和CIFAR-100数据集上对ViT和Swin的实验中，NS-Attn.在所有12组同种子对比中均提升了最终轮次的准确率，平均增益为0.25至0.83个百分点。ViT消融实验表明，一次迭代的平均准确率高于两次迭代。谱分析……（原文截断）

    arXiv:2609.27735v1 Announce Type: new  Abstract: Newton-Schulz (NS) iteration has recently been used in the Muon optimizer to transform update matrices during the training of large language models. Motivated by its spectral effect, we investigate applying NS directly to Transformer attention representations. We introduce Newton-Schulz Attention (NS-Attn.), a parameter-free transformation applied to the output of each attention head. Each head output is arranged as a feature-by-token matrix and normalized by its Frobenius norm. We then apply a finite NS polynomial step and restore the original norm. The objective is to reduce spectral concentration and increase effective rank before standard head merging and output projection. Across ViT and Swin on CIFAR-10 and CIFAR-100, NS-Attn. improves final-epoch accuracy in all 12 matched-seed comparisons, with mean gains of 0.25--0.83 percentage points. ViT ablations show higher mean accuracy with one iteration than with two. Spectral analysis f
    
[^209]: 表格基础模型在上下文中计算了什么？通过注意力门控更新实现的原位表示精炼

    What Do Tabular Foundation Models Compute In Context? In-Situ Representation Refinement through Attention-Gated Updates

    [https://arxiv.org/abs/2609.27679](https://arxiv.org/abs/2609.27679)

    提出“原位表示精炼”机制并构建RefineICL——一种注意力门控、无FFN的上下文学习堆栈，使表格基础模型在不改变参数的情况下利用支持集标签精炼回合表示并迁移至查询，性能超越TabPFN-3等现有模型。

    

    当每张表格都定义一个新的监督任务时，表格基础模型应当学习何种可复用的计算？我们提出了“原位表示精炼”：支持集标签引导对该回合表示的更新，并且这些更新在不改变模型参数的情况下迁移到未标注的查询上。通过正则化的留一法目标，我们得到了支持集校正及其查询扩展。其中的主导项将基于注意力的读取与依赖状态的缩放分离开来，由此启发了RefineICL：一种注意力门控、无FFN的上下文堆栈，包含精选的低秩特征交互和类型化记忆。RefineICL-L24在AMLB29上达到了0.93836的OVR-AUC和0.87173的准确率。经过基准信息引导的继续训练，它在38个数据集的TabArena快照上达到1644.8 Elo，在相同评估条件下比TabPFN-3高出31.4 Elo。在TabZilla的两个视图上，它在所有四个报告指标上也均优于TabPFN-v3。在匹配的10万次更新深度网格中，扩展的FF…（摘要在此处截断）

    arXiv:2609.27679v1 Announce Type: new  Abstract: What reusable computation should a tabular foundation model learn when every table defines a new supervised task? We develop in-situ representation refinement: support labels guide updates to the episode's representations, and these updates transfer to unlabeled queries without changing model parameters. A regularized leave-one-out objective yields a support correction and its query extension. The leading term separates attention-based reading from state-dependent scaling, motivating RefineICL: an attention-gated, FFN-free contextual stack with selected low-rank feature interaction and typed memory. RefineICL-L24 reaches 0.93836 OVR-AUC and 0.87173 accuracy on AMLB29. A benchmark-informed continuation reaches 1644.8 Elo on the 38-dataset TabArena snapshot, 31.4 Elo above TabPFN-3 under the same evaluation. It also improves all four reported metrics over TabPFN-v3 on both TabZilla views. In a matched 100K-update depth grid, an expanded FF
    
[^210]: 基于分解子任务的强化学习

    Reinforcement Learning with Decomposed Subtasks

    [https://arxiv.org/abs/2609.27035](https://arxiv.org/abs/2609.27035)

    该论文提出RLDS方法，其核心是子任务分解优势估计（SDAE），通过在固定分类体系上将轨迹奖励按子任务分解并计算各子任务的组相对优势，解决了GRPO等方法将多轮rollout压缩为单一标量奖励所导致的信息损失问题。

    

    组相对策略优化（GRPO）及用于训练语言模型智能体的相关策略梯度方法，在进入策略更新之前，会将整个多轮rollout压缩为单一标量轨迹奖励。当任务由不同技能组合而成时，尤其是在稀疏且延迟的环境反馈下，这种压缩是有损的：优化器必须隐式地推断是哪种能力导致了最终结果，以及这应当如何改变行为。我们认为正确的基元并非更好的标量，而是分解：轨迹奖励应当在进入策略更新之前沿着子任务进行拆分。我们提出了基于分解子任务的强化学习（RLDS），其核心是子任务分解优势估计（SDAE）：一种替代标量GRPO优势的方法，它在固定的分类体系上将轨迹奖励拆分为每个子任务的份额，为每个子任务计算组相对优势，并将每个token的信用分配……

    arXiv:2609.27035v1 Announce Type: new  Abstract: Group Relative Policy Optimization (GRPO) and related policy-gradient methods for training language model agents collapse an entire multi-turn rollout into a single scalar trajectory reward before it enters the policy update. When the task composes distinct skills, especially under sparse and delayed environmental feedback, this collapsing is lossy: the optimizer must implicitly infer which competency drove the outcome and how that should change behavior. We argue the right primitive is not a better scalar but a decomposition: trajectory reward should be split along subtasks before it enters the policy update. We introduce Reinforcement Learning with Decomposed Subtasks (RLDS), whose core is Subtask-Decomposed Advantage Estimation (SDAE): a replacement for the scalar GRPO advantage that splits trajectory reward into per-subtask shares on a fixed taxonomy, computes a group-relative advantage per subtask, and distributes per-token credit b
    
[^211]: 面向多电网潮流的分层图神经网络：跨运行场景的泛化能力

    Towards Hierarchical GNNs for multi-grid power flow: generalization across operating scenarios

    [https://arxiv.org/abs/2609.26603](https://arxiv.org/abs/2609.26603)

    该论文提出在GENCO校正网络中引入分层潜在通信模块，通过Kron简化和Quotient构建两种简化图在不同电网间交换信息，显著提升了多电网潮流GNN模型对未见运行场景的泛化能力，其中Kron方法将电压误差降低了85.0%。

    

    分层潜在通信提升了多电网潮流模型对新运行场景的泛化能力。该模块在基于GENCO的校正网络中通过两个简化图交换信息。我们在三种电网拓扑上进行了200个epoch的初步训练，比较了基于Kron导出的传输方法、同锚点Quotient构建方法以及平坦骨干网络，每个模型使用三个初始化随机种子。评估采用每个电网200个新生成的预选场景。在训练拓扑上，Kron方法将宏观族群平衡电压误差从5.660 ± 0.899降至0.851 ± 0.110：相比Flat GENCO降低了85.0%，相比达到1.235 ± 0.225的Quotient降低了31.0%。在所有三个随机种子中，两种分层模型在每个训练拓扑上都优于基于训练解拟合的逐母线均值方法。这些结果表明在所研究的拓扑内实现了跨运行场景的泛化。

    arXiv:2609.26603v1 Announce Type: new  Abstract: Hierarchical latent communication improves the generalization of a multi-grid power-flow model to new operating scenarios. The module exchanges information through two reduced graphs within a GENCO-based corrective network. We compare Kron-derived transports, a same-anchor Quotient construction and a flat backbone in preliminary trainings of 200 epochs on three grid topologies, with three initialization seeds per model. Evaluation uses 200 newly generated, preselected scenarios per grid. On the training topologies, Kron reduces the macro family-balanced voltage error from 5.660 +- 0.899 to 0.851 +- 0.110: an 85.0% reduction relative to Flat GENCO and 31.0% relative to Quotient, which reaches 1.235 +- 0.225. Both hierarchical models outperform a per-bus mean fitted on training solutions on every training topology in all three seeds. These results demonstrate generalization across operating scenarios within the studied topologies, with one
    
[^212]: 几何感知的双曲残差量化

    Geometry-Aware Hyperbolic Residual Quantization

    [https://arxiv.org/abs/2609.26342](https://arxiv.org/abs/2609.26342)

    提出一种几何感知的双曲残差量化方法，通过双曲残差聚合恢复前向传播中庞加莱圆盘上的伸缩求和特性，并利用带折扣的双曲直通估计器在反向传播中保留几何信息，从而解决双曲空间残差量化的几何不一致问题。

    

    残差向量量化将连续表示转化为离散的多层级token序列。然而，尽管所产生的编码具有由粗到细的结构，且许多数据域中存在潜在的层次结构，大多数方法仍在欧几里得空间中运行。双曲几何为层次化表示提供了一种自然的替代方案，但朴素的双曲扩展会引入几何上的不一致性：非结合的双曲加法阻碍了一致的残差聚合，而标准的直通梯度估计则忽略了潜在空间的几何特性。我们提出了一种几何感知的双曲残差量化方法，在前向和反向传播两个过程中都解决了这些问题。在前向传播中，双曲残差聚合恢复了庞加莱圆盘上残差量化的伸缩求和行为。在反向传播中，带折扣的双曲直通估计器将重构（梯度）……

    arXiv:2609.26342v1 Announce Type: new  Abstract: Residual Vector Quantization turns continuous representations into discrete, multi-level token sequences. Yet most methods operate in Euclidean space, despite the coarse-to-fine structure of the resulting codes and the latent hierarchies present in many data domains. Hyperbolic geometry offers a natural alternative for hierarchical representations, but naive hyperbolic extensions introduce geometric inconsistencies: non-associative hyperbolic addition prevents consistent residual aggregation, while standard straight-through gradient estimation ignores the geometry of the latent space. We propose a geometry-aware hyperbolic residual quantization that addresses these issues in both the forward and backward passes. In the forward pass, Hyperbolic Residual Aggregation restores the telescoping behavior of residual quantization on the Poincare ball. In the backward pass, a discounted Hyperbolic Straight-Through Estimator routes the reconstruct
    
[^213]: 可扩展的最小体积单纯形估计与非渐近分析

    Scalable Minimum-Volume Simplex Estimation with Non-asymptotic Analysis

    [https://arxiv.org/abs/2609.25576](https://arxiv.org/abs/2609.25576)

    提出 DeepMVSA 方法，通过神经隐式形式（轻量坐标网络加 LU 三角参数化）将最小体积单纯形估计的内存降至与样本量无关的 O(K^2)、单次遍历成本降至 O(NK^2)，并给出非渐近样本复杂度界与神谕不等式等理论保证。

    

    我们研究从 N 个独立同分布、均匀采样自其内部的点中估计一个 K 维单纯形的问题；观测数据是 K+1 个未知原型的凸组合。现有的多项式时间估计器需要每样本立方级的计算量或 O(NK) 的存储空间，在 N 约为 10^6 至 10^8 的规模下不可行。我们提出 DeepMVSA，以神经隐式形式重新表述最小体积原理：一个轻量级坐标网络生成混合权重，一个三角 LU 型参数化表示对偶单纯形矩阵，从而将可训练状态的内存降至与 N 无关的 O(K^2)，并将每次数据遍历的成本降至 O(NK^2)。我们为局部化代理估计器证明了达到多项式时间基准阶数的非渐近样本复杂度界；为神经目标的每个全局最小值点证明了神谕不等式，包含体积膨胀控制和显式收缩偏差；以及一个条件性的端到端误差预算分离……

    arXiv:2609.25576v1 Announce Type: cross  Abstract: We study the estimation of a $K$-dimensional simplex from $N$ i.i.d.\ points sampled uniformly from its interior; the observations are convex combinations of $K+1$ unknown prototypes. Existing polynomial-time estimators need cubic per-sample work or $O(NK)$ storage and are impractical at $N\sim 10^6$--$10^8$. We propose DeepMVSA, which re-expresses the minimum-volume principle in neural implicit form: a lightweight coordinate network generates the mixing weights and a triangular LU-type parameterization the dual simplex matrix, reducing the trainable-state memory to $O(K^2)$, independent of $N$, and the cost per data pass to $O(NK^2)$. We prove a non-asymptotic sample-complexity bound of the polynomial-time benchmark order for a localized surrogate estimator; an oracle inequality for every global minimizer of the neural objective, with volume-inflation control and an explicit shrinkage bias; a conditional end-to-end error budget separa
    
[^214]: 用于测试时扩展的山峰采样：重复采样、进化与训练的简单且更优的替代方案

    Hill Sampling for Test-Time Scaling: A Simple and Better Alternative to Repeated Sampling, Evolution, and Training

    [https://arxiv.org/abs/2609.25510](https://arxiv.org/abs/2609.25510)

    该论文提出了一种简单的“山峰采样”方法——从冻结的大语言模型中反复采样候选程序编辑并保留当前最优程序作为后续采样的条件，无需复杂的进化搜索或测试时训练，就在圆填充问题上创下新的最先进水平，并在Erdős最小重叠问题上超越了AlphaEvolve。

    

    大型语言模型（LLM）可以通过在测试时投入额外的计算来改进可验证的科学和算法问题的解决方案。最近的系统借助日益复杂的进化搜索框架，或在测试时训练过程中更新模型参数，取得了强劲的成果。我们想探究这些复杂机制中究竟有多少是必要的。我们提出了山峰采样，这是一种简单的程序：从冻结的LLM中反复采样候选程序编辑，保留迄今为止找到的最佳程序，并让所有后续采样都以该程序为条件。我们在圆填充、集合的和/差以及Erdős最小重叠问题上，使用三个开源权重模型对该方法进行了评估。山峰采样在已发表方法中为圆填充问题设立了新的最先进水平，在Erdős最小重叠问题上超越了AlphaEvolve参考结果，并在有限集合的和与差问题上取得了强劲成果。

    arXiv:2609.25510v1 Announce Type: new  Abstract: Large language models (LLMs) can improve solutions to verifiable scientific and algorithmic problems by spending additional computation at test time. Recent systems achieve strong results with increasingly elaborate evolutionary search harnesses or by updating model parameters during test-time training. We ask how much of this machinery is necessary. We introduce Hill Sampling, a simple procedure that repeatedly samples candidate program edits from a frozen LLM, retains the best program found so far, and conditions all subsequent samples on that program. We evaluate the method on circle packing, sums/differences of sets, and Erdos' minimum-overlap problem using three open-weight models. Hill Sampling sets a new state of the art on circle packing among published methods, improves over the AlphaEvolve reference on Erdos' minimum-overlap problem, and achieves strong results on sums and differences of finite sets. The circle-packing and Erdo
    
[^215]: MIND 弥合鸿沟：一种具有可调空间尺度的地理隐式神经表示

    MIND the Gap: A Geographic Implicit Neural Representation with Adjustable Spatial Scale

    [https://arxiv.org/abs/2609.25454](https://arxiv.org/abs/2609.25454)

    提出 MIND 方法，通过嵌套监督将专用预训练地理空间模型的嵌入蒸馏为具有可调空间粒度的单一通用坐标嵌入，从而在稀疏标注条件下实现对遥远区域地理量的泛化预测。

    

    地理测量数据通常是稀疏的，导致大片区域缺乏我们想要制图的目标量的标签。地理隐式神经表示（INRs）通过学习可在任意坐标查询的平滑、通用的嵌入来解决这一问题。下游模型将这些嵌入与稀疏标签相结合，即可在推理时无需卫星影像的情况下预测未采样位置的目标值。然而，尽管这对遥感应用十分重要，模型向遥远区域的泛化能力在很大程度上仍未被探索。我们提出了 Matryoshka 隐式神经蒸馏（MIND），它将来自专用预训练地理空间模型的嵌入蒸馏为一个具有可调空间粒度的单一通用坐标嵌入。MIND 在多个嵌入维度上使用嵌套监督，这些维度定义了一系列连续的块。在我们的实验中，较早的块捕获较粗粒度的地理变化，而较后的块……（摘要截断）

    arXiv:2609.25454v1 Announce Type: cross  Abstract: Geographic measurements are often sparse, leaving large areas without labels for the quantities we want to map. Geographic implicit neural representations (INRs) address this by learning smooth, general-purpose embeddings that can be queried at any coordinate. Downstream models combine these embeddings with sparse labels to predict target values at unsampled locations without satellite imagery at inference. However, generalization to distant regions remains largely unexplored, despite its importance for remote sensing applications. We introduce Matryoshka Implicit Neural Distillation (MIND), which distills embeddings from specialist pretrained geospatial models into a single generalist coordinate embedding with adjustable spatial granularity. MIND uses nested supervision at several embedding dimensions, which define a series of contiguous chunks. In our experiments, early chunks capture coarser geographic variation, while later chunks 
    
[^216]: 通过多模态可穿戴传感在自闭症青少年行为升级前检测激越状态

    Detecting Agitation Before Behavioral Escalation in Autistic Youth Through Multimodal Wearable Sensing

    [https://arxiv.org/abs/2609.24791](https://arxiv.org/abs/2609.24791)

    该研究融合惯性测量单元运动、腕戴生理信号和语音三种模态的预训练基础模型，实现了在自闭症青少年挑战性行为升级之前对激越状态的可穿戴多模态检测（AUC达0.724）。

    

    在68%的自闭症青少年中可观察到攻击、自伤和毁物等挑战性行为，这些行为对青少年及其照护者构成风险。此类行为发作之前会出现激越，即一种通过动作、发声和自主神经唤醒表现出来的不断加剧的痛苦状态。其征兆细微且因人而异，其自主神经成分在无仪器辅助的情况下无法被察觉。我们在15名自闭症青少年参与的30次由临床医生主导的会话中，通过惯性测量单元采集上半身运动数据，通过腕戴设备采集生理数据，通过领夹式麦克风采集语音数据，并配以专家行为标注。我们适配了四个预训练基础模型（每个模态各一个），将每个模型投影到共享的128维空间中，并将其融合为一个单一的群体模型。该模型在临床医生标注的激越起始时刻（被试内置换检验 p=0.0005）以0.724的ROC曲线下面积（AUC）检测到激越……

    arXiv:2609.24791v1 Announce Type: cross  Abstract: Challenging behaviors including aggression, self-injury, and property destruction are observed in 68% of autistic youth and pose risks to youth and caregivers. These episodes are preceded by agitation, a rising state of distress expressed through movement, vocalization, and autonomic arousal. Its signs are subtle and individualized, and its autonomic components are invisible without instrumentation. We collected upper-body movement from inertial measurement units, physiology from a wrist-worn device, and vocalizations from lapel microphones across 30 clinician-led sessions with 15 autistic youth, paired with expert behavioral annotations. We adapt four pretrained foundation models, one per modality, project each to a shared 128-dimensional space, and fuse them into a single group model. The model detected agitation with an area under the ROC curve of 0.724 at the clinician-annotated onset (within-participant permutation p=0.0005), decl
    
[^217]: 面向离线强化学习的提升贝尔曼线性规划

    Lifted Bellman Linear Programming for Offline Reinforcement Learning

    [https://arxiv.org/abs/2609.24489](https://arxiv.org/abs/2609.24489)

    提出提升贝尔曼线性规划（LBLP），通过将贝尔曼最优性的线性规划刻画提升到联合(Q,V)空间，用仅涉及数据集中状态-动作对的不等式约束实现样本内贝尔曼最优性，其唯一最优解在确定性动力学下介于数据集最佳回报与最优价值之间。

    

    离线强化学习（RL）通常通过针对自举价值目标最小化回归损失来训练评论家（critic），并利用采用指数移动平均（EMA）更新的目标网络来稳定训练。多步目标包含行为策略的动作，因此需要离策略修正。我们转而通过不等式约束对评论家施加样本内贝尔曼最优性。我们提出了提升贝尔曼线性规划（Lifted Bellman Linear Program, LBLP），它将贝尔曼最优性的线性规划刻画提升到联合 $(Q,V)$ 空间，使得每个约束仅涉及数据集中的状态-动作对。该规划的唯一最小化子即为样本内最优对，且沿数据集轨迹 $K$ 步片段施加的约束对于任何 rollout 策略和任意视野都保持该最小化子不变。在确定性动力学下，该最小化子介于数据集最佳回报与最优价值之间。将约束松弛为合页（损失）……

    arXiv:2609.24489v1 Announce Type: new  Abstract: Offline reinforcement learning (RL) typically trains a critic by minimizing a regression loss against bootstrapped value targets stabilized by target networks with exponential moving average (EMA) updates. Multi-step targets incorporate behavior-policy actions and therefore require off-policy correction. We instead impose in-sample Bellman optimality on the critic through inequality constraints. We formulate the Lifted Bellman Linear Program (LBLP), which lifts the linear programming characterization of Bellman optimality to the joint $(Q,V)$ space so that every constraint involves only state-action pairs in the dataset. Its unique minimizer is the in-sample optimal pair, and constraints along $K$-step segments of dataset trajectories leave this minimizer unchanged for any rollout policy and horizon. Under deterministic dynamics, this minimizer lies between the best dataset return and the optimal value. Relaxing the constraints into hing
    
[^218]: 基于模糊逻辑的海军推进系统可解释预测性状态维护

    Explainable Predictive Condition-based Maintenance of Naval-Propulsion Systems using Fuzzy Logic

    [https://arxiv.org/abs/2609.24250](https://arxiv.org/abs/2609.24250)

    本文提出了一种结合模糊决策树和深度残差神经网络的新型框架，用于实现海军舰艇推进系统的可解释预测性维护，使用户能够理解预测结果背后的故障原因。

    

    航运业对全球经济有着重大影响，这凸显了通过有效的维护技术来保障运营可用性和安全性的必要性。在过去几十年中，预测性维护（PdM）相比现有的传统维护系统已成为一种有前景的解决方案。这是因为它提供了多项优势功能，例如对船舶部件的损坏预测、减少停机时间、改善和延长机械寿命，以及提高航行过程中的安全性。然而，现有的预测性维护方法无法向用户解释其结果，使用户无法理解可能发生的故障。为了解决这一局限性，本文提出了一种基于模糊决策树和深度残差神经网络的新型框架，旨在对海军舰艇执行可解释的预测性维护。该框架能够生成模糊的局部解释……

    arXiv:2609.24250v1 Announce Type: new  Abstract: The shipping industry has a significant impact on the global economy, emphasizing the need for operational availability and safety through the use of effective maintenance techniques. During the last decades, predictive maintenance (PdM) has emerged as a promising solution compared to the existing conventional maintenance systems. This is because it offers several advantageous functions, such as damage predictions for vessel components, reduced downtime, improved and extended life of machinery, as well as higher safety during voyages. However, existing methodologies developed for performing PdM do not provide explanations of their results to users, so that they can understand the failures that may occur. To address this limitation, this paper proposes a novel framework based on a fuzzy decision tree and a deep residual neural network, aiming to perform explainable PdM on naval vessels. The proposed framework is able to generate fuzzy loc
    
[^219]: 矩阵 AdaGrad：按行与按列的自适应次梯度方法

    Matrix AdaGrad: Row-wise and Column-wise Adaptive Subgradient Methods

    [https://arxiv.org/abs/2609.21815](https://arxiv.org/abs/2609.21815)

    本文提出了一个针对矩阵值参数的通用在线镜像下降框架，通过引入按行和按列的自适应近端函数，推导出 Row-AdaGrad 和 Column-AdaGrad 两种优化器，将 AdaGrad 式的自适应次梯度方法推广到了具有矩阵结构的参数优化中。

    

    arXiv:2609.21815v1 公告类型：交叉 摘要：AdaGrad 和 Adam 等自适应优化方法在现代神经网络训练中被广泛使用，但它们的自适应缩放主要是为向量值参数设计的，并未显式地利用矩阵结构。近期的矩阵感知优化器展示了结构化优化的优势，然而，目前仍缺乏一个可用于推导与 AdaGrad 相当的矩阵感知自适应性的通用理论框架。在本工作中，我们为矩阵值参数开发了一个带有自适应近端函数的通用在线镜像下降框架，提供了一种通过在线遗憾最小化来推导矩阵感知自适应优化的原则性方法。通过引入按行和按列的矩阵近端函数，并分析由此产生的遗憾权衡，我们推导出了按行矩阵 AdaGrad和按列矩阵 AdaGrad，其自适应缩放由累积的（按行/按列）梯度信息决定。

    arXiv:2609.21815v1 Announce Type: cross  Abstract: Adaptive optimization methods such as AdaGrad and Adam are widely used in modern neural-network training, but their adaptive scaling is primarily designed for vector-valued parameters and does not explicitly exploit matrix structure. Recent matrix-aware optimizers demonstrate the benefits of structured optimization, yet a general theoretical framework for deriving matrix-aware adaptivity comparable to that of AdaGrad remains lacking. In this work, we develop a general Online Mirror Descent framework with adaptive proximal functions for matrix-valued parameters, providing a principled approach to deriving matrix-aware adaptive optimization through online regret minimization. By introducing row-wise and column-wise matrix proximal functions and analyzing the resulting regret trade-off, we derive Row-wise Matrix AdaGrad (Row-AdaGrad) and Column-wise Matrix AdaGrad (Column-AdaGrad), with adaptive scaling determined by the accumulated row-w
    
[^220]: 弹性阈值注意力：面向长上下文解码的学习型上下文稀疏化

    Elastic Threshold Attention: Learned Contextual Sparsity for Long-Context Decoding

    [https://arxiv.org/abs/2609.20888](https://arxiv.org/abs/2609.20888)

    提出弹性阈值注意力（ETA），一种端到端可训练的架构，通过从查询表示中预测动态上下文阈值，在不牺牲稠密模型质量的前提下实现长上下文解码的硬件加速，解决了KV缓存带来的内存带宽瓶颈。

    

    在长上下文解码过程中，庞大的KV缓存会造成严重的内存带宽瓶颈。稀疏注意力方法通过选择性加载来缓解这一问题，但代价是：僵化的启发式规则会丢弃必要的上下文，导致质量下降。我们提出了弹性阈值注意力（ETA），这是一种端到端可训练的架构，能够在不牺牲稠密模型质量的情况下实现硬件加速的解码速度。ETA直接从查询表示中预测动态的、上下文相关的阈值，使模型能够为困难的检索或推理步骤分配类似稠密的上下文，同时剪枝常规token。为了在不发生表示坍塌的情况下从头学习这一策略，ETA在训练期间通过乘性抑制将低于阈值的logits压向零，而不是直接删除它们。针对这种平滑的均匀注意力底座进行训练，提供了一个分布式的概率储备库，使得定位……

    arXiv:2609.20888v1 Announce Type: new  Abstract: Massive KV caches can cause severe memory-bandwidth bottlenecks during long-context decoding. Sparse attention methods mitigate this via selective loading, but that comes at a cost: rigid heuristics drop necessary context, leading to quality degradation. We introduce \textbf{Elastic Threshold Attention (ETA)}, an end-to-end trainable architecture that achieves hardware-accelerated decoding speed without sacrificing dense model quality. ETA predicts dynamic, contextual thresholds directly from query representations, allowing the model to allocate dense-like context to difficult retrieval or reasoning steps while pruning routine tokens. To learn this policy from scratch without representation collapse, ETA \emph{multiplicatively suppresses} sub-threshold logits toward zero during training rather than deleting them. Training against this smooth uniform attention floor provides a distributed probability reservoir that \textbf{causes localize
    
[^221]: 物理世界模型中：守恒换来稳定性，因式分解换来反事实能力

    Conservation Buys Stability and Factoring Buys Counterfactuals in Physical World Models

    [https://arxiv.org/abs/2609.19674](https://arxiv.org/abs/2609.19674)

    该论文证明物理世界模型的两种失效需要不同的结构性补救：用辛积分器演化学习到的能量可保持保守动力学几何结构、使长时程展开在训练范围100倍内保持稳定，而用显式线性因式分解编码物理耦合则能使模型泛化到从未见过的干预情形。

    

    学习得到的模拟器能够准确地复现其训练条件，但一旦条件发生改变，可能会以两种截然不同的方式失效。在长时间展开时，小误差不断累积，最终导致轨迹偏离物理上合理的行为；而在对某个物理参数进行干预时，模型可能继续遵循训练时所见到的规律，而非被干预后的规律。我们证明这两种失效需要不同的结构性补救措施。用辛积分器来演化学习到的能量函数，可以保持保守动力学的几何结构，使展开过程在长达训练范围100倍的时间内保持有界且物理上有意义，而同等容量的预测器、能量正则化的预测器以及经过调优的神经ODE则会发散。相比之下，通过显式的线性因式分解来编码物理耦合，能够使模型遵循从未见过的该耦合的符号方向，而无限制的参数化则仍然无法做到这一点。

    arXiv:2609.19674v1 Announce Type: new  Abstract: A learned simulator can reproduce its training conditions accurately yet fail in two distinct ways once those conditions change. Over long rollouts, small errors accumulate until the trajectory drifts away from physically plausible behavior; under an intervention on a physical parameter, the model may continue to follow the law seen during training rather than the intervened one. We show that these two failures require different structural remedies. Evolving a learned energy with a symplectic integrator preserves the geometry of the conservative dynamics and keeps rollouts bounded and physically meaningful for up to $100\times$ the training horizon, while equal-capacity predictors, an energy-regularized predictor, and a tuned neural ODE diverge. By contrast, encoding the physical coupling through an explicit linear factorization enables the model to follow a never-seen sign of that coupling, whereas an unrestricted parameterization remai
    
[^222]: 混合专家语言模型中专家的高阶剪枝

    Higher-order pruning of experts in mixture-of-experts language models

    [https://arxiv.org/abs/2609.18916](https://arxiv.org/abs/2609.18916)

    提出二阶剪枝方法HOPE，通过捕捉专家之间的高阶交互作用来可证明地最小化剪枝误差上界，在多个前沿MoE模型和基准测试上的剪枝效果优于忽略专家协作性的一阶方法。

    

    arXiv:2609.18916v1 公告类型：交叉 摘要：混合专家语言模型存在参数量庞大的问题，这造成了显著的内存瓶颈。专家剪枝是减少参数量最直接的方法，然而现有方法对每个专家独立地做出剪枝决策，并假设专家的贡献是纯粹可加的。实际上，混合专家模型中专家的使用本质上是协作性的。我们推导出了HOPE（专家高阶剪枝），这是一种二阶剪枝目标函数，可以证明能够最小化剪枝所产生的误差上界。我们证明REAP（一种最先进的一阶剪枝方法）是HOPE在忽略交互项时的特例。在三个前沿MoE模型（参数量高达1220亿）、两个不同的校准集以及多个基准测试（包括数学、指令遵循、编程和智能体任务套件）上，我们证明HOPE能够比现有方法做出更好的剪枝决策，并且……

    arXiv:2609.18916v1 Announce Type: cross  Abstract: Mixture-of-Experts (MoE) language models suffer from large parameter counts, which create a significant memory bottleneck. Expert pruning is the most direct approach for reducing this parameter count, yet existing methods make pruning decisions for each expert independently, and assume experts' contributions are purely additive. In reality, expert usage in MoEs is inherently cooperative. We derive HOPE (Higher-Order Pruning of Experts), a second-order pruning objective which provably minimizes an upper bound on the error resulting from pruning. We show that REAP (a state-of-the-art first-order pruning method) is a special case of HOPE where interaction terms are ignored. Across three frontier MoE models (up to 122B parameters), two distinct calibration sets, and multiple benchmarks (including math, instruction following, coding, and an agentic suite), we demonstrate that HOPE produces better pruning decisions than existing methods, and
    
[^223]: 超越二次损失：Adam优化器的稳定性相图

    Beyond Quadratic Loss: The Stability Phase Diagram of Adam

    [https://arxiv.org/abs/2609.18314](https://arxiv.org/abs/2609.18314)

    该研究通过绘制Adam优化器在$(\beta_1,\beta_2)$参数平面上的稳定性相图，发现一条近似线性边界$1-\beta_2=C(1-\beta_1)$可用于区分训练中是否出现损失尖峰，并揭示超二次损失景观（如高置信交叉熵损失形成的“核心-墙壁”结构）是决定该边界形状的关键因素。

    

    损失尖峰是神经网络训练中反复出现的不稳定性现象，可能由多种机制引起。特别是对于Adam优化器，宏观损失尖峰已被认为与优化器动力学相关，但其两个动量时间尺度如何支配这些尖峰仍不清楚。我们通过在$(\beta_1,\beta_2)$平面上绘制训练动力学图谱来研究这种依赖关系。在多种模型-任务设置中，一条近似线性的边界$1-\beta_2=C(1-\beta_1)$将出现尖峰与不出现尖峰的动力学区域分隔开来，而一维二次损失则产生近似三次方斜率的边界。一维超二次损失$L(x)\propto|x|^n$则恢复了近线性标度关系，并将边界系数与有效损失指数$n$联系起来。我们进一步表明，高置信度的交叉熵损失会发展出一种“核心-墙壁”景观，由狭窄的二次核心和随后陡峭的墙壁组成，这在优化器步长的尺度上产生有效的超二次行为。

    arXiv:2609.18314v1 Announce Type: new  Abstract: Loss spikes are recurrent instabilities in neural-network training and can arise from multiple mechanisms. For Adam in particular, macroscopic loss spikes have been linked to optimizer dynamics, yet how its two momentum timescales govern them remains unclear. We investigate this dependence by mapping training dynamics across the $(\beta_1,\beta_2)$ plane. Across a range of model--task settings, an approximately linear boundary, $1-\beta_2=C(1-\beta_1)$, separates spiky from non-spiky dynamics, whereas a one-dimensional quadratic loss produces approximately cubic slope. A one-dimensional superquadratic loss $L(x)\propto|x|^n$ recovers the near-linear scaling and links the boundary coefficient to the effective loss exponent $n$. We further show that confident cross-entropy losses develop a core--wall landscape comprising a narrow quadratic core followed by a steep wall, which produces effective superquadratic behavior at the scale of an op
    
[^224]: 高容量核联想记忆中稳定性边缘的信息几何自组织

    Information Geometric Self-Organization at the Edge of Stability in High-Capacity Kernel Associative Memories

    [https://arxiv.org/abs/2609.16827](https://arxiv.org/abs/2609.16827)

    本文通过Hessian特征值谱分析揭示了KLR联想记忆中“优化脊”本质上是秩1谱坍缩附近的几何奇点，并证明梯度下降的学习动力学在稳定性边缘处表现出瞬态自稳定行为，从而自发地组织到该最优区域。

    

    基于核逻辑回归（KLR）的高容量联想记忆展现出卓越的存储能力与鲁棒性。先前的实证研究识别出了一个超参数区域，即“优化脊”，在该区域中吸引子的稳定性达到最大。然而，这一区域的几何本质以及到达该区域所需的优化动力学机制一直不明确。本文研究了采用KLR训练的Hopfield网络中参数空间的静态几何以及梯度下降（GD）的学习轨迹。利用Hessian矩阵的特征值谱，我们揭示了“优化脊”对应于位于秩1谱坍缩附近的一个相边界，它作为一个几何奇点，其主曲率被大幅放大。此外，我们证明了学习动力学表现出一种由稳定性边缘现象驱动的瞬态自稳定行为……

    arXiv:2609.16827v1 Announce Type: new  Abstract: High-capacity associative memories based on Kernel Logistic Regression (KLR) exhibit exceptional storage capabilities and robustness. Previous empirical studies identified a hyperparameter regime, the "Ridge of Optimization," where attractor stability is maximized. However, the geometric nature of this regime and the optimization dynamics required to reach it have remained unclear. In this paper, we investigate the static geometry of the parameter space and the learning trajectory of Gradient Descent (GD) in KLR-trained Hopfield networks. Using the eigenvalue spectrum of the Hessian, we reveal that the Ridge corresponds to a phase boundary located adjacent to a rank-1 spectral collapse, acting as a geometric singularity where the principal curvature is massively amplified. Furthermore, we demonstrate that the learning dynamics exhibit a transient self-stabilizing behavior driven by the Edge of Stability (EoS) phenomenon. Rather than seek
    
[^225]: 一种适应已学习多变量结构的加权核函数近似方法

    A Weighted Kernel Method for Approximation that Adapts to Learned Multivariable Structure

    [https://arxiv.org/abs/2609.16606](https://arxiv.org/abs/2609.16606)

    提出了总敏感度核（TSK）方法，通过加权ANOVA核族学习多变量结构，并借助最小范数RKHS的选择从有限数据中唯一确定各输入的敏感度因子，从而实现对黑箱函数的自适应近似。

    

    从有限数据中近似多变量黑箱函数的输入输出行为极具挑战性，尤其是当对其输入的重要性及输入间相互作用一无所知时。我们引入了总敏感度核（TSKs），这是一种基于加权ANOVA核族的方法，能够学习并适应这种多变量结构。TSKs通过每个输入的因子来参数化目标函数各多变量分量上的权重。我们提出通过选择使目标函数具有最小范数的再生核希尔伯特空间（RKHS），直接从函数评估值中学习这些因子。在适当条件下，我们证明了这一范数最小化问题存在唯一解，并建立了基于最小范数插值的有限数据方法的一致性。学习到的TSK因子刻画了各个输入在相互作用和主效应中的参与程度，提供了一种依赖核的

    arXiv:2609.16606v1 Announce Type: new  Abstract: Approximating the input-output behavior of a multivariable black-box function from limited data is challenging when blind to the importance of its inputs and their interactions. We introduce total sensitivity kernels (TSKs), a method based on families of weighted ANOVA kernels that learn and adapt to this multivariable structure. TSKs parameterize the weights on each multivariable component of the target function by factors for each input. We propose learning these factors directly from function evaluations by selecting the reproducing kernel Hilbert space (RKHS) in which the target function has minimum norm. Under suitable conditions, we show that this norm-minimization problem admits a unique solution, and we establish consistency of a finite-data formulation based on minimum-norm interpolation. The learned TSK factors characterize the participation of individual inputs across interactions and main effects, providing a kernel-dependent
    
[^226]: 通过精确分布式样条合并从碎片化观测中恢复物理参数

    Recovering Physical Parameters from Fragmented Observations via Exact Distributed Spline Merging

    [https://arxiv.org/abs/2609.16579](https://arxiv.org/abs/2609.16579)

    本文提出精确分布式样条合并方法，各数据持有者仅需共享局部Gram矩阵和矩向量即可获得与集中式拟合数学上完全相同的解，并通过场重建与导数提取流程从分布式碎片观测中实现物理参数推断。

    

    科学测量通常分布在不同的地点、时间段和机构之间。将这些碎片组合成连续、可微的场，能够从其导数中恢复控制性物理参数。本文为实现这一目标做出了两项贡献。首先，将固定基岭回归统计量的既定可加结构应用于张量积样条场：每个数据持有者计算局部Gram矩阵和矩向量，合并后的解在数学上与集中式拟合完全相同，无需共享原始数据，也无需迭代同步。这一性质特定于固定特征的平方误差设置；本推导并未为一般联合训练的多层网络建立类似的保证。其次，一个完整的处理流程通过场重建、导数提取等步骤，将分布式观测与物理参数推断连接起来。

    arXiv:2609.16579v1 Announce Type: new  Abstract: Scientific measurements are frequently distributed across locations, time periods, and institutions. Combining such fragments into a continuous, differentiable field enables recovering governing physical parameters from its derivatives. This paper makes two contributions toward that goal. First, the established additive structure of fixed-basis ridge-regression statistics is applied to tensor-product spline fields: each data holder computes a local Gram matrix and moment vector, and the merged solution is mathematically identical to centralized fitting, with no raw data shared and no iterative synchronization. This property is specific to the fixed-feature squared-error setting; the present derivation does not establish an analogous guarantee for general jointly trained multilayer networks. Second, a complete pipeline connects distributed observations to physical parameter inference through field reconstruction, derivative extraction, an
    
[^227]: 监督图预测的图匹配松弛与摊销

    Graph Matching Relaxations and Amortization for Supervised Graph Prediction

    [https://arxiv.org/abs/2609.15437](https://arxiv.org/abs/2609.15437)

    该论文证明了Gromov-Wasserstein目标是监督图预测中最合适的图匹配松弛形式，并提出基于可微Sinkhorn算法的参数化匹配器来摊销图匹配问题，实现图预测模块与匹配器的联合学习。

    

    监督图预测（SGP）的端到端训练需要一个置换不变的损失函数来比较具有任意节点排序的预测图和目标图。这类损失函数通常涉及一个代价高昂的图匹配问题。我们首先研究了该问题的三种最优传输（Optimal Transport）松弛形式，并从理论和实证上表明，Gromov-Wasserstein（GW）目标最适合于监督图预测。随后，为了避免为每个训练样本求解由此产生的内层优化问题，我们提出对图匹配（节点对齐）问题进行摊销。对于每个训练样本，损失函数利用由参数化匹配器提供的传输计划，该匹配器基于应用于经验节点分布的可微Sinkhorn算法构建。图预测模块和匹配器被联合学习。我们在复杂度递增的玩具和现实世界监督图预测问题上展示了该方法的有效性，其中包括一个新颖的质谱到分子骨架（Mass-spectra to Scaffold）预测任务。

    arXiv:2609.15437v1 Announce Type: cross  Abstract: End-to-end Supervised Graph Prediction (SGP) requires a permutation-invariant loss to compare predicted and target graphs with arbitrary node orderings. Such losses typically involve a costly graph-matching problem. We first study three Optimal Transport relaxations of this problem and show, theoretically and empirically, that the Gromov-Wasserstein (GW) objective is the most suitable for SGP. Then, to avoid solving the resulting inner optimization for every training example, we propose to amortize the graph matching (node alignment) problem. For each training sample, the loss function leverages a transport plan provided by a parametric matcher based on the differentiable Sinkhorn algorithm applied on empirical node distributions. The graph prediction module and the matcher are jointly learned. We showcase the efficiency of this approach on toy and real world SGP problems of increasing complexity including a novel Mass-spectra to Scaff
    
[^228]: 远距离大梯度未必可靠：面向长时程自回归预测的可靠性加权信用分配

    Large Distant Gradients Need Not Be Reliable: reliability-weighted credit assignment for long-horizon autoregressive forecasting

    [https://arxiv.org/abs/2609.12890](https://arxiv.org/abs/2609.12890)

    提出Internal-DW方法，通过在反向传播中对每个残差块的恒等路由和非线性路由施加由显式噪声模型估计的有界维纳增益进行可靠性加权，在抑制长时程自回归预测中不可靠远距离梯度噪声的同时保留可预测的学习信号。

    

    在自回归预测中，长预测展开能够提供远距离的监督信号，但通过时间的反向传播（BPTT）需要将这些损失的梯度经过许多自回归步骤逐步传递。反复的雅可比矩阵乘积可能使远距离梯度在参数更新中占据主导地位，同时放大可预测信号与不可预测噪声；因此，大的远距离梯度并不一定携带可靠的学习信号。基于这一观察，我们提出了内部双维纳路由（Internal-DW），这是一种仅作用于反向传播过程的原则性干预方法，它在保留完整前向展开和所有时域损失的同时，对内部梯度路由进行可靠性加权。在每个残差块处，我们为恒等路由和非线性路由推导出有界的维纳增益，以在保留可预测学习信号与抑制不可预测变化之间取得平衡，并通过路由级别的梯度统计量和显式噪声模型对这些增益进行估计。在一个受控的（摘要在此处截断）

    arXiv:2609.12890v1 Announce Type: new  Abstract: In autoregressive forecasting, long prediction rollouts provide distant supervision, but backpropagation through time (BPTT) carries gradients from those losses through many autoregressive steps. Repeated Jacobian products can make distant gradients dominate the update while amplifying predictable signal and unpredictable noise together; a large distant gradient therefore need not carry reliable learning signal. Motivated by this observation, we introduce Internal Dual-Wiener routing (Internal-DW), a principled backward-only intervention that preserves the full forward rollout and all horizon losses while reliability-weighting internal gradient routes. At each residual block, we derive bounded Wiener gains for the identity and nonlinear routes that balance preserving predictable learning signal against suppressing unpredictable variation, and estimate them from route-level gradient statistics and an explicit noise model. In a controlled 
    
[^229]: EGGROLL展开：理解并改进大规模低秩进化策略

    EGGROLL, Unrolled: Understanding and Improving Low-Rank Evolution Strategies at Scale

    [https://arxiv.org/abs/2609.10980](https://arxiv.org/abs/2609.10980)

    本文首次从理论上刻画了面向大语言模型的低秩进化策略EGGROLL的更新场，揭示其可能引入非保守分量并逆转最优点的局部稳定性，同时证明了该方法的二次目标精确性并给出非渐近误差界，为理解与改进该方法奠定理论基础。

    

    EGGROLL通过用低秩高斯乘积（通常为秩一）替代稠密高斯权重扰动，使进化策略（ES）在大语言模型（LLM）上变得实用。这一选择在计算上颇具吸引力，但在几何上却相当严苛：尽管协方差为单位阵，每个秩一扰动都位于环境矩阵空间的一个零体积子集中。我们刻画了在有限秩和非零扰动半径下EGGROLL的平均更新场，并分析了其有限种群估计器的误差。该种群场是通过将一个显式预解式作用于由扰动平滑后的目标函数梯度而得到的。我们证明该预解式可能引入非保守分量，并可能逆转最优点的局部稳定性。尽管如此，EGGROLL在任意秩和任意半径下对所有二次目标函数都是精确的。对于光滑目标函数，其首个局部有限秩修正项为O(σ²/r)，且非渐近界控制着

    arXiv:2609.10980v1 Announce Type: new  Abstract: EGGROLL makes evolution strategies (ES) practical for LLMs by replacing dense Gaussian weight perturbations with low-rank Gaussian products, often of rank one. This choice is computationally attractive but geometrically severe: each rank-one perturbation lies in a zero-volume subset of the ambient matrix space, despite having identity covariance. We characterize the mean EGGROLL update field at finite rank and nonzero perturbation radii, then analyze the error of its finite-population estimator. The population field is obtained by applying an explicit resolvent to the gradient of the objective smoothed by the perturbations. We show that the resolvent can introduce a nonconservative component and can reverse the local stability of an optimum. EGGROLL is nevertheless exact on every quadratic objective at every rank and radius. For smooth objectives, its first local finite-rank correction is $O(\sigma^2/r)$, and nonasymptotic bounds control
    
[^230]: Rockafellar约束条件下极大单调算子之和的非极大性

    Nonmaximal sums of maximally monotone operators under Rockafellar's constraint qualification

    [https://arxiv.org/abs/2609.10487](https://arxiv.org/abs/2609.10487)

    本文通过计算一类图的单调极并施加正秩一扰动的构造定理，在$c_0$和$\ell^1$空间上构造出满足Rockafellar内部域条件但其和并非极大单调的极大单调算子对，从而推翻了Rockafellar和猜想。

    

    我们构造了Rockafellar和猜想（Rockafellar's sum conjecture）的反例，其中两个极大单调算子满足内部域条件，但它们的和不是极大单调的。我们在$c_0$空间上给出一个反例，并在具有通常范数的$\ell^1$空间上给出另一个反例。我们建立了一个一般的构造定理，该定理计算了一类图的完整单调极（monotone polar），给出了其极大单调性的充分必要条件，并展示了在此条件下正秩一扰动如何产生非极大的和。我们在$c_0$上验证了该定理的假设及其极大性判据，从而获得了该猜想的反例。此外，我们构造了一个从$\ell^1$到$c_0$的有界线性满射，并利用它得到了$\ell^1$上的反例。我们还提供了$c_0$反例及拉回引理的Lean形式化证明。

    arXiv:2609.10487v2 Announce Type: replace  Abstract: We construct counterexamples to Rockafellar's sum conjecture in which two maximally monotone operators satisfy the interior-domain condition but their sum is not maximally monotone. We give one counterexample on $c_0$ and another on $\ell^1$ with its usual norm. We establish a general construction theorem that computes the entire monotone polar of a class of graphs, gives a necessary and sufficient condition for their maximal monotonicity, and shows how a positive rank-one perturbation yields a nonmaximal sum under this condition. We verify the theorem's hypotheses and its maximality criterion on $c_0$, thereby obtaining a counterexample to the conjecture. Furthermore, we construct a bounded linear surjection from $\ell^1$ onto $c_0$ and use it to obtain the counterexample on $\ell^1$. Lean formalizations of the $c_0$ counterexample and the pullback lemma are also provided.
    
[^231]: 特征叠加中线性能及性的高概率保证

    High-probability guarantees for linear accessibility in feature superposition

    [https://arxiv.org/abs/2609.09556](https://arxiv.org/abs/2609.09556)

    该论文将特征叠加中的线性能及性建模为压缩感知问题，证明了充分维度只需线性规模（而非此前最坏情况的二次方限制）即可高概率地恢复同时激活的特征，从而量化了线性表示假设的几何约束并为稀疏自编码器和神经可解释性评估提供了理论框架。

    

    神经网络可以利用特征叠加来编码比维度数量更多的概念，但特征间的交叉干扰限制了同时激活特征的线性能及性。通过将线性能及性建模为一个压缩感知问题，我们在次高斯噪声下针对固定支撑集推导出高概率界，证明了充分维度以线性方式扩展（d=O_ε(k log m)），而非此前最坏情况下的二次方限制。随后，我们通过高斯尾近似在各系统参数下验证了这些界。这些结果量化了线性表示假设的几何约束，为评估稀疏自编码器、组合泛化和神经网络可解释性提供了一个框架。

    arXiv:2609.09556v1 Announce Type: cross  Abstract: Neural networks can leverage feature superposition to encode more concepts than dimensions, but cross-feature interference constrains the linear accessibility of simultaneously active features. By framing linear accessibility as a compressed sensing problem, we derive high-probability bounds for fixed supports under subgaussian noise, proving the sufficient dimension scales linearly ($d=O_{\varepsilon}(k \log m)$) rather than prior worst-case quadratic limits. We then validate these bounds across system parameters through Gaussian-tail approximations. These results quantify the geometric constraints of the linear representation hypothesis, providing a framework for evaluating sparse autoencoders, compositional generalization, and neural interpretability.
    
[^232]: 导向干扰反映的是模型的默认倾向，而非行为方向

    Steering Interference Reflects the Model's Defaults, Not the Behavior Directions

    [https://arxiv.org/abs/2609.06951](https://arxiv.org/abs/2609.06951)

    激活导向引发的副作用并非来自被导向的行为方向本身，而是由模型自身的默认偏好决定——无论导向何种行为，模型都会趋向其本已偏好的少数行为（如拒答、谄媚、诗歌化）。

    

    激活导向有望实现对语言模型行为的模块化控制：某种行为（如礼貌）对应模型激活中的一个方向，在模型生成时添加该方向应当能开启该行为，且不影响其他方面。但事实并非如此。我们探究了是什么决定了哪些其他行为会发生变化以及变化程度，发现起决定作用的是模型本身，而非被导向的行为。导向会使模型放松趋向于它本已偏好的一小部分行为，主要是拒答、谄媚和诗歌化倾向，且无论导向什么行为，这一行为集合大体相同。横跨24种行为和十个指令微调模型的三项结果支持这一结论，所有效应均由语言模型裁判从生成文本中读取，而非通过探针读取。这种读取方式很重要：所有24种行为都是线性可解码的，但只有20种行为会改变模型的实际输出内容。第一，一个不含任何行为内容、仅在特定方面与真实导向相匹配的方向……（摘要在此处被截断）

    arXiv:2609.06951v1 Announce Type: cross  Abstract: Activation steering promises modular control of language model behavior: a behavior such as politeness corresponds to a direction in a model's activations, and adding that direction while it generates should switch the behavior on and leave everything else alone. It does not. We ask what decides which other behaviors move, and by how much, and find that it is the model rather than the behavior being steered. A steer relaxes the model toward a small set of behaviors it already favors, chiefly refusal, sycophancy, and poeticism, and that set is much the same whatever is steered.   Three results across 24 behaviors and ten instruction-tuned models support this, every effect read off the generated text by a language-model judge rather than off a probe. That readout matters: all 24 behaviors are linearly decodable, but only 20 change what the model writes. First, a direction carrying no behavioral content, matched to a real steer only in th
    
[^233]: 拒答的几何学：为什么事后安全训练脆弱而预训练期安全却能持久

    The Geometry of Refusal: Why Post-Hoc Safety Is Fragile and Pretraining-Time Safety Persists

    [https://arxiv.org/abs/2609.06934](https://arxiv.org/abs/2609.06934)

    该论文从几何视角证明，事后安全训练（如RLHF）的更新与模型能力方向近乎正交，只是在完好的能力之上叠加一道薄而尖锐的拒答“闸门”而非真正删除能力，因此注定会被越狱等攻击绕过，而持久的安全必须在预训练阶段扎根。

    

    事后安全训练（RLHF、DPO）是目前对齐大语言模型的主流方法，然而越狱攻击（Zou et al., 2023b）、微调攻击（Qi et al., 2024）以及激活空间探测（Arditi et al., 2024）总能恢复出那些本应被移除的行为。我们对这种脆弱性给出了一种几何解释，并将其追溯到预训练过程中安全机制能够真正扎根的时机。我们将安全更新 $\Delta = W_{\text{safe}} - W_{\text{base}}$ 与模型能力的曲率（能力损失的经验 Fisher 信息）进行对照测量。研究发现，事后安全训练始终落入一个“抑制区间”：$\Delta$ 与能力方向近乎正交，其在子空间内的微小分量集中于少数高曲率方向。这一更新“薄而尖锐”——它是在完好的能力之上叠加的一道拒答闸门，而非对能力的真正抹除。一个核不动性引理解释了为什么这样的更新只能掩盖能力而无法将其移除，因此……（摘要原文在此处截断）

    arXiv:2609.06934v1 Announce Type: cross  Abstract: Post-hoc safety training (RLHF, DPO) is the dominant way to align large language models, yet jailbreaks (Zou et al., 2023b), fine-tuning attacks (Qi et al., 2024), and activation-space probes (Arditi et al., 2024) keep recovering the behaviors it was meant to remove. We give this fragility one geometric explanation and trace it to when, during pretraining, safety can take hold. We measure the safety update $\Delta = W_{\text{safe}} - W_{\text{base}}$ against the curvature of the model's capabilities (the empirical Fisher of a capability loss). Post-hoc safety consistently lands in a suppression regime: $\Delta$ is nearly orthogonal to the capability directions, and its small in-subspace part concentrates on a few high-curvature ones. The update is thin but sharp, a refusal gate laid over intact capabilities rather than erasure of them. A kernel-immobility lemma explains why such an update can only mask a capability, not remove it, so a
    
[^234]: SimpleMemVLA：一种简单而有效的面向视觉-语言-动作模型的原生视频记忆

    SimpleMemVLA: A Simple but Effective Native-Video Memory for Vision-Language-Action Models

    [https://arxiv.org/abs/2609.05533](https://arxiv.org/abs/2609.05533)

    SimpleMemVLA提出了一种无需专用记忆模块的视觉-语言-动作模型，通过完整保留历史信息并以时间戳视频格式直接输入骨干网络，利用子任务隐藏状态作为历史到流匹配动作头的唯一通道，从而有效解决长时程操作中的部分可观测问题。

    

    长时程操作是部分可观测的：选择下一个动作所需的信息可能仅出现在几分钟之前的观测中。现有的记忆机制——检索库、学习型压缩器、循环状态——必须在不知道未来决策需要什么的情况下，先决定从过去保留哪些信息。这一设计源于一种假设，即分钟级的历史数据过于庞大而无法直接处理，但现代VLM骨干网络已不再受此限制。在这项工作中，我们提出了SimpleMemVLA，一种无需专用记忆模块的VLA模型。它完整保留采样的历史信息，并以骨干网络预训练时所用的时间戳视频格式将其传递给骨干网络；生成的子任务的隐藏状态随后构成从历史信息到标准流匹配动作头的唯一通道。由于连续决策共享大部分历史信息，在动作执行期间预填充共享前缀可使延迟保持在接近

    arXiv:2609.05533v1 Announce Type: cross  Abstract: Long-horizon manipulation is partially observable: the information needed to choose the next action may appear only in observations from minutes earlier. Existing memory mechanisms: retrieval banks, learned compressors, recurrent states must decide what to keep from the past before knowing what a future decision will require. This was motivated by the assumption that minute-scale history is too large to process directly, which modern VLM backbones no longer make true. In this work, we introduce SimpleMemVLA, a VLA without a dedicated memory module. It keeps the sampled history intact and passes it to the backbone in the timestamped video format the backbone was pretrained to process; the hidden states of a generated sub-task then form the only channel from history to a standard flow-matching action head. Since consecutive decisions share most of their history, prefilling the shared prefix during action execution keeps latency close to 
    
[^235]: WEECFP-SuRGE：具有子结构旋转图距离编码的宽嵌入扩展连接性指纹

    WEECFP-SuRGE: Wide Embedded Extended Connectivity Fingerprint with Substructure Rotary Graph-distance Encoding

    [https://arxiv.org/abs/2609.04672](https://arxiv.org/abs/2609.04672)

    该论文提出了无需参数的分子指纹WEECFP以及结合子结构旋转图距离编码（SuRGE）的transformer架构WEECFP-SuRGE，在不使用任何外部预训练的情况下，在TDC ADMET排行榜的22项基准中取得多项第1名和总体领先的回归性能。

    

    我们提出了WEECFP，一种无需参数的1024维连续分子指纹，它将每个Morgan子结构散射到单个向量的约32个带符号位置上；同时提出了WEECFP-SuRGE，一种transformer架构，其自注意力机制将SuRGE（子结构旋转图距离编码）——一种由分子最短路径图距离参数化的类RoPE旋转——应用于WEECFP子结构token。该架构的7模型混合体（WEECFP-SuRGE Blend）在TDC ADMET排行榜上取得了最低的平均回归排名；在TDC ADMET排行榜上总体排名第2（仅次于预训练的MapLight+GNN），并且在所有不使用外部预训练的方法中总体排名第1；在完整的22项基准测试套件中，在Pgp、亲脂性、CYP2D6底物、微粒体清除率和LD50上获得排行榜第1名（WEECFP-NoSuRGE Blend另在HIA上达到第1名）——且全程无需任何外部预训练。在

    arXiv:2609.04672v1 Announce Type: new  Abstract: We introduce WEECFP, a parameter-free 1024-dimensional continuous molecular fingerprint that scatters each Morgan substructure across roughly thirty-two signed positions of a single vector, and WEECFP-SuRGE, a transformer architecture whose self-attention applies SuRGE (Substructure Rotary Graph-distance Encoding) -- a RoPE-like rotation parameterized by molecular shortest-path graph distance -- to WEECFP substructure tokens. A 7-model blend of this architecture (the WEECFP-SuRGE Blend) achieves the lowest average regression rank on the TDC ADMET leaderboard; is #2 overall on the TDC ADMET leaderboard (behind only pretrained MapLight+GNN), and is #1 overall among methods that use no external pretraining; takes leaderboard #1 finishes on Pgp, Lipophilicity, CYP2D6 Substrate, Clearance Microsome, and LD50 (with the WEECFP-NoSuRGE Blend separately reaching #1 on HIA) across the full 22-benchmark suite -- without any external pretraining. On
    
[^236]: 可证明安全的仿真到现实迁移

    Provably Safe Sim-to-Real Transfer

    [https://arxiv.org/abs/2609.01418](https://arxiv.org/abs/2609.01418)

    该论文提出并形式化了“安全仿真到现实迁移”问题，通过在无奖励安全强化学习框架内构建该问题，使智能体能够在利用不完美模拟器的同时确保现实世界数据收集的安全性，并为目标系统学习到接近最优的可行策略。

    

    为了缓解现实世界强化学习（RL）的样本复杂度问题，一种常见的做法是先在模拟器中训练策略（因为样本成本低廉），然后将学到的策略部署到现实世界中，并希望其能有效泛化。然而，这种直接的仿真到现实迁移并不保证成功：由于仿真与现实之间的失配（sim-to-real mismatch），在模拟器中训练的策略在现实世界中可能是次优的。纠正这种失配需要从真实系统收集数据，但在许多应用中（如机器人技术和医疗保健），这种数据收集过程本身受到安全约束的制约。这就引出了安全仿真到现实迁移的问题：智能体如何利用一个不完美的模拟器，同时确保现实世界数据收集的安全性，并为目标系统学习到接近最优的可行策略？我们通过在无奖励安全强化学习框架内构建安全仿真到现实迁移问题来应对这一挑战……

    arXiv:2609.01418v1 Announce Type: cross  Abstract: To mitigate the sample complexity of real-world reinforcement learning (RL), a common practice is to first train a policy in a simulator, where samples are cheap, and then deploy the learned policy in the real world with the hope that it generalizes effectively. Such direct sim-to-real transfer is not guaranteed to succeed: simulator-trained policies can be suboptimal in the real world due to sim-to-real mismatch. Correcting this mismatch requires collecting data from the real system, but in many applications, such as robotics and healthcare, this data-collection process is itself subject to safety constraints. This gives rise to the problem of safe sim-to-real transfer: how can an agent exploit an imperfect simulator while ensuring safe real-world data collection and learning a near-optimal feasible policy for the target system? We address this problem by formulating safe sim-to-real transfer within the framework of reward-free safe R
    
[^237]: MolLedger：一种具有化学基础ADME归因的加性图神经网络

    MolLedger: An Additive Graph Neural Network with Chemically Grounded ADME Attributions

    [https://arxiv.org/abs/2608.30636](https://arxiv.org/abs/2608.30636)

    提出MolLedger加性图神经网络，通过将ADME预测表示为逐原子分数之和，并利用辅助损失将原子分数锚定于化学性质，在不损失预测性能的前提下实现了精确且化学上可信的模型可解释性。

    

    优化吸收、分布、代谢和排泄（ADME）性质是小分子药物发现中的重要环节。许多机器学习模型已被构建用于预测ADME性质以促进这一优化过程，但解释模型预测结果仍然具有挑战性。我们提出了一种新的图神经网络架构，其内置了有意义的逐原子归因。我们的模型MolLedger输出的预测是逐原子分数之和。MolLedger的加性框架在不损失性能的情况下获得了精确的可解释性，因为全局上下文向量为加性头提供了足够的上下文信息，使其能够产生良好的逐原子分数。此外，MolLedger产生的归因比其他可解释性方法更忠实于化学性质，因为MolLedger中的辅助损失将原子分数锚定到化学性质上。我们的案例研究比较了多种方法对分子的解释……

    arXiv:2608.30636v1 Announce Type: new  Abstract: Optimizing absorption, distribution, metabolism, and excretion (ADME) is an important part of small molecule drug discovery. Many machine learning models have been built to predict ADME properties to facilitate this optimization process, but explaining model predictions is challenging. We propose a new graph neural network architecture with built-in meaningful per-atom attributions. Our model MolLedger outputs predictions that are the sum of per-atom scores. MolLedger's additive framework obtains exact interpretability at no cost to performance because the global context vector gives the additive head enough context to produce good per-atom scores. Furthermore, MolLedger produces attributions that are more faithful to chemical properties than other interpretability methods because the auxiliary loss in MolLedger anchors the atom scores to chemical properties. Our case studies comparing interpretations from multiple methods on molecular p
    
[^238]: 一种能力还是多种能力？检验前沿AI评估的经济效度

    One Capability or Many? Testing the Economic Validity of Frontier AI Evaluation

    [https://arxiv.org/abs/2608.29420](https://arxiv.org/abs/2608.29420)

    本研究通过潜变量模型对421个模型配置和12个基准的分析发现，经济基准测量的并非独立的独特能力，而是与其他基准共享的同一通用能力维度（单一因子解释74.5%的共同方差），从而质疑了前沿AI经济评估的构念效度。

    

    前沿模型排行榜如今基于经济基准对系统进行排名，这些基准测试模型执行专业任务的能力，涵盖从软件工程到银行工作流程等领域，而这些排名影响着组织的采购决策、监管机构的审查方向，以及对工作方式将如何变化的预期。这类基准究竟衡量的是一种有别于一般应试能力的独特能力，还是仅仅是重新表达了随着模型改进所有基准都会随之提升的同一维度，这是一个尚未被研究的构念效度问题。我们在一个固定哈希值的排行榜快照上对这一问题进行了检验，该快照包含421个模型配置和十二个基准测试（其中四个为经济类基准），将基准视为题目、模型视为被试，在一个潜变量模型中检验四个假设，且这些假设及其判定阈值均在分析前预先设定。结果表明，单一因子解释了74.5%的共同方差，并与模型发布日期高度相关（R² = 0.505），因此能力的主导轴在很大程度上是一个时间趋势。

    arXiv:2608.29420v1 Announce Type: new  Abstract: Frontier-model leaderboards now rank systems based on economic benchmarks, tests of how well models carry out professional tasks from software engineering to banking workflows, and those rankings inform what organisations buy, what regulators scrutinise, and expectations of how work will change. Whether such benchmarks measure a capability distinct from general test-taking, or re-express the one axis along which every benchmark rises as models improve, is a question of construct validity that has not yet been studied. We test it on a hash-pinned leaderboard snapshot of 421 model configurations across twelve benchmarks, four of them economic, treating benchmarks as items and models as respondents in a latent-variable model with four hypotheses and their thresholds fixed before analysis. A single factor explains 74.5% of common variance and tracks model release date (R^2 = 0.505), so the leading axis of capability is substantially a time t
    
[^239]: SimCast-S2S：一种通过气候模拟迁移学习实现亚季节降水预报的高效生成模型

    SimCast-S2S: An Efficient Generative Model for Subseasonal Precipitation Forecasting via Transfer Learning from Climate Simulations

    [https://arxiv.org/abs/2608.26594](https://arxiv.org/abs/2608.26594)

    SimCast-S2S通过潜扩散生成框架和气候模拟迁移学习，实现了高效且概率性的亚季节降水预报，解决了不确定性量化和计算成本两大核心瓶颈。

    

    arXiv:2608.26594v1 公告类型：新 摘要：亚季节到季节（S2S）降水预报具有重大的经济和社会影响，但由于预测信号弱、相关不确定性高以及业务系统的计算成本限制了模拟保真度，这一任务仍然具有挑战性。我们引入了SimCast-S2S，一种用于概率性S2S降水预报的生成式潜扩散框架，旨在解决数据驱动预测中的三个主要瓶颈。首先，由于S2S预测需要不确定性量化，而不仅仅是确定性点预报，SimCast-S2S是首个采用基于扩散的生成流程进行S2S预测的数据驱动系统，能够有效从底层条件分布中采样。其次，由于在物理空间中生成大规模概率集成计算成本高昂，SimCast-S2S改为在由变分自编码器学习的紧凑潜空间中运行。

    arXiv:2608.26594v1 Announce Type: new  Abstract: Subseasonal-to-seasonal (S2S) precipitation forecasting has substantial financial and societal impact, yet remains challenging because of weak predictive signals, high associated uncertainty, and the computational cost of operational systems, which constrains simulation fidelity. We introduce SimCast-S2S, a generative latent-diffusion framework for probabilistic S2S precipitation forecasting that addresses three major bottlenecks in data-driven prediction. First, because S2S prediction requires uncertainty quantification rather than only deterministic point forecasts, SimCast-S2S is the first data-driven system that uses a diffusion-based generative pipeline for S2S prediction, enabling effective sampling from the underlying conditional distribution. Second, since generating large probabilistic ensembles is computationally costly in physical space, SimCast-S2S instead operates in a compact latent space learned by variational autoencoders
    
[^240]: 行星预测引擎：通过智能数据选择和基础模型嵌入实现自主地理空间预测

    Planetary Prediction Engine: Autonomous Geospatial Prediction via Intelligent Data Selection and Foundation Model Embeddings

    [https://arxiv.org/abs/2608.26088](https://arxiv.org/abs/2608.26088)

    行星预测引擎是一个自主AI系统，能从自然语言查询直接端到端执行地理空间预测，通过智能数据选择和基础模型嵌入，自动整合多模态数据并搜索最优模型，以应对全球性挑战。

    

    应对从粮食安全、灾害风险到疾病爆发和社会经济脆弱性等关键全球挑战，需要高保真度的地理空间建模。然而，构建预测性行星模型仍受制于碎片化的数据生态系统，需要手动数据检索、多模态数据整理和融合以及迭代模型选择。我们提出了行星预测引擎（PPE），这是一种自主AI系统，可直接从自然语言查询中执行端到端工作流程。PPE动态合成多模态数据集，在开放网络和地球观测平台（Data Commons、Google Earth Engine）上检索时空相关协变量，并将其与地理空间基础模型嵌入（PDFM、AlphaEarth）融合。同时，它通过自动过拟合防护搜索针对任务定制的模型架构家族。在多样化的任务、地理区域和科学领域中，该引擎展现出显著性能。

    arXiv:2608.26088v1 Announce Type: cross  Abstract: Addressing critical global challenges, from food security and disaster risk to disease outbreaks and socio-economic vulnerability, demands high-fidelity geospatial modeling. However, building predictive planetary models remains bottlenecked by a fragmented data ecosystem, requiring manual data retrieval, multimodal data curation and fusion along with iterative model selection. We present the Planetary Prediction Engine (PPE), an autonomous AI system that executes this end-to-end workflow directly from natural-language queries. PPE synthesizes multimodal datasets on the fly, retrieving spatiotemporally relevant covariates across open-web and Earth observation platforms (Data Commons, Google Earth Engine) and fusing them with geospatial foundation model embeddings (PDFM, AlphaEarth). Simultaneously, it searches over task-tailored model architecture families with automated overfitting guards. Across diverse tasks, geographies, and scienti
    
[^241]: 连续吉布斯采样的可证明量子-经典分离

    Provable Quantum--Classical Separation for Continuous Gibbs Sampling

    [https://arxiv.org/abs/2608.24527](https://arxiv.org/abs/2608.24527)

    本文首次证明了连续域吉布斯采样问题中量子算法相比经典算法具有二次加速优势，该优势在低温高维下呈指数增长。

    

    我们证明了在连续域上采样问题的首个量子-经典分离。对于一类吉布斯态 $p\propto e^{-\beta E}$，定义在环面 $\mathbb{T}^d$ 上，具有光滑（$s$-Gevrey）势垒和势垒幅度 $\alpha=e^{\beta\Delta}$（其中 $\Delta = \max E-\min E$），所有经典算法——查询对数值、梯度或任何高阶导数——在总变差距离中实现恒定精度采样需要 $\Omega(\alpha)$ 次查询，而基于量子奇异值阈值和温度退火的量子算法仅需 $\tilde{O}\left(\sqrt{\alpha}\right)$ 次梯度预言机查询。该优势在势垒幅度上是二次的，在低温下随维度呈指数增长，即 $e^{\Omega(d)}$。经典下界是信息论的，适用于所有具有吉布斯势查询访问权限的经典算法。

    arXiv:2608.24527v1 Announce Type: cross  Abstract: We prove the first quantum--classical separation for a sampling problem over a continuous domain. For a class of Gibbs states $p\propto e^{-\beta E}$ on the torus $\mathbb{T}^d$ with smooth ($s$-Gevrey) potential and barrier amplitude $\alpha=e^{\beta\Delta}$, where $\Delta = \max E-\min E$, every classical algorithm---querying the value, gradient, or any higher-order derivatives of the log-density---requires $\Omega(\alpha)$ queries to sample at constant accuracy in total variation distance, while a quantum algorithm based on quantum singular value thresholding and temperature annealing samples with $\tilde{O}\left(\sqrt{\alpha}\right)$ queries to an oracle for the gradient. The advantage is quadratic in the barrier amplitude, which becomes exponential in the dimension, $e^{\Omega(d)}$, at low temperature. The classical bound is information-theoretic, holding for every classical algorithm with query access to the Gibbs potential and i
    
[^242]: PolyChirp：基于TinyML的低功耗声学传感器多物种鸟类鸣声分类

    PolyChirp: Multi-Species Birdsong Classification Using TinyML on Low-Power Acoustic Sensors

    [https://arxiv.org/abs/2608.23101](https://arxiv.org/abs/2608.23101)

    PolyChirp通过结合生物专业知识、自动化数据集和NPU加速的微型多类模型，首次实现了低功耗微控制器上多物种鸟类鸣声的实时分类。

    

    arXiv:2608.23101v1 公告类型：交叉 摘要：TinyML领域的最新进展表明，基于微控制器的低功耗硬件能够利用声学传感器数据，在一次电池充电的情况下，实时监测整个繁殖期内的鸟类物种。然而，目前低功耗微控制器上的先进技术仅限于单一物种的二分类。相比之下，实际的动物监测部署往往需要同时针对多个物种。为应对这一挑战，我们开发了PolyChirp，一种结合生物领域专业知识、自动化数据集整理、神经架构优化和新型硬件的方法，以实现野外多类鸟类物种检测。PolyChirp基于新设计的微型多类模型，利用最新的微控制器和带有神经处理单元（NPU）的硬件加速。我们评估了这些模型的预测性能，并测量了它们的计算效率。

    arXiv:2608.23101v1 Announce Type: cross  Abstract: Recent progress in the field of TinyML has demonstrated that low-power hardware based on microcontrollers can achieve bird species monitoring in real time based on acoustic sensor data for an entire breeding period on a single battery charge. However, the state of the art on low-power microcontrollers was so far limited to binary classification of a single species. In contrast, real fauna monitoring deployments often target multiple species simultaneously. To address this challenge we develop PolyChirp, an approach combining biological domain expertise, automated dataset curation, neural architecture optimization and novel hardware to achieve multiclass bird species detection in the wild. PolyChirp is based on newly designed tiny multiclass models that leverage recent microcontrollers and hardware acceleration with a neural processing unit (NPU). We evaluate the predictive performance of these models, and we measure their computational
    
[^243]: 变压器的通信图谱

    The Communication Map of a Transformer

    [https://arxiv.org/abs/2608.22007](https://arxiv.org/abs/2608.22007)

    提出了一种从权重出发绘制变压器所有潜在通信通道的“通信图谱”方法，能高效计算并揭示大多数注意力头对的耦合或回避模式，且具有广泛适用性。

    

    arXiv:2608.22007v1 公告类型：交叉 摘要：变压器的组件通过写入和读取共享残差流进行通信，机制可解释性已通过手工逐个电路绘制了这些连接。我们提出了通信图谱，它仅从权重出发，绘制了语言模型中所有潜在通信通道，将Elhage等人（2021）的组成分数推广为覆盖所有18类连接（从整个注意力头电路到单个神经元）的单一耦合系数。对所有候选通道的普查，从GPT-2中的$6.3\times10^{8}$到Pythia-6.9B中的$1.3\times10^{11}$，发现70-89%的头对方向偏离随机水平，有些强耦合，另一些则主动避免彼此。完整图谱在单个消费级GPU上计算GPT-2需15秒，Pythia-6.9B需11分钟。两个应用展示了该图谱的实用性。在应用1中，最强的头对头耦合恢复了

    arXiv:2608.22007v1 Announce Type: cross  Abstract: The components of a transformer communicate by writing to and reading from a shared residual stream, and mechanistic interpretability has mapped these connections by hand, one circuit at a time. We present the communication map, which charts every potential communication channel in a language model from weights alone, generalizing the composition score of Elhage et al. (2021) into a single coupling coefficient covering all 18 connection classes, from entire attention head circuits to single neurons. The census of all candidate channels, from $6.3\times10^{8}$ in GPT-2 to $1.3\times10^{11}$ in Pythia-6.9B, finds that 70-89% of head pairs are oriented far from chance, some coupled strongly and others actively avoiding each other. The full map costs 15 seconds for GPT-2 and 11 minutes for Pythia-6.9B on one consumer GPU. Two applications demonstrate the utility of the map. In Application 1, the strongest head-to-head couplings recover the
    
[^244]: DecoVAE：一种轻量级可解释趋势-季节VAE框架，用于高效概率时间序列预测

    DecoVAE: a Lightweight Interpretable Trend-Seasonal VAE Framework for Efficient Probabilistic Time Series Forecasting

    [https://arxiv.org/abs/2608.20052](https://arxiv.org/abs/2608.20052)

    本文提出DecoVAE，一个轻量级可解释的VAE框架，通过趋势差分正则化和频域复高斯VAE显式分解时间序列，在七个基准上持续优于现有方法，同时降低内存和计算开销。

    

    概率时间序列预测仍然具有挑战性，主要因为建模不同的趋势和季节动态需要专门的方法。现有方法往往无法捕捉这些组件的独特内在属性，缺乏可解释性，或遭受沉重的内存和运行时间开销。为解决这些限制，我们提出了DecoVAE，一种轻量级可解释趋势-季节VAE框架，通过应用领域特定的归纳偏置，将时间序列显式分解为趋势和季节组件。趋势流通过潜在轨迹上的差分正则化器强制执行结构平滑性，类似于Hodrick-Prescott滤波器。同时，季节流通过复高斯VAE在频域中操作，天然捕捉周期性模式的幅度和相位。在七个真实世界基准上的广泛评估表明，DecoVAE始终优于现有方法。

    arXiv:2608.20052v1 Announce Type: new  Abstract: Probabilistic time series forecasting remains challenging, largely because modeling distinct trend and seasonal dynamics requires specialized approaches. Existing methods often fail to capture the unique inner properties of these components, lack interpretability, or suffer from heavy memory and runtime overhead. To address these limitations, we propose DecoVAE, a lightweight interpretable trend-seasonal VAE framework that explicitly decomposes time series into trend and seasonal components by applying domain-specific inductive biases. The trend stream enforces structural smoothness using a differential regularizer on the latent trajectory, analogous to the Hodrick-Prescott filter. Concurrently, the seasonal stream operates in the frequency domain via a complex Gaussian VAE, natively capturing the amplitude and phase of periodic patterns. Extensive evaluations across seven real-world benchmarks show that DecoVAE consistently outperforms 
    
[^245]: 过于自信而不安全：用于可靠日志异常检测的模型校准

    Too Sure to Be Safe: Model Calibration for Reliable Log Anomaly Detection

    [https://arxiv.org/abs/2608.17965](https://arxiv.org/abs/2608.17965)

    本文提出LoRD框架，通过从正确分类样本的潜在表示中学习可靠性模型，解决日志异常检测器中置信度校准不良的问题，确保错误预测不被过度自信。

    

    在线日志异常检测对于维护大规模计算系统的可靠性至关重要。尽管基于语言模型的日志异常检测器取得了强大的检测性能，但其置信度估计仍校准不佳。我们表明，这些检测器经常对错误预测赋予过高的置信度，尤其是在严重类别不平衡下的异常日志中。此外，即使传统校准指标显示校准良好，错误预测的置信度仍持续偏高，这为运维监控系统造成了关键可靠性缺口。为解决此问题，我们提出了日志重建与距离（LoRD），一种轻量级的事后校准框架，用于可靠的日志异常检测。LoRD从正确分类的验证样本的潜在表示中学习预测路径特定的可靠性模型，并估计预测可靠性阈值。

    arXiv:2608.17965v1 Announce Type: cross  Abstract: Online log anomaly detection is critical for maintaining the reliability of large-scale computing systems. Although recent language model-based log anomaly detectors achieve strong detection performance, their confidence estimates remain poorly calibrated. We show that these detectors frequently assign excessive confidence to incorrect predictions, particularly for anomalous logs under severe class imbalance. Moreover, confidence on erroneous predictions remains persistently high even when conventional calibration metrics indicate good calibration, creating a critical reliability gap for operational monitoring systems. To address this issue, we propose Log Reconstruction and Distance (LoRD), a lightweight post-hoc calibration framework for reliable log anomaly detection. LoRD learns prediction-route-specific reliability models from latent representations of correctly classified validation samples and estimates prediction reliability th
    
[^246]: LiD-GLM：利普希茨约束的深度广义线性模型

    LiD-GLM: Lipschitz-constrained Deep Generalized Linear Models

    [https://arxiv.org/abs/2608.16340](https://arxiv.org/abs/2608.16340)

    提出一种利用可逆残差网络增强广义线性模型的方法，在保持随机单调性的同时实现非线性参数估计和分布假设的灵活校正。

    

    摘要：arXiv:2608.16340v1 公告类型：交叉 摘要：将传统统计模型与神经网络（NN）组件结合成半结构化混合模型，是一种引人入胜的方法，旨在构建理想情况下兼具传统可解释性与神经网络前所未有的灵活性的模型。为了保持可解释性，通常需要限制神经网络组件，以防止它们主导模型。然而，现有对神经网络组件施加结构约束的方法严重限制了模型的灵活性；相反，仅施加弱且间接约束的方法则失去了有意义的可解释性。因此，我们提出的方法利用可逆残差神经网络（i-ResNets）为广义线性模型配备非线性参数估计和对其分布假设的灵活校正，同时始终保留所建模分布在（原线性）变量上的随机单调性。

    arXiv:2608.16340v1 Announce Type: cross  Abstract: The combination of traditional statistical models and neural network (NN) components into semi-structured hybrid models is an intriguing approach to construct models that, ideally, combine traditional interpretability with the unprecedented flexibility of NNs. In order to preserve interpretability, it is usually necessary to restrict the NN components to prevent them from dominating the model. However, existing methods that enforce structural constraints on their NN components severely limit their models' flexibility; in contrast, methods that only enforce weak, indirect constraints lose meaningful interpretability. The method we propose therefore leverages invertible residual neural networks (i-ResNets) to equip generalized linear models with both nonlinear parameter estimation and a flexible correction of their distributional assumptions while always retaining stochastic monotonicity of the modeled distribution in the (formerly linea
    
[^247]: 非扩张映射的马尔可夫Halpern迭代的巴拿赫空间理论

    A Banach-Space Theory of Markovian Halpern Iteration for Non-Expansive Maps

    [https://arxiv.org/abs/2608.15966](https://arxiv.org/abs/2608.15966)

    本文提出了一种在巴拿赫空间中基于方差缩减的马尔可夫PAGE-Halpern迭代方法，通过泊松方程和位移级别Halpern界，将非扩张映射不动点逼近的样本复杂度从$\tilde O(\epsilon^{-5})$降低至$\tilde O(\epsilon^{-3})$。

    

    我们研究了当预言机样本来自连续马尔可夫轨迹时，非扩张算子不动点的随机逼近问题。直接的分块小批量实现Halpern迭代达到了期望的末次迭代残差为$O(\log N/N)$，但累积了$\tilde O(\epsilon^{-5})$马尔可夫样本的实质性复杂度。因此，我们引入了一种方差缩减的马尔可夫PAGE-Halpern方法，其刷新和同态差分块通过泊松方程进行分析。在希尔伯特空间中，$I-T$的余强性导致$O(\epsilon^{-3})$的样本复杂度。我们的主要结果将这一构造扩展到一般的有限维巴拿赫空间。一个位移级别的Halpern界取代了希尔伯特空间势能，并在原始非扩张范数下产生了$\tilde O(\epsilon^{-3})$的样本复杂度。我们还建立了具有相同主导精度的概率保证。

    arXiv:2608.15966v1 Announce Type: new  Abstract: We study stochastic approximation of fixed points of a non-expansive operator when the oracle samples originate from a continuing Markovian trajectory. A direct block-minibatch implementation of Halpern iteration attains an expected last-iterate residual of order $O(\log N/N)$, but accrues a substantive complexity of $\tilde O(\epsilon^{-5})$ Markovian samples. We therefore introduce a variance-reduced Markovian PAGE-Halpern method whose refresh and same-state difference blocks are analyzed through the Poisson equation. In Hilbert spaces, the cocoercivity of $I-T$ results in an $O(\epsilon^{-3})$ sample complexity. Our main result extends this construction to a general finite-dimensional Banach space. A displacement-level Halpern bound replaces the Hilbert-space potential and yields $\tilde O(\epsilon^{-3})$ sample complexity in the original non-expansiveness norm. We also establish a high-probability guarantee with the same leading accu
    
[^248]: 基于连续深度批处理的循环语言模型深度自适应推理

    Depth-adaptive Inference of Looped Language Models via Continuous Depth Batching

    [https://arxiv.org/abs/2608.09444](https://arxiv.org/abs/2608.09444)

    本文提出了首个针对循环语言模型深度自适应推理的高效方法——连续深度批处理（CDB），通过在循环步骤之间动态重组批次、管理循环KV缓存并提前预测token退出时机，解决了不同循环深度token无法被标准批处理系统高效处理的问题。

    

    循环语言模型的一项主要优势是深度自适应推理。通过将共享层块循环可变次数，模型可以对“简单”token使用更少的计算，而对“困难”token使用更多的计算。然而，具有不同循环次数的token无法共享统一的前向传播过程，因此无法由标准批处理系统（如vLLM）处理。因此，深度自适应推理的实际价值取决于能否实现高效的批处理。我们提出了首个面向深度自适应循环语言模型的高效方法——连续深度批处理，该方法在循环步骤之间形成新的批次。我们的方法能够动态调度架构中的循环部分与非循环部分，管理循环KV缓存，并提前预测哪些token将退出循环，从而可以异步地准备批次。在Ouro 1.4B和Huginn 3.5B上的实验表明，完全循环架构最适合深度自适应

    arXiv:2608.09444v2 Announce Type: replace-cross  Abstract: A main promise of looped language models is depth-adaptive inference. By looping a block of shared layers a variable number of times, the model can use less compute for "easy" tokens and more for "hard" ones. However, tokens with different numbers of loops cannot share a uniform forward pass and therefore cannot be handled by standard batching systems such as vLLM. The practical value of depth-adaptive inference thus hinges on whether batching can be made efficient. We introduce the first efficient method for depth-adaptive looped LMs via continuous depth batching (CDB), which forms new batches between loop steps. Our method dynamically schedules looped and non-looped parts of the architecture, manages looped KV-caching, and predicts which tokens will exit the loop in advance so it can prepare batches asynchronously. Experiments on Ouro 1.4B and Huginn 3.5B show that fully looped architectures are best suited to depth-adaptive 
    
[^249]: Aftab：并行化Q网络中CNN编码器与先进价值函数的综合基准

    Aftab: A Comprehensive Benchmark of CNN Encoders and Advanced Value Functions in Parallelized Q-Networks

    [https://arxiv.org/abs/2608.07335](https://arxiv.org/abs/2608.07335)

    本文系统评估了八种CNN编码器在并行化Q网络中的性能，并结合Hadamax编码与多种价值函数头，提出了一个在Atari-57上表现优异的复合架构。

    

    arXiv:2608.07335v2 公告类型：替换交叉 摘要：深度强化学习的最新进展日益倾向于简化、高度并行化的范式。值得注意的是，并行化Q网络（PQN）算法能够在无需经验回放缓冲区或目标网络的情况下进行离策略价值学习。然而，在这些无缓冲区设置中运行的视觉编码器的表示能力和计算效率仍相对未被充分探索。在本工作中，我们系统性地研究了PQN内卷积神经网络的架构设计空间。我们评估了八种不同的CNN拓扑结构，同时明确表征了它们的参数和计算需求。我们进一步通过将Hadamax编码范式与分类、集成和决斗价值头集成，研究了乘性表示学习和先进价值估计的效果。在Atari-57上的广泛实验表明，我们最终的复合架构...

    arXiv:2608.07335v2 Announce Type: replace-cross  Abstract: Recent advancements in deep reinforcement learning have increasingly favored simplified, highly parallelized paradigms. Notably, the Parallelized Q-Network (PQN) algorithm enables off-policy value learning without relying on experience replay buffers or target networks. However, the representational capacity and computational efficiency of visual encoders operating in these buffer-free settings remain comparatively underexplored. In this work, we systematically investigate the architectural design space of Convolutional Neural Networks within PQN. We evaluate eight distinct CNN topologies while explicitly characterizing their parameter and computational requirements. We further study the effect of multiplicative representation learning and advanced value estimation by integrating the Hadamax encoding paradigm with categorical, ensemble, and dueling value heads. Extensive experiments on Atari-57 show that our final composite arc
    
[^250]: 并非所有发散都应被抑制：在线策略蒸馏中的反事实可恢复性

    Not Every Divergence Should Be Suppressed: Counterfactual Recoverability in On-Policy Distillation

    [https://arxiv.org/abs/2608.04408](https://arxiv.org/abs/2608.04408)

    本文提出反事实可恢复性框架，通过教师续写与回滚分支重放错误状态来区分可恢复与不可逆的错误，并据此决定在线策略蒸馏中对轨迹的保留、回滚或常规监督策略，其可恢复性代理指标AUC达1.000，远超仅依赖发散度指标的0.392。

    

    在线策略蒸馏（OPD）对学生模型访问过的轨迹进行监督，然而基于发散度的规则无法判断一个错误前缀是否仍然可以被纠正。我们将这一决策问题形式化为反事实可恢复性，并通过预算匹配的教师续写分支与回滚分支对每个错误状态进行重放。根据二者的相对成功率，状态被分类为可恢复的、不可逆但可避免的、或模糊的，这些标签指导训练是保留、回滚还是常规监督相应的轨迹。在AIME分支诊断中，可恢复状态的平均“续写减回滚”效应为0.185，而不可逆但可避免的状态为-1.000，表明二者具有截然相反的干预偏好。基于分支实验导出的可恢复性代理指标达到了1.000的AUC，大幅优于仅使用发散度指标的0.392。在冻结评估中，可恢复性感知的控制方法取得了……（原文摘要此处截断）

    arXiv:2608.04408v2 Announce Type: replace-cross  Abstract: On-policy distillation (OPD) supervises student-visited trajectories, yet divergence-based rules cannot determine whether an erroneous prefix remains correctable. We formulate this decision as counterfactual recoverability and replay each error state through budget-matched teacher-continuation and rollback branches. Based on their relative success, states are categorized as recoverable, irreversible-but-avoidable, or ambiguous, and these labels guide whether training retains, rolls back, or conventionally supervises the corresponding trajectory. On AIME branch diagnostics, the mean continuation-minus-rollback effect is 0.185 for recoverable states and -1.000 for irreversible-but-avoidable states, demonstrating opposite intervention preferences. A branch-derived recoverability proxy achieves an AUC of 1.000, substantially outperforming divergence alone at 0.392. Across frozen evaluations, recoverability-aware control achieves th
    
[^251]: DAIF：一种基于近似消息传递的数据驱动多模态监督学习中间融合框架

    DAIF: A Data-Driven Intermediate Fusion Framework for Multimodal Supervised Learning via Approximate Message Passing

    [https://arxiv.org/abs/2608.02769](https://arxiv.org/abs/2608.02769)

    DAIF提出了一种数据自适应的中间融合框架，结合随机矩阵理论与非参数依赖性度量，通过根据模态间依赖性对模态聚类并进行经验贝叶斯先验估计，直接从数据中学习融合结构，克服了传统预定义融合架构无法适应模态间真实依赖关系的缺陷。

    

    多模态监督学习旨在利用多个异构数据源来提升预测性能。其核心挑战在于确定模态间的融合粒度：过度整合可能放大噪声，而整合不足则无法充分利用跨模态依赖关系。现有方法依赖于预先指定的融合架构，从早期融合到晚期融合，这些架构可能无法适应模态之间潜在的依赖结构。我们提出了DAIF，一个数据自适应的中间融合框架，它结合了随机矩阵理论和非参数依赖性度量，直接从数据中学习融合结构。我们在贝叶斯多模态因子模型的框架下进行操作，其中潜在因子的先验分布决定了跨模态依赖关系。我们的方法基于估计的模态间依赖性对模态进行聚类，然后对各簇的先验进行经验贝叶斯估计。这些估计出的先验…

    arXiv:2608.02769v2 Announce Type: replace-cross  Abstract: Multimodal supervised learning seeks to leverage multiple heterogeneous data sources to improve predictive performance. A central challenge is determining the fusion granularity across modalities: over-integration may amplify noise while under-integration fails to exploit cross-modal dependence. Existing approaches rely on pre-specified fusion architectures, from early to late fusion, that may not adapt to the underlying dependence structure among modalities. We propose DAIF, a data adaptive intermediate fusion framework that combines random matrix theory and non-parametric dependence measures to learn fusion structure directly from data. We operate under a Bayesian multimodal factor model where the prior on the latent factors determines the cross-modal dependence. Our method clusters modalities based on estimated intermodal dependence, then performs clusterwise empirical Bayes estimation of the priors. These estimated priors a
    
[^252]: 在多模态持续学习中正则化模态贡献漂移

    Regularizing modality contribution drift in multimodal continual learning

    [https://arxiv.org/abs/2607.27260](https://arxiv.org/abs/2607.27260)

    该论文首次提出“模态贡献漂移（MCD）”概念及其量化评分，揭示模态贡献变化是多模态持续学习中遗忘的关键成因，并设计了相应的持续正则化方法来有效缓解遗忘。

    

    多模态持续学习（MMCL）旨在从多模态数据中获取新知识，同时保留先前学到的知识。现有的多模态持续学习方法主要通过对齐跨模态表示或保持特征级语义相似性来缓解遗忘。然而，不同任务可能依赖不同的模态，学习新任务可能会改变模态对先前学习任务预测结果的贡献方式。目前，模态贡献在增量学习阶段如何演变，以及这种变化与多模态持续学习中的遗忘之间的关系，仍未得到充分研究。我们将这种变化命名为模态贡献漂移，并提出了基于受控模态子集干预的MCD评分。我们的理论和实证分析表明，贡献漂移会导致遗忘，而现有的多模态持续学习方法和传统持续学习方法均无法有效缓解MCD。为解决这一问题，我们提出了持续模态贡献漂移正则化方法。

    arXiv:2607.27260v2 Announce Type: replace  Abstract: Multimodal continual learning (MMCL) aims to acquire new knowledge from multimodal data while retaining previously learned knowledge. Existing MMCL methods primarily mitigate forgetting by aligning cross-modal representations or preserving feature-level semantic similarity. However, different tasks may rely on different modalities, and learning new tasks can alter how modalities contribute to predictions on previously learned tasks. It remains underexplored how modality contributions evolve across incremental stages and how such changes relate to forgetting in MMCL. We term such changes Modality Contribution Drift (MCD) and introduce an MCD score based on controlled modality-subset interventions. Our theoretical and empirical analyses show how contribution drift can lead to forgetting, while existing MMCL and conventional CL methods do not effectively mitigate MCD. To address this issue, we propose Continual Modality Contribution Dri
    
[^253]: 非凸神经网络中噪声解的鲁棒性研究

    On the robustness of noisy solutions in non-convex neural networks

    [https://arxiv.org/abs/2607.27000](https://arxiv.org/abs/2607.27000)

    本文将零温下限制算法可达性的重叠间隙性质（OGP）推广至有限温度情形，证明了冻结一步复制对称破缺解在任意有限温度下依然存在，并给出基于单模式吉布斯权重光滑性的一般性判据，从而刻画了非凸神经网络中噪声解的鲁棒性。

    

    arXiv:2607.27000v2 公告类型：replace-cross 摘要：非凸神经网络模型中的优化过程深受解空间几何结构的影响：稀疏、孤立、点状的解簇通常在算法上难以触及，而宽阔平坦的区域尽管相对罕见，却可以被高效地找到。在零温情形下，这一图景已通过重叠间隙性质（OGP）在二元感知机中得到形式化，该性质限制了在临界约束密度 α_OGP 之上算法对零训练误差构型的可达性。本文将这一描述推广到有限温度情形，此时允许存在正的训练误差并对其进行统计惩罚。我们首先证明，在零温平衡测度中占主导地位的冻结一步复制对称性破缺解，在任意有限温度下依然存续。此外，我们推导出一个一般性判据，该判据基于单模式吉布斯权重在决策边界附近的光滑性……

    arXiv:2607.27000v2 Announce Type: replace-cross  Abstract: Optimization in non-convex neural network models is strongly influenced by the geometry of the solution space: sparse, isolated, point-like clusters are typically algorithmically inaccessible, whereas wide and flat regions can be found efficiently despite being relatively rare. At zero temperature this picture has been formalized in binary perceptrons through the overlap gap property (OGP), which limits algorithmic access to configurations with zero training error above a critical constraint density $\alpha_{\rm OGP}$. Here we extend this description to finite temperature, where a positive training error is allowed and statistically penalized. We first show that the frozen one-step replica-symmetry-breaking solution, dominating the zero temperature equilibrium measure, survives at any finite temperature. We furthermore derive a general criterion, based on the smoothness of the single-pattern Gibbs weight near the decision bound
    
[^254]: 思考精简，智能委托，行动，重复：边缘LLM代理的校准推理与不确定性感知委托

    Think Short, Defer Smart, Act, and Repeat: Calibrated Reasoning and Uncertainty-Aware Deferral for Edge LLM Agents

    [https://arxiv.org/abs/2607.26865](https://arxiv.org/abs/2607.26865)

    TSDS框架通过轻量级收敛探针和基于困惑度的委托规则，在边缘LLM代理中实现推理预算与可靠性的平衡，并利用多目标LTT程序提供同时的有限样本保证。

    

    arXiv:2607.26865v2 公告类型：替换-交叉 摘要：遵循ReAct范式的LLM代理是实现复杂多步任务（包括多跳问答、代码生成和物理AI系统控制）的有前景的使能器。然而，当部署在边缘时，它们必须严格管理推理预算，同时保持可靠性，并且仅在本地不确定性过高而无法安全行动时，才委托给云端模型。我们提出“思考精简，智能委托”（TSDS）框架，该框架协同整合了一个轻量级收敛探针（一旦预期行动稳定即停止设备端推理）与一个基于困惑度的委托规则（将不确定行动升级到云端模型）。两种机制通过多目标“学习-然后-测试”（LTT）程序在端到端情节轨迹上联合校准，同时提供关于预期情节奖励和云端调用率的有限样本保证。我们在四个ReAct基准上评估TSDS，涵盖...

    arXiv:2607.26865v2 Announce Type: replace-cross  Abstract: LLM agents following the ReAct paradigm are promising enablers of complex multi-step tasks, including multi-hop question answering, code generation, and control of physical AI systems. Yet, when deployed at the edge, they must tightly manage their reasoning budget while remaining reliable and deferring to a cloud-side model only when local uncertainty is too high to act safely. We propose Think Short, Defer Smart (TSDS), a framework that synergistically integrates a lightweight convergence probe, which halts on-device reasoning once the intended action has stabilized, with a perplexity-based deferral rule that escalates uncertain actions to a cloud-side model. Both mechanisms are jointly calibrated on end-to-end episode trajectories via a multi-objective Learn-Then-Test (LTT) procedure, providing simultaneous finite-sample guarantees on expected episode reward and cloud-call rate. We evaluate TSDS on four ReAct benchmarks spann
    
[^255]: 盲而非弱：针对编码式VLM越狱攻击的“恢复-重防”防御的套件级最优安全-效用前沿

    Blind, Not Weak: A Best-of-Suite Safety-Utility Frontier for Recover-and-Reguard Defenses Against Encoded VLM Jailbreaks

    [https://arxiv.org/abs/2607.26574](https://arxiv.org/abs/2607.26574)

    该论文构建了一个“恢复-重防”预处理器，在安全防护器之前恢复图像内容并解码编码，将图像渲染类越狱攻击的拦截率从零提升至67-90%，并据此刻画了此类防御在安全性与良性流量效用之间的套件级最优权衡边界。

    

    安全分类器（“防护器”）是视觉语言模型（VLM）的主流黑盒防御手段，然而防护器评判的是输入的表层形式而非其含义：一个有害请求若被重新编码为集合论、形式逻辑、古典语言、代码，或以渲染在图像中的文本形式呈现，便能绕过原本会拦截其明文形式的防护器——这便是“解码鸿沟”。标准的补救方案是在防护器之前部署一个预处理器，用以恢复图像内容并解码编码。我们构建了这样一个预处理器，并在一个由十一种编码攻击组成的集成攻击集上对其进行评估——其中包含六个已发表的攻击实现、一个标准编码基线、一个改编版本以及三个作者自建的渲染攻击——只要任一攻击得手即判定该行为被攻破。正是恢复出防护器从未见过的视图才换来了覆盖率的提升——图像渲染攻击的拦截率从恰好为零提升至67-90%——而其在良性流量上的代价由防护器而非机制本身决定：某个防护器需付出9个百分点的良性拦截代价。

    arXiv:2607.26574v3 Announce Type: replace-cross  Abstract: Safety classifiers ("guards") are the dominant black-box defense for vision-language models, yet a guard judges an input's surface form, not its meaning: a harmful request re-encoded as set theory, formal logic, a classical language, code, or text rendered inside an image slips past a guard that would block it in plain language - the decode gap. The standard fix is a preprocessor that recovers image content and decodes the encoding before the guard. We build one and evaluate it against an ensemble of eleven encoding attacks - six published implementations, one standard encoding baseline, one adapted and three author-constructed renders - counting a behavior as broken if any attack succeeds. Restoring a view the guard never had is what buys coverage - block rates on image renders go from exactly zero to 67-90% - and what it costs in benign traffic is set by the guard, not by the mechanism: one guard pays 9 benign blocking points
    
[^256]: CT-Merging：面向 LoRA 适配器合并的共识方向与任务特定缩放

    CT-Merging: Consensus Directions and Task-Specific Scaling for LoRA Adapter Merging

    [https://arxiv.org/abs/2607.20561](https://arxiv.org/abs/2607.20561)

    提出 CT-Merging 方法，通过平均任务子空间投影器估计共识方向，并为每个任务分配独立的残差缩放因子，从而在 LoRA 适配器合并任务上取得优于现有基线的平均与最差任务准确率。

    

    arXiv:2607.20561v2 公告类型：替换。摘要：LoRA 合并方法越来越多地在任务更新的低秩结构上进行操作，然而公共子空间如何被估计以及在重组后系数如何被分配，却很少被直接比较。我们提出了 CT-Merging，该方法从平均后的任务子空间投影器中估计公共方向，并为每个任务分配一个独立的残差缩放因子。投影器平均在任务子空间中选择被广泛支持的方向，而不按奇异值大小进行加权；同时，任务特定缩放消除了分量级别的幅度变化，并保留了任务之间的尺度差异。在公开发布的 KnOTS CLIP 适配器上，CT-Merging 在两个骨干网络上均取得了最佳的平均归一化准确率和最差任务归一化准确率，相比最强基线分别最多提升 2.56 和 6.65 个百分点。在 DC-Merge 适配器基准测试中，它在九个骨干网络与任务数量设置中的八个里取得了最佳的平均归一化准确率。消融实验表明……

    arXiv:2607.20561v2 Announce Type: replace  Abstract: LoRA merging methods increasingly operate on the low-rank structure of task updates, yet how the common subspace is estimated and how coefficients are assigned after recomposition are rarely compared directly. We propose CT-Merging, which estimates common directions from averaged task subspace projectors and assigns a separate residual scale to each task. Projector averaging selects directions supported across task subspaces without weighting them by singular magnitude, while task-specific scaling removes component-wise magnitude variation and preserves scale differences across tasks. On the released KnOTS CLIP adapters, CT-Merging achieves the best average and worst-task normalized accuracy on both backbones, improving over the strongest baseline by up to 2.56 and 6.65 points, respectively. On the DC-Merge adapter benchmark, it achieves the best average normalized accuracy in eight of nine backbone and task-count settings. Ablations
    
[^257]: DreamSat-Pose：基于单视图三维重建与学习的二维-三维特征匹配的航天器位姿估计

    DreamSat-Pose: Spacecraft Pose Estimation from Single-View 3D Reconstructions and Learned 2D-3D Feature Matching

    [https://arxiv.org/abs/2607.13449](https://arxiv.org/abs/2607.13449)

    本文提出 DreamSat-Pose 框架，仅凭单张图像即可对未知航天器同时完成三维形状重建与六自由度位姿估计，其核心创新在于结合冻结的 DINOv3 图像特征、动态图卷积点云几何特征与双流 Transformer 匹配器，学习密集二维-三维对应关系并由 PnP 求解器恢复位姿。

    

    六自由度（6-DoF）位姿估计是自主交会与邻近操作中的一项关键任务。在目标未知的情况下，该任务变得极具挑战性，因为它必须与目标形状模型的重建相结合。本文提出了一种针对未知航天器目标的单次形状与位姿估计新框架。给定单张图像，我们首先重建目标的三维形状模型，然后通过学习密集的二维-三维对应关系来估计相对六自由度位姿。图像特征使用冻结的 DINOv3 视觉 Transformer 提取，而几何特征则使用可训练的动态图卷积神经网络编码器从重建的点云中计算得到。双流 Transformer 匹配器通过交替的自注意力和交叉注意力对描述子进行精炼，生成软对应关系，并将其传递给透视n点（PnP）求解器以恢复位姿。

    arXiv:2607.13449v2 Announce Type: replace-cross  Abstract: 6-DoF pose estimation is a critical task in autonomous rendezvous and proximity operations. In the case of an unknown target, this task becomes challenging as it shall be paired with the reconstruction of the target shape model. In this article, we propose a novel framework for single-shot shape and pose estimation of unknown spacecraft objects. Given a single image, we first reconstruct a 3D shape model of the target, then estimate the relative six-degrees-of-freedom pose by learning dense 2D-3D correspondences. The image features are extracted using a frozen DINOv3 vision transformer, while the geometric features are computed from the reconstructed point cloud using a trainable dynamic graph convolutional neural network encoder. A dual-stream transformer matcher refines descriptors through alternating self- and cross-attention, producing soft correspondences that are passed to a Perspective-$n$-Point solver for pose recovery.
    
[^258]: 高维M估计中的影响诊断：精确渐近性

    Influence Diagnostics in High-dimensional M-estimation: Precise Asymptotics

    [https://arxiv.org/abs/2607.09250](https://arxiv.org/abs/2607.09250)

    该论文在高维凸M估计中精确刻画了训练点留一影响的渐近分布，发现有影响力的样本平均而言倾向于靠近决策边界，与主动学习中的数据选择启发式方法相契合。

    

    某个给定训练点对统计模型的影响可以通过其对模型参数的留一影响来衡量，该度量量化了将此训练点从训练集中移除对学习到的权重所产生的影响。对于高斯设计下的凸M估计，在高维极限 n ≍ d 情形下，我们证明了训练点间影响的经验分布集中于一个确定性测度附近，并对该测度给出了精确刻画。这一刻画表明，有影响的样本平均而言往往位于接近决策边界的位置，这与主动学习中的标准数据选择启发式方法相呼应。

    arXiv:2607.09250v2 Announce Type: replace  Abstract: The impact of a given training point on a statistical model can be measured through its leave-one-out influence on the model parameters, which quantifies how its removal from the training set affects the learned weights. For convex M-estimation under Gaussian design, in the high-dimensional limit $n\asymp d$, we show that the empirical distribution of influences across training points concentrates around a deterministic measure which we sharply characterize. This characterization suggests that influential samples tend to lie on average close to the decision boundary, making contact with a standard data selection heuristic in active learning.
    
[^259]: 将智能体搜索引入地球观测数据发现

    Bringing Agentic Search to Earth Observation Data Discovery

    [https://arxiv.org/abs/2607.02387](https://arxiv.org/abs/2607.02387)

    该论文提出了一个基于NASA地球观测知识图谱的智能体搜索框架用于地球科学数据发现，构建了包含47k查询-数据集对的开放基准NASA-EO-Bench，并通过微调神经评分器与BM25分数融合，将R@10和MRR提升至余弦基线的5倍以上。

    

    NASA及其数据中心拥有数千个地球科学数据集，以及Worldview、Giovanni、科学发现引擎和Harmony等工具。即使是领域专家，要找到合适的数据或工具也十分困难。我们提出了一个面向地球科学数据发现的智能体搜索框架，该框架接收自然语言研究查询并返回匹配的数据集和工具。我们证明，在大语言模型时代，知识图谱（KG）的潜在价值可以通过智能体搜索得到显著放大。我们从NASA地球观测知识图谱（NASA EO-KG）中构建了NASA-EO-Bench，这是一个包含47k查询-数据集对（其中21k为基于任务的查询）的开放基准。在NASA-EO-Bench上微调的神经评分器优于余弦相似度和BM25基线。进一步通过分数融合将其与BM25结合，使Recall@10（R@10）和MRR均达到未适配余弦基线的5倍以上。在此监督流水线之上，零样本重排序阶段进一步提升了MRR……

    arXiv:2607.02387v2 Announce Type: replace  Abstract: NASA and its data centers hold thousands of geoscience datasets and tools like Worldview, Giovanni, the Science Discovery Engine, and Harmony. Finding the right one is hard even for domain experts. We present an agentic search framework for geoscience data discovery that takes a natural-language research query and returns matching datasets and tools. We demonstrate that, in the era of large language models, the latent value of knowledge graphs (KGs) can be substantially amplified through agentic search. From the NASA Earth Observation Knowledge Graph (NASA EO-KG) we derive NASA-EO-Bench, an open benchmark of 47k query-dataset pairs (21k task-based queries). A neural scorer fine-tuned on NASA-EO-Bench beats cosine and BM25 baselines. Further combining it with BM25 via score fusion raises both Recall@10 (R@10) and MRR to over 5x the unadapted cosine baseline. On top of this supervised pipeline, a zero-shot reranking stage lifts MRR by 
    
[^260]: 超越药物发现：纳米技术分子优化（NMO）基准

    Beyond Drug Discovery: The Nanotechnology Molecular Optimization (NMO) Benchmark

    [https://arxiv.org/abs/2606.30170](https://arxiv.org/abs/2606.30170)

    提出纳米技术分子优化基准，用量子模拟取代代理预测器并引入严格协议，将生成式分子设计从药物发现领域拓展至量子材料科学与纳米技术研究。

    

    生成式分子设计目前主要受针对药物类特性的简单代理基准以及在大型制药数据集上预训练的模型所塑造。这种组合虽然能产生亮眼的基准测试指标，却限制了其向与药物发现在结构上截然不同的领域迁移的能力。为了克服这一局限性，并推动分子发现面向真实的、具有科学依据的目标，我们提出了纳米技术分子优化基准，它连接了机器学习（ML）与量子材料科学。NMO 既可以作为机器学习社区的严格测试平台，也可以作为纳米技术研究的发现引擎。该基准套件用量子模拟取代了代理预测器，并引入了严格的评估协议，优先考虑科学实用性而非面向排行榜的过度拟合。基于物理的 NMO 任务施加了严格的结构约束和崎岖的适应度景观，对生成式模型提出了根本性的全新要求。

    arXiv:2606.30170v2 Announce Type: replace-cross  Abstract: Generative molecular design is shaped by simple proxy benchmarks for drug-like properties and models pretrained on large pharmaceutical datasets. This combination yields strong benchmark metrics but limits transferability to domains structurally distinct from drug discovery. To overcome this limitation and drive discovery toward real, scientifically grounded targets, we introduce the Nanotechnology Molecular Optimization (NMO) Benchmark, which bridges machine learning (ML) and quantum materials science. NMO acts simultaneously as a rigorous testbed for the ML community and a discovery engine for nanotechnology research. The suite replaces proxy oracles with quantum simulations and introduces strict protocols that prioritize scientific utility over leaderboard-oriented overfitting. The physics-based NMO tasks impose hard structural constraints and rugged fitness landscapes, posing fundamentally new requirements on generative mod
    
[^261]: TeDiServe：面向扩散语言模型的高SLO达成率服务系统

    TeDiServe: High SLO Attainment Serving for Diffusion Language Models

    [https://arxiv.org/abs/2606.29094](https://arxiv.org/abs/2606.29094)

    TeDiServe是一个面向扩散语言模型的集群级服务系统，通过截止时间感知调度、基于置信度阈值调整的自适应负载控制以及动态重新配置，在满足延迟SLO的同时实现高吞吐量服务。

    

    arXiv:2606.29094v2 公告类型：替换 摘要：扩散语言模型（DLM）最近成为传统自回归语言模型的一种有前景的替代方案。通过在每个去噪步骤中并行生成多个token，它们在保持有竞争力的生成质量的同时提供了更高的推理吞吐量。然而，要在服务系统中实现这些吞吐量收益并满足延迟SLO，需要应对DLM独特特性所带来的挑战。这些挑战包括：处理基于置信度的去噪所造成的速度-质量权衡、在负载波动情况下为模型实例选择合适的并行化级别，以及协调引入非均匀每步成本的近似KV缓存机制。为了解决这些挑战，我们提出了TeDiServe，一个面向DLM的集群级服务系统。TeDiServe通过置信度阈值调整实现了截止时间感知的调度和自适应负载控制，并能够动态重新配置（摘要在此处截断）

    arXiv:2606.29094v2 Announce Type: replace  Abstract: Diffusion language models (DLMs) have recently emerged as a promising alternative to conventional autoregressive language models. By generating multiple tokens in parallel during each denoising step, they offer higher inference throughput while maintaining competitive quality. However, realizing these throughput gains while meeting latency SLOs in a serving system requires addressing challenges introduced by DLMs' unique characteristics. These include navigating the speed-quality tradeoff created by confidence-based denoising, choosing appropriate parallelization levels across model instances under fluctuating load, and coordinating approximate KV caching mechanisms that introduce non-uniform per-step costs. To address these challenges, we present TeDiServe, a cluster-level serving system for DLMs. TeDiServe enables deadline-aware scheduling and adaptive load control through confidence-threshold adjustment, and dynamically reconfigur
    
[^262]: 统计有效的训练后超参数选择：从调优到保证

    Statistically Valid Post-Training Hyperparameter Selection: From Tuning to Guarantees

    [https://arxiv.org/abs/2606.25601](https://arxiv.org/abs/2606.25601)

    提出以“先学习后测试”（LTT）范式为核心的统一统计框架，将训练后超参数选择转化为多元假设检验问题，为人工智能系统部署中的超参数调优提供正式的可靠性统计保证。

    

    训练后超参数选择是现代人工智能系统部署中的一个关键步骤，因为需要调整预训练模型的自由度，例如推理时参数、实现层面的设置以及驱动决策规则的阈值。尽管具有实际重要性，超参数选择通常采用尽力而为的经验方法（如网格搜索或贝叶斯优化）来执行，而这些方法在可靠性或安全性方面不提供正式的统计保证。本专著面向信号处理和机器学习研究人员，提出了一个以“先学习后测试”范式为核心的统一统计框架，用于实现可靠的训练后超参数选择。LTT将超参数选择问题表述为对候选超参数集合的多元假设检验。该框架能够实现超参数的选择……

    arXiv:2606.25601v2 Announce Type: replace-cross  Abstract: Post-training hyperparameter selection is a critical step in the deployment of modern artificial intelligence systems, given the need to tune degrees of freedom of pre-trained models such as inference-time parameters, implementation-level settings, and thresholds driving decision rules. Despite its practical importance, hyperparameter selection is typically performed using best-effort empirical methods such as grid search or Bayesian optimization, which provide no formal statistical guarantees on reliability or safety. This monograph, intended for an audience of signal processing and machine learning researchers, presents a unified statistical framework for reliable post-training hyperparameter selection, centered on the learn-then-test (LTT) paradigm. LTT formulates the hyperparameter selection problem as multiple hypothesis testing over a candidate set of hyperparameters. The framework enables the choice of hyperparameters th
    
[^263]: ConSolv：溶剂条件化机器学习隐式溶剂势

    ConSolv: Solvent-Conditional Machine Learning Implicit Solvent Potential

    [https://arxiv.org/abs/2606.24983](https://arxiv.org/abs/2606.24983)

    ConSolv提出了一种溶剂条件化的机器学习隐式溶剂势，通过基于注意力的溶剂嵌入模块显式纳入溶剂效应，实现了在66种常见有机溶剂上的迁移与对未见溶剂的泛化，并在溶剂化自由能基准上超越了经典显式溶剂方法和从头算隐式溶剂方法。

    

    隐式溶剂机器学习势（MLP）为弥合分子模拟中精度与效率之间的差距提供了一条强大的途径。然而，现有模型主要集中于水相环境，忽视了非水溶剂在有机合成和电池技术等领域中多样而重要的作用。在本工作中，我们提出了ConSolv，这是一种溶剂条件化的机器学习势架构，通过基于注意力的溶剂嵌入模块，将溶剂效应显式地纳入溶质相互作用之中。通过结合实验溶剂化自由能数据与从头算数据，我们训练出了一个可在66种常见有机溶剂间迁移的单一隐式溶剂机器学习势。在多个溶剂化自由能基准测试中，ConSolv的表现优于经典的显式溶剂方法和选定的从头算隐式溶剂方法，并展现出对未见溶剂的泛化能力。除溶剂化自由能之外……

    arXiv:2606.24983v2 Announce Type: replace-cross  Abstract: Implicit solvent machine learning potentials (MLPs) offer a powerful route to bridging the gap between accuracy and efficiency in molecular simulations. However, existing models have largely focused on aqueous environments, overlooking the diverse and important roles of non-aqueous solvents in areas such as organic synthesis and battery technology. Here, we present ConSolv, a solvent-conditional MLP architecture that explicitly incorporates solvent effects on solute interactions through an attention-based solvent-embedding block. By combining experimental solvation free energy data with ab initio data, we train a single implicit solvent MLP that is transferable across 66 common organic solvents. ConSolv outperforms classical explicit solvent methods and selected ab initio implicit solvent approaches across multiple solvation free energy benchmarks, and demonstrates generalization to unseen solvents. Beyond solvation free energi
    
[^264]: 位置编码器能否捕捉空间效应？跨尺度的GeoShapley基准测试

    Do Location Encoders Capture Spatial Effects? A GeoShapley Benchmark Across Scales

    [https://arxiv.org/abs/2606.23453](https://arxiv.org/abs/2606.23453)

    该论文提出以GeoShapley博弈论解释器为工具，在三个空间尺度上对TorchSpatial框架中的十一种位置编码器进行基准测试，系统评估其嵌入能否恢复已知的空间变化系数，并揭示恢复效果随尺度与编码器架构的变化规律。

    

    位置编码器将地理坐标转换为高维嵌入，供下游机器学习使用，但这些表示能在多大程度上捕捉可解释的空间效应尚不清楚。我们进行基准测试，评估GeoShapley——一种将所有位置特征视为单一联合参与者的博弈论解释器——能否从基于位置编码器嵌入构建的模型中恢复空间变化的系数。我们使用TorchSpatial框架中的十一种编码器，在一个系数已知的合成过程中进行评估，涵盖三种尺度（网格、县级、全球），分别测试嵌入伴随与不伴随原始坐标的情况，以及未训练与对比训练两种条件。我们以估计系数与真实系数之间的相关性来衡量恢复效果，报告其如何随尺度和编码器架构而变化，并将嵌入与原始坐标基线进行比较。主要系数的恢复始终……（原文摘要在此处截断）

    arXiv:2606.23453v2 Announce Type: replace  Abstract: Location encoders transform geographic coordinates into high dimensional embeddings for downstream machine learning, but it is unclear how well these representations capture interpretable spatial effects. We benchmark whether GeoShapley, a game-theoretic explainer that treats all location features as a single joint player, can recover spatially varying coefficients from models built on location-encoder embeddings. Eleven encoders from the TorchSpatial framework are evaluated against a synthetic process with known coefficients, across three scales (grid, county, global), with and without raw coordinates alongside the embedding, and under untrained and contrastively trained conditions. Measuring recovery as the correlation between estimated and true coefficients, we report how it varies with scale and encoder architecture and compare the embeddings against a raw-coordinate baseline. Recovery of the primary coefficient is consistently h
    
[^265]: NAC：面向视觉-语言-动作模型的神经动作编解码器

    NAC: Neural Action Codec for Vision-Language-Action Models

    [https://arxiv.org/abs/2606.21372](https://arxiv.org/abs/2606.21372)

    该论文提出神经动作编解码器（NAC），借鉴神经音频编解码器的设计思想，将机器人动作轨迹视为多通道一维信号并用多尺度RVQGAN架构进行高保真压缩，为视觉-语言-动作模型提供紧凑且有序的离散动作词元空间。

    

    视觉-语言-动作（VLA）模型依赖离散动作分词器来连接连续机器人控制与自回归序列建模，然而现有的分词器往往需要在压缩率、延迟和下游性能之间进行权衡。我们通过神经音频编解码器的视角重新审视这一设计——这是一类采用残差向量量化的卷积编码器-解码器架构，如今已成为音频基础模型的标准前端。受其成功的启发，我们提出了神经动作编解码器（NAC），它将短时机器人动作轨迹视为多通道一维信号，并使用多尺度RVQGAN架构对其进行压缩。通过对动作表示、压缩率和重建目标进行适配，音频编解码器风格的模型无需大幅修改架构即可高保真地对动作进行自编码。NAC通过偏移码本提供了紧凑且有序的词元空间，使标准的……

    arXiv:2606.21372v2 Announce Type: replace-cross  Abstract: Vision-language-action (VLA) models rely on discrete action tokenizers to bridge continuous robot control and autoregressive sequence modeling, yet existing tokenizers often trade off between compression, latency, and downstream performance. We revisit this design through the lens of neural audio codecs - convolutional encoder-decoder architectures with residual vector quantization that serve as the standard front end for audio foundation models. Motivated by their success, we introduce the Neural Action Codec (NAC), which treats short robot action trajectories as multi-channel 1D signals and compresses them using a multi-scale RVQGAN architecture. With adaptations to the action representation, compression rate, and reconstruction objective, audio-codec-style models can autoencode actions with high fidelity without substantial architectural changes. NAC provides a compact, ordered token space via offset codebooks, enabling stan
    
[^266]: ThousandWorlds：潜在宜居系外行星气候模拟的基准测试

    ThousandWorlds: A benchmark for climate emulation of potentially habitable exoplanets

    [https://arxiv.org/abs/2606.18338](https://arxiv.org/abs/2606.18338)

    ThousandWorlds是一个机器学习就绪的系外行星气候模拟基准数据集，包含来自五个全球气候模型的约1800次模拟，旨在突破传统气候模拟的计算瓶颈，加速对潜在宜居系外行星大气的理解与生命信号解读。

    

    寻找地球以外的生命将依赖于探测潜在宜居系外行星大气中的微弱信号。解读这些信号需要理解宿主行星的气候：同一种分子在一颗行星上可能预示生命存在，而在另一颗行星上则可能是非生物化学过程的产物。全球气候模型（GCM）能够提供这种理解，但单次运行可能需要高达数百万核时以及大量领域专家时间。机器学习模拟器有望消除这一瓶颈，但相关进展一直受限于缺乏一个经过精心整理的多模型系外气候数据集。我们推出了ThousandWorlds，这是一个面向系外气候模拟以及更广泛的低数据、多模拟器、参数到场回归任务领域的机器学习就绪基准数据集。该数据集包含来自五个全球气候模型的约1800次模拟，将八个行星参数映射到三维大气场，包括温度、湿度、风、云和辐射。

    arXiv:2606.18338v2 Announce Type: replace  Abstract: The search for life beyond Earth will depend on detecting faint signatures in the atmospheres of potentially habitable exoplanets. Interpreting those signatures requires understanding the host planet's climate: the same molecule may signal life on one planet and abiotic chemistry on another. Global climate models (GCMs) provide this understanding, but individual runs can require up to millions of core-hours and substantial domain expert time. Machine-learning emulators could remove this bottleneck, but progress has been limited by the absence of a curated, multi-model exoclimate dataset. We introduce ThousandWorlds, an ML-ready benchmark for exoclimate emulation and for the broader regime of low-data, multi-simulator, parameter-to-field regression. The dataset contains approximately 1800 simulations from five GCMs, mapping eight planet parameters to 3D atmospheric fields including temperature, humidity, winds, clouds, and radiation. 
    
[^267]: 反问题中后验期望的摊销求积方法

    Amortized quadrature for posterior expectations in inverse problems

    [https://arxiv.org/abs/2606.15871](https://arxiv.org/abs/2606.15871)

    本文提出“求积场”——一种集合等变神经网络，只需在一个后验族上训练一次，即可对任意观测、任意样本数 M 和任意被积函数，通过一次前向传播生成带符号权重的 M 节点求积格式，在保证精度不劣于蒙特卡洛的同时，避免了传统设计求积法需对每个新观测重复求解优化问题的高昂计算成本。

    

    arXiv:2606.15871v2 公告类型：replace-cross 摘要：反问题的解以及在该解上执行的任务的不确定性由后验期望来量化，每个后验期望是某个被积函数在 M 个后验样本上的平均值。虽然精心设计的求积法可以改进蒙特卡洛估计 O(M^{-1/2}) 的误差，但它们需要针对每个新观测求解一个优化问题（通常以后验密度为目标），这在计算上代价高昂。为了解决这一局限，我们引入了“求积场”，这是一个集合等变网络，能够在一次前向传播中将一个观测及其 M 个后验样本映射为具有 M 个节点和带符号权重的求积格式。该网络只需在一个后验族上训练一次，以最小化某类函数上的最坏情况积分误差，此后即可服务于任何观测、任意 M 以及该类中的任何被积函数，无需进一步优化。我们证明，在高概率意义下并带有可计算的松弛量，所得求积的性能从不劣于蒙特卡洛（摘要在此处被截断）。

    arXiv:2606.15871v2 Announce Type: replace-cross  Abstract: Uncertainty in the solution of an inverse problem and in the tasks performed on it is quantified by posterior expectations, each an average of an integrand over $M$ posterior samples. While designed quadratures improve on the $O(M^{-1/2})$ error of Monte-Carlo estimation, they solve an optimization problem, often against the posterior density, for every new observation, which can be computationally costly. To address this limitation, we introduce the quadrature field, a set-equivariant network that maps an observation and its $M$ posterior samples to an $M$-node signed-weight quadrature in one forward pass. Trained once on a family of posteriors to minimize the worst-case integration error over a class of functions, it serves any observation, any $M$ and any integrand in that class with no further optimization. We show that, with high probability and up to a computable slack, the resulting quadrature is never worse than the Mon
    
[^268]: 面向嵌入模型路由的策略后悔：具有低秩专家的上下文赌博机

    Policy Regret for Embedding Model Routing: Contextual Bandits with Low-Rank Experts

    [https://arxiv.org/abs/2606.14929](https://arxiv.org/abs/2606.14929)

    该论文将嵌入模型路由形式化为具有低秩专家的对抗性上下文线性赌博机问题，证明标准后悔度量存在结构性误设或统计不可处理的缺陷，并提出兼具表达能力与高效可学习性的对数二次策略类来实现查询依赖的模型路由。

    

    现代推荐系统日益依赖将多样化的查询动态路由到多个嵌入模型。尽管这一问题具有重要的实践意义，但在对抗性查询、赌博机反馈以及模型可观测性受限等现实条件下，该问题仍未得到充分理解。我们将嵌入模型路由形式化为一个具有低秩专家的对抗性上下文线性赌博机问题，其中上下文对应查询，动作对应物品，专家则对应工作在低秩潜在表示空间上的嵌入模型。我们首先证明了标准的后悔度量会遭遇结构性误设或统计上的不可处理性，并识别出一个对数二次策略类，该策略类既足够富有表现力以刻画依赖于查询的模型路由，又具备足够规整的结构以支持高效的在线学习。聚焦于这一在赌博机反馈下的对数二次策略优化问题——该问题本身亦具有独立的研究价值……

    arXiv:2606.14929v2 Announce Type: replace-cross  Abstract: Modern recommendation systems increasingly rely on dynamically routing diverse queries to multiple embedding models. Despite its practical significance, this problem remains poorly understood under realistic conditions like adversarial queries, bandit feedback, and limited observability of models. We formalize embedding model routing as an adversarial contextual linear bandit with low-rank experts, where contexts are queries, actions are items, and experts are the embedding models working on low-rank latent representation spaces. We first establish that standard regret notions suffer from structural misspecification or statistical intractability, and we identify a log-quadratic policy class that is expressive enough to capture query-dependent model routing, yet structured enough to allow efficient online learning. Focusing on this log-quadratic policy optimization problem under bandit feedback -- which is of independent interes
    
[^269]: 基于优化引导算子的遗传算法

    Genetic Algorithms with Optimization Guided Operators

    [https://arxiv.org/abs/2606.12279](https://arxiv.org/abs/2606.12279)

    该论文提出了带优化引导算子的遗传算法通用模型，将其中优化问题形式化为基于强化学习语言的查询复杂度问题，并揭示了机器学习驱动的变异与重组算子虽更可能改进目标但计算成本更高的基本权衡。

    

    近期机器学习领域的工作在推理阶段应用遗传算法来迭代改进优化问题的解。其中涉及的基本变异和重组算子与经典研究中研究的算子在本质上截然不同。变异不再是随机的；机器学习算法以改进目标函数为目的对解进行变异。同样，重组也不再基于父代解的随机拼接，而是一种基于机器学习优化的算子，其目标是从输入中合成出改进的解。因此，这些变异和重组算子更有可能改进目标函数，但其计算成本要高得多。我们引入了一个遗传算法的通用模型，并使用强化学习的语言将该模型中的优化问题表述为一个查询复杂度问题。我们展示了三种基本现象。首先，我们证明了解池的多样性可……

    arXiv:2606.12279v2 Announce Type: replace-cross  Abstract: Recent work in ML applies genetic algorithms at inference time to iteratively improve solutions to optimization problems. The basic mutation and recombination operators involved are qualitatively different from those studied classically. Mutations are no longer random; an ML algorithm mutates a solution with the goal of improving an objective. Similarly, recombination is not based on random collages of parent solutions. Instead, it is an ML optimization-based operator whose goal is to synthesize improved solutions from its inputs. Thus, these mutation and recombination operators are more likely to improve the objective, but their computational cost is much higher.   We introduce a general model of genetic algorithms and formulate optimization in this model as a query complexity problem, using the language of reinforcement learning. We demonstrate three fundamental phenomena. First, we show that diversity of the solution pool ca
    
[^270]: 评论家架构至关重要：人形机器人运动-操作中双评论家与统一评论家的对比

    Critic Architecture Matters: Dual vs. Unified Critics for Humanoid Loco-Manipulation

    [https://arxiv.org/abs/2606.11891](https://arxiv.org/abs/2606.11891)

    人形机器人运动-操作多目标强化学习中，评论家架构的选择至关重要：双评论家在标准化评估中比统一评论家快 3.5 倍、吞吐量翻倍，但相当一部分差距源于评估中手指控制方式设置的不同。

    

    面向人形机器人的多目标强化学习必须在同一策略中协调运动与操作。一个自然的设计选择是：使用单一（统一）评论家来估计所有目标的组合价值，还是使用奖励信号互不重叠的分离（双）评论家。我们在 NVIDIA Isaac Lab 中于 Unitree G1 人形机器人上对这两种方案进行了比较。在标准化评估的站立模式下，双评论家运行比统一评论家运行快 3.5 倍到达目标（6.5 对 22.6 仿真步），吞吐量达到 2 倍（每 1,000 步 14.3 对 7.0 次经验证的触碰），验证触碰率也更高（65.2% 对 53.8%）。需要注意的是，该评估对所有策略都固定让手指张开，而统一评论家运行在训练时是自行驱动手指的。当统一评论家运行改为驱动自己的手指、其余条件不变时，站立模式下的速度差距从 3.5 倍缩小到 1.3 倍，吞吐量差距从 2 倍缩小到 1.1 倍。这是一次单独的重新评估……（摘要截断）

    arXiv:2606.11891v3 Announce Type: replace-cross  Abstract: Multi-objective reinforcement learning for humanoid robots must coordinate locomotion and manipulation within one policy. A natural design choice is between a single (unified) critic that estimates the combined value of all objectives and separate (dual) critics with disjoint reward signals. We compare the two on the Unitree G1 humanoid in NVIDIA Isaac Lab. In the standing mode of a standardized evaluation, the dual-critic run reaches targets 3.5x faster (6.5 vs. 22.6 simulation steps), achieves 2x the throughput (14.3 vs. 7.0 validated reaches per 1,000 steps) and a higher validated reach rate (65.2% vs. 53.8%) than the unified-critic run. That evaluation pins the fingers open for every policy, whereas the unified run had trained driving its own. When the unified run drives its own fingers, with nothing else changed, the standing-mode gap falls from 3.5x to 1.3x in speed and from 2x to 1.1x in throughput. This is a single re-e
    
[^271]: Bergson：一个用于数据归因的开源库

    Bergson: An Open Source Library for Data Attribution

    [https://arxiv.org/abs/2606.11660](https://arxiv.org/abs/2606.11660)

    Bergson 是一个开源数据归因库，支持扩展至超大规模语言模型和预训练数据集，并首次开源实现了 MAGIC、SOURCE 和 TrackStar 三种前沿数据归因方法。

    

    数据归因是可解释性领域中一个前景广阔的方向，旨在通过训练数据对模型的影响来解释模型行为，其应用包括调试模型的不良行为以及训练数据集的筛选与整理。然而，大规模实施数据归因需要大量的工程投入，许多前沿技术缺乏开源工具和支持。Bergson 是一个开源库，旨在通过提供多种可扩展至超大规模语言模型和预训练数据集的技术，加速该领域的发展。该库原生支持磁盘级梯度存储和多节点分布式训练，并为研究人员提供了易用的实用工具。此外，我们首次开源实现了三种领先的数据归因方法：MAGIC、SOURCE 和 TrackStar。该库可在 https://github.com/EleutherAI/bergson 获取。

    arXiv:2606.11660v2 Announce Type: replace  Abstract: Data attribution is a promising field in interpretability that aims to explain model behavior through the influence of its training data, with applications including debugging undesirable model behavior and training dataset curation. However, significant engineering effort is required to perform it at scale, and many cutting edge techniques lack open-source tooling and support. Bergson is an open source library that aims to enable faster progress in the field by providing a host of techniques that scale to very large language models and pre-training datasets. The library natively supports on-disk gradient stores and multi-node distributed training, and provides quality of life tools for researchers. Finally, we introduce the first open-source implementations of three leading data attribution methods: MAGIC, SOURCE, and TrackStar. The library is available at https://github.com/EleutherAI/bergson .
    
[^272]: INFUSER：影响力引导的自我进化提升推理能力

    INFUSER: Influence-Guided Self-Evolution Improves Reasoning

    [https://arxiv.org/abs/2606.09052](https://arxiv.org/abs/2606.09052)

    INFUSER提出了一种影响力引导的自我进化框架，通过生成器与求解器的协同训练，利用优化器感知的影响力分数来改进问题生成，从而显著提升推理能力。

    

    自我进化为增强推理能力提供了一条可扩展的路径：预训练语言模型仅需极少的外部监督即可自我提升。然而，现有方法要么依赖大量精心策划或教师生成的训练数据，要么在生成器无监督运行时，仅通过难度启发式给予奖励，这未必能改进求解器。我们引入了INFUSER，一种迭代协同训练框架，包含两个共同演化的角色：一个生成器，从自动收集的非结构化文档池中起草问题和参考标准答案；以及一个求解器，通过在这些问题上训练来改进自身。求解器使用标准正确性奖励，依据生成器提供的答案进行训练，而生成器则通过一个优化器感知的影响力分数获得奖励，该分数衡量每个提议的问题是否真正能提升求解器在目标分布上的表现。由于这种连续且嘈杂的影响力分数难以直接处理，我们采用了相应策略进行优化。

    arXiv:2606.09052v4 Announce Type: replace-cross  Abstract: Self-evolution offers a scalable path to stronger reasoning: a pretrained language model improves itself with only minimal external supervision. Yet existing methods either depend on extensively curated or teacher-generated training data, or, when the generator runs unsupervised, reward it by a difficulty heuristic that need not improve the solver. We introduce INFUSER, an iterative co-training framework with two co-evolving roles: a Generator that drafts questions and reference golden answers from a pool of unstructured, automatically collected documents, and a Solver that improves by training on them. The solver is trained with standard correctness rewards against the generator-provided answers, while the generator is rewarded by an optimizer-aware influence score that measures whether each proposed question would actually improve the solver on the target distribution. Because this continuous, noisy influence score is poorly 
    
[^273]: QueryGraph：通过基于大语言模型的图生成实现可靠的多工具查询执行规划

    QueryGraph: Reliable Multi-Tool Query Execution Planning via LLM-Based Graph Generation

    [https://arxiv.org/abs/2606.08300](https://arxiv.org/abs/2606.08300)

    该论文提出QueryGraph系统，将自然语言查询转换为结构化图并通过确定性规划器结合深度优先搜索执行，实现了可靠的跨工具多步骤查询，即使使用小型或本地部署的LLM也能达到高准确率。

    

    许多针对个人数据的真实世界查询跨越多个应用程序，需要进行结构化规划，因为单个工具只能提供部分信息。虽然大语言模型（LLM）展现出强大的推理和工具使用能力，但可靠地执行多步骤、跨工具的查询仍然具有挑战性。我们引入了一个系统，该系统将自然语言查询转换为结构化图，并通过确定性规划器执行。我们的方法使用深度优先搜索来解析依赖关系并组合跨工具的结果，提高了可靠性，并使查询能力超越传统的基于关键字的搜索。我们证明了即使使用较小或本地部署的大语言模型也能实现高准确率。

    arXiv:2606.08300v2 Announce Type: replace  Abstract: Many real-world queries over personal data span multiple applications and require structured planning, as individual tools expose only partial information. While LLMs show strong reasoning and tool use, reliably executing multi-step, cross-tool queries remains challenging. We introduce a system that converts natural language queries into structured graphs and executes them via a deterministic planner. Our approach uses depth-first search to resolve dependencies and combine results across tools, improving reliability and enabling queries beyond traditional keyword-based search. We demonstrate high accuracy even with smaller or locally hosted LLMs.
    
[^274]: 右删失生存数据的适当评分规则

    Proper Scoring Rules for Right-Censored Survival Data

    [https://arxiv.org/abs/2606.06393](https://arxiv.org/abs/2606.06393)

    提出了一个右删失生存数据的适当评分框架，通过先将预测分布经删失机制映射、再在导出的观测数据分布上应用适当评分，统一了右删失似然与IPCW型准则，并给出了CRPS、pinball损失和Brier评分的右删失版本。

    

    适当评分规则为概率预测的训练和评估提供了严谨的理论基础。在生存分析中，此类预测描述的是直到某一事件发生所需时间的分布。然而，由于随访可能在事件发生之前结束，该事件时间往往只能被部分观测，从而产生右删失。我们基于一个简单的思想，提出了一个针对右删失生存结果的适当评分框架：首先，通过删失机制对预测分布进行映射，然后在由此导出的观测数据分布上应用底层的适当评分。这一做法在删失时间固定时给出局部化评分，在删失时间随机或仅被部分观测时给出边际化评分。由此得到的构造在一个统一的框架内恢复了人们熟悉的右删失似然和IPCW型准则，同时还给出了CRPS、pinball损失、Brier评分等指标的右删失版本。

    arXiv:2606.06393v2 Announce Type: replace  Abstract: Proper scoring rules provide a rigorous theoretical basis for the training and evaluation of probabilistic forecasts. In survival analysis, such forecasts describe the distribution of the time until an event occurs. However, this event time is often only partially observed because follow-up may end before the event occurs, resulting in right censoring. We propose a framework for proper scoring of right-censored survival outcomes based on a simple idea: first, map the predictive distribution through the censoring mechanism, then apply the underlying proper score on the induced observed-data law. This yields localized scores for fixed censoring times and marginalized scores when the censoring time is random or only partially observed. The resulting construction recovers familiar right-censored likelihood and IPCW-type criteria within a coherent framework, while also yielding right-censored versions of the CRPS, pinball loss, Brier scor
    
[^275]: QASM-Eval：一个用于训练和评估大语言模型处理超越量子电路的 OpenQASM-3 的数据集

    QASM-Eval: A Dataset to Train and Evaluate LLMs on OpenQASM-3 Beyond Quantum Circuits

    [https://arxiv.org/abs/2605.30358](https://arxiv.org/abs/2605.30358)

    QASM-Eval 是首个用于训练和评估大语言模型生成 OpenQASM 3 面向硬件特性（如中途测量与经典反馈、精确时序及脉冲级控制）代码的综合数据集。

    

    量子计算目前仍处于噪声中等规模量子（NISQ）时代，其性能受噪声制约。要克服这一局限，需要超越门序列的面向硬件的能力：用于量子纠错（QEC）的线路中途测量与经典反馈、用于动力学位解耦（DD）的精确时序控制，以及用于校准的脉冲级波形访问。OpenQASM 3 通过硬件级编程接口提供了这些能力。尽管大语言模型（LLM）在代码生成方面进展迅速，但针对这些高级特性的数据集仍然缺乏。我们提出了 QASM-Eval，这是首个用于在 OpenQASM 3 上训练和评估大语言模型的综合数据集，其目标是在给定的程序上下文中补全面向硬件的构造。QASM-Eval 包含一个由专家精心整理的 1,200 个任务的测试集和一个超过 4,000 个任务的训练集，涵盖经典逻辑、时序调度、脉冲控制……（原文摘要在此处截断）

    arXiv:2605.30358v2 Announce Type: replace  Abstract: Quantum computing remains in the Noisy Intermediate-Scale Quantum (NISQ) era, with performance constrained by noise. Addressing this limitation requires hardware-facing capabilities beyond gate sequences: mid-circuit measurement and classical feedback for quantum error correction (QEC), precise timing for dynamical decoupling (DD), and pulse-level waveform access for calibration. OpenQASM 3 exposes these capabilities through a hardware-level programming interface. Despite rapid progress in large language models (LLMs) for code generation, datasets targeting these advanced features remain lacking. We introduce QASM-Eval, the first comprehensive dataset to train and evaluate LLMs on OpenQASM 3, targeting completion of hardware-facing constructs within a supplied program context. QASM-Eval comprises an expert-curated test set of 1,200 tasks and a training set of over 4,000 tasks, covering classical logic, timing scheduling, pulse contro
    
[^276]: 基于有限标注的大语言模型选择

    Large Language Model Selection with Limited Annotations

    [https://arxiv.org/abs/2605.24981](https://arxiv.org/abs/2605.24981)

    SELECT-LLM是首个大语言模型主动选择框架，通过基于期望信息增益的查询选择规则，仅需少量最具信息量的标注查询即可从开放或黑盒候选模型中识别出给定任务的最佳LLM。

    

    为特定任务选择合适的大语言模型（LLM）需要比较众多强有力的候选模型，然而标准评估依赖于在固定评估集上进行代价高昂的标注。为解决这一挑战，我们开发了SELECT-LLM，这是首个用于大语言模型主动模型选择的框架。SELECT-LLM旨在找到一小组查询，其标注对于识别给定任务的最佳LLM最具信息量。为此，我们引入了一种基于期望信息增益的查询选择规则，该增益由候选模型输出之间的成对相似度计算得出。由于该规则仅使用生成的模型响应，SELECT-LLM可应用于各种候选模型，无需对其架构做出假设或访问模型权重。这使其既适用于开源权重模型，也适用于黑盒LLM。我们在23个数据集、156个被评估模型、多样化的任务类别以及多种文本评估指标上对SELECT-LLM进行了评估。

    arXiv:2605.24981v2 Announce Type: replace  Abstract: Choosing a Large Language Model (LLM) for a given task requires comparing many strong candidates, yet standard evaluation relies on costly annotations over fixed evaluation sets. To address this challenge, we develop SELECT-LLM, the first framework for active model selection of LLMs. SELECT-LLM aims to find a small set of queries whose annotations are most informative for identifying the best LLM for a given task. To this end, we introduce a query selection rule based on expected information gain, computed from pairwise similarities between candidate model outputs. Because this rule only uses generated model responses, SELECT-LLM can be applied across candidate models without assumptions about their architecture or access to model weights. This makes it suitable for both open-weight and black-box LLMs. We evaluate SELECT-LLM across 23 datasets, 156 evaluated models, diverse task families, and multiple text evaluation metrics. Across 
    
[^277]: 神经算子的平滑分段切割方法以处理不连续性与尖锐过渡

    Smooth Piecewise Cutting for Neural Operator to Handle Discontinuities and Sharp Transitions

    [https://arxiv.org/abs/2605.19823](https://arxiv.org/abs/2605.19823)

    提出 Cut-DeepONet 两阶段训练框架，通过将求解域切割为平滑子区域、并将不连续性表示为高维空间中的边界，使神经算子能够高效处理偏微分方程解中的不连续性与尖锐过渡。

    

    神经算子在学习偏微分方程（PDE）的解算子方面已取得出色的性能，但其固有的连续表示难以捕捉不连续性和尖锐过渡。现有方法通常在连续函数空间内近似此类特征，往往需要更大的模型容量和高分辨率数据。在本工作中，我们提出 Cut-DeepONet，这是一个两阶段训练框架，在显式建模不连续性的同时降低了学习复杂度。我们的方法通过一种提升策略重新表述该问题，将求解域划分为平滑的子区域，同时将不连续性表示为更高维空间中的边界。这种分离使算子学习任务与神经网络的归纳偏置相契合，并避免了直接近似不连续性。此外，一个额外的网络用于预测依赖于输入的不连续性位置，以……

    arXiv:2605.19823v2 Announce Type: replace-cross  Abstract: Neural operators have achieved strong performance in learning solution operators of partial differential equations (PDEs), but their inherently continuous representations struggle to capture discontinuities and sharp transitions. Existing approaches typically approximate such features within continuous function spaces, often requiring increased model capacity and high-resolution data. In this work, we propose Cut-DeepONet, a two-stage training framework that explicitly models discontinuities while reducing learning complexity. Our approach reformulates the problem via a lifting strategy, partitioning the domain into smooth subregions while representing discontinuities as boundaries in a higher-dimensional space. This separation aligns the operator learning task with the inductive bias of neural networks and avoids directly approximating discontinuities. An additional network predicts input-dependent discontinuity locations for 
    
[^278]: MSAlign：对齐分子与质谱表征以实现代谢物鉴定

    MSAlign: Aligning Molecule and Mass Spectra representations for Metabolite Identification

    [https://arxiv.org/abs/2605.19752](https://arxiv.org/abs/2605.19752)

    本文提出轻量级模型MSAlign，通过对齐两个冻结的基础模型（质谱模型DreaMS与分子模型MolDeBERTa）实现最先进的代谢物鉴定性能，并以极小的计算开销通过分数融合策略进一步提升了检索效果。

    

    从质谱数据中准确鉴定代谢物（即小分子）仍然是代谢组学领域的核心挑战，其在药物发现、环境分析和临床研究中具有广泛的应用。我们致力于解决分子检索任务，即在给定一组候选分子的情况下，从代谢物的MS/MS谱图中恢复其化学结构。我们做出了三项贡献。首先，我们提出了一个统一框架，涵盖了近期基于表征对齐和对比学习的方法。其次，我们提出了MSAlign，这是一个轻量级模型，通过对齐两个冻结的基础模型（用于质谱的DreaMS和用于分子的MolDeBERTa）实现了最先进的性能，并证明了分数融合策略能以极小的计算成本进一步提升性能。第三，我们研究了一个长期存在的评估问题：分子检索中的数据划分策略（摘要在此处截断）

    arXiv:2605.19752v2 Announce Type: replace  Abstract: Accurately identifying metabolites i.e. small molecules from mass spectrometry data remains a core challenge in metabolomics, with broad applications in drug discovery, environmental analysis, and clinical research. We address the Molecule Retrieval task, which consists in recovering the chemical structure of a metabolite from its MS/MS spectrum given a set of candidate molecules. We make three contributions. First, we propose a unified framework encompassing recent approaches based on representation alignment and contrastive learning. Second, we introduce MSAlign, a lightweight model that achieves state-of-the art performances by aligning two frozen foundation models (DreaMS for mass spectra and MolDeBERTa for molecules) and demonstrate that a score fusion strategy further improves the performance for a very small computational cost. Third, we investigate a long-standing evaluation problem: data splitting strategies in molecule retr
    
[^279]: HiLiftAeroML：一个面向增升构型飞机空气动力学的高保真计算流体力学数据集

    HiLiftAeroML: A High-Fidelity Computational Fluid Dynamics Dataset for High-Lift Aircraft Aerodynamics

    [https://arxiv.org/abs/2605.19565](https://arxiv.org/abs/2605.19565)

    HiLiftAeroML 是首个面向增升构型飞机空气动力学的开放高保真 CFD 数据集，包含基于 NASA 通用研究模型增升构型的 1,800 个 GPU 加速大涡模拟案例，与风洞实验数据吻合良好。

    

    据我们所知，HiLiftAeroML 是首个专门面向增升构型飞机空气动力学的开放高保真计算流体力学（CFD）数据集。它包含 1,800 个模拟案例，涵盖 NASA 通用研究模型（Common Research Model）增升构型的 180 个变体以及从 4° 到 22° 的十个迎角。每个案例均采用 GPU 加速的显式壁面模型大涡模拟方法，在 3 亿至 5 亿单元的自适应求解网格上生成，覆盖了附着流、分离流和失速后流动状态。与参考着陆构型的风洞测量结果的对比显示，积分载荷和截面压力方面具有良好的一致性，且网格自适应显著改进了阻力和俯仰力矩的预测。该数据集以 CC-BY-4.0 许可发布，内容包括几何外形、时间平均的表面和体积场数据、积分载荷、验证材料以及确定性的基准测试数据划分，并附带初步的 GeoTransolver 与 Tr

    arXiv:2605.19565v2 Announce Type: replace-cross  Abstract: HiLiftAeroML is, to our knowledge, the first open high-fidelity computational fluid dynamics dataset dedicated to high-lift aircraft aerodynamics. It contains 1,800 simulations spanning 180 variants of the NASA Common Research Model high-lift configuration and ten angles of attack from $4^\circ$ to $22^\circ$. Each case was generated with a GPU-accelerated explicit wall-modeled large-eddy simulation approach on solution-adapted grids of 300--500 million cells, covering attached, separated, and post-stall flow conditions. Comparisons with wind-tunnel measurements for reference landing configurations show good agreement in integrated loads and sectional pressures, with grid adaptation substantially improving drag and pitching-moment predictions. The CC-BY-4.0 release includes geometries, time-averaged surface and volume fields, integrated loads, validation material, and deterministic benchmark splits. Initial GeoTransolver and Tr
    
[^280]: 联邦鞅后验采样

    Federated Martingale Posterior Samping

    [https://arxiv.org/abs/2605.18554](https://arxiv.org/abs/2605.18554)

    该论文提出联邦鞅后验采样（FMP），通过客户端上传可训练数据嵌入、服务器集中运行预测采样器的一次性并行协议，摆脱了联邦贝叶斯方法对先验设定的依赖，在性能上与中心化方法高度一致并取得最低的期望校准误差。

    

    联邦贝叶斯神经网络需要对模型参数设定一个固定的先验，这是众所周知的难题，而先验设定的偏差会严重损害模型的准确性与校准性能。受预测模型快速发展的启发，鞅后验（也称为预测贝叶斯）用预测分布取代先验-似然对，并通过反复抽取预测样本和重新拟合模型来恢复参数的不确定性。本文提出了联邦鞅后验（FMP）采样，这是一种一次性高度并行的协议，其中每个客户端上传一小组可训练的数据嵌入，由服务器在中心端运行预测采样器。对采样误差的分析展示了数据集压缩率的影响，实验表明FMP与中心化方法的表现十分接近，并取得了最低的平均期望校准误差（ECE）。

    arXiv:2605.18554v2 Announce Type: replace-cross  Abstract: Federated Bayesian neural networks require fixing a prior on the model parameters, which is notoriously difficult, and misspecification of this prior can severely degrade accuracy and calibration. Motivated by the rapid progress of predictive models, the martingale posterior, also known as predictive Bayes, replaces the prior--likelihood pair with a predictive distribution and recovers parameter uncertainty by repeatedly drawing predictive samples and refitting the model. This letter proposes {federated martingale posterior} (FMP) sampling, a one-shot embarrassingly parallel protocol in which each client uploads a small set of trainable data embeddings and the server runs the predictive sampler centrally. Analysis of the sampling error demonstrates the impact of the dataset compression rate, while experiments show that FMP closely matches the centralized counterpart and achieves the lowest mean expected calibration error (ECE) 
    
[^281]: 注意力残差中的注意力汇聚与离群值

    Attention Sinks and Outliers in Attention Residuals

    [https://arxiv.org/abs/2605.17887](https://arxiv.org/abs/2605.17887)

    该论文提出OASIS方法，通过令牌级与深度级的显式空路由和空耦合机制稳定双归一化注意力残差架构，抑制注意力汇聚与激活离群值，并从理论与实验上解释和缓解了AttnResidual的低比特量化敏感性问题。

    

    我们提出OASIS，这是一种感知离群值与注意力汇聚（sink）的方法，通过显式空路由和令牌到深度的空耦合来稳定双归一化的注意力残差架构。AttnResidual引入了一个额外的深度方向归一化通道，提升了层间路由的灵活性，但也可能放大注意力汇聚、激活离群值以及低比特量化误差。OASIS建立在令牌级和深度级上基于Softmax1的显式空路由之上，并利用令牌级的空证据来降低表现出更强空行为的深度分支的权重。在理论上，我们刻画了双归一化条件下类汇聚式注意力集中的机制，为AttnResidual中观察到的低比特敏感性提供了洞见。在实验中，我们在三个语言模型骨干以及多个语言建模、推理和长上下文基准上将OASIS与五个基线方法进行比较，并观察到（原文摘要在此处截断）

    arXiv:2605.17887v2 Announce Type: replace-cross  Abstract: We propose OASIS, an outlier- and sink-aware method that stabilizes dual-normalized attention-residual architectures through explicit null routing and token-to-depth null coupling. AttnResidual introduces an additional depth-wise normalization channel that improves inter-layer routing flexibility but can also amplify attention sinks, activation outliers, and low-bit quantization error. OASIS builds on explicit Softmax1-based null routes at both the token and depth levels and uses token-level null evidence to downweight depth branches exhibiting stronger null behavior. Theoretically, we characterize a conditional mechanism for sink-like attention concentration under dual normalization, offering insight into the low-bit sensitivity observed in AttnResidual. Experimentally, we compare OASIS against five baselines on three language-model backbones and multiple language-modeling, reasoning, and long-context benchmarks and observe co
    
[^282]: 表格数据不平衡学习：综述、基准测试与实践指南

    Tabular Imbalanced Learning: A Survey, Benchmark, and Practical Guide

    [https://arxiv.org/abs/2605.14915](https://arxiv.org/abs/2605.14915)

    本文对表格数据不平衡学习进行了系统综述并构建统一分类体系，同时推出了大规模实证基准TILBench，在57个数据集上标准化评估了40多种方法，发现没有任何单一方法能在所有场景下始终占优。

    

    不平衡学习仍然是表格数据应用中的一项根本性挑战。尽管经过数十年的研究并提出了众多方法，但对于不同的不平衡处理策略在各种数据场景和计算约束下的表现，目前仍缺乏系统性的理解，这使得实际方法选择变得困难。在本工作中，我们对表格数据不平衡学习进行了系统性综述，并提出了表格数据不平衡学习基准测试集，这是一个用于评估现有方法的大规模实证基准。我们首先将不平衡学习方法归纳为一个统一的分类体系，然后在标准化评估协议下，对57个表格数据集上的40多种代表性方法进行基准测试，考察其整体预测性能、对数据集特征的敏感性以及计算可扩展性。我们的结果表明，没有任何单一方法能够在所有设置中始终保持优势。

    arXiv:2605.14915v2 Announce Type: replace  Abstract: Imbalanced learning remains a fundamental challenge in tabular data applications. Despite decades of research and numerous proposed methods, there is still limited systematic understanding of how different imbalance-handling strategies perform across diverse data regimes and computational constraints, making practical method selection difficult. In this work, we provide a systematic survey of tabular imbalanced learning and introduce Tabular Imbalanced Learning Benchmark (TILBench), a large-scale empirical benchmark for evaluating existing methods. We first organize imbalanced learning approaches into a unified taxonomy and then benchmark more than 40 representative methods across 57 tabular datasets under a standardized evaluation protocol, examining overall predictive performance, sensitivity to dataset characteristics, and computational scalability. Our results show that no single method consistently dominates across all settings.
    
[^283]: 针对具有离散和连续参数的模拟器的混合神经后验估计

    Mixed neural posterior estimation for simulators with discrete and continuous parameters

    [https://arxiv.org/abs/2605.13551](https://arxiv.org/abs/2605.13551)

    本文提出混合神经后验估计方法，通过将联合后验分解为离散分量（自回归分类器）和连续分量（生成模型）并在单一仿真目标下联合训练，将神经后验估计扩展到了同时包含离散和连续参数的混合参数空间，同时提供了评估混合后验校准的诊断工具。

    

    神经后验估计（NPE）能够为具有难以处理的似然函数的复杂模拟器实现快速参数推断。NPE训练一个推断网络来估计给定数据下参数的概率密度，通常假设参数是连续的。然而，许多科学模型涉及的是混合参数空间，即同时包含离散和连续维度。我们通过将NPE扩展到混合参数空间来解决这一局限性，设计了一个联合处理离散和连续参数的推断网络。该推断网络将联合后验分解为离散和连续两个组成部分，将针对离散参数的自回归分类器与针对连续参数的生成模型相结合，并在单一的基于仿真的目标下进行联合训练。此外，我们提出了一种诊断工具来评估混合后验近似的校准情况。

    arXiv:2605.13551v2 Announce Type: replace  Abstract: Neural Posterior Estimation (NPE) enables rapid parameter inference for complex simulators with intractable likelihoods. NPE trains an inference network to estimate a probability density over parameters given data, typically assumed to be \emph{continuous}. However, many scientific models involve parameter spaces that are \emph{mixed}, that is, they contain both discrete and continuous dimensions. We address this limitation by extending NPE to mixed parameter spaces through an inference network that jointly handles discrete and continuous parameters. The inference network factorizes the joint posterior into discrete and continuous components, combining an autoregressive classifier for the discrete parameters with a generative model for the continuous parameters, trained jointly under a single simulation-based objective. In addition, we propose a diagnostic tool to assess the calibration of the mixed posterior approximation. Across tr
    
[^284]: 用于可解释脑网络分析的有监督深度多模态矩阵分解

    Supervised Deep Multimodal Matrix Factorization for Interpretable Brain Network Analysis

    [https://arxiv.org/abs/2605.13312](https://arxiv.org/abs/2605.13312)

    提出有监督深度多模态矩阵分解（SD3MF），将对称非负矩阵三因子分解从无监督单图聚类推广为面向多模态脑图的有监督深度框架，通过层级化、部分式的可解释表示与数据驱动的模态融合，弥合了脑网络分析中预测精度与可解释性之间的权衡。

    

    多模态脑网络分析长期面临预测精度与可解释性之间难以兼顾的权衡。深度神经网络虽然能够取得很高的预测精度，但其行为如同黑箱，几乎无法揭示驱动其决策的脑模块；而矩阵分解方法虽然能够提供基于部分（parts-based）的可解释性，却在很大程度上仍停留在浅层、无监督且局限于单一视角的阶段，往往需要通过预定义或启发式的融合规则来整合各个模态。为了弥合这一差距，我们提出了一种将层级建模能力与结构化、可解释的表示以及数据驱动融合相结合的公式化方法——有监督深度多模态矩阵分解，这是一个面向整合式脑网络分析的可解释框架。SD3MF 将对称非负矩阵三因子分解从无监督的单图聚类推广到面向多模态图总体的有监督预测。SD3MF 学习……（摘要在此处被截断）

    arXiv:2605.13312v2 Announce Type: replace  Abstract: Multimodal brain network analysis faces a persistent trade-off between predictive accuracy and interpretability. Deep neural networks achieve high accuracy but behave as black boxes that reveal little about the brain modules driving their decisions, whereas matrix factorization methods provide parts-based interpretability yet remain largely shallow, unsupervised, and restricted to a single view, integrating modalities through predefined or heuristic fusion rules. To bridge this gap with a formulation that couples hierarchical modeling capacity with structured, interpretable representations and data-driven fusion, we present Supervised Deep Multimodal Matrix Factorization (SD3MF), an interpretable framework for integrative brain network analysis that generalizes Symmetric Nonnegative Matrix Tri-Factorization (SNMTF) from unsupervised single-graph clustering to supervised prediction over populations of multimodal graphs. SD3MF learns d
    
[^285]: 复数值相位一致Transformer

    Complex-Valued Phase-Coherent Transformer

    [https://arxiv.org/abs/2605.10123](https://arxiv.org/abs/2605.10123)

    提出相位一致Transformer（PCT），通过对L2归一化的复数查询-键相似度施加实值平滑门控，以无token竞争的注意力机制跨层保留相位信息，在中规模基准测试中一致优于标准softmax Transformer及其复数值对应模型。

    

    复数值Transformer在很大程度上从实数值架构中继承了softmax注意力机制。然而，按行归一化的token竞争并不一定与保相计算相一致。在本文中，我们提出了相位一致Transformer（Phase-Coherent Transformer, PCT），它将一个实值的、与元素无关的平滑门控应用于经L2归一化的复数查询-键相似度。PCT用无token竞争的注意力取代了token竞争，旨在跨层保留相位信息。在涵盖长程记忆、层次化长程推理、位置检索、基于相位的记忆与叠加以及图像分类的中等规模基准测试中，PCT在各任务类别上展现出强大的泛化能力。在参数公平比较下，PCT始终优于标准softmax Transformer及其直接的复数值对应版本。此外，即使在一些传统上被认为困难的任务上……（摘要原文在此处截断）

    arXiv:2605.10123v3 Announce Type: replace  Abstract: Complex-valued Transformers have largely inherited softmax attention from real-valued architectures. However, row-normalised token competition is not necessarily aligned with phase-preserving computation. In this paper, we introduce the Phase-Coherent Transformer (PCT), which applies a real-valued, element-independent, smooth gate to L2-normalised complex query-key similarities. PCT replaces token competition with token-non-competing attention and is designed to preserve phase information across layers.   Across mid-scale benchmarks spanning long-range memory, hierarchical long-range reasoning, positional retrieval, phase-based memory and superposition, and image classification, PCT shows strong generalisation across task categories. Under parameter-fair comparison, PCT consistently outperforms both the standard softmax Transformer and its direct complex-valued counterpart. Moreover, even on tasks traditionally considered difficult f
    
[^286]: Transformer 可通过预条件 Richardson 迭代实现上下文内高斯核回归

    Transformers Can Implement Preconditioned Richardson Iteration for In-Context Gaussian Kernel Regression

    [https://arxiv.org/abs/2605.08475](https://arxiv.org/abs/2605.08475)

    本文从理论和实验上证明，标准 softmax 注意力 Transformer 的前向传播可通过实现预条件 Richardson 迭代来近似高斯核岭回归预测器，其中注意力负责跨词元的核算子运算、MLP 负责词元内的标量算术，并以 O(log(1/ε)) 的深度达到 ε 精度。

    

    本文研究了基于高斯核的上下文内核岭回归（KRR），并从理论和实证两方面证明，标准的 softmax 注意力 Transformer 能够在其前向传播过程中近似 KRR 预测器。在有界数据假设下，我们构建了一个单头 Transformer，其前向传播可近似地在相应的核系统上实现预条件 Richardson 迭代。该构造仅需 O(log(1/ε)) 个模块和宽度为 O(√(N/ε)) 的 MLP，即可对长度为 N 的提示实现 ε 精度的预测。我们的构造揭示了 Transformer 架构内部的功能分解：softmax 注意力负责生成跨词元交互所需的行归一化高斯核算子，而 MLP 层则在局部近似更新所需的词元内标量运算。在实证方面，我们训练了 GPT-2 风格的 Transformer……（原文摘要在此处截断）

    arXiv:2605.08475v3 Announce Type: replace-cross  Abstract: In this paper, we study in-context kernel ridge regression (KRR) with Gaussian kernels and show, both theoretically and empirically, that a standard softmax-attention transformer can approximate the KRR predictor during its forward pass. Under bounded-data assumptions, we construct a single-head transformer whose forward pass approximately implements \textit{preconditioned Richardson iteration} on the associated kernel system. The construction uses $O(\log(1/\epsilon))$ blocks and MLP width $O(\sqrt{N/\epsilon})$ to achieve $\epsilon$-accurate prediction for prompts of length $N$. Our construction reveals a functional decomposition within the transformer architecture: softmax attention produces a row-normalized Gaussian-kernel operator needed for \emph{cross-token} interactions, while MLP layers act locally to approximate the \emph{intra-token} scalar arithmetic required by the update. Empirically, we train GPT-2-style transfor
    
[^287]: 基于气动压力测量的可解释结构损伤检测研究

    Towards Interpretable Damage Detection based on Aerodynamic Pressure Measurements

    [https://arxiv.org/abs/2605.08187](https://arxiv.org/abs/2605.08187)

    提出利用新型无侵入、低成本的Aerosense气动压力传感系统，结合卷积神经网络从风洞实验数据中检测风力涡轮机叶片结构损伤并评估其严重程度，实现可解释的结构健康监测。

    

    现代大型风力涡轮机叶片的柔性日益增强，这使得经济高效且可靠的结构监测方案成为迫切需求。为此，我们提出利用通过Aerosense获取的气动压力测量数据，Aerosense是一种新型、无侵入且经济的传感系统。在先前的工作中 [Franz et al., 2025]，我们研究了气动压力测量在弹性和气动载荷结构上进行结构损伤检测的潜力。我们在开放式风洞中，对安装在垂直振动悬臂梁上的NACA 633418翼型开展了实验研究。通过在梁支撑附近进行受控的锯切操作，逐步引入结构损伤，并在不同来流条件和结构状态下记录气动压力分布。基于该数据集，我们开发了一种卷积神经网络，用于检测结构损伤并对其严重程度进行分类。

    arXiv:2605.08187v2 Announce Type: replace-cross  Abstract: The increasing flexibility of modern large wind turbine blades necessitates cost-efficient and reliable structural monitoring solutions. For this purpose, we propose to use aerodynamic pressure measurements obtained via Aerosense, a novel, non-intrusive and economical sensing system. In former work [Franz et al., 2025], we investigated the potential of aerodynamic pressure measurements for structural damage detection on elastic and aerodynamically loaded structures. An experimental campaign was conducted on a NACA 633418 airfoil mounted on a vertically vibrating cantilever beam within an open wind tunnel. Structural damage was introduced progressively through controlled saw cuts near the beam support. Aerodynamic pressure distributions were recorded under varying inflow conditions and structural states. Based on this data set, we developed a convolutional neural network to detect structural damage and classify its severity usin
    
[^288]: 几何感知单纯形消息传递

    Geometry-Aware Simplicial Message Passing

    [https://arxiv.org/abs/2605.06061](https://arxiv.org/abs/2605.06061)

    提出几何单纯形Weisfeiler–Lehman（GSWL）测试，将顶点坐标引入颜色细化过程，证明了几何感知单纯形消息传递方案的表达能力上界及其判别能力的可匹配性，并结合欧拉示性数变换给出了几何表达能力的完整刻画与近似框架。

    

    Weisfeiler–Lehman（WL）测试及其单纯形扩展（SWL）刻画了消息传递网络的组合表达能力，但它们对几何不敏感，即具有相同连接性但不同嵌入的网格无法被区分。我们提出了几何单纯形Weisfeiler–Lehman（GSWL）测试，该方法将顶点坐标纳入几何单纯复形的颜色细化过程。此外，我们证明：(i) 几何感知单纯形消息传递方案的表达能力以GSWL为上界；(ii) 存在参数设置，使得在任意固定的有限几何单纯复形族上，这些方案的判别能力能够与GSWL相匹配。结合欧拉示性数变换（ECT）——几何单纯复形的完全不变量——这一结果给出了几何表达能力的刻画以及一个近似框架。实验在……

    arXiv:2605.06061v2 Announce Type: replace  Abstract: The Weisfeiler--Lehman (WL) test and its simplicial extension (SWL) characterize the combinatorial expressivity of message passing networks, but they are blind to geometry, i.e., meshes with identical connectivity but different embeddings are indistinguishable. We introduce the Geometric Simplicial Weisfeiler--Lehman (GSWL) test, which incorporates vertex coordinates into color refinement for geometric simplicial complexes. In addition, we show that (i) the expressivity of geometry-aware simplicial message passing schemes is bounded above by GSWL, and (ii) that there exist parameters such that the discriminating power of GSWL is matched by these schemes on any fixed finite family of geometric simplicial complexes. Combined with the Euler Characteristic Transform (ECT), a complete invariant for geometric simplicial complexes, this yields a geometric expressivity characterization together with an approximation framework. Experiments on
    
[^289]: QuadraSHAP：基于高斯-勒让德求积的乘积博弈中稳定且可扩展的Shapley值

    QuadraSHAP: Stable and Scalable Shapley Values for Product Games via Gauss-Legendre Quadrature

    [https://arxiv.org/abs/2605.05870](https://arxiv.org/abs/2605.05870)

    本文提出QuadraSHAP，证明乘积博弈中每个玩家的Shapley值可精确表示为一维积分，从而利用高斯-勒让德求积以仅需⌈d/2⌉个节点即可实现可证明精确、稳定且可扩展的高效计算。

    

    我们研究了乘积博弈中Shapley值的高效计算——乘积博弈是一类联盟价值可分解为各玩家项乘积的合作博弈。当价值函数从底层模型继承乘法结构时，例如具有乘积核的核方法和基于树的模型，这类博弈会出现在机器学习可解释性问题中。我们的关键结果是：乘积博弈中每个玩家的Shapley值都存在精确的一维积分表示，即对指数级数量特征联盟的加权和可坍缩为一个次数为(d-1)的多项式在[0,1]区间上的积分，其中d为特征总数。由此得到一种高斯-勒让德求积方案：当节点数满足m_q ≥ ⌈d/2⌉时，该方案可证明是精确的；否则可提供近似精确的估计，其误差可证明随m_q呈几何级数衰减。

    arXiv:2605.05870v3 Announce Type: replace  Abstract: We study the efficient computation of Shapley values for \emph{product games} -- cooperative games in which the coalition value factorizes as a product of per-player terms. Such games arise in machine learning explainability whenever the value function inherits a multiplicative structure from the underlying model, as in kernel methods with product kernels and tree-based models. Our key result is that the Shapley value of each player in a product game admits an exact one-dimensional integral representation: the weighted sum over exponentially many feature coalitions collapses to the integral of a degree-$(d-1)$ polynomial over $[0,1]$, where $d$ is the total number of features. This yields a Gauss--Legendre quadrature scheme that is \emph{provably exact} whenever the number of nodes satisfies $m_q \geq \lceil d/2 \rceil$, and otherwise provides a \emph{near-exact} approximation with error provably decaying geometrically in $m_q$. In p
    
[^290]: 梯度-动量耦合：一种参数空间的学习进度代理指标

    Gradient-Momentum Coupling: A Parameter-Space Proxy for Learning Progress

    [https://arxiv.org/abs/2605.05856](https://arxiv.org/abs/2605.05856)

    提出梯度-动量耦合（GMC），通过梯度与动量的归一化乘积在参数空间中衡量学习进度，相比基于预测误差的方法更能抵抗噪声干扰，并按改进速度而非难度对任务排序。

    

    衡量学习进度是基于好奇心驱动的探索的核心，这种方法会奖励智能体前往其模型仍在学习的地方。然而，学习进度这一抽象概念无法被直接测量，现有方法通常从输出空间中的预测误差来推导它。本文提出了梯度-动量耦合，它衡量一个样本在参数空间中驱动变化的强度，由其梯度与先前梯度动量的归一化绝对乘积给出。在样本之间持续存在的变化方向会在动量中不断累积，而噪声则会相互抵消。在受控实验中，GMC 在具有不同噪声水平的任务之间分配近乎均匀的优先级，而预测误差则会追逐噪声最大的任务；GMC 按照改进速度而非难度对可学习的任务进行排序。在四个 MiniGrid MultiRoom 任务上，将内在好奇心模块中的预测误差替换为 GMC（摘要在此处被截断）……

    arXiv:2605.05856v2 Announce Type: replace  Abstract: Measuring learning progress is at the core of curiosity-driven exploration, which rewards an agent for going where its model is still learning. However, the abstract notion of learning progress is not directly measurable, and existing methods often derive it from the prediction error in the output space. This paper proposes Gradient-Momentum Coupling (GMC), which measures how strongly a sample drives change in the parameter space, given by the normalized absolute product of its gradient with the momentum of previous gradients. Directions of change that persist across samples accumulate in momentum, while noise cancels out. In controlled experiments GMC allocates near uniform priority across tasks with varying levels of noise, where prediction error chases the noisiest, and orders learnable tasks by improvement speed rather than difficulty. On four MiniGrid MultiRoom tasks, substituting GMC for prediction error inside the Intrinsic Cu
    
[^291]: 基于动力系统预测的大语言模型幻觉低成本黑盒检测方法

    Low-Cost Black-Box Detection of LLM Hallucinations via Dynamical System Prediction

    [https://arxiv.org/abs/2605.05134](https://arxiv.org/abs/2605.05134)

    该论文提出将大语言模型视为黑盒动力系统，利用Koopman算子理论分别对事实性与幻觉性响应的状态空间转移算子进行拟合，并通过差分残差评分实现无需二次采样或外部知识检索的低成本单次幻觉检测。

    

    大语言模型（LLM）经常生成看似合理但不符合事实的内容，这种现象被称为幻觉。现有的检测方法通常依赖于计算成本高昂的基于采样的一致性检查或外部知识检索，而我们提出了一种新方法，将大语言模型视为黑盒动力系统。通过嵌入模型将大语言模型的响应投影到高维流形中，我们将由此得到的向量序列表征为模型潜在状态空间动力学的可观测实现。利用Koopman算子理论，我们分别为事实性和幻觉性两种状态拟合其转移算子，并基于各自的预测误差定义了差分残差评分。这种方法能够在单次样本传递中实现低成本的幻觉检测，避免了二次采样或外部接地（grounding）的需要。在三个数据基准上的广泛测试证明了……

    arXiv:2605.05134v2 Announce Type: replace  Abstract: Large Language Models (LLMs) frequently generate plausible but non-factual content, a phenomenon known as hallucination. While existing detection methods typically rely on computationally expensive sampling-based consistency checks or external knowledge retrieval, we propose a new method that treats the LLM as a black-box dynamical system. By projecting LLM responses into a high-dimensional manifold via an embedding model, we characterize the resulting vector sequences as observable realizations of the model's latent state-space dynamics. Leveraging Koopman operator theory, we fit the transition operators for both factual and hallucinated regimes and define a differential residual score based on their respective prediction errors. This approach enables low-cost hallucination detection in a single-sample pass, avoiding the need for secondary sampling or external grounding. Extensive testing across three data benchmarks demonstrates th
    
[^292]: 无限宽度能持续多久？长程线性递归中的信号传播

    How Long Does Infinite Width Last? Signal Propagation in Long-Range Linear Recurrences

    [https://arxiv.org/abs/2605.05113](https://arxiv.org/abs/2605.05113)

    本文推导出复高斯初始化下线性递归模型隐状态信号能量的精确有限宽度公式，并揭示了无限宽度近似仅在深度满足 $t=o(\sqrt n)$ 的亚临界区域有效，而在 $t\sim c\sqrt n$ 的临界区域会出现不可忽略的偏差。

    

    我们研究了有限宽度下线性递归模型中的信号传播。尽管现有的信号传播理论主要依赖于无限宽度极限，但当递归深度 $t$ 与宽度 $n$ 共同增长时，该近似能在多长时间内保持准确仍不清楚。这一问题对现代递归序列模型尤为重要，因为其自然运行场景涉及长输入序列，即较大的 $t$。我们在复高斯初始化下推导出了线性递归中隐状态信号能量的精确有限宽度公式。利用这些公式，我们确定了支配信号传播的深度-宽度联合缩放区域：(i) 亚临界区域 $t=o(\sqrt n)$，其中无限宽度近似仍然有效；(ii) 临界区域 $t\sim c\sqrt n$，其中出现对无限宽度预测不可忽略的偏离，并呈现非平凡的联合缩放极限

    arXiv:2605.05113v2 Announce Type: replace  Abstract: We study signal propagation in linear recurrent models at finite width. While existing signal propagation theory relies predominantly on the infinite-width limit, it remains unclear for how long that approximation remains accurate when recurrent depth $t$ grows jointly with width $n$. This question is especially relevant for modern recurrent sequence models, whose natural operating regime involves long input sequences, i.e., large $t$. We derive exact finite-width formulas for the hidden state signal energies in linear recurrences under complex Gaussian initialization. Using these formulas, we identify the joint depth--width scaling regimes that govern signal propagation: (i) a \emph{subcritical regime} $t=o(\sqrt n)$, in which the infinite-width approximation remains valid; (ii) a \emph{critical regime} $t\sim c\sqrt n$, in which non-negligible deviations from infinite-width predictions appear and a nontrivial joint scaling limit em
    
[^293]: 基于复杂网络的时间序列持续同调

    Persistent Homology of Time Series through Complex Networks

    [https://arxiv.org/abs/2605.01624](https://arxiv.org/abs/2605.01624)

    该论文提出了一个结合复杂网络与持续同调的统一时间序列分类流程，通过系统比较五种图构造发现没有任何单一构造普遍最优（最优图类型取决于信号的判别结构），且图距离度量是影响分类性能的首要设计因素。

    

    我们提出了一种通过复杂网络和持续同调进行单变量时间序列分类的统一流程。时间序列通过三大家族（可见性（自然可见性图和水平可见性图）、转换和邻近性）中的五种图构造之一被映射为图，然后将图转换为相异度矩阵，并通过 Vietoris-Rips 过滤从中生成持续图。这些持续图通过持续景观和拓扑摘要统计被向量化为固定长度的特征。通过标准化下游处理流程，分类性能的差异可以仅归因于网络构造和距离度量本身。在十二个 UCR 基准数据集上的实验表明：(i) 没有单一的图构造占绝对优势——最优的图类型取决于信号的判别结构；(ii) 图距离度量是一阶设计选择，扩散距离一致地……（摘要在此处截断）

    arXiv:2605.01624v1 Announce Type: cross  Abstract: We present a unified pipeline for univariate time series classification via complex networks and persistent homology. A time series is mapped to a graph through one of five constructions across three families (visibility (natural and horizontal visibility graphs), transition, and proximity) and the graph is converted to a dissimilarity matrix from which a Vietoris-Rips filtration yields persistence diagrams. These diagrams are vectorized into fixed-length features through persistence landscapes and topological summary statistics. By standardizing the downstream processing, differences in classification performance are attributable to the network construction and distance metric alone. Experiments on twelve UCR benchmarks show that (i) no single construction dominates: the optimal graph type depends on the signal's discriminative structure; (ii) the graph distance metric is a first-order design choice, with diffusion distance uniformly 
    
[^294]: 解锁预测经济：面向预测市场全生命周期的数据集套件：[实验与分析]

    Unlocking the Forecasting Economy: A Suite of Datasets for the Full Lifecycle of Prediction Market: [Experiments \& Analysis]

    [https://arxiv.org/abs/2604.20421](https://arxiv.org/abs/2604.20421)

    本文提出了首个覆盖去中心化预测市场全生命周期六个阶段（市场创建、代币注册、交易、预言机交互、争议处理与最终结算）的持续同步数据集套件，解决了链上智能合约与链下数据源之间数据严重碎片化、难以整合追踪的关键难题。

    

    预测市场是针对普遍性未来事件（如总统选举）的相关声明进行交易的市场。在超过500亿美元交易量的迅猛增长推动下，预测市场已成为一种颇具前景的预测机制，其价格能够提供集体信念的持续更新信号。在去中心化平台（如Polymarket）中，预测市场的生命周期包括六个阶段：市场创建、代币注册、交易、预言机交互、争议处理以及最终结算。然而，全面追踪这一完整流程仍然是一个重大挑战，因为底层数据严重碎片化地分布在异构的链上智能合约和链下数据源之中。为填补这一关键空白，我们提出了首个针对去中心化预测市场全生命周期的持续同步数据集套件，以实现大规模的跨数据源集成、不完整数据链接以及持续同步。

    arXiv:2604.20421v2 Announce Type: replace  Abstract: Prediction markets are markets for trading claims on universal future events (e.g., presidential elections). Fueled by a meteoric surge with over \$50 billion trading volume, they have emerged as a promising forecasting mechanism, where their prices provide continuously updated signals of collective beliefs. In decentralized platforms (e.g., Polymarket), the prediction market lifecycle include six stages: market creation, token registration, trading, oracle interaction, dispute, and final settlement. However, comprehensively tracking this complete pipeline remains a major challenge, as the underlying data are severely fragmented across heterogeneous on-chain smart contracts and off-chain sources. To fill this critical gap, we present the first continuously synchronized dataset suite for the full-lifecycle of decentralized prediction markets. To achieve large-scale cross-source integration, incomplete linkage, and continuous synchroni
    
[^295]: 地下真菌生物多样性可利用自监督学习卫星特征进行监测

    Below-ground Fungal Biodiversity Can be Monitored Using Self-Supervised Learning Satellite Features

    [https://arxiv.org/abs/2604.09818](https://arxiv.org/abs/2604.09818)

    该研究证明，利用自监督学习从卫星影像提取的特征可有效预测地下菌根真菌物种丰富度，其预测能力超越气候、土壤和土地覆盖等传统基线，并将监测空间分辨率提升了10,000倍。

    

    菌根真菌对陆地生态系统功能至关重要。然而，由于时间和成本的限制，在景观尺度上监测其生物多样性往往难以实现。目前的预测表明，90%的菌根多样性热点区域仍未受到保护，这引发了如何广泛且有效地绘制地下真菌群落分布图的问题。我们展示了将自监督学习（SSL）应用于卫星影像，可以预测不同环境下的地下外生菌根真菌物种丰富度。我们的模型在覆盖欧洲和亚洲的约12,000个野外样本中解释了超过一半的物种丰富度方差。SSL衍生特征是所测试的所有预测变量组中信息量最大的，其表现超越了现有的气候、土壤和土地覆盖等基线方法。与现有技术相比，我们实现了10,000倍的空间分辨率提升，从1公里的景观平均值提升到10米尺度的栖息地观测。随着卫星观测的……

    arXiv:2604.09818v2 Announce Type: replace  Abstract: Mycorrhizal fungi are vital to terrestrial ecosystem functioning. Yet monitoring their biodiversity at landscape scales is often unfeasible due to time and cost constraints. Current predictions suggest that 90% of mycorrhizal diversity hotspots remain unprotected, opening questions of how to broadly and effectively map underground fungal communities. We show that self-supervised learning (SSL) applied to satellite imagery can predict below-ground ectomycorrhizal fungal richness across diverse environments. Our models explain over half the variance in species richness across ~12,000 field samples spanning Europe and Asia. SSL-derived features are the most informative tested predictor group, and outperform each of the established climate, soil, and land cover baselines. We achieve a 10,000-fold increase in spatial resolution over existing techniques, moving from 1km landscape averages to 10m habitat-scale observations. As satellite obs
    
[^296]: 使用单一代理变量识别因果效应

    Identifying Causal Effects Using a Single Proxy Variable

    [https://arxiv.org/abs/2604.09135](https://arxiv.org/abs/2604.09135)

    该论文提出SPICE假设，证明在已知混杂因素生成单一（多维）代理变量机制的前提下因果效应可识别，将经典代理变量方法扩展到多维连续场景，并开发了适用于离散和连续处理的神经网络估计框架SPICE-Net。

    

    未观测的混杂因素是估计从处理变量到结果变量的因果效应时的一个关键挑战。在这项工作中，我们假设观察到未观测混杂因素的一个单一（可能是多维的）代理变量，并且已知混杂因素生成该代理变量的机制。在一个称为“因果效应的单一代理可识别性”（Single Proxy Identifiability of Causal Effects，简称SPICE）的假设下，我们证明了该误差机制是完备的，且因果效应是可识别的。我们将Kuroki和Pearl (2014)以及Pearl (2010)基于代理变量的因果可识别性结果扩展到多维连续设置、更灵活的函数关系以及更广泛的分布类别。此外，我们开发了一个基于神经网络的估计框架SPICE-Net来估计因果效应，该框架同时适用于离散和连续的处理变量。

    arXiv:2604.09135v2 Announce Type: replace  Abstract: Unobserved confounding is a key challenge when estimating causal effects from a treatment on an outcome. In this work, we assume that we observe a single, potentially multi-dimensional proxy variable of the unobserved confounder and that we know the mechanism that generates the proxy from the confounder. Under an assumption called Single Proxy Identifiability of Causal Effects or simply SPICE, we prove that this error mechanism is complete and causal effects are identifiable. We extend the proxy-based causal identifiability results by Kuroki and Pearl (2014); Pearl (2010) to multi-dimensional continuous settings, more flexible functional relationships and a broader class of distributions. Further, we develop a neural network based estimation framework, SPICE-Net, to estimate causal effects, which is applicable to both discrete and continuous treatments.
    
[^297]: 面向模型预测控制的RC建筑热模型神经参数估计

    Neural Parameter Estimation of RC Thermal Building Models for Model Predictive Control

    [https://arxiv.org/abs/2604.05904](https://arxiv.org/abs/2604.05904)

    该论文提出一种将RC模型物理方程嵌入神经网络训练过程的神经参数估计方法，并通过多建筑数据预训练进一步提升精度，为建筑节能模型预测控制提供了无需初始猜测、计算高效且准确的RC参数估计方案。

    

    arXiv:2604.05904v2 公告类型：replace-cross 摘要：灰箱RC模型被广泛用于实现建筑中节能的模型预测控制（MPC）。然而，RC参数估计仍然困难，因为传统的基于优化的算法容易陷入局部极小值、严重依赖良好的初始猜测，且计算成本高昂。为解决这些问题，我们提出了一种新颖的神经参数估计方法——Estimator from Scratch（从零估计器），该方法将物理方程嵌入到神经网络的训练过程中来估计RC参数。为进一步提高估计精度并消除对初始猜测的依赖，我们通过在多个源建筑的数据上预训练神经网络来扩展该方法，即Pretrained Estimator（预训练估计器）。我们将这两种方法与基于遗传算法的RC估计器以及完全黑箱神经网络进行了基准比较。所有方法均在八座模拟建筑和三座真实建筑上针对两种RC配置进行了评估（摘要在此处被截断）。

    arXiv:2604.05904v2 Announce Type: replace-cross  Abstract: Gray-box RC models are widely used to enable energy-efficient model predictive control (MPC) in buildings. However, estimating RC parameters remains difficult, as conventional optimization-based algorithms are prone to local minima, rely heavily on good initial guesses, and incur high computational cost. To address these issues, we propose the Estimator from Scratch, a novel neural parameter estimation approach that embeds the physical equations into a neural network's training process to estimate RC parameters. To further improve estimation accuracy and eliminate dependence on an initial guess, we extend this approach by pretraining the neural network on data from multiple source buildings, the Pretrained Estimator. We benchmark both methods against a genetic-algorithm-based RC estimator and a fully black-box neural network. All methods are evaluated across eight simulated and three real-world buildings for two RC configuratio
    
[^298]: Combee：为自我改进的语言模型智能体扩展提示学习

    Combee: Scaling Prompt Learning for Self-Improving Language Model Agents

    [https://arxiv.org/abs/2604.04247](https://arxiv.org/abs/2604.04247)

    Combee 提出了一种新颖的框架，通过有原则的并行扩展策略解决现有提示学习方法在高并行度下质量下降的问题，从而同时提升自我改进语言模型智能体提示学习的效率和效果。

    

    提示学习（prompt learning）的最新进展使大型语言模型智能体能够在不改变参数的情况下，从推理时上下文中获取与任务相关的知识。例如，现有方法（如 ACE 或 GEPA）可以基于先前的智能体运行记录来学习系统提示，从而提升准确率。然而，这些方法主要集中于单智能体或低并行度的设置，这从根本上限制了它们从大量收集的智能体轨迹中高效学习的能力。鉴于从大量智能体轨迹或并行智能体执行中学习的趋势日益增长，以并行方式运行提示学习将是高效且有益的。然而，由于缺乏有原则的扩展策略，当前方法在高并行度下会出现质量下降的问题。为了同时提升提示学习的效率和效果，我们提出了 Combee，一个用于扩展自我改进智能体并行提示学习的新颖框架。Combee 加速了学习……

    arXiv:2604.04247v2 Announce Type: replace  Abstract: Recent advances in prompt learning allow large language model agents to acquire task-relevant knowledge from inference-time context without parameter changes. For example, existing methods (like ACE or GEPA) can learn system prompts to improve accuracy based on previous agent runs. However, these methods primarily focus on single-agent or low-parallelism settings. This fundamentally limits their ability to efficiently learn from a large set of collected agentic traces. It would be efficient and beneficial to run prompt learning in parallel to accommodate the growing trend of learning from many agentic traces or parallel agent executions. Yet without a principled strategy for scaling, current methods suffer from quality degradation with high parallelism. To improve both the efficiency and quality of prompt learning, we propose Combee, a novel framework to scale parallel prompt learning for self-improving agents. Combee speeds up learn
    
[^299]: 基于行列式的尖锐范数不等式与Buzano不等式

    A Sharp Norm Inequality and Buzano's Inequality via Determinants

    [https://arxiv.org/abs/2604.01525](https://arxiv.org/abs/2604.01525)

    本文利用参数化二次型族的行列式结构，给出了联系三种基本范数的尖锐不等式的简短证明，证明了常数 $(1+\sqrt{n})/2$ 的最优性，并同时证明了实向量情形的 Buzano 不等式。

    

    我们给出不等式 $$ \|x\|_1\,\|x\|_\infty \le \frac{1+\sqrt{n}}{2}\,\|x\|_2^2 $$ 的一个简短的线性代数证明，该不等式对每个 $x\in\mathbb{R}^n$ 都成立。这个不等式联系了有限维空间上的三种基本范数，并在优化和数值分析中有应用。我们的证明利用了一个参数化二次型族的行列式结构，并证明常数 $(1+\sqrt{n})/2$ 是最优的。该不等式是 Buzano 不等式的特例，同样的方法也证明了实向量情形的 Buzano 不等式。

    arXiv:2604.01525v2 Announce Type: replace-cross  Abstract: We give a short linear-algebraic proof of the inequality $$ \|x\|_1\,\|x\|_\infty \le \frac{1+\sqrt{n}}{2}\,\|x\|_2^2, $$ valid for every $x\in\mathbb{R}^n$. This inequality relates three fundamental norms on finite-dimensional spaces and has applications in optimization and numerical analysis. Our proof exploits the determinantal structure of a parametrized family of quadratic forms, and we show the constant $(1+\sqrt{n})/2$ is optimal. The inequality is a special case of Buzano's inequality, and the same method also proves Buzano's inequality for real vectors.
    
[^300]: 关于具有巨大容量的稀疏模式联想神经网络

    On associative neural networks for sparse patterns with huge capacities

    [https://arxiv.org/abs/2603.26217](https://arxiv.org/abs/2603.26217)

    本文通过将高阶交互与稀疏联想记忆相结合，将Amari和Willshaw模型的存储容量提升至 $N^n/(\log N)^n$ 量级，并在交互阶数对数增长时实现超多项式的存储规模。

    

    具有高阶或指数交互项的广义Hopfield模型已被证明比经典二次模型具有大得多的存储容量。另一方面，针对稀疏模式的联想记忆模型，如Willshaw和Amari模型，在稀疏情形下已经表现出增强的存储容量。本文将这两种机制结合起来，引入了稀疏联想记忆模型的高阶版本，并在固定模式稳定性的意义下研究它们的存储容量。对于具有固定交互阶数 $n$ 的Amari和Willshaw模型，我们得到了量级为 $\frac{N^n}{(\log N)^n}$ 的存储规模。当交互阶数随神经元数量对数增长时，所得的存储规模变为超多项式级。我们还研究了块结构Gripon–Berrou架构中的高阶交互，其自然存储规模为 $c^n$ 量级。

    arXiv:2603.26217v2 Announce Type: replace-cross  Abstract: Generalized Hopfield models with higher-order or exponential interaction terms are known to have substantially larger storage capacities than the classical quadratic model. On the other hand, associative memories for sparse patterns, such as the Willshaw and Amari models, already exhibit enhanced storage capacities in the sparse regime.   In this paper we combine these two mechanisms. We introduce higher-order versions of sparse associative memory models and study their storage capacities in the sense of fixed-pattern stability. For the Amari and Willshaw models with fixed interaction order $n$, we obtain storage scales of order $\frac{N^n}{(\log N)^n}$. When the interaction order grows logarithmically with the number of neurons, the resulting storage scale becomes super-polynomial. We also study higher-order interactions in the block-structured Gripon--Berrou architecture, where the natural storage scale is of order $c^n$.   O
    
[^301]: 论Transformer对上下文关系的表达能力

    On the Expressive Power of Transformers for Contextual Relations

    [https://arxiv.org/abs/2603.25860](https://arxiv.org/abs/2603.25860)

    本文基于概率与最优传输理论构建了数学框架，揭示了注意力归一化与最优传输的深刻联系——softmax归一化产生条件关系而Sinkhorn归一化产生联合关系——并证明了Transformer在表示上下文关系上的通用逼近能力。

    

    Transformer通过将注意力作为建模上下文内交互的核心机制，彻底改变了机器学习。尽管注意力扮演着核心角色，但Transformer在表示上下文关系方面的理论能力仍不明确。在本工作中，我们通过构建一个基于概率和最优传输的数学框架来研究这一问题。我们将文本视为其表示的分布，并将注意力视为这些表示之间的概率关系。这一视角揭示了注意力归一化与最优传输之间的联系：标准的softmax归一化产生条件关系，而Sinkhorn归一化产生具有指定边缘分布的联合关系。因此，这两种机制都能从注意力分数中提供结构化的概率关系。在温和的条件下，我们为这两种设置建立了通用逼近结果。我们证明Transformer架构……（摘要原文此处截断）

    arXiv:2603.25860v4 Announce Type: replace  Abstract: Transformers have revolutionized machine learning by making attention a central mechanism for modeling interactions within a context. Despite the central role of attention, the theoretical capabilities of Transformers for representing contextual relations remain unclear. In this work, we address this question by developing a mathematical framework based on probability and optimal transport. We view a text as a distribution of its representations and attention as a probabilistic relation between them. This perspective reveals a connection between attention normalization and optimal transport: standard softmax normalization produces conditional relations, while Sinkhorn normalization produces joint relations with prescribed marginals. Thus, both mechanisms provide structured probabilistic relations from attention scores. Under mild conditions, we establish universal approximation results for both settings. We show that Transformer arch
    
[^302]: 谱球约束超连接

    Spectral-Sphere-Constrained Hyper-Connections

    [https://arxiv.org/abs/2603.20896](https://arxiv.org/abs/2603.20896)

    针对双随机约束超连接存在的恒等退化、表达能力瓶颈和参数化开销三重局限，提出将残差矩阵约束在谱球流形上的谱球约束超连接，在保持恒等映射性质以稳定训练的同时，恢复跨流混合的谱自由度和表达能力。

    

    摘要：超连接将残差连接扩展为多条流，并利用残差矩阵进行跨流混合，以丰富模型的表达能力。然而，不加约束的混合会破坏残差连接固有的恒等映射性质，导致训练不稳定。为解决这一问题，流形约束超连接及其变体通过 Sinkhorn-Knopp（SK）算法或基于置换的参数化，将这些矩阵限制为双随机矩阵。我们揭示了这种双随机约束的三个局限：(1) 恒等退化，即学习到的矩阵坍缩在恒等初始化附近，削弱了跨流交互；(2) 表达能力瓶颈，双随机约束限制了残差矩阵次主导谱的自由度，使模型无法有选择地保留或衰减跨流变化；(3) 参数化……（原摘要在此处截断）

    arXiv:2603.20896v2 Announce Type: replace-cross  Abstract: Hyper-Connections (HC) extend residual connections into multiple streams, employing residual matrices for cross-stream mixing to enrich model expressivity. However, unconstrained mixing disrupts the identity mapping property intrinsic to the residual connection, causing unstable training. To address this, Manifold-Constrained Hyper-Connections (mHC) and its variants restrict these matrices to be doubly stochastic via Sinkhorn-Knopp (SK) algorithm or permutation-based parameterizations. We reveal three limitations of this doubly stochastic constraint: (1) identity degeneration, where learned matrices collapse around the identity initialization and diminish cross-stream interactions, (2) a expressivity bottleneck, where the doubly stochastic constraint restricts the freedom of the subdominant spectrum of the residual matrices, preventing the model from selectively preserving or attenuating cross-stream variations, and (3) paramet
    
[^303]: 逐层目标传播：通过以目标为中心的传播实现高效的组件归因

    Layer-wise Target Propagation: Efficient Component Attribution through Target Centric Propagation

    [https://arxiv.org/abs/2603.19742](https://arxiv.org/abs/2603.19742)

    提出逐层目标传播（LTP）框架，仅需一次前向和一次反向传播即可在冻结的Transformer上忠实地追踪信息流，在模型组件数量方面实现O(1)时间复杂度的高效密集组件归因。

    

    理解基于Transformer的大型语言模型（LLM）的内部机制对于其可靠部署和有效运行至关重要。尽管近期的研究已经产生了大量试图在忠实性与计算效率之间取得平衡的归因方法，但密集的组件归因的代价仍然高得令人望而却步。在本工作中，我们提出了逐层目标传播（Layer-wise Target Propagation, LTP），这是一种新颖的框架，能够在冻结的Transformer上通过一次前向传播和一次反向传播忠实地追踪信息流，而无需反事实样本。LTP以解析方式将Transformer的计算结构分解并线性化为不同的路径，并沿着这些路径传播一个目标化的反嵌入（unembedding）向量，从而在每个残差位置获得有效表示。这种以目标为中心的传播在模型组件数量方面实现了O(1)的时间复杂度，并可扩展至长输入场景。

    arXiv:2603.19742v3 Announce Type: replace-cross  Abstract: Understanding the internal mechanisms of transformer-based large language models (LLMs) is crucial for their reliable deployment and effective operation. While recent efforts have yielded a plethora of attribution methods attempting to balance faithfulness and computational efficiency, dense component attribution remains prohibitively expensive. In this work, we introduce Layer-wise Target Propagation (LTP), a novel framework that faithfully traces information flow on the frozen transformer in one forward and one backward pass without requiring counterfactual examples. LTP analytically decomposes and linearizes the computational structure of the Transformers into distinct pathways along which it propagates a targeted unembedding vector to receive the effective representation at each residual position. This target-centric propagation achieves O(1) time complexity with respect to the number of model components, scaling to long in
    
[^304]: 差分隐私下的极小极大与自适应协方差矩阵估计

    Minimax and Adaptive Covariance Matrix Estimation under Differential Privacy

    [https://arxiv.org/abs/2603.19703](https://arxiv.org/abs/2603.19703)

    本文提出了在差分隐私下针对不同协方差矩阵类别的最优估计器，揭示了隐私约束与数据几何结构如何共同影响高维协方差估计的极小极大速率。

    

    协方差矩阵的估计是广泛统计应用的基础。本文研究了在$\rho$-零集中差分隐私（$\rho$-zCDP）下，对三个嵌套类别的高维协方差矩阵进行极小极大和自适应估计：逐点衰减类$\mathcal{H}_\alpha$、行尾类$\mathcal{G}_\alpha$和分块类$\mathcal{F}_\alpha$。我们考虑了平方算子范数损失和归一化平方Frobenius范数损失。对于$\mathcal{H}_\alpha$和$\mathcal{G}_\alpha$，我们开发了针对这两类精细几何结构量身定制的中心-外部二进估计器，而对于$\mathcal{F}_\alpha$，我们开发了分块三对角估计器。所得的极小极大最优速率揭示了平滑度$\alpha$、损失函数、协方差类几何结构和隐私约束之间非平凡的相互作用。与非隐私设置相比，隐私约束显著改变了估计的收敛速率和自适应性质。

    arXiv:2603.19703v2 Announce Type: replace-cross  Abstract: Estimating covariance matrices is fundamental to a wide range of statistical applications. This paper studies minimax and adaptive estimation of high-dimensional covariance matrices under $\rho$-zero-concentrated differential privacy ($\rho$-zCDP) over three nested classes: the pointwise-decay class $\mathcal{H}_\alpha$, the row-tail class $\mathcal{G}_\alpha$, and the separated-block class $\mathcal{F}_\alpha$. We consider both squared operator norm loss and normalized squared Frobenius norm loss.   For $\mathcal{H}_\alpha$ and $\mathcal{G}_\alpha$, we develop center--outer dyadic estimators tailored to the refined geometry of the two classes, while for $\mathcal{F}_\alpha$, we develop a blockwise tridiagonal estimator. The resulting minimax-optimal rates reveal a nontrivial interplay among the smoothness $\alpha$, the loss, the geometry of the covariance class, and the privacy constraint. In contrast to the non-private settin
    
[^305]: 分布条件化传输

    Distribution-Conditioned Transport

    [https://arxiv.org/abs/2603.04736](https://arxiv.org/abs/2603.04736)

    提出了分布条件化传输（DCT）框架，通过将传输映射条件化于源分布和目标分布的学习嵌入表示上，实现对未见分布对的泛化，并支持利用单条件观测分布的半监督学习和多种底层传输机制。

    

    学习一个将源分布映射到目标分布的传输模型是机器学习中的经典问题，但科学应用越来越需要能够泛化到训练期间未见过的源分布和目标分布的模型。我们提出了分布条件化传输（Distribution-Conditioned Transport, DCT），这是一个将传输映射条件化于源分布和目标分布的学习嵌入表示上的框架，从而实现对未见过的分布对的泛化。DCT 还支持分布预测问题的半监督学习：由于它可以从任意分布对中学习，因此能够利用仅在单一条件下观测到的分布来改进传输预测。DCT 对底层传输机制是不可知的，支持从流匹配到基于分布散度的模型（如 Wasserstein、MMD）等多种模型。我们在合成基准上展示了 DCT 的实际性能优势。

    arXiv:2603.04736v2 Announce Type: replace  Abstract: Learning a transport model that maps a source distribution to a target distribution is a canonical problem in machine learning, but scientific applications increasingly require models that can generalize to source and target distributions unseen during training. We introduce distribution-conditioned transport (DCT), a framework that conditions transport maps on learned embeddings of source and target distributions, enabling generalization to unseen distribution pairs. DCT also allows semi-supervised learning for distributional forecasting problems: because it learns from arbitrary distribution pairs, it can leverage distributions observed at only one condition to improve transport prediction. DCT is agnostic to the underlying transport mechanism, supporting models ranging from flow matching to distributional divergence-based models (e.g. Wasserstein, MMD). We demonstrate the practical performance benefits of DCT on synthetic benchmar
    
[^306]: 尺度不变的高斯导数残差网络

    Scale-invariant Gaussian derivative residual networks

    [https://arxiv.org/abs/2603.02843](https://arxiv.org/abs/2603.02843)

    本文提出了一种可证明尺度不变的高斯导数残差网络，通过在高斯导数层中引入残差跳跃连接构建更深的网络，在显著提升精度的同时保持优异的跨尺度泛化能力，并为任意维度下的尺度协变与尺度不变性质提供了严格的数学证明。

    

    arXiv:2603.02843v2 公告类型：replace-cross。摘要：跨图像尺度的泛化能力仍然是深度网络面临的一个根本性挑战，深度网络通常无法处理训练中未见过的尺度的图像（即分布外问题）。在本文中，我们提出了可证明具有尺度不变性的高斯导数残差网络，该网络由级联耦合的尺度协变高斯导数残差块构成，旨在解决这一问题。通过在先前的高斯导数层概念基础上添加残差跳跃连接，可以构建更深的网络，其精度得到显著提升，同时保持了非常好的尺度泛化特性。本文为任意维度下的尺度协变和尺度不变性质提供了明确的数学证明。为了分析GaussDerResNets泛化到新尺度的能力，我们将其应用于STL-10数据集的一个新的重新缩放版本，其中训练是在单一固定[尺度上进行的……（摘要在此处截断）

    arXiv:2603.02843v2 Announce Type: replace-cross  Abstract: Generalisation across image scales remains a fundamental challenge for deep networks, which often fail to handle images at scales not seen during training (the out-of-distribution problem). In this paper, we present provably scale-invariant Gaussian derivative residual networks (GaussDerResNets), constructed out of scale-covariant Gaussian derivative residual blocks coupled in cascade, aimed at addressing this problem. By adding residual skip connections to the previous notion of Gaussian derivative layers, deeper networks with substantially increased accuracy can be constructed, while preserving very good scale generalisation properties. Explicit proofs are provided for the underlying scale-covariant and scale-invariant properties in arbitrary dimensions.   To analyse the ability of GaussDerResNets to generalise to new scales, we apply them on a new rescaled version of the STL-10 dataset, where training is done at a single fix
    
[^307]: 面向可泛化长期物理模拟的潜在生成求解器

    Latent Generative Solvers for Generalizable Long-Term Physics Simulation

    [https://arxiv.org/abs/2602.11229](https://arxiv.org/abs/2602.11229)

    本文提出潜在生成求解器（LGS），通过物理VAE压缩十二个PDE族到共享潜在流形、金字塔流强制Transformer进行流匹配生成，以及训练时输入加噪的稳定性保证，首次实现了跨异构PDE族的泛化能力与长时域自回归物理模拟稳定性的兼顾。

    

    可靠的物理模拟需要两种当今神经偏微分方程（PDE）求解器无法同时具备的能力：跨异构PDE族的泛化能力，以及长时间自回归滚动预测下的稳定性。确定性算子会以几何级数累积误差，而现有的概率求解器则局限于单一PDE族或较短的预测时域。我们通过潜在生成求解器（Latent Generative Solver, LGS）弥合了这一差距，该求解器由三个相互耦合的组件构成：(i) 物理VAE（PhyVAE），将十二个PDE族压缩到共享的潜在流形中；(ii) 金字塔流强制Transformer（PFlowFT），通过流匹配生成下一个潜在状态，并以基于模型自身预测而更新的每条轨迹上下文作为条件；(iii) 训练期间的输入加噪，我们为此推导了一个充分条件下的收缩界，用以解释所观察到的长时域稳定性。LGS在一个包含250万条轨迹、16个系统、分辨率为128²的语料库上进行预训练，其匹配……（摘要原文在此处被截断）

    arXiv:2602.11229v3 Announce Type: replace  Abstract: Reliable physics simulation demands two capabilities that today's neural PDE solvers do not deliver together: generalization across heterogeneous PDE families, and stability under long autoregressive rollouts. Deterministic operators accumulate error geometrically, while existing probabilistic solvers are confined to a single PDE family or short horizons. We close this gap with the \textbf{Latent Generative Solver} (LGS), three coupled components: (i) a Physics VAE (PhyVAE) compressing twelve PDE families into a shared latent manifold; (ii) a Pyramidal Flow-Forcing Transformer (PFlowFT) that generates the next latent by flow matching, conditioned on a per-trajectory context updated on the model's own predictions; and (iii) input noising during training, for which we derive a sufficient-condition contraction bound explaining the observed long-horizon stability. Pretrained on a 2.5\,M-trajectory, 16-system corpus at $128^2$, LGS matche
    
[^308]: 伪可逆神经网络

    Pseudo-Invertible Neural Networks

    [https://arxiv.org/abs/2602.06042](https://arxiv.org/abs/2602.06042)

    本文提出满射伪可逆神经网络（SPNN），将摩尔-彭若斯伪逆自然推广到非线性神经网络，并形式化非线性反投影（NLBP）方法，从而扩展了零样本逆问题的求解范围。

    

    摩尔-彭若斯伪逆（PInv）是线性系统的基本求解方法。在本文中，我们提出了PInv向非线性领域的自然推广，特别是向神经网络的推广。我们引入了满射伪可逆神经网络（SPNN），这是一类专门设计用于支持可处理的非线性伪逆的架构。所提出的非线性伪逆及其在SPNN中的实现满足基本的几何性质。其中一个性质是零空间投影或“反投影”，$x' = x + A^\dagger(y-Ax)$，它将样本$x$移动到满足$Ax=y$的最接近的一致状态$x'$。我们形式化了非线性反投影（NLBP），这是一种通过我们定义的伪逆为非线性映射$f(x)=y$保证相同一致性约束的方法。我们利用SPNN来扩展零样本逆问题的适用范围。基于扩散的零空间投影已经彻底改变了零样本……

    arXiv:2602.06042v2 Announce Type: replace  Abstract: The Moore-Penrose Pseudo-inverse (PInv) serves as the fundamental solution for linear systems. In this paper, we propose a natural generalization of PInv to the nonlinear regime in general and to neural networks in particular. We introduce Surjective Pseudo-invertible Neural Networks (SPNN), a class of architectures explicitly designed to admit a tractable non-linear PInv. The proposed non-linear PInv and its implementation in SPNN satisfy fundamental geometric properties. One such property is null-space projection or "Back-Projection", $x' = x + A^\dagger(y-Ax)$, which moves a sample $x$ to its closest consistent state $x'$ satisfying $Ax=y$. We formalize Non-Linear Back-Projection (NLBP), a method that guarantees the same consistency constraint for non-linear mappings $f(x)=y$ via our defined PInv. We leverage SPNNs to expand the scope of zero-shot inverse problems. Diffusion-based null-space projection has revolutionized zero-shot
    
[^309]: 学习关键之处：面向鲁棒偏好对齐的几何锚定

    Learning Where It Matters: Geometric Anchoring for Robust Preference Alignment

    [https://arxiv.org/abs/2602.04909](https://arxiv.org/abs/2602.04909)

    提出几何锚定偏好优化（GAPO），用当前策略的对抗性局部扰动作为动态几何感知锚点取代DPO的固定参考策略，并通过自适应重加权和锚点间隔机制，在噪声偏好监督下实现更鲁棒的大语言模型对齐。

    

    直接偏好优化（DPO）及相关方法通过针对固定参考策略对更新进行正则化，从成对偏好中对齐大型语言模型。然而，随着策略发生漂移，静态参考可能变得越来越失准，导致分布不匹配，并在噪声监督下放大虚假的偏好信号。相反，无参考的变体虽然避免了分布不匹配问题，但常常受到无约束奖励漂移的困扰。我们提出几何锚定偏好优化（GAPO），用一个动态的、几何感知的锚点取代固定参考：该锚点是当前策略在小半径内的对抗性局部扰动，作为悲观的基线。这一锚点使自适应重加权机制成为可能，能够根据每个偏好对的局部敏感性来调节其重要性。我们进一步引入“锚点间隔”，即策略与其锚点之间的奖励差异……

    arXiv:2602.04909v4 Announce Type: replace  Abstract: Direct Preference Optimization (DPO) and related methods align large language models from pairwise preferences by regularizing updates against a fixed reference policy. As the policy drifts, a static reference, however, can become increasingly miscalibrated, leading to distributional mismatch and amplifying spurious preference signals under noisy supervision. Conversely, reference-free variants avoid mismatch but often suffer from unconstrained reward drift. We propose Geometric Anchor Preference Optimization (GAPO), which replaces the fixed reference with a dynamic, geometry-aware anchor: an adversarial local perturbation of the current policy within a small radius that serves as a pessimistic baseline. This anchor enables an adaptive reweighting mechanism, modulating the importance of each preference pair based on its local sensitivity. We further introduce the Anchor Gap, the reward discrepancy between the policy and its anchor, a
    
[^310]: RAPTOR：岭自适应逻辑回归探针

    RAPTOR: Ridge-Adaptive Logistic Probes

    [https://arxiv.org/abs/2602.00158](https://arxiv.org/abs/2602.00158)

    提出了一种简单的L2正则化逻辑回归探针RAPTOR，通过验证集调优的岭回归强度从归一化权重中提取准确且方向稳定的概念向量，可用于大语言模型的“先探针后引导”激活引导流程。

    

    探针技术通过在冻结的大语言模型层表示之上训练轻量级预测器，来研究这些层表示中编码了哪些信息。除了分析用途之外，探针还经常被实际应用于“先探针后引导”流程中：从探针中提取学习到的概念向量，并通过加性激活引导的方式将其注入，即在前向传播过程中将其添加到某一层的表示上。该流程的有效性取决于所估计的概念向量是否准确、在消融下方向是否稳定，以及获取成本是否低廉。基于这些目标，我们提出了RAPTOR（岭自适应逻辑回归探针），这是一种简单的L2正则化逻辑回归探针，其通过验证集调优的岭回归强度从归一化权重中生成概念向量。在指令微调大语言模型和人类撰写的概念数据集上的大量实验中，RAPTOR在准确率上匹敌或超越强基线，同时实现了具有竞争力的方向稳定性。

    arXiv:2602.00158v3 Announce Type: replace-cross  Abstract: Probing studies what information is encoded in a frozen LLM's layer representations by training a lightweight predictor on top of them. Beyond analysis, probes are often used operationally in probe-then-steer pipelines: a learned concept vector is extracted from a probe and injected via additive activation steering by adding it to a layer representation during the forward pass. The effectiveness of this pipeline hinges on estimating concept vectors that are accurate, directionally stable under ablation, and inexpensive to obtain. Motivated by these desiderata, we propose RAPTOR (Ridge-Adaptive Logistic Probe), a simple L2-regularized logistic probe whose validation-tuned ridge strength yields concept vectors from normalized weights. Across extensive experiments on instruction-tuned LLMs and human-written concept datasets, RAPTOR matches or exceeds strong baselines in accuracy while achieving competitive directional stability an
    
[^311]: MeshGraphNet-Transformer：面向固体力学的可扩展基于网格的学习仿真

    MeshGraphNet-Transformer: Scalable Mesh-based Learned Simulation for Solid Mechanics

    [https://arxiv.org/abs/2601.23177](https://arxiv.org/abs/2601.23177)

    提出了一种融合Transformer全局建模能力与MeshGraphNets几何归纳偏置的新架构MGN-T，通过物理注意力机制直接捕获长程物理相互作用，无需深层消息传递或网格粗化，即可在工业规模的高分辨率网格上实现高效固体力学仿真学习。

    

    我们提出了MeshGraphNet-Transformer（MGN-T），这是一种新颖的架构，它将Transformer的全局建模能力与MeshGraphNets的几何归纳偏置相结合，同时保留了基于网格的图表示。MGN-T克服了标准MGN的一个关键局限，即在大型高分辨率网格上，由迭代消息传递导致的低效长程信息传播问题。物理注意力Transformer充当全局处理器，在显式保留节点和边属性的同时，同步更新所有节点状态。通过直接捕获长程物理相互作用，MGN-T消除了对深层消息传递堆栈或分层粗化网格的需求，从而能够在工业规模上，对具有不同几何形状、拓扑结构和边界条件的高分辨率网格进行高效学习。我们证明MGN-T能够成功处理冲击动力学问题的工业规模网格……

    arXiv:2601.23177v4 Announce Type: replace  Abstract: We present MeshGraphNet-Transformer (MGN-T), a novel architecture that combines the global modeling capabilities of Transformers with the geometric inductive bias of MeshGraphNets, while preserving a mesh-based graph representation. MGN-T overcomes a key limitation of standard MGN, the inefficient long-range information propagation caused by iterative message passing on large, high-resolution meshes. A physics-attention Transformer serves as a global processor, updating all nodal states simultaneously while explicitly retaining node and edge attributes. By directly capturing long-range physical interactions, MGN-T eliminates the need for deep message-passing stacks or hierarchical, coarsened meshes, enabling efficient learning on high-resolution meshes with varying geometries, topologies, and boundary conditions at an industrial scale.   We demonstrate that MGN-T successfully handles industrial-scale meshes for impact dynamics, a set
    
[^312]: 使用几何潜在子空间对离散数据进行生成建模

    Generative Modeling of Discrete Data Using Geometric Latent Subspaces

    [https://arxiv.org/abs/2601.21831](https://arxiv.org/abs/2601.21831)

    该论文提出一种几何潜在子空间框架，在类别分布乘积流形的指数参数空间中通过几何主成分分析（GPCA）学习高维离散数据的低维表示，并借助等距黎曼几何实现一致的流匹配生成建模。

    

    我们提出了一种用于离散数据生成建模的几何潜在子空间框架。具体而言，我们在类别分布乘积流形的指数参数空间中引入潜在子空间，作为学习高维离散数据低维表示的一种新方法。由此得到的低维潜在空间能够捕获统计依赖关系，并去除类别变量之间冗余的自由度。我们为参数域配备了黎曼几何，使得潜在子空间与所诱导的数据流形之间保持等距关系，从而实现一致的流匹配。利用这一结构，我们提出了一种几何感知的降维目标，称为几何主成分分析（GPCA），并将其表述为一种正则化的交叉熵最小化，以鼓励数据与其重构之间的黎曼距离尽可能小。特别是，在所诱导的……（摘要原文在此处截断）

    arXiv:2601.21831v3 Announce Type: replace  Abstract: We propose a geometric latent-subspace framework for generative modeling of discrete data. Specifically, we introduce latent subspaces in the exponential parameter space of product manifolds of categorical distributions as a novel approach to learning low-dimensional representations of high-dimensional discrete data. The resulting low-dimensional latent space captures statistical dependencies and removes redundant degrees of freedom among the categorical variables. We equip the parameter domain with a Riemannian geometry such that the latent subspace and induced data manifold are related isometrically, enabling consistent flow matching. Exploiting this structure, we propose a geometry-aware dimensionality reduction objective, called geometric PCA (GPCA), which we formulate as a regularized cross-entropy minimization that encourages small Riemannian distances between the data and their reconstructions. In particular, under the induced
    
[^313]: DeepFedNAS：基于Pareto引导超网训练的异构物联网联邦高效硬件感知架构自适应

    DeepFedNAS: Efficient Hardware-Aware Architecture Adaptation for Heterogeneous IoT Federations via Pareto-Guided Supernet Training

    [https://arxiv.org/abs/2601.15127](https://arxiv.org/abs/2601.15127)

    DeepFedNAS提出了一种基于多目标适应度函数的两阶段联邦神经架构搜索框架，通过Pareto最优超网训练与免预测器搜索，为异构物联网设备高效生成硬件感知的定制化网络架构并大幅降低搜索成本。

    

    在异构物联网设备集群上部署联邦学习需要为每类设备量身定制神经网络架构，然而现有的联邦神经架构搜索方法存在超网训练缺乏引导以及训练后搜索流程代价过高的问题——这些方法需要验证数千个子网来构建学习型精度预测器。我们提出了DeepFedNAS，这是一个建立在多目标适应度函数之上的两阶段框架，该函数将信息论网络指标与架构启发式方法相融合。在第一阶段，联邦Pareto最优超网训练用预先计算的精英高适应度架构缓存取代了随机子网采样，从而训练出性能更优的超网。在第二阶段，无预测器搜索直接将结构适应度函数用作精度代理，无需构建学习型子网精度预测器。在我们的CIFAR-10基准测试中，准备基准……（摘要原文被截断）

    arXiv:2601.15127v4 Announce Type: replace  Abstract: Deploying federated learning across heterogeneous IoT device fleets requires tailored neural network architectures for each device class, yet existing Federated Neural Architecture Search (FedNAS) methods suffer from unguided supernet training and prohibitively costly post-training search pipelines that validate thousands of subnets to construct learned accuracy predictors. We introduce DeepFedNAS, a two-phase framework built on a multi-objective fitness function that synthesizes information-theoretic network metrics with architectural heuristics. In the first phase, Federated Pareto Optimal Supernet Training replaces random subnet sampling with a pre-computed cache of elite, high-fitness architectures, yielding a superior supernet. In the second phase, a Predictor-Free Search uses the structural fitness function as an accuracy proxy without constructing a learned subnet-accuracy predictor. In our CIFAR-10 benchmark, preparing the ba
    
[^314]: FLAME：用于通用时间序列预测的流增强勒让德记忆模型

    FLAME: Flow Enhanced Legendre Memory Models for General Time Series Forecasting

    [https://arxiv.org/abs/2512.14253](https://arxiv.org/abs/2512.14253)

    FLAME是一种轻量级时间序列基础模型，通过在编解码阶段结合平移与缩放勒让德记忆变体来捕获数据归纳偏置并实现高效长程推理，并采用基于归一化流的预测头以生成式方式建模复杂分布，在多个基准测试中实现了高效且准确的通用时间序列预测。

    

    在这项工作中，我们提出了FLAME，这是一系列极其轻量且功能强大的时间序列基础模型，它通过生成式概率建模支持多种预测任务，同时确保了效率和鲁棒性。FLAME利用勒让德记忆来实现强大的泛化能力。通过在编码和解码阶段采用勒让德记忆的变体，即平移勒让德和缩放勒让德，FLAME能够有效捕获数据中固有的归纳偏置，并进行高效的长程推理。为了在保持高效的同时提高概率预测的准确性，FLAME采用了基于归一化流的预测头，该预测头能够以生成的方式对预测时域内的复杂分布进行建模。在TSFM-Bench、ProbTS和TFB三个广受认可的基准测试上进行的全面实验表明，FLAME是一款强大的开箱即用工具。

    arXiv:2512.14253v4 Announce Type: replace  Abstract: In this work, we introduce FLAME, a family of extremely lightweight and capable Time Series Foundation Models, which support versatile forecasting tasks via generative probabilistic modeling, while ensuring both efficiency and robustness. FLAME utilizes the Legendre Memory for strong generalization capabilities. By adapting variants of Legendre Memory, i.e., translated Legendre (LegT) and scaled Legendre (LegS), in the Encoding and Decoding phases, FLAME can effectively capture the inherent inductive bias within data and make efficient long-range inferences. To enhance the accuracy of probabilistic forecasting while remaining efficient, FLAME adopts a normalizing-flow-based forecasting head, which can model complex distributions over the forecasting horizon in a generative manner. Comprehensive experiments on three well-recognized benchmarks, including TSFM-Bench, ProbTS, and TFB, demonstrate that FLAME is a strong out-of-the-box too
    
[^315]: 演示：生成式AI基于用户偏好辅助放射治疗计划设计

    Demo: Generative AI helps Radiotherapy Planning with User Preference

    [https://arxiv.org/abs/2512.08996](https://arxiv.org/abs/2512.08996)

    本文提出一种仅依据用户自定义偏好即可预测三维剂量分布的生成式模型，使计划制定者能够个性化权衡危及器官与靶区之间的取舍，并在适应性上超越Varian RapidPlan。

    

    放射治疗计划是一个高度复杂的过程，在不同机构和不同计划制定者之间往往存在显著差异。现有的大多数用于三维剂量预测的深度学习方法在训练时依赖参考计划作为真值（ground truth），这可能会无意中使模型偏向特定的计划风格或机构偏好。在本研究中，我们提出了一种新颖的生成式模型，仅根据用户自定义的偏好风格来预测三维剂量分布。这些可定制的偏好使计划制定者能够优先考虑危及器官（OARs）与计划靶区（PTVs）之间的特定权衡，提供了更大的灵活性和个性化。我们的方法旨在与临床治疗计划系统无缝集成，帮助用户高效地生成高质量计划。对比评估表明，我们的方法在适应性方面可以超越Varian RapidPlan模型。

    arXiv:2512.08996v2 Announce Type: replace-cross  Abstract: Radiotherapy planning is a highly complex process that often varies significantly across institutions and individual planners. Most existing deep learning approaches for 3D dose prediction rely on reference plans as ground truth during training, which can inadvertently bias models toward specific planning styles or institutional preferences. In this study, we introduce a novel generative model that predicts 3D dose distributions based solely on user-defined preference flavors. These customizable preferences enable planners to prioritize specific trade-offs between organs-at-risk (OARs) and planning target volumes (PTVs), offering greater flexibility and personalization. Designed for seamless integration with clinical treatment planning systems, our approach assists users in generating high-quality plans efficiently. Comparative evaluations demonstrate that our method can surpasses the Varian RapidPlan model in both adaptability
    
[^316]: 模型达人秀：无需训练即可甄别高性能的可穿戴人体活动识别模型

    Models Got Talent: Identifying High Performing Wearable Human Activity Recognition Models Without Training

    [https://arxiv.org/abs/2511.06157](https://arxiv.org/abs/2511.06157)

    本研究证明零成本代理（ZCPs）可以在六个可穿戴人体活动识别基准数据集上，无需完整训练就能高效甄别出接近最优的模型架构，预测出的顶尖架构性能与大规模随机训练结果的差距不超过7%。

    

    为基于可穿戴设备的人体活动识别（HAR）应用寻找高性能的模型架构极具挑战性。由于传感器佩戴位置、记录设备、活动类型等因素带来的显著多样性和可变性，成熟的架构在并非为其专门设计的数据集或任务上可能表现不佳。作为神经架构搜索（NAS）的一种有前景的补充方案，零成本代理（Zero Cost Proxies, ZCPs）与实际训练后的性能具有良好的相关性，且仅需在随机采样的数据批次上进行一次前向/反向传播即可完成计算。本文在六个基准HAR数据集上研究了八种ZCPs的有效性，并证明被预测为最优的架构所取得的性能与对2000个随机采样架构进行完整训练所能达到的性能相差不超过7%。此外，训练预测排名前十的架构所获得的性能……（原文摘要在此截断）

    arXiv:2511.06157v3 Announce Type: replace-cross  Abstract: Discovering high performing model architectures for wearables-based Human Activity Recognition (HAR) applications is challenging. The astonishing diversity and variability due to differing sensor locations, recording apparatus, activities, etc., can cause established architectures to perform worse on datasets/tasks they were not designed for. A promising complement to Neural Architecture Search (NAS) involves the development of Zero Cost Proxies (ZCPs), which correlate well with trained performance, but can be computed through a single forward/backward pass on a randomly sampled batch of data. In this paper, we investigate the effectiveness of eight ZCPs on six benchmark HAR datasets, and demonstrate that the top-predicted architectures obtain performance within 7% of that attained by full-scale training of 2,000 randomly sampled architectures. Furthermore, training the top-10 predicted architectures results in performance with
    
[^317]: Transformers 在无图先验的情况下发现分子结构

    Transformers Discover Molecular Structure Without Graph Priors

    [https://arxiv.org/abs/2510.02259](https://arxiv.org/abs/2510.02259)

    该研究表明，Transformer模型无需嵌入图结构或几何局部性等物理先验，仅通过数据学习就能自动发现分子结构等物理模式。

    

    计算模拟在科学发现中扮演着核心角色，而机器学习（ML）已成为传统基于物理建模的一种有前景的替代方案。然而，科学建模需要具有物理意义的预测，这为数据驱动方法提出了一个根本性问题：物理归纳偏置——即关于物理世界结构的先验假设——能在多大程度上仅通过从数据中学习而涌现？例如，在原子建模领域，机器学习架构历来都嵌入了强物理归纳偏置——如几何局部性和图结构——其依据是假设这些先验对于物理预测是必不可少的。我们系统地研究了物理模式如何能够直接从数据中被发现：通过训练一个不含领域特定先验的模型，其中包括任何手动定义的原子间成对相互作用。我们发现……

    arXiv:2510.02259v2 Announce Type: replace  Abstract: Computational simulations play a central role in scientific discovery, and machine learning (ML) has emerged as a promising alternative to traditional physics-based modeling. However, scientific modeling requires physically meaningful predictions, raising a fundamental question for data-driven methods: to what extent can physical inductive biases - that is, prior assumptions about the structure of the physical world - emerge by learning from data alone? In atomistic modeling, for example, ML architectures have historically embedded strong physical inductive biases - such as geometric locality and graph structure - based on the assumption that these priors are necessary for physical predictions. We systematically develop an understanding of how physical patterns can alternatively be discovered directly from data by training a model without domain-specific priors, including any manually defined atomistic pairwise interactions. We find 
    
[^318]: 它们修复了什么？利用大语言模型辅助对关键内存漏洞的安全补丁进行分类

    What Do They Fix? LLM-Aided Categorization of Security Patches for Critical Memory Bugs

    [https://arxiv.org/abs/2509.22796](https://arxiv.org/abs/2509.22796)

    该论文提出利用大语言模型对Linux内核中修复关键内存漏洞（如越界访问和释放后使用）的安全补丁进行自动分类，以解决安全补丁难以识别、下游维护者采用延迟的问题。

    

    开源软件项目是现代软件生态系统的基础，其中Linux内核因其普遍性和复杂性而成为关键典范。尽管安全补丁不断被集成到Linux主线内核中，但下游维护者往往延迟采用这些补丁，从而造成漏洞暴露窗口期。造成这种延迟的一个关键原因是难以识别安全关键补丁，特别是那些针对可利用漏洞（如越界（OOB）访问和释放后使用（UAF）漏洞）的补丁。由于有意保持静默的漏洞修复、不完整或缺失的CVE编号分配、CVE发布的延迟，以及Linux内核CVE分配标准的近期变更，这一挑战更加严峻。虽然存在细粒度的补丁分类方法，但它们在覆盖范围和准确性方面都存在局限。在这项工作中，我们识别了此前未被探索的机会，以显著……

    arXiv:2509.22796v2 Announce Type: replace-cross  Abstract: Open-source software projects are foundational to modern software ecosystems, with the Linux kernel standing out as a critical exemplar due to its ubiquity and complexity. Although security patches are continuously integrated into the Linux mainline kernel, downstream maintainers often delay their adoption, creating windows of vulnerability. A key reason for this lag is the difficulty in identifying security-critical patches, particularly those addressing exploitable vulnerabilities such as out-of-bounds (OOB) accesses and use-after-free (UAF) bugs. This challenge is exacerbated by intentionally silent bug fixes, incomplete or missing CVE assignments, delays in CVE issuance, and recent changes to the CVE assignment criteria for the Linux kernel. While fine-grained patch classification approaches exist, they exhibit limitations in both coverage and accuracy. In this work, we identify previously unexplored opportunities to signif
    
[^319]: AUWave：一种利用稀疏观测重建有效波高的数据驱动模型

    AUWave: A Data-Driven Model for Reconstructing Significant Wave Heights Using Sparse Observations

    [https://arxiv.org/abs/2509.19384](https://arxiv.org/abs/2509.19384)

    提出了AUWave混合深度学习框架，结合站点编码器与自注意力增强的多尺度U-Net，从稀疏浮标观测中高精度重建区域有效波高场，并通过浮标消融分析识别关键站点以指导海洋观测网络设计。

    

    从稀疏的浮标观测中重建高分辨率区域有效波高（SWH）场是海洋监测中的一项关键挑战。我们提出了AUWave，这是一个混合深度学习框架，它将基于站点的编码器与通过自注意力机制增强的多尺度U-Net相融合，以恢复区域有效波高场。AUWave利用夏威夷地区的NDBC浮标观测数据和ERA5再分析数据进行训练和验证，实现了较高的精度。它始终优于一个代表性基线模型，尤其是在配置多于单个浮标的情形下，展示了其多尺度架构的优势。空间误差分析表明，正如预期的那样，模型性能在观测站点附近最高。此外，浮标消融研究识别出了关键的锚定站点，这些站点一旦被移除会导致性能不成比例地下降，从而为观测网络设计提供了可操作的指导。AUWave为填补数据空白提供了一条可扩展的路径。

    arXiv:2509.19384v2 Announce Type: replace-cross  Abstract: Reconstructing high-resolution regional significant wave height (SWH) fields from sparse buoy observations is a critical challenge for ocean monitoring. We introduce AUWave, a hybrid deep learning framework that fuses a station-wise encoder with a multi-scale U-Net enhanced by self-attention to recover regional SWH fields. Trained and validated using NDBC buoy observations and ERA5 reanalysis over the Hawaii region, AUWave achieves high accuracy. It consistently outperforms a representative baseline, especially in configurations with more than a single buoy, demonstrating the benefit of its multi-scale architecture. Spatial error analysis shows performance is highest near observation sites, as expected. Further, buoy ablation studies identify critical anchor stations whose removal disproportionately degrades performance, offering actionable guidance for observational network design. AUWave provides a scalable pathway for gap-fi
    
[^320]: 重尾噪声下的随机双层优化

    Stochastic Bilevel Optimization with Heavy-Tailed Noise

    [https://arxiv.org/abs/2509.14952](https://arxiv.org/abs/2509.14952)

    本文提出了一种针对重尾噪声下随机双层优化的嵌套循环归一化随机双层近似算法（N²SBA），在噪声中心矩阶为 p∈(1,2] 的条件下，以 O(κ^((7p-3)/(p-1)) σ^(p/(p-1)) ε^(-(4p-2)/(p-1))) 的随机一阶预言机复杂度找到 ε-稳定点，并将该方法推广至非凸-强凹极小极大优化问题。

    

    本文考虑光滑双层优化问题，其中下层问题是强凸的，而上层问题可能是非凸的。我们关注随机设置，即算法可以访问带有重尾噪声的无偏随机梯度评估，这种噪声在许多机器学习应用中普遍存在，例如大型语言模型的训练和强化学习。我们提出了一种嵌套循环归一化随机双层近似方法，用于寻找 ε-稳定点，其随机一阶预言机复杂度为 $\tilde{\mathcal{O}}\big(\kappa^{\frac{7p-3}{p-1}} \sigma^{\frac{p}{p-1}} \epsilon^{-\frac{4 p - 2}{p-1}}\big)$，其中 $\kappa$ 是条件数，$p\in(1,2]$ 是噪声中心矩的阶数，$\sigma$ 是噪声水平。此外，我们将这一思想专门应用于求解非凸-强凹的极小极大优化问题。

    arXiv:2509.14952v3 Announce Type: replace  Abstract: This paper considers the smooth bilevel optimization in which the lower-level problem is strongly convex and the upper-level problem is possibly nonconvex. We focus on the stochastic setting where the algorithm can access the unbiased stochastic gradient evaluation with heavy-tailed noise, which is prevalent in many machine learning applications, such as training large language models and reinforcement learning. We propose a nested-loop normalized stochastic bilevel approximation (N$^2$SBA) for finding an $\epsilon$-stationary point with the stochastic first-order oracle (SFO) complexity of $\tilde{\mathcal{O}}\big(\kappa^{\frac{7p-3}{p-1}} \sigma^{\frac{p}{p-1}} \epsilon^{-\frac{4 p - 2}{p-1}}\big)$, where $\kappa$ is the condition number, $p\in(1,2]$ is the order of central moment for the noise, and $\sigma$ is the noise level. Furthermore, we specialize our idea to solve the nonconvex-strongly-concave minimax optimization problem,
    
[^321]: 神经桥过程

    Neural Bridge Processes

    [https://arxiv.org/abs/2508.07220](https://arxiv.org/abs/2508.07220)

    提出神经桥过程（NBP），用输入锚定的桥轨迹替代无条件前向核，使条件输入信息在扩散的含噪状态中就被编码，从而实现对随机函数更具表达力且强条件依赖的学习。

    

    从部分观测的上下文-目标对中学习随机函数，需要模型具备强表达能力、不确定性感知能力以及对输入的强条件依赖。神经扩散过程（NDPs）通过去噪扩散提升了表达能力，但其前向过程与输入无关；输入仅进入反向去噪器，因此含噪的训练状态本身并不编码条件输入信息。我们提出神经桥过程（NBPs），用输入锚定的桥轨迹替代无条件的前向核。当输入与输出维度不同时，NBP学习一个输出空间锚点 $a_\psi(x)=P_\psi(x)$，使坐标或其他输入能够引导生成路径，而无需改变去噪主干网络。我们从理论上证明，过程级锚定诱导了逐路径的输入可区分性，将关于 x 的信息注入含噪状态，并创建了传统方法中不可得的直接梯度通路。

    arXiv:2508.07220v4 Announce Type: replace-cross  Abstract: Learning stochastic functions from partially observed context-target pairs requires models that are expressive, uncertainty-aware, and strongly conditioned on inputs. Neural Diffusion Processes (NDPs) improve expressivity with denoising diffusion, but their forward process is input-independent; inputs only enter the reverse denoiser, so the noisy training states themselves do not encode the conditioning inputs. We propose Neural Bridge Processes (NBPs), which replace the unconditional forward kernel with an input-anchored bridge trajectory. When input and output dimensions differ, NBP learns an output-space anchor $a_\psi(x)=P_\psi(x)$, allowing coordinates or other inputs to guide the generative path without changing the denoising backbone. We show theoretically that process-level anchoring induces pathwise input distinguishability, injects information about x into noisy states, and creates a direct gradient pathway unavailabl
    
[^322]: DeepC4：用于城市形态大规模多任务空间分解的深度条件化人口普查约束聚类

    DeepC4: Deep Conditional Census-Constrained Clustering for Large-scale Multitask Spatial Disaggregation of Urban Morphology

    [https://arxiv.org/abs/2507.22554](https://arxiv.org/abs/2507.22554)

    本文提出DeepC4，一种基于深度学习的人口普查约束聚类方法，用于在弱监督条件下解决城市形态大规模多任务空间分解中与人口普查数据的局部差异及模型不确定性传播的难题。

    

    为了解许多发展中经济体在可持续发展和灾害风险减缓方面的全球进展，近期两项重大倡议——全球地震模型（GEM）基金会的非洲统一暴露数据集以及通过地球观测程序建模暴露（METEOR）项目——采用了经典的空间分解技术，利用各类卫星影像及其衍生产品、建筑环境地理空间数据集以及次国家级人口普查统计数据，生成大规模的城市形态制图。然而，在此类从粗粒度到细粒度的制图问题中，与经过充分验证的人口普查统计数据之间的局部差异以及模型不确定性的传播仍然是一个挑战，尤其是受到弱且带有条件约束的标签监督的限制。因此，我们提出了深度条件化人口普查约束聚类，这是一种新颖的基于深度学习的空间分解方法……

    arXiv:2507.22554v4 Announce Type: replace  Abstract: To understand our global progress for sustainable development and disaster risk reduction in many developing economies, two recent major initiatives - the Uniform African Exposure Dataset of the Global Earthquake Model (GEM) Foundation and the Modelling Exposure through Earth Observation Routines (METEOR) Project - implemented classical spatial disaggregation techniques to generate large-scale mapping of urban morphology using the information from various satellite imagery and its derivatives, geospatial datasets of the built environment, and subnational census statistics. However, the local discrepancy with well-validated census statistics and the propagated model uncertainties remain a challenge in such coarse-to-fine-grained mapping problems, specifically constrained by weak and conditional label supervision. Therefore, we present Deep Conditional Census-Constrained Clustering (DeepC4), a novel deep learning-based spatial disaggre
    
[^323]: LapDDPM：用于鲁棒单细胞流形生成的谱扰动扩散模型

    LapDDPM: Spectral Perturbation Diffusion for Robust Single-Cell Manifold Generation

    [https://arxiv.org/abs/2506.13344](https://arxiv.org/abs/2506.13344)

    提出LapDDPM，一种结合谱对抗扰动机制的条件图扩散概率模型，通过谱扰动实现分布鲁棒优化，能够生成高保真且对结构噪声鲁棒的单细胞RNA测序数据。

    

    生成高保真且生物学上合理的合成单细胞RNA测序数据是计算生物学中的一个关键挑战，这源于对高维、稀疏且非线性细胞流形建模的需求。现有的生成模型往往无法捕捉细胞分化的复杂拓扑结构，或对技术噪声和结构变异缺乏鲁棒性。我们提出了LapDDPM，这是一种新颖的条件图扩散概率模型，旨在实现鲁棒的流形学习和高保真生成。LapDDPM将基于图的归纳偏置与基于分数的生成建模相结合，并通过一种新颖的谱对抗扰动机制加以增强。通过在训练过程中沿主谱模式系统地扰动图边权重，我们的方法充当了一个分布鲁棒优化（DRO）框架，从而强制实现对结构噪声的不变性。我们进一步扩展……

    arXiv:2506.13344v2 Announce Type: replace-cross  Abstract: Generating high-fidelity and biologically plausible synthetic single-cell RNA sequencing (scRNA-seq) data is a critical challenge in computational biology, driven by the need to model high-dimensional, sparse, and non-linear cellular manifolds. Existing generative models often fail to capture the complex topology of cellular differentiation or lack robustness against technical noise and structural variability. We introduce LapDDPM, a novel conditional Graph Diffusion Probabilistic Model designed for robust manifold learning and high-fidelity generation. LapDDPM integrates graph-based inductive biases with score-based generative modeling, enhanced by a novel spectral adversarial perturbation mechanism. By systematically perturbing graph edge weights along principal spectral modes during training, our method acts as a Distributionally Robust Optimization (DRO) framework, enforcing invariance to structural noise. We further extend
    
[^324]: 评估即一切：通过评估设计对大语言模型推理能力的策略性夸大

    Evaluation is All You Need: Strategic Overclaiming of LLM Reasoning Capabilities Through Evaluation Design

    [https://arxiv.org/abs/2506.04734](https://arxiv.org/abs/2506.04734)

    本研究揭示评估条件的细微差异会导致Deepseek-R1-Distill系列等推理模型的基准测试结果大幅波动，使其声称的性能提升难以可靠复现，并倡导建立更严格的模型性能评估范式。

    

    以Deepseek-R1-Distill系列为代表的推理模型因其在数学、科学、编程等领域的出色表现而被开源社区广泛采用。然而，我们的研究揭示，其基准测试评估结果会受到多种因素的影响而产生显著波动，评估条件的细微差异就可能导致结果的重大变化。在基于Deepseek-R1-Distill系列微调的其他开源推理模型以及QwQ-32B模型中也观察到类似现象，使得其所声称的性能提升难以可靠复现。因此，我们倡导建立更严格的模型性能评估范式，并呈现了我们对Deepseek-R1-Distill系列模型的实证评估。

    arXiv:2506.04734v3 Announce Type: replace  Abstract: Reasoning models represented by the Deepseek-R1-Distill series have been widely adopted by the open-source community due to their strong performance in mathematics, science, programming, and other domains. However, our study reveals that their benchmark evaluation results are subject to significant fluctuations caused by various factors. Subtle differences in evaluation conditions can lead to substantial variations in results. Similar phenomena are observed in other open-source inference models fine-tuned based on the Deepseek-R1-Distill series, as well as in the QwQ-32B model, making their claimed performance improvements difficult to reproduce reliably. Therefore, we advocate for the establishment of a more rigorous paradigm for model performance evaluation and present our empirical assessments of the Deepseek-R1-Distill series models.
    
[^325]: ChemMLLM：化学多模态大语言模型

    ChemMLLM: Chemical Multimodal Large Language Model

    [https://arxiv.org/abs/2505.16326](https://arxiv.org/abs/2505.16326)

    提出了ChemMLLM，一个统一的化学多模态大语言模型，能够同时处理分子理解与生成任务，首次将化学多模态大模型的能力扩展到图像生成领域。

    

    近年来，多模态大语言模型（MLLMs）在化学领域取得了快速进展。然而，能够处理跨模态理解与生成的化学多模态大语言模型仍未得到充分探索。为填补这一空白，我们提出了ChemMLLM，一个用于分子理解与生成的统一化学多模态大语言模型。在这项工作中，我们设计了五种跨文本、分子SMILES字符串和图像的多模态任务，并整理了相应的数据集。我们在这些任务上将ChemMLLM与一系列领先的通用多模态大语言模型、化学大语言模型和专用模型进行了基准测试。实验结果表明，ChemMLLM在所有评估任务中的表现均优于通用多模态大语言模型，且接近专用模型的性能。我们的工作将化学多模态大语言模型的能力扩展到了图像生成领域，展示了统一多种跨模态化学任务的可行性。

    arXiv:2505.16326v3 Announce Type: replace  Abstract: Recent years have seen rapid progress in multimodal large language models (MLLMs) in the field of chemistry. However, chemical MLLMs that can handle cross-modal understanding and generation remain underexplored. To fill this gap, we propose ChemMLLM, a unified chemical multimodal large language model for molecule understanding and generation. In this work, we design five types of multimodal tasks across text, molecular SMILES strings and images, and curate the datasets. We benchmark ChemMLLM against a range of general leading MLLMs, Chemical LLMs and specialized models on these tasks. Experimental results show that ChemMLLM achieves superior performance among general-purpose MLLMs and close performance to specialized models across all evaluated tasks. Our work extends the capabilities of chemical multimodal large language models to the realm of image generation, demonstrating the feasibility of unifying multiple cross-modal chemical 
    
[^326]: 基于算子值核的正则化随机梯度下降学习算子

    Learning Operators by Regularized Stochastic Gradient Descent with Operator-valued Kernels

    [https://arxiv.org/abs/2504.18184](https://arxiv.org/abs/2504.18184)

    本文针对从波兰空间到可分希尔伯特空间的算子学习问题，分析了算子值核再生核希尔伯特空间中在线与有限时域两种设置下的正则化随机梯度下降算法，建立了对输出空间维度无显式依赖、且在期望意义下接近最优的误差界，并给出了高概率估计与几乎必然收敛的结论。

    

    我们考虑一类统计逆问题，即估计从波兰空间到可分希尔伯特空间的回归算子，其中目标函数位于由算子值核诱导的向量值再生核希尔伯特空间中。为了解决相关的病态性问题，我们分析了在线和有限时域两种设置下的正则化随机梯度下降（SGD）算法：前者使用多项式衰减的步长和正则化参数，后者采用固定值。在适当的结构和分布假设下，我们建立了预测误差和估计误差界，且对输出空间维度没有显式依赖。所得到的收敛率在期望意义下是接近最优的，我们还推导了高概率估计，这意味着几乎必然收敛。我们的分析引入了一种通用的技术，用于获得高概率保证

    arXiv:2504.18184v5 Announce Type: replace  Abstract: We consider a class of statistical inverse problems involving the estimation of a regression operator from a Polish space to a separable Hilbert space, where the target lies in a vector-valued reproducing kernel Hilbert space induced by an operator-valued kernel. To address the associated ill-posedness, we analyze regularized stochastic gradient descent (SGD) algorithms in both online and finite-horizon settings. The former uses polynomially decaying step sizes and regularization parameters, while the latter adopts fixed values. Under suitable structural and distributional assumptions, we establish prediction and estimation error bounds with no explicit dependence on the dimension of the output space. The resulting convergence rates are near-optimal in expectation, and we also derive high-probability estimates that imply almost sure convergence. Our analysis introduces a general technique for obtaining high-probability guarantees in 
    
[^327]: AYLA：在浅层神经网络中架构损失景观以加速特征恢复

    AYLA: Architecting a loss landscape in shallow neural networks to accelerate feature recovery

    [https://arxiv.org/abs/2504.01875](https://arxiv.org/abs/2504.01875)

    AYLA是一种损失重参数化框架，通过对损失施加sigmoid控制的幂律变换动态调节梯度大小，在不改变临界点位置的前提下，加速浅层神经网络特征学习在平坦和鞍点区域的下降，并稳定后期优化过程。

    

    浅层神经网络中的特征学习呈现出丰富而脆弱的动力学特性，包括长时间的停滞平台、突然的相变以及对优化超参数的敏感性。尽管近期的理论工作已通过损失景观的几何结构、鞍点逃逸机制和涌现的标度律对这些问题进行了刻画，但主动塑造这些动力学的实用方法仍然有限。在本文中，我们提出了AYLA，这是一个有理论依据的损失重参数化框架，它在训练过程中动态调节梯度大小，而不改变驻点或最优解的位置。AYLA对经验损失施加一个平滑的、由sigmoid控制的幂律变换，产生依赖于状态的有效学习率，从而在平坦或鞍点主导的区域加速下降，同时稳定优化后期的训练。至关重要的是，AYLA保留了原始目标函数的所有临界点。

    arXiv:2504.01875v4 Announce Type: replace  Abstract: Feature learning in shallow neural networks exhibits rich yet fragile dynamics, including prolonged plateaus, abrupt phase transitions, and sensitivity to optimization hyperparameters. While recent theoretical work has characterized these behaviors through the geometry of loss landscapes, saddle escape mechanisms, and emergent scaling laws, practical methods for actively shaping these dynamics remain limited. In this paper, we introduce AYLA, a principled loss reparameterization framework that dynamically modulates gradient magnitudes during training without altering the location of stationary points or optimal solutions. AYLA applies a smooth, sigmoid-controlled power-law transformation to empirical loss, yielding a state-dependent effective learning rate that accelerates descent in flat or saddle-dominated regions while stabilizing late-stage optimization. Crucially, AYLA preserves all critical points of the original objective, act
    
[^328]: LEAD：用于阿尔茨海默病检测的脑电图基础模型

    LEAD: An EEG Foundation Model for Alzheimer's Disease Detection

    [https://arxiv.org/abs/2502.01678](https://arxiv.org/abs/2502.01678)

    本文构建了迄今最大的EEG-AD数据集（2,238名受试者），并提出首个脑电图阿尔茨海默病检测基础模型LEAD，其门控时-空Transformer可适应异构EEG数据，配合被试正则化训练策略提升了跨被试泛化能力。

    

    脑电图（EEG）为检测阿尔茨海默病（AD）提供了一种无创、高度可及且经济高效的方法。然而，现有方法，无论是基于手工特征工程还是标准深度学习，都面临三大挑战：1）缺乏大规模基于EEG的AD数据集用于稳健的表示学习和评估；2）跨被试泛化能力有限；3）难以适应高度异构的数据。为应对这些挑战，我们构建了迄今世界上最大的EEG-AD语料库，包含2,238名受试者。利用这一独特资源，我们提出了LEAD，这是首个用于基于EEG的AD检测的基础模型。具体而言，我们设计了一种门控时-空Transformer，能够适应具有不同长度、通道配置和采样率的EEG记录。此外，我们引入了一种被试正则化训练策略，以增强端到端的子（摘要原文在此处截断）。

    arXiv:2502.01678v5 Announce Type: replace-cross  Abstract: Electroencephalography (EEG) provides a non-invasive, highly accessible, and cost-effective approach for detecting Alzheimer's disease (AD). However, existing methods, whether based on handcrafted feature engineering or standard deep learning, face three major challenges: 1) the lack of large-scale EEG-based AD datasets for robust representation learning and evaluation; 2) limited cross-subject generalizability; and 3) difficulty in adapting to highly heterogeneous data. To address these challenges, we curate the world's largest EEG-AD corpus to date, comprising 2,238 subjects. Leveraging this unique resource, we propose LEAD, the first foundation model for EEG-based AD detection. Specifically, we design a gated temporal-spatial Transformer that can adapt to EEG recordings with diverse lengths, channel configurations, and sampling rates. In addition, we introduce a subject-regularized training strategy to enhance end-to-end sub
    
[^329]: 强化学习与交互式决策的基础

    Foundations of Reinforcement Learning and Interactive Decision Making

    [https://arxiv.org/abs/2312.16730](https://arxiv.org/abs/2312.16730)

    该专著在统一的统计框架下系统阐述了从多臂老虎机到基于函数逼近的强化学习的交互式决策算法设计与复杂度理论，并展示了如何将监督学习方法转化为决策算法、分析其性能以及判定问题的可解性。

    

    交互式决策是指在未知环境中学习采取良好行动的问题，即利用自身行动所产生的数据来持续改进，这一场景广泛存在于在线平台、机器人技术和医疗治疗等领域。本专著从统计学视角探讨交互式决策的算法设计与复杂度问题，在一个统一的框架内，从多臂老虎机逐步延伸至上下文老虎机、结构化老虎机，再到具有函数逼近的强化学习。书中特别关注函数逼近以及神经网络等灵活模型，并着重阐述监督学习与决策之间的联系：读者将学会如何将任意监督学习方法转化为决策算法、如何分析所得结果，以及如何判断给定问题能否通过少量交互求解。一个贯穿全文的统一主题是……

    arXiv:2312.16730v2 Announce Type: replace-cross  Abstract: Interactive decision making is the problem of learning to act well in an unknown environment, using the data that one's own actions generate to continuously improve, and arises in situations ranging from online platforms and robotics to medical treatments. This monograph gives a statistical perspective on algorithm design and complexity for interactive decision making, building from multi-armed bandits through contextual and structured bandits to reinforcement learning with function approximation within a single, unified framework. Special attention is paid to function approximation and flexible models such as neural networks, and to the connection between supervised learning and decision making: the reader will learn how to turn any supervised learning method into a decision making algorithm, how to analyze the result, and how to determine whether a given problem can be solved with few interactions.   A unifying theme is that 
    
[^330]: 持久同调上的一种新的非阿基米德度量

    A New Non-archimedean Metric on Persistent Homology

    [https://arxiv.org/abs/2012.02655](https://arxiv.org/abs/2012.02655)

    本文提出了一种适用于所有维度持久同调类的新非阿基米德度量——共表型度量，并证明其与层次聚类结合能提供统计上可验证的等价拓扑信息，且所得聚类在轮廓系数和兰德指数等评估指标上表现优异。

    

    在本文中，我们在所有维度的持久同调类上定义了一种新的非阿基米德度量结构，称为共表型度量。随后，基于我们在不同数据集上获得的实验结果，我们证明了零维持久同调结合共表型度量，以及采用多种不同度量的层次聚类算法，确实能够提供统计上可验证的等价拓扑信息。我们还观察到，由共表型距离得到的聚类在轮廓系数和兰德指数等不同评估指标方面表现出色。此外，由于共表型度量是为所有同调维度定义的，现在可以通过有根树展示所有维度的持久同调类之间的相互关系。

    arXiv:2012.02655v4 Announce Type: cross  Abstract: In this article, we define a new non-archimedean metric structure, called cophenetic metric, on persistent homology classes of all degrees. We then show that zeroth persistent homology together with the cophenetic metric and hierarchical clustering algorithms with a number of different metrics do deliver statistically verifiable commensurate topological information based on experimental results we obtained on different datasets. We also observe that the resulting clusters coming from cophenetic distance do shine in terms of different evaluation measures such as silhouette score and the Rand index. Moreover, since the cophenetic metric is defined for all homology degrees, one can now display the inter-relations of persistent homology classes in all degrees via rooted trees.
    
[^331]: 快速图搜索算法与动态优化和减少直方图用于二分类问题的区分 (arXiv:2401.04282v1 [cs.LG])

    A Fast Graph Search Algorithm with Dynamic Optimization and Reduced Histogram for Discrimination of Binary Classification Problem. (arXiv:2401.04282v1 [cs.LG])

    [http://arxiv.org/abs/2401.04282](http://arxiv.org/abs/2401.04282)

    本研究提出了一种用于二分类问题的快速图搜索算法，通过动态优化和减少直方图的方法来提高区分结果。该算法在支持向量机模型的基础上应用，显著提高了真正例并减少了假正例。

    

    本研究开发了一种图搜索算法，用于找到二分类问题的最优区分路径。目标函数被定义为真正例（TP）和假正例（FP）之间变异性的差异。它使用深度优先搜索（DFS）算法来寻找自顶向下的区分路径。它提出了一种动态优化过程，以在上层优化TP，然后在下层减少FP。为了加速计算速度并提高准确性，它提出了一种带有可变箱大小的减小直方图算法，而不是循环遍历所有数据点，以找到区分的特征阈值。该算法应用于支持向量机（SVM）模型上，用于预测一个人是否健康。它显著提高了SVM结果的TP并减少了FP （例如，FP减少了90%，而TP仅损失了5%）。图搜索自动生成了39个排序的区分路径。

    This study develops a graph search algorithm to find the optimal discrimination path for the binary classification problem. The objective function is defined as the difference of variations between the true positive (TP) and false positive (FP). It uses the depth first search (DFS) algorithm to find the top-down paths for discrimination. It proposes a dynamic optimization procedure to optimize TP at the upper levels and then reduce FP at the lower levels. To accelerate computing speed with improving accuracy, it proposes a reduced histogram algorithm with variable bin size instead of looping over all data points, to find the feature threshold of discrimination. The algorithm is applied on top of a Support Vector Machine (SVM) model for a binary classification problem on whether a person is fit or unfit. It significantly improves TP and reduces FP of the SVM results (e.g., reduced FP by 90% with a loss of only\ 5% TP). The graph search auto-generates 39 ranked discrimination paths withi
    
[^332]: 差分隐私决策树与对数据篡改的可靠性证明

    Differentially-Private Decision Trees and Provable Robustness to Data Poisoning. (arXiv:2305.15394v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2305.15394](http://arxiv.org/abs/2305.15394)

    本论文提出了一种名为PrivaTree的差分隐私决策树方法，通过使用私有直方图选择分割点来在隐私保护与模型效用之间取得更好的平衡。这种方法能够接收混合的数值和类别数据，并且能够在数据篡改方面表现出可靠性。

    

    决策树是适用于非线性学习问题的可解释模型。关于将差分隐私引入决策树学习算法的研究已经很多，差分隐私能够确保训练数据中样本的隐私性。然而，目前用于此目的的最先进算法在获得一点点隐私保护的同时牺牲了较多的模型效用。这些解决方案引入了随机决策节点，降低了决策树的准确性，或者在标记叶子节点上使用过多的隐私预算。此外，很多方法不支持连续特征或者泄露与连续特征相关的信息。我们提出了一种基于私有直方图的新方法，称为PrivaTree，它在消耗一小部分隐私预算的同时选择合适的分割点。由此产生的决策树在隐私效用权衡方面取得了显著的提升，而且能够接受混合的数值和类别数据而不泄露与数值特征相关的信息。最后，尽管给出可靠性保证一直很难，我们的方法在数据篡改方面表现出了可靠性。

    Decision trees are interpretable models that are well-suited to non-linear learning problems. Much work has been done on extending decision tree learning algorithms with differential privacy, a system that guarantees the privacy of samples within the training data. However, current state-of-the-art algorithms for this purpose sacrifice much utility for a small privacy benefit. These solutions create random decision nodes that reduce decision tree accuracy or spend an excessive share of the privacy budget on labeling leaves. Moreover, many works do not support continuous features or leak information about them. We propose a new method called PrivaTree based on private histograms that chooses good splits while consuming a small privacy budget. The resulting trees provide a significantly better privacy-utility trade-off and accept mixed numerical and categorical data without leaking information about numerical features. Finally, while it is notoriously hard to give robustness guarantees a
    

