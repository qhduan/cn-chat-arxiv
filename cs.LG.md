# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [QGPINNs: A Physics-Informed Neural Network Framework for Nonlocal Differential Equations on Quantum Graphs](https://arxiv.org/abs/2608.28589) | 提出了QGPINNs——一个基于PyTorch的物理信息神经网络框架，通过逐边神经网络逼近和统一的图损失函数（融合连续性、Kirchhoff-Neumann顶点条件与Dirichlet边界条件），实现量子图上多阶分数阶椭圆问题和时间分数阶演化方程等非局部微分方程的数值求解。 |
| [^2] | [Aero Hand Open: A Simulation-Ready Tendon-Driven Hand for Dexterous Manipulation Learning](https://arxiv.org/abs/2608.28578) | 提出了Aero Hand Open——一款仿真就绪的腱驱动拟人灵巧手，附带可复现缆绳传动的仿真模型和双向辨识执行映射，解决了腱驱动手在灵巧操作学习中的仿真建模难题。 |
| [^3] | [Learning a Size-Weight Frontier for Synthetic-Augmented Inference](https://arxiv.org/abs/2608.28576) | 提出合成增强推断框架，通过从历史任务中学习“规模-权重前沿”，为所有规模-权重配置提供有限样本覆盖保证，在真实数据稀缺时安全利用合成数据并显著收窄置信区间。 |
| [^4] | [On two proofs of $d^2$ mixing of weighted Dikin walks](https://arxiv.org/abs/2608.28566) | 该论文提出了两种证明方法——在高概率区域上控制Metropolis-Hastings接受概率以及基于新的四阶自举条件的 $\chi^2$-散度分析——证明了加权Dikin行走从多面体采样可在 $\widetilde O(d^2)$ 步内实现混合。 |
| [^5] | [Learning between the peaks: sharp asymptotics for kernel ridge regression under power-law anisotropy](https://arxiv.org/abs/2608.28564) | 该论文针对幂律各向异性高斯数据下的核岭回归，推导出核谱与泛化误差的渐近精确表达式，揭示了弱各向异性会使方差峰值随 α 增大而逐渐衰减、且与主方向对齐的目标的偏差在分数样本复杂度处下降并与插值峰值解耦，而强各向异性（α>1）则会改变有效维数。 |
| [^6] | [Blog: Survey of Optimizers](https://arxiv.org/abs/2608.28557) | 本综述从时间估计、更新几何、周期管理和表示与系统四个维度系统梳理了2025-2026年的优化器进展，指出以Muon、Shampoo、SOAP为代表的矩阵感知方法是真正的进步，但尚不存在适用于所有场景的通用最优优化器。 |
| [^7] | [Advancing Interaction-Sensitive Feature Selection: Novel Relief-Based Algorithms, Expanded Comparisons, and Recommendations for Biomedical Data Mining](https://arxiv.org/abs/2608.28552) | 本研究重构并扩展了scikit-rebate Python包，提出了5种新型基于Relief的交互敏感特征选择算法变体，并通过多样化的基因组模拟基准测试为生物医学数据挖掘提供了比较结果与实用建议。 |
| [^8] | [DARTS: Decoder-Aware Representation Tuning via Surgery for Model Merging](https://arxiv.org/abs/2608.28547) | 该论文提出DARTS方法，首次分析解码器模型合并中的表示偏差问题，指出因果注意力掩码导致偏差跨位置累积以及高熵决策关键位置更为重要这两大挑战，并通过手术式的解码器感知表示微调来校正偏差。 |
| [^9] | [An Enclosed Mode Is a Gauge Choice: Topology Relative to Reach in Certified Code World Models](https://arxiv.org/abs/2608.28541) | 该论文提出“危险是相对于可达性的拓扑”这一核心原则，证明认证代码世界模型中超出可达范围的封闭错误模式是一种规范选择——错误拓扑的伪影在不可达时不可证伪且无害，而一旦通过宽度为γ的通道变得可达，就会依次经历可证伪且有代价、立即被证伪等状态。 |
| [^10] | [REPLICANT: Learning Policies for Evading and Hardening Malware Detectors](https://arxiv.org/abs/2608.28499) | 提出深度强化学习框架Replicant，在严格的仅标签黑盒威胁模型下学习可跨样本、检测器和特征空间迁移的恶意软件规避策略，在七个Android恶意软件检测器上实现78.8%的平均攻击成功率，较现有最优方法相对提升20.9%至39.2%。 |
| [^11] | [How Proper Scoring Rules Shape LLM Forecasting](https://arxiv.org/abs/2608.28482) | 尽管五种适当的评分规则在理论上都激励真实概率报告，但作为大语言模型预测的训练目标时，会训练出在校准、偏差、信息和噪声特征上各不相同的模型，表明奖励函数的选择并非可以互换。 |
| [^12] | [Acquire, Repair, Preserve: A Diagnosis-Guided Post-Training Recipe for Small-Model Dialogue Game Agents](https://arxiv.org/abs/2608.28458) | 该论文提出一种诊断引导的三步后训练方案（获取、修复、保持），使2B小型模型在对话游戏挑战中的clemscore从10.67大幅提升至38.92，同时保持了一般能力不退化。 |
| [^13] | [Generalized Splines and Gaussian Processes](https://arxiv.org/abs/2608.28446) | 本章将有限维高斯线性逆问题中“最小均方误差估计等价于正则化最小二乘拟合”的经典结论推广到无穷维设定，建立了广义样条与核空间上广义高斯过程之间的对应等价关系。 |
| [^14] | [Sliding-window beats linear attention](https://arxiv.org/abs/2608.28444) | 本研究表明，带sink的滑动窗口注意力（SWA）在多项下游任务和长上下文推理任务上的表现与经过后训练的线性注意力模型相当甚至更优，说明这个更简单的基线方法被严重低估了。 |
| [^15] | [Curvature-Conditioned Multiscale Momentum with Sphere Constraints for LLM Pretraining](https://arxiv.org/abs/2608.28442) | 该论文提出了一种带球面约束的曲率条件多尺度动量优化方法，通过在平坦方向上结合慢衰减降噪分量与快衰减曲率适应分量的互补优势，加速大语言模型预训练中主导最终损失下降的平坦方向进展。 |
| [^16] | [Euclidean Fourier Neural Operators](https://arxiv.org/abs/2608.28425) | 提出欧几里得傅里叶神经算子（EFNO），通过将谱核参数化为物理波矢的连续函数，使神经算子能够跨不同形状和大小的周期域保持一致地学习与迁移，克服了传统FNO的域依赖性问题。 |
| [^17] | [SymboLLM-FE: LLM-Accelerated Symbolic Regression for Automated Feature Engineering on Tabular Data](https://arxiv.org/abs/2608.28408) | 提出SymboLLM-FE方法，将符号回归与大语言模型相结合，通过符号回归提取与目标强相关、可解释的数学公式来指导特征生成，从而解决了传统AutoFE可解释性差、以及基于LLM的方法迭代成本高且易产生偏见与幻觉的问题。 |
| [^18] | [Post-Training VLMs for Video Mistake Detection](https://arxiv.org/abs/2608.28406) | 该论文提出了错误检测视频问答（MD-VQA）协议与基准，并首次通过后训练视觉语言模型来学习错误的通用概念，从而实现对已见和未见动作的视频错误检测。 |
| [^19] | [Timing-Aware Repurchase Prediction for Web-Scale E-Commerce: Survival Models for Multi-Surface Grocery Recommendation](https://arxiv.org/abs/2608.28393) | 该研究用生存模型直接预测复购时间，取代按固定时间窗口逐一训练的二元分类模型，并发现生鲜复购的边际风险随时间轻微递减（对数正态分布拟合最佳），与“越久越可能复购”的传统直觉相反。 |
| [^20] | [Quantum Federated Learning Based on Bures--Uhlmann Geometry for Heterogeneous Noisy Clients](https://arxiv.org/abs/2608.28379) | 该论文提出将参数空间几何从纯态扩展到噪声客户端实际制备的混合态，利用混合态几何张量的实部（Bures度量）作为局部预条件子、虚部（平均Uhlmann曲率）构建动态聚合规则，从而有效应对量子联邦学习中噪声量子设备的数据与硬件异构性问题。 |
| [^21] | [Localizing Global Discrepancies: Marginal Contributions and Contextual Anomaly Detection](https://arxiv.org/abs/2608.28375) | 该论文提出了一个将全局分布差异定位到具体观测值的框架，通过为每个观测分配其在随机统计情境中的边际贡献，统一了重采样诊断、数据估值与事件级异常检测，并由此获得更高效的估计量。 |
| [^22] | [Real-Time Monitoring of MHD Liquid Metal Flows with Shallow Recurrent Decoders](https://arxiv.org/abs/2608.28366) | 该研究提出将浅层循环解码器（SHRED）与主成分分析相结合的数据驱动降阶模型框架，通过稀疏温度测量实现对DEMO聚变堆液态金属增殖包层中三维磁流体流动的实时状态监测。 |
| [^23] | [GRACE:Gradient-guided Coreset Selection for LLM Unlearning](https://arxiv.org/abs/2608.28361) | 提出GRACE，一种梯度引导的核心集选择方法，能够仅凭少量不良行为种子示例自动构建大语言模型遗忘所需的遗忘集和保留集，并有效保持模型效用。 |
| [^24] | [BanglaMed-QA: A Question Answering System for Healthcare Support in Bangla](https://arxiv.org/abs/2608.28329) | 本文提出了BanglaMed-QA——首个专为孟加拉语医疗领域设计的问答系统，通过构建包含506种疾病、4,493个问答对的结构化知识库，结合SVM问题分类、领域专用词典与同义词集以及多种相似度度量与投票机制，为低资源语言的医疗健康信息支持提供了有效解决方案。 |
| [^25] | [Deriving Scaling Laws for OpenEuroLLM Models: Learning Rate, Batch Size and Loss](https://arxiv.org/abs/2608.28308) | 该论文为OpenEuroLLM模型推导了学习率、批量大小和损失的缩放定律，研究了它们随模型容量和数据规模的边际演变及最优超参数在训练阶段间的迁移性，并验证了显式建模容量与数据交互的缩放形式能有效捕捉欠训练与过训练两种情形。 |
| [^26] | [VISTA: Verifier-Informed Student-to-Teacher Adaptation for On-Policy Self-Distillation](https://arxiv.org/abs/2608.28306) | 提出VISTA方法，在保留标准在策略自蒸馏学生更新的同时，利用结果验证的rollout使特权教师向学生分布自适应，解决了教师分布与学生有效推理不匹配时单向监督误导学生的问题。 |
| [^27] | [Parser States Already Know: Structure-Conditioned KV Persistence for Structured Generation](https://arxiv.org/abs/2608.28276) | 提出PASK方法，将受限解码中解析器状态所暴露的结构信号转化为按层组划分的KV缓存持久化决策，通过任务错误敏感度设定最低保护底线、注意力输出失真分配剩余容量，从而在结构化生成中保护模式关键的KV缓存。 |
| [^28] | [An algebraic proof of Colombo's difference-power determinant conjecture](https://arxiv.org/abs/2608.28274) | 本文通过将假想的核向量转化为实二元型的方法，证明了超临界奇数指数下差-幂矩阵的非奇异性，从而完整证明了 Colombo 于 1928 年提出的差-幂行列式猜想，并由此确立了秩公式 rank A_d(λ)=min{n,d+1}。 |
| [^29] | [Learning to Transfer Across Modes: Towards Unified Urban Mobility Forecasting](https://arxiv.org/abs/2608.28273) | 提出TransMod统一框架，通过构建共享的区级空间表示对齐不同空间粒度的出行系统，实现异构出行模式间的知识迁移，从而解决多模式城市出行需求预测中的空间异质性与新兴模式数据稀缺问题。 |
| [^30] | [Residual-Guided Randomized Neural Networks](https://arxiv.org/abs/2608.28267) | 提出一种残差引导的贪心构建方法，通过闭式残差下降准则逐步筛选随机候选隐藏单元并重新拟合输出层，克服了随机化神经网络一次性随机特征构建所导致的冗余表示和模型容量利用不足的问题。 |
| [^31] | [SinkSLOT: Sinkhorn via Sparse Lifted Optimal Transport](https://arxiv.org/abs/2608.28262) | SinkSLOT通过期望切片提升传输计划对Gibbs核进行稀疏化并采用非独立先验耦合，将每次Sinkhorn迭代的计算复杂度从O(N²)降至O(LN)，同时所得目标函数无需去偏处理。 |
| [^32] | [I-FLOP: Fast Learning of Order and Parents from Interventional Data](https://arxiv.org/abs/2608.28245) | I-FLOP将FLOP算法从观测数据扩展到干预数据场景，通过将干预BIC评分适配到基于Cholesky的迭代评分更新机制中，实现了兼具速度优势与理论保证（可恢复正确干预马尔可夫等价类）的快速因果结构学习。 |
| [^33] | [Spectral Features Dominate BCG Respiratory-Event Detection: A Large-Scale Patient-Independent Comparison of Feature Groups in Sleep Apnea Patients](https://arxiv.org/abs/2608.28242) | 该大规模患者无关研究表明，频谱特征在心冲击图（BCG）睡眠呼吸暂停呼吸事件检测中占主导地位，随机森林和直方图梯度提升模型达到了0.967的AUC-ROC。 |
| [^34] | [Efficient Online Continual Foundation Model Fine-Tuning for Predictive Process Monitoring](https://arxiv.org/abs/2608.28237) | 提出了首个面向预测性流程监控的基础模型在线持续微调框架COMPASS，通过自适应损失平台期漂移检测自主识别任务边界，并利用统一知识子空间有效缓解概念漂移带来的冷启动问题。 |
| [^35] | [D-TAIA: Domain-Aware LLM Adaptation for Multi-Task Predictive Process Monitoring](https://arxiv.org/abs/2608.28236) | 提出D-TAIA框架，通过领域感知训练与基于注意力的推理架构对大语言模型进行参数高效微调，在数据稀缺、高流程熵和分布偏移条件下实现下一个活动与剩余时间的联合预测。 |
| [^36] | [Stay Within Your Bounds: Distance-Guided Decoding for Guaranteed Context-Free Grammar Compliance](https://arxiv.org/abs/2608.28229) | 提出一种基于下推自动机的距离引导解码框架，通过离线计算可达性标签与到接受状态的距离上界、在线进行视野感知剪枝与束搜索，保证大模型生成结果百分之百符合目标上下文无关文法，同时提升补全质量。 |
| [^37] | [Generalized Context in Cross Attention for Transfer Learning of Disjoint Tabular Data](https://arxiv.org/abs/2608.28209) | 本文提出CATTLE方法，利用Transformer中key和query投影权重捕获广义上下文，在源域与目标域表格数据完全不共享特征的情况下，以数据无关的方式实现跨领域注意力迁移学习。 |
| [^38] | [Explainable Diabetic Retinopathy Classification Using Vision Foundation Models](https://arxiv.org/abs/2608.28207) | 该研究提出了一种基于视觉基础模型（DINOv2、CLIP、ViT）结合多种迁移学习策略的可解释糖尿病视网膜病变分类框架，其中DINOv2-LoRA内部AUROC达0.758，外部AUROC达0.920，并通过Grad-CAM和HiResCAM结合专家标注病灶掩码实现了模型可解释性评估。 |
| [^39] | [Performative Privacy: When Differential Privacy Maximizes Utility](https://arxiv.org/abs/2608.28198) | 该论文提出“表演性隐私”新框架，首次形式化了隐私保护与用户参与度之间的动态关系，并证明当数据泄露导致用户流失时，采用有限隐私预算的差分隐私机制在长期内可以优于非隐私估计。 |
| [^40] | [EXPOSE: Explainable and Domain-Robust Embeddings from Pathology Vision Foundation Models using Sparse Autoencoders](https://arxiv.org/abs/2608.28191) | 提出了EXPOSE框架，利用稀疏自编码器作为可解释瓶颈，识别并抑制病理视觉基础模型嵌入中的域特定信息，从而在不重新训练骨干模型的情况下提升跨域泛化能力。 |
| [^41] | [Beyond Flat Netlist: Hierarchical Graph Representation Learning for Scalable Analysis of Sequential Circuits](https://arxiv.org/abs/2608.28188) | DeepSeq3提出了一种层次化图表示学习框架，通过触发器划分的组合子图与超节点图两级表示、双GNN架构以及基于状态可达性的预训练方案，实现了对大规模时序电路的可扩展分析。 |
| [^42] | [Biologically Inspired Mechanisms for Facilitating Grokking in Multilayer Perceptrons](https://arxiv.org/abs/2608.28184) | 本文在多层感知机中引入七种受生物学启发的机制（如输入门控、结构可塑性、稳态调节、侧抑制等），通过消融实验发现稳态调节对促进grokking（从记忆到泛化的延迟转变）的作用最强且最一致，结构稀疏化也具有重要作用。 |
| [^43] | [Conformal Risk-Averse Decision Making with Optimized Certainty Equivalent Risk Control](https://arxiv.org/abs/2608.28179) | 该论文提出了基于优化确定性等价（OCE）度量的风险规避决策框架，证明CVaR下的最优策略可归结为基于预测集的解，从而为保形预测提供了操作性解释，并针对未知分布设计了基于合成似然模型与留出校准数据的数据驱动校准策略，实现对OCE风险的高概率控制。 |
| [^44] | [Empowering Local Agriculture: A Deep Learning-Powered Web System for Identifying Bangladeshi Mango Varieties](https://arxiv.org/abs/2608.28161) | 本研究构建了一个基于深度学习的网络系统，利用自建的包含九个孟加拉国芒果品种的图像数据集微调EfficientNetB0等CNN模型，实现了高达97.36%的测试准确率，可自动准确识别孟加拉国芒果品种。 |
| [^45] | [HARTS: Efficient Agentic Reinforcement Learning for Hybrid-Attention Models over Arbitrary Rollout Trees](https://arxiv.org/abs/2608.28158) | 提出了HARTS系统，通过联合规划微批次、数据并行副本分配与调度，以及线性时间算法协调分块边界状态恢复与重放，实现了在任意回放树上对混合注意力模型的高效智能体强化学习训练，避免了共享前缀的重复计算。 |
| [^46] | [Under-Mattress Temporal Sensing for Next-Day Agitation Risk Scoring in Dementia Wards](https://arxiv.org/abs/2608.28152) | 该研究利用床垫下非接触式传感采集的前夜分钟级时序信号来预测痴呆患者次日的激越风险，并证明保留时间结构的建模方法优于传统的整夜汇总特征。 |
| [^47] | [The Approximation Rank of Softmax Attention: Sharp Geometric Laws and Robust Interaction Dimension](https://arxiv.org/abs/2608.28150) | 本文刻画了控制softmax注意力逼近秩的几何定律——球面自注意力的秩随维度指数增长而全球几何仅多项式增长，并证明固定注意力头下可见查询-键交互维度r给出极小极大紧的r/2秩上界。 |
| [^48] | [Conditional Diffusion Models for Energy-Efficient Driving](https://arxiv.org/abs/2608.28142) | 提出了一种结合潜在条件编码器与时序一维U-Net去噪骨干的条件扩散框架，能以车辆速度和环境温度等路线特征为条件生成真实的电动汽车电池电流曲线，为能耗感知的车队运营决策提供支持。 |
| [^49] | [CheXtriev: Anatomy-Centered Representation for Case-Based Retrieval of Chest Radiographs](https://arxiv.org/abs/2608.28137) | CheXtriev提出了一种基于图Transformer的解剖感知胸部X线片检索框架，通过从特定解剖区域提取特征并建模解剖位置与影像发现的相互作用，在检索准确率和排序质量上分别超越现有最先进方法18%-26%和11%-23%。 |
| [^50] | [Learning to Difference: Adaptive Reversible Differencing (AdaRDiff) for Time Series Forecasting](https://arxiv.org/abs/2608.28134) | 提出AdaRDiff，一种利用可学习权重的自适应可逆差分方法，以单一算子同时去除并自回归恢复趋势与季节性成分，从而提升长时程时间序列预测的可靠性。 |
| [^51] | [VICT: Verifier-Instrumented Credit Tracing for Long-Horizon LLM Agent Reinforcement Learning](https://arxiv.org/abs/2608.28128) | VICT提出一种训练时接口，将终端验证器内部的可执行或证据支持的原子检查通过依赖证明边追溯到具体动作，并仅沿这些边重新分配组相对优势，从而解决长程LLM智能体强化学习中的细粒度功劳分配问题。 |
| [^52] | [Generalized Gibbs Ensemble Weighting for Forecast Combination](https://arxiv.org/abs/2608.28116) | 本文提出了广义吉布斯集成加权（GGEW）概率框架，将预测模型视为专家并通过归一化预测损失的吉布斯式指数变换分配集成权重，进一步扩展出数值稳定、多样性感知与在线自适应的一系列方法，以提升预测组合的稳健性与性能。 |
| [^53] | [Do Medical Vision Models Reason About Anatomy? Probing the Spatial Inductive Biases of Learned Visual Representations](https://arxiv.org/abs/2608.28092) | 医学视觉模型在解剖空间推理上存在根本性缺陷，其看似准确的表现实则源于对典型解剖结构的记忆而非对图像的真正空间理解。 |
| [^54] | [Comparing Classical and Quantum Machine Learning for Regression in High Energy Physics Collision Data](https://arxiv.org/abs/2608.28084) | 该论文系统比较了四种经典机器学习模型与其量子对应模型在CERN开放数据质子-质子对撞事件的横向动量回归任务上的表现，发现经典架构（尤其是CNN和LSTM）略优于量子模型。 |
| [^55] | [Landau theory of quenched criticality in linear in-context learning](https://arxiv.org/abs/2608.28059) | 该论文将线性上下文学习中的双下降奇异性表述为淬火无序系统的临界现象，构建了以重整化岭参数为序参量的朗道理论，并揭示学习参数的样本间涨落是奇异误差的微观起源。 |
| [^56] | [Explainable Uncertainty Estimation for Reliable Medical AI](https://arxiv.org/abs/2608.28052) | 本文提出将不确定性估计与可解释人工智能相统一的新方法egRUE，能够量化医疗AI预测的不确定性并将其分解为特征层面的贡献，从而提升临床决策的可靠性与可解释性。 |
| [^57] | [Emergent aggregation from collective foraging](https://arxiv.org/abs/2608.28046) | 该研究表明，在没有任何直接群聚奖励的情况下，聚集行为可以从纯粹个体化的最优觅食目标中间接涌现——随着视觉范围增大，强化学习觅食者会从个体搜索策略急剧转变为集体搜索策略，并自发形成空间聚集。 |
| [^58] | [Characterization of Request and Token Energy Costs for LLM Inference Workloads on GPU Platforms](https://arxiv.org/abs/2608.28044) | 该论文提出一个分解式LLM推理能耗模型，揭示令牌归一化能耗指标的局限性，并在H100/H200 GPU上系统表征了请求能耗与令牌能耗随模型类型、批大小、上下文长度和输出长度的变化规律。 |
| [^59] | [Twin Worlds: Equivariance-Based Abstention for Evidence-Grounded Reasoning](https://arxiv.org/abs/2608.28018) | 该论文提出“双生世界”（TW）框架，通过等变性检验模型的推理是否真正以证据为依据，使模型在证据不足时能够弃答，从而避免生成看似合理却缺乏证据支撑的答案。 |
| [^60] | [When Can Conditional Flow Matching Replace Pointwise Negative Log-Likelihood?](https://arxiv.org/abs/2608.28010) | 本文通过将端点NLL精确分解为熵、加权CFM目标、内部速度-得分残差和边界残差，刻画了条件流匹配损失何时能替代逐点负对数似然，证明普通CFM通常不是逐点NLL估计器，而特定加权 \(w_{\mathrm{sc}}(t)=(1-t)/t\) 在离策略最优下可消除内部残差，但该结论不能推广到训练或在线策略对齐场景。 |
| [^61] | [Exact Risk Ratios for Weighted Data Selection in Linear Regression](https://arxiv.org/abs/2608.28007) | 本文解决了 Hanneke 等人提出的加权数据选择公开问题，精确确定了最小范数 ERM 下的最坏风险比率 F_w(d,n) 的多个取值，包括证明 F_w(d,2d-1)=1+1/d，并给出了中间预算 n=d+k 情形下基于平衡划分调和量的紧致下界 1+Γ_{d,k}。 |
| [^62] | [A Method for Layer Bit-Width Allocation in LLM Quantization via Performance Maximization Under a Quality-Degradation Constraint](https://arxiv.org/abs/2608.28003) | 该论文提出一种在质量退化预算约束下通过最大化推理性能来为LLM逐层分配量化位宽的方法，并借助TensorRT-LLM实测区分了FFN、Attention和lm_head各模块对整体加速的贡献。 |
| [^63] | [Is Monte Carlo Tree Search Just Every-Visit Monte Carlo Control?](https://arxiv.org/abs/2608.27985) | 本文论证了MCTS的四个阶段（选择、扩展、模拟、回传）本质上可归结为“当前策略下的轨迹采样”和“每次访问蒙特卡洛更新”两个基本操作，即MCTS实质上就是每次访问蒙特卡洛控制。 |
| [^64] | [PhyMamba: Physics-Modulated Mamba for Robust Battery Health Prognostics](https://arxiv.org/abs/2608.27978) | 提出PhyMamba两阶段物理调制Mamba框架，将电化学老化机理融入序列建模，仅利用BMS信号即可实现无需侵入式测量的鲁棒长期电池健康预测。 |
| [^65] | [Not to Break, but to Attest: Adversarial Probes for Privacy-Preserving LLM Verification](https://arxiv.org/abs/2608.27954) | 提出一种基于zk-SNARK的隐私保护审计框架，利用对抗性探针放大模型部署后修改所导致的logit漂移，从而在无需访问专有模型权重的情况下验证大语言模型是否被篡改。 |
| [^66] | [Temporal Memory-Aware Online Test-Time Adaptation on Dynamic Graphs](https://arxiv.org/abs/2608.27948) | 提出了DGOTTA框架，通过时序记忆感知的在线测试时自适应方法，有效应对动态图上DGNN模型因结构和语义持续演化而面临的分布偏移挑战。 |
| [^67] | [TI$^2$PS: A Topology-Informed Inverse Design Framework for Stochastic Multicellular Pattern Formation](https://arxiv.org/abs/2608.27931) | 该研究提出TI²PS框架，通过结合拓扑数据分析中的贝蒂数向量与逆向代理建模，实现了从目标多细胞空间模式直接推断基于智能体模型的细胞级参数，并在斑马鱼色素模式形成模型中验证了其有效性。 |
| [^68] | [TACIT-Switch: Cost-Aware Model Escalation for LLM Agents from Censored Supervision](https://arxiv.org/abs/2608.27911) | 提出TACIT-Switch方法，利用教师标注的删失干预时间学习永久性移交策略，以成本感知的方式决定何时将LLM智能体从小模型骨干升级至大模型骨干，部署时无需教师参与即可将成功率提升7.4-11.1个百分点。 |
| [^69] | [OpenStamp: A Watermark for Open-Source Language Models](https://arxiv.org/abs/2608.27899) | OpenStamp通过仅修改开源语言模型的反嵌入层，将水印逻辑直接编码进模型权重，解决了传统采样概率水印在白盒场景下可被用户禁用的问题，在几乎不损失模型能力的前提下实现了更优的检测性能和更强的鲁棒性。 |
| [^70] | [There and Back Again: Bidirectional Diffusion Bridges for Multimodality Translation](https://arxiv.org/abs/2608.27885) | 提出了双向图像-文本扩散桥BIT，直接从文本出发插值到图像，实现了源感知的生成路径与可逆的双向多模态生成框架。 |
| [^71] | [Beyond Pairwise Graphs in Science: Hypergraph Adaptive Wavelet Operators for Parametric PDEs](https://arxiv.org/abs/2608.27883) | 提出超图自适应小波算子HALO，将求解域提升至超图并在其谱小波域中学习，突破了成对图神经算子的局限，可直接捕获网格单元间的群体耦合，从而提升非结构化网格上参数化偏微分方程求解的精度与稳定性。 |
| [^72] | [SOMTab: Set-Order Mamba for Efficient Tabular In-Context Learning](https://arxiv.org/abs/2608.27882) | SOMTab提出一种集合序Mamba架构，用基于Mamba的状态空间混合替代表格上下文学习中构建行列表示所需的注意力机制，仅在最终预测阶段保留注意力以实现查询条件检索，从而实现更高效的表格上下文学习。 |
| [^73] | [What Do Interaction Representations Actually Measure? Pre-Event Separability in Weakly-Supervised Violence Detection](https://arxiv.org/abs/2608.27879) | 该研究在固定下游流程的严格控制条件下，通过弱监督早期暴力检测系统比较五种交互表征，发现基于人体姿态的表征并不优于粗略的边界框几何特征。 |
| [^74] | [Anchored Scenario Coverage for Failure-Aware First-Hit Batch Inverse Design](https://arxiv.org/abs/2608.27873) | 本文提出ARC-SC批量采集方法，通过锚定强边际候选对象并在风险支持约束下最大化对预测目标场景的互补覆盖，避免了冗余推荐，在易失败的闭环逆向设计中显著提升了首次命中发现的效率。 |
| [^75] | [FedEHR-Agents: Federated Agentic Optimization for Automated EHR Modeling](https://arxiv.org/abs/2608.27856) | 提出FedEHR-Agents框架，将联邦学习从以模型为中心转变为以经验为中心的智能体优化范式，使各医院部署的自主临床智能体能够在保护隐私的前提下共享建模经验，实现自动化EHR建模。 |
| [^76] | [RealSWE: A Compositional Evaluation of Coding Agents under Realistic User Requests](https://arxiv.org/abs/2608.27831) | 该论文提出RealSWE基准，通过381个源自SWE-bench的多变体任务族来模拟简短、随意、信息稀疏的真实用户请求，从而更真实地评估编程智能体，并揭示了现有基准与真实用户请求在信息完整度和语言风格上的显著差距。 |
| [^77] | [Personalized and Multi-View Representation for Federated Cold-Start Recommendation](https://arxiv.org/abs/2608.27826) | 提出PMFRec方法，利用个性化多视图表示，解决联邦冷启动推荐中缺乏个性化、异构语义组合性失效以及训练通信低效三大问题。 |
| [^78] | [Actionable CBFI: Integrating Structural Decomposition and Causal Counterfactual Recourse for Tabular Machine Learning](https://arxiv.org/abs/2608.27821) | 该论文提出A-CBFI框架，基于结构因果模型将特征重要性分解与因果反事实补救相结合，通过隔离协同交互瓶颈并释放抑制性结构锁，为表格机器学习生成因果有效、可操作的针对性干预方案，克服了现有方法因果无效与干预分散的问题。 |
| [^79] | [CURA: Certified Runtime Alarms for Computer-Use Agents](https://arxiv.org/abs/2608.27808) | 提出CURA外部监视器，仅利用工具可见的遥测数据，通过带认证虚警控制的CUSUM序贯检验，能在6.6%的实际虚警率下检测出42.3%的智能体失败，且比任务终止中位数提前31步，无需模型内部信息或额外LLM调用。 |
| [^80] | [Node-wise Feature Encoding for Neural Performance Prediction](https://arxiv.org/abs/2608.27794) | 本文提出FeatureFormer，通过在门控图注意力架构中显式编码每个节点的FLOPs、参数量和内存代理指标，并配合新的大规模能耗数据集NNEQ，在延迟和能耗预测任务上实现了最先进的性能。 |
| [^81] | [Initialization Is Critical: Advancing Federated Short-Term Load Forecasting under Load Heterogeneity via Model Initialization](https://arxiv.org/abs/2608.27791) | 本文揭示了联邦学习客户端负荷数据中存在的结构化异构性问题，并从全局和局部两个视角提出模型初始化策略，以提升负荷异构场景下联邦短期负荷预测的性能。 |
| [^82] | [Memorization Is Not Extraction: Tight Differential-Privacy Bounds and Audit Blind Spots](https://arxiv.org/abs/2608.27782) | 该论文精确刻画了差分隐私对反事实记忆化与自适应提取这两个度量的紧致控制界，证明二者互不控制，从而揭示了差分隐私作为统一防护代理时存在的审计盲区。 |
| [^83] | [Beyond Procrustes distances: a multilinear Gromov-Wasserstein distance capturing chirality](https://arxiv.org/abs/2608.27774) | 该论文提出了Gromov-Wasserstein目标的多线性推广，由此定义出对手性敏感的手性Gromov-Wasserstein（CGW）距离，能够区分形状与其镜像，并具备鲁棒性保证和高效的计算算法。 |
| [^84] | [Fast Weight Attention for Continual Learning](https://arxiv.org/abs/2608.27763) | 该论文在“写后读”自回归语义下将快速权重记忆与状态空间模型的状态转移统一视为在线学习规则，并推导出面向持续学习前缀预测的归一化一阶更新家族（Falcon 系列回归与内积变体）。 |
| [^85] | [Beyond Search-Imitation: Prior-Directed Exploration for Searchless Chess](https://arxiv.org/abs/2608.27757) | 该论文提出用朝向网络自身MCTS先验的前向质量覆盖KL散度（先验引导探索）替代传统熵奖励，并结合由价值头不确定性驱动的熵自适应采样温度，通过自我对弈强化学习将无搜索国际象棋网络的谜题准确率从93.9%提升至94.9%。 |
| [^86] | [The Calls are Coming from Inside the Model: Investigating Probe-based Detection of Tool-Calling Errors in LLMs](https://arxiv.org/abs/2608.27750) | 本研究提出利用线性探针读取大语言模型隐藏状态来检测工具调用错误，在18个模型上验证了该方法能有效捕获包括参数值错误在内的各类调用错误，且检测效果受模型大小、探针层级和后训练类型的影响。 |
| [^87] | [Diffusion Distillation for Efficient Weather Ensembles](https://arxiv.org/abs/2608.27728) | 该论文提出一种有监督的能量距离蒸馏方法，将多步扩散天气集合预报模型压缩为单步学生模型，在每次自回归步骤仅需一次神经函数评估的情况下达到或超越教师模型性能，并在极端事件预报中保持预报能力。 |
| [^88] | [Leveraging a Foundation Model for the EEG-Based Diagnosis of Alzheimer's Disease](https://arxiv.org/abs/2608.27719) | 该研究利用在2,500多小时EEG数据上预训练的大脑基础模型LaBraM，结合非线性随机森林分类器，仅凭8秒EEG片段即可高精度区分阿尔茨海默病患者与健康对照，性能超越传统频谱特征方法。 |
| [^89] | [Beyond Non-IID: Learner--Client Distribution Mismatch in Federated Learning](https://arxiv.org/abs/2608.27715) | 该论文率先研究并缓解联邦学习中学习者目标分布与客户端数据分布之间的失配问题，提出在学习者仅持有小型代理数据集的实际场景下评估并利用各客户端异质贡献的新方法。 |
| [^90] | [DART-FL: Burst-Aware Multitask Federated Learning under Dynamic Inference Demand at the Edge](https://arxiv.org/abs/2608.27713) | 该论文提出DART-FL框架，一种SLO感知、需求驱动的多任务联邦学习方法，能够在边缘设备上联合优化推理与训练的资源分配，并根据任务需求动态调整各任务的训练优先级，以应对动态变化且存在突发的推理需求。 |
| [^91] | [On the Computational and Statistical Efficiency of the Empirical Maximum Entropy on the Mean Method](https://arxiv.org/abs/2608.27705) | 本文将经验均值最大熵（MEM）方法的期望收敛速率从 $O(n^{-1/4})$ 提升至参数化的 $O(n^{-1/2})$，并通过将MEM对偶问题重构为期望风险最小化问题，使其融入现代随机优化框架。 |
| [^92] | [RiskBlend: A Multi-Signal Framework for Test Input Prioritization in Machine Learning Regression Testing](https://arxiv.org/abs/2608.27704) | RiskBlend提出了一种与分类器无关的多信号测试输入优先级排序框架，通过融合历史失败模式、预测偏移、决策边界偏移和邻域变化四种互补风险信号，在有限验证预算下更有效地发现机器学习模型重训练引发的回归缺陷。 |
| [^93] | [CARDINAL Predicts Cardiovascular Risk From Non-contrast Cardiac CT](https://arxiv.org/abs/2608.27690) | CARDINAL是一个从常规非增强心脏CT中学习紧凑表示的深度学习框架，在17,659名患者中，其对1至10年主要不良心血管事件的预测性能超越了传统临床方程（PCE、PREVENT）、冠状动脉钙化评分及放射组学特征，且预测时间越长优势越明显。 |
| [^94] | [SafeStep: An Interactive Demonstration of Semantic Communication for Pedestrian Safety Monitoring](https://arxiv.org/abs/2608.27688) | SafeStep是一个基于浏览器的交互式语义通信演示平台，可从实时交通摄像头中提取行人信息进行安全监控，并展示了仅416万参数的Meta-VIB模型无需在线重训练即可在不同SNR、码长和AoI条件下泛化的优势。 |
| [^95] | [SegBench-GC: Testing Segmentation Invariance in Multi-Step Offline Goal-Conditioned Reinforcement Learning](https://arxiv.org/abs/2608.27678) | 该论文提出SegBench-GC基准，通过严格控制变量的压力测试揭示多步离线目标条件强化学习对轨迹分割方式高度敏感：即使人工切分保留延续价值（CVT）也会使成功率从50.5%降至39.1%，若被视为绝对终止则进一步骤降至19.1%。 |
| [^96] | [Unsupervised Continual Learning with Growing Self-Organizing Maps and Synthetic Replay](https://arxiv.org/abs/2608.27662) | 该论文提出了一种基于生长自组织映射和分布统计记忆的完全无监督持续学习框架，通过生成合成样本进行重放，无需存储原始数据或依赖任务边界与类别标签，即可达到与有监督最先进方法相当的性能。 |
| [^97] | [More Data Cannot Break a Symmetry: Identifiability by Design](https://arxiv.org/abs/2608.27651) | 该论文证明刺激几何的对称性从结构上决定了对应关系的可辨识上限——再多的数据或计算也无法弥补，并提出在设计阶段通过诊断工具选择非对称的刺激集合，从根本上保证可辨识性。 |
| [^98] | [Curvature-Aware Radius Shrinkage for Adaptive Nearest Neighbor Classification](https://arxiv.org/abs/2608.27634) | 提出了几何驱动的CARSANN框架，通过基于形状算子的局部平均曲率估计来自适应收缩最近邻邻域半径，使高曲率区域获得更紧凑的邻域、平坦区域保留更宽的空间支撑，从而让最近邻分类适应流形上变化的局部几何。 |
| [^99] | [Depth-Aware Pothole Detection Using YOLO and RT-DETR at the Edge](https://arxiv.org/abs/2608.27633) | 该论文提出了一种基于RGB-D传感器融合的深度感知坑洼检测框架，在边缘设备上比较了YOLOv8n、YOLOv8nSeg、YOLOv9t、RT-DETR-L和RT-DETR-X五种架构，并结合RANSAC地面正射校正实现坑洼深度的自动化测量。 |
| [^100] | [Quantum SEDONet: Spectrally-Embedded Quantum Deep Operator Networks for Partial Differential Equations](https://arxiv.org/abs/2608.27626) | 量子SEDONet根据边界条件为量子DeepONet主干网络的各坐标分别嵌入傅里叶（周期）或切比雪夫（非周期）谱特征，在不增加量子比特和电路深度的前提下提升了量子神经算子求解偏微分方程的能力。 |
| [^101] | [Tensor-Accelerated Eager Multi-Resolution Grids for Evolving Large-Scale Substrates](https://arxiv.org/abs/2608.27612) | 提出用张量加速的即时多分辨率网格替代 ES-HyperNEAT 中难以张量化且无法批处理的四叉树细分方法，从而在 JAX 框架下高效演化大规模神经基底。 |
| [^102] | [Physics-informed learning for the inverse problem in resonant ultrasound spectroscopy](https://arxiv.org/abs/2608.27590) | 该论文提出一种物理信息学习框架，将共振超声谱的弹性常数反演问题转化为约束逆等谱问题，通过低维特征变量与解析式尺度恢复相结合，实现了高精度的弹性常数重构。 |
| [^103] | [Towards Large-Scale Heterogeneous Data Organization for Scientific Foundation Models: A Nuclear Fusion Case Study](https://arxiv.org/abs/2608.27578) | 本文以核聚变为案例，系统表征了训练科学基础模型所需的大规模异构稀疏数据（超过20种传感器、采样率跨越5个数量级、混合张量结构），并提出了大规模多模态波动数据表示的通用模板。 |
| [^104] | [Self-Explainable Multi-Label Graph Neural Network for Correlated Evidence Attribution](https://arxiv.org/abs/2608.27574) | 本文提出SEMGNN，一种端到端的自解释多标签图神经网络，能够在进行多标签节点分类的同时识别对各预测标签有显著贡献的关键边，从而解决事后解释方法无法建模标签间证据共享与分离的问题。 |
| [^105] | [Towards a mathematical theory of superposition](https://arxiv.org/abs/2608.27540) | 该论文利用框架理论与压缩感知工具，首次为神经网络中的叠加现象建立了严格的数学恢复理论，在随机和最坏情况支撑集设定下均证明了特征恢复定理，并精确确定了等角紧框架的恢复阈值。 |
| [^106] | [Ab initio Modeling of MoS2/Oxide Device Interfaces with Machine Learned Electronic Structures](https://arxiv.org/abs/2608.27533) | 该研究提出了一种结合机器学习电子结构模型与量子输运求解器的新型从头算方法，实现了比密度泛函理论快一万倍的加速，可处理超过两万个原子的器件，并揭示了半导体-氧化物界面附近配位不足的金属原子对MoS2器件中电子电流及其传播的显著影响。 |
| [^107] | [Dandelion: A Spherical Flower for Neural Simulation of Planetary Dynamics](https://arxiv.org/abs/2608.27521) | 提出Dandelion——基于warp的神经PDE求解器Flower的球面版本，通过沿大圆传输特征并在球谐域中实现分层池化，构建了无卷积的球面神经网络架构，用于行星动力学的神经模拟。 |
| [^108] | [When Muon Meets Task Interference: A Spectral Perspective on Continual Learning and Model Merging](https://arxiv.org/abs/2608.27518) | 该论文揭示持续学习中的灾难性遗忘与模型合并中的权重解缠误差本质上是同一现象——任务干扰，并将其统一归结为逐层Frobenius内积，进而从理论上证明优化器可通过控制参数更新的谱范数来缓解任务干扰。 |
| [^109] | [Destroy Me: Automatic Artifact Generation for Histopathology Images](https://arxiv.org/abs/2608.27516) | 提出"Destroy Me"混合框架，结合微调的Stable Diffusion与基于物理的程序化建模，自动合成六种逼真伪影用于数据增强，使病理图像分析模型在不完美数据环境下保持鲁棒性。 |
| [^110] | [A Deeper Analysis of Block-Sparse Featurizers](https://arxiv.org/abs/2608.27515) | 本文深入分析了块稀疏特征化器（BSF）的优缺点，提出包括锦标赛Top-K选择规则在内的多项架构改进以显著减少特征分裂，并将块范式扩展到了交叉编码器。 |
| [^111] | [DAMP: Decay-Aware Mixed-Precision Recurrent-State Quantization](https://arxiv.org/abs/2608.27513) | 该论文首次研究了GDN/KDA语言模型中循环状态的训练后量化问题，发现均匀量化效果不佳且量化误差集中在少数与衰减相关的通道中，据此提出感知衰减的混合精度量化方法DAMP，以更优的精度-显存权衡加速解码。 |
| [^112] | [Quantization-Triggered Backdoors in Language Models: Cross-Quantizer Transferability and the Validation--Deployment Gap](https://arxiv.org/abs/2608.27512) | 该论文提出量化行为等价类（QBEC）理论，证明源精度下的模型验证无法保证量化部署后的行为等价，并构建三阶段对抗微调框架，使后门仅在模型被INT8或4比特量化部署时才被触发激活，揭示了量化流程中的安全隐患。 |
| [^113] | [How Do Linear Probes Emerge? A Circuit-Tracing Framework with Concept-Targeted Attribution](https://arxiv.org/abs/2608.27510) | 该论文提出概念定向归因（CTA）框架，通过针对线性探针方向训练归因图，首次将线性探针的性能与模型内部可解释的电路结构联系起来，不仅能判断探针是否有效，还能揭示是哪些内部计算使探针起作用。 |
| [^114] | [Marginal Coverage Credit Reduces Redundant Exploration in Parallel State-Entropy Optimization](https://arxiv.org/abs/2608.27507) | 本文提出MCC-PGPSE方法，通过结合留一策略覆盖与状态所有者专业化的边际覆盖信用来重新分配辅助内在奖励，从而减少并行状态熵优化中的冗余探索并促进互补性状态覆盖。 |
| [^115] | [Optimal Transport for Network Comparison: A Review with Machine Learning Applications](https://arxiv.org/abs/2608.27500) | 本文综述了基于最优传输的网络比较方法，系统梳理了Wasserstein、Gromov-Wasserstein和Bures-Wasserstein三种距离，突出传输方案可解释图间差异的节点来源，并利用拉普拉斯谱为Bures-Wasserstein距离推导高效边界，进而在聚类和时间序列网络任务中验证了这些方法。 |
| [^116] | [Multiscale Community-Based Fingerprinting of Signed Functional Networks](https://arxiv.org/abs/2608.27483) | 该论文提出了一种基于符号多层社区检测的多尺度功能连接组指纹识别框架，通过融合正负相关脑活动的中尺度社区结构生成低维社区级指纹，从而实现跨任务、跨会话且更鲁棒的个体识别。 |
| [^117] | [Effectiveness of IoT and Deep Learning for Detection and Severity Assessment of Postelectrotermes militaris in Tea Plantations](https://arxiv.org/abs/2608.27480) | 该研究提出了一种结合深度学习的物联网声学监测框架，通过采集茶树树干音频信号并训练CNN模型，实现了茶园白蚁侵染的早期检测与严重程度评估。 |
| [^118] | [Hypothesize, Evaluate, Refine: A Scientific Agent for PDE Discovery with Unknown Spatial Coefficient Fields](https://arxiv.org/abs/2608.27475) | 提出了HER-PDE科学智能体框架，通过“假设-评估-精炼”流程联合发现非均匀介质中偏微分方程的组合结构与未知时不变系数场，并借助双向跨激励迁移评估假设，避免系数场的灵活性掩盖结构性误差。 |
| [^119] | [SciReC: Diagnostic Evaluation of Multimodal, Multi-Turn Relational Reasoning with Adaptive Interaction](https://arxiv.org/abs/2608.27461) | 该论文提出了SciReC——一个模型自适应的多模态学术对话基准，以及DMRA缺陷诊断框架，用于系统评估多模态大语言模型在多轮关系推理中的表现，并量化视觉理解、知识展示和记忆回忆等因素对失败案例的贡献。 |
| [^120] | [Accelerating LLM Inference via Vector Index Based Output Embeddings](https://arxiv.org/abs/2608.27460) | 本文将大语言模型的输出投影重新表述为基于HNSW向量索引的最大内积搜索，仅检索高分候选词元以替代稠密词表投影，在CPU推理中最高可将解码吞吐量提升82%且不损失生成质量。 |
| [^121] | [Understanding Evolution Strategies for LLM Reasoning: Broader Reasoning Coverage than GRPO](https://arxiv.org/abs/2608.27351) | 本文通过理论和实证证明进化策略（ES）比GRPO提供更广泛的推理覆盖，从而提升大语言模型推理性能，并提出了GRPO-ES顺序训练策略。 |
| [^122] | [Profit based evaluation of machine learning for nitrogen recommendations in winter wheat](https://arxiv.org/abs/2608.27205) | 本文提出直接以放弃利润评估氮肥推荐，发现机器学习在利润上不如标准建议，但简单修正可带来收益。 |
| [^123] | [Neural Regression with Embeddings for Numerical Attribute Prediction in Knowledge Graphs](https://arxiv.org/abs/2608.26729) | 本文提出LitEm神经回归模型和协同训练框架，使知识图谱嵌入模型能预测数值属性，并提升双线性模型的链接预测性能。 |
| [^124] | [TraceML: An Empirical Analysis of Human-Agent Planning in Machine Learning Development](https://arxiv.org/abs/2608.26086) | 本文通过引入TraceML数据集，将人类与智能体在同一机器学习竞赛中的开发轨迹进行版本级对比，揭示了智能体在自主开发中的性能差距及其具体成因。 |
| [^125] | [Trust the Mass: Forced Weights in KV-Cache Eviction](https://arxiv.org/abs/2608.25230) | 本文发现KV缓存驱逐中保留最大权重已接近最优，已发布方法间的差异主要源于存储方式而非选择策略，并揭示了评估中的内存与性能权衡。 |
| [^126] | [On-policy Distillation with Verifiable Reward](https://arxiv.org/abs/2608.24696) | 提出了一种无需额外超参数的无缝结合在线策略蒸馏和可验证奖励强化学习的方法，通过基于轨迹正确性的奖励重构和ReLU门控机制，有效提升大型语言模型后训练性能。 |
| [^127] | [XP-JEPA: Cross-Predictive Physics Grounding for Forecastable Latent Dynamics](https://arxiv.org/abs/2608.24044) | XP-JEPA通过跨模态共享预测器将视觉潜在动力学与物理轨迹结合，训练后仅保留视觉模型，从而在部署时减少滚动漂移并增强物理可预测性。 |
| [^128] | [Semantic Overlays: Mitigating Prompt Injection with Annotations Beyond Tokens and Steering Vectors](https://arxiv.org/abs/2608.23873) | 该论文提出了一种名为“语义覆盖层”的新技术，通过向模型输入添加非文本通道来缓解提示注入攻击，利用小型学习的适配器在冻结模型的残差流中创建带外注释，从而增强模型对片段身份的理解。 |
| [^129] | [RIBOSPAN: A Long-Context RNA Foundation Model for Versatile RNA Modeling](https://arxiv.org/abs/2608.22849) | 提出了RIBOSPAN——一个16.1亿参数、原生支持最长10,240核苷酸上下文的双向RNA基础模型，通过密集双向自注意力、单核苷酸分词和注意力隔离序列打包，实现了对完整长RNA的单核苷酸分辨率建模。 |
| [^130] | [GAN-Diff : Coupling Pretrained WGAN-GP Features with Conditional Diffusion U-Nets](https://arxiv.org/abs/2608.22272) | 本文提出一种混合GAN引导扩散框架，通过将预训练WGAN-GP的中间特征固定融入条件扩散U-Net，并解决多种训练不稳定性，从而提升图像恢复质量。 |
| [^131] | [How Architecture and Training Affect TPC Representations Across Experiments](https://arxiv.org/abs/2608.21756) | 该工作以TPC数据为测试平台，提出利用冻结编码器上的探针并结合随机权重对照，评估事件表征的跨实验、跨探测器复用性，并区分架构与编码器训练各自的贡献。 |
| [^132] | [What Neural Network Field Theory Can and Cannot Realise on a Computer](https://arxiv.org/abs/2608.21523) | 该论文提出了一个适用于标准网络架构的不可能性定理，据此将神经网络场理论分为四种版本，并证明有限宽度系综无论被解释为量子场论还是有效场论，都无法在计算机上自洽实现。 |
| [^133] | [JuryProbe: An Empirical Consensus-Risk Diagnostic for Routing Reference-Free Factuality Judge Panels to Grounded Verification](https://arxiv.org/abs/2608.20607) | 本文提出JuryProbe，一种通过仅假阴性相关性和假共识提升度来诊断无参考事实性评审团共识风险的方法，并在高风险时路由到有参考验证，以减少因共享盲点导致的错误接受。 |
| [^134] | [Why GPT-Style Models Do Not Directly Transfer to Symbolic Music: Compression in the Wrong Coordinate System](https://arxiv.org/abs/2608.18025) | 本文指出GPT风格模型无法直接迁移到符号音乐的原因在于压缩发生在错误的坐标系统中，并提出一个有效性-无损框架，强调分词化的核心是发现使音乐规律可预测压缩的坐标系统，而非单纯寻找更大的音乐组合。 |
| [^135] | [Recirculation](https://arxiv.org/abs/2608.17981) | 本文提出“再循环”技术，通过推理时引入特定循环机制，使现成基础模型能跟踪信念状态，显著降低困惑度并提升生成和推理准确性，且几乎不增加生成延迟。 |
| [^136] | [Establishing Boundary KKT Convergence of Mirror Descent through Reparameterization](https://arxiv.org/abs/2608.07248) | 本文通过在重参数化变量下分析镜像下降（使Hessian度量在边界附近保持平坦且非退化），并在延拓性与可定义性条件下，为一类广泛的结构化非凸问题首次建立了镜像下降收敛到边界KKT点的理论保证，克服了Legendre核梯度在边界爆炸的固有困难。 |
| [^137] | [ED-CSP: Crystal Structure Prediction from Electron Diffraction](https://arxiv.org/abs/2608.06448) | ED-CSP是一个新框架，能从电子衍射数据直接预测晶体结构，通过结合关系集编码器和周期性流生成器，在大型模拟数据集上训练，实现了从稀疏观测到完整结构的生成式重建。 |
| [^138] | [Locked Evaluation Surfaces: Transfer Failure and Sampling-Depth Entanglement in CRISPRi Perturbation-Effect Prediction](https://arxiv.org/abs/2608.00152) | 该论文在锁定且预注册的评估协议下评估冻结的Geneformer表示，发现其在虚拟细胞挑战赛（VCC）分布内数据上具有显著超越随机特征对照的预测信息量，但在零样本跨筛选迁移中失败，并揭示了迁移失败与采样深度等设计因素之间的纠缠。 |
| [^139] | [Where Steering Signals Come From: Activation Source Selection in Activation Steering](https://arxiv.org/abs/2607.25270) | 该论文首次将激活引导中常被忽视的“激活源选择”作为核心研究对象，发现引导信号的有效性关键取决于激活是否取自模型即将执行目标行为的“执行边界状态”，而非源文本中是否包含期望行为。 |
| [^140] | [On the Depth Scalability of Logic Gate Networks](https://arxiv.org/abs/2607.21633) | 提出输入锚定逻辑门网络（IALGN），通过让每个门结合私有隐藏主干与直接输入锚点的拓扑设计，解决了逻辑门网络加深时优化崩塌与信用分配退化的问题，在MNIST、CIFAR-10和CIFAR-100上实现了高达150层的一致深度-准确率扩展。 |
| [^141] | [Robust Chance-Constrained Optimization using a Continuous Parameter Space Wasserstein-2 Ambiguity Set of Gaussian Mixtures](https://arxiv.org/abs/2607.17018) | 该论文提出了一种基于连续参数空间的Wasserstein-2模糊集新方法，用于高斯混合模型下的分布鲁棒机会约束优化，允许最坏情况分布内生地确定混合分量的数量与质量分配，克服了传统有限支撑方法只能对拟合的名义混合模型进行压力测试、无法应对结构性误设的局限。 |
| [^142] | [An End-to-End Hybrid Quantum--Classical Sampling Workflow for Discrete Markov Random Fields: A Reproducible Case Study](https://arxiv.org/abs/2607.09893) | 该论文构建了一个可复现的端到端量子-经典混合采样工作流，实证表明现代经典MCMC采样器能大幅缩小与量子采样的差距，且在计入$O(2^n)$预处理成本后，量子方法在实际运行时间上并无优势。 |
| [^143] | [SpecGradFilter: A Spectral Gradient Filtering Framework for Taming Federated Heterogeneity](https://arxiv.org/abs/2607.04189) | 提出频谱梯度过滤框架SpecGradFilter，从频域视角揭示联邦学习客户端漂移主要集中于低频梯度分量的“漂移频谱偏差”，并通过抑制不协调的低频信号来驯服统计异构性。 |
| [^144] | [Closing the Operational Gap in Semantic Caching](https://arxiv.org/abs/2606.19719) | 该论文指出PR-AUC指标会误导语义缓存系统的部署决策，提出了缓存感知的P-CHR AUC指标和运营保留率ORR，并将离线与部署质量间的运营差距分解为可恢复的阈值效用部分和由数据集正例率决定的不可约简结构部分。 |
| [^145] | [The Discrete-Log Clock: How a Transformer Learns Modular Multiplication](https://arxiv.org/abs/2606.17399) | 该论文发现Transformer学习模乘法时嵌入频谱的“稠密性”是在错误基下分析的伪象，改用乘法特征变换后频谱高度稀疏，且96.9%的MLP神经元被调谐到单一乘法频率。 |
| [^146] | [TokenPilot: Cache-Efficient Context Management for LLM Agents](https://arxiv.org/abs/2606.17016) | TokenPilot提出了一种双粒度上下文管理框架，通过全局的感知摄取压缩来稳定提示前缀并消除环境噪声，结合局部的生命周期感知驱逐机制仅在任务相关性过期时卸载内容，从而在降低大语言模型智能体推理成本的同时保持提示缓存的连续性。 |
| [^147] | [RecourseBench: A Modular Framework for Reproducible Algorithmic Recourse Evaluation](https://arxiv.org/abs/2606.16113) | RecourseBench是一个以模块化、可复现性和交互性为核心的算法追索统一评估框架，它将评估流程解耦为五个独立层次，并对所有集成方法进行可复现性分类及其核心实证结论的系统性验证。 |
| [^148] | [ToolSense: A Diagnostic Framework for Auditing Parametric Tool Knowledge in LLMs](https://arxiv.org/abs/2606.12451) | ToolSense是一个开源的LLM驱动诊断框架，通过自动生成包含真实查询在内的三个基准来审计大语言模型的参数化工具知识，揭示了模型在标准ToolBench基准上的优异表现并不能证明其真正理解工具。 |
| [^149] | [ERP-XTTN: Interpretable Prototype-Guided Cross-Attention for Cross-Subject ERP Classification](https://arxiv.org/abs/2606.02939) | 提出ERP-XTTN架构，通过无值投影的查询-键交叉注意力将EEG峰值路由到从差异波自动提取的原型上，实现了无需校准、跨被试且内在可解释的跨范式ERP分类。 |
| [^150] | [Can Subgraph Explanations Be Weaponized to Steal Graph Neural Networks?](https://arxiv.org/abs/2605.30470) | 图机器学习即服务平台中的可解释性接口可被武器化，攻击者仅凭离散类别标签和二值解释掩码即可在严格黑盒约束下窃取图神经网络模型。 |
| [^151] | [LongDS-Bench: On the Failure of Long-Horizon Agentic Data Analysis](https://arxiv.org/abs/2605.30434) | 提出LongDS基准，基于真实Kaggle笔记本构建68个长时程多轮数据分析任务，揭示当前最先进模型在维护和演变分析状态方面存在严重缺陷，最佳模型平均准确率仅48.45%，且长时程错误占失败案例的52%–69%。 |
| [^152] | [Negligible in Size, Significant in Effect: On Scale Vectors in Large Language Models](https://arxiv.org/abs/2605.26895) | 缩放向量虽仅占大语言模型参数的极小部分，但并非用于增强表达能力，而是通过自放大预条件效应改善优化过程，对模型预训练效果至关重要。 |
| [^153] | [More Expressive Feedforward Layers: Part I. Token-Adaptive Mixing of Activations](https://arxiv.org/abs/2605.26647) | 本文提出令牌自适应的激活混合（MoA）前馈层设计，通过轻量级输入相关门控混合多个激活函数，并从理论上证明了其表达能力严格超越可学习激活（LA）和固定激活FFN。 |
| [^154] | [WINO: A Weak-Form Physics Informed Neural Operator for Hyperelasticity on Variable Domains](https://arxiv.org/abs/2605.24651) | 该论文提出WINO——一种无需数据的弱形式物理信息神经算子框架，将神经算子的高效性与φ-FEM的几何灵活性相结合，通过最小化弱形式残差进行训练，能够在无需贴体网格和大规模参考解数据集的情况下求解可变几何域上的超弹性问题。 |
| [^155] | [Learned Relay Representations for Forward-Thinking Discrete Diffusion Models](https://arxiv.org/abs/2605.22967) | 提出Relay方法，通过可学习的逐token通道在去噪步骤之间传递潜在信息，使掩码扩散模型具备前瞻性，避免每轮去噪后的信息硬重置，并可扩展至最先进的扩散语言模型。 |
| [^156] | [ImplicitTerrainV2: Wavelet-Guided Spatially Adaptive Neural Terrain Representation](https://arxiv.org/abs/2605.22556) | 本文提出了ImplicitTerrainV2，一种结合小波引导空间自适应、导数感知监督和模型压缩的紧凑高效神经地形表示方法，解决了现有地形INRs在频率控制、梯度结构和部署成本上的不足。 |
| [^157] | [Online Learning-to-Defer with Varying Experts](https://arxiv.org/abs/2605.12340) | 本文提出了一种将查询动作的老虎机反馈与动态变化专家池相结合的在线多分类学习延迟算法，实现了次线性真实延迟遗憾 $O(T^{2/3})$，并在集中评分条件下提升至 $O(\sqrt T)$。 |
| [^158] | [SkillSafetyBench: Evaluating Agent Safety under Skill-Facing Attack Surfaces](https://arxiv.org/abs/2605.12015) | 该论文提出 SkillSafetyBench 基准，通过 155 个覆盖 6 大风险领域的对抗性案例，首次系统评估了隐藏在技能指导、本地文件等非用户输入中的攻击面，发现此类攻击可稳定诱发大语言模型智能体的不安全行为。 |
| [^159] | [D3-Gym: Constructing Real-World Verifiable Environments for Data-Driven Discovery](https://arxiv.org/abs/2604.27977) | 本文提出了首个为科学数据驱动发现自动构建的可验证环境数据集D3-Gym，包含565个来自真实科学代码库的任务，其自动评估脚本与人工标注达到87.5%的一致性，且基于其轨迹训练可显著提升Qwen3模型在ScienceAgentBench上的表现。 |
| [^160] | [ABC: Any-Subset Autoregression via Non-Markovian Diffusion Bridges in Continuous Time and Space](https://arxiv.org/abs/2604.27443) | 提出ABC模型，通过连续时空中的非马尔可夫扩散桥构建单一连续SDE，使时间变量与中间状态跟踪真实物理时间，从而实现以任意状态子集（如不规则采样或未来观测）为条件的连续时间连续空间随机过程生成。 |
| [^161] | [DiffAnon: Diffusion-based Prosody Control for Voice Anonymization](https://arxiv.org/abs/2604.26281) | DiffAnon是首个基于扩散模型与无分类器引导（CFG）的语音匿名化框架，可在推理时对韵律保留进行显式、连续且可插值的控制，在单一模型内平滑地权衡匿名化强度与韵律保真度。 |
| [^162] | [Budget-Constrained Causal Bandits: Bridging Uplift Modeling and Sequential Decision-Making](https://arxiv.org/abs/2604.26169) | 该论文提出预算约束因果老虎机（BCCB）在线框架，将个体处理效应学习、不确定性探索与预算节奏控制三者统一起来，并基于拉格朗日松弛的 KKT 条件推导出决策规则，从而解决了冷启动场景下数字广告的预算分配问题。 |
| [^163] | [G-Loss: Graph-Guided Fine-Tuning of Language Models](https://arxiv.org/abs/2604.25853) | 提出了一种图引导的损失函数G-Loss，通过结合半监督标签传播与文档相似度图来捕捉全局语义结构，引导预训练语言模型学习更具判别性和鲁棒性的嵌入表示。 |
| [^164] | [Beyond Output Correctness: Benchmarking and Evaluating Large Language Model Reasoning in Coding Tasks](https://arxiv.org/abs/2604.12379) | 该论文提出了首个覆盖代码生成、摘要与分类三类编程任务的推理质量评估基准CodeRQ-Bench，并通过分析评估器失配案例得出设计启示，进而提出结合证据验证与歧义感知评分修正的两阶段评估器VERA，显著提升了编程任务中大语言模型推理质量的评估效果。 |
| [^165] | [SemEnrich: Self-Supervised Semantic Enrichment of Radiology Reports for Vision-Language Learning](https://arxiv.org/abs/2604.09887) | 提出SemEnrich方法，利用自监督语义聚类对放射学报告进行增强，通过添加阳性/中性发现来缓解医学视觉-语言数据集的阴性偏差，并在多项评估指标上取得一致的性能提升。 |
| [^166] | [PolicyLong: Towards On-Policy Context Extension](https://arxiv.org/abs/2604.07809) | PolicyLong 提出动态在线策略的数据构建范式，通过用当前模型迭代重新执行熵计算、检索与验证的数据筛选流程，使训练分布持续跟踪模型能力的演化，从而解决了静态离线数据构建导致的离策略分布漂移问题。 |
| [^167] | [Prompts Without Evidence: How Neuroimaging Mentions Shift Clinical Vision-Language Model Predictions](https://arxiv.org/abs/2603.28387) | 该研究发现在提示词中仅提及神经影像（如MRI）而无需实际提供任何图像，就能显著提升小型视觉语言模型在临床分类任务中的性能与校准度，揭示了这类模型可能依赖表面线索而非真实的影像证据。 |
| [^168] | [Deflation-PINNs: Learning Multiple Solutions for PDEs and Landau-de Gennes](https://arxiv.org/abs/2603.27936) | 提出Deflation-PINNs框架，通过在PINNs与DeepONet结合的架构中引入收缩损失项，系统性地寻找并收敛到非线性偏微分方程（如液晶Landau-de Gennes模型）的有限多个不同解分支，并提供相应的理论逼近结果。 |
| [^169] | [Camera-Agnostic Pruning of 3D Gaussian Splats via Descriptor-Based Beta Evidence](https://arxiv.org/abs/2603.21933) | 本文提出了一种仅基于属性描述符的相机无关3D高斯溅射修剪方法，通过混合描述符和Beta证据模型量化每个溅射的可靠性，实现了无需相机参数的一次性高效修剪。 |
| [^170] | [Var-JEPA: A Variational Formulation of the Joint-Embedding Predictive Architecture - Bridging Predictive and Generative Self-Supervised Learning](https://arxiv.org/abs/2603.20111) | 论文提出Var-JEPA，将JEPA重新诠释为变分推断在耦合潜变量模型上的确定性特例，通过显式优化证据下界（ELBO）弥合了预测式与生成式自监督学习之间的鸿沟。 |
| [^171] | [The Autonomy Tax: Defense Training Breaks LLM Agents](https://arxiv.org/abs/2603.19423) | 该论文揭示了“能力-对齐悖论”：为提升安全性而进行的防御训练会系统性地摧毁LLM智能体在多步工具使用任务中的能力（引发智能体无能偏差与级联放大偏差等三种系统性偏差），同时却无法阻止复杂的提示注入攻击。 |
| [^172] | [InfoMamba: An Attention-Free Hybrid Mamba-Transformer Model](https://arxiv.org/abs/2603.18031) | InfoMamba提出了一种无注意力的Mamba-Transformer混合架构，用概念瓶颈线性过滤层替代自注意力，并通过信息最大化融合（IMF）动态注入全局上下文，从而在线性复杂度下兼顾细粒度局部建模与长程全局依赖捕捉。 |
| [^173] | [Simplex-to-Euclidean Bijection for Conjugate and Calibrated Multiclass Gaussian Process Classification](https://arxiv.org/abs/2603.16621) | 该论文提出利用Aitchison几何将概率单纯形上的类别概率双射映射到欧几里得空间，把多类分类转化为共轭的GP回归问题，从而在无需分布近似的情况下实现共轭推断、良好校准的预测概率以及基于稀疏GP技术的可扩展推断。 |
| [^174] | [Agentic-Kube: A Graph-Enhanced Multi-Agent Reinforcement Learning Framework for Multi-Objective Kubernetes Scheduling](https://arxiv.org/abs/2603.12031) | Agentic-Kube提出了一种图增强的协作式多智能体强化学习框架，将Kubernetes多目标调度分解为由专门子智能体负责的成本最小化、容错和资源平衡三方优化空间，并结合二部图卷积网络与QMIX值分解，解决了传统单智能体强化学习的梯度干扰和奖励稀释问题。 |
| [^175] | [FlowCorrect: Efficient Interactive Correction of Generative Flow Policies for Robotic Manipulation](https://arxiv.org/abs/2602.22056) | FlowCorrect提出一种交互式模仿学习方法，通过轻量级VR界面收集稀疏的人类纠正信号，在部署时对流匹配操作策略进行局部适配，无需重新训练即可修复“擦肩而过”式的失败案例，同时保持已学场景的性能。 |
| [^176] | [Mine and Refine: Optimizing Graded Relevance in E-commerce Semantic Search Retrieval](https://arxiv.org/abs/2602.17654) | 提出两阶段对比训练框架“挖掘与精炼”，借助经参与度审计微调的轻量级LLM标注器，同时解决大规模电商语义搜索中分级相关性建模、硬样本挖掘假阴性以及相似度得分可分性不稳定三大难题。 |
| [^177] | [Robust Assortment Optimization from Observational Data](https://arxiv.org/abs/2602.10696) | 提出了一个鲁棒的数据驱动商品组合优化框架，通过建模顾客选择行为中潜在的分布偏移，克服了传统方法因假设偏好稳定和选择模型正确而在现实中导致的泛化差和收益损失问题。 |
| [^178] | [SCALE: Self-uncertainty Conditioned Adaptive Looking and Execution for Vision-Language-Action Models](https://arxiv.org/abs/2602.04208) | SCALE是一种受主动推理理论启发的推理策略，基于模型自不确定性在单次前向传播中联合自适应调节视觉感知与动作，无需额外训练或验证器即可提升VLA模型在感知模糊情境下的鲁棒性与部署实用性。 |
| [^179] | [Learning Fast Monomial Orders for Gr\"obner Basis Computations](https://arxiv.org/abs/2602.02972) | 该论文将Gröbner基计算中单项式序的选择建模为强化学习问题，所学策略在系统生物学和计算机视觉基准问题上持续超越GrevLex等标准启发式方法，大幅降低计算成本。 |
| [^180] | [Bayesian Experimental Design for Model Discrepancy Calibration: A Rivalry between Kullback--Leibler Divergence and Wasserstein Distance](https://arxiv.org/abs/2601.16425) | 本文通过玩具示例揭示了Wasserstein距离作为贝叶斯实验设计效用函数的缺陷——固定形状后验的距离值取决于其主体质量在支撑集内的相对位置，可能产生与信息增益无关的虚假奖励，凸显了KL散度在模型差异校准实验设计中的优势。 |
| [^181] | [Aligning Agentic World Models via Knowledgeable Experience Learning](https://arxiv.org/abs/2601.13247) | 提出WorldMind框架，通过综合环境反馈自主构建符号化世界知识库，使LLM智能体世界模型无需昂贵再训练即可遵循物理法则、避免物理幻觉。 |
| [^182] | [The Instability of Safety: How Random Seeds and Temperature Expose Inconsistent LLM Refusal Behavior](https://arxiv.org/abs/2512.12066) | 该研究揭示大语言模型的安全拒绝决策在随机种子和温度变化下并不稳定，18-28%的有害提示词会出现“拒绝”与“配合”之间的决策翻转，且温度越高稳定性越差，表明单次安全评估无法真实反映模型的安全对齐水平。 |
| [^183] | [Prequential posteriors](https://arxiv.org/abs/2511.17721) | 本文提出基于预测序列损失函数的prequential后验方法，解决了深度生成预测模型因似然函数不可解而无法应用标准贝叶斯数据同化的难题，并证明了其在温和条件下的理论一致性保证。 |
| [^184] | [Aspiration-based Perturbed Learning Automata in Games with Noisy Utility Measurements. Part A: Stochastic Stability in Non-zero-Sum Games](https://arxiv.org/abs/2511.11602) | 本文提出了一种新颖的基于收益的分布式学习方案——基于期望的扰动学习自动机（APLA），使多人弱无环非零和博弈中各参与者独立学习时也能收敛到纯纳什均衡，突破了以往方法仅适用于势博弈和协调博弈的局限。 |
| [^185] | [Think-at-Hard: Dynamic Looped Transformers for Improved Reasoning](https://arxiv.org/abs/2511.08577) | 针对循环Transformer中存在的“潜在过度思考”现象，提出TaH方法，利用轻量级神经决策器仅在可能出错的token处动态触发潜在迭代，从而在参数受限条件下将大语言模型的推理性能提升高达7.3%。 |
| [^186] | [One Model for All: Universal Pre-training for EEG based Emotion Recognition across Heterogeneous Datasets and Paradigms](https://arxiv.org/abs/2511.08444) | 提出了一种跨异构EEG数据集与范式的通用预训练框架"One Model for All"，通过统一通道模式进行单通道自监督对比学习预训练，再结合ART与GAT架构进行多变量微调，有效解决了数据集异构性难题，并在SEED、DEAP和DREAMER数据集上取得显著性能提升。 |
| [^187] | [Multilingual Lexical Feature Analysis of Spoken Language for Predicting Major Depression Symptom Severity](https://arxiv.org/abs/2511.07011) | 该研究基于来自英国、荷兰和西班牙467名参与者的多语言智能手机录音数据，利用可解释的线性混合效应模型识别出与抑郁症状严重程度相关的口语词汇特征，并通过机器学习验证了这些特征对PHQ-8评分预测的增益作用。 |
| [^188] | [GREAT: Generalizable Backdoor Attacks in RLHF via Emotion-Aware Trigger Synthesis](https://arxiv.org/abs/2510.09260) | GREAT框架通过在模型潜在嵌入空间中利用降维与聚类技术识别愤怒情绪触发器，并构建包含5000余个触发器的Erinyes数据集，实现了针对RLHF的自然分布、可泛化后门攻击。 |
| [^189] | [Large Reasoning Models Learn Better Alignment from Flawed Thinking](https://arxiv.org/abs/2510.00938) | RECAP 是一种基于强化学习的后训练方法，通过在合成生成的反向对齐（有缺陷）思维链预填充上训练，教模型识别并覆盖错误的推理轨迹、转向安全有用的回答，从而在不增加额外训练成本的情况下显著提升安全性与抗越狱鲁棒性，同时减少过度拒绝并保留核心推理能力。 |
| [^190] | [OceanGym: A Benchmark Environment for Underwater Embodied Agents](https://arxiv.org/abs/2509.26536) | OceanGym是首个面向水下具身智能体的综合性基准环境，涵盖八个真实任务领域和基于多模态大语言模型的统一智能体框架，实验揭示当前最先进的智能体与人类专家之间仍存在显著差距。 |
| [^191] | [Examining the robustness of Physics-Informed Neural Networks to noise for Inverse Problems](https://arxiv.org/abs/2509.20191) | 本研究通过在一维Burgers方程和二维/三维Taylor-Green涡的粘度识别反问题上，将PINNs与结合数值优化器的有限元传统方法在含加性高斯噪声数据下的表现进行对比，系统检验了PINNs对噪声的鲁棒性。 |
| [^192] | [Probabilistic Symbolic Regression for Equation Discovery via Operator-induced and Regularized Symbolic Forests](https://arxiv.org/abs/2509.19710) | 该论文提出一种概率符号回归框架，将数学表达式表示为符号树集成，通过树拓扑上的正则化先验控制表达式复杂度，并利用基于奥卡姆窗口的后验摘要刻画多个合理符号模型的不确定性，为方程发现提供了兼具精度、简洁性与不确定性量化的统一解决方案。 |
| [^193] | [Shift Before You Learn: Enabling Low-Rank Representations in Reinforcement Learning](https://arxiv.org/abs/2509.05193) | 该论文发现后继测度本身并非低秩，但跳过初始转移的“移位后继测度”自然具有低秩结构，并为其低秩估计提供了有限样本保证，误差由新提出的“谱可恢复性”指标刻画。 |
| [^194] | [Automatic Pronunciation Error Detection and Correction of the Holy Quran's Learners Using Deep Learning](https://arxiv.org/abs/2509.00094) | 该论文提出了一套98%自动化的《古兰经》诵读数据构建流程，发布了848小时音频数据集（28.6万条标注语句）以及涵盖Tajweed规则的基准qdat_bench，实现了对《古兰经》学习者发音错误的自动检测与纠正。 |
| [^195] | [Class Incremental Continual Learning with Self-Organizing Maps and Synthetic Replay](https://arxiv.org/abs/2508.21240) | 提出一种基于自组织映射（SOM）的生成式持续学习框架，通过为每个SOM单元存储分布统计量来生成合成样本进行重放，在不保存原始数据的情况下实现类增量学习。 |
| [^196] | [Attention as Conditioning: What Classical Learning Theory Predicts About Linear Transformers](https://arxiv.org/abs/2508.08289) | 论文揭示线性注意力的状态更新与动物学习理论中的经典模型（Hebbian接近性、Rescorla-Wagner误差校正、刺激痕迹衰减）逐项等价，从而将条件反射现象转化为线性Transformer上下文行为的可检验预测。 |
| [^197] | [RegCL: Compact Continual SAM Adaptation for Visual Grounding in Multi-Sensorial Media](https://arxiv.org/abs/2507.12297) | 提出RegCL框架，通过增量模型合并将多领域分割知识整合进单个SAM适配器，无需重放数据即可实现多感官媒体中紧凑高效的持续视觉定位适配。 |
| [^198] | [Ampere: Communication-Efficient and High-Accuracy Split Federated Learning](https://arxiv.org/abs/2507.07130) | Ampere是一种新型分割联邦学习系统，通过基于局部损失的单向块间训练消除了梯度传输，在最小化设备端计算和通信开销的同时，提高了非独立同分布数据下的模型精度。 |
| [^199] | [Meta-Prompt Optimization for LLM-Based Sequential Decision Making](https://arxiv.org/abs/2502.00728) | 该论文提出EXPO算法，借鉴对抗性老虎机算法处理非平稳奖励观测的能力，实现基于大语言模型的序贯决策智能体元提示词的自动优化。 |
| [^200] | [Off the Normal Path: Learning Spatial Density Models of Node Mobility](https://arxiv.org/abs/2411.10997) | 该论文引入Möbius分布混合模型来学习二维地形上移动节点的稳态空间密度，相比混合密度网络和归一化流等现成方法，提供了更可解释、更简洁且性能相当或更优的模型。 |
| [^201] | [Mixture of Multicenter Experts in Multimodal AI for Debiased Radiotherapy Target Delineation](https://arxiv.org/abs/2410.00046) | 提出多中心专家混合框架，无需跨机构数据共享即可整合多中心临床专业知识、解决医学AI偏见，在前列腺癌放疗靶区勾画任务中显著提升了模型的泛化能力和适应性。 |
| [^202] | [Rethinking Speaker Embeddings for Speech Generation: Sub-Center Modeling for Capturing Intra-Speaker Diversity](https://arxiv.org/abs/2407.04291) | 该论文提出一种说话人嵌入的子中心建模框架，通过为每位说话人学习多个子中心而非单一原型来保留说话人内部的结构化多样性，从而在零样本语音转换中提升可懂度、音高变化性和自然度，同时保持说话人验证性能。 |
| [^203] | [Amortizing intractable inference in diffusion models for vision, language, and control](https://arxiv.org/abs/2405.20971) | 本文提出“相对轨迹平衡”这一具有渐近正确性保证的无数据学习目标，借助生成流网络视角与深度强化学习技术，训练扩散模型以摊销方式从扩散先验与黑盒约束构成的后验分布中进行精确采样，从而解决视觉、语言与控制任务中的难解后验推断问题。 |
| [^204] | [On diffusion models for amortized inference: Benchmarking and improving stochastic control and sampling](https://arxiv.org/abs/2402.05098) | 本研究探讨了训练扩散模型以从给定分布中采样的问题，并针对随机控制和采样提出了一种新的探索策略，通过基准测试比较了不同推断方法的相对优劣，并对过去的工作提出了质疑。 |
| [^205] | [Diffusion models as plug-and-play priors](https://arxiv.org/abs/2206.09012) | 本文提出将独立训练的扩散模型作为即插即用的先验模块，通过与可微辅助约束结合并在去噪网络上进行迭代微分实现近似推理，从而支持条件生成、图像分割等新任务的应用。 |
| [^206] | [Biases in Expected Goals Models Confound Finishing Ability.](http://arxiv.org/abs/2401.09940) | 本研究旨在解决使用期望进球（xG）统计评估射门能力时的限制和偏见。研究发现，持续超出累积xG需要高射门频率。 |
| [^207] | [Joint Bayesian Inference of Graphical Structure and Parameters with a Single Generative Flow Network.](http://arxiv.org/abs/2305.19366) | 本文提出了在单一生成流网络中联合建模贝叶斯网络结构和参数的方法，包括非离散样本空间，提高了贝叶斯网络局部概率模型的灵活性。 |
| [^208] | [Let the Flows Tell: Solving Graph Combinatorial Optimization Problems with GFlowNets.](http://arxiv.org/abs/2305.17010) | 本文提出了一种名为GFlowNets的机器，可以有效地解决组合优化问题，同时在训练方面进行了优化，结果表明其可以高效地找到高质量的解决方案。 |
| [^209] | [Transformer-based models and hardware acceleration analysis in autonomous driving: A survey.](http://arxiv.org/abs/2304.10891) | 本文综述了基于Transformer的模型在自动驾驶中的应用，探讨了不同体系结构和运算符的优缺点，重点讨论了针对便携计算平台的硬件加速方案，并对卷积神经网络和Transformer的层进行了对比。 |
| [^210] | [Trajectory balance: Improved credit assignment in GFlowNets.](http://arxiv.org/abs/2201.13259) | GFlowNets使用轨迹平衡作为一种更高效的学习目标，解决了先前学习目标中信用传播效率低下的问题，并且在实验中证明了其在收敛性、生成样本多样性以及鲁棒性方面的优势。 |

# 详细

[^1]: QGPINNs：量子图上非局部微分方程的物理信息神经网络框架

    QGPINNs: A Physics-Informed Neural Network Framework for Nonlocal Differential Equations on Quantum Graphs

    [https://arxiv.org/abs/2608.28589](https://arxiv.org/abs/2608.28589)

    提出了QGPINNs——一个基于PyTorch的物理信息神经网络框架，通过逐边神经网络逼近和统一的图损失函数（融合连续性、Kirchhoff-Neumann顶点条件与Dirichlet边界条件），实现量子图上多阶分数阶椭圆问题和时间分数阶演化方程等非局部微分方程的数值求解。

    

    我们提出了QGPINNs，这是一个基于PyTorch开发的物理信息神经网络框架，用于量子图上非局部微分方程的数值求解。该框架被设计为一种通用的计算实现，其中图每条边上的解由神经网络逼近，同时统一的基于图的损失函数将控制方程与初始条件、边界条件及顶点传输条件共同纳入约束。特别地，该公式将标准的连续性与Kirchhoff-Neumann顶点条件以及Dirichlet边界条件融入学习过程，从而将局部的逐边神经逼近耦合为图上的全局解。该框架针对两类具有代表性的非线性模型开发：量子图上的多阶分数阶椭圆问题以及时间分数阶演化方程。为提高精度与训练稳定性，QGPINNs集成（原文此处截断）……

    arXiv:2608.28589v1 Announce Type: new  Abstract: We propose QGPINNs, a physics-informed neural network framework developed in PyTorch for the numerical solution of nonlocal differential equations on quantum graphs. The framework is designed as a general computational implementation in which the solution on each edge of the graph is approximated by a neural network, while a unified graph-based loss function enforces the governing equations together with initial, boundary, and vertex transmission conditions. In particular, the formulation incorporates standard continuity and Kirchhoff-Neumann vertex conditions and Dirichlet boundary conditions into the learning process to couple the local edge-wise neural approximations into a global solution on the graph. The framework is developed for two representative classes of nonlinear models: multi-order fractional elliptic problems and time-fractional evolution equations on quantum graphs. To improve accuracy and training stability, QGPINNs inte
    
[^2]: Aero Hand Open：一款面向灵巧操作学习的仿真就绪腱驱动灵巧手

    Aero Hand Open: A Simulation-Ready Tendon-Driven Hand for Dexterous Manipulation Learning

    [https://arxiv.org/abs/2608.28578](https://arxiv.org/abs/2608.28578)

    提出了Aero Hand Open——一款仿真就绪的腱驱动拟人灵巧手，附带可复现缆绳传动的仿真模型和双向辨识执行映射，解决了腱驱动手在灵巧操作学习中的仿真建模难题。

    

    腱驱动灵巧手具有拟人化结构，而将执行器从关节处移开，正是这类高性能手部能够以低成本制造的关键。这种成本节约来自两方面：通过缆绳传递力，使得电机无需安装在所驱动的关节内部，因此可以使用更小、更便宜的电机；同时一个电机可以通过单根缆绳驱动多个关节，从而减少所需电机的数量。然而，与直驱灵巧手相比，腱驱动手更难以用于学习。产生成本节约的欠驱动传动系统本身在仿真器中就难以建模，而且由同一根缆绳驱动的关节无法被独立控制。我们提出了Aero Hand Open，一款以仿真就绪状态发布的腱驱动拟人灵巧手。该产品附带三项内容：一个能够复现缆绳传动本身的仿真模型；一个经过辨识的执行映射，可在两个方向上将该模型与电机指令相互连接，包括（原文在此处截断）

    arXiv:2608.28578v1 Announce Type: cross  Abstract: Tendon-driven hands are anthropomorphic, and moving the actuators off the joints is what makes a hand of this capability affordable to build. Two effects produce that saving. Routing force through a cable removes the requirement that a motor fit inside the joint it drives, so smaller and cheaper motors suffice, and one motor can drive several joints through a single cable, so fewer motors are needed. They are also harder to learn on than a direct-drive hand. The underactuated transmission that produces the saving is itself difficult to represent in a simulator, and the joints one cable drives are not independently commandable. We present Aero Hand Open, a tendon-driven anthropomorphic hand that is released simulation-ready. Three things ship with it. A simulation model reproduces the cable transmission itself. An identified actuation map connects that model to the motor commands in both directions, including the three-way coupling of t
    
[^3]: 学习合成增强推断中的规模-权重前沿

    Learning a Size-Weight Frontier for Synthetic-Augmented Inference

    [https://arxiv.org/abs/2608.28576](https://arxiv.org/abs/2608.28576)

    提出合成增强推断框架，通过从历史任务中学习“规模-权重前沿”，为所有规模-权重配置提供有限样本覆盖保证，在真实数据稀缺时安全利用合成数据并显著收窄置信区间。

    

    当真实数据稀缺时，合成数据可以改善统计推断，但简单地将合成样本当作真实数据会引入偏差并导致不可靠的推断。我们开发了一个面向相关任务总体的合成增强推断通用框架。该框架通过合成观测的数量及其权重来刻画合成增强。框架的核心是一个规模-权重前沿，它为每个权重指定了最大的合成样本量，使得所有不大于该样本量的配置都能达到目标任务边际覆盖率。我们从历史任务中估计这一前沿，并对估计前沿上或以下的所有规模-权重配置同时建立了有限样本覆盖保证。在使用大语言模型响应来增强舆论调查数据的实验中，我们的方法实现了目标覆盖率，并大幅收窄了置信区间。

    arXiv:2608.28576v1 Announce Type: cross  Abstract: Synthetic data can improve statistical inference when real data are scarce, but naively treating synthetic samples as real data can introduce bias and lead to unreliable inference. We develop a general framework for synthetic-augmented inference across a population of related tasks. It characterizes synthetic augmentation by the number of synthetic observations and their weight. Central to our framework is a size-weight frontier that specifies, for each weight, the largest synthetic sample size for which all smaller sizes attain the target task-marginal coverage. We estimate this frontier from historical tasks, and establish a finite-sample coverage guarantee simultaneously for all size-weight configurations on or below the estimated frontier. In experiments using large language model responses to augment opinion survey data, our procedure achieves target coverage and substantially narrows confidence intervals.
    
[^4]: 关于加权Dikin行走 $d^2$ 混合时间的两种证明方法

    On two proofs of $d^2$ mixing of weighted Dikin walks

    [https://arxiv.org/abs/2608.28566](https://arxiv.org/abs/2608.28566)

    该论文提出了两种证明方法——在高概率区域上控制Metropolis-Hastings接受概率以及基于新的四阶自举条件的 $\chi^2$-散度分析——证明了加权Dikin行走从多面体采样可在 $\widetilde O(d^2)$ 步内实现混合。

    

    我们研究了加权Dikin行走的混合时间，用于从多面体和截断半正定（PSD）锥上的指数分布中采样。我们的第一个结果在强自协和性、$\bar{\nu}$-对称性以及局部度量的混合迹正则性条件下，给出了一个一般的全变差混合界。其关键思想是在高概率区域上控制Metropolis--Hastings接受概率，而非在每一点上进行控制。将这一框架应用于Lee--Sidford度量、Lewis权重度量和John度量，可以得到从多面体采样的 $\widetilde O(d^2)$ 混合界；而将其应用于混合障碍函数，则得到从截断PSD锥采样的 $\widetilde O(d^4)$ 混合界。我们的第二个结果利用一个新的四阶自举条件，建立了更强的 $\chi^2$-散度保证和逐点接受控制。对于适当缩放的Lee--Sidford度量，这给出了 $\widetilde O(d^2)$ 的混合（摘要在此处被截断）。

    arXiv:2608.28566v1 Announce Type: cross  Abstract: We study the mixing time of weighted Dikin walks for sampling from exponential distributions on polytopes and truncated positive-semidefinite (PSD) cones. Our first result gives a general total-variation mixing bound under strong self-concordance, $\bar{\nu}$-symmetry, and mixed-trace regularity on the local metric. The key idea is to control the Metropolis--Hastings acceptance probability on a high-probability region rather than at every point. Applying this framework to the Lee--Sidford, Lewis-weight, and John metrics yields an $\widetilde O(d^2)$ mixing bound for sampling from polytopes, while applying it to a hybrid barrier yields an $\widetilde O(d^4)$ mixing bound for sampling from truncated PSD cones. Our second result establishes stronger $\chi^2$-divergence guarantees and pointwise acceptance control using a new fourth-order bootstrap condition. For a suitably scaled Lee--Sidford metric, this yields an $\widetilde O(d^2)$ mixi
    
[^5]: 峰间学习：幂律各向异性下核岭回归的精确渐近分析

    Learning between the peaks: sharp asymptotics for kernel ridge regression under power-law anisotropy

    [https://arxiv.org/abs/2608.28564](https://arxiv.org/abs/2608.28564)

    该论文针对幂律各向异性高斯数据下的核岭回归，推导出核谱与泛化误差的渐近精确表达式，揭示了弱各向异性会使方差峰值随 α 增大而逐渐衰减、且与主方向对齐的目标的偏差在分数样本复杂度处下降并与插值峰值解耦，而强各向异性（α>1）则会改变有效维数。

    

    我们研究了各向异性高斯数据下的核岭回归问题，其中对于多项式内积核，输入协方差以指数 α≥0 的幂律衰减。我们在多项式高维区域 n=Θ(d^κ) 下推导出了核谱与泛化误差的渐近精确表达式，揭示了各向异性如何重塑学习曲线。对于弱各向异性（0<α<1），问题在本质上仍是高维的，既保留了各向同性情形的某些特征，又在其他方面有所偏离：方差仍然在整数样本复杂度 κ∈ℕ 处出现峰值，但随着 α 的增大，这些峰值会被逐渐衰减；同时，对于与数据主方向高度对齐的目标函数，偏差会在分数样本复杂度处下降，从而使偏差的转变与插值峰值解耦。对于强各向异性（α>1），有效维数……（原文摘要在此处截断）

    arXiv:2608.28564v1 Announce Type: cross  Abstract: We study kernel ridge regression under anisotropic Gaussian data, where the input covariance decays as a power law with exponent $\alpha\geq 0$ for polynomial inner-product kernels. We derive asymptotically sharp expressions for the kernel spectrum and the generalization error in the polynomial high-dimensional regime $n=\Theta(d^\kappa)$, revealing how anisotropy reshapes the learning curves. For weak anisotropy ($0<\alpha<1$), the problem remains effectively high-dimensional and retains some features of the isotropic case, while departing from it in others: the variance still peaks at integer sample complexities $\kappa\in\mathbb{N}$, but these peaks are progressively damped as $\alpha$ grows; meanwhile, for targets strongly aligned with the data's principal directions, the bias drops at fractional sample complexities, decoupling the bias transitions from the interpolation peaks. For strong anisotropy ($\alpha > 1$), the effective di
    
[^6]: 博客：优化器综述

    Blog: Survey of Optimizers

    [https://arxiv.org/abs/2608.28557](https://arxiv.org/abs/2608.28557)

    本综述从时间估计、更新几何、周期管理和表示与系统四个维度系统梳理了2025-2026年的优化器进展，指出以Muon、Shampoo、SOAP为代表的矩阵感知方法是真正的进步，但尚不存在适用于所有场景的通用最优优化器。

    

    2025-2026年的神经网络优化已不能再被简单地描述为一系列新Adam变体的更迭。设计空间已从坐标扩展到矩阵和层，从固定的训练周期扩展到随时间变化的策略，从数学更新规则扩展到必须在分片（sharding）和低精度计算中保持稳健的状态表示。本综述沿着四个基本独立的维度组织近期的优化器和训练优化方法：时间估计、更新几何、周期管理以及表示与系统。它串联了Muon的谱归一化、Shampoo和SOAP的历史矩阵统计、自适应与混合矩阵方法、内存高效优化器、无调度（schedule-free）训练、小批量修正以及量化优化器状态。核心经验结论刻意保持审慎：矩阵感知方法代表了真正的进步，但并不存在与具体场景无关的通用替代方案。

    arXiv:2608.28557v1 Announce Type: new  Abstract: Neural-network optimization in 2025-2026 is no longer well described as a succession of new Adam variants. The design space has expanded from coordinates to matrices and layers, from fixed training horizons to policies over time, and from mathematical update rules to state representations that must survive sharding and low-precision computation. This survey organizes recent optimizers and training optimization methods along four largely independent axes: temporal estimation, update geometry, horizon management, and representation and systems. It connects the spectral normalization of Muon, the historical matrix statistics of Shampoo and SOAP, adaptive and hybrid matrix methods, memory-efficient optimizers, schedule-free training, small-batch corrections, and quantized optimizer states. The central empirical conclusion is deliberately non-triumphal: matrix-aware methods represent a genuine advance, but there is no context-independent repl
    
[^7]: 推进交互敏感特征选择：新型基于Relief的算法、扩展的比较研究以及面向生物医学数据挖掘的建议

    Advancing Interaction-Sensitive Feature Selection: Novel Relief-Based Algorithms, Expanded Comparisons, and Recommendations for Biomedical Data Mining

    [https://arxiv.org/abs/2608.28552](https://arxiv.org/abs/2608.28552)

    本研究重构并扩展了scikit-rebate Python包，提出了5种新型基于Relief的交互敏感特征选择算法变体，并通过多样化的基因组模拟基准测试为生物医学数据挖掘提供了比较结果与实用建议。

    

    作为高维生物医学数据建模的前置步骤，可靠的特征选择能够降低计算成本、提升建模性能，并产生更简单、更易解释的模型。然而，大多数基于过滤式的特征选择方法难以检测特征间的交互作用，而封装式或嵌入式特征选择方法则计算成本高昂。基于Relief的算法（RBAs）是一类对特征交互敏感的过滤式方法，同时能够缓解上述其他局限性。本研究（1）重构、优化并扩展了scikit-rebate Python包，纳入了现有及新提出的RBA变体；（2）在多样化的基因组模拟中开展了严格的RBA基准比较。我们扩展了scikit-rebate，新增了SWRF*、mu-Relief以及5种采用替代性邻居选择和特征评分策略的新型RBA变体。所有RBA均被评估以比较其预测性特征排名能力……

    arXiv:2608.28552v1 Announce Type: new  Abstract: As a precursor to high-dimensional biomedical data modeling, reliable feature selection can reduce computational expense, improve modeling performance, and yield simpler, more interpretable models. However, most filter-based feature selection methods struggle to detect feature interactions, while wrapper or embedded feature selection methods are computationally expensive. Relief-based algorithms (RBAs) are filter methods that are sensitive to feature interactions while mitigating these other limitations. This study (1) refactors, optimizes, and expands the scikit-rebate Python package with existing and newly proposed RBA variants and (2) conducts rigorous RBA benchmark comparisons across diverse genomic simulations. We expand scikit-rebate to include SWRF*, mu-Relief, and 5 novel RBA variants implementing alternative strategies for neighbor selection and feature scoring. All RBAs were evaluated to compare predictive feature ranking and r
    
[^8]: DARTS：面向模型合并的基于手术式解码器感知表示微调

    DARTS: Decoder-Aware Representation Tuning via Surgery for Model Merging

    [https://arxiv.org/abs/2608.28547](https://arxiv.org/abs/2608.28547)

    该论文提出DARTS方法，首次分析解码器模型合并中的表示偏差问题，指出因果注意力掩码导致偏差跨位置累积以及高熵决策关键位置更为重要这两大挑战，并通过手术式的解码器感知表示微调来校正偏差。

    

    模型合并能够在无需额外训练的情况下，将多个针对特定任务微调的大语言模型（LLM）合并为单个多任务模型。然而，众所周知，合并后的模型会遭受表示偏差问题：即合并模型的隐藏状态与各个源模型的隐藏状态之间存在系统性漂移。先前的工作（Yang等人，2024a）使用由L1损失训练的轻量级校正模块，研究并缓解了基于编码器的视觉模型中的这种偏差。然而，由于解码器模型的自回归特性，这种偏差尚未在解码器模型中得到研究。我们分析了解码器模型中的表示偏差问题，并指出了编码器中不存在的两个挑战：（1）因果注意力掩码导致偏差在各个token位置之间累积，需要位置相关的校正；（2）并非所有token位置都同等重要，即高熵（决策关键）位置远比低熵位置更重要。为解决这些挑战，我们提出Deco（摘要在此处截断）

    arXiv:2608.28547v1 Announce Type: new  Abstract: Model merging combines multiple task-specific fine-tuned LLMs into a single multi-task model without additional training. However, merged models are known to suffer from representation bias: systematic drift between the merged model's hidden states and those of each individual source model. Prior work (Yang et al., 2024a) study and mitigate this bias for encoder-based vision models using a lightweight correction module trained with L1 loss. However, such bias is not studied for decoder models due to their autoregressive nature. We analyze the problem of representation bias in decoder models, and show two challenges absent in encoders: (1) the causal attention mask causes bias to accumulate across token positions, requiring position-dependent correction; and (2) not all token positions are equally important, i.e., high-entropy (decision-critical) positions matter far more than low-entropy ones. To address these challenges, we propose Deco
    
[^9]: 封闭模式是一种规范选择：认证代码世界模型中相对于可达性的拓扑

    An Enclosed Mode Is a Gauge Choice: Topology Relative to Reach in Certified Code World Models

    [https://arxiv.org/abs/2608.28541](https://arxiv.org/abs/2608.28541)

    该论文提出“危险是相对于可达性的拓扑”这一核心原则，证明认证代码世界模型中超出可达范围的封闭错误模式是一种规范选择——错误拓扑的伪影在不可达时不可证伪且无害，而一旦通过宽度为γ的通道变得可达，就会依次经历可证伪且有代价、立即被证伪等状态。

    

    通过采样门认证的代码世界模型，可以在门所能观察到的一切上完全正确，而在其之外任意错误。当遗漏是一个包围着不可达内部的环形冻结模式时，我们刻画了认证模型能够知道什么，以及其错误可能造成什么代价。门商（gate quotient）使这一问题变得精确：确定性的接受在可达查询集上精确地确定模型；超出可达范围的部分则是规范自由度。在一个最小环形实验装置上，我们证明了极端情形（一种拓扑错误的填充圆盘伪影，任何采样门都无法证伪，且在运行中逐位无害），并通过跨三个模型族的LLM合成实验，测量了一个旋钮（宽度为γ的通道）如何使同一伪影经历三种状态：不可证伪且无害、可证伪且有代价、以及立即被证伪。三条原则组织了这些实证结果。首先，危险是相对于可达性的拓扑：规划者可以使用的通道会坍缩……

    arXiv:2608.28541v1 Announce Type: new  Abstract: A code world model accepted by a sampling gate can be exactly right on everything the gate can see and arbitrarily wrong beyond it. We characterize what a certified model can know, and what its errors can cost, when the omission is an annular freeze mode enclosing an unreachable interior. The gate quotient makes the question precise: acceptance-with-certainty determines the model exactly on the reachable query set; beyond reach is gauge. On a minimal ring instrument we prove the extreme case (a wrong-topology filled-disc artifact unfalsifiable by any sampling gate and bitwise harmless at play) and measure, with LLM synthesis across three model families, how one knob (a channel of width gamma) walks the same artifact through three regimes: unfalsifiable-and-harmless, falsifiable-and-costly, and instantly falsified. Three principles organize the empirics. First, danger is topology relative to reach: a channel the planner can use collapses 
    
[^10]: REPLICANT：学习规避和加固恶意软件检测器的策略

    REPLICANT: Learning Policies for Evading and Hardening Malware Detectors

    [https://arxiv.org/abs/2608.28499](https://arxiv.org/abs/2608.28499)

    提出深度强化学习框架Replicant，在严格的仅标签黑盒威胁模型下学习可跨样本、检测器和特征空间迁移的恶意软件规避策略，在七个Android恶意软件检测器上实现78.8%的平均攻击成功率，较现有最优方法相对提升20.9%至39.2%。

    

    为了确定基于机器学习的恶意软件检测在现实世界中的有效性，评估其面对高能力对手时的鲁棒性至关重要。然而，当前最先进的攻击方法并不能有效模拟现实中的对手，因为它们通常假设攻击者可以获取特权信息，例如目标的训练数据、特征空间或置信度分数。在这项工作中，我们提出了Replicant，一个深度强化学习框架，它在严格的仅标签（label-only）黑盒威胁模型下学习现实的规避任务。Replicant学习了一个可复用的策略，用于决定如何修改恶意软件样本以及何时查询目标，该策略可以在不同样本、检测器和特征空间之间迁移。在七个Android恶意软件检测器和三种特征空间上的实验中，Replicant是最强且查询效率最高的方法，平均攻击成功率达到78.8%，相比最先进方法有20.9%至39.2%的相对提升。

    arXiv:2608.28499v1 Announce Type: new  Abstract: To determine the real-world effectiveness of machine learning based malware detection, it is vital to evaluate its robustness against highly capable adversaries. However, state-of-the-art attacks do not effectively model realistic adversaries, as they often assume access to privileged information such as the training data, feature space, or confidence scores of the target. In this work, we present Replicant, a deep reinforcement learning framework that learns the realistic task of evasion under a strict label-only black-box threat model. Replicant learns a reusable policy on how to modify a malware sample and when to query the target, which transfers across samples, detectors, and feature spaces. Across seven Android malware detectors and three feature spaces, Replicant is the strongest and most query-efficient approach achieving a mean attack success rate of 78.8%, a relative improvement of 20.9%-39.2% over the state-of-the-art. Further
    
[^11]: 适当的评分规则如何塑造大语言模型预测

    How Proper Scoring Rules Shape LLM Forecasting

    [https://arxiv.org/abs/2608.28482](https://arxiv.org/abs/2608.28482)

    尽管五种适当的评分规则在理论上都激励真实概率报告，但作为大语言模型预测的训练目标时，会训练出在校准、偏差、信息和噪声特征上各不相同的模型，表明奖励函数的选择并非可以互换。

    

    本文评估了奖励函数的选择如何塑造大语言模型预测器的性能与行为。我们比较了五种适当的评分规则作为训练目标，用于对已有结果的现实世界事件进行二元预测。尽管这些规则在理论上都具有激励真实概率报告的相同特性，但由此训练出的模型在校准、概率使用以及偏差、信息和噪声的估计特征方面存在差异，而在总体准确率和区分度上的差异较小。使用Brier规则训练的模型具有最低的观测Brier分数和最高的AUC-ROC，而使用对数规则训练的模型具有最高的观测对数分数和最低的校准误差。总体性能相似的模型也是通过偏差、信息和噪声的不同组合来达到该性能的。因此，适当的评分规则作为训练目标时并不必然可以互换。奖励的选择不仅可能影响大语言模型预测的好坏，还可能……

    arXiv:2608.28482v1 Announce Type: new  Abstract: This paper evaluates how reward function choice shapes the performance and behavior of LLM forecasters. We compare five proper scoring rules as training objectives for binary forecasts of resolved real-world events. Although the rules share the same theoretical incentive for truthful probability reporting, the resulting models differ in calibration, probability use, and estimated profiles of bias, information, and noise, with smaller differences in aggregate accuracy and discrimination. The Brier-trained model has the lowest observed Brier score and highest AUC-ROC, while the log-trained model has the highest observed log score and lowest calibration error. Models with similar aggregate performance also reach that performance through different combinations of bias, information, and noise. Proper scoring rules therefore need not behave interchangeably as training objectives. Reward choice may shape not only how well an LLM forecasts, but 
    
[^12]: 获取、修复、保持：一种面向小型模型对话游戏智能体的诊断引导式后训练方案

    Acquire, Repair, Preserve: A Diagnosis-Guided Post-Training Recipe for Small-Model Dialogue Game Agents

    [https://arxiv.org/abs/2608.28458](https://arxiv.org/abs/2608.28458)

    该论文提出一种诊断引导的三步后训练方案（获取、修复、保持），使2B小型模型在对话游戏挑战中的clemscore从10.67大幅提升至38.92，同时保持了一般能力不退化。

    

    交互式对话游戏考验一种静态基准测试大多未能明确涉及的能力：模型必须在多轮对话中携带状态、解读反馈，并在不断变化的约束下选择有效动作。我们在LM Playschool Challenge中使用2B参数的开源权重模型研究这一场景，发现许多失败不仅是广泛的知识性失败，还包括局部决策失败：重复猜测、格式错误的动作，以及违反模型刚刚看到的反馈。这些诊断结果启发了一种围绕三个步骤组织的训练方案：通过监督微调获取广泛的游戏参与能力，使用轮次局部的偏好对在单一目标对话游戏族内修复可机械验证的失败，并保持这些对话游戏之外的一般能力。在官方最终评估中，我们的提交将公开clemscore从10.67提升至38.92，封闭域内得分从13.41提升至41.17，同时……

    arXiv:2608.28458v1 Announce Type: new  Abstract: Interactive dialogue games test a capability that static benchmarks largely leave implicit: a model must carry state across turns, interpret feedback, and choose valid actions under changing constraints. We study this setting in the LM Playschool Challenge with a 2B open-weight model, and find that many failures are not only broad knowledge failures but also local decision failures: repeated guesses, malformed actions, and violations of feedback that the model has just seen. These diagnostics motivate a training recipe organized around three steps: acquire broad game participation through supervised fine-tuning, repair mechanically verifiable failures within one targeted dialogue-game family using turn-local preference pairs, and preserve general capabilities beyond these dialogue games. In the official final evaluation, our submission improves public clemscore from 10.67 to 38.92 and closed in-domain score from 13.41 to 41.17, while app
    
[^13]: 广义样条与高斯过程

    Generalized Splines and Gaussian Processes

    [https://arxiv.org/abs/2608.28446](https://arxiv.org/abs/2608.28446)

    本章将有限维高斯线性逆问题中“最小均方误差估计等价于正则化最小二乘拟合”的经典结论推广到无穷维设定，建立了广义样条与核空间上广义高斯过程之间的对应等价关系。

    

    对于变量服从高斯分布的有限维线性逆问题，众所周知，最小均方误差估计器表现为正则化最小二乘数据拟合的形式。在本章中，我们证明这一等价性可以推广到一个更为广泛的无穷维设定：其中广义样条充当线性回归器的角色，而核空间 $S$ 上的广义高斯过程则对应于高斯随机向量。这一扩展的范畴在性质上类似于从经典函数概念到分布（也称为“广义函数”）的转变。我们的形式化体系涉及一个白化/正则化算子 $L: S\to S'$，其连续延拓诱导出一个本征希尔伯特空间 $H\subset S'$，该空间在我们的刻画中起着核心作用。本阐述在大部分内容上是自包含的，并且具有极高的普适性与威力。它能够恢复……（原文摘要在此处截断）

    arXiv:2608.28446v1 Announce Type: cross  Abstract: For finite-dimensional linear inverse problems where the variables are Gaussian, it is well-known that the minimum-mean-square error estimator takes the form of a regularized least-squares data fit. In this chapter, we show that this equivalence extends to a much broader infinite-dimensional setting where generalized splines take the role of linear regressors and generalized Gaussian processes on a nuclear space $S$ are the counterpart of Gaussian random vectors. The scope of this extension is of the same nature as the switch from the classic notion of function to that of a distribution, also known as a "generalized function." Our formalism involves a whitening/regularization operator $L: S\to S'$ whose continuous extension induces a native Hilbert space $H\subset S'$ that plays a central role in our characterization. The presentation is self-contained for the most part and remarkably general and powerful. It allows for the recovery of
    
[^14]: 滑动窗口注意力优于线性注意力

    Sliding-window beats linear attention

    [https://arxiv.org/abs/2608.28444](https://arxiv.org/abs/2608.28444)

    本研究表明，带sink的滑动窗口注意力（SWA）在多项下游任务和长上下文推理任务上的表现与经过后训练的线性注意力模型相当甚至更优，说明这个更简单的基线方法被严重低估了。

    

    由于二次方复杂度注意力的固有特性，大语言模型（LLM）消耗大量内存和能源。每个新token的成本都比前一个更高。对于每个新增的token，其键和值必须无限期地存储在内存中，这是不可持续的。为了解决二次方扩展问题，研究者们已提出多种替代方案，其中之一是将LLM改造为使用线性注意力。由于线性注意力有望以低成本实现最先进的性能并解决二次方扩展问题，这一想法引起了广泛关注。然而，这一研究方向尚未与更简单的基线方法进行恰当的比较。在本工作中，我们证明了带sink的滑动窗口注意力（SWA）的表现与经过后训练的线性注意力模型相当甚至更好。我们在多个LLM和多种下游任务上都观察到了这一结果。对于长上下文推理任务（Needle-in-a-Haystack和BABILong），SWA实现了大幅更高的性能。

    arXiv:2608.28444v1 Announce Type: new  Abstract: Due to the nature of quadratic attention, Large Language Models (LLMs) consume a lot of memory and energy. Every new token costs more than the previous one. For each additional token, the keys and values must be stored in memory indefinitely, which is unsustainable.   Several alternatives have been proposed to fix the quadratic scaling problem, one of which is retrofitting LLMs to use Linear Attention. This idea has attracted a lot of attention, given its promise to solve the quadratic scaling problem with state-of-the-art performance at low cost. However, this line of research has not been properly compared to simpler baselines.   In this work, we show that Sliding Window Attention (SWA) with sinks performs as well or better than post-trained Linear Attention models. We observe this across multiple LLMs on various downstream tasks. For long-context reasoning tasks (Needle-in-a-Haystack and BABILong), SWA achieves massively higher perfor
    
[^15]: 面向大语言模型预训练的曲率条件多尺度动量与球面约束方法

    Curvature-Conditioned Multiscale Momentum with Sphere Constraints for LLM Pretraining

    [https://arxiv.org/abs/2608.28442](https://arxiv.org/abs/2608.28442)

    该论文提出了一种带球面约束的曲率条件多尺度动量优化方法，通过在平坦方向上结合慢衰减降噪分量与快衰减曲率适应分量的互补优势，加速大语言模型预训练中主导最终损失下降的平坦方向进展。

    

    预训练占据了大语言模型训练总计算成本的很大一部分。然而，噪声主导的梯度和高度病态的损失地形带来了严峻挑战。尽管 AdamW 和 Muon 等现代自适应优化器在大规模预训练中取得了巨大成功，但它们对梯度归一化的依赖只能有限地缓解病态曲率问题。沿平坦方向（即小特征值对应的特征方向）的进展主导着最终损失的降低，但这一进展仍然相对缓慢。为了增强沿平坦方向的训练动态，我们提出了一种带球面约束的曲率条件多尺度动量方法，为大语言模型预训练提供稳定的加速。这种多尺度动量仅应用于平坦方向，将用于噪声抑制的慢衰减分量与用于快速曲率适应的快衰减分量相结合，利用它们互补的优势（摘要在此处被截断）。

    arXiv:2608.28442v1 Announce Type: new  Abstract: Pretraining accounts for a large fraction of the total computational cost in LLM training. However, noise-dominant gradients and the highly ill-conditioned loss landscape bring severe challenges. Although modern adaptive optimizers such as AdamW and Muon have achieved great success in large-scale pretraining, their reliance on gradient normalization offers limited mitigation of the ill-conditioned curvature. The progress along flat directions (eigen-directions of small eigenvalues), which dominates the final loss reduction, remains relatively slow. To enhance training dynamics along flat directions, we propose a curvature-conditioned multiscale momentum method with sphere constraints, delivering steady acceleration in LLM pretraining. This multiscale momentum, applied only along flat directions, pairs a slow-decay component for noise reduction with a fast-decay component for rapid curvature adaptation, harnessing their complementary stre
    
[^16]: 欧几里得傅里叶神经算子

    Euclidean Fourier Neural Operators

    [https://arxiv.org/abs/2608.28425](https://arxiv.org/abs/2608.28425)

    提出欧几里得傅里叶神经算子（EFNO），通过将谱核参数化为物理波矢的连续函数，使神经算子能够跨不同形状和大小的周期域保持一致地学习与迁移，克服了传统FNO的域依赖性问题。

    

    傅里叶神经算子（FNO）为学习函数空间之间的映射提供了一个高效的框架，因为其构建方式使其独立于训练和评估时所用的网格分辨率。然而，FNO 并不独立于其所应用的周期域：其离散谱权重由整数傅里叶模数索引，而这些模数对应于物理波矢。当应用于不同的域时，相同的训练权重会在不同的波矢处起作用，FNO 会在不被察觉的情况下表示一个不同的算子。这使得 FNO 不适用于跨域迁移至关重要的任务。我们提出欧几里得傅里叶神经算子（EFNO）作为 FNO 的域独立替代方案。通过将谱核参数化为物理波矢的连续函数，EFNO 能够学习在不同形状和大小的周期域上保持一致作用的算子。我们在（摘要在此处截断）

    arXiv:2608.28425v1 Announce Type: new  Abstract: Fourier neural operators (FNOs) provide an efficient framework for learning mappings between function spaces as they are, by construction, independent of the grid resolution at which they are trained and evaluated. However, FNOs are not independent of the periodic domain they are applied to: their discrete spectral weights are indexed by integer Fourier mode numbers, which correspond to physical wavevectors. When applied to a different domain, the same trained weights act at different wavevectors, and the FNO silently represents a different operator. This makes FNOs unsuitable for tasks where transfer across domains is crucial. We propose Euclidean Fourier neural operators~(EFNOs) as a domain-independent alternative to FNOs. By parameterizing the spectral kernel as a continuous function of the physical wavevector, the EFNO can learn operators that act consistently across periodic domains of varying shape and size. We evaluate the EFNO on
    
[^17]: SymboLLM-FE：基于大语言模型加速的符号回归，用于表格数据的自动化特征工程

    SymboLLM-FE: LLM-Accelerated Symbolic Regression for Automated Feature Engineering on Tabular Data

    [https://arxiv.org/abs/2608.28408](https://arxiv.org/abs/2608.28408)

    提出SymboLLM-FE方法，将符号回归与大语言模型相结合，通过符号回归提取与目标强相关、可解释的数学公式来指导特征生成，从而解决了传统AutoFE可解释性差、以及基于LLM的方法迭代成本高且易产生偏见与幻觉的问题。

    

    表格数据作为机器学习中的核心数据格式，常因特征信息量不足而缺乏高性能建模所需的判别能力。自动化特征工程（AutoFE）通过自动化特征生成与选择来克服这一问题，同时兼顾模型性能与运行效率。然而，传统AutoFE由于依赖盲目的数学变换，往往生成可解释性较差的特征；而基于大语言模型（LLM）的AutoFE则需要代价高昂的多轮迭代才能生成高效用特征以有效提升模型性能，并且还面临偏见与幻觉等固有风险。本文将符号回归与大语言模型相结合用于特征工程（SymboLLM-FE），以解决上述挑战。我们通过符号回归提取与目标强相关的、数学上表达能力强的公式，这些公式可以增强……

    arXiv:2608.28408v1 Announce Type: new  Abstract: Tabular data, as a core data format in machine learning, often lacks the discriminative power needed for high-performance modeling due to insufficient feature informativeness. Automated Feature Engineering (AutoFE) overcomes this by automating feature generation and selection, ensuring both model performance and operational efficiency. However, traditional AutoFE often yield features with poor interpretability because they rely on blind mathematical transformations, while large language models (LLM)-based AutoFE faces challenges in requiring costly multi-round iterations to generate high-utility features to effectively enhance model performance, compounded by inherent risks of bias and hallucination. In this paper, we combine symbolic regression with LLMs for feature engineering (SymboLLM-FE) to solve these challenges. We extract mathematically expressive formulas strongly correlated with the target via symbolic regression, which can enh
    
[^18]: 面向视频错误检测的视觉语言模型后训练

    Post-Training VLMs for Video Mistake Detection

    [https://arxiv.org/abs/2608.28406](https://arxiv.org/abs/2608.28406)

    该论文提出了错误检测视频问答（MD-VQA）协议与基准，并首次通过后训练视觉语言模型来学习错误的通用概念，从而实现对已见和未见动作的视频错误检测。

    

    人类在遵循指令时难免会犯错，而这些错误可能导致严重后果。因此，开发用于检测视频中错误的方法引起了越来越多的关注，但现有方法大多专注于闭集协议。尽管这类方法在受控环境中取得了成功，但闭集假设限制了其更广泛的应用，因为任务的任何改变都需要收集新数据并重新训练模型。相反，我们认为错误检测方法应该学习错误的通用概念，而不是过拟合于特定步骤的细节。为体现这一点，我们提出了错误检测视频问答（MD-VQA）协议及配套基准。MD-VQA 测试方法能否判断某个步骤是否按照其描述正确执行，涵盖已见和未见动作。为应对这一重要挑战，我们提出了首个视觉语言模型后训练方法……

    arXiv:2608.28406v1 Announce Type: cross  Abstract: Human mistakes are inevitable when following instructions, yet they can lead to severe consequences. As such, there has been an increased interest in developing methods for detecting mistakes in videos, with current methods mostly focusing on closed-set protocols. While successful in controlled settings, the closed-set assumption limits their wider applicability, as any changes to the task require collecting new data and re-training models. Instead, we argue that mistake detection methods should learn the general concept of a mistake, rather than overfitting to step-specific details. To reflect this, we introduce the Mistake Detection Video Question Answering (MD-VQA) protocol and accompanying benchmark. MD-VQA tests whether methods can discern if a step was executed correctly with respect to its description, for both seen and unseen actions. To address this important challenge, we propose the first video-language-model post-training t
    
[^19]: 面向网络规模电商的时间感知复购预测：用于多场景生鲜推荐的生存模型

    Timing-Aware Repurchase Prediction for Web-Scale E-Commerce: Survival Models for Multi-Surface Grocery Recommendation

    [https://arxiv.org/abs/2608.28393](https://arxiv.org/abs/2608.28393)

    该研究用生存模型直接预测复购时间，取代按固定时间窗口逐一训练的二元分类模型，并发现生鲜复购的边际风险随时间轻微递减（对数正态分布拟合最佳），与“越久越可能复购”的传统直觉相反。

    

    电商中的复购推荐通常被表述为一个二元问题——“该顾客是否会在W天内购买该商品”，这种表述需要为每个感兴趣的时间跨度单独训练一个模型。我们用生存模型取代这种堆叠式方案，直接预测复购时间，并在某大型生鲜电商平台的数百万顾客数据上，通过三十多种消融配置进行了评估。我们的研究有三项贡献。首先，经验风险分析揭示了轻微递减的边际风险（k ≈ 0.9），这与“生鲜商品距上次购买时间越长越可能被复购”（递增风险，k > 1）的普遍直觉不同。尽管Weibull分布在条件残差拟合上表现最佳，但对数正态分布却取得了最佳的边际拟合（R^2 = 0.998）和最佳排序效果，我们对这一明显的差异进行了详细分析。其次，单一加速……（原文摘要在此处截断）

    arXiv:2608.28393v1 Announce Type: cross  Abstract: Repurchase recommenders in e-commerce are commonly framed as a binary question asking "will this customer buy this item within W days", a formulation that requires a separately trained model for every horizon of interest. We replace this stack with survival models that predict time-to-repurchase directly, and evaluate them on millions of customers from a major grocery e-commerce platform across more than thirty ablation configurations. Our study makes three contributions. First, an empirical hazard analysis reveals a slightly decreasing marginal hazard (k ~ 0.9), differing from the common intuition that grocery items become more likely to be repurchased the longer since the last purchase (increasing hazard, k > 1). Log-Normal achieves the best marginal fit (R^2 = 0.998) and the best ranking, despite Weibull providing the best conditional residual fit, revealing an apparent discrepancy we analyze in detail. Second, a single Accelerated 
    
[^20]: 基于Bures-Uhlmann几何的面向异构噪声客户端的量子联邦学习

    Quantum Federated Learning Based on Bures--Uhlmann Geometry for Heterogeneous Noisy Clients

    [https://arxiv.org/abs/2608.28379](https://arxiv.org/abs/2608.28379)

    该论文提出将参数空间几何从纯态扩展到噪声客户端实际制备的混合态，利用混合态几何张量的实部（Bures度量）作为局部预条件子、虚部（平均Uhlmann曲率）构建动态聚合规则，从而有效应对量子联邦学习中噪声量子设备的数据与硬件异构性问题。

    

    量子联邦学习使量子设备之间能够在不共享原始数据的情况下协同训练模型，但它面临着噪声量子设备固有的数据与硬件异构性问题。利用量子几何张量是一种自然的解决方案，然而纯态方法和对角近似会丢弃编码参数不可兼容性的关联信息。为解决这一问题，我们将参数空间几何扩展到噪声客户端实际制备的混合态。所得到的混合态几何张量的实部是Bures度量，它衡量物理状态在参数变化下的变化速度；虚部是平均Uhlmann曲率，它量化了同时估计多个参数的不可兼容性。据此，我们采用Bures度量作为局部预条件子，并利用平均Uhlmann曲率开发了一种动态的可达精度聚合规则……

    arXiv:2608.28379v1 Announce Type: cross  Abstract: Quantum federated learning enables collaborative model training across quantum devices without sharing raw data, and it faces the data and hardware heterogeneity inherent to noisy quantum devices. Utilizing the quantum geometric tensor is a natural remedy, yet pure-state approaches and diagonal approximations discard the correlations that encode parameter incompatibility. To address this, we extend the parameter-space geometry to the mixed states that noisy clients actually prepare. The real part of the resulting mixed-state geometric tensor is the Bures metric, which measures how fast the physical state changes under parameter variation, and the imaginary part is the mean Uhlmann curvature, which quantifies the incompatibility of estimating multiple parameters simultaneously. Accordingly, we employ the Bures metric as a local preconditioner and use the mean Uhlmann curvature to develop an achievable-precision aggregation rule that dyn
    
[^21]: 定位全局差异：边际贡献与情境异常检测

    Localizing Global Discrepancies: Marginal Contributions and Contextual Anomaly Detection

    [https://arxiv.org/abs/2608.28375](https://arxiv.org/abs/2608.28375)

    该论文提出了一个将全局分布差异定位到具体观测值的框架，通过为每个观测分配其在随机统计情境中的边际贡献，统一了重采样诊断、数据估值与事件级异常检测，并由此获得更高效的估计量。

    

    全局拟合优度与差异统计量能够判定一个样本偏离了参考分布，但无法识别是哪些观测值导致了这种偏离。我们为这一定位问题开发了一个框架，通过为每个观测值分配其在随机统计情境中的条件贡献或边际贡献。这一方法将重采样诊断与数据估值同投影理论以及事件级异常检测联系起来。对于对称统计量，固定大小的替换与中心化条件定位完全等价。对于U-统计量，添加得分恰好等于第一阶Hoeffding/Hájek贡献；对于光滑分布泛函，其在一阶近似下与影响函数相关；而对于已知背景的无偏MMD，它则恰好简化为MMD见证函数。这一视角还带来了更高效的估计量。匹配情境减除法能够消除与观测对象无关的波动……

    arXiv:2608.28375v1 Announce Type: cross  Abstract: Global goodness-of-fit and discrepancy statistics can establish that a sample departs from a reference distribution without identifying which observations drive the departure. We develop a framework for this localization problem by assigning to each observation its conditional or marginal contribution across random statistical contexts. This connects resampling diagnostics and data valuation to projection theory and event-level anomaly detection. For symmetric statistics, fixed-size replacement is exactly equivalent to centered conditional localization. For U-statistics, the addition score equals the first Hoeffding/H\'ajek contribution; for smooth distributional functionals it is related at leading order to the influence function; and for unbiased known-background MMD it reduces exactly to the MMD witness.   This viewpoint also yields more efficient estimators. Matched-context subtraction removes fluctuations unrelated to the observat
    
[^22]: 基于浅层循环解码器的磁流体液态金属流动实时监测

    Real-Time Monitoring of MHD Liquid Metal Flows with Shallow Recurrent Decoders

    [https://arxiv.org/abs/2608.28366](https://arxiv.org/abs/2608.28366)

    该研究提出将浅层循环解码器（SHRED）与主成分分析相结合的数据驱动降阶模型框架，通过稀疏温度测量实现对DEMO聚变堆液态金属增殖包层中三维磁流体流动的实时状态监测。

    

    磁流体动力学流动中的状态估计对于托卡马克聚变反应堆中液态金属包层的实时监测至关重要。由于这些现象的多物理场特性，高保真度模拟对于实时应用而言在计算上过于昂贵。本工作研究了一种数据驱动的降阶模型框架：将浅层循环解码器（SHRED）与主成分分析相结合，将稀疏的温度测量映射到完整的热工水力系统状态。本工作的主要贡献在于对代表DEMO增殖包层构型的全三维域进行了双参数分析。在该构型中，流动受到方向和强度均变化的外部磁场作用，并受到两个作为水冷系统的圆柱体的阻碍，这两个圆柱体在其表面上施加了温度边界条件。这种双参数磁场变化诱导出（摘要在此处截断）

    arXiv:2608.28366v1 Announce Type: cross  Abstract: State estimation in magnetohydrodynamic flows is critical for real-time monitoring of liquid metal blankets in tokamak fusion reactors. Due to the multiphysics nature of these phenomena, high-fidelity simulations are computationally prohibitive for real-time applications. This work investigates a data- driven Reduced Order Model framework: the Shallow Recurrent Decoder (SHRED) coupled with Principal Component Analysis, to map sparse temperature measurements to the full thermo-hydraulic system's state. The major contribution of this work lies in the two-parameter analysis of a fully three-dimensional domain representative of the DEMO breeding blanket configuration. Here, the flow is subjected to an external magnetic field varying in direction and intensity and is hindered by two cylinders acting as a water-cooling system, which impose a temperature boundary condition on their surfaces. This double-parametric magnetic variation induces n
    
[^23]: GRACE：面向大语言模型遗忘的梯度引导核心集选择方法

    GRACE:Gradient-guided Coreset Selection for LLM Unlearning

    [https://arxiv.org/abs/2608.28361](https://arxiv.org/abs/2608.28361)

    提出GRACE，一种梯度引导的核心集选择方法，能够仅凭少量不良行为种子示例自动构建大语言模型遗忘所需的遗忘集和保留集，并有效保持模型效用。

    

    大语言模型的机器遗忘方法通常假设预先指定了遗忘集和保留集。然而在现实场景中，用户请求可能仅提供少量体现不良行为的示例，这就需要从异构语料库中推断出遗忘集和保留集。我们研究了这一数据选择问题，并提出了GRACE——一种面向大语言模型遗忘的梯度引导核心集选择方法，能够同时构建遗忘集和保留集。GRACE首先从引发不良行为的种子示例中计算出遗忘方向，然后利用非负正交匹配追踪算法选择一个梯度能够近似该遗忘方向的紧凑遗忘核心集。为了保持模型效用，该方法在投影去除遗忘方向后，在剩余梯度空间中应用聚类正交匹配追踪来选择保留样本。在两个目标领域、两个模型家族和四种遗忘算法上的实验表明，GRACE提升了模型遗忘效果（原文此处截断）。

    arXiv:2608.28361v1 Announce Type: cross  Abstract: Machine Unlearning methods for Large Language Models typically assume pre-specified forget and retain sets. In realistic settings, however, requests may provide only a few examples of undesired behavior, requiring forget and retain sets to be inferred from heterogeneous corpora. We study this data-selection problem and propose GRACE , a gradient-guided coreset selection method that constructs both forget and retain sets for LLM unlearning. GRACE first computes a forget direction from seed examples that elicit the undesired behavior, then selects a compact forget coreset whose gradients approximate this direction using non-negative orthogonal matching pursuit. To preserve model utility, it selects retain examples after projecting out the forget direction and applying clustered orthogonal matching pursuit in the remaining gradient space. Across two target domains, two model families, and four unlearning algorithms, GRACE improves model u
    
[^24]: BanglaMed-QA：面向孟加拉语医疗健康支持的问答系统

    BanglaMed-QA: A Question Answering System for Healthcare Support in Bangla

    [https://arxiv.org/abs/2608.28329](https://arxiv.org/abs/2608.28329)

    本文提出了BanglaMed-QA——首个专为孟加拉语医疗领域设计的问答系统，通过构建包含506种疾病、4,493个问答对的结构化知识库，结合SVM问题分类、领域专用词典与同义词集以及多种相似度度量与投票机制，为低资源语言的医疗健康信息支持提供了有效解决方案。

    

    医疗问答（QA）系统已成为提供可靠健康信息的重要工具。但由于数据集有限以及缺乏针对孟加拉语等低资源语言定制的系统，这些语言在医疗问答领域仍鲜有探索。为解决这一问题，我们推出了BanglaMed-QA，一个专为孟加拉语医疗领域设计的稳健问答系统。该流程首先构建了一个结构化的医疗知识库，其中包含506种疾病下9个类别的4,493个问答对。为提升语义理解能力，我们提出了领域专用的词根词典和同义词集，并采用词性标注技术进行指代消解。我们采用了监督机器学习模型，其中支持向量机（SVM）被发现是对问题进行分类的最佳模型。我们应用了多种相似度度量方法，包括余弦相似度、Jaccard、BM25和Levenshtein距离，并结合软投票和硬投票方法进行查询匹配。该问答系统的性能表现……（原文摘要在此处截断）

    arXiv:2608.28329v1 Announce Type: new  Abstract: Medical question answering (QA) systems have become crucial tools for providing reliable health information. But they remain very unexplored for low-resource languages like Bangla due to limited datasets and systems tailored to these languages. To address this, we introduce BanglaMed-QA, a robust QA system specifically designed for the Bangla medical domain. The process begins with building a structured medical knowledge base that includes 4,493 QA pairs in 9 categories under 506 diseases. To improve semantic comprehension, domain-specific root word dictionaries and synonym sets are proposed, in addition to part-of-speech tagging for anaphora resolution. We adopt supervised machine learning models in which SVM is found to be the best model to categorize questions. Multiple similarity metrics, including cosine, Jaccard, BM25, and Levenshtein, are applied with soft and hard voting methods for query matching. The performance of the QA syste
    
[^25]: 推导OpenEuroLLM模型的缩放定律：学习率、批量大小与损失

    Deriving Scaling Laws for OpenEuroLLM Models: Learning Rate, Batch Size and Loss

    [https://arxiv.org/abs/2608.28308](https://arxiv.org/abs/2608.28308)

    该论文为OpenEuroLLM模型推导了学习率、批量大小和损失的缩放定律，研究了它们随模型容量和数据规模的边际演变及最优超参数在训练阶段间的迁移性，并验证了显式建模容量与数据交互的缩放形式能有效捕捉欠训练与过训练两种情形。

    

    我们研究了在以英语为主的语料库上预训练稠密大型语言模型时学习率和批量大小的缩放行为。除了对联合最优的学习率和批量大小进行缩放之外，我们还研究了它们随模型容量和数据规模的边际演变，并开发了一个能够捕捉这些关系的模型。由于我们采用预热-稳定-衰减的学习率调度，我们进一步研究了在广泛的超参数设置、模型和数据预算范围内学习率退火所带来的收益，以及最优学习率和批量大小能否在稳定阶段与衰减阶段之间迁移。最后，我们刻画了损失对模型容量和数据集大小的依赖关系，并评估了最近提出的显式建模两者交互作用的缩放形式。我们发现这些方法在捕捉实验中欠训练和过训练两种情形方面尤为有效。

    arXiv:2608.28308v1 Announce Type: new  Abstract: We study the scaling behavior of learning rate and batch size in pretraining dense large language models on English-prevalent corpora. Beyond scaling \textit{jointly optimal} learning rates and batch sizes, we investigate their \textit{marginal} evolution with model capacity and data scale and develop a model that captures these relationships. As we employ a Warmup-Stable-Decay learning rate schedule, we further investigate the gains from learning rate annealing over a broad range of hyperparameters settings, models and data budgets, and whether the optimal learning rate and batch size \textit{transfer} between the stable and decay phases. Finally, we characterize the dependence of loss on model capacity and dataset size, evaluating recently proposed scaling forms that explicitly model their interaction. We find these approaches particularly effective at capturing both undertraining and overtraining regimes across our experiments. This s
    
[^26]: VISTA：基于验证器信息的学生到教师自适应的在策略自蒸馏

    VISTA: Verifier-Informed Student-to-Teacher Adaptation for On-Policy Self-Distillation

    [https://arxiv.org/abs/2608.28306](https://arxiv.org/abs/2608.28306)

    提出VISTA方法，在保留标准在策略自蒸馏学生更新的同时，利用结果验证的rollout使特权教师向学生分布自适应，解决了教师分布与学生有效推理不匹配时单向监督误导学生的问题。

    

    在策略自蒸馏（OPSD）通过训练一个仅见问题的学生模型，在其自身生成的 rollout 上进行学习，并由一个同时能看见参考答案的特权教师模型提供密集的词元级监督，从而提升推理能力。然而，标准 OPSD 将教师分布视为学生 rollout 上的固定目标，且只更新学生模型——尽管特权条件化并不能保证教师总是为仅见问题的推理提供最合适的目标。因此，当教师分布与学生的有效推理不一致时，这种单向监督可能会误导学生。为此，我们提出了基于验证器信息的学生到教师自适应方法（VISTA），该方法在保留标准 OPSD 学生更新的同时，利用经结果验证的 rollout 使教师分布向学生分布自适应。在每个经验证的 rollout 内，VISTA 进一步将这种自适应限制在 top-k 位置上……

    arXiv:2608.28306v1 Announce Type: cross  Abstract: On-policy self-distillation (OPSD) improves reasoning by training a problem-only student on its own rollouts using dense token-level supervision from a privileged teacher that also sees a reference solution. However, standard OPSD treats the teacher distribution as a fixed target along the student's rollout and updates only the student %, although -- even though privileged conditioning does not guarantee that the teacher always provides the most appropriate target for problem-only reasoning. This one-way supervision can therefore misdirect the student when the teacher distribution is misaligned with valid student reasoning. We therefore introduce Verifier-Informed Student-to-Teacher Adaptation (VISTA), which preserves the standard OPSD student update while using outcome-verified rollouts to adapt the teacher toward the student distribution. Within each verified rollout, VISTA further restricts this adaptation to the top-$k$ positions w
    
[^27]: 解析器状态已然知晓：面向结构化生成的结构条件化KV持久化

    Parser States Already Know: Structure-Conditioned KV Persistence for Structured Generation

    [https://arxiv.org/abs/2608.28276](https://arxiv.org/abs/2608.28276)

    提出PASK方法，将受限解码中解析器状态所暴露的结构信号转化为按层组划分的KV缓存持久化决策，通过任务错误敏感度设定最低保护底线、注意力输出失真分配剩余容量，从而在结构化生成中保护模式关键的KV缓存。

    

    结构化生成是大型语言模型（LLM）智能体生成 JSON、SQL 和函数调用的基础，其中一个错误的字段就可能导致下游操作失败。受限解码（constrained decoding）已经跟踪解析器转换以确保形式有效性，而这些转换揭示了生成的 token 如何在活跃语法下参与模式关键决策，例如必填字段、参数和结构边界。然而，现有的 KV 压缩方法在很大程度上忽略了这一与任务相关的结构信号。我们提出 PASK（Parser-Aware Structural KV Persistence，解析器感知的结构化 KV 持久化），它将解析器派生的结构信息转化为按层组划分的 KV 持久化决策。PASK 通过利用任务错误敏感度来设定最低保护底线，并利用注意力输出失真来分配剩余的 KV 容量，从而解决了模型侧 KV 敏感性与任务级结构化风险之间的不匹配问题。一个离线校准阶段将这些信号编译为（摘要在此处截断）

    arXiv:2608.28276v1 Announce Type: new  Abstract: Structured generation underpins large language model (LLM) agents that produce JSON, SQL, and function calls, where a single wrong field can cause the downstream action to fail. Constrained decoding already tracks parser transitions to enforce formal validity, and these transitions expose how generated tokens participate in schema-critical decisions such as required fields, arguments, and structural boundaries under the active grammar. Existing KV compression largely leaves this task-relevant structural signal unused. We introduce PASK (Parser-Aware Structural KV Persistence), which turns parser-derived structure into layer-group-specific KV persistence decisions. PASK addresses the mismatch between model-side KV sensitivity and task-level structured risk by using task-error sensitivity to set minimum protection floors and attention-output distortion to allocate residual KV capacity. An offline calibration stage compiles these signals in
    
[^28]: Colombo 差-幂行列式猜想的代数证明

    An algebraic proof of Colombo's difference-power determinant conjecture

    [https://arxiv.org/abs/2608.28274](https://arxiv.org/abs/2608.28274)

    本文通过将假想的核向量转化为实二元型的方法，证明了超临界奇数指数下差-幂矩阵的非奇异性，从而完整证明了 Colombo 于 1928 年提出的差-幂行列式猜想，并由此确立了秩公式 rank A_d(λ)=min{n,d+1}。

    

    设 n≥2 为偶数，λ=(λ₁,…,λₙ)∈ℝⁿ 的坐标两两互异，定义差-幂矩阵 A_d(λ) := [(λ_r−λ_s)^d]_{r,s=1}^n，其中 d∈ℕ。1928 年，Colombo 证明了 det A_{n−1}(λ)≠0（因而 det A_{n−1}(λ)>0），并且当 0≤d<n−1 时 rank A_d(λ)=d+1。他猜想：对于每个 d≥n−1，均有 det A_d(λ)≠0。对于偶数 d，该猜想的非奇异性可由已发表的关于距离-幂矩阵的结果推出。因此，剩余的未解决情形是超临界的奇数指数 d≥n+1。我们证明了所有这些奇数指数下的非奇异性，从而完整证明了 Colombo 的猜想。由此可得 rank A_d(λ)=min{n,d+1}（d∈ℕ）。我们的证明将假想的核向量转化为一个实二元型……

    arXiv:2608.28274v1 Announce Type: new  Abstract: Let $n\ge2$ be even, let $\lambda=(\lambda_1,\ldots,\lambda_n)\in\mathbb{R}^n$ have pairwise distinct coordinates, and define the difference-power matrix \[ A_d(\lambda) := \bigl[(\lambda_r-\lambda_s)^d\bigr]_{r,s=1}^n, \qquad d\in\mathbb{N}. \] In 1928, Colombo proved that $\det A_{n-1}(\lambda)\ne0$---and hence $\det A_{n-1}(\lambda)>0$---and that $\operatorname{rank} A_d(\lambda)=d+1$ for $0\le d<n-1$. He conjectured that \[ \det A_d(\lambda)\ne0 \qquad\text{for every } d\ge n-1. \] For even $d$, the conjectured nonsingularity follows from previously published results on distance-power matrices. The remaining open cases were therefore the supercritical odd exponents $d\ge n+1$. We prove nonsingularity for all these odd exponents, thereby completing Colombo's conjecture. Consequently, \[ \operatorname{rank} A_d(\lambda)=\min\{n,d+1\} \qquad(d\in\mathbb{N}). \] Our proof converts a hypothetical kernel vector into a real binary form havi
    
[^29]: 跨模式迁移学习：迈向统一的城市出行需求预测

    Learning to Transfer Across Modes: Towards Unified Urban Mobility Forecasting

    [https://arxiv.org/abs/2608.28273](https://arxiv.org/abs/2608.28273)

    提出TransMod统一框架，通过构建共享的区级空间表示对齐不同空间粒度的出行系统，实现异构出行模式间的知识迁移，从而解决多模式城市出行需求预测中的空间异质性与新兴模式数据稀缺问题。

    

    城市交通系统由多种出行模式组成，这些模式在同一城市内共存，并表现出复杂的相互依赖关系，导致各模式间的需求动态相互关联。然而，由于空间上存在显著异质性，以及新兴出行模式的历史数据有限，跨不同模式联合预测需求仍然极具挑战性。现有的预测方法大多针对单一出行模式开发，并隐含地假设源系统与目标系统之间具有兼容的空间结构，这严重限制了它们在多模式场景中的适用性。为应对这些挑战，我们提出了TransMod，一个统一的城市出行需求预测框架，能够在异构出行模式之间实现有效的知识迁移。TransMod构建了一个共享的区级空间表示，将具有不同空间粒度的出行系统对齐到统一的表示空间中。

    arXiv:2608.28273v1 Announce Type: new  Abstract: Urban transportation systems consist of multiple mobility modes that coexist within the same city and exhibit complex interdependencies, leading to correlated demand dynamics across modes. However, forecasting demand jointly across different modes remains challenging due to substantial heterogeneity in space and the limited availability of historical data for emerging modes. Existing forecasting methods are largely developed for individual mobility modes and implicitly assume compatible spatial structures between source and target systems, which severely restricts their applicability in multi-modal settings. To address these challenges, we propose \textbf{TransMod}, a unified framework for urban mobility demand forecasting that enables effective knowledge transfer across heterogeneous mobility modes. TransMod constructs a shared zone-level spatial representation that aligns mobility systems with different spatial granularities into a com
    
[^30]: 残差引导的随机化神经网络

    Residual-Guided Randomized Neural Networks

    [https://arxiv.org/abs/2608.28267](https://arxiv.org/abs/2608.28267)

    提出一种残差引导的贪心构建方法，通过闭式残差下降准则逐步筛选随机候选隐藏单元并重新拟合输出层，克服了随机化神经网络一次性随机特征构建所导致的冗余表示和模型容量利用不足的问题。

    

    随机化神经网络通过随机固定输入到隐藏层的参数，并以闭式解的方式学习输出权重，从而实现快速且可解析分析的训练；然而，其性能严重依赖于对隐藏单元的一次无信息随机抽取。这种一次性且与任务无关的特征构建往往导致冗余的表示以及模型容量的次优利用。为了解决这一局限性，我们提出了一种简单且广泛适用的残差引导方法，该方法使用闭式残差下降准则贪心地构建隐藏层。在每一阶段，我们（i）生成一个随机候选单元池，（ii）根据每个候选单元在岭回归正则化目标中引起的精确下降量对其进行评分，（iii）选出得分最高的k个单元，以及（iv）使用带直接输入连接的标准设计以闭式形式重新拟合读出层。该过程产生了一个渐进式的训练过程，并具有保证……

    arXiv:2608.28267v1 Announce Type: new  Abstract: Randomized neural networks enable fast and analytically tractable training by fixing the input to hidden layer parameters at random and learning the output weights in closed form; however, their performance critically depends on a single uninformed draw of hidden units. This one shot and task uninformed feature construction often leads to redundant representations and suboptimal utilization of model capacity. To address this limitation, we propose a simple and broadly applicable residual guided procedure that greedily constructs the hidden layer using a closed form residual decrease criterion. At each stage, we (i) generate a pool of random candidate units, (ii) score each candidate by the exact reduction it induces in the ridge regularized objective, (iii) select the top k units, and (iv) refit the readout in closed form using the standard design with direct input links. This procedure yields a progressive training process with a guaran
    
[^31]: SinkSLOT：基于稀疏提升最优传输的Sinkhorn算法

    SinkSLOT: Sinkhorn via Sparse Lifted Optimal Transport

    [https://arxiv.org/abs/2608.28262](https://arxiv.org/abs/2608.28262)

    SinkSLOT通过期望切片提升传输计划对Gibbs核进行稀疏化并采用非独立先验耦合，将每次Sinkhorn迭代的计算复杂度从O(N²)降至O(LN)，同时所得目标函数无需去偏处理。

    

    熵正则化最优传输（EOT）已被证明能为精确最优传输提供计算上可行的近似。然而，标准的Sinkhorn-Knopp算法存在两个主要局限。第一，对于包含 $N$ 个点的离散测度，每次迭代需要 $O(N^2)$ 次运算，这限制了其在大规模数据集（例如 $N\geq10^4$）上的应用。第二，该算法使用独立耦合作为正则化的参考测度，这会在中等正则化强度下将质量分配给高成本的传输边。我们提出SinkSLOT，通过引入期望切片提升传输计划作为以非独立先验耦合稀疏化Gibbs核的自然方法，来解决这两个局限。我们证明：1）SinkSLOT是收敛的；2）使用 $L$ 个切片时，每次稀疏Sinkhorn迭代的成本为 $O(LN)$；3）所得目标函数是一个无需去偏（debiasing）的散度。在合成基准上的实验表……

    arXiv:2608.28262v1 Announce Type: new  Abstract: Entropic optimal transport (EOT) has been shown to offer a computationally tractable approximation to exact optimal transport. However, the standard Sinkhorn-Knopp algorithm has two main limitations. First, given discrete measures with $N$ points, each iteration requires $O(N^2)$ operations, which restricts its use on large-scale datasets (e.g. $N\geq10^4$). Second, it uses the independent coupling as a reference measure for regularisation. This assigns mass to high-cost transport edges at moderate regularisation strengths. We propose SinkSLOT, which addresses both limitations by putting forth the expected sliced lifted transport plan as a natural way to sparsify the Gibbs kernel with a non-independent prior coupling. We prove that: 1) SinkSLOT converges; 2) with $L$ slices, each resulting sparse Sinkhorn iteration costs $O(LN)$; and 3) the resulting objective is a divergence requiring no debiasing. Experiments on synthetic benchmarks sh
    
[^32]: I-FLOP：基于干预数据的序与父节点快速学习

    I-FLOP: Fast Learning of Order and Parents from Interventional Data

    [https://arxiv.org/abs/2608.28245](https://arxiv.org/abs/2608.28245)

    I-FLOP将FLOP算法从观测数据扩展到干预数据场景，通过将干预BIC评分适配到基于Cholesky的迭代评分更新机制中，实现了兼具速度优势与理论保证（可恢复正确干预马尔可夫等价类）的快速因果结构学习。

    

    我们将Wienöbst等人（2026）近期提出的FLOP（序与父节点快速学习）算法从观测数据扩展到干预数据。特别地，我们采用Hauser和Bühlmann（2012）提出的干预BIC评分，并将其适配到基于Cholesky分解的迭代评分更新框架中，而后者正是FLOP算法速度优势的部分来源。我们证明，在样本极限情况下，I-FLOP能够恢复出与数据生成DAG处于同一干预马尔可夫等价类中的DAG。我们在真实和模拟的干预数据上将I-FLOP与现有的因果结构学习算法进行比较，结果表明I-FLOP在性能和运行时间两方面均表现优异。

    arXiv:2608.28245v1 Announce Type: cross  Abstract: We extend the FLOP (fast learning of order and parents) algorithm recently proposed by Wien\"obst et al. (2026) from observational to interventional data. In particular, we use the interventional BIC score of Hauser and B\"uhlmann (2012), adapting it to be used with the iterative Cholesky-based score updates that are partly responsible for FLOP's speed. We show that, in the sample limit, I-FLOP recovers a DAG in the same interventional Markov equivalence class as the data-generating DAG. We compare I-FLOP to existing causal structure learning algorithms on real and simulated interventional data, where it performs favorably in terms of both performance and run time.
    
[^33]: 频谱特征主导BCG呼吸事件检测：睡眠呼吸暂停患者中特征组的大规模患者无关比较

    Spectral Features Dominate BCG Respiratory-Event Detection: A Large-Scale Patient-Independent Comparison of Feature Groups in Sleep Apnea Patients

    [https://arxiv.org/abs/2608.28242](https://arxiv.org/abs/2608.28242)

    该大规模患者无关研究表明，频谱特征在心冲击图（BCG）睡眠呼吸暂停呼吸事件检测中占主导地位，随机森林和直方图梯度提升模型达到了0.967的AUC-ROC。

    

    无感式心冲击图（BCG）传感是长期睡眠呼吸暂停监测的一种有前景的模态，然而目前尚不清楚哪些信号特征对呼吸事件检测最具判别力。我们提出了一项基于文献引导的、患者无关的十组BCG特征组比较研究，使用512传感器电容式压力垫与呼吸多导睡眠图同步记录，研究对象为155名（52名女性，103名男性）正在接受阻塞性睡眠呼吸暂停住院评估的患者。特征从六个空间上不同的信号通道中提取，生成了一个涵盖通用统计、时域、频域、小波、帧能量和非线性复杂度描述符的191维特征向量。在严格的留一患者交叉验证下，对呼吸事件窗口与无事件参考窗口进行二分类，随机森林和直方图梯度提升模型分别达到了0.967和0……（摘要在此处截断）

    arXiv:2608.28242v1 Announce Type: new  Abstract: Unobtrusive ballistocardiographic (BCG) sensing is a promising modality for long-term sleep-apnea monitoring, yet it remains unclear which signal features are most discriminative for respiratory-event detection. We present a literature-guided, patient-independent comparison of ten BCG feature groups using a 512-sensor capacitive pressure mat recorded simultaneously with respiratory polygraphy in 155 patients (52 female, 103 male) undergoing in-hospital evaluation for obstructive sleep apnea. Features were extracted from six spatially distinct signal channels, yielding a 191-dimensional feature vector spanning general statistical, time-domain, frequency-domain, wavelet, frame-energy, and nonlinear complexity descriptors. Under strict leave-one-patient-out cross-validation for binary classification of respiratory-event windows versus event-free reference windows, Random Forest and Histogram Gradient Boosting achieved AUC-ROC of 0.967 and 0
    
[^34]: 面向预测性流程监控的高效在线持续基础模型微调

    Efficient Online Continual Foundation Model Fine-Tuning for Predictive Process Monitoring

    [https://arxiv.org/abs/2608.28237](https://arxiv.org/abs/2608.28237)

    提出了首个面向预测性流程监控的基础模型在线持续微调框架COMPASS，通过自适应损失平台期漂移检测自主识别任务边界，并利用统一知识子空间有效缓解概念漂移带来的冷启动问题。

    

    预测性流程监控（PPM）模型越来越多地被部署在动态环境中，概念漂移会导致底层流程分布随时间发生变化。尽管近期研究已转向在线持续学习，但现有方法仍完全从零开始训练紧凑的、面向特定任务的网络，留下了持久的冷启动问题。基础模型（FMs）为这一问题提供了极具吸引力的解决方案，但其在流程挖掘领域的持续微调尚未被探索。我们提出了COMPASS（基于持续在线基础模型的自适应子空间预测性流程监控），这是首个面向PPM的基础模型在线持续微调框架。COMPASS改进了损失平台期漂移检测方法，能够自主识别事件流中的任务边界，并维护一个同时包含预训练方向和特定任务方向的统一知识子空间。我们在涵盖合成与真实数据的九个事件流上评估了该方法。

    arXiv:2608.28237v1 Announce Type: new  Abstract: Predictive Process Monitoring (PPM) models are increasingly deployed in dynamic environments where concept drift causes the underlying process distribution to shift over time. While recent work has moved toward online continual learning, existing methods train compact, task-specific networks entirely from scratch, leaving a persistent cold-start problem. Foundation Models (FMs) offer a compelling solution to this problem, but their continual fine-tuning in the process mining domain remains unexplored. We propose COMPASS (Continual Online foundation Model-based PPM with Adaptive SubSpaces), the first framework for online continual fine-tuning of FMs for PPM. COMPASS adapts loss-plateau drift detection to autonomously identify task boundaries in event streams and maintains a unified knowledge subspace including both pre-trained and task-specific directions. We evaluate our approach on nine event streams covering synthetic and real-world co
    
[^35]: D-TAIA：面向多任务预测性流程监控的领域感知大语言模型适配

    D-TAIA: Domain-Aware LLM Adaptation for Multi-Task Predictive Process Monitoring

    [https://arxiv.org/abs/2608.28236](https://arxiv.org/abs/2608.28236)

    提出D-TAIA框架，通过领域感知训练与基于注意力的推理架构对大语言模型进行参数高效微调，在数据稀缺、高流程熵和分布偏移条件下实现下一个活动与剩余时间的联合预测。

    

    预测性流程监控（PPM）使组织能够预测未来的流程行为，例如进行中案例的下一个活动及剩余时间。在实践中，三种情况会导致现有方法性能下降，即数据稀缺、高流程熵和分布偏移。尽管基础模型（FMs），尤其是大语言模型（LLMs），凭借广泛的序列推理能力提供了一种新范式，但在这些条件下将其适配到多任务PPM仍然是一个悬而未决的挑战。现有基于基础模型的方法要么缺乏处理分布偏移的机制，要么依赖于与连续时间预测任务在结构上可能不匹配的直接回归头。本文提出了D-TAIA（领域感知训练与基于注意力的推理架构），一个通过对基础模型骨干进行参数高效微调来联合预测下一个活动和剩余时间的框架。我们的方法结合了领域……（摘要在此处被截断）

    arXiv:2608.28236v1 Announce Type: new  Abstract: Predictive Process Monitoring (PPM) enables organizations to forecast future process behavior, such as the next activity and remaining time of ongoing cases. In practice, three conditions cause existing methods to degrade, namely data scarcity, high process entropy and distributional shift. While Foundation Models (FMs), especially Large Language Models (LLMs), offer a new paradigm through broad sequential reasoning, adapting them to multi-task PPM under these conditions remains an open challenge. Existing FM-based approaches either lack mechanisms for handling distributional shift or rely on direct regression heads that can be structurally misaligned with continuous time prediction tasks. This paper introduces D-TAIA (Domain-aware Training and Attention-based Inference Architecture), a framework for a joint next activity and remaining time prediction task via parameter-efficient fine-tuning of an FM backbone. Our approach combines domai
    
[^36]: 恪守边界：基于距离引导的解码方法，保证输出符合上下文无关文法

    Stay Within Your Bounds: Distance-Guided Decoding for Guaranteed Context-Free Grammar Compliance

    [https://arxiv.org/abs/2608.28229](https://arxiv.org/abs/2608.28229)

    提出一种基于下推自动机的距离引导解码框架，通过离线计算可达性标签与到接受状态的距离上界、在线进行视野感知剪枝与束搜索，保证大模型生成结果百分之百符合目标上下文无关文法，同时提升补全质量。

    

    文法约束解码可帮助大型语言模型生成语法有效的结构化输出，例如代码、JSON和SQL。针对上下文无关文法，许多实用解码器强制执行局部前缀可行性：每个token必须保证当前前缀能够扩展到某个有效的完整结果。然而，在分词器与文法不匹配以及token预算有限的情况下，可行前缀仍可能无法到达接受状态。我们提出了一种基于下推自动机的、面向上下文无关文法的前瞻引导解码框架。离线阶段，我们计算带有可达性标签以及到接受状态距离上界的有界下推摘要；在线阶段，这些估计值引导具有视野感知的剪枝与束搜索。由此得到的解码器在语法上是可靠的：每个输出都会被目标文法接受。在JSON、SQL和线性时序逻辑（LTL）上的实验表明，与现有基线相比，该方法既保持了一致的语法有效性，又提升了补全质量。

    arXiv:2608.28229v1 Announce Type: cross  Abstract: Grammar-constrained decoding helps large language models produce syntactically valid structured outputs, such as code, JSON, and SQL. For context-free grammars, many practical decoders enforce local prefix feasibility: each token must keep the current prefix extendable to some valid completion. Yet, under tokenizer-grammar mismatch and finite token budgets, feasible prefixes may still fail to reach acceptance. We propose a lookahead-guided decoding framework for context-free grammars based on pushdown automata. Offline, we compute bounded pushdown summaries with reachability labels and upper-bound distances to acceptance. Online, these estimates guide horizon-aware pruning and beam search. The resulting decoder is syntactically sound: every output is accepted by the target grammar. Experiments on JSON, SQL, and Linear Temporal Logic (LTL) show both consistent syntactic validity and improved completion quality over existing baselines.
    
[^37]: 面向不相交表格数据迁移学习的交叉注意力广义上下文方法

    Generalized Context in Cross Attention for Transfer Learning of Disjoint Tabular Data

    [https://arxiv.org/abs/2608.28209](https://arxiv.org/abs/2608.28209)

    本文提出CATTLE方法，利用Transformer中key和query投影权重捕获广义上下文，在源域与目标域表格数据完全不共享特征的情况下，以数据无关的方式实现跨领域注意力迁移学习。

    

    与图像和文本不同，将迁移学习应用于表格数据极具挑战性，原因在于不同领域之间存在特征类型、结构和语义上的异质性。现有方法通常假设数据表之间共享特征才能实现跨领域知识迁移，这在实际场景中并不现实。本文提出广义上下文学习方法，消除了对跨领域共享特征的要求。由Transformer中key、value和query投影权重捕获的广义上下文，能够提供基于规则的泛化能力，而非传统上从Transformer激活中学习到的领域特定上下文。源领域的key投影权重与目标领域的query权重相互作用，以数据无关的方式实现跨领域注意力迁移学习（CATTLE）。在十对不相交的源-目标数据集上的实验表明，CATTLE能够学习泛化能力。

    arXiv:2608.28209v1 Announce Type: new  Abstract: Unlike images and text, applying transfer learning to tabular data is challenging due to heterogeneity in feature types, structures, and semantics across disparate domains. Existing methods assume shared features across data tables to enable knowledge transfer between domains, which is unrealistic in practice. \mds{This paper introduces generalized context learning to remove the requirement of shared features across domains. The generalized context captured by transformer projection weights for $key$, $value$, and $query$ provides rule-based generalization rather than the domain-specific context conventionally learned from transformer activations. Projection weights for $key$ from the source domain interact with the weight for $query$ in the target domain to achieve Cross-domain Attention Transfer Learning (CATTLE) in a data-agnostic manner. Our experiments on ten pairs of disjoint source-target data sets show that CATTLE can learn gener
    
[^38]: 使用视觉基础模型的可解释糖尿病视网膜病变分类

    Explainable Diabetic Retinopathy Classification Using Vision Foundation Models

    [https://arxiv.org/abs/2608.28207](https://arxiv.org/abs/2608.28207)

    该研究提出了一种基于视觉基础模型（DINOv2、CLIP、ViT）结合多种迁移学习策略的可解释糖尿病视网膜病变分类框架，其中DINOv2-LoRA内部AUROC达0.758，外部AUROC达0.920，并通过Grad-CAM和HiResCAM结合专家标注病灶掩码实现了模型可解释性评估。

    

    糖尿病视网膜病变（DR）是可预防性失明的主要原因之一，因此需要准确且可信的自动化筛查。本研究探索了一个使用视觉基础模型和多种迁移学习策略的可解释DR分类框架。研究评估了三种骨干网络——DINOv2、CLIP和视觉Transformer（ViT），并采用了全量微调、线性探测和低秩适应（LoRA）三种策略。模型在ODIR数据集上进行训练和内部评估，并在APTOS数据集上进行外部评估以检验泛化能力。DINOv2-LoRA取得了最高的内部AUROC（0.758），而DINOv2全量微调和ViT全量微调取得了最高的外部AUROC（0.920）。在等渗回归之后，通过可靠性分析进一步评估了模型的校准性能。在可解释性方面，使用IDRiD数据集中专家标注的病灶掩码，通过Dice系数和交并比（IoU）等指标对Grad-CAM和HiResCAM进行了评估。

    arXiv:2608.28207v1 Announce Type: cross  Abstract: Diabetic retinopathy (DR) is a major cause of preventable blindness, creating a need for accurate and trustworthy automated screening. This study investigates an explainable DR classification framework using vision foundation models and multiple transfer learning strategies. Three backbones, DINOv2, CLIP, and Vision Transformer (ViT), were evaluated using full fine-tuning, linear probing, and Low-Rank Adaptation (LoRA). Models were trained and internally evaluated on the ODIR dataset and externally evaluated on APTOS to assess generalization. DINOv2-LoRA achieved the highest internal AUROC of 0.758, while DINOv2 full fine-tuning and ViT full fine-tuning achieved the highest external AUROC of 0.920. Calibration was further assessed using reliability analysis after isotonic regression. For explainability, Grad-CAM and HiResCAM were evaluated against expert-annotated lesion masks from the IDRiD dataset using Dice, Intersection over Union 
    
[^39]: 表演性隐私：差分隐私何时能最大化效用

    Performative Privacy: When Differential Privacy Maximizes Utility

    [https://arxiv.org/abs/2608.28198](https://arxiv.org/abs/2608.28198)

    该论文提出“表演性隐私”新框架，首次形式化了隐私保护与用户参与度之间的动态关系，并证明当数据泄露导致用户流失时，采用有限隐私预算的差分隐私机制在长期内可以优于非隐私估计。

    

    保护隐私的学习通常源于这样一种理念：保护用户数据可以维持信任，从而保持用户参与，进而在长期内提升效用。然而，这一论点迄今为止尚未被形式化。与此同时，表演性学习为研究部署行为会影响其后续观测数据的学习系统提供了一个框架。在本工作中，我们将这两种视角结合起来，提出了“表演性隐私”的概念，即数据泄露会降低未来的用户参与度。我们研究了一个简单模型：智能体反复贡献数据用于均值估计，但当其数据被泄露时可能会退出系统。隐私通过差分隐私机制来实现，从而在估计噪声与未来参与度之间形成权衡。通过对该动态过程的理论研究和数值实验，我们证明了在某些条件下，有限的隐私预算在长期内可以优于非隐私估计。

    arXiv:2608.28198v1 Announce Type: new  Abstract: Privacy-preserving learning is often motivated by the idea that protecting users' data can preserve trust and thus participation, improving utility in the long term. However, this claim has not been formalized so far. In parallel, performative learning provides a framework for studying learning systems whose deployment affects the data they later observe. In this work, we bring these two perspectives together and introduce \emph{performative privacy}, where data leakage reduces future participation. We study a simple model where agents repeatedly contribute data for mean estimation but may leave the system when their data is leaked. Privacy is implemented through differentially private mechanisms, creating a trade-off between estimation noise and future participation. We show, through a theoretical study of the dynamics and numerical experiments, that a finite privacy budget can outperform non-private estimation in the long term when the
    
[^40]: EXPOSE：利用稀疏自编码器从病理学视觉基础模型中获取可解释且域鲁棒的嵌入

    EXPOSE: Explainable and Domain-Robust Embeddings from Pathology Vision Foundation Models using Sparse Autoencoders

    [https://arxiv.org/abs/2608.28191](https://arxiv.org/abs/2608.28191)

    提出了EXPOSE框架，利用稀疏自编码器作为可解释瓶颈，识别并抑制病理视觉基础模型嵌入中的域特定信息，从而在不重新训练骨干模型的情况下提升跨域泛化能力。

    

    视觉基础模型（VFMs）在计算病理学中被广泛使用，但对由染色、组织制备和扫描仪硬件差异所引起的域偏移仍然敏感。一个关键限制在于VFM嵌入将生物学信息与特定域信息纠缠在一起，阻碍了跨域泛化能力。我们提出了跨域稀疏嵌入的可解释探测框架（EXPOSE），该框架使用稀疏自编码器（SAEs）作为可解释的瓶颈，来识别并抑制VFM嵌入中的特定域成分。我们训练VFM特征的稀疏表示，使用线性分类器识别特定域的潜在维度，并在下游复发预测之前掩蔽这些特征，而无需重新训练骨干模型。在包含多个采集域的大型前列腺癌数据集上的实验表明，SAE特征能够同时捕获域特定信息和任务特定信息，且二者部分可分离（原文在此处截断）。

    arXiv:2608.28191v1 Announce Type: cross  Abstract: Vision Foundation Models (VFMs) are widely used in computational pathology but remain sensitive to domain shifts arising from variations in staining, tissue preparation, and scanner hardware. A key limitation is that VFM embeddings entangle biological with domain-specific information, hindering cross-domain generalization. We propose Explainable Probing of Cross-Domain Sparse Embeddings (EXPOSE), a framework that uses Sparse Autoencoders (SAEs) as an explainable bottleneck to identify and suppress domain-specific components in VFM embeddings. We train a sparse representation of VFM features, use a linear classifier to identify domain-specific latent dimensions, and mask these features prior to downstream relapse prediction without retraining the backbone model. Experiments on a large prostate cancer dataset with multiple acquisition domains show that SAE features capture both domain- and task-specific information, which are partially d
    
[^41]: 超越扁平网表：面向时序电路可扩展分析的层次图表示学习

    Beyond Flat Netlist: Hierarchical Graph Representation Learning for Scalable Analysis of Sequential Circuits

    [https://arxiv.org/abs/2608.28188](https://arxiv.org/abs/2608.28188)

    DeepSeq3提出了一种层次化图表示学习框架，通过触发器划分的组合子图与超节点图两级表示、双GNN架构以及基于状态可达性的预训练方案，实现了对大规模时序电路的可扩展分析。

    

    电路表示学习（CRL）为指导和优化核心电子设计自动化（EDA）任务提供了强大的范式，但其实际应用受到工业级网表巨大规模以及未能显式建模寄存器级时序动态的阻碍。为克服这些障碍，我们提出了DeepSeq3，一种新颖的层次化框架，它将电路抽象为两级表示：由触发器（FF）划分的细粒度组合逻辑子图，以及建模寄存器传输结构的高层超节点图（SNG）。双图神经网络（GNN）架构在两个层级上学习表示，同时捕获局部布尔逻辑和全局状态转换。至关重要的是，我们引入了一种以状态为中心的预训练方案，该方案预测触发器状态之间的可达性，从而赋予模型对时序行为的深刻理解。在大规模基准测试上的验证表明，DeepSeq3……

    arXiv:2608.28188v1 Announce Type: new  Abstract: Circuit Representation Learning (CRL) offers a powerful paradigm to guide and optimize core Electronic Design Automation (EDA) tasks, but its practical adoption is hindered by the immense scale of industrial netlists and a failure to explicitly model register-level temporal dynamics. To overcome these barriers, we introduce DeepSeq3, a novel hierarchical framework that abstracts circuits into a two-level representation: fine-grained combinational subgraphs partitioned by flip-flops (FFs), and a high-level Super-Node Graph (SNG) that models the register-transfer structure. A dual Graph Neural Network (GNN) architecture learns representations at both levels, capturing local Boolean logic and global state transitions. Crucially, we introduce a state-centric pre-training scheme that predicts the reachability between FF states, endowing the model with a deep understanding of temporal behavior. Demonstrated on large-scale benchmarks, DeepSeq3'
    
[^42]: 受生物学启发的促进多层感知机中Grokking（顿悟）现象的机制

    Biologically Inspired Mechanisms for Facilitating Grokking in Multilayer Perceptrons

    [https://arxiv.org/abs/2608.28184](https://arxiv.org/abs/2608.28184)

    本文在多层感知机中引入七种受生物学启发的机制（如输入门控、结构可塑性、稳态调节、侧抑制等），通过消融实验发现稳态调节对促进grokking（从记忆到泛化的延迟转变）的作用最强且最一致，结构稀疏化也具有重要作用。

    

    Grokking是一种从记忆到泛化的延迟转变，通常伴随着内部表示的大规模重组。本文研究了受生物学启发的机制——其中许多机制通常未被纳入人工神经网络——能否通过在神经元活动、响应和有效连接等层面调节隐藏层计算，从而主动促进这一转变。我们为多层感知机增加了输入门控、结构可塑性、增益调制、阈值调制、稳态调节、侧抑制和激活去相关等机制，并在两个经典grokking基准任务（稀疏奇偶校验和带噪XOR分类）上通过系统性消融实验评估这些机制。结果表明，各机制对泛化的贡献并不均等：稳态调节提供了最强且最一致的收益，而结构稀疏化则成为（摘要在此处截断）。

    arXiv:2608.28184v1 Announce Type: new  Abstract: Grokking is a delayed transition from memorization to generalization that is often accompanied by substantial reorganization of internal representations. This paper studies whether biologically inspired mechanisms, many of which are not commonly incorporated into artificial neural networks, can actively promote this transition by regulating hidden-layer computation at the levels of neuronal activity, response, and effective connectivity. We augment a multilayer perceptron with input gating, structural plasticity, gain modulation, threshold modulation, homeostasis, lateral inhibition, and activation decorrelation, and evaluate these mechanisms through systematic ablations on two established grokking benchmarks: sparse parity and noisy XOR classification. The results show that the mechanisms contribute unequally to generalization. Homeostasis provides the strongest and most consistent benefit, while structural sparsification emerges as the
    
[^43]: 基于优化确定性等价风险控制的保形风险规避决策

    Conformal Risk-Averse Decision Making with Optimized Certainty Equivalent Risk Control

    [https://arxiv.org/abs/2608.28179](https://arxiv.org/abs/2608.28179)

    该论文提出了基于优化确定性等价（OCE）度量的风险规避决策框架，证明CVaR下的最优策略可归结为基于预测集的解，从而为保形预测提供了操作性解释，并针对未知分布设计了基于合成似然模型与留出校准数据的数据驱动校准策略，实现对OCE风险的高概率控制。

    

    我们研究风险规避决策问题，其中智能体在对真实系统状态不确定的情况下选择动作。风险通过优化确定性等价（OCE）度量来衡量，该度量推广了均值-方差风险和条件风险价值（CVaR）等流行准则。我们在分布已知的情况下刻画了最优策略，并证明对于CVaR，该策略可简化为基于预测集的解，这为保形预测类型的预测集提供了一种操作性的解释。对于分布未知的情况，我们基于似然的合成模型和留出的校准数据，开发了一种数据驱动的校准策略，从而实现对OCE风险的高概率控制。该方法在两个无线波束成形场景中进行了评估。

    arXiv:2608.28179v1 Announce Type: cross  Abstract: We study risk-averse decision making, in which an agent selects actions while being uncertain about the true system state. The risk is measured via optimized certainty equivalent (OCE) metrics, which generalize popular criteria such as mean-variance risk and conditional value-at-risk (CVaR). We characterize the optimal policy under known distributions, and show that it reduces to a prediction set-based solution for the CVaR. This provides an operational interpretation of conformal prediction-type prediction sets. For unknown distributions, we develop a data-driven calibration strategy, based on a synthetic model for the likelihood and held-out calibration data, yielding high-probability control of the OCE risk. The approach is evaluated on two wireless beamforming settings.
    
[^44]: 赋能当地农业：基于深度学习的孟加拉国芒果品种识别网络系统

    Empowering Local Agriculture: A Deep Learning-Powered Web System for Identifying Bangladeshi Mango Varieties

    [https://arxiv.org/abs/2608.28161](https://arxiv.org/abs/2608.28161)

    本研究构建了一个基于深度学习的网络系统，利用自建的包含九个孟加拉国芒果品种的图像数据集微调EfficientNetB0等CNN模型，实现了高达97.36%的测试准确率，可自动准确识别孟加拉国芒果品种。

    

    在孟加拉国进行芒果品种识别具有挑战性，因为密切相关的栽培品种可能具有相似的视觉特征，且图像通常是在不同的真实世界条件下拍摄的。本工作提出了一种基于深度学习的网络系统，用于自动识别孟加拉国芒果品种。我们从当地市场和农场收集了2,013张高质量芒果图像（3024x4032像素），并将其组织为九个类别，将Bari-4和Bari-7合并为单一的Bari类。数据集被划分为训练集（70%）、验证集（15%）和测试集（15%），并应用图像增强来提高模型的泛化能力。三种预训练的CNN架构——ResNet18、ResNet50和EfficientNetB0——在一致的训练设置下进行了微调。EfficientNetB0取得了最佳性能，获得了98.01%的验证准确率和97.36%的测试准确率，相比之下，ResNet18和ResNe…（原文在此处截断）

    arXiv:2608.28161v1 Announce Type: cross  Abstract: Mango variety identification in Bangladesh is challenging because closely related cultivars can have similar visual characteristics and images are often captured under varying real-world conditions. This work presents a deep learning-based web system for automatic identification of Bangladeshi mango varieties. We collected 2,013 high-quality mango images (3024x4032 pixels) from local markets and farms and organized them into nine classes, combining Bari-4 and Bari-7 as a single Bari class. The dataset was divided into training (70%), validation (15%), and test (15%) sets, with image augmentation applied to improve model generalization. Three pretrained CNN architectures, ResNet18, ResNet50, and EfficientNetB0, were fine-tuned under consistent training settings. EfficientNetB0 achieved the best performance, obtaining 98.01% validation accuracy and 97.36% test accuracy, compared with 86.47% and 78.55% test accuracy for ResNet18 and ResNe
    
[^45]: HARTS：面向任意回放树上混合注意力模型的高效智能体强化学习

    HARTS: Efficient Agentic Reinforcement Learning for Hybrid-Attention Models over Arbitrary Rollout Trees

    [https://arxiv.org/abs/2608.28158](https://arxiv.org/abs/2608.28158)

    提出了HARTS系统，通过联合规划微批次、数据并行副本分配与调度，以及线性时间算法协调分块边界状态恢复与重放，实现了在任意回放树上对混合注意力模型的高效智能体强化学习训练，避免了共享前缀的重复计算。

    

    智能体强化学习（RL）通常会产生具有共享历史的不规则回放树。独立训练从根到叶的轨迹会重复计算这些共享前缀。现有系统主要针对全注意力模型，缺乏与激活重计算兼容的稠密、可微分的混合注意力执行。我们提出了HARTS（面向树结构的混合注意力强化学习）。HARTS在前缀压缩后，联合规划微批次、数据并行（DP）副本分配以及微批次槽调度，以利用无需重放的紧凑token工作。对于分块线性注意力，一种线性时间算法协调分块边界的状态恢复与重放，并在我们的打包执行模型下产生最少次数的顺序线性注意力调用。HARTS保留了按轨迹训练的分块状态划分方式：它不重复投影、MLP/MoE计算或最终输出，并且仅执行有界的（摘要在此处截断）

    arXiv:2608.28158v1 Announce Type: new  Abstract: Agentic reinforcement learning (RL) often produces irregular rollout trees with shared histories. Training root-to-leaf trajectories independently recomputes these shared prefixes. Existing systems primarily target full-attention models and lack dense, differentiable hybrid-attention execution compatible with activation recomputation. We present HARTS (Hybrid-Attention RL over Tree Structures). HARTS jointly plans microbatches, data-parallel (DP) replica assignments, and microbatch-slot schedules using non-replay compact-token work after prefix compression. For chunkwise linear attention, a linear-time algorithm coordinates chunk-boundary state recovery and replay and produces the minimum number of sequential linear-attention calls under our packed execution model. HARTS preserves the chunkwise state partitioning of trajectory-wise training: it does not repeat projections, MLP/MoE computation, or final outputs, and performs only bounded 
    
[^46]: 面向痴呆病房次日激越风险评分的床垫下时序传感

    Under-Mattress Temporal Sensing for Next-Day Agitation Risk Scoring in Dementia Wards

    [https://arxiv.org/abs/2608.28152](https://arxiv.org/abs/2608.28152)

    该研究利用床垫下非接触式传感采集的前夜分钟级时序信号来预测痴呆患者次日的激越风险，并证明保留时间结构的建模方法优于传统的整夜汇总特征。

    

    痴呆患者的激越症状会在短时间内发生波动，然而用于预测次日风险的连续生理信息却十分有限。我们评估了前一整夜的非接触式床垫下信号能否提示次日的激越风险，以及保留分钟级时间结构是否比传统的整夜汇总特征能带来更好的性能。我们利用两种床垫下传感系统，分析了一家专业医院痴呆病房中65名受试者的423个患者夜晚数据。通过统一的四范式基准，比较了整夜手工汇总特征、三时段手工特征、整夜序列建模以及滑动窗口多示例学习。研究采用了针对数据来源的预处理和五折按患者分组的交叉验证，性能通过汇总的折外预测进行估计。评估指标包括区分度、校准度、固定阈值指标，以及对p（摘要在此处截断）

    arXiv:2608.28152v1 Announce Type: cross  Abstract: Agitation fluctuates over short time horizons in people living with dementia, yet continuous physiological information for anticipating next-day risk is limited. We assessed whether contactless under-mattress signals from the preceding night inform next-day agitation risk and whether preserving minute-level temporal structure improves performance over conventional nightly summaries. We analyzed 423 patient-nights from 65 subjects in a specialized hospital dementia unit using two under-mattress sensing systems. A unified four-paradigm benchmark compared nightly handcrafted summaries, three-period handcrafted features, full-night sequence modeling, and sliding-window multiple-instance learning. Source-specific preprocessing and five-fold patient-grouped cross-validation were used, with performance estimated from pooled out-of-fold predictions. Evaluation included discrimination, calibration, fixed-threshold metrics, and a comparison of p
    
[^47]: Softmax注意力的逼近秩：尖锐的几何定律与稳健的交互维度

    The Approximation Rank of Softmax Attention: Sharp Geometric Laws and Robust Interaction Dimension

    [https://arxiv.org/abs/2608.28150](https://arxiv.org/abs/2608.28150)

    本文刻画了控制softmax注意力逼近秩的几何定律——球面自注意力的秩随维度指数增长而全球几何仅多项式增长，并证明固定注意力头下可见查询-键交互维度r给出极小极大紧的r/2秩上界。

    

    究竟哪种几何结构控制着归一化softmax注意力的秩复杂度？我们研究了最大行-ℓ₁逼近秩，即恰好能够保持每一个有界向量值输出所需的最低无约束秩。两个尖锐的最坏情形定律将支撑集几何分离出来：对于固定的维度d和误差ε，球面自注意力的秩为Θ_{d,ε}(min{n,(1+β)^{(d-1)/2}})，而全球几何增加了一个径向自由度，当β≥β₀(d,ε)且n≥C_d e^{β/8}时，其秩为Θ_{d,ε}(β^{d/2})。对于固定的注意力头，行softmax将行标量logit方向商掉：剩余的可见查询-键交互维度r产生每实例r/2的上界定律，且有界构造表明该指数在极小极大意义下是紧的。近似交互子空间会引入显式的残差输出误差，并给出以容差为索引的SVD维度。在一个包含84个头的BERT-base校准集上，w……（原文摘要至此截断）

    arXiv:2608.28150v1 Announce Type: new  Abstract: Which geometry controls the rank complexity of normalized softmax attention? We study maximum-row-$\ell_1$ approximation rank, exactly the least unrestricted rank preserving every bounded vector-valued output. Two sharp worst-case laws isolate support geometry: for fixed $d$ and error $\varepsilon$, spherical self-attention has rank $\Theta_{d,\varepsilon}(\min\{n,(1+\beta)^{(d-1)/2}\})$, while full-ball geometry adds one radial degree and, for $\beta\ge\beta_0(d,\varepsilon)$ and $n\ge C_d e^{\beta/8}$, gives $\Theta_{d,\varepsilon}(\beta^{d/2})$. For a fixed head, row-softmax quotients out row-scalar logit directions: the remaining visible query--key interaction dimension $r$ yields an $r/2$ per-instance upper law, and bounded constructions show this exponent is minimax sharp. Approximate interaction subspaces incur an explicit residual output error and yield a tolerance-indexed SVD dimension. On an 84-head BERT-base calibration set, w
    
[^48]: 面向节能驾驶的条件扩散模型

    Conditional Diffusion Models for Energy-Efficient Driving

    [https://arxiv.org/abs/2608.28142](https://arxiv.org/abs/2608.28142)

    提出了一种结合潜在条件编码器与时序一维U-Net去噪骨干的条件扩散框架，能以车辆速度和环境温度等路线特征为条件生成真实的电动汽车电池电流曲线，为能耗感知的车队运营决策提供支持。

    

    商业配送车队的电气化正在推动车队路径规划从基于距离和时间的优化转向基于能耗感知的决策。现有的序列模型主要提供确定性的点估计或有限的不确定性描述，无法捕捉运营决策所需的多种合理能耗轨迹范围。在本工作中，我们提出了一种条件扩散框架，该框架以车辆速度和环境温度等路线特征为条件，生成电动汽车电池电流曲线。该模型将潜在条件编码器与时序一维U-Net去噪骨干网络相结合，使行程相关条件能够映射到共享表示中，并引导反向扩散过程。我们在一个包含9辆车、12,000次行程的开放获取商业电动汽车遥测数据集上对该框架进行了评估。所提出的潜在条件扩散模型能够生成真实的……

    arXiv:2608.28142v1 Announce Type: new  Abstract: Electrification of commercial delivery fleets is shifting fleet routing from distance- and time-based optimization toward energy-aware decision-making. Existing sequence models primarily provide deterministic point estimates or limited uncertainty summaries, which do not capture the range of plausible energy-consumption trajectories required for operational decision-making. In this work, we introduce a conditional diffusion framework that generates EV battery-current profiles conditioned on route features such as vehicle velocity and ambient temperature. The model combines a latent conditioning encoder with a temporal 1D U-Net denoising backbone that enables trip-related conditions to be mapped into a shared representation and guides the reverse diffusion process. We evaluate the framework on an open-access commercial EV telemetry dataset containing 12k trips from 9 vehicles. The proposed latent-conditioned diffusion model generates real
    
[^49]: CheXtriev：面向胸部X线片基于病例检索的以解剖为中心的表示

    CheXtriev: Anatomy-Centered Representation for Case-Based Retrieval of Chest Radiographs

    [https://arxiv.org/abs/2608.28137](https://arxiv.org/abs/2608.28137)

    CheXtriev提出了一种基于图Transformer的解剖感知胸部X线片检索框架，通过从特定解剖区域提取特征并建模解剖位置与影像发现的相互作用，在检索准确率和排序质量上分别超越现有最先进方法18%-26%和11%-23%。

    

    我们提出了CheXtriev，一个基于图的、解剖感知的胸部X线片检索框架。与以往专注于全局特征的方法不同，我们的方法利用图Transformer从特定解剖区域提取有信息量的特征。此外，它还捕获了空间上下文以及解剖位置与影像发现之间的相互作用。这种基于循证解剖学的上下文化处理产生了更丰富的解剖感知表示，从而实现更准确、有效且高效的检索，尤其是对于较少见的影像发现。CheXtriev在检索准确率上比最先进的全局和局部方法高出18%至26%，在排序质量上高出11%至23%。代码已在 https://github.com/cvit-mip/chextriev 公开。

    arXiv:2608.28137v1 Announce Type: cross  Abstract: We present CheXtriev, a graph-based, anatomy-aware framework for chest radiograph retrieval. Unlike prior methods focussed on global features, our method leverages graph transformers to extract informative features from specific anatomical regions. Furthermore, it captures spatial context and the interplay between anatomical location and findings. This contextualization, grounded in evidence-based anatomy, results in a richer anatomy-aware representation and leads to more accurate, effective and efficient retrieval, particularly for less prevalent findings. CheXtriv outperforms state-of-the-art global and local approaches by 18% to 26% in retrieval accuracy and 11% to 23% in ranking quality. The code is available at https://github.com/cvit-mip/chextriev.
    
[^50]: 学习差分：面向时间序列预测的自适应可逆差分

    Learning to Difference: Adaptive Reversible Differencing (AdaRDiff) for Time Series Forecasting

    [https://arxiv.org/abs/2608.28134](https://arxiv.org/abs/2608.28134)

    提出AdaRDiff，一种利用可学习权重的自适应可逆差分方法，以单一算子同时去除并自回归恢复趋势与季节性成分，从而提升长时程时间序列预测的可靠性。

    

    可靠的长时程时间序列预测是一个重要却困难的问题。趋势和季节性引入了复杂的时序结构，给基于学习的预测模型带来了挑战。差分法通过减去邻近的过去值来消除这类结构，是经典的处理手段，但其依赖人工挑选的阶数和周期，因此近年来基本缺席于深度学习架构。我们提出自适应可逆差分，这是一种广义化的差分方法，利用可学习的权重，通过与先前时间点进行加权差分来简化序列。该方法在稳定化的残差上执行预测，随后以自回归方式恢复被移除的成分来重构预测结果，从而通过单一算子同时捕捉趋势与季节性。这种重构具有闭式解……（摘要在此处被截断）

    arXiv:2608.28134v1 Announce Type: new  Abstract: Reliable long-horizon time series forecasting is an important yet difficult problem. Trends and seasonality introduce complex temporal structure that challenges learning-based forecasting models. Differencing, which subtracts nearby past values to remove such structure, is the classical remedy, but its reliance on hand-picked orders and periods has kept it largely absent from recent deep architectures. We propose \textbf{\underline{Ada}}ptive \textbf{\underline{R}}eversible \textbf{\underline{Diff}}erencing \textbf{(AdaRDiff)}, a generalized differencing approach that uses learnable weights to simplify the series through weighted differencing with previous time instants. This yields stabilized residuals on which forecasting is performed, after which the removed components are restored autoregressively to reconstruct the forecast, capturing trend and seasonality jointly through a single operator. This reconstruction admits a closed-form c
    
[^51]: VICT：面向长程LLM智能体强化学习的验证器插桩式功劳追踪

    VICT: Verifier-Instrumented Credit Tracing for Long-Horizon LLM Agent Reinforcement Learning

    [https://arxiv.org/abs/2608.28128](https://arxiv.org/abs/2608.28128)

    VICT提出一种训练时接口，将终端验证器内部的可执行或证据支持的原子检查通过依赖证明边追溯到具体动作，并仅沿这些边重新分配组相对优势，从而解决长程LLM智能体强化学习中的细粒度功劳分配问题。

    

    细粒度的功劳分配是长程LLM智能体强化学习中的核心挑战。标准目标通常基于可编程验证的终端奖励进行训练，将每个稀疏的结果广播到轨迹中的每一个动作。现有方法通常从rollout侧寻求更细粒度的功劳信号，通过构建辅助轨迹信号或额外的比较来估计动作的重要性。尽管有效，这些方法仍然将判定成功的验证器当作一个标量奖励，丢弃了其内部蕴含的任务结构。我们的关键洞察是：许多可验证任务已经在终端验证器中编码了相关的检查逻辑。我们提出VICT（验证器插桩式功劳追踪），这是一个训练时接口，它将可执行或有证据支持的原子检查暴露出来，并通过依赖有效的证明边将这些检查追溯到具体动作。VICT仅沿着这些边重新分配组相对优势，从而转移……

    arXiv:2608.28128v1 Announce Type: new  Abstract: Fine-grained credit assignment is a central challenge in reinforcement learning for long horizon LLM agents. Standard objectives often train from programmatically verifiable terminal rewards by broadcasting each sparse outcome to every action in a trajectory. Existing methods typically seek finer credit from the rollout side, constructing auxiliary trajectory signals or additional comparisons to estimate action importance. Although useful, these approaches still treat the verifier that judged success as a scalar reward, discarding its internal task structure. Our key insight is that many verifiable tasks already encode the relevant checks inside their terminal verifier. We propose VICT (VerifierInstrumented Credit Tracing), a training-time interface that exposes executable or evidence backed atoms and traces them back to actions through dependency-valid proof edges. VICT redistributes group-relative advantage only along those edges, shif
    
[^52]: 面向预测组合的广义吉布斯集成加权

    Generalized Gibbs Ensemble Weighting for Forecast Combination

    [https://arxiv.org/abs/2608.28116](https://arxiv.org/abs/2608.28116)

    本文提出了广义吉布斯集成加权（GGEW）概率框架，将预测模型视为专家并通过归一化预测损失的吉布斯式指数变换分配集成权重，进一步扩展出数值稳定、多样性感知与在线自适应的一系列方法，以提升预测组合的稳健性与性能。

    

    当有多个预测模型可用时，预测组合是提高预测性能的可靠方法。简单的聚合规则（如均值、中位数、截尾均值、逆损失加权和指数加权）通常是很强的基线方法，但它们的相对性能可能因数据集、预测时间跨度、部署设置以及基础预测器之间的分歧程度而有所不同。我们提出了广义吉布斯集成加权（GGEW），这是一个概率框架，它将预测模型视为专家，并使用归一化预测损失的吉布斯式指数变换来分配集成权重。该框架通过数值稳定化、多样性感知的得分修正以及在线超参数自适应来扩展这一基本加权规则。GGEW产生了一系列相关方法，包括稳定吉布斯加权、方向性吉布斯-NCL和对称吉布斯-NCL。这些变体共享一个核心……

    arXiv:2608.28116v1 Announce Type: new  Abstract: Forecast combination is a reliable way to improve predictive performance when several forecasting models are available. Simple aggregation rules such as the mean, median, trimmed mean, inverse-loss weighting, and exponential weighting are often strong baselines, but their relative performance can vary across datasets, forecast horizons, deployment settings, and levels of disagreement among base forecasters. We develop Generalized Gibbs Ensemble Weighting (GGEW), a probabilistic framework that treats forecasting models as experts and assigns ensemble weights using a Gibbs-style exponential transformation of normalized predictive loss. The framework extends this basic weighting rule through numerical stabilization, diversity-aware score corrections, and online hyperparameter adaptation. GGEW produces a family of related methods, including Stable Gibbs weighting, Directional Gibbs-NCL, and Symmetric Gibbs-NCL. These variants share one core 
    
[^53]: 医学视觉模型真的能理解解剖结构吗？探究学习视觉表征的空间归纳偏置

    Do Medical Vision Models Reason About Anatomy? Probing the Spatial Inductive Biases of Learned Visual Representations

    [https://arxiv.org/abs/2608.28092](https://arxiv.org/abs/2608.28092)

    医学视觉模型在解剖空间推理上存在根本性缺陷，其看似准确的表现实则源于对典型解剖结构的记忆而非对图像的真正空间理解。

    

    解读CT扫描意味着比较两侧的结构、判断器官之间的距离、以及知道每个器官应该位于何处。医学视觉编码器通常仅通过诊断准确性进行评估，或通过组装的多模态系统进行评估（其中失败难以归因），因此尚不清楚其表征是否真正支持这些空间能力。我们构建了SPAR-Bench，这是一套在多器官腹部CT上的八个探测任务，将坐标定位、关系推理和空间查询分离开来，并将其应用于五种架构配置和三个医学基础模型（包括冻结和微调两种状态）。结果显示，要求在切片内进行比较的探测任务停留在随机水平，且预训练规模、微调和架构都无法缩小这一差距。在领域内看似已被解决的探测任务在零样本迁移下降至随机水平，表明其准确性反映的是对典型解剖结构的记忆，而非对图像本身的计算。

    arXiv:2608.28092v1 Announce Type: cross  Abstract: Interpreting a CT scan means comparing structures on either side, judging how far apart organs sit, and knowing where each one belongs. Medical vision encoders are evaluated on diagnostic accuracy, or through assembled multimodal systems where a failure is hard to attribute, so it remains unclear whether their representations support any of this. We construct SPAR-Bench, eight probes over multi-organ abdominal CT that separate coordinate localization, relational reasoning, and spatial queries, and apply them to five architectural configurations and three medical foundation models, frozen and finetuned. Probes that ask for a comparison within the slice stay at chance, and neither pretraining scale, finetuning, nor architecture closes the gap. Probes that appear solved in domain fall to chance under zero-shot transfer, indicating that their accuracy reflects recall of canonical anatomy rather than computation over the image. Reading the 
    
[^54]: 比较经典与量子机器学习在高能物理碰撞数据回归任务中的表现

    Comparing Classical and Quantum Machine Learning for Regression in High Energy Physics Collision Data

    [https://arxiv.org/abs/2608.28084](https://arxiv.org/abs/2608.28084)

    该论文系统比较了四种经典机器学习模型与其量子对应模型在CERN开放数据质子-质子对撞事件的横向动量回归任务上的表现，发现经典架构（尤其是CNN和LSTM）略优于量子模型。

    

    粒子碰撞事件的分类与回归构成了实验高能物理中一个持续的计算挑战，在这一领域中，海量模拟数据必须以兼顾速度与精度的方式进行处理。本工作对四种经典机器学习架构——支持向量机（SVM）、人工神经网络（ANN）、卷积神经网络（CNN）和长短期记忆网络——与其量子对应模型——量子支持向量机（QSVM）、量子神经网络（QNN）、量子卷积神经网络（QCNN）和量子长短期记忆网络（QLSTM）——进行了系统性比较。所有模型均在来自CERN开放数据门户的模拟质子-质子对撞事件上训练，末态为电子-正电子和μ子-反μ子，以横向动量分量作为输入特征，横向动量大小作为回归目标。经典架构，尤其是CNN和LSTM，取得了边际优势。

    arXiv:2608.28084v1 Announce Type: new  Abstract: The classification and regression of particle collision events constitute a persistent computational challenge in experimental high energy physics, where large volumes of simulated data must be processed with both speed and precision. This work carries out a systematic comparison of four classical machine learning architectures, support vector machines (SVM), artificial neural networks (ANN), convolutional neural networks (CNN), and long short-term memory (LSTM) networks against their quantum counterparts: quantum SVM (QSVM), quantum neural networks (QNN), quantum CNN (QCNN), and quantum LSTM (QLSTM). All models are trained on simulated proton-proton collision events with electron-positron and muon-antimuon final states from the CERN Open Data portal, using transverse-momentum components as input features and transverse-momentum magnitude as the regression target. Classical architectures, and in particular the CNN and LSTM, achieve margi
    
[^55]: 线性上下文学习中淬火临界性的朗道理论

    Landau theory of quenched criticality in linear in-context learning

    [https://arxiv.org/abs/2608.28059](https://arxiv.org/abs/2608.28059)

    该论文将线性上下文学习中的双下降奇异性表述为淬火无序系统的临界现象，构建了以重整化岭参数为序参量的朗道理论，并揭示学习参数的样本间涨落是奇异误差的微观起源。

    

    上下文学习（ICL）使预训练模型能够从提示中提供的示例推断新任务，而无需更新其参数。在ICL的线性模型中，当预训练样本数量与可学习参数数量相当时，预测误差会出现双下降奇异性。我们将这种插值奇异性表述为淬火无序系统的临界现象。通过比较同一线性ICL模型的退火与淬火描述，我们确定学习参数的样本间连通涨落是奇异误差的微观起源。通过对重整化岭参数 $\xi$ 的腔自洽方程进行积分，构建了朗道势。其中 $\xi$ 扮演（磁化）序参量的角色，而裸岭参数 $\lambda$ 则成为其共轭磁场。归一化样本复杂度 $\tau$ 作……（摘要在此处截断）

    arXiv:2608.28059v1 Announce Type: cross  Abstract: In-context learning (ICL) allows a pretrained model to infer a new task from examples supplied in its prompt without updating its parameters. In linear models of ICL, the prediction error develops a double-descent singularity when the number of pretraining samples becomes comparable to the number of learnable parameters. We formulate this interpolation singularity as a critical phenomenon of a quenched disordered system. By comparing annealed and quenched descriptions of the same linear ICL model, we identify the connected sample-to-sample fluctuations of the learned parameters as the microscopic origin of the singular error. A Landau potential is constructed by integrating the cavity self-consistency equation for the renormalized ridge parameter $\xi$. The role of (magnetization) order parameter is played by $\xi$, while the bare ridge parameter $\lambda$ becomes its conjugate magnetic field. The normalized sample complexity $\tau$ ac
    
[^56]: 面向可靠医疗人工智能的可解释不确定性估计

    Explainable Uncertainty Estimation for Reliable Medical AI

    [https://arxiv.org/abs/2608.28052](https://arxiv.org/abs/2608.28052)

    本文提出将不确定性估计与可解释人工智能相统一的新方法egRUE，能够量化医疗AI预测的不确定性并将其分解为特征层面的贡献，从而提升临床决策的可靠性与可解释性。

    

    人工智能在支持临床决策方面具有巨大潜力，但由于缺乏信任，其在医疗保健领域的应用仍然有限。不确定性估计可以标记不可靠的预测，可解释人工智能（XAI）可以阐明预测是如何做出的，但现有方法将两者分开处理，无法在特征层面提供洞察，即无法解释为什么某个预测是不确定的，或者应该优先进行哪些检查来降低这种不确定性。为了填补这一空白，我们提出了可解释不确定性估计，该方法将不确定性估计与XAI统一起来，既能量化不确定性，又能解释特征层面的贡献。我们引入了期望梯度重构不确定性估计，该方法将预测解释融入其不确定性计算中，并将不确定性分解为特征层面的贡献。我们证明了egRUE的理论性质，并通过实验表明它提高了可靠性和可解释性。

    arXiv:2608.28052v1 Announce Type: new  Abstract: Artificial intelligence has strong potential to support clinical decision-making, yet its adoption in healthcare remains limited due to a lack of trust. Uncertainty estimation can signal unreliable predictions, and explainable AI (XAI) can clarify how predictions are made but existing methods treat them separately, providing no feature-level insight into why a prediction is uncertain or which tests to prioritize to reduce it. To address this gap, we propose explainable uncertainty estimation, which unifies uncertainty estimation and XAI to both quantify uncertainty and explain feature-level contributions. We introduce the Expected Gradients Reconstruction Uncertainty Estimate (egRUE), which incorporates prediction explanations into its uncertainty computation and decomposes uncertainty into feature-wise contributions. We prove theoretical properties of egRUE and show through experiments that it improves reliability and interpretability c
    
[^57]: 从群体觅食中涌现的聚集现象

    Emergent aggregation from collective foraging

    [https://arxiv.org/abs/2608.28046](https://arxiv.org/abs/2608.28046)

    该研究表明，在没有任何直接群聚奖励的情况下，聚集行为可以从纯粹个体化的最优觅食目标中间接涌现——随着视觉范围增大，强化学习觅食者会从个体搜索策略急剧转变为集体搜索策略，并自发形成空间聚集。

    

    生物系统中的集体行为通常被建模为直接社会驱动的结果：个体因获得奖励或天生被设定为与邻居对齐或靠近邻居。本研究表明，聚集现象可以转而从间接目标中涌现。我们让强化学习觅食者（初始时进行随机游走）仅依据找到可再生目标的纯个体奖励来优化其动力学，同时它们只能感知同类而永远无法感知目标本身。随着视觉范围的增大，个体经历从适应环境的个体搜索到尺度无关的集体搜索的急剧转变，且这一转变与空间聚集的出现相吻合。因此，集体相作为最优觅食的副产品而产生，无需任何对群聚的直接奖励。一个极简的解析首达时间模型将这一转变再现为两种搜索策略之间的转变。

    arXiv:2608.28046v1 Announce Type: cross  Abstract: Collective behaviour in living systems is usually modelled as the outcome of a \emph{direct} social drive: agents are rewarded, or hard-wired, to align with or approach their neighbours. Here we show that aggregation can instead emerge from an \emph{indirect} objective. We let reinforcement learning foragers, initially performing a random walk, optimize their dynamics from a purely individual reward for finding replenishable targets, while perceiving only their conspecifics and never the targets themselves. As the visual range grows, the agents undergo a sharp crossover from an environment-tuned individual search to a scale-agnostic collective one, and this crossover coincides with the onset of spatial aggregation. Thus a collective phase arises as a by-product of optimal foraging, without any direct reward for grouping. A minimal analytical first-passage model reproduces the transition as a crossover between the two search strategies.
    
[^58]: GPU平台上LLM推理工作负载的请求与令牌能耗成本表征

    Characterization of Request and Token Energy Costs for LLM Inference Workloads on GPU Platforms

    [https://arxiv.org/abs/2608.28044](https://arxiv.org/abs/2608.28044)

    该论文提出一个分解式LLM推理能耗模型，揭示令牌归一化能耗指标的局限性，并在H100/H200 GPU上系统表征了请求能耗与令牌能耗随模型类型、批大小、上下文长度和输出长度的变化规律。

    

    大语言模型（LLM）推理服务按令牌计费，但GPU能耗却是在整个推理时间窗口内消耗的。这种核算方式的不匹配使得基于令牌归一化的能耗指标并不完整，因为即使总请求能耗在增加，平均输出令牌能耗仍可能下降。我们通过一个分解式能耗模型来表征这种行为：包含一次性的固定预填充（prefill）成本和固定的生成设置成本，而每个输出令牌的生成步骤则增加边际步骤能耗。我们在NVIDIA H100和H200 GPU上，针对稠密模型和混合专家模型评估了这一LLM推理能耗模型，报告了请求能耗和令牌能耗作为模型类型（M）、阶段（P）、批大小（B）、上下文长度（C）和输出长度（N）函数的变化规律。对于H200上批大小为16、上下文长度为4K的Llama-3.2-1B模型，当输出长度从10个令牌增加到512个令牌时，令牌能耗从7.46 J/token降至0.72 J/token，而总的批处理推理窗口能耗则从1.19持续增加。

    arXiv:2608.28044v1 Announce Type: cross  Abstract: Large language model (LLM) inference serving is priced by tokens, but GPU energy is consumed over inference windows. This accounting mismatch makes token-normalized metrics incomplete, since average output-token energy can decrease even when total request energy increases. We characterize this behavior with a decomposed energy model: a fixed one-time prefill with a fixed generation setup cost, while each output-token generation step adds marginal step energy. We evaluate this LLM inference energy model on NVIDIA H100 and H200 GPUs across dense and mixture-of-experts (MoE) models, reporting both request energy and token energy as functions of model type (M), phase (P), batch size (B), context length (C), and output length (N). For Llama-3.2-1B on H200 at batch-16 and context-4K, increasing output length from 10 to 512 tokens reduces token energy from 7.46 to 0.72 J/token while total batched inference-window energy increases from 1.19 to
    
[^59]: 双生世界：基于等变性的弃答机制实现证据接地推理

    Twin Worlds: Equivariance-Based Abstention for Evidence-Grounded Reasoning

    [https://arxiv.org/abs/2608.28018](https://arxiv.org/abs/2608.28018)

    该论文提出“双生世界”（TW）框架，通过等变性检验模型的推理是否真正以证据为依据，使模型在证据不足时能够弃答，从而避免生成看似合理却缺乏证据支撑的答案。

    

    知识密集型推理要求大语言模型（LLM）将答案建立在所提供的证据之上。当证据不足时，理想的情形是模型选择弃答，而不是自信地生成缺乏依据的答案。现有的弃答方法依赖于不确定性估计或证据充分性检查，但二者均无法检验生成过程——即由所提供证据与模型内部记忆参数的交互所驱动的推理过程——是否真正以证据为依据。一个关键因素是，上下文中的实体提及会激活模型记忆中的关联，导致模型生成看似合理却缺乏证据支撑的回答。我们提出了双生世界（Twin Worlds, TW）框架，通过基于等变性的弃答机制来提升知识密集型推理的可靠性：与要求输出保持不变的不变性不同，等变性要求输出在实体……（原文摘要在此处截断）

    arXiv:2608.28018v1 Announce Type: new  Abstract: Knowledge-intensive reasoning requires Large Language Models (LLMs) to ground answers in provided evidence. When evidence is insufficient, it is desirable that models abstain rather than confidently generating unsupported answers. Existing abstention methods rely on uncertainty estimation or evidence sufficiency checks, but neither tests whether the reasoning process for generation, driven by the interaction of provided evidence and the model's internal memory parameters, is actually grounded in the evidence. A key contributing factor is that entity mentions in context activate memorised associations, causing models to generate plausible responses ungrounded in evidence. We propose Twin Worlds (TW), a framework for improving reliability in knowledge-intensive reasoning through equivariance-based abstention: unlike invariance, which requires outputs to remain unchanged, equivariance requires outputs to transform correspondingly under enti
    
[^60]: 条件流匹配何时可以替代逐点负对数似然？

    When Can Conditional Flow Matching Replace Pointwise Negative Log-Likelihood?

    [https://arxiv.org/abs/2608.28010](https://arxiv.org/abs/2608.28010)

    本文通过将端点NLL精确分解为熵、加权CFM目标、内部速度-得分残差和边界残差，刻画了条件流匹配损失何时能替代逐点负对数似然，证明普通CFM通常不是逐点NLL估计器，而特定加权 \(w_{\mathrm{sc}}(t)=(1-t)/t\) 在离策略最优下可消除内部残差，但该结论不能推广到训练或在线策略对齐场景。

    

    流匹配使得无似然训练成为可能，然而对齐方法越来越多地将条件流匹配（CFM）损失重新用作端点负对数似然（NLL），并将其新旧差异用作对数似然比。我们刻画了这些替代何时有效。对于线性高斯路径，我们将端点NLL精确分解为熵、加权CFM目标、内部速度-得分残差和边界残差。因此，仅基于CFM的估计和差异只有在相应残差相互抵消时才是精确的。在离策略总体最优处，普通CFM通常并不是逐点NLL估计器，而权重函数 \(w_{\mathrm{sc}}(t)=(1-t)/t\) 可以消除内部残差；但这一积极结果通常不能推广到训练或在线策略对齐。即使对于相同的端点分布，或在替代目标优化之后，在线策略的对数似然比仍可能保持有偏。在跨维度、分布和几何结构的实验中（原文在此处截断）……

    arXiv:2608.28010v1 Announce Type: new  Abstract: Flow matching enables likelihood-free training, yet alignment methods increasingly reuse conditional flow matching (CFM) losses as endpoint negative log-likelihoods (NLLs) and their old/new differences as log-likelihood ratios. We characterize when these substitutions are valid. For linear Gaussian paths, we exactly decompose endpoint NLL into entropy, a weighted CFM objective, an interior velocity--score residual, and a boundary residual. Thus CFM-only estimates and differences are exact only when the corresponding residuals cancel. At the off-policy population optimum, ordinary CFM is not generally a pointwise NLL estimator, whereas \(w_{\mathrm{sc}}(t)=(1-t)/t\) removes the interior residual; this positive result does not extend generally to training or on-policy alignment. On-policy log-ratios can remain biased even for identical endpoint laws or after surrogate optimization. Experiments across dimensions, distributions, and geometri
    
[^61]: 线性回归中加权数据选择的精确风险比率

    Exact Risk Ratios for Weighted Data Selection in Linear Regression

    [https://arxiv.org/abs/2608.28007](https://arxiv.org/abs/2608.28007)

    本文解决了 Hanneke 等人提出的加权数据选择公开问题，精确确定了最小范数 ERM 下的最坏风险比率 F_w(d,n) 的多个取值，包括证明 F_w(d,2d-1)=1+1/d，并给出了中间预算 n=d+k 情形下基于平衡划分调和量的紧致下界 1+Γ_{d,k}。

    

    Hanneke、Moran、Shlimovich 和 Yehudayoff（COLT 2025）提出了如下公开问题：一个选择者看到有限数据集 D ⊆ R^d × R，挑选至多 n 个样本并赋予非负权重，然后将加权最小二乘目标交给最小范数 ERM 求解。记 F_w(d,n) 为返回预测器在全部数据 D 上的损失与最优损失之间的最坏情况比率，他们证明了当 n<2d 时 F_w(d,n)=∞。本文在若干情形下确定了该值。对每个 d，我们证明了 F_w(d,2d-1)=1+1/d，这证实了原始笔记中一个未给出证明的论断。我们进一步证明了 F_w(3,4)=5/3 和 F_w(4,5)=2，这是端点公式未能覆盖的两个最小的格子。对每个中间预算 n=d+k，我们证明了下界 F_w(d,d+k) ≥ 1+Γ_{d,k}，其中 Γ_{d,k} 是关于平衡划分的一个显式调和量，并且我们证明该下界即为精确的最小值。

    arXiv:2608.28007v1 Announce Type: new  Abstract: Hanneke, Moran, Shlimovich and Yehudayoff (COLT 2025) posed the following open problem. A selector sees a finite dataset $D \subseteq \mathbb{R}^d \times \mathbb{R}$, picks at most $n$ examples together with nonnegative weights, and hands the weighted least squares objective to the minimum-norm ERM. Writing $F_w(d,n)$ for the worst-case ratio between the loss of the returned predictor on all of $D$ and the optimal loss, they proved $F_w(d,n)=\infty$ for $n<2d$. We determine this value in several cases. For every $d$ we prove $F_w(d,2d-1)=1+1/d$, which confirms a claim stated without proof in the original note. We further prove $F_w(3,4)=5/3$ and $F_w(4,5)=2$, the two smallest cells not covered by the endpoint formula. For every intermediate budget $n=d+k$ we prove the lower bound $F_w(d,d+k) \ge 1+\Gamma_{d,k}$, where $\Gamma_{d,k}$ is an explicit harmonic quantity over balanced partitions, and we show that this bound is the exact minima
    
[^62]: 一种基于质量退化约束下性能最大化的LLM量化层位宽分配方法

    A Method for Layer Bit-Width Allocation in LLM Quantization via Performance Maximization Under a Quality-Degradation Constraint

    [https://arxiv.org/abs/2608.28003](https://arxiv.org/abs/2608.28003)

    该论文提出一种在质量退化预算约束下通过最大化推理性能来为LLM逐层分配量化位宽的方法，并借助TensorRT-LLM实测区分了FFN、Attention和lm_head各模块对整体加速的贡献。

    

    本文提出了一种针对Gemma-3-1B的层位分配方法，将该问题形式化为在退化预算约束（即允许的生成质量损失水平）下的性能最大化问题（降低延迟）。该方法不同于文献中耗时耗资源的统一层量化方法（如GPTQ或AWQ），也不同于未经验证具有性能加速效果的分配方法（如MixLLM或TorchAO）。我们先前工作SA-PTQ所得的层敏感度曲线通过TensorRT-LLM内部的激活直通模式加以应用。对于每一层，根据前一步骤引入的分组方式（5+5、10+10、all26），以块为单位逐一确定精度，从而区分FFN、Attention和lm_head对整体加速的贡献。我们在RTX 5090上测量了13种W8A8变体的时钟速度。我们发现，对于FFN和lm_head，量化带来的时间成本……（摘要原文在此处截断）

    arXiv:2608.28003v1 Announce Type: new  Abstract: This paper proposes a layer bit allocation method for Gemma-3-1B, formulating the problem as performance maximization (latency decrease) given a degradation budget constraint (allowable level of generation quality loss). This approach is different from time- and resource-consuming uniform layer quantization methods that are used in the literature (like GPTQ or AWQ) or allocation methods without proven performance-accelerating effect (like MixLLM or TorchAO). The layer sensitivity profile resulting from our prior work SA-PTQ is applied using the activation pass-through mode inside TensorRT-LLM. For each layer precision is determined individually in blocks, according to a grouping introduced in the prior step (5+5, 10+10, all26), differentiating the contribution of FFN, Attention, and lm_head to the overall speedup. The clock speed was measured for 13 W8A8 variants on an RTX 5090. We find that for FFN and lm_head the time cost of quantizat
    
[^63]: 蒙特卡洛树搜索仅仅是每次访问蒙特卡洛控制吗？

    Is Monte Carlo Tree Search Just Every-Visit Monte Carlo Control?

    [https://arxiv.org/abs/2608.27985](https://arxiv.org/abs/2608.27985)

    本文论证了MCTS的四个阶段（选择、扩展、模拟、回传）本质上可归结为“当前策略下的轨迹采样”和“每次访问蒙特卡洛更新”两个基本操作，即MCTS实质上就是每次访问蒙特卡洛控制。

    

    蒙特卡洛树搜索（MCTS）和每次访问蒙特卡洛（MC）控制通常被呈现为两种不同的方法。MCTS是用搜索的语言来描述的（选择、扩展、模拟和回传），而MC控制则是用强化学习的语言来描述的（轨迹采样、回报估计、动作价值更新和策略改进）。本文认为，在轨迹生成和动作价值更新的层面上，这两者的区别在很大程度上只是术语上的差异。树策略和模拟策略可以被视为同一个不断演化的策略中“已学习”和“尚未学习”的部分；扩展对应于首次访问和初始化；而回传就是普通的每次访问蒙特卡洛更新。在这种解释下，MCTS的四个阶段可以简化为两个基本操作：在当前策略下的轨迹采样和每次访问蒙特卡洛更新。从这个意义上说，MCTS仅仅是每次访问蒙特卡洛……

    arXiv:2608.27985v1 Announce Type: new  Abstract: Monte Carlo Tree Search (MCTS) and every-visit Monte Carlo (MC) control are usually presented as different methods. MCTS is described in the language of search (selection, expansion, simulation, and backup), whereas MC control is described in the language of reinforcement learning (trajectory sampling, return estimation, action-value updating, and policy improvement). This note argues that, at the level of trajectory generation and action-value updating, the distinction is largely terminological. The tree policy and rollout policy can be viewed as the learned and not-yet-learned parts of a single evolving policy; expansion corresponds to first visit and initialization; and backup is the ordinary every-visit Monte Carlo update. Under this interpretation, the four stages of MCTS reduce to two basic operations: trajectory sampling under the current policy and every-visit Monte Carlo updating. In this sense, MCTS is simply every-visit Monte 
    
[^64]: PhyMamba：用于鲁棒电池健康预测的物理调制Mamba

    PhyMamba: Physics-Modulated Mamba for Robust Battery Health Prognostics

    [https://arxiv.org/abs/2608.27978](https://arxiv.org/abs/2608.27978)

    提出PhyMamba两阶段物理调制Mamba框架，将电化学老化机理融入序列建模，仅利用BMS信号即可实现无需侵入式测量的鲁棒长期电池健康预测。

    

    电池健康预测是电池管理系统（BMS）中的核心功能，然而由于对运行条件的依赖性以及传感器噪声的影响，基于BMS信号进行长期健康预测仍然极具挑战性。在本文中，我们提出了PhyMamba，这是一个两阶段的物理调制Mamba框架，将电化学老化机理融入序列建模之中。PhyMamba无需显式识别内部老化参数，而这类参数识别通常依赖于侵入式测量。在第一阶段，一个轻量级的Mamba编码器首先处理BMS信号并生成潜在表示，随后通过老化参数化模块将其转化为物理信息驱动的老化特征。在第二阶段，一个定制的Mamba预测骨干网络执行多周期预测，其中物理机制被紧密集成，用以调节模型内部的时间更新，使其朝着与退化规律一致的演化方向进行。在三个公开数据集上、多种工况条件下的实验……

    arXiv:2608.27978v1 Announce Type: new  Abstract: Battery health prognostics is a core function in battery management systems (BMSs), yet long-horizon health forecasting from BMS signals remains challenging due to operating-condition dependency and sensor noise. In this paper, we propose PhyMamba, a two-stage physics-modulated Mamba framework that integrates electrochemical aging into sequence modelling. PhyMamba does not require explicit identification of internal aging parameters, which often relies on intrusive measurements. In stage-1, a lightweight Mamba encoder first processes BMS signals and produces a latent representation that is transformed via an aging parameterization module, into physics-informed aging features. In stage-2, a customized Mamba forecasting backbone performs multi-cycle prediction, where physics is tightly integrated to regulate the model's internal temporal updates toward degradation-consistent evolution. Experiments on three public datasets under multiple fo
    
[^65]: 非为破坏，而为见证：用于隐私保护大语言模型验证的对抗性探针

    Not to Break, but to Attest: Adversarial Probes for Privacy-Preserving LLM Verification

    [https://arxiv.org/abs/2608.27954](https://arxiv.org/abs/2608.27954)

    提出一种基于zk-SNARK的隐私保护审计框架，利用对抗性探针放大模型部署后修改所导致的logit漂移，从而在无需访问专有模型权重的情况下验证大语言模型是否被篡改。

    

    大语言模型在部署后的改动可能在保持常规输出大体不变的情况下改变模型行为，这在模型权重为专有资产时给AI治理带来了挑战。我们提出了一个基于zk-SNARK的隐私保护审计框架，该框架搜索以对抗样本思路设计的探针，以放大已批准模型与修改后部署之间的logit漂移。我们的框架在不同的访问模型下探索互补的探针族：基于词元的探针在黑盒设置下运行，仅需要输入接口、分词器和词表；基于嵌入的探针需要对嵌入接口的灰盒访问；压力探针依赖额外的接口能力，但不需要对模型权重或架构的完全白盒访问。这一范围使探针选择能够在敏感性、访问需求和部署成本之间取得平衡。我们在多个大语言模型上评估了探针构建方法。

    arXiv:2608.27954v1 Announce Type: cross  Abstract: Post-deployment changes to large language models can alter behavior while leaving routine outputs largely unchanged, creating a challenge for AI governance when model weights are proprietary. We present a privacy-preserving zk-SNARK-based audit framework that searches for probes designed in the spirit of adversarial examples to amplify logit drift between an approved model and a modified deployment. Our framework explores complementary probe families under different access models. Token-based probes operate in a black-box setting and require only the input interface, tokenizer, and vocabulary. Embedding-based probes require gray-box access to the embedding interface. Stress probes rely on additional interface capabilities but do not require full white-box access to model weights or architecture. This range allows probe selection to balance sensitivity, access requirements, and deployment cost. We evaluate probe constructions across LLM
    
[^66]: 基于时序记忆感知的动态图在线测试时自适应

    Temporal Memory-Aware Online Test-Time Adaptation on Dynamic Graphs

    [https://arxiv.org/abs/2608.27948](https://arxiv.org/abs/2608.27948)

    提出了DGOTTA框架，通过时序记忆感知的在线测试时自适应方法，有效应对动态图上DGNN模型因结构和语义持续演化而面临的分布偏移挑战。

    

    arXiv:2608.27948v1 公告类型：新论文 摘要：图上的测试时自适应（TTA）旨在将在训练图上训练良好的图神经网络（GNN）适配到测试图上，其中存在可能损害模型泛化能力和测试时推理的潜在分布偏移。虽然近期的研究已经探索了静态图上的TTA，但在使用动态GNN（DGNN）模型学习的动态图上仍存在研究空白，因为动态图中结构连接和节点语义都会随时间持续演化。这使得适配DGNN模型以获得可靠的测试时性能面临巨大挑战。为填补这一空白，本工作提出了一种新颖的时序记忆感知的动态图在线测试时自适应框架，命名为DGOTTA，以在测试时有效地适配训练良好的DGNN。具体而言，所提出的DGOTTA包含三个模块：（1）时序感知增强，用于扩展测试动态图的多样性以应对复杂的时序……（摘要内容在此处被截断）

    arXiv:2608.27948v1 Announce Type: new  Abstract: Test-time adaptation (TTA) on graphs aims to adapt a graph neural network (GNN) that is well-trained on the training graph to the test graph, which involves potential distribution shifts that may harm model generalization and test-time inference. While recent efforts have investigated TTA on static graphs, there is still a research gap on dynamic graphs learned with dynamic GNN (DGNN) models, where both structural connectivity and node semantics evolve continuously over time. This makes adapting a DGNN model for reliable test-time performance substantially challenging. To fill this gap, in this work, we propose a novel framework of temporal memory-aware Online Test-Time Adaptation on Dynamic Graphs, named DGOTTA, to effectively adapt well-trained DGNNs during test time. Specifically, the proposed DGOTTA contains three modules: (1) temporal-aware augmentation, to extend the diversity of test dynamic graphs for addressing complex temporal 
    
[^67]: TI$^2$PS：一个基于拓扑信息的随机多细胞模式形成逆向设计框架

    TI$^2$PS: A Topology-Informed Inverse Design Framework for Stochastic Multicellular Pattern Formation

    [https://arxiv.org/abs/2608.27931](https://arxiv.org/abs/2608.27931)

    该研究提出TI²PS框架，通过结合拓扑数据分析中的贝蒂数向量与逆向代理建模，实现了从目标多细胞空间模式直接推断基于智能体模型的细胞级参数，并在斑马鱼色素模式形成模型中验证了其有效性。

    

    本研究提出了一种新颖的框架，用于估计基于智能体模型（ABM）中重现目标多细胞模式所需的参数。多细胞ABM面临的两大挑战是：估计细胞级别的参数（智能体特定变量），以及在随机细胞增殖和死亡条件下定量评估多细胞排列的拓扑特征。为解决这些挑战，我们整合了两种方法：贝蒂数向量（Betti vectors）和逆向代理建模。通过拓扑数据分析获得的贝蒂数向量能够一致地表示广泛的多细胞空间构型特征。逆向代理建模则能够从目标模式直接推断出相应的ABM参数。我们使用斑马鱼色素模式形成——一个由多细胞相互作用驱动的模式形成的代表性模型——验证了所提出的框架。结果表明……

    arXiv:2608.27931v1 Announce Type: new  Abstract: This study proposes a novel framework to estimate parameters for reproducing target multicellular patterns using an agent-based model (ABM). Two major challenges in multicellular ABMs are estimating cell-level parameters (agent-specific variables) and quantitatively evaluating the topological characteristics of multicellular arrangements under stochastic cell proliferation and death. To address these challenges, we integrate two approaches: Betti vectors and inverse surrogate modeling. The Betti vectors obtained through topological data analysis can consistently represent features of a wide range of multicellular spatial configurations. The inverse surrogate modeling enables direct inference of the corresponding ABM parameters from the target patterns. We validated the proposed framework using zebrafish pigment pattern formation, a representative model of pattern formation driven by multicellular interactions. The results demonstrate tha
    
[^68]: TACIT-Switch：基于删失监督的LLM智能体成本感知模型升级方法

    TACIT-Switch: Cost-Aware Model Escalation for LLM Agents from Censored Supervision

    [https://arxiv.org/abs/2608.27911](https://arxiv.org/abs/2608.27911)

    提出TACIT-Switch方法，利用教师标注的删失干预时间学习永久性移交策略，以成本感知的方式决定何时将LLM智能体从小模型骨干升级至大模型骨干，部署时无需教师参与即可将成功率提升7.4-11.1个百分点。

    

    采用较小语言模型骨干的智能体成本较低，但可能陷入持续的失败模式；而采用较大骨干的智能体通常更可靠，但成本更高。这种可靠性-成本之间的权衡促使人们研究路由方法，以决定何时调用具有更大骨干的智能体：可以在执行之前、在固定的轨迹前缀之后，或在各个步骤局部进行。我们的方法TACIT-SWITCH从累积的轨迹证据和教师标注的删失干预时间中学习永久性移交策略。它将每个标注表示为累积风险尺度上的区间删失观测。由此得到的混合治愈阈值模型可估计配对的强模型rollout成功的概率，以及在成功条件下的移交阈值；部署时无需教师参与。在基于机制的多步模拟中，TACIT-SWITCH相比基线方法将成功率提升了7.4-11.1个百分点。

    arXiv:2608.27911v1 Announce Type: new  Abstract: Agents with smaller language-model backbones are less expensive but can drift into persistent failure modes, whereas those with larger backbones are generally more reliable but more costly. This reliability-cost trade-off motivates routing methods that decide when to invoke an agent with a larger backbone: before execution, after a fixed trajectory prefix, or locally at individual steps. Our method, TACIT-SWITCH, learns permanent handoff policies from accumulated trajectory evidence and Teacher-Annotated Censored Intervention Times (TACIT). It represents each annotation as an interval-censored observation on a cumulative-risk scale. The resulting mixture-cure threshold model estimates the probability that the paired Strong rollout succeeds and, conditional on success, the handoff threshold; no teacher is required at deployment. In a mechanism-based multi-step simulation, TACIT-SWITCH improves success by 7.4-11.1 percentage points over ta
    
[^69]: OpenStamp：面向开源语言模型的水印技术

    OpenStamp: A Watermark for Open-Source Language Models

    [https://arxiv.org/abs/2608.27899](https://arxiv.org/abs/2608.27899)

    OpenStamp通过仅修改开源语言模型的反嵌入层，将水印逻辑直接编码进模型权重，解决了传统采样概率水印在白盒场景下可被用户禁用的问题，在几乎不损失模型能力的前提下实现了更优的检测性能和更强的鲁棒性。

    

    随着大语言模型（LLM）生成内容的日益普及，水印技术被视为一种将文本归属于LLM并与人类撰写内容相区分的有前景的方法。一类突出的技术通过修改token的采样概率，在生成文本中嵌入细微但可检测的信号。然而，这类方法并不适用于开源模型，因为用户拥有白盒访问权限，可以在推理过程中轻松禁用水印。在这项工作中，我们提出了OpenStamp，一种水印技术，它通过仅修改最终的投影层（即反嵌入层，unembedding layer），将水印逻辑直接编码到模型权重中。通过在两个模型上的实验，我们证明OpenStamp实现了更优的检测性能，且与先前方法相比，模型能力的退化极小。植入的水印经过专门设计，并经实验证实，对扰动（p…

    arXiv:2608.27899v1 Announce Type: new  Abstract: With the growing prevalence of large language model (LLM) generated content, watermarking is considered a promising approach for attributing text to LLMs and distinguishing it from human-written content. A prominent class of techniques embeds subtle but detectable signals in generated text by modifying token sampling probabilities. However, such methods are unsuitable for open-source models, where users have white-box access and can easily disable watermarking during inference. In this work, we introduce OpenStamp, a watermarking technique that encodes the watermarking logic directly into the model weights by modifying only the final projection, or unembedding, layer. Through experiments across two models, we show that OpenStamp achieves superior detection performance, with minimal degradation in model capabilities compared to prior methods. The implanted watermark is explicitly designed, and empirically confirmed, to be more robust to p
    
[^70]: 往返之间：用于多模态翻译的双向扩散桥

    There and Back Again: Bidirectional Diffusion Bridges for Multimodality Translation

    [https://arxiv.org/abs/2608.27885](https://arxiv.org/abs/2608.27885)

    提出了双向图像-文本扩散桥BIT，直接从文本出发插值到图像，实现了源感知的生成路径与可逆的双向多模态生成框架。

    

    多模态翻译（例如文本生成图像）是生成式人工智能的一项核心任务。然而，现有方法（1）所遵循的生成路径并不直接表示源模态，限制了某些采样算法的灵活性；（2）是单向的，无法进行逆过程（例如图像生成文本）。我们提出了BIT：双向图像-文本扩散桥。与以往方法不同，BIT直接从文本出发并插值到图像，提供（1）源感知的生成路径，支持多样且灵活的采样算法；（2）端点条件化的过程，可以从图像遍历到文本，提供了一个统一的双向生成框架。BIT通过随机微积分推导得出，得到了适合模拟的SDE形式以及可扩展到高维度的易处理损失函数。我们的实验表明，BIT与去噪扩散和确定性流基线方法相比具有竞争力。

    arXiv:2608.27885v1 Announce Type: new  Abstract: Multimodality translation (e.g., text-to-image) is a core generative AI task. However, existing approaches (1) follow generative paths that do not directly represent the source modality, limiting the flexibility of some sampling algorithms; and (2) are unidirectional, preventing inversion (e.g., image-to-text). We propose BIT: Bidirectional Image-Text Diffusion Bridges. In contrast to previous approaches, BIT starts directly from text and interpolates into images, providing (1) a source-aware generative path that enables diverse and flexible sampling algorithms; and (2) an endpoint-conditioned process that can be traversed from image to text, providing a unified, bidirectional generative framework. BIT is derived through stochastic calculus, yielding SDE forms amenable to simulation and tractable loss functions that scale to high dimensions. Our experiments show that BIT is competitive with denoising-diffusion and deterministic-flow base
    
[^71]: 科学中超越成对图：面向参数化偏微分方程的超图自适应小波算子

    Beyond Pairwise Graphs in Science: Hypergraph Adaptive Wavelet Operators for Parametric PDEs

    [https://arxiv.org/abs/2608.27883](https://arxiv.org/abs/2608.27883)

    提出超图自适应小波算子HALO，将求解域提升至超图并在其谱小波域中学习，突破了成对图神经算子的局限，可直接捕获网格单元间的群体耦合，从而提升非结构化网格上参数化偏微分方程求解的精度与稳定性。

    

    物理系统通常由解算子来建模，这些算子将输入场、参数、几何形状或过去状态映射到稳态或未来的物理状态。学习这些映射是困难的，尤其是对于必须融合历史信息并在自回归滚动中保持稳定的时间依赖系统。许多神经算子在规则的、结构化的网格上表现最佳，而现实模拟往往需要非结构化网格或点云来刻画复杂几何；在这种设置下，以网格为中心的表示可能会损失精度。图神经算子通过消息传递或谱图滤波来处理这些域，但成对的边无法直接捕获网格单元、局部邻域或守恒体积之间的群体耦合。我们提出了超图自适应小波算子（HALO），它将求解域提升到超图上，并在其谱小波域中进行学习。HALO避免了显式的超图-拉普拉斯……

    arXiv:2608.27883v1 Announce Type: new  Abstract: Physical systems are often modeled by solution operators that map input fields, parameters, geometries, or past states to steady or future physical states. Learning these maps is difficult, especially for time-dependent systems that must assimilate history and remain stable under autoregressive rollout. Many neural operators work best on regular, structured grids, while realistic simulations often require unstructured meshes or point clouds to resolve complex geometries; in such settings, grid-centric representations can lose accuracy. Graph neural operators handle these domains through message passing or spectral graph filtering, but pairwise edges do not directly capture group-wise couplings among mesh cells, local neighborhoods, or conservation volumes. We introduce the Hypergraph Adaptive waveLet Operator (HALO), which lifts the domain to a hypergraph and learns in its spectral wavelet domain. HALO avoids explicit hypergraph-Laplacia
    
[^72]: SOMTab：面向高效表格上下文学习的集合序Mamba架构

    SOMTab: Set-Order Mamba for Efficient Tabular In-Context Learning

    [https://arxiv.org/abs/2608.27882](https://arxiv.org/abs/2608.27882)

    SOMTab提出一种集合序Mamba架构，用基于Mamba的状态空间混合替代表格上下文学习中构建行列表示所需的注意力机制，仅在最终预测阶段保留注意力以实现查询条件检索，从而实现更高效的表格上下文学习。

    

    基于上下文学习的表格基础模型近来已成为任务特定模型拟合的有力替代方案。然而，当前的性能前沿仍由重度依赖注意力机制的架构主导，注意力机制被用于整个建模流程之中。这引出了一个自然的问题：表格上下文学习的每个阶段都需要注意力机制吗？我们提出SOMTab，一种面向高效表格上下文学习的集合序Mamba架构。SOMTab将表示构建与基于查询条件的检索分离开来。在行与列表示方面，它将无序的表格标记映射到稳定的潜在槽位中，并应用基于Mamba的状态空间混合来构建紧凑的表示；在最终预测方面，它保留基于注意力的上下文学习，以保持从带标签的上下文样本中进行查询条件检索的能力。我们进一步引入DCH-TailMix，一种合成先验，它结合了度相关（摘要在此处被截断）……

    arXiv:2608.27882v1 Announce Type: new  Abstract: Tabular foundation models based on in-context learning have recently emerged as strong alternatives to task-specific model fitting. However, the current performance frontier remains dominated by attention-heavy architectures, where attention is used throughout the modeling pipeline. This raises a natural question: is attention necessary at every stage of tabular in-context learning? We introduce SOMTab, a Set-Order Mamba architecture for efficient tabular in-context learning. SOMTab separates representation construction from query-conditioned retrieval. For row and column representations, it maps unordered table tokens into stable latent slots and applies Mamba-based state-space mixing to construct compact representations. For final prediction, it retains attention-based in-context learning to preserve query-conditioned retrieval from labeled context examples. We further introduce DCH-TailMix, a synthetic prior that combines degree-corre
    
[^73]: 交互表征究竟衡量了什么？弱监督暴力检测中的事件前可分性

    What Do Interaction Representations Actually Measure? Pre-Event Separability in Weakly-Supervised Violence Detection

    [https://arxiv.org/abs/2608.27879](https://arxiv.org/abs/2608.27879)

    该研究在固定下游流程的严格控制条件下，通过弱监督早期暴力检测系统比较五种交互表征，发现基于人体姿态的表征并不优于粗略的边界框几何特征。

    

    关节化人体姿态提供了超越粗略空间关系的精细身体构型信息，但在下游处理流程保持固定的情况下，这种细节是否能带来更强的判别信息仍不清楚。我们通过早期暴力检测来考察这一问题。在保持跟踪器、时序模块、监督方式、数据划分和评估方法全部固定的情况下，我们比较了五种交互表征，涵盖粗略的边界框几何、与之匹配的手工姿态类似物、丰富的姿态描述符，以及从原始关节点学习的容量匹配编码器，并采用视频级评估和聚类自助法置信区间。没有任何一种基于姿态的表征优于粗略几何，尽管在十五个异常视频的子集上无法排除小效应的存在。将流程扩展到冻结的视觉编码器，并在XD-Violence数据集（137个异常视频，是我们UCF-Crime样本的九倍）上重复该比较，人物裁剪外观和整体……（原文摘要在此处截断）

    arXiv:2608.27879v1 Announce Type: cross  Abstract: Articulated human pose provides detailed body-configuration information beyond coarse spatial relationships, but whether this detail yields greater discriminative information when the downstream pipeline is held fixed remains unclear. We examine this through early violence detection. Holding the tracker, temporal head, supervision, folds, and evaluation fixed, we compare five interaction representations spanning coarse bounding-box geometry, a matched handcrafted pose analogue, enriched pose descriptors, and a matched-capacity encoder learned from raw joints, under video-level evaluation with cluster-bootstrap intervals. No pose-based representation outperforms coarse geometry, though with fifteen anomalous videos this subset cannot rule out small effects. Extending the pipeline to frozen visual encoders, and repeating the comparison on XD-Violence (137 anomalous videos, nine times our UCF-Crime sample), person-crop appearance and whol
    
[^74]: 面向失效感知首次命中批量逆向设计的锚定场景覆盖方法

    Anchored Scenario Coverage for Failure-Aware First-Hit Batch Inverse Design

    [https://arxiv.org/abs/2608.27873](https://arxiv.org/abs/2608.27873)

    本文提出ARC-SC批量采集方法，通过锚定强边际候选对象并在风险支持约束下最大化对预测目标场景的互补覆盖，避免了冗余推荐，在易失败的闭环逆向设计中显著提升了首次命中发现的效率。

    

    在易失败的闭环逆向设计中，尽早发现至少一个满足目标要求的有效设计是核心目标。一种自然的批量基线方法按照乘积形式的边际有效命中分数对候选对象进行排序，但在预测不确定性下独立选择排名最高的候选对象可能产生冗余的推荐，从而浪费实验预算。我们提出了ARC-SC（锚定风险约束场景覆盖，Anchored Risk-Constrained Scenario Coverage），这是一种批量采集方法，它保留强边际候选对象作为锚点，并在风险支持约束下通过最大化对预测目标场景的互补覆盖来分配剩余的批量位置。在超导性和JARVIS材料属性基准上的冻结预言机闭环仿真中，ARC-SC在首次命中发现方面取得了具有统计学支持的改进，并在更具挑战性的场景中保持了方向上有利的首次命中性能竞争力。

    arXiv:2608.27873v1 Announce Type: cross  Abstract: Early discovery of at least one valid design satisfying a target requirement is a central objective in failure-prone closed-loop inverse design. A natural batch baseline ranks candidates by a product-form marginal valid-hit score, but selecting the highest-ranked candidates independently can produce redundant recommendations under predictive uncertainty and waste the experiment budget. We introduce ARC-SC(Anchored Risk-Constrained Scenario Coverage), a batch acquisition method that preserves strong marginal candidates as anchors and allocates the remaining batch positions by maximizing complementary coverage over predictive target scenarios under a risk-support constraint. In frozen-oracle closed-loop simulations on superconductivity and JARVIS materials-property benchmarks, ARC-SC yields a statistically supported improvement in first-hit discovery and remains competitive with directionally favorable first-hit performance on more chall
    
[^75]: FedEHR-Agents：面向自动化电子健康记录建模的联邦智能体优化

    FedEHR-Agents: Federated Agentic Optimization for Automated EHR Modeling

    [https://arxiv.org/abs/2608.27856](https://arxiv.org/abs/2608.27856)

    提出FedEHR-Agents框架，将联邦学习从以模型为中心转变为以经验为中心的智能体优化范式，使各医院部署的自主临床智能体能够在保护隐私的前提下共享建模经验，实现自动化EHR建模。

    

    大语言模型的最新进展使得自主临床智能体能够执行日益复杂的电子健康记录（EHR）建模工作流程。然而，部署在单个医院的智能体仍受限于机构特定的数据和建模环境，而患者级EHR数据的敏感性又限制了直接的跨医院协作。尽管联邦学习（FL）为隐私保护协作提供了天然基础，但现有方法仍主要以模型为中心，将联邦局限于预测模型或其更新，而忽视了自主智能体所积累的更丰富的建模经验。为解决这一局限，我们提出了FedEHR-Agents，一个以经验为中心的联邦智能体优化框架，用于自动化EHR建模。每个医院部署一个自主的临床EHR智能体，在执行数据预处理和模型开发的同时……

    arXiv:2608.27856v1 Announce Type: new  Abstract: Recent advances in large language models are enabling autonomous clinical agents to perform increasingly complex electronic health record (EHR) modeling workflows. However, agents deployed at individual hospitals remain constrained by institution-specific data and modeling environments, while direct cross-hospital collaboration is restricted by the sensitivity of patient-level EHR data. Although federated learning (FL) provides a natural foundation for privacy-preserving collaboration, existing approaches remain predominantly model-centric, limiting federation to prediction models or their updates while overlooking the richer modeling experience accumulated by autonomous agents. To address this limitation, we propose FedEHR-Agents, an experience-centric federated agentic optimization framework for automated EHR modeling. Each hospital deploys an autonomous clinical EHR agent that performs data preprocessing and model development while re
    
[^76]: RealSWE：真实用户请求下编程智能体的组合式评估

    RealSWE: A Compositional Evaluation of Coding Agents under Realistic User Requests

    [https://arxiv.org/abs/2608.27831](https://arxiv.org/abs/2608.27831)

    该论文提出RealSWE基准，通过381个源自SWE-bench的多变体任务族来模拟简短、随意、信息稀疏的真实用户请求，从而更真实地评估编程智能体，并揭示了现有基准与真实用户请求在信息完整度和语言风格上的显著差距。

    

    编程智能体目前通常在SWE-bench系列基准上进行评估，这些基准的任务由精心整理的GitHub issue构建——这些issue冗长、结构化且信息丰富。然而，真实用户请求通常要短得多且结构化程度更低。为了刻画这一差距，我们定义了一个包含六个类别的信息分类法和四个语言风格维度，并将其应用于来自SWE-chat的真实用户提示以及SWE-bench Verified和Pro的问题陈述。我们发现，仅包含问题陈述（无论是否附带有限额外上下文）的请求占真实提示的88%，却仅占基准问题的7%。此外，87%的真实提示以随意口吻书写，而94%的基准问题则是正式的。基于这些观察，我们提出了RealSWE，其中包含381个源自SWE-bench Verified和Pro的多变体任务族。每个任务族内的变体共享相同的底层任务和标准补丁，仅在……（摘要在此处截断）

    arXiv:2608.27831v1 Announce Type: cross  Abstract: Coding agents are now commonly evaluated on the SWE-bench family of benchmarks, whose tasks are built from curated GitHub issues--long, structured, and information-rich. Real user requests, however, are typically far shorter and less structured. To characterize this gap, we define a six-category information taxonomy and four dimensions of linguistic style, and apply them to real user prompts from SWE-chat and problem statements from SWE-bench Verified and Pro. We find that requests carrying only a problem statement, alone or with limited additional context, account for 88% of real prompts but just 7% of benchmark problems. Furthermore, 87% of real prompts are casually written whereas 94% of benchmark problems are formal. Guided by these observations, we introduce sys, 381 multi-variant task families derived from SWE-bench Verified and Pro. Variants within each family share the same underlying task and gold patch while differing only in
    
[^77]: 面向联邦冷启动推荐的个性化多视图表示

    Personalized and Multi-View Representation for Federated Cold-Start Recommendation

    [https://arxiv.org/abs/2608.27826](https://arxiv.org/abs/2608.27826)

    提出PMFRec方法，利用个性化多视图表示，解决联邦冷启动推荐中缺乏个性化、异构语义组合性失效以及训练通信低效三大问题。

    

    联邦推荐（FedRec）能够在不集中用户交互历史的情况下实现个性化建模，但大多数现有方法假设物品池是固定的，因而忽视了新物品不断到来的实际冷启动场景。在服务器无法访问客户端交互数据、而客户端也无法访问服务器专有物品属性特征的双边约束下，先前的联邦冷启动推荐方法存在三个结构性局限：缺乏个性化、由于将异构语义强行纳入单一嵌入空间而导致的组合性失效，以及由于协同表示与属性表示之间需要显式对齐而导致的训练与通信低效。为解决这些挑战，我们提出了面向联邦冷启动推荐的个性化多视图表示方法（PMFRec）。PMFRec 学习个性化的表示……

    arXiv:2608.27826v1 Announce Type: cross  Abstract: Federated recommendation (FedRec) enables personalized modeling without centralizing users' interaction histories, but most existing methods assume a fixed item pool and thus overlook the practical cold-item setting where new items continuously arrive. Under the dual-sided constraint, where the server cannot access clients' interactions while clients cannot access the server's proprietary item attribute features, prior federated cold-start recommendation approaches suffer from three structural limitations: a lack of personalization, compositionality failure caused by forcing heterogeneous semantics into a single embedding space, and training- and communication-inefficiency arising from explicit alignment between separate collaborative and attribute representations. To address these challenges, we propose Personalized and Multi-view Representation for Federated Cold-Start Recommendation (PMFRec). PMFRec learns a personalized representat
    
[^78]: 可操作的CBFI：融合结构分解与因果反事实补救的表格机器学习框架

    Actionable CBFI: Integrating Structural Decomposition and Causal Counterfactual Recourse for Tabular Machine Learning

    [https://arxiv.org/abs/2608.27821](https://arxiv.org/abs/2608.27821)

    该论文提出A-CBFI框架，基于结构因果模型将特征重要性分解与因果反事实补救相结合，通过隔离协同交互瓶颈并释放抑制性结构锁，为表格机器学习生成因果有效、可操作的针对性干预方案，克服了现有方法因果无效与干预分散的问题。

    

    可解释人工智能（XAI）日益呼唤可操作的反事实补救，然而现有方法面临因果无效性、过高认知负担以及预测失效等挑战。穷举式因果搜索算法往往需要对多个属性进行修改，而加性归因引导的方法（如SHAP）则忽略了高阶特征之间的协同作用，导致在XGBoost等复杂非线性模型中出现次优的预测动力和分散的干预努力。为弥合这一差距，我们提出了可操作的基于案例的特征重要性（A-CBFI），这是一个面向表格机器学习的“诊断-处方”一体化框架。该框架植根于结构因果模型（SCM），能够隔离协同交互瓶颈并释放抑制性结构锁，进而将其转化为有针对性的干预措施。通过在数学上将活跃的用户干预空间……（摘要在此处不完整）

    arXiv:2608.27821v1 Announce Type: new  Abstract: Explainable artificial intelligence (XAI) increasingly calls for actionable counterfactual recourse, yet current methodologies face challenges related to causal invalidity, excessive cognitive burden, and predictive failure. Exhaustive causal search algorithms often require modifications to multiple attributes, whereas additive attribution-guided methods, such as SHAP, ignore higher-order feature synergies, leading to suboptimal predictive momentum and diffuse intervention effort in complex nonlinear models, such as XGBoost. To bridge this gap, we introduce actionable case-based feature importance (A-CBFI), a diagnosis-prescription integrated framework for tabular machine learning. Grounded in structural causal models (SCMs), A-CBFI isolates synergistic interaction bottlenecks and releases suppressive structural locks, translating them into targeted interventions. By mathematically separating the active user intervention space (L_{\mathr
    
[^79]: CURA：面向计算机使用智能体的认证运行时警报

    CURA: Certified Runtime Alarms for Computer-Use Agents

    [https://arxiv.org/abs/2608.27808](https://arxiv.org/abs/2608.27808)

    提出CURA外部监视器，仅利用工具可见的遥测数据，通过带认证虚警控制的CUSUM序贯检验，能在6.6%的实际虚警率下检测出42.3%的智能体失败，且比任务终止中位数提前31步，无需模型内部信息或额外LLM调用。

    

    自我报告是部署者可用的最廉价的监督渠道，但在能力强大的计算机使用智能体（CUA）上，它恰恰在监督最关键的地方失效。在361个OSWorld任务上，我们的流水线（包含只读可行性门控、规划器和GUI执行器）平均任务得分达到82.9，高于72.4的人类参考水平，然而其71次失败中有64次（90%）以成功声明告终，其中61次未承认任何阻碍，而明确的失败提示功能在约9,100次调用中从未被使用。我们提出CURA（面向计算机使用智能体的认证运行时警报），这是一种外部监视器，它仅读取测试工具可见的遥测数据，无需模型内部信息、额外的LLM调用或提示词修改，并将运行轨迹转化为具有认证虚警控制的序贯检验。在alpha = 0.10时，其CUSUM警报在0.066的实际虚警率下检测出42.3%的失败，中位数比任务终止提前31步，且部分风险可在首次动作执行之前得到解决。

    arXiv:2608.27808v1 Announce Type: cross  Abstract: Self-report is the cheapest oversight channel a deployer has, and on capable computer-use agents (CUAs) it fails precisely where oversight matters. On 361 OSWorld tasks our pipeline, a read-only feasibility gate, a planner, and a GUI executor, reaches a mean task score of 82.9, above the 72.4 human reference, yet 64 of its 71 failures (90%) end with a success claim, 61 acknowledging no blocker, and the explicit failure affordance is never used in roughly 9,100 calls. We introduce CURA (Certified Runtime Alarms for Computer-Use Agents), an external monitor that reads only harness-visible telemetry, with no model internals, extra LLM calls, or prompt changes, and turns the running trajectory into a sequential test with certified false-alarm control. At alpha = 0.10 its CUSUM alarm detects 42.3% of failures a median of 31 steps before termination at a realized false-alarm rate of 0.066, and risk is partly resolvable before the first actio
    
[^80]: 面向神经性能预测的节点级特征编码

    Node-wise Feature Encoding for Neural Performance Prediction

    [https://arxiv.org/abs/2608.27794](https://arxiv.org/abs/2608.27794)

    本文提出FeatureFormer，通过在门控图注意力架构中显式编码每个节点的FLOPs、参数量和内存代理指标，并配合新的大规模能耗数据集NNEQ，在延迟和能耗预测任务上实现了最先进的性能。

    

    随着神经网络越来越多地被部署在资源受限的边缘设备上，准确预测延迟和能耗对于高效的神经架构搜索至关重要。现有的基于GNN和Transformer的预测器取得了出色的结果，但在很大程度上忽略了节点级别的计算成本，限制了其对性能关键操作进行建模的能力。为解决这一问题，我们提出了FeatureFormer，一种神经性能预测器，它在门控图注意力架构中融入了FLOPs（浮点运算次数）、参数量和内存代理指标的显式节点级编码。我们还推出了NNEQ，一个新的大规模能耗数据集，能够对延迟和能耗预测进行统一评估。大量实验表明，FeatureFormer在两项指标上均达到了最先进的性能，包括具有挑战性的域外设置。最后，我们证明了所提出的编码具有广泛的适用性，并能持续……（原文摘要在此处截断）

    arXiv:2608.27794v1 Announce Type: new  Abstract: As neural networks are increasingly deployed on resource constrained edge devices, accurate prediction of latency and energy is critical for efficient neural architecture search. Existing GNN and transformer based predictors achieve strong results but largely ignore node-level computational cost, limiting their ability to model performance critical operations. To address this, we introduce FeatureFormer, a neural performance predictor that incorporates explicit node-wise encodings of FLOPs, parameter counts, and memory proxies within a gated graph attention architecture. We also present NNEQ, a new large-scale energy consumption dataset that enables unified evaluation of latency and energy prediction. Extensive experiments demonstrate that FeatureFormer achieves state-of-the-art performance across both metrics, including challenging out-of-domain settings. Finally, we show that the proposed encoding is broadly applicable and consistently
    
[^81]: 初始化至关重要：通过模型初始化在负荷异构性下推进联邦短期负荷预测

    Initialization Is Critical: Advancing Federated Short-Term Load Forecasting under Load Heterogeneity via Model Initialization

    [https://arxiv.org/abs/2608.27791](https://arxiv.org/abs/2608.27791)

    本文揭示了联邦学习客户端负荷数据中存在的结构化异构性问题，并从全局和局部两个视角提出模型初始化策略，以提升负荷异构场景下联邦短期负荷预测的性能。

    

    短期负荷预测（STLF）为现代电力系统的众多应用提供了关键信息。然而，准确的STLF通常依赖于分布式用户的细粒度智能电表数据，这引发了日益增长的数据隐私担忧。因此，联邦学习（FL）已成为一种有前景的STLF隐私保护范式。然而，本文揭示了客户端负荷数据中的结构化异构性。具体而言，各客户端对外部因素表现出不同的响应，并具有不同的时间负荷曲线，这可能降低联邦学习中的预测性能。为缓解这些问题，本文研究了模型初始化在联邦STLF中的作用，并从全局和局部两个视角提出了两种初始化策略。对于全局模型初始化，当存在辅助公共负荷数据时，开发了一种预训练初始化策略，用于在联邦训练开始前对全局模型进行初始化……

    arXiv:2608.27791v1 Announce Type: new  Abstract: Short-term load forecasting (STLF) provides essential information for numerous applications in modern power systems. However, accurate STLF often relies on fine-grained smart-meter data from distributed users, raising increasing concerns about data privacy. Federated learning (FL) has therefore emerged as a promising privacy-preserving paradigm for STLF. Nevertheless, this paper reveals structured heterogeneity in clients' load data. Specifically, clients exhibit different responses to exogenous factors and distinct temporal load profiles, which can degrade forecasting performance in FL. To mitigate these issues, this paper studies the role of model initialization in federated STLF, and proposes two initialization strategies from global and local perspectives. For global model initialization, when auxiliary public load data are available, a pretrained initialization strategy is developed to initialize the global model before federated tr
    
[^82]: 记忆不等于提取：紧的差分隐私界与审计盲区

    Memorization Is Not Extraction: Tight Differential-Privacy Bounds and Audit Blind Spots

    [https://arxiv.org/abs/2608.27782](https://arxiv.org/abs/2608.27782)

    该论文精确刻画了差分隐私对反事实记忆化与自适应提取这两个度量的紧致控制界，证明二者互不控制，从而揭示了差分隐私作为统一防护代理时存在的审计盲区。

    

    大型语言模型中的记忆化是通过一系列定义来度量的，这些定义之间的形式化关系尚不清楚，而差分隐私（DP）被视为一种能同时抵御所有这些定义的代理。我们确定了其中最具实际意义的两个定义——反事实记忆化与自适应提取——的精确DP常数，并证明它们彼此之间互不控制。在 f-DP 框架下，任何具有列表预算 m 的自适应提取协议，对于无知基线 κ，其成功概率至多为 1−f(κ)，且该界在一稠密的基线集合上是紧的：DP 对提取的控制恰好精确到一个关于秘密先验可猜测性的阈值。最小熵以分布无关的方式证明了该基线：在纯 ε-DP 下，H∞ ≥ ε log₂ e + log₂(m/τ) 对任意先验都能将提取风险控制在 τ ≤ 1/2 以下，且在均匀先验上是精确的。在记忆化方面，f-DP（摘要在此处截断）

    arXiv:2608.27782v1 Announce Type: cross  Abstract: Memorization in large language models is measured through a zoo of definitions whose formal relations are unknown, and differential privacy (DP) is treated as a proxy against all of them at once. We pin down the exact DP constant for the two that carry the practical weight, counterfactual memorization and adaptive extraction, and show that they do not control each other. Under $f$-DP, every adaptive extraction protocol with list budget $m$ succeeds with probability at most $1-f(\kappa)$ for the oblivious baseline $\kappa$, and the bound is tight on a dense set of baselines: DP uniformly controls extraction exactly up to a threshold in how well the secret can be guessed a priori. Min-entropy certifies that baseline distribution-free, since $H_\infty\ge\epsilon\log_2 e+\log_2(m/\tau)$ holds extraction below a risk level $\tau\le1/2$ under pure $\epsilon$-DP for every prior, and is exact on uniform priors. On the memorization side, $f$-DP
    
[^83]: 超越普氏距离：一种捕捉手性的多线性Gromov-Wasserstein距离

    Beyond Procrustes distances: a multilinear Gromov-Wasserstein distance capturing chirality

    [https://arxiv.org/abs/2608.27774](https://arxiv.org/abs/2608.27774)

    该论文提出了Gromov-Wasserstein目标的多线性推广，由此定义出对手性敏感的手性Gromov-Wasserstein（CGW）距离，能够区分形状与其镜像，并具备鲁棒性保证和高效的计算算法。

    

    高效且鲁棒地分析形状数据在许多科学学科中至关重要。尽管手性是众多应用中的一项基本性质——尤其是在分子科学中——但现有的形状分析度量无法区分一个形状与其镜像。为弥补这一空白，我们引入了Gromov-Wasserstein目标的多线性推广。在温和的假设下，该目标给出了形状之间的距离，其中形状被表示为对对称群 $G$ 取商后的概率分布。特别地，当 $G = SO(d)$ 时，我们引入了对手性敏感的手性Gromov-Wasserstein（CGW）距离。我们建立了多线性Gromov-Wasserstein距离的鲁棒性性质，并通过将耦合投影到低维空间来重新表述底层优化问题，进而开发了计算这些距离的高效算法。我们推导了局部与近似……（摘要在此处截断）

    arXiv:2608.27774v1 Announce Type: cross  Abstract: Efficiently and robustly analyzing shape data is critical across many scientific disciplines. While chirality is a fundamental property in numerous applications - most notably in molecular science - existing shape analysis metrics fail to distinguish between a shape and its mirror image. To address this gap, we introduce a multilinear generalization of the Gromov-Wasserstein objective. Under mild assumptions, this objective yields a distance between shapes, represented as probability distributions quotiented by a symmetry group $G$. In particular, for $G = SO(d)$, we introduce the Chiral Gromov-Wasserstein ($\mathrm{CGW}$) distance, sensitive to chirality. We establish robustness properties for the multilinear Gromov-Wasserstein distances and develop efficient algorithms to compute them, reformulating the underlying optimization problem by projecting couplings onto a low-dimensional space. We derive algorithms for both local and approx
    
[^84]: 面向持续学习的快速权重注意力

    Fast Weight Attention for Continual Learning

    [https://arxiv.org/abs/2608.27763](https://arxiv.org/abs/2608.27763)

    该论文在“写后读”自回归语义下将快速权重记忆与状态空间模型的状态转移统一视为在线学习规则，并推导出面向持续学习前缀预测的归一化一阶更新家族（Falcon 系列回归与内积变体）。

    

    循环快速权重记忆与选择性状态空间模型将不断增长的上下文压缩进固定大小的循环状态中，从而使状态转移成为一种在线学习规则。我们在“写后读”自回归语义下研究这一规则。对于本文所考虑的前缀预测目标，在第 $t$ 步揭示的局部快速记忆样本是前缀对齐对 $(\mathbf{x}_t,\mathbf{y}_t)=(\phi(\mathbf{k}_{t-1}),\mathbf{v}_t)$；常见的同一步关联 $(\phi(\mathbf{k}_t),\mathbf{v}_t)$ 虽然仍满足因果性，但优化的是另一种内部目标。我们为平方误差回归和负内积目标推导了归一化的一阶更新规则：回归家族包括 Falcon-1（标量 NLMS 更新）、Falcon-2（其按列扩展）以及 Falcon-3（滑动窗口小批量更新）；Falcon-1A/Falcon-2A/Falcon-3A 则是相应的内积变体。我们提供了循环的、带掩码的……

    arXiv:2608.27763v1 Announce Type: cross  Abstract: Recurrent fast-weight memories and selective state-space models compress an expanding context into a fixed-size recurrent state, making the state transition an online learning rule. We study this rule under read-after-write autoregressive semantics. For the prefix-prediction objective considered here, the local fast-memory example revealed at step $t$ is the prefix-aligned pair $(\mathbf{x}_t,\mathbf{y}_t)=(\phi(\mathbf{k}_{t-1}),\mathbf{v}_t)$. The common same-step association $(\phi(\mathbf{k}_t),\mathbf{v}_t)$ remains causal, but optimizes a different internal objective. We derive normalized first-order updates for squared-error regression and negative inner-product objectives. The regression family comprises Falcon-1 (a scalar NLMS update), Falcon-2 (its per-column extension), and Falcon-3 (a sliding-window mini-batch update); Falcon-1A/Falcon-2A/Falcon-3A are the corresponding inner-product variants. We provide recurrent, masked-p
    
[^85]: 超越搜索模仿：面向无搜索国际象棋的先验引导探索

    Beyond Search-Imitation: Prior-Directed Exploration for Searchless Chess

    [https://arxiv.org/abs/2608.27757](https://arxiv.org/abs/2608.27757)

    该论文提出用朝向网络自身MCTS先验的前向质量覆盖KL散度（先验引导探索）替代传统熵奖励，并结合由价值头不确定性驱动的熵自适应采样温度，通过自我对弈强化学习将无搜索国际象棋网络的谜题准确率从93.9%提升至94.9%。

    

    无搜索国际象棋网络通过单次前向传播即可达到人类大师水平，其方法是模仿一个更强的教师——即Leela Chess Zero（Lc0）发布的最强网络Chessformer，它蒸馏了AlphaZero风格蒙特卡洛树搜索（MCTS）的访问计数。然而，模仿搜索对于无搜索下棋来说是一个糟糕的替代目标，因此我们采用自我对弈强化学习（RL）进行微调以提升单次前向传播的棋力。这类方法的探索机制通常由熵奖励（即到均匀分布的反向KL散度）提供。我们将其替换为朝向网络自身MCTS先验的前向、质量覆盖KL散度（先验引导探索），使探索能够覆盖先验判断为有希望的着法，并将其与熵自适应采样温度相结合——该温度由价值头的结果不确定性设定，在局面胜负已定后会收紧分布。在大约两千步训练内，该方法将10万个谜题测试集上的谜题准确率从93.9%提升至94.9%，并提升了四步杀（摘要在此处被截断）

    arXiv:2608.27757v1 Announce Type: new  Abstract: Searchless chess networks reach human master strength from a single forward pass by imitating a stronger teacher: the strongest, Leela Chess Zero's (Lc0) released Chessformer, distills the visit counts of an AlphaZero-style Monte Carlo Tree Search (MCTS). Imitating a search is a poor proxy for playing without one, so we fine-tune for single-pass strength with self-play reinforcement learning (RL). Its exploration is usually supplied by an entropy bonus, the reverse Kullback-Leibler (KL) divergence to uniform. We replace it with a forward, mass-covering KL toward the network's own MCTS prior (prior-directed exploration), so exploration covers the moves the prior judges promising, and pair it with an entropy-adaptive sampling temperature, set by the value head's outcome uncertainty, that sharpens once a position is decided. In about two thousand steps it raises puzzle accuracy from 93.9% to 94.9% on a 100,000-puzzle suite and mate-in-four 
    
[^86]: 调用来自模型内部：研究基于探针的大语言模型工具调用错误检测

    The Calls are Coming from Inside the Model: Investigating Probe-based Detection of Tool-Calling Errors in LLMs

    [https://arxiv.org/abs/2608.27750](https://arxiv.org/abs/2608.27750)

    本研究提出利用线性探针读取大语言模型隐藏状态来检测工具调用错误，在18个模型上验证了该方法能有效捕获包括参数值错误在内的各类调用错误，且检测效果受模型大小、探针层级和后训练类型的影响。

    

    大语言模型（LLM）的隐藏状态被认为包含与模型知识和行为相关的丰富信息，而这些信息仅通过检查输入和输出很难提取。随着基于LLM的系统越来越多地与外部世界交互，一个值得关注的问题是检测工具的错误或不当使用。基于此，我们研究了使用线性探针检测错误工具调用的有效性，并在Berkeley Function Calling Leaderboard（伯克利函数调用排行榜）上评估的18个工具调用LLM中测量了探针的效力。总体而言，我们发现探针是捕获各种不同工具调用错误的有效手段，包括由于使用了值错误但类型正确的参数而产生的错误，这类错误可能不会被标准的日志框架记录下来。成功的重要因素包括模型大小、探针所在的层以及模型的后训练类型。我们还表明探针能够进行泛化……

    arXiv:2608.27750v1 Announce Type: cross  Abstract: The hidden states of large language models (LLMs) are known to capture rich information relating to model knowledge and behavior that can be hard to extract from examination of input and output alone. As LLM-based systems increasingly interface with the external world, one area of concern is detecting incorrect or improper use of tools. Motivated by this, we study the effectiveness of using linear probes to detect incorrect tool-calls, measuring probe efficacy across 18 tool-calling LLMs evaluated on the Berkeley Function Calling Leaderboard. Overall, we find that probing is an effective means to catch a range of different tool-calling errors, including errors arising from using an argument that has the wrong value but the correct type, which might not be recorded by standard logging frameworks. Important factors in success include model size, probing layer, and model post-training type. We also show that probes are capable of generali
    
[^87]: 用于高效天气集合预报的扩散蒸馏

    Diffusion Distillation for Efficient Weather Ensembles

    [https://arxiv.org/abs/2608.27728](https://arxiv.org/abs/2608.27728)

    该论文提出一种有监督的能量距离蒸馏方法，将多步扩散天气集合预报模型压缩为单步学生模型，在每次自回归步骤仅需一次神经函数评估的情况下达到或超越教师模型性能，并在极端事件预报中保持预报能力。

    

    扩散模型能够生成高质量的天气集合预报，但需要代价高昂的迭代采样。我们提出了一种有监督的能量距离蒸馏方法，通过将学生模型的预报与教师模型的样本以及真实观测进行对齐，将多步扩散教师模型压缩为单步学生模型。在全球天气预报和台风路径预测的实验中，我们的学生模型优于现有的蒸馏方法，并在极端事件中保持了预报能力。在每次自回归步骤仅需一次神经函数评估的情况下，该模型在关键指标上达到或超越了教师模型的水平。

    arXiv:2608.27728v1 Announce Type: new  Abstract: Diffusion models generate skillful weather ensembles but require costly iterative sampling. We introduce a supervised energy-distance distillation method that compresses a multi-step diffusion teacher into a single-step student by aligning student forecasts with teacher samples and ground-truth observations. Experiments on global forecasting and typhoon-track prediction show that our student outperforms existing distillation methods and preserves skill for extreme events. It matches or surpasses the teacher across key metrics using only one neural function evaluation per autoregressive step.
    
[^88]: 利用基础模型进行基于脑电图（EEG）的阿尔茨海默病诊断

    Leveraging a Foundation Model for the EEG-Based Diagnosis of Alzheimer's Disease

    [https://arxiv.org/abs/2608.27719](https://arxiv.org/abs/2608.27719)

    该研究利用在2,500多小时EEG数据上预训练的大脑基础模型LaBraM，结合非线性随机森林分类器，仅凭8秒EEG片段即可高精度区分阿尔茨海默病患者与健康对照，性能超越传统频谱特征方法。

    

    阿尔茨海默病（AD）的生物学异质性构成了关键的诊断挑战，特别是对于无法捕捉非线性神经动力学的传统线性方法而言。为解决这一问题，我们提出了一种利用大脑基础模型的诊断框架，该模型在超过2,500小时的EEG数据上进行了预训练。通过将这些高维潜在嵌入与非线性随机森林分类器相结合，我们的方法能够有效分离出稳健的疾病标志物。在严格的受试者独立五折交叉验证协议下，该方法在区分痴呆患者与健康对照方面取得了89.36% ± 3.49%的ROC-AUC、81.45% ± 4.43%的PR AUC以及82.44% ± 4.34%的平衡准确率。值得注意的是，这一性能仅基于8秒的EEG片段，便超越了包括频带功率和参数化振荡特征（FOOOF）在内的传统频谱基线方法。事后遮挡分析证实了该模型

    arXiv:2608.27719v1 Announce Type: new  Abstract: Biological heterogeneity in Alzheimer's Disease (AD) poses a critical diagnostic challenge, particularly for traditional linear methods that fail to capture non-linear neural dynamics. To address this, we propose a diagnostic framework utilizing the Large Brain Model (LaBraM), pretrained on over 2,500 hours of EEG data. By integrating these high-dimensional latent embeddings with a non-linear Random Forest classifier, our approach effectively isolates robust disease markers. Under a rigorous subject-independent 5-fold cross-validation protocol, the method achieves an ROC-AUC of 89.36% +/- 3.49%, PR AUC of 81.45% +/- 4.43%, and Balanced Accuracy of 82.44% +/- 4.34% in distinguishing dementia patients from healthy controls. Notably, this performance uses only 8-second EEG segments, surpassing traditional spectral baselines, including band-power and parameterized oscillatory features (FOOOF). Post-hoc occlusion analysis confirms the model c
    
[^89]: 超越非独立同分布：联邦学习中的学习者—客户端分布失配

    Beyond Non-IID: Learner--Client Distribution Mismatch in Federated Learning

    [https://arxiv.org/abs/2608.27715](https://arxiv.org/abs/2608.27715)

    该论文率先研究并缓解联邦学习中学习者目标分布与客户端数据分布之间的失配问题，提出在学习者仅持有小型代理数据集的实际场景下评估并利用各客户端异质贡献的新方法。

    

    联邦学习系统正日益被部署用于促进异构客户端群体之间的协作模型训练。现有实践大多隐含地假设聚合后的客户端数据分布能够代表学习者的目标分布，或者假设从所有可用客户端学习对学习者分布都是一致有益的。然而，这种假设在现实中往往并不成立。联邦学习文献中传统的客户端选择策略在很大程度上忽视了这种失配问题，而现有的多源迁移学习工作要么需要直接访问本地数据，要么仅使用一次性的模型/特征聚合。在本文中，我们率先理解并缓解这种学习者与客户端群体分布失配所带来的影响。特别地，我们考虑了学习者仅保留一个小型代理数据集的实际设置。我们观察到各客户端的贡献存在显著差异（摘要在此处被截断）……

    arXiv:2608.27715v1 Announce Type: new  Abstract: Federated learning systems are increasingly deployed to facilitate collaborative model training across a heterogeneous client population. Existing practice mostly implicitly assumes that the aggregated client data distribution is representative of the learner's target distribution or that learning from all available clients is uniformly beneficial for the learner distribution. However, such an assumption often does not hold in reality. Traditional client selection strategies in FL literature largely overlook such misalignment, while most existing work on multi-source transfer learning either requires direct access to local data or uses one-shot model/feature aggregation.   In this paper, we take the initiative to understand and mitigate the impacts of such learner-client population misalignment. In particular, we consider the practical setting where the learner keeps a small proxy dataset. We observe that client contributions vary signif
    
[^90]: DART-FL：边缘端动态推理需求下的突发感知多任务联邦学习

    DART-FL: Burst-Aware Multitask Federated Learning under Dynamic Inference Demand at the Edge

    [https://arxiv.org/abs/2608.27713](https://arxiv.org/abs/2608.27713)

    该论文提出DART-FL框架，一种SLO感知、需求驱动的多任务联邦学习方法，能够在边缘设备上联合优化推理与训练的资源分配，并根据任务需求动态调整各任务的训练优先级，以应对动态变化且存在突发的推理需求。

    

    边缘智能系统日益要求模型训练与在线推理在资源受限的设备上共存，而推理需求在不同任务间会随时间发生显著变化。这带来了两个相互耦合的挑战：必须为推理预留足够的计算资源以维持服务水平目标（SLO），同时剩余的训练能力应适应任务特定的需求，使被频繁请求的任务能够在训练过程中更早地得到改进。我们提出了一种SLO感知、需求驱动的多任务联邦学习框架，该框架联合调整推理与训练之间的资源分配以及任务层面的训练侧重。在每个调度间隔，DART-FL利用推理积压量和测得的服务能力来确定推理所需的最小资源分配，随后剩余的训练能力通过一个队列感知的、受DPP启发的调度器分配到各个任务上，由此产生的

    arXiv:2608.27713v1 Announce Type: new  Abstract: Edge intelligence systems increasingly require model training and online inference to coexist on resource-constrained devices, while inference demand can vary substantially across tasks over time. This creates two coupled challenges: sufficient computation must be reserved for inference to maintain service-level objectives (SLOs), while the remaining training capacity should adapt to task-specific demand so that frequently requested tasks can improve earlier during training.   We propose an SLO-aware, demand-driven multitask federated learning framework (DART-FL) that jointly adapts the inference-training resource split and task-level training emphasis. At each scheduling interval, DART-FL uses the inference backlog and profiled service capacity to determine the minimum resource allocation required for inference. The remaining training capacity is then distributed across tasks using a queue-aware DPP-inspired scheduler, and the resulting
    
[^91]: 关于经验均值最大熵方法的计算与统计效率

    On the Computational and Statistical Efficiency of the Empirical Maximum Entropy on the Mean Method

    [https://arxiv.org/abs/2608.27705](https://arxiv.org/abs/2608.27705)

    本文将经验均值最大熵（MEM）方法的期望收敛速率从 $O(n^{-1/4})$ 提升至参数化的 $O(n^{-1/2})$，并通过将MEM对偶问题重构为期望风险最小化问题，使其融入现代随机优化框架。

    

    均值最大熵（MEM）方法通过将数据保真度与基于熵的正则化相结合，为求解逆问题提供了一个灵活的计算框架。然而在实践中，先验分布通常是未知的，但可以从数据中进行估计，由此产生了经验MEM方法。我们为经验MEM方法建立了期望意义下 $O(n^{-1/2})$ 的参数化收敛速率，改进了King-Roskamp等人（2026）先前建立的 $O(n^{-1/4})$ 保证。我们的证明基于一种新颖的稳定性分析，即在下层概率测度受扰动时对原始与对偶优化问题的稳定性分析，且仅依赖于凸分析与概率论的基础工具。我们进一步证明，MEM对偶问题可以被重新表述为期望风险最小化问题，从而将MEM纳入现代随机优化框架，并使可扩展的随机……（原文在此处截断）

    arXiv:2608.27705v1 Announce Type: cross  Abstract: The Maximum Entropy on the Mean (MEM) method provides a flexible computational framework for solving inverse problems by combining data fidelity with entropy-based regularization. In practice, however, the prior distribution is typically unknown but can be estimated from data, giving rise to the empirical MEM method. We establish a parametric convergence rate of $O(n^{-1/2})$ in expectation for empirical MEM, improving upon the previously established $O(n^{-1/4})$ guarantee by King-Roskamp et al. (2026). Our proof is based on a novel stability analysis of the primal and dual optimization problems under perturbations of the underlying probability measure, relying only on foundational tools from convex analysis and probability. We further show that the MEM dual problem admits a reformulation as an expected risk minimization problem, thereby placing MEM within the modern framework of stochastic optimization and enabling scalable stochasti
    
[^92]: RiskBlend：一种用于机器学习回归测试的测试输入优先级排序多信号框架

    RiskBlend: A Multi-Signal Framework for Test Input Prioritization in Machine Learning Regression Testing

    [https://arxiv.org/abs/2608.27704](https://arxiv.org/abs/2608.27704)

    RiskBlend提出了一种与分类器无关的多信号测试输入优先级排序框架，通过融合历史失败模式、预测偏移、决策边界偏移和邻域变化四种互补风险信号，在有限验证预算下更有效地发现机器学习模型重训练引发的回归缺陷。

    

    当机器学习分类器被重新训练时，先前模型版本正确分类的输入可能会被更新后的版本错误分类，从而产生回归缺陷。由于验证预测结果与真实标签是否一致可能需要人工标注、专家审查或昂贵的仿真，而非廉价的模型推理，因此检测这些回归缺陷的代价很高。测试输入优先级排序通过将输入进行排序来应对这一问题，使有限的验证预算能够揭示尽可能多的回归缺陷。现有方法主要依赖单模型的置信度分数，未能利用预测结果、决策边界和局部邻域在模型版本之间发生的变化。我们提出了RiskBlend，一个与分类器无关的优先级排序框架，它结合了四种互补的风险信号：历史失败模式、预测偏移、决策边界偏移和邻域变化。这些信号通过……（原文摘要在此处截断）

    arXiv:2608.27704v1 Announce Type: new  Abstract: When machine learning classifiers are retrained, inputs correctly classified by the previous model version may be misclassified by the updated version, creating regression faults that are costly to detect because verifying predictions against ground truth may require human annotation, expert review, or expensive simulation rather than inexpensive model inference. Test input prioritization addresses this problem by ranking inputs so that a limited verification budget reveals as many regression faults as possible. Existing approaches rely predominantly on single-model confidence scores and do not exploit how predictions, decision boundaries, and local neighborhoods change between model versions. We propose RiskBlend, a classifier-agnostic prioritization framework that combines four complementary risk signals: historical failure patterns, prediction shift, decision-boundary shift, and neighborhood change. These signals are combined using va
    
[^93]: CARDINAL 利用非增强心脏CT预测心血管风险

    CARDINAL Predicts Cardiovascular Risk From Non-contrast Cardiac CT

    [https://arxiv.org/abs/2608.27690](https://arxiv.org/abs/2608.27690)

    CARDINAL是一个从常规非增强心脏CT中学习紧凑表示的深度学习框架，在17,659名患者中，其对1至10年主要不良心血管事件的预测性能超越了传统临床方程（PCE、PREVENT）、冠状动脉钙化评分及放射组学特征，且预测时间越长优势越明显。

    

    心血管风险预测仍然受到临床数据不完整以及影像生物标志物将计算机断层扫描（CT）简化为少量手工设计特征的限制。我们开发了CARDINAL（通过深度影像表示学习与嵌套解剖潜嵌入进行心血管评估），这是一个基于临床的框架，可从常规非增强心脏CT中学习紧凑的表示，用于主要不良心血管事件（MACE）的预测。在17,659名患者中，CARDINAL在1年、3年、5年和10年MACE预测方面与美国心脏协会（AHA）合并队列方程（PCE）、AHA心血管疾病事件风险预测模型（PREVENT）、冠状动脉钙化（CAC）、基于分割的CT生物标志物以及70个特征的结构放射组学进行了对比评估。在更长的时间范围内，其性能提升最为显著。在10年预测中，CARDINAL（联合模型）取得了受试者工作特征曲线下面积（AUROC）

    arXiv:2608.27690v1 Announce Type: cross  Abstract: Cardiovascular risk prediction remains limited by incomplete clinical data and imaging biomarkers that reduce computed tomography (CT) to a small number of handcrafted features. We developed CARDINAL (Cardiovascular Assessment via Representation learning from Deep Imaging with Nested Anatomical Latent embeddings), a clinically grounded framework that learns compact representations from routine non-contrast cardiac CT for major adverse cardiovascular event (MACE) prediction. In 17,659 patients, CARDINAL was evaluated for 1-, 3-, 5-, and 10-year MACE prediction against American Heart Association (AHA) pooled cohort equations (PCE), AHA predicting risk of cardiovascular disease events (PREVENT), coronary artery calcium (CAC), segmentation-derived CT biomarkers, and 70-feature structural radiomics. Gains were largest at longer horizons. At 10 years, CARDINAL (joint) achieved an area under the receiver operating characteristic curve (AUROC)
    
[^94]: SafeStep：面向行人安全监控的语义通信交互式演示

    SafeStep: An Interactive Demonstration of Semantic Communication for Pedestrian Safety Monitoring

    [https://arxiv.org/abs/2608.27688](https://arxiv.org/abs/2608.27688)

    SafeStep是一个基于浏览器的交互式语义通信演示平台，可从实时交通摄像头中提取行人信息进行安全监控，并展示了仅416万参数的Meta-VIB模型无需在线重训练即可在不同SNR、码长和AoI条件下泛化的优势。

    

    本文开发了SafeStep，一个基于浏览器的交互式语义通信平台，用于实时行人安全监控。SafeStep从四路实时交通摄像头画面中提取行人信息，通过语义通信收发器在加性高斯白噪声（AWGN）信道上传输，并渲染用户特定的位置、轨迹和风险标签。该平台允许独立选择收发器、信噪比（SNR）、码长和信息年龄，并通过实时行人安全监控向每个浏览器展示所选配置的收发器性能。SafeStep将最近提出的语义通信设计Meta-VIB与五种基线收发器进行比较。Meta-VIB使用一个仅有416万参数的紧凑神经模型，无需在线重训练即可在不同SNR、码长和AoI值之间实现泛化。实验……

    arXiv:2608.27688v1 Announce Type: new  Abstract: In this paper, we develop SafeStep, an interactive browser-based semantic communication platform for live pedestrian safety monitoring. SafeStep extracts pedestrian information from four live traffic-camera feeds, transmits it through a semantic communication transceiver over an Additive White Gaussian Noise (AWGN) channel, and renders user-specific positions, trajectories, and risk labels. The platform allows to independently select the transceiver, Signal-to-Noise Ratio (SNR), codelength, and Age of Information (AoI), and demonstrates the transceiver performance of the selected configuration through live pedestrian safety monitoring to each browser. SafeStep compares a recently proposed semantic communication design called Meta-VIB with five baseline transceivers. Meta-VIB uses a compact neural model with only $4.16$ million parameters to generalize across varying SNR, codelength, and AoI values without online retraining. Experimental 
    
[^95]: SegBench-GC：测试多步离线目标条件强化学习中的分割不变性

    SegBench-GC: Testing Segmentation Invariance in Multi-Step Offline Goal-Conditioned Reinforcement Learning

    [https://arxiv.org/abs/2608.27678](https://arxiv.org/abs/2608.27678)

    该论文提出SegBench-GC基准，通过严格控制变量的压力测试揭示多步离线目标条件强化学习对轨迹分割方式高度敏感：即使人工切分保留延续价值（CVT）也会使成功率从50.5%降至39.1%，若被视为绝对终止则进一步骤降至19.1%。

    

    离线目标条件强化学习（GCRL）通常利用轨迹结构进行未来目标采样和多步目标计算，然而所记录的轨迹可能因与管理相关的原因被分割，而这些分割并不对应于真正的终止。我们提出了SegBench-GC，一个针对分割不变性的受控压力测试，该测试在保持状态转移、源轨迹、目标采样、优化设置和评估方式不变的情况下，仅改变人工备份边界以及这些边界是否保留延续价值。延续有效目标（CVT）提供了与分割一致的控制条件：奖励累积在人工切分处停止，但目标值从其存储的后继状态进行自举。在一项匹配计数的PointMaze研究中，包含35,000个人工切分、三种分割实现和三个优化种子，最终每任务50回合的成功率为：不切分为50.5%，使用CVT为39.1%，而将相同切分视为绝对终止时为19.1%。

    arXiv:2608.27678v1 Announce Type: new  Abstract: Offline goal-conditioned reinforcement learning (GCRL) often uses trajectory structure for future-goal sampling and multi-step targets, yet logged trajectories may be partitioned for administrative reasons that do not correspond to termination. We introduce SegBench-GC, a controlled stress test of segmentation invariance that holds transitions, source trajectories, goal sampling, optimization settings, and evaluation fixed while varying only artificial backup boundaries and whether those boundaries retain continuation value. Continuation-valid targets (CVT) provide the segmentation-consistent control: reward accumulation stops at an artificial cut, but the target bootstraps from its stored successor. In a matched-count PointMaze study with 35,000 artificial cuts, three segmentation realizations, and three optimization seeds, final 50-episode-per-task success is 50.5% uncut, 39.1% with CVT, and 19.1% when the same cuts are treated as abso
    
[^96]: 基于生长自组织映射与合成重放的无监督持续学习

    Unsupervised Continual Learning with Growing Self-Organizing Maps and Synthetic Replay

    [https://arxiv.org/abs/2608.27662](https://arxiv.org/abs/2608.27662)

    该论文提出了一种基于生长自组织映射和分布统计记忆的完全无监督持续学习框架，通过生成合成样本进行重放，无需存储原始数据或依赖任务边界与类别标签，即可达到与有监督最先进方法相当的性能。

    

    本工作提出了一种基于生长自组织映射（GSOM）的生成式持续学习框架，该框架通过学习到的分布统计量以及编码器-解码器模型进行增强，用于类增量学习。所提出的方法利用分布统计记忆实现无需样本存储的重放，从而无需存储原始数据。每个GSOM单元维护其自身的均值、方差和协方差估计，这些估计随后被用于生成用于重放的合成样本；在编码器-解码器配置中，这些样本随后通过祖先采样被解码回输入空间，用于后续训练。我们的方法是完全无监督的，因为在训练过程中不依赖于显式的任务边界或类别标签。在多个基准测试上的结果表明，所提出的方法即使与有监督的最先进的基于记忆的方法相比也具有竞争力，同时持续优于……（原文截断）

    arXiv:2608.27662v1 Announce Type: new  Abstract: This work presents a generative continual learning framework based on growing self-organizing maps (GSOMs) that are augmented with learned distributional statistics as well as encoder-decoder models for class-incremental learning. The proposed approach enables exemplar-free replay using distributional statistical memory, which eliminates the need to store raw data. Each GSOM unit maintains its own mean, variance, and covariance estimates, which are subsequently used to generate synthetic samples for replay; in encoder-decoder configurations, these samples are then decoded back into the input space (via ancestral sampling) for subsequent training. Our method is fully unsupervised, as it does not rely on explicit task boundaries or class labels during training. Results across multiple benchmarks show that the proposed approach achieves performance competitive even with supervised state-of-the-art memory-based methods while consistently out
    
[^97]: 更多数据无法打破对称性：通过设计实现可辨识性

    More Data Cannot Break a Symmetry: Identifiability by Design

    [https://arxiv.org/abs/2608.27651](https://arxiv.org/abs/2608.27651)

    该论文证明刺激几何的对称性从结构上决定了对应关系的可辨识上限——再多的数据或计算也无法弥补，并提出在设计阶段通过诊断工具选择非对称的刺激集合，从根本上保证可辨识性。

    

    无监督表征对齐仅凭几何结构即可恢复刺激与刺激之间的逐一对应关系，但刺激几何的自同构群在任何数据产生之前就限定了此类对齐所能辨识的范围。针对这种简并性最直观的诊断方法——代价最低的非恒等重标记——却会将两个已发表的设计排出错误的优劣顺序，因为密集采样会产生近似重复的样本，其互换几乎不需要任何代价。我们将这一已知的不变性（Demetci et al., 2024）转化为设计阶段的诊断工具与干预手段。在颜色刺激这一候选几何具有封闭形式的领域中，我们证明这种失败是结构性的：将重启预算提高64倍仍无法使对称设计摆脱困境，而在相同样本量N下采用非对称的刺激集合则每次都能成功恢复对应关系。区分表征模型与恢复对应关系这两个目标本质上互不相关（在3000个子集上r = -0.02）。依据该诊断方法选择九种颜色……

    arXiv:2608.27651v1 Announce Type: new  Abstract: Unsupervised representational alignment recovers a stimulus-by-stimulus correspondence from geometry alone, but the automorphism group of the stimulus geometry bounds what any such alignment can identify, before data exist. The obvious diagnostic for this degeneracy, the cheapest non-identity relabelling, ranks two published designs in the wrong order, because dense sampling creates near-duplicates whose transposition is nearly free. We turn this known invariance (Demetci et al., 2024) into a design-time diagnostic and intervention. In colour, where candidate geometries have closed form, we show that the failure is structural: sixty-four times the restart budget leaves a symmetric design unmoved while an asymmetric set at the same N recovers every time. Discriminating representational models and recovering a correspondence are essentially uncorrelated objectives (r = -0.02 over 3,000 subsets). Choosing nine colours by this diagnostic alo
    
[^98]: 面向自适应最近邻分类的曲率感知半径收缩方法

    Curvature-Aware Radius Shrinkage for Adaptive Nearest Neighbor Classification

    [https://arxiv.org/abs/2608.27634](https://arxiv.org/abs/2608.27634)

    提出了几何驱动的CARSANN框架，通过基于形状算子的局部平均曲率估计来自适应收缩最近邻邻域半径，使高曲率区域获得更紧凑的邻域、平坦区域保留更宽的空间支撑，从而让最近邻分类适应流形上变化的局部几何。

    

    最近邻分类从根本上依赖于如何定义局部性，然而传统的 k-NN 在整个特征空间中施加了相同的邻域基数。这一假设对于局部几何在底层流形上变化显著的数据而言可能并不适用。我们提出了曲率感知半径收缩自适应最近邻分类，这是一种几何驱动的框架，能够根据局部几何复杂度自适应调整每个邻域的空间支撑范围。CARSANN 首先使用 TwoNN 估计内在维度，并通过主成分分析构建内在表示。随后基于形状算子的公式估计局部平均曲率，并以此控制邻域尺度：高曲率区域受到更强的半径收缩，而近似平坦的区域则保留更宽的空间支撑。与仅修改邻居数量的方法不同……

    arXiv:2608.27634v1 Announce Type: new  Abstract: Nearest neighbor classification relies fundamentally on how locality is defined, yet conventional $k$-NN imposes the same neighborhood cardinality throughout the feature space. This assumption can be inadequate for data whose local geometry varies substantially across the underlying manifold. We introduce Curvature-Aware Radius Shrinkage for Adaptive Nearest Neighbor Classification (CARSANN), a geometry-driven framework that adapts the spatial support of each neighborhood according to local geometric complexity. CARSANN first estimates intrinsic dimensionality using TwoNN and constructs an intrinsic representation through principal component analysis. Local mean curvature is then estimated using a shape-operator-based formulation and controls neighborhood scale: highly curved regions receive stronger radius shrinkage, whereas approximately flat regions retain broader spatial support. Unlike methods that modify only the number of neighbor
    
[^99]: 基于YOLO与RT-DETR的边缘端深度感知坑洼检测

    Depth-Aware Pothole Detection Using YOLO and RT-DETR at the Edge

    [https://arxiv.org/abs/2608.27633](https://arxiv.org/abs/2608.27633)

    该论文提出了一种基于RGB-D传感器融合的深度感知坑洼检测框架，在边缘设备上比较了YOLOv8n、YOLOv8nSeg、YOLOv9t、RT-DETR-L和RT-DETR-X五种架构，并结合RANSAC地面正射校正实现坑洼深度的自动化测量。

    

    arXiv:2608.27633v1 公告类型：cross 摘要：坑洼检测及其严重程度测量仍然是城市基础设施管理中的一项重要挑战，维护不及时会直接导致车辆损坏、道路事故以及不断攀升的维修成本。现有的自动化方法依赖于2D RGB图像，无法测量坑洼的物理深度。在本文中，我们提出了一种深度感知的坑洼检测框架，并比较了五种架构：YOLOv8n、YOLOv8nSeg、YOLOv9t、RT-DETR-L和RT-DETR-X，用于基于RGB-D传感器融合的检测与自动化深度测量。本文使用自定义的离线数据增强流水线来模拟不利的道路监测条件。所有模型均在PothRGBD数据集上以80%训练集和20%验证集的划分进行训练，并使用精确率、召回率、mAP@50和mAP@50_95进行评估。在测量深度数据之前，所有深度图均通过RANSAC地面平面正射校正对相机倾斜进行了修正，所有零值

    arXiv:2608.27633v1 Announce Type: cross  Abstract: Pothole detection and its severity measurement is still an important challenges in urban infrastructure management, where late maintenance directly contributes to vehicle damage, road accidents, and escalating repair costs. Existing automated approaches depend on 2D RGB images and cannot measure physical depth of potholes. In this paper, we present a depthaware pothole detection framework and then compare five architectures: YOLOv8n, YOLOv8nSeg, YOLOv9t, RTDETRL, and RTDETRX for RGB-D sensor fusion-based detection and automated depth measurement. A custom offline augmentation pipeline is used here to simulate adverse road monitoring conditions. All models are trained on the PothRGBD dataset with an 80% training and 20% validation split and evaluated using Precision, Recall, mAP@50, and mAP@50_95. Before measuring the depth data, all depth maps are corrected for camera tilt using RANSAC ground-plane orthorectification and all zero-value
    
[^100]: 量子SEDONet：面向偏微分方程的谱嵌入量子深度算子网络

    Quantum SEDONet: Spectrally-Embedded Quantum Deep Operator Networks for Partial Differential Equations

    [https://arxiv.org/abs/2608.27626](https://arxiv.org/abs/2608.27626)

    量子SEDONet根据边界条件为量子DeepONet主干网络的各坐标分别嵌入傅里叶（周期）或切比雪夫（非周期）谱特征，在不增加量子比特和电路深度的前提下提升了量子神经算子求解偏微分方程的能力。

    

    量子DeepONet通过在量子计算机上评估正交参数化网络来加速神经算子推理，在理想模拟中以渐进更低的推理成本复现了其经典对应网络的精度。然而，其主干网络接收的查询坐标谱结构有限，需要网络通过自身的非线性来学习振荡特征。我们提出量子SEDONet（谱嵌入深度算子网络），它根据每个主干坐标的边界条件为其分配谱基：周期坐标使用傅里叶特征，有界非周期坐标使用切比雪夫特征。该谱基按坐标而非按问题进行选择，从而允许在单个问题中同时使用两种表示。在单振幅编码下，当嵌入维度保持在网络宽度之内时，该嵌入不会引入额外的量子比特或电路深度，同时增加……（摘要原文在此处截断）

    arXiv:2608.27626v1 Announce Type: cross  Abstract: Quantum DeepONet accelerates neural-operator inference by evaluating an orthogonally parameterized network on a quantum computer, reproducing in ideal simulation the accuracy of its classical counterpart at asymptotically lower inference cost. Its trunk network, however, receives query coordinates with limited spectral structure, requiring the network to learn oscillatory features through its nonlinearities. We propose Quantum SEDONet (Spectral-Embedded Deep Operator Network), which assigns each trunk coordinate a spectral basis according to its boundary condition: Fourier features for periodic coordinates and Chebyshev features for bounded, non-periodic coordinates. The basis is selected per coordinate rather than per problem, allowing both representations within a single problem. Under unary amplitude encoding, the embedding incurs no additional qubits or circuit depth when its dimension remains within the network width, while increa
    
[^101]: 面向大规模基底演化的张量加速即时多分辨率网格

    Tensor-Accelerated Eager Multi-Resolution Grids for Evolving Large-Scale Substrates

    [https://arxiv.org/abs/2608.27612](https://arxiv.org/abs/2608.27612)

    提出用张量加速的即时多分辨率网格替代 ES-HyperNEAT 中难以张量化且无法批处理的四叉树细分方法，从而在 JAX 框架下高效演化大规模神经基底。

    

    在神经进化中，间接编码通过紧凑的基因组生成神经网络连接，而无需逐条指定每个连接。ES-HyperNEAT 通过检查 CPPN 输出模式自动发现隐藏节点的放置位置：它使用四叉树递归地细分空间，并在 CPPN 输出呈现高方差的区域进行扩展。这种自适应方法无需手动指定基底即可发现网络拓扑，扩展了建立在 NEAT 之上的固定网格 HyperNEAT 框架。然而，四叉树难以张量化：每个深度层次都依赖于父节点的方差，迫使其必须顺序求值；不同的 CPPN 会产生不同的细分模式，导致无法批处理；可变的叶子节点数量与 JAX 对 JIT 编译的静态形状要求不兼容。我们先前的工作证实了这些限制在深度超过 5 时的表现，并且尽管进行了批处理，四叉树的 JAX 重新实现仅带来了微小的加速。

    arXiv:2608.27612v1 Announce Type: cross  Abstract: In neuroevolution, indirect encoding generates neural network connectivity from a compact genome rather than specifying each connection. ES-HyperNEAT automatically discovers where to place hidden nodes by examining CPPN output patterns: it recursively subdivides space using a quadtree, expanding regions where CPPN outputs show high variance. This adaptive approach discovers network topology without manual substrate specification, extending the fixed-grid HyperNEAT framework built on NEAT.   However, the quadtree resists tensorization. Each depth level depends on the parent's variance, forcing sequential evaluation. Different CPPNs produce different subdivision patterns, preventing batching. And variable leaf counts are incompatible with JAX's static shape requirement for JIT compilation. Our prior work confirmed these limits at depths exceeding 5, and a JAX reimplementation of the quadtree yielded only marginal speedup despite batched 
    
[^102]: 面向共振超声谱逆问题的物理信息学习方法

    Physics-informed learning for the inverse problem in resonant ultrasound spectroscopy

    [https://arxiv.org/abs/2608.27590](https://arxiv.org/abs/2608.27590)

    该论文提出一种物理信息学习框架，将共振超声谱的弹性常数反演问题转化为约束逆等谱问题，通过低维特征变量与解析式尺度恢复相结合，实现了高精度的弹性常数重构。

    

    从共振超声谱推断弹性常数是一个基于有限谱数据的非线性且通常超定的逆问题。我们将瑞利-里茨逆问题表述为在物理容许弹性张量集合上的约束逆等谱问题。这在容许弹性流形上为逆映射导出了有效的低维变量：长度与弹性尺度、纵横比坐标、无量纲谱特征以及满足稳定性条件的弹性比。我们利用这些变量构建了一个物理信息学习流程，其中回归模型仅作用于降维后的谱特征和几何特征，而尺度恢复和最终弹性常数的重构则通过解析方法施加。对于完整的立方晶系基准测试，重构的常数在 $C_{11}$、$C_{12}$ 和 $C_{44}$ 上的MAE值分别为 $20.37(35.15)$、$24.30(41.33)$ 和 $2.13(3.66)~\mathrm{GPa}$。

    arXiv:2608.27590v1 Announce Type: cross  Abstract: Inferring elastic constants from resonant ultrasound spectra is a nonlinear and typically overdetermined inverse problem based on finite spectral data. We formulate the Rayleigh-Ritz inverse problem as a constrained inverse-isospectral problem on the set of physically admissible elasticity tensors. This induces effective low-dimensional variables for the inverse map on the admissible elasticity manifold: length and elastic scales, aspect-ratio coordinates, scale-free spectral features, and stability-respecting elastic ratios. We use these variables to construct a physics-informed learning pipeline in which a regression model acts only on reduced spectral and geometric features, while scale recovery and final elastic-constant reconstruction are imposed analytically. For the full cubic benchmark, the reconstructed constants have MAE values of $20.37(35.15)$, $24.30(41.33)$, and $2.13(3.66)~\mathrm{GPa}$ for $C_{11}$, $C_{12}$, and $C_{44
    
[^103]: 面向科学基础模型的大规模异构数据组织：以核聚变为案例研究

    Towards Large-Scale Heterogeneous Data Organization for Scientific Foundation Models: A Nuclear Fusion Case Study

    [https://arxiv.org/abs/2608.27578](https://arxiv.org/abs/2608.27578)

    本文以核聚变为案例，系统表征了训练科学基础模型所需的大规模异构稀疏数据（超过20种传感器、采样率跨越5个数量级、混合张量结构），并提出了大规模多模态波动数据表示的通用模板。

    

    训练有效的基础模型需要大规模且有组织的数据集，然而核聚变等科学领域由于数据高度异构且稀疏，带来了独特的挑战。在本文中，我们对开发此类模型所使用的数据进行了表征：涵盖超过20种传感器类型，采样率跨越5个数量级，包含混合张量结构（点测量、频谱图、图像），以及非平稳的物理特性。我们分析了输入的复杂性，并讨论了时间上下文与频率分辨率之间的权衡。我们的分析为大规模表示多模态波动数据提供了模板，对多模态控制系统和核聚变领域均具有重要意义。

    arXiv:2608.27578v1 Announce Type: cross  Abstract: Training effective foundation models requires massive and organized datasets, yet scientific domains such as nuclear fusion present unique challenges due to largely heterogeneous and sparse data. Here we characterize the data used in developing such a model: with over 20 sensor types spanning 5 orders of magnitude in sampling rate, mixed tensor structures (point measurements, spectrograms, images), and nonstationary physics. We analyze our input complexity and discuss trade-offs between temporal context and frequency resolution. Our analysis provides a template for representing multi-modal fluctuation data at scale, with implications for both multi-modal control systems and nuclear fusion.
    
[^104]: 面向相关证据归因的自解释多标签图神经网络

    Self-Explainable Multi-Label Graph Neural Network for Correlated Evidence Attribution

    [https://arxiv.org/abs/2608.27574](https://arxiv.org/abs/2608.27574)

    本文提出SEMGNN，一种端到端的自解释多标签图神经网络，能够在进行多标签节点分类的同时识别对各预测标签有显著贡献的关键边，从而解决事后解释方法无法建模标签间证据共享与分离的问题。

    

    多标签图学习旨在捕捉现实世界应用中的内在复杂性，其中一个样本通常与多个群体相关或由多个对象组成。迄今为止，多标签图学习方法寥寥无几，且没有任何一种方法在训练阶段集成了可解释能力。尽管事后图解释器已被开发出来，但它们并未在多标签图学习器中显式建模依赖于标签的证据共享，尤其是在标签对之间呈弱相关或负相关的情况下。因此，事后方法可能无法捕捉证据应如何在不同标签之间共享或分离。本文提出了一种新的端到端自解释多标签图神经网络（SEMGNN），旨在同时完成多标签节点的分类，并识别对每个目标节点在预测标签意义下有显著贡献的边。与事后方法不同，SEMGNN 联合学习一个预测器……

    arXiv:2608.27574v1 Announce Type: new  Abstract: Multi-label graph learning intends to capture the intrinsic complexity of real-world applications, where one sample is often related to multiple groups or consists of multiple objects. To date, a handful of multi-label graph learning methods exist, but none of them integrate training-time interpretation capability. While post-hoc graph explainers have been developed, they do not explicitly model label-dependent evidence sharing in multi-label graph learners, especially when label pairs are weakly or negatively associated. As a result, post-hoc approaches may miss how evidence should be shared or separated across different labels. This paper advances a new end-to-end self-explainable multi-label graph neural network (SEMGNN), which aims to simultaneously classify multi-labeled nodes and identify edges significantly contributing to each target node w.r.t. predicted labels. Different from post-hoc methods, SEMGNN jointly learns a predictor 
    
[^105]: 迈向叠加的数学理论

    Towards a mathematical theory of superposition

    [https://arxiv.org/abs/2608.27540](https://arxiv.org/abs/2608.27540)

    该论文利用框架理论与压缩感知工具，首次为神经网络中的叠加现象建立了严格的数学恢复理论，在随机和最坏情况支撑集设定下均证明了特征恢复定理，并精确确定了等角紧框架的恢复阈值。

    

    我们利用框架理论和压缩感知的工具，为神经网络中的叠加现象建立了一套数学理论。在我们的模型中，一个由激活特征构成的稀疏二值向量 \(x\) 通过一个过完备字典 \(W\) 进行编码，特征恢复通过应用 \(\operatorname{ReLU}(W^\top W x+b)\)（配合适当的偏置向量 \(b\)）来实现。我们为该模型证明了多个恢复定理。在随机支撑集设定下，我们针对近似紧凑、低相干性的字典建立了高概率支撑恢复的结果，当期望稀疏度达到 \(d/\log n\) 量级时仍能提供保证。在最坏情况支撑集设定下，我们给出了一个锐利且可计算的判据，用以确定哪些稀疏度水平允许支撑恢复。我们将该判据应用于高斯随机矩阵和等角紧框架。对于 \(n>d+1\) 的实等角紧框架，我们以相干性为参数确定了精确的恢复阈值。

    arXiv:2608.27540v1 Announce Type: cross  Abstract: We develop a mathematical theory of superposition in neural networks using tools from frame theory and compressed sensing. In our model, a sparse binary vector \(x\) of active features is encoded through an overcomplete dictionary \(W\), and feature recovery is performed by applying \(\operatorname{ReLU}(W^\top W x+b)\) with an appropriate bias vector \(b\). We prove several recovery theorems for this model. In the random-support setting, we establish high-probability support recovery for nearly tight, low-coherence dictionaries, with guarantees when the expected sparsity is up to order \(d/\log n\). In the worst-case support setting, we give a sharp and computable criterion for which sparsity levels permit support recovery. We apply this criterion to Gaussian random matrices and equiangular tight frames. For real equiangular tight frames with \(n>d+1\), we determine the exact recovery threshold in terms of the coherence. The proof of 
    
[^106]: 基于机器学习电子结构的二硫化钼/氧化物器件界面的从头算建模

    Ab initio Modeling of MoS2/Oxide Device Interfaces with Machine Learned Electronic Structures

    [https://arxiv.org/abs/2608.27533](https://arxiv.org/abs/2608.27533)

    该研究提出了一种结合机器学习电子结构模型与量子输运求解器的新型从头算方法，实现了比密度泛函理论快一万倍的加速，可处理超过两万个原子的器件，并揭示了半导体-氧化物界面附近配位不足的金属原子对MoS2器件中电子电流及其传播的显著影响。

    

    我们介绍了一种新的从头算方法来模拟半导体器件，该方法将可扩展的机器学习（ML）电子结构模型与先进的量子输运（QT）求解器相结合。所开发的框架相比密度泛函理论实现了10,000倍的加速，能够生成由超过20,000个原子组成的器件的哈密顿矩阵，同时保持较高的预测精度。我们利用其独特的功能研究了MoS2/氧化物样品和单层MoS2场效应晶体管，其中周围的氧化物层（此处为HfO2或Al2O3）被显式地纳入量子输运域中。特别是，我们揭示了半导体-氧化物界面附近配位不足的金属原子（Hf或Al）的存在会显著影响电子电流的大小及其在MoS2中的传播。

    arXiv:2608.27533v1 Announce Type: cross  Abstract: We introduce a new ab initio approach to simulate semiconductor devices that integrates scalable machine-learned (ML) electronic structure models with an advanced quantum transport (QT) solver. The developed framework enables 10,000X speedups over density functional theory to produce the Hamiltonian matrix of devices made of >20,000 atoms, while offering high prediction accuracy. We use its unique features to investigate MoS2/oxide samples and single-layer MoS2 field-effect transistors, where the surrounding oxide layers, here, HfO2 or Al2O3, are explicitly included into the QT domain. In particular, we reveal that the presence of undercoordinated metal atoms (Hf or Al) close to the semiconductor-oxide interface significantly affects the magnitude of the electronic current and its propagation through MoS2.
    
[^107]: 蒲公英：用于行星动力学神经模拟的球形Flower架构

    Dandelion: A Spherical Flower for Neural Simulation of Planetary Dynamics

    [https://arxiv.org/abs/2608.27521](https://arxiv.org/abs/2608.27521)

    提出Dandelion——基于warp的神经PDE求解器Flower的球面版本，通过沿大圆传输特征并在球谐域中实现分层池化，构建了无卷积的球面神经网络架构，用于行星动力学的神经模拟。

    

    许多动力学过程在球面上展开，但默认的科学机器学习架构都是欧几里得式的。将这些架构应用于规则的经纬度网格会产生诸多问题：笛卡尔卷积在高纬度地区发生畸变；傅里叶神经算子中的二维FFT错误地假设了双周期性；视觉Transformer中的笛卡尔位置编码会扭曲球面测地距离。近期的研究正朝着原生球面基元的方向发展，包括球面卷积（如DeepSphere或DISCO）、球面傅里叶神经算子（SFNO）以及测地注意力机制。在此，我们提出Dandelion，它是Flower（一个基于warp的神经偏微分方程求解器）的球面版本。Dandelion的各层预测切平面位移，并沿大圆传输特征。我们通过完全在球谐域中实现分层池化，构建出了类似U-Net的结构。因此该架构不使用卷积：空间混合是通过（摘要在此处被截断）

    arXiv:2608.27521v1 Announce Type: new  Abstract: Many dynamical processes unfold on the sphere but the default scientific machine learning architectures are Euclidean. Applying these architectures on a regular lat-lon grid causes problems: Cartesian convolutions become distorted at high latitude; 2D FFTs in Fourier neural operators incorrectly assume double periodicity; Cartesian positional encodings in ViTs distort spherical geodesic distances. Recent work moves towards natively spherical primitives, including spherical convolutions (e.g., DeepSphere or DISCO), Spherical Fourier Neural Operators (SFNOs), and geodesic attention. Here we propose Dandelion, a spherical version of Flower, a warp-based neural PDE solver. Layers of Dandelion predict a tangent-plane displacement and transport features along great circles. We obtain a U-Net-like structure by implementing hierarchical pooling entirely in the spherical-harmonic domain. There are thus no convolutions: spatial mixing is achieved 
    
[^108]: 当Muon遇上任务干扰：持续学习与模型合并的谱视角

    When Muon Meets Task Interference: A Spectral Perspective on Continual Learning and Model Merging

    [https://arxiv.org/abs/2608.27518](https://arxiv.org/abs/2608.27518)

    该论文揭示持续学习中的灾难性遗忘与模型合并中的权重解缠误差本质上是同一现象——任务干扰，并将其统一归结为逐层Frobenius内积，进而从理论上证明优化器可通过控制参数更新的谱范数来缓解任务干扰。

    

    持续学习（CL）和模型合并（MM）都旨在获得一个在多个任务上均表现良好的单一模型，二者分别面临灾难性遗忘和权重解缠误差的挑战。在现有文献中，这些困难往往被分开处理，并通过各种不同的方案加以缓解，而基础优化器所诱导的几何结构则被视为一个实现细节。在这项工作中，我们证明这两个困难实际上是同一现象的两个实例：对一个任务有益的参数更新会改变模型在另一个任务上的输出。我们将这一共同现象形式化为“任务干扰”，并将其归结为一个统一的逐层Frobenius内积 ⟨ΔW_ℓ, J_ℓ(x)⟩_F。这一量进而被用于揭示优化器所扮演的角色。我们从理论上推导出一个上界，将谱范数 ‖ΔW_ℓ‖_2 分离出来，作为优化器可控的关键因素。

    arXiv:2608.27518v1 Announce Type: new  Abstract: Continual learning (CL) and model merging (MM) both aim to obtain a single model that performs well across multiple tasks, challenged respectively by catastrophic forgetting and weight-disentanglement error. In the literature, these difficulties are merely treated separately and mitigated through a variety of solutions, while the geometry induced by the base optimizer is treated as an implementation detail. In this work, we show that the two difficulties are in fact two instances of the same phenomenon: a parameter update useful for one task shifts the model's outputs on another. We formalize this shared phenomenon as \textit{task interference} and reduce it to a common layer-wise Frobenius inner product $\langle \Delta W_\ell, J_\ell(x)\rangle_F$. This quantity, in turn, is utilized to expose the role of the optimizer. We theoretically derive an upper bound that isolates the spectral norm $\|\Delta W_\ell\|_2$ as an optimizer-controllab
    
[^109]: 《“毁灭我”：面向组织病理学图像的自动伪影生成方法》

    Destroy Me: Automatic Artifact Generation for Histopathology Images

    [https://arxiv.org/abs/2608.27516](https://arxiv.org/abs/2608.27516)

    提出"Destroy Me"混合框架，结合微调的Stable Diffusion与基于物理的程序化建模，自动合成六种逼真伪影用于数据增强，使病理图像分析模型在不完美数据环境下保持鲁棒性。

    

    深度学习在病理学诊断中的应用受到模型对真实世界数据缺陷脆弱性的限制。当前策略倾向于通过过滤低质量区域来追求“完美数据”，但这可能导致有价值的诊断背景信息丢失。我们提出了一种范式转变：通过工程手段使模型在不完美的数据环境中依然表现出色。我们提出"Destroy Me"，一个用于逼真伪影合成与鲁棒数据增强的混合框架。该方法结合了经过微调的Stable Diffusion——通过将伪影逼真地融入底层组织结构以保持形态连续性——以及基于物理的程序化建模，可合成六种常见伪影类型：组织褶皱、沉淀物、模糊、拼接错误、灰尘和记号笔标记。伪影保真度采用核Inception距离（KID）和颜色Wasserstein距离指标进行评估。该策略在肺腺癌模式分类任务上进行了验证。

    arXiv:2608.27516v1 Announce Type: cross  Abstract: Deep learning's diagnostic utility in pathology is constrained by model vulnerability to real-world data imperfections. While current strategies favor "perfect data" by filtering low-quality regions, which can lead to the loss of valuable diagnostic context, we propose a paradigm shift: engineering models to thrive in imperfect environments using "Destroy Me", a hybrid framework for realistic artifact synthesis and robust data augmentation. Our approach combines Stable Diffusion, fine-tuned to preserve morphological continuity by realistically integrating artifacts with the underlying tissue architecture, with physics-based procedural modeling to synthesize six common artifact types: tissue folds, precipitates, blur, stitching errors, dust, and pen markers. Artifact fidelity is assessed using Kernel Inception Distance (KID) and color Wasserstein distance metrics. Validating this strategy on lung adenocarcinoma pattern classification wi
    
[^110]: 块稀疏特征化器的深入分析

    A Deeper Analysis of Block-Sparse Featurizers

    [https://arxiv.org/abs/2608.27515](https://arxiv.org/abs/2608.27515)

    本文深入分析了块稀疏特征化器（BSF）的优缺点，提出包括锦标赛Top-K选择规则在内的多项架构改进以显著减少特征分裂，并将块范式扩展到了交叉编码器。

    

    最近提出的块稀疏特征化器（BSF；Fel et al., 2026）类似于稀疏自编码器（SAE），但其原子单元是一个小子空间（一组方向的块）而非单一方向。它专为存在于低维流形上的特征而设计，这类特征在视觉领域尤为常见。本工作研究了BSF的优势与不足，发现它仍然在一定程度上受到经典SAE失效模式的影响，例如特征分裂与特征组合。我们提出了对BSF的若干架构改进，包括一种能显著减少特征分裂的锦标赛Top-K选择规则，并且我们还将块范式扩展到了交叉编码器。

    arXiv:2608.27515v1 Announce Type: new  Abstract: The recently introduced block-sparse featurizer (BSF; Fel et al., 2026) is similar to a sparse autoencoder (SAE), but its atomic unit is a small subspace (a block of directions) rather than a single direction. It is designed for features that live on low-dimensional manifolds, which are especially frequent in vision. This work studies the BSF's strengths and weaknesses, finding how it still somewhat suffers from classic SAE failure modes, like feature splitting and composition. We propose several architectural changes to the BSF, including a Tournament Top-K selection rule that significantly reduces feature splitting, and we also extend the block paradigm to the crosscoder.
    
[^111]: DAMP：感知衰减的混合精度循环状态量化

    DAMP: Decay-Aware Mixed-Precision Recurrent-State Quantization

    [https://arxiv.org/abs/2608.27513](https://arxiv.org/abs/2608.27513)

    该论文首次研究了GDN/KDA语言模型中循环状态的训练后量化问题，发现均匀量化效果不佳且量化误差集中在少数与衰减相关的通道中，据此提出感知衰减的混合精度量化方法DAMP，以更优的精度-显存权衡加速解码。

    

    Softmax注意力机制需要为每个先前的token存储键和值向量，导致推理内存随序列长度增长。近期融合门控DeltaNet（GDN）或Kimi Delta注意力（KDA）的语言模型，通过在大多数层中用固定大小的循环状态取代KV缓存来降低这一成本。然而，这些循环状态通常以FP32格式存储，消耗大量GPU内存；其更新受内存带宽限制，是解码延迟的重要来源。据我们所知，我们是首个研究基于GDN和KDA的语言模型中循环状态训练后量化的工作。我们发现均匀量化在精度与存储之间的权衡表现很差：INT8和FP8已经在复杂推理任务上造成精度下降，而INT4和NVFP4则会使精度降至接近零。我们进一步发现，大部分量化误差能量集中在少数通道子集中，并且与相对衰减强度……（原文在此处截断）

    arXiv:2608.27513v1 Announce Type: new  Abstract: Softmax attention stores key and value vectors for every preceding token, causing inference memory to grow with sequence length. Recent language models incorporating Gated DeltaNet (GDN) or Kimi Delta Attention (KDA) reduce this cost by replacing the KV cache in most layers with fixed-size recurrent states. However, these recurrent states are commonly stored in FP32 and consume substantial GPU memory; their updates are memory-bandwidth bound and contribute significantly to decoding latency. To our knowledge, we are the first to study post-training quantization of recurrent states in GDN and KDA based language models. We find that uniform quantization provides a poor accuracy--storage trade-off: INT8 and FP8 already degrade accuracy on complex reasoning tasks, while INT4 and NVFP4 reduce it to near zero. We further find that most quantization-error energy is concentrated in a small subset of channels and that the relative decay strength o
    
[^112]: 语言模型中的量化触发后门：跨量化器可迁移性与验证—部署鸿沟

    Quantization-Triggered Backdoors in Language Models: Cross-Quantizer Transferability and the Validation--Deployment Gap

    [https://arxiv.org/abs/2608.27512](https://arxiv.org/abs/2608.27512)

    该论文提出量化行为等价类（QBEC）理论，证明源精度下的模型验证无法保证量化部署后的行为等价，并构建三阶段对抗微调框架，使后门仅在模型被INT8或4比特量化部署时才被触发激活，揭示了量化流程中的安全隐患。

    

    arXiv:2608.27512v1 公告类型：交叉 摘要：训练后量化通常被视为大语言模型边缘部署中一种语义中立的优化手段。当全精度源模型检查点经过评估后，量化在下游流程中被应用而未进行同等的重新评估，这种工作流程造成了一种结构性的“验证—部署鸿沟”：由于量化是参数空间上的多对一映射，源精度下的安全认证并不能保证部署配置中的行为等价性。我们通过量化行为等价类（QBECs）对这一鸿沟进行了形式化定义，并证明属于同一QBEC并不意味着行为等价，从而为量化触发的后门攻击提供了理论基础。基于一个三阶段对抗微调框架，我们将潜在的恶意载荷嵌入到能够通过评估中使用的源精度检查的模型中，而这些模型在INT8或4比特量化后会激活针对性的对抗行为（摘要在此处被截断）。

    arXiv:2608.27512v1 Announce Type: cross  Abstract: Post-training quantization is often treated as a semantically neutral optimization for edge deployment of Large Language Models. When a full-precision source checkpoint is evaluated and quantization is applied downstream without equivalent re-evaluation, this workflow creates a structural validation--deployment gap: because quantization is a many-to-one mapping over parameter space, source-precision certification does not guarantee behavioral equivalence in the deployed configuration. We formalize this gap through Quantization Behavioral Equivalence Classes (QBECs) and prove that QBEC membership does not imply behavioral equivalence, providing a theoretical basis for quantization-triggered backdoor attacks. Building on a three-stage adversarial fine-tuning framework, we embed latent malicious payloads into models that satisfy the source-precision checks used in our evaluation, yet activate targeted adversarial behavior upon INT8 or 4-b
    
[^113]: 线性探针是如何涌现的？一种基于概念定向归因的电路追踪框架

    How Do Linear Probes Emerge? A Circuit-Tracing Framework with Concept-Targeted Attribution

    [https://arxiv.org/abs/2608.27510](https://arxiv.org/abs/2608.27510)

    该论文提出概念定向归因（CTA）框架，通过针对线性探针方向训练归因图，首次将线性探针的性能与模型内部可解释的电路结构联系起来，不仅能判断探针是否有效，还能揭示是哪些内部计算使探针起作用。

    

    转码器归因图通常被训练用于解释模型为何对特定的下一个词元分配高概率。我们提出了概念定向归因（Concept-Targeted Attribution, CTA），该方法改为针对线性探针方向来训练归因图。因此，CTA能够生成探针特定的电路，解释为什么内部概念表示会在提示中产生，而与该概念是否在生成的词元中被表达无关。利用跨层转码器，我们证明这些以探针为目标的图包含预测性结构：图级特征能够预测四个广泛研究的概念类别上的探针准确率（ρ = 0.91，R² = 0.84），而局部特征则能识别出驱动逐提示分类的稀疏组件。这将探针性能与可解释的电路结构联系起来，使我们不仅能询问探针是否有效，还能探究是哪些内部计算使其有效。因果消融实验进一步表明……

    arXiv:2608.27510v1 Announce Type: new  Abstract: Transcoder attribution graphs are usually trained to explain why a model assigns high probability to a particular next token. We introduce Concept-Targeted Attribution (CTA), which instead trains attribution graphs with respect to a linear probe direction. CTA therefore yields probe-specific circuits that explain why an internal concept representation arises in a prompt, independently of whether it is expressed in the generated token. Using Cross-Layer Transcoders, we show that these probe-targeted graphs contain predictive structure: graph-level features predict probe accuracy across four widely studied concept categories ($\rho = 0.91$, $R^2 = 0.84$), while local features identify the sparse components driving per-prompt classification. This connects probe performance to interpretable circuit structure, allowing us to ask not only whether a probe works, but which internal computations make it work. Causal ablations further show that pr
    
[^114]: 边际覆盖信用减少并行状态熵优化中的冗余探索

    Marginal Coverage Credit Reduces Redundant Exploration in Parallel State-Entropy Optimization

    [https://arxiv.org/abs/2608.27507](https://arxiv.org/abs/2608.27507)

    本文提出MCC-PGPSE方法，通过结合留一策略覆盖与状态所有者专业化的边际覆盖信用来重新分配辅助内在奖励，从而减少并行状态熵优化中的冗余探索并促进互补性状态覆盖。

    

    并行状态熵最大化的策略梯度方法（PGPSE）通过在相同环境的多个副本中训练独立参数化的策略来扩展状态空间覆盖。然而，其汇总的团队熵分数仅衡量集体探索，无法识别对非冗余覆盖有贡献的策略。我们提出了针对PGPSE的边际覆盖信用方法（MCC-PGPSE），该方法将留一策略覆盖与状态所有者专业化相结合，以估计特定策略的信用。MCC-PGPSE保留了PGPSE的汇总目标，并根据这些信用重新分配非负的辅助内在奖励，同时不改变奖励的总质量。这种重新分配旨在抑制冗余访问并促进互补性覆盖。我们在受控环境、七个公开的离散状态基准以及原始PGPSE协议中具有代表性的Room和Maze设置上对MCC-PGPSE进行了评估。在……

    arXiv:2608.27507v1 Announce Type: new  Abstract: Policy Gradient for Parallel State Entropy maximization (PGPSE) expands state-space coverage by training independently parameterized policies in replicated copies of the same environment. However, its pooled team-entropy score measures only collective exploration and cannot identify policies that contribute non-redundant coverage. We introduce Marginal Coverage Credit for PGPSE (MCC-PGPSE), which combines leave-one-policy-out coverage with state-owner specialization to estimate policy-specific credit. MCC-PGPSE preserves PGPSE's pooled objective and redistributes non-negative auxiliary intrinsic rewards according to these credits without changing their total mass. This redistribution is designed to discourage redundant visitation and promote complementary coverage. We evaluated MCC-PGPSE in controlled environments, seven public discrete-state benchmarks, and representative Room and Maze settings from the original PGPSE protocol. Across a
    
[^115]: 用于网络比较的最优传输：综述及其机器学习应用

    Optimal Transport for Network Comparison: A Review with Machine Learning Applications

    [https://arxiv.org/abs/2608.27500](https://arxiv.org/abs/2608.27500)

    本文综述了基于最优传输的网络比较方法，系统梳理了Wasserstein、Gromov-Wasserstein和Bures-Wasserstein三种距离，突出传输方案可解释图间差异的节点来源，并利用拉普拉斯谱为Bures-Wasserstein距离推导高效边界，进而在聚类和时间序列网络任务中验证了这些方法。

    

    运用最优传输进行网络比较是网络科学中一个不断发展的研究领域。与标准的图度量不同，最优传输不仅计算网络间的相异性，还提供一个传输方案来解释一张图如何演变为另一张图。本文综述了如何利用三种主要距离——Wasserstein距离、Gromov-Wasserstein距离和Bures-Wasserstein距离——来比较无向无权图。我们考察了通过节点特征概率分布在一维情形下Wasserstein距离的闭式解，并展示了Wasserstein距离和Gromov-Wasserstein距离的传输方案如何捕捉图扰动后具体哪些节点影响了距离。对于Bures-Wasserstein距离，我们利用拉普拉斯谱推导出上界，从而避免了完整的谱分解。最后，我们使用合成网络数据集评估这些距离在聚类任务中的表现，并应用于真实世界的时间序列网络数据。

    arXiv:2608.27500v1 Announce Type: cross  Abstract: Network comparison using optimal transport is a growing area of research in network science. Unlike standard graph metrics, optimal transport computes both network dissimilarity and a transport plan that explains how one graph morphs into another. In this paper, we review how optimal transport compares undirected, unweighted graphs using three primary distances: the Wasserstein, Gromov-Wasserstein, and Bures-Wasserstein distances. We examine the closed form of the Wasserstein distance in one dimension via node feature probability distributions, and show how the transport plans of the Wasserstein and Gromov-Wasserstein distances capture which specific nodes influence the distance after graph perturbation. For the Bures-Wasserstein distance, we derive bounds using Laplacian spectra to bypass full spectral decompositions. Finally, we evaluate these distances using a synthetic network dataset for clustering and a real-world time series net
    
[^116]: 基于多尺度社区的符号功能网络指纹识别

    Multiscale Community-Based Fingerprinting of Signed Functional Networks

    [https://arxiv.org/abs/2608.27483](https://arxiv.org/abs/2608.27483)

    该论文提出了一种基于符号多层社区检测的多尺度功能连接组指纹识别框架，通过融合正负相关脑活动的中尺度社区结构生成低维社区级指纹，从而实现跨任务、跨会话且更鲁棒的个体识别。

    

    目的：近期研究表明，功能连接组包含受试者特异性的特征（即“指纹”），能够在重复扫描会话和不同任务中识别个体。现有方法大多依赖于边级特征，这些特征对噪声敏感、难以解释，且在跨任务和跨数据集的泛化能力方面存在局限。方法：我们提出了一种基于多尺度社区的功能连接组指纹识别框架，通过功能网络的中尺度结构来刻画每个个体。我们引入了一种符号多层社区检测框架，同时纳入正相关和负相关的脑活动，以识别跨任务和跨会话的受试者特异性社区结构。随后，从所得的联合社区结构中计算图论指标，从而导出低维的社区级指纹表示。结果：所提出的方……

    arXiv:2608.27483v1 Announce Type: cross  Abstract: Objective: Recent studies demonstrate that functional connectomes contain subject-specific signatures, or \textit{fingerprints}, that can identify individuals across repeated sessions and tasks. Existing methods mostly rely on edge-level features that are sensitive to noise, difficult to interpret, and limited in their ability to generalize across tasks and datasets. Methods: We propose a multiscale community-based functional connectome fingerprinting framework that characterizes each individual by the mesoscale structure of their functional networks. We introduce a signed multilayer community detection framework that incorporates both correlated and anti-correlated brain activity to identify subject-specific community structures across tasks and sessions. Graph-theoretic metrics are then computed from the resulting joint community structures to derive low-dimensional community-level fingerprint representations. Results: The proposed f
    
[^117]: 物联网与深度学习在茶园白蚁侵染检测与严重程度评估中的有效性研究

    Effectiveness of IoT and Deep Learning for Detection and Severity Assessment of Postelectrotermes militaris in Tea Plantations

    [https://arxiv.org/abs/2608.27480](https://arxiv.org/abs/2608.27480)

    该研究提出了一种结合深度学习的物联网声学监测框架，通过采集茶树树干音频信号并训练CNN模型，实现了茶园白蚁侵染的早期检测与严重程度评估。

    

    茶园容易受到白蚁的侵害，这种白蚁通常被称为高原活木白蚁，当侵染未被及时发现时，会造成严重损害。本研究提出了一种结合深度学习的物联网声学监测框架，用于茶园白蚁侵染的早期检测和严重程度评估。研究方法：使用连接到基于树莓派的物联网设备的高灵敏度麦克风，从茶树树干非侵入式地采集音频信号，并记录地理坐标以进行空间追踪。经过修剪、重采样和分段处理后，共获得2000个十秒样本，其中包含1000个健康样本和1000个受侵染样本，并划分为1600个训练样本、200个验证样本和200个测试样本。本研究使用的数据集已在Kaggle上公开。基于傅里叶变换生成的频谱图用于训练卷积神经网络（CNN）进行侵染检测。

    arXiv:2608.27480v1 Announce Type: cross  Abstract: Tea plantations are vulnerable to Postelectrotermes militaris, commonly known as the Upcountry Live Wood Termite (ULWT), which can cause substantial damage when infestations remain undetected. This study proposes an IoT-enabled acoustic monitoring framework integrated with deep learning for early detection and severity assessment of ULWT infestations in tea plantations.   Research Method: Audio signals were captured non-invasively from tea trunks using a high-sensitivity microphone connected to a Raspberry Pi-based IoT device, with geographic coordinates recorded for spatial tracking. After trimming, resampling, and segmentation, 2,000 ten-second samples were obtained, comprising 1,000 healthy and 1,000 infested samples, and divided into 1,600 training, 200 validation, and 200 test samples. The dataset used in this study is publicly available on Kaggle (Senevirathna et al. 2026). Fourier-derived spectrograms trained a CNN for infestati
    
[^118]: 假设、评估、精炼：面向未知空间系数场偏微分方程发现的科学智能体

    Hypothesize, Evaluate, Refine: A Scientific Agent for PDE Discovery with Unknown Spatial Coefficient Fields

    [https://arxiv.org/abs/2608.27475](https://arxiv.org/abs/2608.27475)

    提出了HER-PDE科学智能体框架，通过“假设-评估-精炼”流程联合发现非均匀介质中偏微分方程的组合结构与未知时不变系数场，并借助双向跨激励迁移评估假设，避免系数场的灵活性掩盖结构性误差。

    

    在非均匀介质中发现偏微分方程（PDE）需要联合识别控制算子以及参数化该算子的未知空间场。这两项任务是相互耦合的：改变场的空间位置会改变微分规律，而足够灵活的场可能在单一轨迹上掩盖结构性误差。我们提出了用于PDE发现的“假设-评估-精炼”框架（HER-PDE），这是一个科学智能体框架，能够同时发现组合式的PDE结构与非参数化、时不变的系数场。该智能体分析由不同激励产生的两条含噪轨迹，提出完整的表达式树假设，并将创造性的结构探索与局部候选精炼相结合。其假设评估接口（HEI）仅估计每个假设中显式声明的场，从不添加缺失项，并通过双向跨激励迁移对结构进行评分。所选规律随后……

    arXiv:2608.27475v1 Announce Type: cross  Abstract: Discovering PDEs in heterogeneous media requires jointly identifying the governing operator and the unknown spatial fields that parameterize it. These tasks are coupled: changing field placement changes the differential law, while a sufficiently flexible field can conceal structural error on a single trajectory. We present Hypothesize, Evaluate, Refine for PDE Discovery (HER-PDE), a scientific-agent framework that discovers compositional PDE structure together with nonparametric, time-invariant coefficient fields. The Agent analyzes two noisy trajectories generated by different excitations, proposes complete expression-tree hypotheses, and combines creative structural exploration with local candidate refinement. Its Hypothesis Evaluation Interface (HEI) estimates only the fields explicitly declared in each hypothesis, never adds missing terms, and scores structures by bidirectional cross-excitation transfer. The selected law is subsequ
    
[^119]: SciReC：基于自适应交互的多模态多轮关系推理诊断评估

    SciReC: Diagnostic Evaluation of Multimodal, Multi-Turn Relational Reasoning with Adaptive Interaction

    [https://arxiv.org/abs/2608.27461](https://arxiv.org/abs/2608.27461)

    该论文提出了SciReC——一个模型自适应的多模态学术对话基准，以及DMRA缺陷诊断框架，用于系统评估多模态大语言模型在多轮关系推理中的表现，并量化视觉理解、知识展示和记忆回忆等因素对失败案例的贡献。

    

    关系推理需要对概念之间的潜在关系进行感知理解、比较和整合的过程。这种能力包含多个类别，例如类比推理、结构推理和因果关系推理，每种类型都捕捉了高阶理解的不同方面。为了检验多模态大语言模型（MLLM）在这些关系推理任务上的表现，我们开发了 SciReC，一个模型自适应的多模态学术对话基准。由于关系推理过程涉及多种表示和多种因素（视觉理解、知识展示和记忆回忆），我们提出了 DMRA，一种基于缺陷的诊断框架，用于量化这些组成部分的贡献，以识别失败案例的主要原因。Claude 4.6 在总体关系得分上取得了最佳表现，达到 73%，其次是 GPT 5.4，得分为 68%。性能趋势表明

    arXiv:2608.27461v1 Announce Type: new  Abstract: Relational reasoning requires the process of perceptual understanding, comparing, and integrating the underlying relationships between concepts. This ability consists of multiple categories, such as analogical, structural, and cause-effect, each capturing a different aspect of higher-order understanding. To examine the performance of multimodal large language models (MLLM) on these relational inference tasks, we developed SciReC, a model-adaptive multimodal academic dialog benchmark. As the relational reasoning process involves multiple representations and various factors (visual understanding, exhibiting knowledge, and memory recall), we propose DMRA, a deficit-based diagnostic framework that quantifies the contribution of these components to identify the primary cause of unsuccessful cases. Claude 4.6 achieved the best performance on the overall relational score with 73\%, followed by GPT 5.4 with 68\%. Performance trends indicate that
    
[^120]: 基于向量索引输出嵌入加速大语言模型推理

    Accelerating LLM Inference via Vector Index Based Output Embeddings

    [https://arxiv.org/abs/2608.27460](https://arxiv.org/abs/2608.27460)

    本文将大语言模型的输出投影重新表述为基于HNSW向量索引的最大内积搜索，仅检索高分候选词元以替代稠密词表投影，在CPU推理中最高可将解码吞吐量提升82%且不损失生成质量。

    

    大型输出嵌入矩阵在自回归解码过程中会造成显著的内存带宽瓶颈，尤其是对于拥有庞大多语言词表的紧凑型大语言模型。我们将输出投影及随后的top-k词元选择重新表述为对词元嵌入的最大内积搜索，并用基于HNSW的向量索引取代稠密的词表投影。由此得到的输出头仅检索一小部分高分候选词元，并可通过将检索到的logits散射到稀疏的全词表张量中，集成到现有的解码流水线中。在Gemma 3、Llama 3.2和Qwen 3模型的CPU推理实验中，我们的方法显著加速了输出投影，使Gemma 3 270M的端到端批大小为1的解码吞吐量最高提升了82%，同时在AlpacaEval评估下保持了生成质量。这些结果表明，近似检索是稠密输出投影的一种实用替代方案。

    arXiv:2608.27460v1 Announce Type: new  Abstract: Large output embedding matrices create a significant memory bandwidth bottleneck during autoregressive decoding, especially for compact LLMs with large multilingual vocabularies. We reformulate the output projection followed by top-k token selection as a maximum inner product search over token embeddings and replace the dense vocabulary projection with an HNSW-based vector index. The resulting output head retrieves only a small candidate set of high-scoring tokens and can be integrated into existing decoding pipelines by scattering retrieved logits into a sparse full-vocabulary tensor. On CPU inference with Gemma 3, Llama 3.2, and Qwen 3 models, our method substantially accelerates the output projection and improves end-to-end batch-size-one decoding throughput by up to 82% for Gemma 3 270M, while preserving generation quality under AlpacaEval evaluation. These results suggest approximate retrieval is a practical alternative to dense out
    
[^121]: 理解用于大语言模型推理的进化策略：比GRPO更广泛的推理覆盖

    Understanding Evolution Strategies for LLM Reasoning: Broader Reasoning Coverage than GRPO

    [https://arxiv.org/abs/2608.27351](https://arxiv.org/abs/2608.27351)

    本文通过理论和实证证明进化策略（ES）比GRPO提供更广泛的推理覆盖，从而提升大语言模型推理性能，并提出了GRPO-ES顺序训练策略。

    

    arXiv:2608.27351v1 公告类型：新公告 摘要：进化策略（ES）最近作为一种内存高效的后训练范式出现，用于大语言模型推理。然而，ES的优化行为仍未得到充分研究，这使得难以定义其相对于主流后训练范式（如组相对策略优化（GRPO））的优势范围。通过系统研究ES的动态和机制，本文首先识别出ES相对于GRPO的性能优势，从理论和实证上表明ES可以导致更广泛的推理覆盖，从而更好地利用预训练大语言模型的推理能力。理论上，我们展示了ES群体中验证器投影的Jensen-Shannon多样性有助于更高的Pass@K性能。实证上，与表现出熵坍缩的GRPO不同，ES在提高Pass@1的同时，实现了比GRPO更高的Pass@K。我们进一步开发了一种顺序的GRPO-ES训练策略，结合了GRPO的优势。

    arXiv:2608.27351v1 Announce Type: new  Abstract: Evolution Strategies (ES) have recently emerged as a memory-efficient post-training paradigm for LLM reasoning. However, the optimization behavior of ES remains understudied, making it hard to define its advantage scope compared to mainstream post-training paradigms (e.g., Group Relative Policy Optimization (GRPO)). By systematically investigating ES dynamics and mechanisms, this paper first identifies a performance advantage of ES over GRPO, theoretically and empirically showing that ES can lead to broader reasoning coverage, thereby better exploiting the reasoning capabilities of pretrained LLMs. Theoretically, we show that verifier-projected Jensen-Shannon diversity across the ES population is helpful to higher Pass@K performances. Empirically, unlike GRPO, which exhibits entropy collapse, ES improves Pass@1 while attaining higher Pass@K than GRPO. We further develop a sequential GRPO-ES training strategy that combines GRPO's strength
    
[^122]: 基于利润的机器学习评估在冬小麦氮肥推荐中的应用

    Profit based evaluation of machine learning for nitrogen recommendations in winter wheat

    [https://arxiv.org/abs/2608.27205](https://arxiv.org/abs/2608.27205)

    本文提出直接以放弃利润评估氮肥推荐，发现机器学习在利润上不如标准建议，但简单修正可带来收益。

    

    摘要：冬小麦的氮肥施用量是在季节前设定的，此时价格和天气未知。英国的标准建议不响应价格变化，然而近期的价格波动使最有利的施氮量每公顷变动了数十公斤。机器学习常被提议作为解决方案。然而，它通常以预测精度来评判，而准确的预测本身并不能使推荐的施氮量更有利可图。我们的见解是直接通过实测产量响应曲线上放弃的利润来评分氮肥建议。我们在来自两个长期英国实验的892条这样的曲线上建立了一个测试基准，并扫描氮肥与谷物价格比以覆盖所有价格情景。在这个基准上，机器学习作为预测器表现不佳。没有模型能在农场容忍度内恢复最佳施氮量，基准噪声表明没有模型能做到。在正常价格下，每个模型在利润上也输给了标准建议。收益在其他地方。一个简单的修正...

    arXiv:2608.27205v1 Announce Type: new  Abstract: Nitrogen rates for winter wheat are set before the season, under unknown prices and weather. The standard UK advice does not respond to prices, yet recent price swings moved the most profitable rate by tens of kilograms per hectare. Machine learning is often proposed as the fix. However, it is usually judged on prediction accuracy, and accurate prediction does not by itself make the recommended rate more profitable. Our insight is to score nitrogen advice directly by the profit it forgoes on measured yield response curves. We build a test bench on 892 such curves from two long running UK experiments, and sweep the nitrogen to grain price ratio to cover all price scenarios. On this bench, machine learning fails as a predictor. No model recovers the best rate within farm tolerance, and the benchmark noise shows none can. At normal prices, every model also loses to the standard advice on profit. The gain sits elsewhere. A simple correction 
    
[^123]: 基于嵌入的神经网络回归用于知识图谱中数值属性预测

    Neural Regression with Embeddings for Numerical Attribute Prediction in Knowledge Graphs

    [https://arxiv.org/abs/2608.26729](https://arxiv.org/abs/2608.26729)

    本文提出LitEm神经回归模型和协同训练框架，使知识图谱嵌入模型能预测数值属性，并提升双线性模型的链接预测性能。

    

    arXiv:2608.26729v1 公告类型：新 摘要：近年来，直推式知识图谱嵌入模型已被应用于链接预测和查询回答等任务。尽管知识图谱通常包含丰富的数值属性，但大多数嵌入模型忽略了这些属性，限制了它们表示具有多样化信息的现实世界知识图谱的能力。在这项工作中，我们提出了一种神经回归模型（LitEm），该模型使直推式知识图谱嵌入模型能够预测知识图谱中的数值属性。实验结果表明，LitEm在FB15K-237、YAGO15K、DB15K和Mutagenesis数据集上对大多数属性达到了最佳或次佳结果。此外，我们提出了一种协同训练框架，将最先进的直推式知识图谱嵌入模型与LitEm联合训练，这主要提高了双线性模型的链接预测性能，同时使其能够预测数值属性。此外，该框架还...

    arXiv:2608.26729v1 Announce Type: new  Abstract: In recent years, transductive knowledge graph embedding models have been applied to tasks such as link prediction and query answering. Although knowledge graphs often contain rich numerical attributes, most embedding models neglect them, limiting their ability to represent real-world knowledge graphs with diverse information. In this work, we propose a neural regression model (LitEm) that enables transductive knowledge graph embedding models to predict numerical attributes within knowledge graphs. Experimental results demonstrate that LitEm achieves the best or second-best results on most attributes across FB15K-237, YAGO15K, DB15K, and Mutagenesis. Furthermore, we propose a co-training framework that jointly trains state-of-the-art transductive knowledge graph embedding models with LitEm, which improves link prediction performance mainly for bilinear models and simultaneously enables them to predict numerical attributes. In addition, th
    
[^124]: TraceML：机器学习开发中人机协同规划的经验分析

    TraceML: An Empirical Analysis of Human-Agent Planning in Machine Learning Development

    [https://arxiv.org/abs/2608.26086](https://arxiv.org/abs/2608.26086)

    本文通过引入TraceML数据集，将人类与智能体在同一机器学习竞赛中的开发轨迹进行版本级对比，揭示了智能体在自主开发中的性能差距及其具体成因。

    

    大型语言模型在解决孤立问题时能编写正确的代码，但在自主机器学习开发方面仍远远落后，因为智能体必须在数小时的反馈中修订数据管道、模型和验证过程，并且在大多数竞赛中仍低于强大的人类竞争者。基于结果的基准测试记录了这种差距，但未揭示其原因，因为它们只评估最终提交结果，而忽略了其背后的开发过程。我们引入了TraceML，它在统一的版本级模式框架下配对人类和智能体在同一竞赛中的工作：涵盖134个竞赛中的4,465条人类Kaggle轨迹，其中7个竞赛也由两个智能体框架完成，产生了430条配对的人类轨迹和207条智能体轨迹。每个代码版本都带有其得分、时间戳以及所采取行动、意图、编辑大小和得分效果的标签。通过这种方式解读，差距变得具体化。专家交替进行数据工作、验证、模型更改和评估。

    arXiv:2608.26086v1 Announce Type: new  Abstract: Large language models write correct code for isolated problems but remain far weaker at autonomous machine-learning development, where an agent must revise data pipelines, models, and validation over hours of feedback, and on most competitions still finishes below strong human competitors. Outcome-based benchmarks record this gap but not its cause, because they grade the final submission and discard the development process behind it. We introduce TraceML, which pairs human and agent work on the same competitions under one version-level schema: 4,465 human Kaggle trajectories across 134 competitions, seven of which are also worked by two agent scaffolds, giving 430 paired human and 207 agent trajectories. Every code version carries its score, its timestamp, and labels for the action taken, its intent, the edit size, and the score effect. Read this way, the gap becomes concrete. Experts alternate data work, validation, model changes, and e
    
[^125]: 信任大众：KV缓存驱逐中的强制权重

    Trust the Mass: Forced Weights in KV-Cache Eviction

    [https://arxiv.org/abs/2608.25230](https://arxiv.org/abs/2608.25230)

    本文发现KV缓存驱逐中保留最大权重已接近最优，已发布方法间的差异主要源于存储方式而非选择策略，并揭示了评估中的内存与性能权衡。

    

    arXiv:2608.25230v1 公告类型：交叉 摘要：每个部署的稀疏注意力或KV缓存驱逐规则都会保留一部分键，丢弃其余部分，并对保留集上的注意力权重进行重新归一化。在来自五个模型的168,192个注意力行上，枚举该约束下的精确最优子集表明，保留最大权重已经接近最优，因为最优子集仅将剩余差距中位数缩小了2%到5%。如果选择带来的改进如此之小，那么已发布的驱逐方法之间的差异必然来自其他方面，因此我们测量了每种方法持有的字节数。在共享评估流程中，最强的查询无关方法持有完整缓存，因为它们的按头选择存储为掩码，只有不规则按头存储才能释放该内存。在固定选择上强制执行名义预算会损失14到62个基准点。我们将一个87.6点的检索差距追溯到在问题可见时计算的排名。

    arXiv:2608.25230v1 Announce Type: cross  Abstract: Every deployed sparse-attention or KV-cache-eviction rule keeps a subset of the keys, discards the rest, and renormalizes the attention weights over the kept set. Enumerating the exact best subset under that constraint on $168{,}192$ attention rows from five models shows that keeping the largest weights is already near-optimal, since the best subset closes only a median $2$ to $5\%$ of the remaining gap to full attention. If selection closes this little, published margins between eviction methods must come from elsewhere, so we measure the bytes each method holds. In the shared evaluation pipeline, the strongest query-agnostic methods hold the full cache because their per-head selections are stored as masks, and only ragged per-head storage frees that memory. Enforcing a nominal budget on one fixed selection costs $14$ to $62$ benchmark points. We trace an $87.6$-point retrieval margin to rankings computed while the question is visible
    
[^126]: 基于可验证奖励的在线策略蒸馏

    On-policy Distillation with Verifiable Reward

    [https://arxiv.org/abs/2608.24696](https://arxiv.org/abs/2608.24696)

    提出了一种无需额外超参数的无缝结合在线策略蒸馏和可验证奖励强化学习的方法，通过基于轨迹正确性的奖励重构和ReLU门控机制，有效提升大型语言模型后训练性能。

    

    arXiv:2608.24696v1 公告类型：交叉 摘要：基于可验证奖励的强化学习（RLVR）和在线策略蒸馏（OPD）已成为大型语言模型后训练中广泛采用的两种范式。然而，RLVR面临稀疏的任务级反馈问题，而OPD提供密集的令牌级指导但忽略轨迹正确性，将其性能限制在教师模型水平。将两者结合是一个有前景的方向：OPD提供密集的监督信号，而RLVR提供任务级正确性。尽管如此，现有集成方法通常依赖加权组合或启发式切换，引入了额外超参数和权衡。我们提出了一种名为基于可验证奖励的在线策略蒸馏（OPDVR）的方法，这是一种简单而有效的方法，无需添加任何超参数即可无缝结合OPD和RLVR。我们首先根据轨迹正确性重新表述采样令牌OPD的隐式奖励，然后应用ReLU门控机制以确保正确的轨迹获得正向更新。

    arXiv:2608.24696v1 Announce Type: cross  Abstract: Reinforcement Learning with Verifiable Rewards (RLVR) and on-policy distillation (OPD) have become two widely adopted paradigms for post-training large language models. However, RLVR suffers from sparse task-level feedback, while OPD provides dense token-level guidance but ignores trajectory correctness, limiting its performance to that of the teacher. Combining them is a promising direction: OPD supplies dense supervisory signals, while RLVR provides task-level correctness. Nevertheless, existing integrations often rely on weighted combination or heuristic switching, introducing extra hyperparameters and trade-offs. We propose On-policy Distillation with Verifiable Reward (OPDVR), a simple yet effective method that seamlessly combines OPD and RLVR without adding any hyperparameters. We first reformulate the implicit reward of sampled-token OPD based on trajectory correctness, then apply a ReLU gating mechanism to ensure that correct t
    
[^127]: XP-JEPA：用于可预测潜在动力学的跨预测物理基础

    XP-JEPA: Cross-Predictive Physics Grounding for Forecastable Latent Dynamics

    [https://arxiv.org/abs/2608.24044](https://arxiv.org/abs/2608.24044)

    XP-JEPA通过跨模态共享预测器将视觉潜在动力学与物理轨迹结合，训练后仅保留视觉模型，从而在部署时减少滚动漂移并增强物理可预测性。

    

    潜在世界模型通过预测候选动作如何转换学习到的表示来进行规划。然而，在自预测模型中，编码器和预测器被联合优化，可能共同适应于易于预测但受场景物理演化约束较弱的潜在转换。我们引入了跨预测JEPA（XP-JEPA），该方法将视觉潜在动力学锚定在特权物理轨迹上。XP-JEPA分别编码视觉观察和物理状态，通过共享的动作条件预测器推进两者，并将每个预测与两种未来表示进行匹配。这一目标鼓励跨两种模态的统一潜在动力学，并基于底层物理转换进行基础化。训练后，物理分支被丢弃，部署时仅保留视觉模型。在一个涵盖六个评估子族的多任务套件上，XP-JEPA减少了新拟合p的滚动漂移。

    arXiv:2608.24044v1 Announce Type: new  Abstract: Latent world models plan by predicting how candidate actions transform learned representations. In self-predictive models, however, the encoder and predictor are optimized jointly and can co-adapt to latent transitions that are easy to predict but only weakly constrained by the physical evolution of the scene. We introduce the cross-predictive JEPA (XP-JEPA), which grounds visual latent dynamics in privileged physical trajectories. XP-JEPA separately encodes visual observations and physical states, advances both through a shared action-conditioned predictor, and matches each prediction to both future representations. This objective encourages unified latent dynamics across the two modalities, grounded in the underlying physical transitions. The physical branch is discarded after training, leaving a visual-only model at deployment. On a multi-task suite spanning six evaluation subfamilies, XP-JEPA reduces rollout drift of a newly fitted p
    
[^128]: 语义覆盖层：通过超越令牌和引导向量的注释缓解提示注入

    Semantic Overlays: Mitigating Prompt Injection with Annotations Beyond Tokens and Steering Vectors

    [https://arxiv.org/abs/2608.23873](https://arxiv.org/abs/2608.23873)

    该论文提出了一种名为“语义覆盖层”的新技术，通过向模型输入添加非文本通道来缓解提示注入攻击，利用小型学习的适配器在冻结模型的残差流中创建带外注释，从而增强模型对片段身份的理解。

    

    摘要：arXiv:2608.23873v1 公告类型：新 摘要：语言模型看到的一切都是令牌。服务堆栈知道每个片段是什么——用户输入、工具输出、指令——但模型必须自己跟踪这些，它可能会失去跟踪或被混淆：文本可以被写成看起来像任何东西。提示注入是对这种现象的自然利用。通过扰乱模型对片段身份的理解，攻击者可以诱导不必要的、可能危险的行为。在模型输入中添加一个非文本通道——一种超越文本传达片段身份的方式——缓解了这类攻击。因此，我们引入了一种通用的引导技术，称为语义覆盖层：小型学习的适配器，应用于冻结模型的残差流中的选定预填充位置。在片段上铺设覆盖层创建了一个带外注释通道，该通道无法通过令牌复制。与引导向量不同，语义覆盖层是经过训练的、可适应的，并有选择性地应用。一个覆盖层...

    arXiv:2608.23873v1 Announce Type: new  Abstract: Everything a language model sees is tokens. The serving stack knows what each span is -- user input, tool output, instructions -- but the model must keep track of that itself, and it can lose track or be confused: text can be written to read like anything. Prompt injection is a natural exploit of this phenomenon. By scrambling the model's understanding of span identity, an attacker can induce unwanted and potentially dangerous actions. Adding a non-textual channel to the model's input -- a way to communicate span identity beyond text -- mitigates this class of attack. We thus introduce a general steering technique called Semantic Overlays: small learned adapters applied at chosen prefill positions to a frozen model's residual stream. Laying an overlay over a span creates an out-of-band annotation channel that cannot be replicated by tokens. Unlike steering vectors, Semantic Overlays are trained, adaptable, and selectively applied. An ove
    
[^129]: RIBOSPAN：一种用于多功能RNA建模的长上下文RNA基础模型

    RIBOSPAN: A Long-Context RNA Foundation Model for Versatile RNA Modeling

    [https://arxiv.org/abs/2608.22849](https://arxiv.org/abs/2608.22849)

    提出了RIBOSPAN——一个16.1亿参数、原生支持最长10,240核苷酸上下文的双向RNA基础模型，通过密集双向自注意力、单核苷酸分词和注意力隔离序列打包，实现了对完整长RNA的单核苷酸分辨率建模。

    

    全长RNA，尤其是信使RNA（mRNA），通常超出现有RNA基础模型预训练时所使用的上下文长度，限制了在单核苷酸分辨率下对完整转录本的建模。我们提出了RIBOSPAN，一个16.1亿参数的双向RNA基础模型，以最长10,240个核苷酸的上下文长度进行原生预训练。RIBOSPAN结合了密集双向自注意力机制、单核苷酸分词以及注意力隔离的序列打包技术，实现了对完整长RNA的高分辨率建模。原生10K预训练在10,240个token长度下保持了强大的重建能力，并在一个受控的长上下文基准测试中，维持了较强的上下文响应能力和上下文特异的表示分离，同时使扰动引起的变化保持高度局部化。推理阶段的YaRN扩展恢复了直接短上下文外推所丢失的大部分上下文组织结构，但会引起明显更大的距离……（原文摘要在此处截断）

    arXiv:2608.22849v2 Announce Type: replace  Abstract: Full-length RNAs, particularly messenger RNAs, often exceed the context lengths used to pretrain existing RNA foundation models, limiting complete-transcript modeling at single-nucleotide resolution. We present RIBOSPAN, a 1.61-billion-parameter bidirectional RNA foundation model natively pretrained with context lengths up to 10,240 nt. RIBOSPAN combines dense bidirectional self-attention, single-nucleotide tokenization, and attention-isolated sequence packing to enable high-resolution modeling of complete long RNAs. Native 10K pretraining preserves strong reconstruction at 10,240 tokens and, in a controlled long-context benchmark, maintains strong contextual responsiveness and context-specific representation separation while keeping perturbation-induced changes highly localized. Inference-time YaRN scaling recovers much of the contextual organization lost by direct short-context extrapolation, but induces substantially greater dista
    
[^130]: GAN-Diff：耦合预训练WGAN-GP特征与条件扩散U-Net

    GAN-Diff : Coupling Pretrained WGAN-GP Features with Conditional Diffusion U-Nets

    [https://arxiv.org/abs/2608.22272](https://arxiv.org/abs/2608.22272)

    本文提出一种混合GAN引导扩散框架，通过将预训练WGAN-GP的中间特征固定融入条件扩散U-Net，并解决多种训练不稳定性，从而提升图像恢复质量。

    

    生成对抗网络（GAN）能够提供高效的图像生成，而扩散模型则提供高质量的图像恢复，但需要迭代采样。本文提出了一种混合的GAN引导扩散框架，该框架使用预训练的带梯度惩罚的Wasserstein GAN（WGAN-GP）作为条件扩散图像恢复的特征先验。来自冻结的WGAN-GP生成器的中间特征通过交叉注意力被整合到扩散U-Net中，并在DDIM采样过程中保持固定。该框架在两个恢复任务上进行了评估，即高斯去噪和2倍超分辨率，使用CelebA人脸图像。在开发过程中，识别并解决了多个不稳定性来源，包括对抗学习率不平衡、不适当的扩散初始化、过度损坏以及参数平均不足。所得框架持续改善了质量。

    arXiv:2608.22272v1 Announce Type: cross  Abstract: Generative adversarial networks (GANs) can provide efficient image generation, while diffusion models offer high-quality image restoration but require iterative sampling. This paper presents a hybrid GAN-guided diffusion framework that uses a pretrained Wasserstein GAN with gradient penalty (WGAN-GP) as a feature prior for conditional diffusion-based image restoration. Intermediate features from the frozen WGAN-GP generator are incorporated into a diffusion U-Net through cross-attention and remain fixed during the DDIM sampling process. The framework is evaluated on two restoration tasks, Gaussian denoising and 2Xsuper-resolution, using CelebA face images. During development, several sources of instability were identified and addressed, including adversarial learning-rate imbalance, inappropriate diffusion initialization, excessive corruption, and insufficient parameter averaging. The resulting framework consistently improves the quali
    
[^131]: 架构与训练如何影响跨实验的TPC（时间投影室）表征

    How Architecture and Training Affect TPC Representations Across Experiments

    [https://arxiv.org/abs/2608.21756](https://arxiv.org/abs/2608.21756)

    该工作以TPC数据为测试平台，提出利用冻结编码器上的探针并结合随机权重对照，评估事件表征的跨实验、跨探测器复用性，并区分架构与编码器训练各自的贡献。

    

    深度学习的研究正日益转向基础模型方法。在实验物理学中，这使得模型及其学到的表征能够在开发它们的实验之外被复用。本工作利用冻结编码器上的探针，评估了表征在不同实验和探测器系统之间的可复用性。这些探针能够在下游适配之前揭示与任务相关的结构，与微调形成互补。结合随机权重对照，它们能够区分架构与编码器训练各自的贡献，而这是仅凭下游性能无法分辨的。时间投影室（TPC）数据提供了一个有用的测试平台，因为TPC系统的事件可以表示为可变长度的稀疏张量，而探测器的几何结构、事件拓扑和科学任务可能存在很大差异。我们研究了固定维度的TPC事件表征能否在不同分类任务之间复用……（原文摘要在此处截断）

    arXiv:2608.21756v2 Announce Type: replace  Abstract: Deep-learning efforts have increasingly shifted toward foundation model approaches. In experimental physics, this allows models and learned representations to be reused beyond the experiments in which they were developed. This work evaluates the reusability of representations across experiments and detector systems using probes on frozen encoders. These probes reveal task-relevant structure before downstream adaptation, complementing fine-tuning. Together with random-weight controls, they distinguish contributions from architecture and encoder training that downstream performance alone cannot resolve.   Time projection chamber (TPC) data provide a useful testbed because events from TPC systems can be represented as variable-length sparse tensors, while detector geometries, event topologies, and scientific tasks can differ substantially. We investigate whether fixed-dimensional TPC event representations can be reused across classifica
    
[^132]: 神经网络场理论在计算机上能实现与不能实现什么

    What Neural Network Field Theory Can and Cannot Realise on a Computer

    [https://arxiv.org/abs/2608.21523](https://arxiv.org/abs/2608.21523)

    该论文提出了一个适用于标准网络架构的不可能性定理，据此将神经网络场理论分为四种版本，并证明有限宽度系综无论被解释为量子场论还是有效场论，都无法在计算机上自洽实现。

    

    神经网络场理论的一个目标是将量子场论或有效场论放到计算机上实现，并以网络系综本身作为该理论。我们探究对于一类足够正则、因而可被计算的函数，这一目标能够推进到何种程度。我们的主要结果是一个不可能性定理，其假设条件适用于标准网络架构。我们利用该定理将神经网络场理论区分为四种版本：其划分依据是定义对象为有限宽度系综还是其无穷宽度极限，以及欲计算的目标是量子场论还是有效场论。两种有限宽度解释都无法直接自洽：对于每点方差均有限的有限宽度系综而言，量子场论解释不满足反射正性，而有效场论解释则无法建立尺度分离，因而无法将这种正性破坏置于其适用范围之外。至于两种极限版本……（原文摘要在此处被截断）

    arXiv:2608.21523v2 Announce Type: replace-cross  Abstract: One aim of neural network field theory is to put a quantum or effective field theory on a computer, with the network ensemble itself as the theory. We ask how far that aim can be pushed for a function class regular enough to be computed with. Our main result is a no-go theorem with assumptions that hold for standard network architectures. We use it to separate four versions of neural network field theory, according to whether the defining object is the finite width ensemble or its infinite width limit, and whether the target we want to compute is a quantum or an effective field theory. Neither finite width interpretation is straightforwardly consistent. For finite width ensembles with finite variance at each point, the QFT interpretation fails reflection positivity, while the EFT interpretation establishes no scale separation by which the positivity violation can be placed outside its domain of validity. Of the two limit versio
    
[^133]: JuryProbe：一种用于路由无参考事实性评审团到有依据验证的经验共识风险诊断方法

    JuryProbe: An Empirical Consensus-Risk Diagnostic for Routing Reference-Free Factuality Judge Panels to Grounded Verification

    [https://arxiv.org/abs/2608.20607](https://arxiv.org/abs/2608.20607)

    本文提出JuryProbe，一种通过仅假阴性相关性和假共识提升度来诊断无参考事实性评审团共识风险的方法，并在高风险时路由到有参考验证，以减少因共享盲点导致的错误接受。

    

    arXiv:2608.20607v1 公告类型：交叉 摘要：由廉价LLM评审员组成的小组越来越多地做出接受或升级的决策。在事实性设置中，因为多个无参考评审员一致同意而接受一个声明可能会产生隐藏风险：这种一致性可能反映的是共同的假阴性盲点，而非独立的证据。我们引入了JuryProbe，一种针对无参考事实性评审团的经验共识风险诊断方法，并配以基于校准的路由策略。JuryProbe通过使用仅假阴性（FN-only）评审员相关性和假共识提升度，从标记的校准探针中估计共识风险；当标记为高风险时，无参考多数接受会被路由到带有可信参考的相同评审员。在审计的FEVER腐败数据上，无参考评审团显示出相关的假阴性（FN-only相关性为0.402和0.368；提升度分别为3.13倍和18.13倍），而在可信参考最佳案例诊断下，两种情形的一致假共识均降至零。

    arXiv:2608.20607v1 Announce Type: cross  Abstract: Panels of inexpensive LLM judges increasingly make accept-or-escalate decisions. In factuality settings, accepting a claim because several reference-free judges agree can create a hidden risk: agreement may reflect shared false-negative blind spots rather than independent evidence. We introduce JuryProbe, an empirical consensus-risk diagnostic for reference-free factuality judge panels, paired with a calibration-based routing policy. JuryProbe estimates consensus risk from a labeled calibration probe using false-negative-only (FN-only) judge correlation and false-consensus lift; when flagged high-risk, reference-free majority accepts are routed to the same judges with trusted references. On audited FEVER corruptions, reference-free panels show correlated false negatives (FN-only correlations 0.402 and 0.368; lifts 3.13x and 18.13x), while unanimous false consensus drops to zero under a trusted-reference best-case diagnostic on both min
    
[^134]: 为什么GPT风格模型不能直接迁移到符号音乐：在错误的坐标系统中进行压缩

    Why GPT-Style Models Do Not Directly Transfer to Symbolic Music: Compression in the Wrong Coordinate System

    [https://arxiv.org/abs/2608.18025](https://arxiv.org/abs/2608.18025)

    本文指出GPT风格模型无法直接迁移到符号音乐的原因在于压缩发生在错误的坐标系统中，并提出一个有效性-无损框架，强调分词化的核心是发现使音乐规律可预测压缩的坐标系统，而非单纯寻找更大的音乐组合。

    

    摘要：arXiv:2608.18025v1 公告类型：交叉 摘要：GPT风格模型通过使用有限的、可复用的离散令牌词汇来表示语言，从而实现了强大的性能。这一成功促使符号音乐分词化将重复出现的音乐结构，如和弦、动机和乐句，视为类似于语言令牌的可复用单元。然而，分词化的优势不仅来自于可复用的组合，更来自于压缩：有效的压缩需要坐标系统，在该系统中，重复出现的规律形成稳定且可预测的条件分布。因此，关键问题不在于寻找更大的音乐组合，而在于发现使音乐事实变得可预测压缩的坐标系统。我们提出了“有效性-无损性框架”，并将分词化定义为构建一个预测有效且关系无损的坐标系统。预测有效性原则定义了“事实-令牌边界”：

    arXiv:2608.18025v1 Announce Type: cross  Abstract: GPT-style models achieve strong performance by representing language with finite vocabularies of reusable discrete tokens. This success has motivated symbolic music tokenizations to treat recurring musical structures, such as chords, motifs, and phrases, as reusable units analogous to linguistic tokens. However, tokenization derives its advantage not from reusable combinations alone, but from compression: effective compression requires coordinates in which recurring regularities form stable and predictable conditional distributions. The key problem is therefore not to find larger musical combinations, but to discover the coordinate system in which musical facts become predictively compressible. We formulate the Effectiveness--Losslessness Framework and define tokenization as the construction of a predictively effective and relationally lossless coordinate system. The Predictive Effectiveness Principle defines the Fact--Token Boundary: 
    
[^135]: 再循环

    Recirculation

    [https://arxiv.org/abs/2608.17981](https://arxiv.org/abs/2608.17981)

    本文提出“再循环”技术，通过推理时引入特定循环机制，使现成基础模型能跟踪信念状态，显著降低困惑度并提升生成和推理准确性，且几乎不增加生成延迟。

    

    arXiv:2608.17981v1 公告类型：新 摘要：我们描述了一种针对现成基础模型的推理时架构增强方法，该方法显著降低了困惑度，并在生成和推理任务中提升了准确性。我们的方法在生成过程中几乎不增加额外延迟，但在预填充阶段需要串行处理。受前馈变压器中状态更新受模型深度限制这一根本性局限的启发，我们的技术——再循环——引入了一种特定形式的循环，使模型能够作为动态系统运行并跟踪信念状态。我们将此技术与思维链计算区分开来——后者更适合复杂推理而非基本状态跟踪——同时区别于流行的深度循环技术（循环）和昂贵的循环变压器训练。我们还提出并评估了一种自适应再循环变体，该变体仅需对超参数进行轻度调整。

    arXiv:2608.17981v1 Announce Type: new  Abstract: We describe an inference-time architectural enhancement for off-the-shelf foundation models that markedly reduces perplexity and boosts accuracy across generation and reasoning tasks. Our approach incurs essentially no additional latency during generation, though it requires serial processing in the prefill phase. Motivated by the fundamental limitation that state updates in feedforward transformers are bounded by model depth, our technique, recirculation, introduces a specific form of recurrence that allows the model to act as a dynamical system and track belief states. We distinguish this technique from chain-of-thought computation---which is better reserved for complex inferences rather than basic state tracking---as well as from popular depth-recurrence techniques (looping) and the costly training of recurrent transformers. We also propose and evaluate an adaptive variant of recirculation which requires only light tuning of hyperpara
    
[^136]: 通过重参数化建立镜像下降的边界KKT收敛性

    Establishing Boundary KKT Convergence of Mirror Descent through Reparameterization

    [https://arxiv.org/abs/2608.07248](https://arxiv.org/abs/2608.07248)

    本文通过在重参数化变量下分析镜像下降（使Hessian度量在边界附近保持平坦且非退化），并在延拓性与可定义性条件下，为一类广泛的结构化非凸问题首次建立了镜像下降收敛到边界KKT点的理论保证，克服了Legendre核梯度在边界爆炸的固有困难。

    

    对于采用Legendre核的非凸镜像下降算法，其序列能否收敛到边界Karush–Kuhn–Tucker（KKT）点这一问题长期以来悬而未决。困难源于Legendre核的梯度在边界处会发生爆炸（发散）。近期研究表明，即使目标函数值不断下降，镜像下降仍可能收敛到非KKT的边界点，因此在一般情况下无法保证其收敛到KKT点。尽管存在这一负面结果，镜像下降在许多实际应用中依然有效。受这一反差启发，本文直接应对边界难题，为一类广泛的结构化非凸问题建立了镜像下降的KKT收敛性。我们在重参数化变量下分析镜像下降，在该变量下Hessian度量被“拉平”，并且在逼近边界时保持非退化。在目标函数、Legendre核等要素联合耦合的延拓性与可定义性条件下……（摘要此处被截断）

    arXiv:2608.07248v3 Announce Type: replace-cross  Abstract: Sequence convergence to a boundary Karush--Kuhn--Tucker (KKT) point has long remained unclear for nonconvex mirror descent with Legendre kernels. The difficulty arises from the blow-up of the gradient of the Legendre kernel at the boundary. Recent work~\cite{dingtoh2026nonkkt} shows that mirror descent can accumulate at non-KKT boundary points despite decreasing objective values, precluding a convergence guarantee to KKT points in general. Despite this negative result, mirror descent remains effective in many real applications. Motivated by this contrast, we address the boundary difficulty directly and establish KKT convergence of mirror descent for a broad class of structured nonconvex problems. We analyze mirror descent in reparameterized variables, where the Hessian metric is flattened and remains nondegenerate as the boundary is approached. Under extension and definability conditions jointly coupling the objective, the Lege
    
[^137]: ED-CSP：从电子衍射预测晶体结构

    ED-CSP: Crystal Structure Prediction from Electron Diffraction

    [https://arxiv.org/abs/2608.06448](https://arxiv.org/abs/2608.06448)

    ED-CSP是一个新框架，能从电子衍射数据直接预测晶体结构，通过结合关系集编码器和周期性流生成器，在大型模拟数据集上训练，实现了从稀疏观测到完整结构的生成式重建。

    

    从稀疏、未索引的电子衍射（ED）观测中恢复周期性三维晶体结构是一个具有挑战性的生成性逆问题。现有的基于ED的学习方法主要预测晶体学标签、从索引反射重建结构，或从有限的结构库中检索候选结构。在此，我们引入了ED-CSP，一个机器学习框架，它根据化学成分、原子计数和多个探测器平面的ED点集预测晶体结构。ED-CSP结合了关系集编码器、排列不变的多视角聚合和周期性流生成器，共同预测晶格参数和分数原子坐标。为了训练模型，我们构建了ED-CS数据集，包含485万个模拟的多视角ED晶体结构，这些结构经过七个材料数据库的去重，并过滤以排除CHILI-100K重叠。在2075个保留的CHILI-100K材料上，ED-C...

    arXiv:2608.06448v2 Announce Type: replace-cross  Abstract: Recovering a periodic 3D crystal structure from sparse, unindexed electron diffraction (ED) observations is a challenging generative inverse problem. Existing ED-based learning methods mainly predict crystallographic labels, reconstruct structures from indexed reflections, or retrieve candidates from finite structure libraries. Here, we introduce ED-CSP, a machine learning framework that predicts crystal structures from chemical composition, atom count, and multiple detector-plane ED spot sets. ED-CSP combines a relational set encoder, permutation-invariant multi-view aggregation, and a periodic flow generator to jointly predict lattice parameters and fractional atomic coordinates.   To train the model, we construct ED-CS, a dataset of 4.85 million simulated multi-view ED crystal structures, deduplicated across seven materials repositories and filtered to exclude CHILI-100K overlaps. On 2,075 held-out CHILI-100K materials, ED-C
    
[^138]: 锁定评估面：CRISPRi扰动效应预测中的迁移失败与采样深度纠缠

    Locked Evaluation Surfaces: Transfer Failure and Sampling-Depth Entanglement in CRISPRi Perturbation-Effect Prediction

    [https://arxiv.org/abs/2608.00152](https://arxiv.org/abs/2608.00152)

    该论文在锁定且预注册的评估协议下评估冻结的Geneformer表示，发现其在虚拟细胞挑战赛（VCC）分布内数据上具有显著超越随机特征对照的预测信息量，但在零样本跨筛选迁移中失败，并揭示了迁移失败与采样深度等设计因素之间的纠缠。

    

    预测保留的目标基因如何响应CRISPRi扰动，以及此类预测能否在不同生物筛选之间迁移，是很难评估的：一种表示可能在单个筛选内具有信息量，却在跨筛选时失效，同时终点定义和采样深度等设计因素在不同数据集之间也存在差异。我们在一个锁定且预注册的协议下评估冻结的Geneformer表示：分类头与模型选择在测试评估之前冻结，外部结果标签在最终揭盲前保密，且支配分析的决策在其所支配的评估之前即已固定。在虚拟细胞挑战赛（VCC）的分布内数据上，该冻结表示携带可测量的预测信息，超过维度匹配的随机特征对照（ΔR² = +0.1645，95%置信区间 [+0.1375, +0.1920]），满足了在解释迁移之前所要求的预注册信息量门槛。随后它在零样本……[摘要在此处被截断]

    arXiv:2608.00152v2 Announce Type: replace  Abstract: Predicting how held-out target genes respond to CRISPRi perturbation, and whether such predictions transfer across biological screens, is hard to evaluate: a representation can be informative within one screen yet fail across screens, while endpoint definitions and design factors such as sampling depth differ between datasets. We evaluate a frozen Geneformer representation under a locked, pre-registered protocol, with heads and model selection frozen before test evaluation, external outcome labels withheld until final unblinding, and analysis-governing decisions fixed before the evaluations they govern. In-distribution on the Virtual Cell Challenge (VCC), the frozen representation carries measurable predictive information beyond a dimension-matched random-feature control (Delta R^2 = +0.1645, 95% CI [+0.1375, +0.1920]), satisfying the pre-registered informativeness gate required before interpreting transfer. It then fails zero-shot t
    
[^139]: 引导信号从何而来：激活引导中的激活源选择

    Where Steering Signals Come From: Activation Source Selection in Activation Steering

    [https://arxiv.org/abs/2607.25270](https://arxiv.org/abs/2607.25270)

    该论文首次将激活引导中常被忽视的“激活源选择”作为核心研究对象，发现引导信号的有效性关键取决于激活是否取自模型即将执行目标行为的“执行边界状态”，而非源文本中是否包含期望行为。

    

    激活引导通过在推理时向隐藏状态中添加向量或特征来控制语言模型，但这些引导信号的上游来源通常被视为次要细节。我们将这一来源选择作为“激活源选择”进行研究：即用于收集隐藏状态（并从中构建引导信号）的源上下文与激活读取策略的组合。在保持下游干预不变的情况下，我们在三个指令微调模型和四个引导任务族上证明，仅改变源激活就会显著改变引导的成功率。我们进一步发现，有效的引导并不能简单地用源文本中是否出现期望行为来解释。相反，强信号来自“执行边界状态”，即模型即将产生或继续目标行为时的状态。这种实现前/实现后的区分解释了为什么基于答案的源有时有效：……

    arXiv:2607.25270v2 Announce Type: replace  Abstract: Activation steering controls language models by adding vectors or features to hidden states at inference time, but the upstream source of these steering signals is often treated as a secondary detail. We study this source choice as activation source selection: the combination of source context and activation readout policy used to collect the hidden states from which a steering signal is built. Holding the downstream intervention fixed, we show across three instruction-tuned models and four steering task families that changing only the source activations substantially changes steering success. We further find that effective steering is not explained simply by whether the desired behavior appears in the source text. Instead, strong signals come from execution-boundary states, where the model is about to produce or continue the target behavior. This pre-/post-realization distinction explains why answer-based sources sometimes work: the
    
[^140]: 关于逻辑门网络的深度可扩展性研究

    On the Depth Scalability of Logic Gate Networks

    [https://arxiv.org/abs/2607.21633](https://arxiv.org/abs/2607.21633)

    提出输入锚定逻辑门网络（IALGN），通过让每个门结合私有隐藏主干与直接输入锚点的拓扑设计，解决了逻辑门网络加深时优化崩塌与信用分配退化的问题，在MNIST、CIFAR-10和CIFAR-100上实现了高达150层的一致深度-准确率扩展。

    

    逻辑门网络通过布尔运算的组合进行计算，然而现有的逻辑门网络并不能可靠地从深度的增加中获益。我们确定了两个原因：一是优化崩塌，二是由拓扑结构引起的输出特定信用分配退化——即使通过偏置跳跃初始化和直通估计稳定了训练，这种退化依然存在。我们提出了输入锚定逻辑门网络，其中每个门都将一个私有的隐藏主干与一个直接的输入锚点相结合。这种拓扑结构在保证每一层都能访问输入的同时，防止了输出路径的合并。信用诊断实验表明，随机布线会稀释或产生冲突的输出特定梯度，而IALGN则能够保持可用且连贯的信用分配。Random-$k_x$松弛方法在不放松主干的情况下改进了锚点的选择。在MNIST、CIFAR-10和CIFAR-100数据集上，IALGN展现出一致的固定宽度深度-准确率扩展性，最深可达150层，而其他拓扑结构……

    arXiv:2607.21633v3 Announce Type: replace  Abstract: Logic Gate Networks (LGNs) compute through compositions of Boolean operations, yet existing LGNs do not reliably benefit from increased depth. We identify two causes: optimization collapse and topology-induced degradation of output-specific credit that persists even after skip-biased initialization and straight-through estimation stabilize training.   We introduce Input-Anchored Logic Gate Networks (IALGNs), in which each gate combines a private hidden spine with a direct input anchor. This topology prevents output-path merging while retaining input access at every layer. Credit diagnostics show that random wiring dilutes or conflicts output-specific gradients, whereas IALGN maintains usable and coherent credit. Random-$k_x$ relaxation improves anchor selection without relaxing the spine.   Across MNIST, CIFAR-10, and CIFAR-100, IALGN exhibits consistent fixed-width depth--accuracy scaling up to 150 layers, while alternative topologi
    
[^141]: 基于连续参数空间高斯混合Wasserstein-2模糊集的鲁棒机会约束优化

    Robust Chance-Constrained Optimization using a Continuous Parameter Space Wasserstein-2 Ambiguity Set of Gaussian Mixtures

    [https://arxiv.org/abs/2607.17018](https://arxiv.org/abs/2607.17018)

    该论文提出了一种基于连续参数空间的Wasserstein-2模糊集新方法，用于高斯混合模型下的分布鲁棒机会约束优化，允许最坏情况分布内生地确定混合分量的数量与质量分配，克服了传统有限支撑方法只能对拟合的名义混合模型进行压力测试、无法应对结构性误设的局限。

    

    我们研究了分布鲁棒线性机会约束问题，其中不确定性由高斯混合模型（GMM）建模。有限支撑分布鲁棒（FDR）公式在数据驱动的鲁棒优化中被广泛使用，它在经验混合支撑点上进行鲁棒化，因此主要对拟合的名义混合模型进行压力测试。当服务可靠性依赖于名义混合-支撑参数的结构性误设时，这种方法可能是不够的。为了解决这一局限性，我们通过开发一种新颖的Wasserstein-2度量公式来描述分布的模糊集，该公式在具有有限二阶矩的概率测度上使用Bures-Wasserstein（BW）度量。与FDR通常先验地设置有限个经验支撑点不同，所提出的模糊集允许最坏情况分布内生地确定有多少个混合分量获得质量以及……（摘要截断）

    arXiv:2607.17018v2 Announce Type: replace-cross  Abstract: We study distributionally robust linear chance-constrained problems in which uncertainty is modeled by a Gaussian mixture model (GMM). Finite-support distributionally robust (FDR) formulations, widely used in data-driven robust optimization, robustify over empirical mixture support points and therefore primarily stress-test the fitted nominal mixture. This can be insufficient when service reliability depends on structural misspecification of the nominal mixture-support parameters. To address this limitation, we describe the ambiguity set of distributions by developing a novel formulation of a Wasserstein-2 metric that uses the Bures-Wasserstein (BW) metric over probability measures with finite second moments. Unlike FDR, which generally sets finitely many empirical support points a priori, the proposed ambiguity set allows the worst-case distribution to endogenously determine both how many mixture components receive mass and wh
    
[^142]: 面向离散马尔可夫随机场的端到端量子-经典混合采样工作流：一项可复现的案例研究

    An End-to-End Hybrid Quantum--Classical Sampling Workflow for Discrete Markov Random Fields: A Reproducible Case Study

    [https://arxiv.org/abs/2607.09893](https://arxiv.org/abs/2607.09893)

    该论文构建了一个可复现的端到端量子-经典混合采样工作流，实证表明现代经典MCMC采样器能大幅缩小与量子采样的差距，且在计入$O(2^n)$预处理成本后，量子方法在实际运行时间上并无优势。

    

    从离散马尔可夫随机场（MRF）中采样是一个困难问题。我们研究了针对小型MRF的振幅编码独立同分布采样，其中$2^n$个目标概率通过经典方法预先计算。这消除了量子指数级加速的可能性，但使得能够基于独立电路样本（$\tau \approx 1$）与经典MCMC进行清晰的对比。在跨越五个图族的60个实例上（1k步预热，保留3k个样本），量子方法相对于单点Gibbs、块Gibbs、调优块采样和并行回火的平均有效样本量（ESS）比率分别为16.35、7.29、1.82和1.79，表明现代经典采样器已大幅缩小了这一差距。将$O(2^n)$预处理成本摊销到实际运行时间中，精确的逆CDF采样可达到1770万ESS/s，而量子采样器仅为48.8万ESS/s（平均速率相差36倍，单实例最高相差153倍），证实量子方法在实际运行时间上并无优势。我们刻画了MCMC自相关成本并进行了基准测试。

    arXiv:2607.09893v2 Announce Type: replace-cross  Abstract: Sampling from discrete Markov random fields (MRFs) is a hard problem. We study amplitude-encoded i.i.d. sampling for small MRFs where $2^n$ target probabilities are precomputed classically. This removes quantum exponential speedup but allows a clean comparison against classical MCMC based on independent circuit samples ($\tau \approx 1$). Across 60 instances spanning five graph families (1k-step burn-in, 3k retained samples), the mean ESS ratios of Quantum to Single-Site Gibbs, Block Gibbs, Tuned-Block, and Parallel Tempering are $16.35$, $7.29$, $1.82$, and $1.79$, showing modern classical samplers substantially close this gap. Amortizing $O(2^n)$ preprocessing into wall-clock time, exact inverse-CDF sampling yields $17.7\text{M}$ ESS/s versus $488\text{K}$ ESS/s for the quantum sampler ($36\times$ mean rate, $153\times$ per-instance), confirming no wall-clock advantage. We characterize MCMC autocorrelation costs and benchmark
    
[^143]: SpecGradFilter：一个用于驯服联邦异构性的频谱梯度过滤框架

    SpecGradFilter: A Spectral Gradient Filtering Framework for Taming Federated Heterogeneity

    [https://arxiv.org/abs/2607.04189](https://arxiv.org/abs/2607.04189)

    提出频谱梯度过滤框架SpecGradFilter，从频域视角揭示联邦学习客户端漂移主要集中于低频梯度分量的“漂移频谱偏差”，并通过抑制不协调的低频信号来驯服统计异构性。

    

    联邦学习（FL）从根本上受到统计异构性的挑战，其中非独立同分布（non-IID）的数据会引起客户端漂移，严重阻碍全局收敛。虽然现有方法试图通过空间域梯度校正或正则化来缓解这种漂移，但它们忽视了优化信号的内在频谱结构。在本工作中，我们从新颖的频域视角重新审视客户端漂移，并揭示了一个关键的漂移频谱偏差：客户端间的梯度发散主要集中于编码客户端特定分布偏移的低频分量，而代表细粒度特征的高频分量则保持相对一致。受此启发，我们提出了SpecGradFilter，一个统一的频谱梯度过滤框架，通过抑制不协调的低频信号来驯服异构性。至关重要的是，我们证明……（摘要原文在此处被截断）

    arXiv:2607.04189v2 Announce Type: replace  Abstract: Federated Learning (FL) is fundamentally challenged by statistical heterogeneity, where non-identically distributed (non-IID) data induces client drift that severely hampers global convergence. While existing approaches attempt to mitigate this drift through spatial-domain gradient correction or regularization, they overlook the intrinsic spectral structure of optimization signals. In this work, we revisit client drift from a novel frequency-domain perspective and uncover a critical Spectral Bias of Drift: inter-client gradient divergence is predominantly concentrated in low-frequency components which encode client-specific distributional shifts, while high-frequency components representing fine-grained features remain relatively consistent. Motivated by this, we propose SpecGradFilter, a unified Spectral Gradient Filtering Framework that tames heterogeneity by suppressing discordant low-frequency signals. Crucially, we demonstrate t
    
[^144]: 弥合语义缓存中的运营差距

    Closing the Operational Gap in Semantic Caching

    [https://arxiv.org/abs/2606.19719](https://arxiv.org/abs/2606.19719)

    该论文指出PR-AUC指标会误导语义缓存系统的部署决策，提出了缓存感知的P-CHR AUC指标和运营保留率ORR，并将离线与部署质量间的运营差距分解为可恢复的阈值效用部分和由数据集正例率决定的不可约简结构部分。

    

    语义缓存通过为语义相似的查询提供缓存响应来降低大语言模型（LLM）的推理成本。标准做法是使用PR-AUC来评估这些系统，但该指标仅衡量分数的排序质量，而忽略了分数在固定阈值下是否可用。我们证明这种错位会导致系统性的糟糕部署选择，因为PR-AUC最高的模型在实际运行中往往表现最差。我们引入了精确率-缓存命中率（P-CHR）AUC这一缓存感知指标，用于衡量不同缓存利用率水平下的精确率；以及运营保留率（ORR），用于捕捉离线排序质量在部署时的保留程度。我们将离线质量与部署质量之间的运营差距分解为可恢复的阈值效用部分，以及由数据集正例率固定的不可约简的结构部分。我们的实验表明，阈值效用差距由训练目标决定，而非……（摘要原文在此处截断）

    arXiv:2606.19719v3 Announce Type: replace-cross  Abstract: Semantic caching cuts LLM inference costs by serving a cached response to semantically similar queries. Standard practice evaluates these systems using PR-AUC, a metric that only measures how well scores rank and ignores whether they are usable at a fixed threshold. We show this mismatch leads to systematically poor deployment choices, as models with the highest PR-AUC are often the worst in operation. We introduce Precision--Cache Hit Ratio (P-CHR) AUC, a cache-aware metric that measures precision across cache utilization levels, and Operational Retention Rate (ORR), which captures how much offline ranking quality survives at deployment. We decompose the operational gap between offline and deployed quality into a recoverable threshold-utility component and an irreducible structural component fixed by the dataset's positive rate. Our experiments show that the threshold-utility gap is governed by the training objective rather th
    
[^145]: 离散对数时钟：Transformer如何学习模乘法

    The Discrete-Log Clock: How a Transformer Learns Modular Multiplication

    [https://arxiv.org/abs/2606.17399](https://arxiv.org/abs/2606.17399)

    该论文发现Transformer学习模乘法时嵌入频谱的“稠密性”是在错误基下分析的伪象，改用乘法特征变换后频谱高度稀疏，且96.9%的MLP神经元被调谐到单一乘法频率。

    

    当小型Transformer对模乘法实现“顿悟”（grokking）时，先前的研究报告称学习到的嵌入具有需要所有频率的“稠密”傅里叶频谱。这与模加法形成对比，后者只需一组稀疏的关键频率即可。我们证明这种稠密性是在错误基下进行分析所产生的伪象。乘法的自然傅里叶变换并非标准的加性DFT，而是乘法特征变换，它将乘法群 $(\mathbb{Z}/p\mathbb{Z})^*$ 上的函数分解为其不可约表示。将此变换应用于在 $a \cdot b \bmod 113$ 上训练并出现顿悟现象的Transformer，我们发现嵌入频谱变得高度稀疏（基尼系数为0.58，而加性基下为0.07），仅有4个关键频率携带显著能量。此外，96.9%的MLP神经元被清晰地调谐到单一的乘法频率，且神经元激活热图揭示了二维周期性结构。

    arXiv:2606.17399v2 Announce Type: replace  Abstract: When small transformers grok modular multiplication, prior work reports that the learned embedding has a "dense" Fourier spectrum requiring all frequencies. This contrasts with modular addition, where only a sparse set of key frequencies suffices. We show this density is an artifact of analyzing in the wrong basis. The natural Fourier transform for multiplication is not the standard additive DFT but the multiplicative character transform, which decomposes functions on the multiplicative group $(\mathbb{Z}/p\mathbb{Z})^*$ into its irreducible representations. Applying this transform to a grokked transformer trained on $a \cdot b \bmod 113$, we find the embedding spectrum becomes highly sparse (Gini coefficient 0.58 vs. 0.07 in the additive basis) with only 4 key frequencies carrying significant energy. Furthermore, 96.9% of MLP neurons are cleanly tuned to a single multiplicative frequency, and neuron activation heatmaps reveal 2D-per
    
[^146]: TokenPilot：面向大语言模型智能体的缓存高效上下文管理

    TokenPilot: Cache-Efficient Context Management for LLM Agents

    [https://arxiv.org/abs/2606.17016](https://arxiv.org/abs/2606.17016)

    TokenPilot提出了一种双粒度上下文管理框架，通过全局的感知摄取压缩来稳定提示前缀并消除环境噪声，结合局部的生命周期感知驱逐机制仅在任务相关性过期时卸载内容，从而在降低大语言模型智能体推理成本的同时保持提示缓存的连续性。

    

    随着大语言模型智能体被部署在长时程会话中，上下文的不断累积推高了推理成本。现有方法利用文本剪枝或动态内存驱逐来最小化令牌占用；然而，它们不受约束的序列变更会改变布局，引入前缀不匹配和缓存失效问题。这揭示了文本稀疏性与提示缓存连续性之间的关键权衡。为解决这一问题，我们提出了TokenPilot，一个双粒度的上下文管理框架。在全局层面，感知摄取的压缩机制充当框架治理工具，在摄取入口处稳定提示前缀并消除开放世界的环境噪声。在局部层面，生命周期感知的驱逐机制监控上下文片段的持续残余效用，执行保守的批轮次调度，仅在任务相关性过期时才卸载内容片段。在PinchBench和Claw-Eval基准上以隔离和连续两种模式进行的实验证明了……（原文摘要在此处截断）

    arXiv:2606.17016v2 Announce Type: replace  Abstract: As LLM agents are deployed in long-horizon sessions, context accumulation drives up inference costs. Existing approaches utilize text pruning or dynamic memory eviction to minimize token footprints; however, their unconstrained sequence mutations alter layouts, introducing prefix mismatches and cache invalidation. This reveals a critical trade-off between text sparsity and prompt cache continuity. To address this, we present TokenPilot, a dual-granularity context management framework. Globally, Ingestion-Aware Compaction acts as a framework harness to stabilize prompt prefixes and eliminate open-world environmental noise at the ingestion gate. Locally, Lifecycle-Aware Eviction monitors the ongoing residual utility of context segments, enforcing a conservative batch-turn schedule to offload content segments only when task relevance expires. Experiments on PinchBench and Claw-Eval under both isolated and continuous modes demonstrate th
    
[^147]: RecourseBench：一个用于可复现算法追索评估的模块化框架

    RecourseBench: A Modular Framework for Reproducible Algorithmic Recourse Evaluation

    [https://arxiv.org/abs/2606.16113](https://arxiv.org/abs/2606.16113)

    RecourseBench是一个以模块化、可复现性和交互性为核心的算法追索统一评估框架，它将评估流程解耦为五个独立层次，并对所有集成方法进行可复现性分类及其核心实证结论的系统性验证。

    

    算法追索方法通过反事实解释告知个体推翻不利模型决策所需的行动。尽管方法论进展迅速，但原则性的比较仍然难以实现；现有框架往往难以扩展，既缺乏互操作性，也缺乏对集成方法能否忠实复现其原始报告结论的系统性验证。我们提出了RecourseBench，这是一个围绕三大承诺构建的统一评估框架：模块化、可复现性和交互性。该框架将流程分解为五个完全解耦的层——数据、预处理、模型、追索方法和评估——由抽象接口和动态注册表进行管理。每个集成方法都根据工件可用性被归类到四层可复现性分类体系中，随后对其核心实证结论进行系统性验证。

    arXiv:2606.16113v2 Announce Type: replace-cross  Abstract: Algorithmic recourse methods provide counterfactual explanations that inform individuals of the actions required to overturn an unfavorable model decision. Despite rapid methodological progress, principled comparison remains elusive; existing frameworks are often difficult to extend and lack both interoperability and systematic verification that integrated methods faithfully reproduce their originally reported claims. We introduce RecourseBench, a unified evaluation framework built around three commitments: modularity, reproducibility, and interactivity. The framework decomposes the pipeline into five fully decoupled layers---Data, Preprocessing, Model, Recourse Method, and Evaluation---governed by abstract interfaces and a dynamic registry. Every integrated method is classified into a four-tier reproducibility taxonomy based on artifact availability, followed by a systematic verification of its core empirical claims. We furthe
    
[^148]: ToolSense：一个用于审计大语言模型参数化工具知识的诊断框架

    ToolSense: A Diagnostic Framework for Auditing Parametric Tool Knowledge in LLMs

    [https://arxiv.org/abs/2606.12451](https://arxiv.org/abs/2606.12451)

    ToolSense是一个开源的LLM驱动诊断框架，通过自动生成包含真实查询在内的三个基准来审计大语言模型的参数化工具知识，揭示了模型在标准ToolBench基准上的优异表现并不能证明其真正理解工具。

    

    作为智能体部署在大型工具目录上的大语言模型面临关键的工具检索瓶颈。由于基于嵌入的检索方法依赖紧凑的编码器，可能无法充分捕捉专业化的工具语义，参数化工具检索通过将每个工具编码为附加到LLM词表中的虚拟token来解决这一问题，并采用两阶段微调（先记忆、后检索的有监督微调）将LLM用作检索器，在标准ToolBench检索基准上取得了出色性能。然而，这些基准使用冗长且完全指定的查询，其评估还采用受限解码将输出限制在有效的token路径上，两者都无法揭示模型是否真正理解其工具。我们提出ToolSense，一个开源的、由LLM驱动的诊断框架，它以任意工具目录为输入，自动生成三个基准：包括带有真实查询的现实检索基准（RRB）……

    arXiv:2606.12451v2 Announce Type: replace-cross  Abstract: Large language models deployed as agents over large tool catalogs face a critical tool-retrieval bottleneck. As embedding-based retrieval approaches rely on compact encoders that may under-capture specialized tool semantics, parametric tool retrieval addresses this by encoding each tool as a virtual token appended to the LLM vocabulary, fine-tuned in two stages (memorization then retrieval SFT) to use the LLM as a retriever, achieving strong performance on standard ToolBench retrieval benchmarks. Yet these benchmarks use verbose, fully-specified queries, and their evaluation applies constrained decoding that restricts outputs to valid token paths, neither reveals whether the model actually understands its tools. We introduce \textbf{ToolSense}, an open-source LLM-powered diagnostic framework that takes any tool catalog as input and automatically generates three benchmarks: a Realistic Retrieval Benchmark (RRB) with queries at t
    
[^149]: ERP-XTTN：面向跨被试ERP分类的可解释原型引导交叉注意力方法

    ERP-XTTN: Interpretable Prototype-Guided Cross-Attention for Cross-Subject ERP Classification

    [https://arxiv.org/abs/2606.02939](https://arxiv.org/abs/2606.02939)

    提出ERP-XTTN架构，通过无值投影的查询-键交叉注意力将EEG峰值路由到从差异波自动提取的原型上，实现了无需校准、跨被试且内在可解释的跨范式ERP分类。

    

    无需校准即可跨被试泛化的可解释脑机接口分类器仍然是一个悬而未决的挑战。我们评估了基于原型的交叉注意力能否在部署兼容条件下，跨范式提供具有竞争力且内在可解释的ERP分类。我们提出了ERP-XTTN（ERP交叉注意力），这是一种交叉注意力架构，通过仅使用查询-键（query-key-only）的交叉注意力（不含值投影）将输入的EEG峰值路由到固定的差异波原型。分类直接基于原型相似性以及一个独立的成分幅值度量，因此原型内容在结构上必然参与每一次决策。原型自动地从训练折的总体平均差异波中的显著极值点提取。我们在三个公开数据源（BNCI Horizon 2020、HRI Cursor和ERP CORE）上进行了评估，涵盖八种ERP成分（ERN、LRP、ErrP、N170、P300、N……（摘要原文在此处截断）。

    arXiv:2606.02939v2 Announce Type: replace  Abstract: Interpretable brain-computer interface classifiers that generalize across subjects without calibration remain an open challenge. We evaluated whether prototype-based cross-attention can provide competitive, inherently interpretable ERP classification across paradigms under deployment-compatible conditions. We propose ERP-XTTN (ERP Cross-Attention), a cross-attention architecture that routes input EEG peaks to fixed difference-wave prototypes via query-key-only cross-attention with no value projection. Classification is based directly on prototype similarity and a separate measure of component amplitude, so that prototype content contributes to every decision by construction. Prototypes are derived automatically from prominent extrema in the training-fold grand-average difference wave. We evaluated across three public sources (BNCI Horizon 2020, HRI Cursor, and ERP CORE) encompassing eight ERP components (ERN, LRP, ErrP, N170, P300, N
    
[^150]: 子图解释能否被武器化以窃取图神经网络？

    Can Subgraph Explanations Be Weaponized to Steal Graph Neural Networks?

    [https://arxiv.org/abs/2605.30470](https://arxiv.org/abs/2605.30470)

    图机器学习即服务平台中的可解释性接口可被武器化，攻击者仅凭离散类别标签和二值解释掩码即可在严格黑盒约束下窃取图神经网络模型。

    

    图机器学习即服务（GMLaaS）平台日益增加地实现可解释性接口以满足监管透明度要求。然而，这种透明度为模型提取攻击创造了可利用的漏洞。我们提出了首个专门针对图分类任务的模型提取攻击，适用于严格的黑盒约束场景——攻击者只能观察到离散的类别标签和二值解释掩码（没有概率分数、梯度或置信度值）。我们的方法（1）利用模型解释输出引导蒙特卡洛边敏感度估计向决策边界推进，并在估计精度上提供Hoeffding集中性保证；（2）利用解释子图高效缩小边界搜索空间。在跨多个领域的基准图数据集上进行的大量实验证明了我们的方法优于同类基线方法。这些发现表明……

    arXiv:2605.30470v2 Announce Type: replace  Abstract: Graph Machine Learning as a Service (GMLaaS) platforms increasingly implement explainability interfaces to meet regulatory transparency requirements. However, this transparency creates exploitable vulnerabilities for model extraction attacks. We present the first model extraction attack specifically designed for graph classification under strict black-box constraints where the attacker observes only discrete class labels and binary explanation masks (no probability scores, gradients, or confidence values). Our method (1) uses model explanation outputs to guide Monte Carlo edge sensitivity estimation toward decision boundaries, with Hoeffding concentration guarantees on estimation accuracy and (2) exploits explanation subgraphs to efficiently narrow the boundary search space. Extensive experiments on benchmark graph datasets across multiple domains demonstrate our method's superiority over comparable baselines. These findings demonstr
    
[^151]: LongDS-Bench：论长时程智能体数据分析的失败

    LongDS-Bench: On the Failure of Long-Horizon Agentic Data Analysis

    [https://arxiv.org/abs/2605.30434](https://arxiv.org/abs/2605.30434)

    提出LongDS基准，基于真实Kaggle笔记本构建68个长时程多轮数据分析任务，揭示当前最先进模型在维护和演变分析状态方面存在严重缺陷，最佳模型平均准确率仅48.45%，且长时程错误占失败案例的52%–69%。

    

    现实世界的数据分析本质上是迭代式的，然而现有的基准测试大多评估孤立的或短交互的任务，未能检验智能体在长时程中跟踪不断演变的分析上下文的能力。我们提出了LongDS，这是一个面向长时程、多轮数据分析的基准，要求智能体维护、更新、恢复和组合不断演变的分析状态。LongDS包含从真实Kaggle笔记本构建的68个任务，涵盖地球科学、商业和教育等六个领域，共2,225轮。任务围绕状态演变模式（例如反事实扰动、回滚、多状态组合）设计，平均依赖跨度为11.3轮。通过对五个最先进模型的评估，我们发现表现最好的模型平均准确率仅为48.45%，性能从早期到后期轮次下降近47个百分点，长时程错误占失败案例的52%–69%。

    arXiv:2605.30434v2 Announce Type: replace-cross  Abstract: Real-world data analysis is inherently iterative, yet existing benchmarks mostly evaluate isolated or short interactive tasks, leaving agents' ability to track evolving analytical context over long horizons untested. We introduce LongDS, a benchmark for long-horizon, multi-turn data analysis where agents must maintain, update, restore, and compose evolving analytical states. LongDS comprises 68 tasks constructed from real-world Kaggle notebooks, spanning 2,225 turns across six domains including Geoscience, Business, and Education. Tasks are designed around state-evolution patterns (e.g., counterfactual perturbation, rollback, multi-state composition), with an average dependency span of 11.3 turns. Evaluating five state-of-the-art models, we find that the best model reaches only 48.45% average accuracy, performance drops nearly 47 points from early to late turns, and long-horizon errors account for 52%--69% of failures. Further 
    
[^152]: 微不足道的规模，举足轻重的作用：论大语言模型中的缩放向量

    Negligible in Size, Significant in Effect: On Scale Vectors in Large Language Models

    [https://arxiv.org/abs/2605.26895](https://arxiv.org/abs/2605.26895)

    缩放向量虽仅占大语言模型参数的极小部分，但并非用于增强表达能力，而是通过自放大预条件效应改善优化过程，对模型预训练效果至关重要。

    

    现代大语言模型（LLM）中的归一化层由一个确定性的归一化操作和一个可学习的缩放向量组成。尽管归一化操作已被广泛研究，但缩放向量虽然被普遍使用，却仍然缺乏深入理解。在这项工作中，我们从表达能力、优化和架构结构的角度对大语言模型中的缩放向量进行了系统研究。首先，我们通过实验证明，尽管缩放向量仅占模型参数中极小的一部分，但移除它们会显著降低大语言模型的预训练效果。我们的理论进一步表明，在Pre-Norm架构中，缩放向量并不能增加模型的表达能力；相反，它们通过对后续线性映射的自放大预条件效应来改善优化过程。其次，我们研究了权重衰减对缩放向量的作用。通过区分Input-Norm和Output-Norm层，我们从理论上……

    arXiv:2605.26895v2 Announce Type: replace  Abstract: Normalization layers in modern large language models (LLMs) consist of a deterministic normalization operation and a learnable scale vector. While the normalization operation has been extensively studied, the scale vector remains poorly understood despite its ubiquitous use. In this work, we present a systematic study of scale vectors in LLMs from the perspectives of expressivity, optimization, and architectural structure. First, we show empirically that although scale vectors constitute only a negligible fraction of model parameters, removing them substantially degrades LLM pre-training. Our theory further shows that, in Pre-Norm architectures, scale vectors do not increase expressivity; instead, they improve optimization through a self-amplifying preconditioning effect on subsequent linear mappings. Second, we investigate the role of weight decay for scale vectors. By distinguishing Input-Norm and Output-Norm layers, we theoretical
    
[^153]: 更具表达力的前馈层：第一部分：令牌自适应的激活混合

    More Expressive Feedforward Layers: Part I. Token-Adaptive Mixing of Activations

    [https://arxiv.org/abs/2605.26647](https://arxiv.org/abs/2605.26647)

    本文提出令牌自适应的激活混合（MoA）前馈层设计，通过轻量级输入相关门控混合多个激活函数，并从理论上证明了其表达能力严格超越可学习激活（LA）和固定激活FFN。

    

    前馈网络（FFN）层在基于Transformer的大语言模型（LLM）中占据了相当大的参数比例和非线性表达能力。尽管激活函数已从ReLU和GELU演进到SwiGLU等门控变体，但大多数FFN设计仍然使用单一固定的激活函数，对所有令牌（token）应用相同的非线性变换。在这项工作中，我们提出了激活混合（Mixture of Activations, MoA），这是一种令牌自适应的FFN设计，它使用轻量级的、依赖于输入的门控机制来混合一个激活函数字典，同时共享相同的线性投影。作为输入无关的对应方案，我们还引入了可学习激活（Learnable Activations, LA），它为ReLU型和SwiGLU型FFN构造激活函数的线性组合。在理论方面，我们在固定激活FFN、LA和MoA之间建立了严格的有限宽度表达能力分离关系：LA严格包含固定激活FFN的表达能力，而MoA又严格包含LA，并进一步……

    arXiv:2605.26647v2 Announce Type: replace  Abstract: Feedforward network (FFN) layers account for a large fraction of parameters and nonlinear expressivity in Transformer-based large language models (LLMs). Despite the evolution from ReLU and GELU to gated variants such as SwiGLU, most FFN designs still use a single fixed activation function, applying the same nonlinear transformation to all tokens. In this work, we propose Mixture of Activations (MoA), a token-adaptive FFN design that mixes a dictionary of activation functions using lightweight input-dependent gates while sharing the same linear projections. As an input-independent counterpart, we also introduce learnable activations (LA), which form linear combinations of activation functions for both ReLU-type and SwiGLU-type FFNs. Theoretically, we establish strict finite-width expressive separations among fixed-activation FFNs, LA, and MoA: LA strictly contains fixed-activation FFNs, while MoA strictly contains LA, with the additi
    
[^154]: WINO：一种面向可变域超弹性问题的弱形式物理信息神经算子

    WINO: A Weak-Form Physics Informed Neural Operator for Hyperelasticity on Variable Domains

    [https://arxiv.org/abs/2605.24651](https://arxiv.org/abs/2605.24651)

    该论文提出WINO——一种无需数据的弱形式物理信息神经算子框架，将神经算子的高效性与φ-FEM的几何灵活性相结合，通过最小化弱形式残差进行训练，能够在无需贴体网格和大规模参考解数据集的情况下求解可变几何域上的超弹性问题。

    

    我们提出了一种弱形式物理信息神经算子（WINO），这是一个无需数据的框架，它将神经算子的高效性与φ-有限元方法（φ-FEM）的几何灵活性相结合。φ-FEM是一种非贴体方法，无需贴体网格即可适应几何变化，其中区域几何由水平集函数φ表示。为了施加边界条件，Dirichlet问题采用φ-FEM提升技术，因此只需学习齐次位移贡献；而对于由牵引力驱动的Neumann问题，则额外预测非贴体弱形式所需的辅助场。模型参数通过最小化与φ-FEM对齐的平方弱形式残差，以及对切割单元辅助方程施加平方惩罚来进行训练，从而无需大规模成对的收敛参考解数据集。当存在标注的参考（摘要在此处截断）……

    arXiv:2605.24651v3 Announce Type: replace-cross  Abstract: We propose a Weak-form Physics-Informed Neural Operator (WINO), a data-free framework that combines the efficiency of neural operators with the geometric flexibility of the $\varphi$-finite element method ($\varphi$-FEM). $\varphi$-FEM is an unfitted method that accommodates geometric variations without body-fitted meshes, where the domain geometry is represented by the level-set function $\varphi$. To impose the boundary conditions, Dirichlet problems adopt the $\varphi$-FEM lifting so only the homogeneous displacement contribution is learned, whereas traction-driven Neumann problems additionally predict the auxiliary fields necessary for the unfitted weak formulation. Parameters are trained by minimizing squared weak-form residuals aligned with $\varphi$-FEM together with squared penalties on the cut-cell auxiliary equations, which removes the need for large paired datasets of converged reference solutions. When labeled refer
    
[^155]: 面向前瞻性离散扩散模型的学习式接力表示

    Learned Relay Representations for Forward-Thinking Discrete Diffusion Models

    [https://arxiv.org/abs/2605.22967](https://arxiv.org/abs/2605.22967)

    提出Relay方法，通过可学习的逐token通道在去噪步骤之间传递潜在信息，使掩码扩散模型具备前瞻性，避免每轮去噪后的信息硬重置，并可扩展至最先进的扩散语言模型。

    

    当掩码扩散模型通过迭代精炼生成序列时，掩码位置上丰富的内部计算结果会被丢弃，迫使每个后续的精炼步骤重新计算以模型表示形式存储的宝贵内部信息。为了避免去噪轮次之间的硬重置，我们提出了学习式接力表示，这是一种让掩码扩散模型在去噪时具备前瞻性的方法，通过显式学习如何传播潜在信息以造福未来的去噪步骤。Relay引入了一个可微分的逐token通道，用于在前向传递之间传递信息，并通过截断的随时间反向传播（BPTT）进行训练。我们证明该框架可以扩展到最先进的扩散语言模型，并且与块扩散和KV缓存等技术无缝兼容。我们首先对Relay中的设计选择提供了详尽的论证……

    arXiv:2605.22967v3 Announce Type: replace  Abstract: When Masked Diffusion Models (MDMs) generate sequences through iterative refinement, the rich internal computation over masked positions is discarded, forcing every subsequent refinement step to recompute the valuable internal information stored as model representations. To avoid a hard reset between denoising rounds, we propose Learned Relay Representations (Relay), a method that allows MDMs to be forward-thinking when denoising by explicitly learning how to propagate latent information for the benefit of future denoising steps. Relay introduces a differentiable per-token channel that passes information between forward passes and is trained via truncated backpropagation through time (BPTT). We show that this framework can be scaled to state-of-the-art Diffusion Language Models (DLMs), and is seamlessly compatible with techniques like block diffusion and KV caching. We first provide a thorough justification of the design choices in R
    
[^156]: ImplicitTerrainV2：小波引导的空间自适应神经地形表示

    ImplicitTerrainV2: Wavelet-Guided Spatially Adaptive Neural Terrain Representation

    [https://arxiv.org/abs/2605.22556](https://arxiv.org/abs/2605.22556)

    本文提出了ImplicitTerrainV2，一种结合小波引导空间自适应、导数感知监督和模型压缩的紧凑高效神经地形表示方法，解决了现有地形INRs在频率控制、梯度结构和部署成本上的不足。

    

    数字高程模型（DEM）支撑着地理信息系统（GIS）中的地形分析，但通常作为栅格表示，它们依赖插值进行离网格采样，并依赖有限差分算子进行基于导数的分析。隐式神经表示（INRs）提供了一种连续的替代方案，但先前的地形INRs缺乏显式频率控制，忽略了地形的梯度结构，并且对于实际部署而言仍然过大且训练成本高昂。我们提出了ImplicitTerrainV2，通过结合频谱控制机制与小波引导的空间自适应性、导数感知监督以及训练后模型压缩，将地形INRs推向一种紧凑、高效的神经地形数据格式。其核心在于，一个小波复杂度场（WCF）从解析计算的小波系数中推导出空间自适应频率掩码，将高频能力定位到复杂地形区域。

    arXiv:2605.22556v2 Announce Type: replace  Abstract: Digital elevation models (DEMs) underpin terrain analysis in Geographic Information Systems (GIS), but commonly as raster representation, they rely on interpolation for off-grid sampling and finite-difference operators for derivative-based analysis. Implicit neural representations (INRs) offer a continuous alternative, but prior terrain INRs lack explicit frequency control, neglect the gradient structure of terrain, and remain too large and costly to train for practical deployment. We present ImplicitTerrainV2, which advances terrain INRs toward a compact, efficient neural terrain data format by combining a spectral control mechanism with wavelet-guided spatial adaptivity, derivative-aware supervision, and post-training model compression. At its core, a wavelet complexity field (WCF) derives spatially-adaptive frequency masks from analytically computed wavelet coefficients, localizing high-frequency capacity to complex terrain region
    
[^157]: 面向动态变化专家的在线学习延迟决策

    Online Learning-to-Defer with Varying Experts

    [https://arxiv.org/abs/2605.12340](https://arxiv.org/abs/2605.12340)

    本文提出了一种将查询动作的老虎机反馈与动态变化专家池相结合的在线多分类学习延迟算法，实现了次线性真实延迟遗憾 $O(T^{2/3})$，并在集中评分条件下提升至 $O(\sqrt T)$。

    

    学习延迟（Learning-to-Defer, L2D）方法将每个查询要么路由给预测模型，要么路由给外部专家。现实世界的部署需要处理流式数据、不断变化的专家可用性、专家可靠性的漂移，以及仅针对所选动作才能观测到的反馈。我们提出了一种在线多分类L2D算法，该算法将查询动作的老虎机反馈与动态变化的专家池相结合。设 $N=n+n_e$，设 $B$ 为线性评分矩阵的 Frobenius 范数的上界，$\rho$ 为增广输入范数的上界。在线性校准以及投影比较器类上代理最小化间隙为零的假设下，我们的方法达到了期望真实延迟遗憾 $O((BN^{3/2}\rho+1)T^{2/3})$，并在集中评分条件下改进为 $O(BN^{3/2}\rho\sqrt T+B^2N^3\rho^2)$。该分析将在线 $\mathcal H$-一致性转移界与投影在线凸优化相结合。在合成数据（摘要在此处被截断）……

    arXiv:2605.12340v5 Announce Type: replace-cross  Abstract: Learning-to-Defer (L2D) methods route each query either to a predictive model or to external experts. Real-world deployments require handling streaming data, changing expert availability, shifting expert reliability, and feedback observed only for the selected action. We introduce an online multiclass L2D algorithm that combines queried-action bandit feedback with a dynamically varying pool of experts. Let $N=n+n_e$, let $B$ bound the Frobenius norm of the linear score matrix, and let $\rho$ bound the augmented input norm. Assuming linear calibration and zero surrogate minimizability gap for the projected comparator class, our method achieves expected true-deferral regret $O((BN^{3/2}\rho+1)T^{2/3})$, improving to $O(BN^{3/2}\rho\sqrt T+B^2N^3\rho^2)$ under a concentrated-score condition. The analysis combines an online $\mathcal H$-consistency transfer bound with projected online convex optimization. Experiments on synthetic a
    
[^158]: SkillSafetyBench：评估面向技能攻击面下的智能体安全性

    SkillSafetyBench: Evaluating Agent Safety under Skill-Facing Attack Surfaces

    [https://arxiv.org/abs/2605.12015](https://arxiv.org/abs/2605.12015)

    该论文提出 SkillSafetyBench 基准，通过 155 个覆盖 6 大风险领域的对抗性案例，首次系统评估了隐藏在技能指导、本地文件等非用户输入中的攻击面，发现此类攻击可稳定诱发大语言模型智能体的不安全行为。

    

    可复用技能正在成为扩展大语言模型智能体的常见接口，它将程序性指导与对文件、工具、记忆和执行环境的访问打包在一起。然而，这种模块化引入了现有安全评估大多忽视的攻击面：即使用户请求是良性的，不安全的影响也可能存在于技能指导、本地工件或执行环境文件中，从而引导智能体做出不安全的操作。我们提出了 SkillSafetyBench，一个用于评估此类面向技能安全故障的可运行基准。SkillSafetyBench 包含 155 个对抗性案例，涵盖 47 个任务、6 个风险领域和 30 个安全类别，每个案例均通过特定于案例的基于规则的验证器进行评估。使用多个 CLI 智能体和模型后端的实验表明，非用户攻击能够持续诱发不安全行为，且在不同领域、攻击方法和脚手架-模型组合下呈现出截然不同的失败模式。

    arXiv:2605.12015v3 Announce Type: replace-cross  Abstract: Reusable skills are becoming a common interface for extending large language model agents, packaging procedural guidance with access to files, tools, memory, and execution environments. However, this modularity introduces attack surfaces that are largely missed by existing safety evaluations: even when the user request is benign, unsafe influence may reside in skill guidance, local artifacts, or execution-environment files that steer the agent toward unsafe actions. We present SkillSafetyBench, a runnable benchmark for evaluating such skill-facing safety failures. SkillSafetyBench includes 155 adversarial cases across 47 tasks, 6 risk domains, and 30 safety categories, each evaluated with a case-specific rule-based verifier. Experiments with multiple CLI agents and model backends show that non-user attacks can consistently induce unsafe behavior, with distinct failure patterns across domains, attack methods, and scaffold-model 
    
[^159]: D3-Gym：为数据驱动发现构建真实世界可验证环境

    D3-Gym: Constructing Real-World Verifiable Environments for Data-Driven Discovery

    [https://arxiv.org/abs/2604.27977](https://arxiv.org/abs/2604.27977)

    本文提出了首个为科学数据驱动发现自动构建的可验证环境数据集D3-Gym，包含565个来自真实科学代码库的任务，其自动评估脚本与人工标注达到87.5%的一致性，且基于其轨迹训练可显著提升Qwen3模型在ScienceAgentBench上的表现。

    

    尽管面向科学数据驱动发现的语言模型和智能体近期取得了进展，但由于缺乏能够代表真实世界科学任务的可验证环境，其能力的提升受到了阻碍。为填补这一空白，我们推出了D3-Gym，这是首个为科学数据驱动发现自动构建的、带有可验证环境的数据集。D3-Gym包含来自四个学科、239个真实科学代码库的565个任务，每个任务均配有自然语言指令、预装依赖的可执行环境、数据集预览、参考解决方案以及自动合成的评估脚本。我们的评估脚本与人工标注的黄金标准达到87.5%的一致性，并在领域特定评估逻辑上表现出高度对齐。在从D3-Gym采样的轨迹上进行训练，使Qwen3系列模型在ScienceAgentBench上获得一致提升，其中Qwen3-32B绝对提升7.8个百分点（原文在此处截断）。

    arXiv:2604.27977v3 Announce Type: replace-cross  Abstract: Despite recent progress in language models and agents for scientific data-driven discovery, advancing their capabilities is held back by the absence of verifiable environments representing real-world scientific tasks. To fill this gap, we introduce D3-Gym, the first automatically constructed dataset with verifiable environments for scientific Data-Driven Discovery. D3-Gym comprises 565 tasks from 239 real scientific repositories across four disciplines, each with a natural language instruction, an executable environment with pre-installed dependencies, dataset previews, a reference solution, and an automatically synthesized evaluation script. Our evaluation scripts achieve 87.5% agreement with human-annotated gold standards and strong alignment in domain-specific evaluation logic. Training on trajectories sampled from D3-Gym yields consistent gains across Qwen3 models on ScienceAgentBench, boosting Qwen3-32B by 7.8 absolute poi
    
[^160]: ABC：基于连续时间与空间中非马尔可夫扩散桥的任意子集自回归

    ABC: Any-Subset Autoregression via Non-Markovian Diffusion Bridges in Continuous Time and Space

    [https://arxiv.org/abs/2604.27443](https://arxiv.org/abs/2604.27443)

    提出ABC模型，通过连续时空中的非马尔可夫扩散桥构建单一连续SDE，使时间变量与中间状态跟踪真实物理时间，从而实现以任意状态子集（如不规则采样或未来观测）为条件的连续时间连续空间随机过程生成。

    

    生成连续时间、连续空间的随机过程（例如视频、天气预报），并以部分观测（例如首帧和末帧）为条件，是一个基础性挑战。现有方法（例如扩散模型）存在几个关键局限：（1）从噪声到数据的演化无法捕捉物理时间上相近状态之间的结构相似性，且在低步数情况下积分不稳定；（2）注入的随机噪声对物理过程所经过的真实时间不敏感，导致错误的动力学特性；（3）它们忽视了以任意状态子集为条件（例如不规则采样的时间步、未来观测）。我们提出ABC：基于连续时间与空间中非马尔可夫扩散桥的任意子集自回归模型。关键在于，我们用一个连续的随机微分方程（SDE）对该过程建模，其时间变量和中间状态能够跟踪真实时间和过程状态。这具有可证明的优势：（1

    arXiv:2604.27443v3 Announce Type: replace  Abstract: Generating continuous-time, continuous-space stochastic processes (e.g., videos, weather forecasts) conditioned on partial observations (e.g., first and last frames) is a fundamental challenge. Existing approaches, (e.g., diffusion models), suffer from key limitations: (1) noise-to-data evolution fails to capture structural similarity between states close in physical time and has unstable integration in low-step regimes; (2) random noise injected is insensitive to the physical process's time elapsed, resulting in incorrect dynamics; (3) they overlook conditioning on arbitrary subsets of states (e.g., irregularly sampled timesteps, future observations). We propose ABC: Any-Subset Autoregressive Models via Non-Markovian Diffusion Bridges in Continuous Time and Space. Crucially, we model the process with one continual SDE whose time variable and intermediate states track the real time and process states. This has provable advantages: (1
    
[^161]: DiffAnon：基于扩散模型的语音匿名化韵律控制

    DiffAnon: Diffusion-based Prosody Control for Voice Anonymization

    [https://arxiv.org/abs/2604.26281](https://arxiv.org/abs/2604.26281)

    DiffAnon是首个基于扩散模型与无分类器引导（CFG）的语音匿名化框架，可在推理时对韵律保留进行显式、连续且可插值的控制，在单一模型内平滑地权衡匿名化强度与韵律保真度。

    

    在语音匿名化中，保留还是不保留韵律是一个核心问题。韵律承载着语义与情感信息，却与说话人身份紧密耦合。现有方法要么为了隐私而完全丢弃韵律，要么缺乏一种有原则的机制来控制效用与隐私之间的权衡，只能在固定的设计点上运行。我们提出DiffAnon，一种基于扩散模型的匿名化方法，借助无分类器引导（CFG）在推理阶段对韵律保留提供显式、连续的控制。DiffAnon在RVQ编解码器的语义嵌入之上细化声学细节，从而在单一模型内实现匿名化强度与韵律保真度之间的平滑插值。据我们所知，这是首个能够提供结构化、可插值的推理时韵律控制的语音匿名化框架。实验展示了结构化的权衡行为，在保持竞争力的同时实现了强大的效用。

    arXiv:2604.26281v2 Announce Type: replace-cross  Abstract: To preserve or not to preserve prosody is a central question in voice anonymization. Prosody conveys meaning and affect, yet is tightly coupled with speaker identity. Existing methods either discard prosody for privacy or lack a principled mechanism to control the utility-privacy trade-off, operating at fixed design points. We propose DiffAnon, a diffusion-based anonymization method with classifier-free guidance (CFG) that provides explicit, continuous inference-time control over prosody preservation. DiffAnon refines acoustic detail over semantic embeddings of an RVQ codec, enabling smooth interpolation between anonymization strength and prosodic fidelity within a single model. To the best of our knowledge, it is the first voice anonymization framework to provide structured, interpolatable inference-time prosody control. Experiments demonstrate structured trade-off behavior, achieving strong utility while maintaining competiti
    
[^162]: 预算约束下的因果老虎机：连接增益建模与序贯决策

    Budget-Constrained Causal Bandits: Bridging Uplift Modeling and Sequential Decision-Making

    [https://arxiv.org/abs/2604.26169](https://arxiv.org/abs/2604.26169)

    该论文提出预算约束因果老虎机（BCCB）在线框架，将个体处理效应学习、不确定性探索与预算节奏控制三者统一起来，并基于拉格朗日松弛的 KKT 条件推导出决策规则，从而解决了冷启动场景下数字广告的预算分配问题。

    

    预算约束下的处理分配是数字广告中的核心挑战。标准做法是在历史数据上训练离线增益模型，然后通过求解约束优化来分配预算，但在几乎没有历史数据的冷启动场景中这种方法会失效。我们提出了预算约束因果老虎机，这是一个在线框架，能够在花费预算的同时学习哪些用户会对广告作出响应。BCCB 统一了三个组件：学习个体层面的处理效应、探索响应不确定的用户、以及随时间推移对预算进行节奏控制。我们将每次用户到达时的决策规则推导为预算化因果分配目标的拉格朗日松弛的 KKT 条件，为算法提供了有原则的理论基础。我们在 Criteo Uplift 数据集上使用 20 个随机种子并进行配对统计检验进行评估。我们的核心发现是在 n = 7,500 处存在一个数据效率交叉点。

    arXiv:2604.26169v2 Announce Type: replace  Abstract: Treatment allocation under budget constraints is a central challenge in digital advertising. The standard approach trains an offline uplift model on historical data, then solves a constrained optimization to allocate budget. This fails in cold-start settings where little historical data exists. We propose Budget-Constrained Causal Bandits (BCCB), an online framework that learns which users respond to ads while simultaneously spending the budget. BCCB unifies three components: learning individual-level treatment effects, exploring users whose response is uncertain, and pacing the budget over time. We derive the per-arrival decision rule as the KKT condition of a Lagrangian relaxation of the budgeted causal-allocation objective, providing a principled foundation for the algorithm. We evaluate on the Criteo Uplift dataset using 20 random seeds with paired statistical tests. Our central finding is a data-efficiency crossover at n = 7,500
    
[^163]: G-Loss：图引导的语言模型微调

    G-Loss: Graph-Guided Fine-Tuning of Language Models

    [https://arxiv.org/abs/2604.25853](https://arxiv.org/abs/2604.25853)

    提出了一种图引导的损失函数G-Loss，通过结合半监督标签传播与文档相似度图来捕捉全局语义结构，引导预训练语言模型学习更具判别性和鲁棒性的嵌入表示。

    

    传统的损失函数，包括用于微调BERT等预训练语言模型的交叉熵、对比损失、三元组损失和监督对比损失，仅在局部邻域内运作，未能考虑全局语义结构。我们提出了G-Loss，这是一种图引导的损失函数，它结合了半监督标签传播来利用嵌入流形中的结构关系。G-Loss构建了一个能够捕捉全局语义关系的文档相似度图，从而引导模型学习更具判别性和鲁棒性的嵌入表示。我们在涵盖关键下游分类任务的五个基准数据集上评估了G-Loss：MR（情感分析）、R8和R52（主题分类）、Ohsumed（医学文档分类）以及20NG（新闻分类）。在大多数实验设置中，G-Loss收敛更快，并能产生语义连贯的嵌入空间……

    arXiv:2604.25853v4 Announce Type: replace  Abstract: Traditional loss functions, including cross-entropy, contrastive, triplet, and su pervised contrastive losses, used for fine-tuning pre-trained language models such as BERT, operate only within local neighborhoods and fail to account for the global semantic structure. We present G-Loss, a graph-guided loss function that incorporates semi-supervised label propagation to use structural relationships within the embedding manifold. G-Loss builds a document-similarity graph that captures global semantic relationships, thereby guiding the model to learn more discriminative and robust embeddings. We evaluate G-Loss on five benchmark datasets covering key downstream classification tasks: MR (sentiment analysis), R8 and R52 (topic categorization), Ohsumed (medical document classification), and 20NG (news categorization). In the majority of experimental setups, G-Loss converges faster and produces semantically coherent embedding spaces, result
    
[^164]: 超越输出正确性：编程任务中大语言模型推理能力的基准测试与评估

    Beyond Output Correctness: Benchmarking and Evaluating Large Language Model Reasoning in Coding Tasks

    [https://arxiv.org/abs/2604.12379](https://arxiv.org/abs/2604.12379)

    该论文提出了首个覆盖代码生成、摘要与分类三类编程任务的推理质量评估基准CodeRQ-Bench，并通过分析评估器失配案例得出设计启示，进而提出结合证据验证与歧义感知评分修正的两阶段评估器VERA，显著提升了编程任务中大语言模型推理质量的评估效果。

    

    大语言模型（LLM）在解决编程任务时越来越依赖显式推理，然而评估这种推理的质量仍然具有挑战性。现有的推理评估器并非为编程任务设计，而当前的基准测试主要关注代码生成，其他编程任务在很大程度上尚未被探索。我们提出了CodeRQ-Bench，这是首个用于评估大语言模型在三类编程任务（生成、摘要和分类）中推理质量的基准。利用该基准，我们分析了来自现有评估器的1,069个失配案例，识别出五个反复出现的局限性，并由此得出四项针对编程任务推理评估的设计启示。基于这些启示，我们提出了VERA——一种结合基于证据的验证与歧义感知评分修正的两阶段评估器。在CodeRQ-Bench上的实验表明，VERA在四个数据集上持续优于强大的基线方法。

    arXiv:2604.12379v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) increasingly rely on explicit reasoning to solve coding tasks, yet evaluating the quality of this reasoning remains challenging. Existing reasoning evaluators are not designed for coding, and current benchmarks focus primarily on code generation, leaving other coding tasks largely unexplored. We introduce CodeRQ-Bench, the first benchmark for evaluating LLM reasoning quality across three coding task categories: generation, summarization, and classification. Using this benchmark, we analyze 1,069 mismatch cases from existing evaluators, identify five recurring limitations, and derive four design insights for reasoning evaluation in coding tasks. Guided by these insights, we propose VERA, a two-stage evaluator that combines evidence-grounded verification with ambiguity-aware score correction. Experiments on CodeRQ-Bench show that VERA consistently outperforms strong baselines across four datasets, imp
    
[^165]: SemEnrich：面向视觉-语言学习的放射学报告自监督语义增强

    SemEnrich: Self-Supervised Semantic Enrichment of Radiology Reports for Vision-Language Learning

    [https://arxiv.org/abs/2604.09887](https://arxiv.org/abs/2604.09887)

    提出SemEnrich方法，利用自监督语义聚类对放射学报告进行增强，通过添加阳性/中性发现来缓解医学视觉-语言数据集的阴性偏差，并在多项评估指标上取得一致的性能提升。

    

    医学视觉-语言数据集通常规模有限，且偏向于阴性发现，因为临床医生主要报告异常情况，但可能会省略一些阳性/中性发现，因为这些发现可能被认为与患者的病情无关。我们提出了一种自监督数据增强方法，该方法利用报告句子的语义聚类。然后，我们以自监督的方式，通过添加来自不同聚类的阳性/中性观察结果来丰富训练集中医学报告的发现。我们的方法在监督微调中带来了一致的性能提升（在COMET分数、Bert分数、Sentence Bleu、CheXbert-F1和RadGraph-F1分数上分别平均提升5.63%、3.04%、7.40%、5.30%、7.47%）。消融研究证实，这些提升源于语义聚类而非随机增强。此外，我们还引入了一种将语义聚类信息纳入GRPO奖励设计的方法。

    arXiv:2604.09887v2 Announce Type: replace  Abstract: Medical vision-language datasets are often limited in size and biased toward negative findings, as clinicians report abnormalities mostly but might omit some positive/neutral findings because they might be considered as irrelevant to the patient's condition. We propose a self-supervised data enrichment method that leverages semantic clustering of report sentences. Then we enrich the findings in the medical reports in the training set by adding positive/neutral observations from different clusters in a self-supervised manner. Our approach yields consistent gains in supervised fine-tuning (5.63%, 3.04%, 7.40%, 5.30%, 7.47% average gains on COMET score, Bert score, Sentence Bleu, CheXbert-F1 and RadGraph-F1 scores respectively). Ablation studies confirm that improvements stem from semantic clustering rather than random augmentation. Furthermore, we introduce a way to incorporate semantic cluster information into the reward design for GR
    
[^166]: PolicyLong：迈向在线策略的上下文扩展

    PolicyLong: Towards On-Policy Context Extension

    [https://arxiv.org/abs/2604.07809](https://arxiv.org/abs/2604.07809)

    PolicyLong 提出动态在线策略的数据构建范式，通过用当前模型迭代重新执行熵计算、检索与验证的数据筛选流程，使训练分布持续跟踪模型能力的演化，从而解决了静态离线数据构建导致的离策略分布漂移问题。

    

    扩展大语言模型（LLM）的上下文窗口一直受到高质量长上下文数据稀缺的阻碍。最近的方法通过信息论验证来合成具有真实长程依赖关系的数据，即选择能够降低基础模型预测熵的上下文。然而，这类方法采用固定模型进行单次离线数据构建，产生了根本性的离策略（off-policy）差距：静态的筛选格局与模型不断演进的能力不相匹配，导致训练分布发生漂移。我们提出 PolicyLong，将数据构建转向动态的在线策略（on-policy）范式。通过使用当前模型迭代地重新执行数据筛选流程（熵计算、检索和验证），PolicyLong 确保训练分布能够跟踪模型不断演进的能力，从而产生一种涌现式的自我课程。至关重要的是，正样本和难负样本上下文均来源于当前模型的熵格局，与模型学习的内容共同演化。

    arXiv:2604.07809v2 Announce Type: replace  Abstract: Extending LLM context windows is hindered by scarce high-quality long-context data. Recent methods synthesize data with genuine long-range dependencies via information-theoretic verification, selecting contexts that reduce a base model's predictive entropy. However, their single-pass offline construction with a fixed model creates a fundamental off-policy gap: the static screening landscape misaligns with the model's evolving capabilities, causing the training distribution to drift. We propose PolicyLong, shifting data construction towards a dynamic on-policy paradigm. By iteratively re-executing data screening (entropy computation, retrieval, and verification) using the current model, PolicyLong ensures the training distribution tracks evolving capabilities, yielding an emergent self-curriculum. Crucially, both positive and hard negative contexts derive from the current model's entropy landscape, co-evolving what the model learns to
    
[^167]: 无证据的提示词：神经影像的提及如何改变临床视觉语言模型的预测

    Prompts Without Evidence: How Neuroimaging Mentions Shift Clinical Vision-Language Model Predictions

    [https://arxiv.org/abs/2603.28387](https://arxiv.org/abs/2603.28387)

    该研究发现在提示词中仅提及神经影像（如MRI）而无需实际提供任何图像，就能显著提升小型视觉语言模型在临床分类任务中的性能与校准度，揭示了这类模型可能依赖表面线索而非真实的影像证据。

    

    值得信赖的临床AI必须使用真实证据，避免依赖表面层次的伪线索。我们在两个临床神经影像队列上评估了12个开放权重的视觉语言模型（VLM），用于情感障碍和认知衰退的二分类任务。两个队列均包含在其原始研究方案下采集的结构磁共振成像（MRI）。先前的研究并未确立这些神经影像输入可作为当前任务的可靠独立诊断证据。然而，当在提示中引入神经影像相关上下文时，较小的VLM在所评估的增强条件下最多可获得0.66的F1提升，从而与规模大一个数量级的模型相比具有竞争力。置信度估计表明，对于所分析的较小模型，大部分校准改进发生在将MRI参考添加到提示词之后、在任何图像被提供之前。我们的初步专家案例研究发现，忠实度……

    arXiv:2603.28387v3 Announce Type: replace-cross  Abstract: Trustworthy clinical AI must use real evidence and avoid relying on surface-level artifacts. We evaluate 12 open-weight vision-language models (VLMs) on two clinical neuroimaging cohorts for binary classification of affective disorders and cognitive decline. Both cohorts include structural magnetic resonance imaging (MRI) acquired under their original research protocols. Prior work does not establish the included neuroimaging inputs as reliable stand-alone diagnostic evidence for the present tasks. Nevertheless, when neuroimaging context is introduced, smaller VLMs gain up to 0.66 F1 under the evaluated augmented conditions, becoming competitive with models an order of magnitude larger. Confidence estimation shows that most of the calibration improvement for the analyzed smaller models occurs after the MRI reference is added to the prompt, before any image is supplied. Our preliminary expert case study finds that faithfulness r
    
[^168]: Deflation-PINNs：学习偏微分方程与Landau-de Gennes理论的多重解

    Deflation-PINNs: Learning Multiple Solutions for PDEs and Landau-de Gennes

    [https://arxiv.org/abs/2603.27936](https://arxiv.org/abs/2603.27936)

    提出Deflation-PINNs框架，通过在PINNs与DeepONet结合的架构中引入收缩损失项，系统性地寻找并收敛到非线性偏微分方程（如液晶Landau-de Gennes模型）的有限多个不同解分支，并提供相应的理论逼近结果。

    

    非线性偏微分方程在数学物理和工程领域中无处不在。尽管物理信息神经网络（PINNs）已成为求解偏微分方程问题的强大工具，但由于其设计初衷是每次只寻找一个解，因此通常难以识别多个不同的解。为了解决这一局限性，我们提出了Deflation-PINNs，这是一种将收缩损失与基于PINNs和深度算子网络的新架构相结合的新型框架。通过在损失函数中加入收缩项，我们的方法能够系统地促使Deflation-PINN寻找并收敛到有限多个不同的解分支。我们提供了关于该模型逼近能力的理论结果，并通过在液晶Landau-de Gennes模型（一个以其复杂能量景观著称的系统）上的数值实验，验证了Deflation-PINNs的有效性。

    arXiv:2603.27936v3 Announce Type: replace-cross  Abstract: Nonlinear Partial Differential Equations (PDEs) are ubiquitous in mathematical physics and engineering. Although Physics-Informed Neural Networks (PINNs) have emerged as a powerful tool for solving PDE problems, they typically struggle to identify multiple distinct solutions, since they are designed to find one solution at a time. To address this limitation, we introduce Deflation-PINNs, a novel framework that integrates a deflation loss with an architecture based on PINNs and Deep Operator Networks (DeepONets). By incorporating a deflation term into the loss function, our method systematically forces the Deflation-PINN to seek and converge upon distinct finitely many solution branches. We provide theoretical results on the approximation capabilities of our model and demonstrate the efficacy of Deflation-PINNs through numerical experiments on the Landau-de Gennes model of liquid crystals, a system renowned for its complex energ
    
[^169]: 基于描述符Beta证据的相机无关3D高斯溅射修剪方法

    Camera-Agnostic Pruning of 3D Gaussian Splats via Descriptor-Based Beta Evidence

    [https://arxiv.org/abs/2603.21933](https://arxiv.org/abs/2603.21933)

    本文提出了一种仅基于属性描述符的相机无关3D高斯溅射修剪方法，通过混合描述符和Beta证据模型量化每个溅射的可靠性，实现了无需相机参数的一次性高效修剪。

    

    arXiv:2603.21933v2 公告类型：替换交叉 摘要：3D高斯溅射的修剪对于降低其复杂度以实现高效存储、传输和下游处理至关重要。然而，现有的大多数修剪策略依赖于相机参数、渲染图像或视图相关度量。这种依赖性在新兴的相机无关交换场景中成为障碍，在这些场景中，溅射直接作为基于点的表示（例如.ply）共享。在本文中，我们提出了一种相机无关、一次性、训练后修剪3D高斯溅射的方法，该方法仅依赖于从属性派生的邻域描述符。作为我们的主要贡献，我们引入了一个混合描述符框架，直接从溅射表示中捕获结构和外观一致性。基于这些描述符，我们将修剪表述为统计证据估计问题，并引入了一个Beta证据模型，用于量化每个溅射的可靠性。

    arXiv:2603.21933v2 Announce Type: replace-cross  Abstract: The pruning of 3D Gaussian splats is essential for reducing their complexity to enable efficient storage, transmission, and downstream processing. However, most of the existing pruning strategies depend on camera parameters, rendered images, or view-dependent measures. This dependency becomes a hindrance in emerging camera-agnostic exchange settings, where splats are shared directly as point-based representations (e.g., .ply). In this paper, we propose a camera-agnostic, one-shot, post-training pruning method for 3D Gaussian splats that relies solely on attribute-derived neighbourhood descriptors. As our primary contribution, we introduce a hybrid descriptor framework that captures structural and appearance consistency directly from the splat representation. Building on these descriptors, we formulate pruning as a statistical evidence estimation problem and introduce a Beta evidence model that quantifies per-splat reliability t
    
[^170]: Var-JEPA：联合嵌入预测架构的变分形式——连接预测式与生成式自监督学习

    Var-JEPA: A Variational Formulation of the Joint-Embedding Predictive Architecture - Bridging Predictive and Generative Self-Supervised Learning

    [https://arxiv.org/abs/2603.20111](https://arxiv.org/abs/2603.20111)

    论文提出Var-JEPA，将JEPA重新诠释为变分推断在耦合潜变量模型上的确定性特例，通过显式优化证据下界（ELBO）弥合了预测式与生成式自监督学习之间的鸿沟。

    

    联合嵌入预测架构（JEPA）通常被视为基于似然的自监督学习的一种非生成式替代方案，强调在表示空间中进行预测，而非在观测空间中进行重构。我们认为，由此产生的与概率生成建模之间的分离在很大程度上是修辞性的而非结构性的：经典的JEPA设计（由上下文到目标的预测器耦合的编码器）对应于将变分推断应用于一类特定的耦合潜变量模型时所得到的变分后验与学习到的条件先验，而标准JEPA可以被视为一种确定性特例，其中正则化是通过架构和训练方面的启发式方法而非显式似然来施加的。基于这一观点，我们推导出了变分JEPA（Var-JEPA），它通过优化单一的证据下界（ELBO）使潜在生成结构显式化（摘要在此处被截断）。

    arXiv:2603.20111v2 Announce Type: replace  Abstract: The Joint-Embedding Predictive Architecture (JEPA) is often seen as a non-generative alternative to likelihood-based self-supervised learning, emphasizing prediction in representation space rather than reconstruction in observation space. We argue that the resulting separation from probabilistic generative modeling is largely rhetorical rather than structural: the canonical JEPA design (coupled encoders with a context-to-target predictor) mirrors the variational posteriors and learned conditional priors obtained when variational inference is applied to a particular class of coupled latent-variable models, and standard JEPA can be viewed as a deterministic specialization in which regularization is imposed via architectural and training heuristics rather than an explicit likelihood. Building on this view, we derive the Variational JEPA (Var-JEPA), which makes the latent generative structure explicit by optimizing a single Evidence Lowe
    
[^171]: 自主性税：防御训练破坏大语言模型智能体

    The Autonomy Tax: Defense Training Breaks LLM Agents

    [https://arxiv.org/abs/2603.19423](https://arxiv.org/abs/2603.19423)

    该论文揭示了“能力-对齐悖论”：为提升安全性而进行的防御训练会系统性地摧毁LLM智能体在多步工具使用任务中的能力（引发智能体无能偏差与级联放大偏差等三种系统性偏差），同时却无法阻止复杂的提示注入攻击。

    

    大语言模型（LLM）智能体日益依赖外部工具（文件操作、API调用、数据库事务）来自主完成复杂的多步骤任务。从业者部署经过防御训练的模型，以防范通过恶意观测内容或检索内容操纵智能体行为的提示注入攻击。我们揭示了一个根本性的“能力-对齐悖论”：旨在提升安全性的防御训练系统性地摧毁了智能体的能力，却未能阻止复杂的攻击。通过在97个智能体任务和1,000个对抗性提示上，将经过防御的模型与未防御的基线模型进行对比评估，我们发现了多步骤智能体特有的三种系统性偏差。“智能体无能偏差”表现为工具执行立即失效，模型在尚未观测到任何外部内容之前，就会在良性任务上拒绝执行或生成无效动作。“级联放大偏差”导……（原文摘要在此处截断）

    arXiv:2603.19423v3 Announce Type: replace-cross  Abstract: Large language model (LLM) agents increasingly rely on external tools (file operations, API calls, database transactions) to autonomously complete complex multi-step tasks. Practitioners deploy defense-trained models to protect against prompt injection attacks that manipulate agent behavior through malicious observations or retrieved content. We reveal a fundamental \textbf{capability-alignment paradox}: defense training designed to improve safety systematically destroys agent competence while failing to prevent sophisticated attacks. Evaluating defended models against undefended baselines across 97 agent tasks and 1,000 adversarial prompts, we uncover three systematic biases unique to multi-step agents. \textbf{Agent incompetence bias} manifests as immediate tool execution breakdown, with models refusing or generating invalid actions on benign tasks before observing any external content. \textbf{Cascade amplification bias} cau
    
[^172]: InfoMamba：一种无注意力的Mamba-Transformer混合模型

    InfoMamba: An Attention-Free Hybrid Mamba-Transformer Model

    [https://arxiv.org/abs/2603.18031](https://arxiv.org/abs/2603.18031)

    InfoMamba提出了一种无注意力的Mamba-Transformer混合架构，用概念瓶颈线性过滤层替代自注意力，并通过信息最大化融合（IMF）动态注入全局上下文，从而在线性复杂度下兼顾细粒度局部建模与长程全局依赖捕捉。

    

    arXiv:2603.18031v2 公告类型：替换 摘要：在计算约束下平衡细粒度局部建模与长程依赖捕捉，仍然是序列建模中的核心挑战。Transformer虽然具有强大的词元混合能力，但其复杂度为二次方；而Mamba式的选择性状态空间模型（SSM）虽然具有线性复杂度，却往往难以捕捉高秩和同步的全局交互。我们提出了一种一致性边界分析，刻画了对角短记忆SSM何时能够逼近因果注意力，并识别了仍然存在的结构性差距。基于这一分析，我们提出了InfoMamba，一种无注意力的混合架构。InfoMamba用一个概念瓶颈线性过滤层取代了词元级自注意力，该层充当最小带宽的全局接口，并通过信息最大化融合（IMF）将其与选择性递归流集成。IMF动态地将全局上下文注入SSM……

    arXiv:2603.18031v2 Announce Type: replace  Abstract: Balancing fine-grained local modeling with long-range dependency capture under computational constraints remains a central challenge in sequence modeling. While Transformers provide strong token mixing, they suffer from quadratic complexity, whereas Mamba-style selective state-space models (SSMs) scale linearly but often struggle to capture high-rank and synchronous global interactions. We present a consistency boundary analysis that characterizes when diagonal short-memory SSMs can approximate causal attention and identifies structural gaps that remain. Motivated by this analysis, we propose InfoMamba, an attention-free hybrid architecture. InfoMamba replaces token-level self-attention with a concept bottleneck linear filtering layer that serves as a minimal-bandwidth global interface and integrates it with a selective recurrent stream through information-maximizing fusion (IMF). IMF dynamically injects global context into the SSM d
    
[^173]: 单纯形到欧几里得空间的双射：用于共轭且校准的多类高斯过程分类

    Simplex-to-Euclidean Bijection for Conjugate and Calibrated Multiclass Gaussian Process Classification

    [https://arxiv.org/abs/2603.16621](https://arxiv.org/abs/2603.16621)

    该论文提出利用Aitchison几何将概率单纯形上的类别概率双射映射到欧几里得空间，把多类分类转化为共轭的GP回归问题，从而在无需分布近似的情况下实现共轭推断、良好校准的预测概率以及基于稀疏GP技术的可扩展推断。

    

    我们提出了一种共轭且校准的高斯过程（GP）模型，用于多类分类，其核心在于利用概率单纯形的几何结构。我们的方法采用Aitchison几何将单纯形取值的类别概率映射到无约束的欧几里得表示，从而将分类问题转化为一个潜在维度少于标准多类GP分类器的GP回归问题。这使得推理具有共轭性，并产生可靠的预测概率，而无需在模型构建中依赖分布近似。该方法与标准的稀疏GP回归技术兼容，能够在更大规模的数据集上实现可扩展的推断。实证结果表明，该方法在合成数据集和真实世界数据集上均表现出良好的校准性能和竞争力。

    arXiv:2603.16621v2 Announce Type: replace  Abstract: We propose a conjugate and calibrated Gaussian process (GP) model for multi-class classification by exploiting the geometry of the probability simplex. Our approach uses Aitchison geometry to map simplex-valued class probabilities to an unconstrained Euclidean representation, turning classification into a GP regression problem with fewer latent dimensions than standard multi-class GP classifiers. This yields conjugate inference and reliable predictive probabilities without relying on distributional approximations in the model construction. The method is compatible with standard sparse GP regression techniques, enabling scalable inference on larger datasets. Empirical results show well-calibrated and competitive performance across synthetic and real-world datasets.
    
[^174]: Agentic-Kube：一种用于多目标Kubernetes调度的图增强多智能体强化学习框架

    Agentic-Kube: A Graph-Enhanced Multi-Agent Reinforcement Learning Framework for Multi-Objective Kubernetes Scheduling

    [https://arxiv.org/abs/2603.12031](https://arxiv.org/abs/2603.12031)

    Agentic-Kube提出了一种图增强的协作式多智能体强化学习框架，将Kubernetes多目标调度分解为由专门子智能体负责的成本最小化、容错和资源平衡三方优化空间，并结合二部图卷积网络与QMIX值分解，解决了传统单智能体强化学习的梯度干扰和奖励稀释问题。

    

    云原生容器编排需要能够平衡基础设施支出、故障韧性和节点利用率的资源调度器。传统强化学习方法通常依赖单一的单智能体模型，在将相互冲突的运营目标映射为单一标量奖励时，会出现梯度干扰和奖励稀释的问题。我们提出了Agentic-Kube，一个专为实时Kubernetes Pod放置设计的协作式多智能体强化学习框架。该架构将多目标调度分解为三方优化空间，由专门的子智能体分别负责成本最小化、反亲和性容错以及向量资源平衡。Agentic-Kube集成了二部图卷积网络以捕获动态的主机-Pod依赖关系、两阶段单调QMIX值分解网络以保持联合动作值的一致性，以及一个多元化的……

    arXiv:2603.12031v4 Announce Type: replace-cross  Abstract: Cloud-native container orchestration requires resource schedulers capable of balancing infrastructure expenditure, fault resilience, and node utilisation. Conventional reinforcement learning approaches typically rely on monolithic single-agent models that suffer from gradient interference and reward dilution when mapping conflicting operational goals into a single scalar reward. We present Agentic-Kube, a cooperative multi-agent reinforcement learning framework designed for real-time Kubernetes pod placement. The architecture decomposes multi-objective scheduling into a tripartite optimisation space managed by dedicated sub-agents for cost minimisation, anti-affinity fault tolerance, and vector resource balancing. Agentic-Kube integrates a bipartite Graph Convolutional Network to capture dynamic host-pod dependencies, a two-stage monotonic QMIX value factorisation network to maintain joint action value coherence, and a pluralit
    
[^175]: FlowCorrect：面向机器人操作的高效交互式生成流策略修正

    FlowCorrect: Efficient Interactive Correction of Generative Flow Policies for Robotic Manipulation

    [https://arxiv.org/abs/2602.22056](https://arxiv.org/abs/2602.22056)

    FlowCorrect提出一种交互式模仿学习方法，通过轻量级VR界面收集稀疏的人类纠正信号，在部署时对流匹配操作策略进行局部适配，无需重新训练即可修复“擦肩而过”式的失败案例，同时保持已学场景的性能。

    

    生成式操作策略在部署时遇到分布偏移可能会发生灾难性失败，然而许多失败其实只是“擦肩而过”：机器人到达了几乎正确的姿态，只需一个微小的纠正动作就能成功。我们提出FlowCorrect，这是一种模块化的交互式模仿学习方法，能够在部署时利用稀疏的相对人类纠正信号来适配流匹配操作策略，而无需重新训练。在执行过程中，人类通过轻量级VR界面提供简短的纠正性姿态微调。FlowCorrect利用这些稀疏纠正对策略进行局部适配，在不重新训练主干网络的情况下改进动作，同时保持模型在已学习场景上的性能。我们在真实机器人上评估了四个桌面任务：抓取放置、倒水、杯子扶正和插入。在低纠正预算下，FlowCorrect在之前失败的案例上实现了80%的成功率。

    arXiv:2602.22056v3 Announce Type: replace-cross  Abstract: Generative manipulation policies can fail catastrophically under deployment-time distribution shift, yet many failures are near-misses: the robot reaches almost-correct poses and would succeed with a small corrective motion. We propose FlowCorrect, a modular interactive imitation learning approach that enables deployment-time adaptation of flow-matching manipulation policies from sparse, relative human corrections without retraining. During execution, a human provides brief corrective pose nudges via a lightweight VR interface. FlowCorrect uses these sparse corrections to locally adapt the policy, improving actions without retraining the backbone while preserving the model performance on previously learned scenarios. We evaluate on a real-world robot across four tabletop tasks: pick-and-place, pouring, cup uprighting, and insertion. With a low correction budget, FlowCorrect achieves an 80% success rate on previously failed case
    
[^176]: 挖掘与精炼：优化电商语义搜索检索中的分级相关性

    Mine and Refine: Optimizing Graded Relevance in E-commerce Semantic Search Retrieval

    [https://arxiv.org/abs/2602.17654](https://arxiv.org/abs/2602.17654)

    提出两阶段对比训练框架“挖掘与精炼”，借助经参与度审计微调的轻量级LLM标注器，同时解决大规模电商语义搜索中分级相关性建模、硬样本挖掘假阴性以及相似度得分可分性不稳定三大难题。

    

    基于嵌入的检索（EBR）在大规模电商搜索中面临三个相互交织的挑战：分级（非二元）相关性——其中用户参与度信号存在噪声且意图多变，而业务相关性准则允许“可接受但不完全精确”的匹配；硬样本挖掘中的假阴性问题；以及跨相关性等级的相似度得分可分性不稳定，最后一个问题使混合搜索中的得分融合和下游排序变得复杂。我们提出“挖掘与精炼”，一个解决上述三个问题的两阶段对比训练框架。一个经过参与度驱动审计微调的轻量级大语言模型（LLM），在整个训练过程中充当与业务准则对齐的可扩展标注器。第一阶段通过标签感知的监督对比学习建立稳健的全局嵌入空间；第二阶段挖掘难样本，利用LLM标注器对其进行重新标注以减轻虚假负样本，并通过圈损失的多级扩展来精炼模型。

    arXiv:2602.17654v2 Announce Type: replace-cross  Abstract: Embedding-based retrieval (EBR) for large-scale e-commerce search faces three intertwined challenges: graded (non-binary) relevance where engagement signals are noisy and intent-varying while business relevance guidelines admit acceptable-but-not-exact matches, false negatives in hard sample mining, and unstable similarity score separability across relevance levels, the last of which complicates hybrid search score fusion and downstream ranking. We propose Mine and Refine, a two-stage contrastive training framework that addresses all three. A lightweight LLM, fine-tuned with engagement-driven audit, serves as a guideline-aligned scalable labeler throughout training. Stage 1 establishes a robust global embedding space via label-aware supervised contrastive learning; Stage 2 mines hard samples, re-annotates them with the LLM labeler to mitigate spurious negatives, and refines the model through a multi-level extension of circle lo
    
[^177]: 基于观测数据的鲁棒商品组合优化

    Robust Assortment Optimization from Observational Data

    [https://arxiv.org/abs/2602.10696](https://arxiv.org/abs/2602.10696)

    提出了一个鲁棒的数据驱动商品组合优化框架，通过建模顾客选择行为中潜在的分布偏移，克服了传统方法因假设偏好稳定和选择模型正确而在现实中导致的泛化差和收益损失问题。

    

    商品组合优化是现代零售和推荐系统中的一项根本性挑战，其目标是在复杂的顾客选择行为下，选择能够最大化预期收益的产品子集。尽管数据驱动方法的最新进展已利用历史数据来学习和优化商品组合，但这些方法通常依赖于较强的假设——即顾客偏好的稳定性以及底层选择模型的正确性。然而，在现实场景中，由于偏好漂移和模型误设，这些假设经常失效，导致泛化能力差和收益损失。受此局限性的启发，我们提出了一个鲁棒的数据驱动商品组合优化框架，该框架考虑了顾客选择行为中潜在的分布偏移。我们的方法对相对于生成数据的标称选择模型可能发生的偏好偏移进行建模，并寻求……（摘要在此处截断）

    arXiv:2602.10696v3 Announce Type: replace-cross  Abstract: Assortment optimization is a fundamental challenge in modern retail and recommendation systems, where the goal is to select a subset of products that maximizes expected revenue under complex customer choice behaviors. While recent advances in data-driven methods have leveraged historical data to learn and optimize assortments, these approaches typically rely on strong assumptions -- namely, the stability of customer preferences and the correctness of the underlying choice models. However, such assumptions frequently break in real-world scenarios due to preference shifts and model misspecification, leading to poor generalization and revenue loss. Motivated by this limitation, we propose a robust framework for data-driven assortment optimization that accounts for potential distributional shifts in customer choice behavior. Our approach models potential preference shift from a nominal choice model that generates data and seeks to 
    
[^178]: SCALE：面向视觉-语言-动作模型的自不确定性条件化自适应观察与执行

    SCALE: Self-uncertainty Conditioned Adaptive Looking and Execution for Vision-Language-Action Models

    [https://arxiv.org/abs/2602.04208](https://arxiv.org/abs/2602.04208)

    SCALE是一种受主动推理理论启发的推理策略，基于模型自不确定性在单次前向传播中联合自适应调节视觉感知与动作，无需额外训练或验证器即可提升VLA模型在感知模糊情境下的鲁棒性与部署实用性。

    

    视觉-语言-动作（VLA）模型已成为通用机器人控制的一种有前景的范式，而测试时扩展（TTS）因能在训练之外增强鲁棒性而日益受到关注。然而，现有的VLA测试时扩展方法需要额外的训练、验证器以及多次前向传播，使其难以实际部署。此外，这些方法仅在动作解码阶段进行干预，而保持视觉表征固定——在感知模糊的情况下这是不足够的，因为在模糊情境下，重新思考如何感知与决定做什么同等重要。为了解决这些局限，我们提出了SCALE，这是一种简单的推理策略，受主动推理理论中不确定性驱动探索的启发，基于“自不确定性”联合调节视觉感知与动作——无需额外训练、无需验证器，且仅需单次前向传播。SCALE在高不确定性下拓宽感知与动作两个层面的探索……

    arXiv:2602.04208v3 Announce Type: replace-cross  Abstract: Vision-Language-Action (VLA) models have emerged as a promising paradigm for general-purpose robotic control, with test-time scaling (TTS) gaining attention to enhance robustness beyond training. However, existing TTS methods for VLAs require additional training, verifiers, and multiple forward passes, making them impractical for deployment. Moreover, they intervene only at action decoding while keeping visual representations fixed-insufficient under perceptual ambiguity, where reconsidering how to perceive is as important as deciding what to do. To address these limitations, we propose SCALE, a simple inference strategy that jointly modulates visual perception and action based on 'self-uncertainty', inspired by uncertainty-driven exploration in Active Inference theory-requiring no additional training, no verifier, and only a single forward pass. SCALE broadens exploration in both perception and action under high uncertainty, w
    
[^179]: 面向Gröbner基计算的学习型快速单项式排序

    Learning Fast Monomial Orders for Gr\"obner Basis Computations

    [https://arxiv.org/abs/2602.02972](https://arxiv.org/abs/2602.02972)

    该论文将Gröbner基计算中单项式序的选择建模为强化学习问题，所学策略在系统生物学和计算机视觉基准问题上持续超越GrevLex等标准启发式方法，大幅降低计算成本。

    

    Gröbner基计算是求解多项式方程组的标准引擎，其效率取决于单项式序的选择。尽管可能的单项式序近乎连续，但大多数实现仍依赖于以专家直觉为指导的静态启发式方法，如GrevLex。我们通过将单项式序的选择建模为在可允许序空间上的强化学习问题来弥补这一空白。我们的方法利用了能够准确反映Gröbner基计算成本、且支持高效蒙特卡洛估计的领域知识驱动的奖励信号。在来自系统生物学和计算机视觉的基准问题上的实验表明，所得的学习策略始终优于标准启发式方法，显著降低了计算成本。此外，我们发现这些策略难以被蒸馏为简单的可解释模型。

    arXiv:2602.02972v2 Announce Type: replace-cross  Abstract: The efficiency of Gr\"obner basis computation, the standard engine for solving systems of polynomial equations, depends on the choice of monomial ordering. Despite a near-continuum of possible monomial orders, most implementations rely on static heuristics such as GrevLex, guided primarily by expert intuition. We address this gap by casting the selection of monomial orderings as a reinforcement learning problem over the space of admissible orderings. Our approach leverages domain-informed reward signals that accurately reflect the computational cost of Gr\"obner basis computations and admits efficient Monte Carlo estimation. Experiments on benchmark problems from systems biology and computer vision show that the resulting learned policies consistently outperform standard heuristics, yielding substantial reductions in computational cost. Moreover, we find that these policies resist distillation into simple interpretable models, 
    
[^180]: 用于模型差异校准的贝叶斯实验设计：Kullback-Leibler散度与Wasserstein距离之争

    Bayesian Experimental Design for Model Discrepancy Calibration: A Rivalry between Kullback--Leibler Divergence and Wasserstein Distance

    [https://arxiv.org/abs/2601.16425](https://arxiv.org/abs/2601.16425)

    本文通过玩具示例揭示了Wasserstein距离作为贝叶斯实验设计效用函数的缺陷——固定形状后验的距离值取决于其主体质量在支撑集内的相对位置，可能产生与信息增益无关的虚假奖励，凸显了KL散度在模型差异校准实验设计中的优势。

    

    arXiv:2601.16425v2 公告类型：替换 摘要：设计能够系统性地从复杂物理系统中采集数据的实验，是加速科学发现的核心所在。贝叶斯实验设计（BED）提供了一个有原则的、基于信息的框架，将实验规划与概率推断相结合，而BED中效用函数的选择是一个长期存在且活跃的研究课题，不同的准则强调不同的信息概念。尽管Kullback-Leibler（KL）散度一直是最常见的选择之一，但近期有研究提出将Wasserstein距离作为替代方案。在这项工作中，我们首先通过一个玩具示例阐明了Wasserstein距离存在的一个问题——固定形状后验分布的Wasserstein距离值取决于其主体质量在支撑集内的相对位置，可能表现出与信息增益无关的虚假奖励，尤其是在使用非信息性先验（例如均匀分布）的情况下。

    arXiv:2601.16425v2 Announce Type: replace  Abstract: Designing experiments that systematically gather data from complex physical systems is central to accelerating scientific discovery. While Bayesian experimental design (BED) provides a principled, information-based framework that integrates experimental planning with probabilistic inference, the selection of utility functions in BED is a long-standing and active topic, where different criteria emphasize different notions of information. Although Kullback--Leibler (KL) divergence has been one of the most common choices, recent studies have proposed Wasserstein distance as an alternative. In this work, we first employ a toy example to illustrate an issue of Wasserstein distance - the value of Wasserstein distance of a fixed-shape posterior depends on the relative position of its main mass within the support and can exhibit false rewards unrelated to information gain, especially with a non-informative prior (e.g., uniform distribution).
    
[^181]: 通过知识性经验学习对齐智能体世界模型

    Aligning Agentic World Models via Knowledgeable Experience Learning

    [https://arxiv.org/abs/2601.13247](https://arxiv.org/abs/2601.13247)

    提出WorldMind框架，通过综合环境反馈自主构建符号化世界知识库，使LLM智能体世界模型无需昂贵再训练即可遵循物理法则、避免物理幻觉。

    

    当前的大型语言模型（LLMs）存在一个关键的模态脱节问题：它们拥有海量的语义知识，却缺乏遵循物理世界不变法则的程序性基础。因此，尽管这些智能体隐式地充当着世界模型，它们的模拟常常受到物理幻觉的困扰——生成逻辑上合理但物理上无法执行的计划。现有的对齐策略主要依赖资源密集型的训练或微调，试图将动态的环境规则压缩进静态的模型参数中。然而，这种参数化封装本质上是僵硬的，若不进行持续且昂贵的再训练，难以适应物理动力学的开放式变化。为弥合这一差距，我们提出了WorldMind，一个通过综合环境反馈来自主构建符号化世界知识库的框架。具体而言，它统一了过程经验……

    arXiv:2601.13247v2 Announce Type: replace  Abstract: Current Large Language Models (LLMs) exhibit a critical modal disconnect: they possess vast semantic knowledge but lack the procedural grounding to respect the immutable laws of the physical world. Consequently, while these agents implicitly function as world models, their simulations often suffer from physical hallucinations-generating plans that are logically sound but physically unexecutable. Existing alignment strategies predominantly rely on resource-intensive training or fine-tuning, which attempt to compress dynamic environmental rules into static model parameters. However, such parametric encapsulation is inherently rigid, struggling to adapt to the open-ended variability of physical dynamics without continuous, costly retraining. To bridge this gap, we introduce WorldMind, a framework that autonomously constructs a symbolic World Knowledge Repository by synthesizing environmental feedback. Specifically, it unifies Process Ex
    
[^182]: 安全的不稳定性：随机种子与温度如何暴露大语言模型不一致的拒绝行为

    The Instability of Safety: How Random Seeds and Temperature Expose Inconsistent LLM Refusal Behavior

    [https://arxiv.org/abs/2512.12066](https://arxiv.org/abs/2512.12066)

    该研究揭示大语言模型的安全拒绝决策在随机种子和温度变化下并不稳定，18-28%的有害提示词会出现“拒绝”与“配合”之间的决策翻转，且温度越高稳定性越差，表明单次安全评估无法真实反映模型的安全对齐水平。

    

    当前大语言模型的安全评估依赖于单次测试，隐含地假设模型响应是确定性的，并能代表模型的安全对齐水平。我们通过研究安全拒绝决策在不同随机种子和温度设置下的稳定性来挑战这一假设。我们在20种采样配置（4种温度 × 5个随机种子）下，对来自三个系列的四个指令微调模型（Llama 3.1 8B、Qwen 2.5 7B、Qwen 3 8B、Gemma 3 12B）在876个有害提示词上进行了测试，发现18-28%的提示词表现出决策翻转——即模型在某些配置下拒绝回答，而在其他配置下则予以配合——具体比例因模型而异。我们的安全稳定性指数（SSI）显示，更高的温度会显著降低决策稳定性（Friedman卡方 = 396.81，p < 0.001），温度内平均SSI从温度0.0时的0.977下降至温度1.0时的0.942。我们在……（原文摘要在此处被截断）

    arXiv:2512.12066v3 Announce Type: replace-cross  Abstract: Current safety evaluations of large language models rely on single-shot testing, implicitly assuming that model responses are deterministic and representative of the model's safety alignment. We challenge this assumption by investigating the stability of safety refusal decisions across random seeds and temperature settings. Testing four instruction-tuned models from three families (Llama 3.1 8B, Qwen 2.5 7B, Qwen 3 8B, Gemma 3 12B) on 876 harmful prompts across 20 sampling configurations (4 temperatures x 5 seeds), we find that 18-28% of prompts exhibit decision flips--the model refuses in some configurations but complies in others--depending on the model. Our Safety Stability Index (SSI) reveals that higher temperatures significantly reduce decision stability (Friedman chi-squared = 396.81, p < 0.001), with mean within-temperature SSI dropping from 0.977 at temperature 0.0 to 0.942 at temperature 1.0. We validate findings acro
    
[^183]: 预测序后验

    Prequential posteriors

    [https://arxiv.org/abs/2511.17721](https://arxiv.org/abs/2511.17721)

    本文提出基于预测序列损失函数的prequential后验方法，解决了深度生成预测模型因似然函数不可解而无法应用标准贝叶斯数据同化的难题，并证明了其在温和条件下的理论一致性保证。

    

    数据同化是在观测到新数据时更新预测模型的一项基础任务，其应用涵盖天气预报到在线强化学习等领域。深度生成预测模型（DGFMs）在这些领域表现出色，但由于其似然函数难以处理，将数据同化到此类模型中极具挑战性。这一局限性限制了标准贝叶斯数据同化方法在DGFMs中的应用。为了克服这一问题，我们提出了基于预测序列损失函数的prequential后验；该方法天然适用于时间相关数据，而这正是预测任务的核心关注点。由于真实的数据生成过程往往超出了所假设的模型类别，我们采用了一种替代的一致性概念，并证明在温和的条件下，prequential损失最小化器和prequential后验均会集中在……

    arXiv:2511.17721v2 Announce Type: replace-cross  Abstract: Data assimilation is a fundamental task in updating forecasting models upon observing new data, with applications ranging from weather prediction to online reinforcement learning. Deep generative forecasting models (DGFMs) have shown excellent performance in these areas, but assimilating data into such models is challenging due to their intractable likelihood functions. This limitation restricts the use of standard Bayesian data assimilation methodologies for DGFMs. To overcome this, we introduce prequential posteriors, based upon a predictive-sequential (prequential) loss function; an approach naturally suited for temporally dependent data which is the focus of forecasting tasks. Since the true data-generating process often lies outside the assumed model class, we adopt an alternative notion of consistency and prove that, under mild conditions, both the prequential loss minimizer and the prequential posterior concentrate aroun
    
[^184]: 具有噪声效用测量的博弈中基于期望的扰动学习自动机。A部分：非零和博弈中的随机稳定性

    Aspiration-based Perturbed Learning Automata in Games with Noisy Utility Measurements. Part A: Stochastic Stability in Non-zero-Sum Games

    [https://arxiv.org/abs/2511.11602](https://arxiv.org/abs/2511.11602)

    本文提出了一种新颖的基于收益的分布式学习方案——基于期望的扰动学习自动机（APLA），使多人弱无环非零和博弈中各参与者独立学习时也能收敛到纯纳什均衡，突破了以往方法仅适用于势博弈和协调博弈的局限。

    

    基于强化的学习在建模人类行为以及工程设计基于测量或收益的优化方案方面都引起了相当大的关注。这类学习方案表现出若干优势，尤其是在滤除噪声观测方面。然而，当应用于分布式环境时，它们可能存在若干局限性。在多人弱无环博弈中，当每个参与者应用学习动力学的独立副本时，无法保证收敛到（通常是理想的）纯纳什均衡。先前的工作仅关注于一小类博弈，即势博弈和协调博弈。为了解决这一主要局限，本文提出了一种新颖的用于分布式优化的基于收益的学习方案，即基于期望的扰动学习自动机（APLA）。在这类动力学中，与标准的基于强化的学习方案相反，每个……（摘要至此不完整）

    arXiv:2511.11602v3 Announce Type: replace  Abstract: Reinforcement-based learning has attracted considerable attention both in modeling human behavior as well as in engineering, for designing measurement- or payoff-based optimization schemes. Such learning schemes exhibit several advantages, especially in relation to filtering out noisy observations. However, they may exhibit several limitations when applied in a distributed setup. In multi-player weakly-acyclic games, and when each player applies an independent copy of the learning dynamics, convergence to (usually desirable) pure Nash equilibria cannot be guaranteed. Prior work has only focused on a small class of games, namely potential and coordination games. To address this main limitation, this paper introduces a novel payoff-based learning scheme for distributed optimization, namely aspiration-based perturbed learning automata (APLA). In this class of dynamics, and contrary to standard reinforcement-based learning schemes, each 
    
[^185]: Think-at-Hard（难处深思）：面向推理能力提升的动态循环Transformer

    Think-at-Hard: Dynamic Looped Transformers for Improved Reasoning

    [https://arxiv.org/abs/2511.08577](https://arxiv.org/abs/2511.08577)

    针对循环Transformer中存在的“潜在过度思考”现象，提出TaH方法，利用轻量级神经决策器仅在可能出错的token处动态触发潜在迭代，从而在参数受限条件下将大语言模型的推理性能提升高达7.3%。

    

    提升大语言模型（LLM）的推理能力，尤其是在参数受限的条件下，对实际应用至关重要。循环Transformer通过执行多次潜在迭代来优化每个token的表示，超越了单次前向传播的能力。然而，我们识别出一种“潜在过度思考”现象：大多数token预测在第一次前向传播后就已经正确，但在后续迭代中有时反而被修改成错误。我们探究了选择性地跳过潜在迭代能否提升准确率，并通过一个先验迭代策略揭示了显著的潜力，该策略可将性能提升高达7.3%。受此启发，我们提出了Think-at-Hard（TaH），一种针对选择性迭代进行优化的循环Transformer。TaH采用一个轻量级神经决策器，仅在标准前向传播后可能出错的token上触发潜在迭代。在潜在迭代过程中，深度感知的低秩适应（LoRA）模块……

    arXiv:2511.08577v4 Announce Type: replace  Abstract: Improving the reasoning abilities of Large Language Models (LLMs), especially under parameter constraints, is crucial for real-world applications. Looped transformers address this by performing multiple latent iterations to refine each token beyond a single forward pass. However, we identify a latent overthinking phenomenon: most token predictions are already correct after the first pass, but are sometimes revised into errors in later iterations. We ask whether selectively skipping latent iterations can improve accuracy, and reveal significant potential with an oracle iteration policy that boosts performance by up to 7.3%. Motivated by this, we propose Think-at-Hard (TaH), a looped transformer optimized for selective iteration. TaH employs a lightweight neural decider to trigger latent iteration, only at tokens likely to be incorrect after the standard forward pass. During latent iterations, depth-aware Low-Rank Adaptation (LoRA) mod
    
[^186]: 一模型通用：面向异构数据集与范式的基于EEG情绪识别的通用预训练

    One Model for All: Universal Pre-training for EEG based Emotion Recognition across Heterogeneous Datasets and Paradigms

    [https://arxiv.org/abs/2511.08444](https://arxiv.org/abs/2511.08444)

    提出了一种跨异构EEG数据集与范式的通用预训练框架"One Model for All"，通过统一通道模式进行单通道自监督对比学习预训练，再结合ART与GAT架构进行多变量微调，有效解决了数据集异构性难题，并在SEED、DEAP和DREAMER数据集上取得显著性能提升。

    

    基于EEG的情绪识别受到数据集深度异构性（通道/被试差异性）的阻碍，难以构建可泛化的模型，现有方法也难以有效地迁移知识。我们提出了"One Model for All"（一模型通用），一个跨不同数据集的EEG分析通用预训练框架。该范式将学习过程解耦为两个阶段：(1) 单变量预训练，通过在单个通道上进行自监督对比学习实现，这得益于统一通道模式（UCS），它利用了各数据集的通道并集（例如SEED-62通道、DEAP-32通道）；(2) 多变量微调，采用新颖的"ART"（自适应重采样Transformer）与"GAT"（图注意力网络）架构来捕获复杂的时空依赖关系。实验表明，通用预训练是一种关键的稳定器，可防止模型在SEED上从头训练时的性能崩溃，并在DEAP（+7.65%）和DREAMER（+3.55%）上带来显著提升。我们的框架实现了新的……

    arXiv:2511.08444v2 Announce Type: replace  Abstract: EEG-based emotion recognition is hampered by profound dataset heterogeneity (channel/subject variability), hindering generalizable models. Existing approaches struggle to transfer knowledge effectively. We propose 'One Model for All', a universal pre-training framework for EEG analysis across disparate datasets. Our paradigm decouples learning into two stages: (1) Univariate pre-training via self-supervised contrastive learning on individual channels, enabled by a Unified Channel Schema (UCS) that leverages the channel union (e.g., SEED-62ch, DEAP-32ch); (2) Multivariate fine-tuning with a novel 'ART' (Adaptive Resampling Transformer) and 'GAT' (Graph Attention Network) architecture to capture complex spatio-temporal dependencies. Experiments show universal pre-training is an essential stabilizer, preventing collapse on SEED (vs. scratch) and yielding substantial gains on DEAP (+7.65%) and DREAMER (+3.55%). Our framework achieves new
    
[^187]: 用于预测重度抑郁症症状严重程度的多语言口语词汇特征分析

    Multilingual Lexical Feature Analysis of Spoken Language for Predicting Major Depression Symptom Severity

    [https://arxiv.org/abs/2511.07011](https://arxiv.org/abs/2511.07011)

    该研究基于来自英国、荷兰和西班牙467名参与者的多语言智能手机录音数据，利用可解释的线性混合效应模型识别出与抑郁症状严重程度相关的口语词汇特征，并通过机器学习验证了这些特征对PHQ-8评分预测的增益作用。

    

    背景：远程采集的口语语言可以为抑郁症症状严重程度提供客观、定期的指标。然而，迄今为止的研究主要使用非临床、横断面的书面语言以及可解释性有限的复杂机器学习方法。方法：我们使用线性混合效应模型，在RADAR-MDD研究的数据中识别与症状严重程度相关的可解释词汇特征，该数据包含来自英国、荷兰和西班牙467名参与者的5,846个智能手机录音以及患者健康问卷（PHQ-8）评分。随后，我们开发了机器学习模型，并通过嵌套交叉验证系统地评估了可解释的词汇特征或高维向量嵌入是否能比社会人口统计学和混杂因素特征提高PHQ-8预测的准确性。结果：抑郁症状严重程度与五个词汇特征相关，包括词语（原文在此处截断）

    arXiv:2511.07011v2 Announce Type: replace  Abstract: Background: Remotely captured spoken language could provide objective, regular indicators of depression symptom severity. However, research to date has largely used non-clinical, cross-sectional written language and complex machine learning (ML) approaches with limited interpretability. Methods: We used linear mixed-effect models to identify interpretable lexical features associated with symptom severity in data from the RADAR-MDD study that comprised 5,846 smartphone recordings and Patient Health Questionnaire (PHQ-8) scores from 467 participants in the UK, Netherlands and Spain. We then developed ML models and systematically assessed via nested cross-validation whether interpretable lexical features or high-dimensional vector embeddings improved the accuracy of PHQ-8 prediction over sociodemographic and confounding features. Results: Depression symptom severity was associated with five lexical features, including reductions in word
    
[^188]: GREAT：基于情感感知触发器合成的RLHF可泛化后门攻击

    GREAT: Generalizable Backdoor Attacks in RLHF via Emotion-Aware Trigger Synthesis

    [https://arxiv.org/abs/2510.09260](https://arxiv.org/abs/2510.09260)

    GREAT框架通过在模型潜在嵌入空间中利用降维与聚类技术识别愤怒情绪触发器，并构建包含5000余个触发器的Erinyes数据集，实现了针对RLHF的自然分布、可泛化后门攻击。

    

    近期研究表明，基于人类反馈的强化学习（RLHF）极易受到后门攻击。然而，现有方法通常依赖稀有词元或固定触发器，限制了其在现实场景中的攻击效果。在本工作中，我们提出了GREAT，一个用于在RLHF中构建自然分布后门的新型框架。具体而言，GREAT针对一个易受攻击的用户子群体诱导生成有害响应，该群体的特征是语义上暴力的请求搭配情感上愤怒的触发器。我们框架的核心是一个在模型潜在嵌入空间中运行的触发器识别流水线，利用降维和聚类技术来识别代表性触发器。为此，我们引入了一种分层且多样性驱动的提示策略，构建了Erinyes——一个从GPT-4.1中精选的包含超过5,000个愤怒触发器的高质量数据集。实验表明，GREAT在攻击效果上显著优于基线方法。

    arXiv:2510.09260v3 Announce Type: replace-cross  Abstract: Recent work has shown that RLHF is highly susceptible to backdoor attacks. However, existing methods often rely on rare tokens or fixed triggers, limiting their impact in realistic scenarios. In this work, we develop GREAT, a novel framework for crafting natural distributional backdoors in RLHF. Specifically, GREAT targets harmful response generation for a vulnerable user subpopulation featured by semantically violent requests paired with emotionally angry triggers. At the core of our framework is a trigger identification pipeline that operates in the model's latent embedding space, leveraging dimensionality reduction and clustering techniques to identify representative triggers. To enable this, we introduce a hierarchical and diversity-driven prompting strategy to construct Erinyes, a high-quality dataset of over 5,000 angry triggers curated from GPT-4.1. Our experiments show that GREAT significantly outperforms baselines in a
    
[^189]: 大型推理模型从有缺陷的思考中学习更好的对齐

    Large Reasoning Models Learn Better Alignment from Flawed Thinking

    [https://arxiv.org/abs/2510.00938](https://arxiv.org/abs/2510.00938)

    RECAP 是一种基于强化学习的后训练方法，通过在合成生成的反向对齐（有缺陷）思维链预填充上训练，教模型识别并覆盖错误的推理轨迹、转向安全有用的回答，从而在不增加额外训练成本的情况下显著提升安全性与抗越狱鲁棒性，同时减少过度拒绝并保留核心推理能力。

    

    大型推理模型（LRMs）通过在给出最终答案之前生成结构化的思维链（CoT）来进行“思考”，然而它们仍然缺乏对安全对齐进行批判性推理的能力，并且当有缺陷的前提被注入其思考过程时容易被误导。我们提出了 RECAP（通过反向对齐预填充实现鲁棒安全对齐），这是一种有原则的强化学习（RL）后训练方法，明确地教模型覆盖有缺陷的推理轨迹，并重新路由到安全且有用的回答。RECAP 在合成生成的反向对齐 CoT 预填充和标准提示的混合数据上进行训练，除了原始的人类反馈强化学习（RLHF）之外，不需要额外的训练成本或修改，并且显著提升了安全性和抗越狱鲁棒性，减少了过度拒绝，同时保留了核心推理能力——所有这些都在保持推理 token 预算的前提下实现。广泛的分析……

    arXiv:2510.00938v3 Announce Type: replace  Abstract: Large reasoning models (LRMs) "think" by generating structured chain-of-thought (CoT) before producing a final answer, yet they still lack the ability to reason critically about safety alignment and are easily biased when a flawed premise is injected into their thought process. We propose RECAP (Robust Safety Alignment via Counter-Aligned Prefilling), a principled reinforcement learning (RL) method for post-training that explicitly teaches models to override flawed reasoning trajectories and reroute to safe and helpful responses. RECAP trains on a mixture of synthetically generated counter-aligned CoT prefills and standard prompts, requires no additional training cost or modifications beyond vanilla reinforcement learning from human feedback (RLHF), and substantially improves safety and jailbreak robustness, reduces overrefusal, and preserves core reasoning capability -- all while maintaining inference token budget. Extensive analysi
    
[^190]: OceanGym：面向水下具身智能体的基准测试环境

    OceanGym: A Benchmark Environment for Underwater Embodied Agents

    [https://arxiv.org/abs/2509.26536](https://arxiv.org/abs/2509.26536)

    OceanGym是首个面向水下具身智能体的综合性基准环境，涵盖八个真实任务领域和基于多模态大语言模型的统一智能体框架，实验揭示当前最先进的智能体与人类专家之间仍存在显著差距。

    

    我们提出了OceanGym，这是首个面向海洋水下具身智能体的综合性基准测试环境，旨在推动人工智能在最苛刻的真实环境之一中的发展。与陆地或空中领域不同，水下环境带来了极端的感知与决策挑战，包括低能见度和动态洋流，使得智能体的有效部署异常困难。OceanGym涵盖八个真实的任务领域，以及一个由多模态大语言模型（MLLM）驱动的统一智能体框架，该框架集成了感知、记忆和序列决策能力。智能体需要理解光学和声呐数据，自主探索复杂环境，并在这些恶劣条件下完成长程目标。大量实验表明，最先进的MLLM驱动智能体与人类专家之间存在显著差距，凸显了感知、规划等能力方面持续存在的困难。

    arXiv:2509.26536v3 Announce Type: replace  Abstract: We introduce OceanGym, the first comprehensive benchmark for ocean underwater embodied agents, designed to advance AI in one of the most demanding real-world environments. Unlike terrestrial or aerial domains, underwater settings present extreme perceptual and decision-making challenges, including low visibility, dynamic ocean currents, making effective agent deployment exceptionally difficult. OceanGym encompasses eight realistic task domains and a unified agent framework driven by Multi-modal Large Language Models (MLLMs), which integrates perception, memory, and sequential decision-making. Agents are required to comprehend optical and sonar data, autonomously explore complex environments, and accomplish long-horizon objectives under these harsh conditions. Extensive experiments reveal substantial gaps between state-of-the-art MLLM-driven agents and human experts, highlighting the persistent difficulty of perception, planning, and 
    
[^191]: 检验物理信息神经网络在反问题中对噪声的鲁棒性

    Examining the robustness of Physics-Informed Neural Networks to noise for Inverse Problems

    [https://arxiv.org/abs/2509.20191](https://arxiv.org/abs/2509.20191)

    本研究通过在一维Burgers方程和二维/三维Taylor-Green涡的粘度识别反问题上，将PINNs与结合数值优化器的有限元传统方法在含加性高斯噪声数据下的表现进行对比，系统检验了PINNs对噪声的鲁棒性。

    

    对偏微分方程（PDE）解的近似是科学与工程中动力系统建模的基础。物理信息神经网络（PINNs）是一种新兴的基于机器学习的方法，其许多性质和局限性仍然未知。人们普遍认为，PINNs在计算效率和求解精度上不如求解偏微分方程的传统方法（如有限元方法）。然而，人们通常声称PINNs在求解反问题以及处理含噪或不完整数据方面显示出潜力。我们将PINNs在求解反问题方面的性能与一种使用有限元方法结合数值优化器的传统方法进行了比较。这些模型在一维Burgers方程和二维/三维Taylor-Green涡的粘度识别任务上进行了测试，在所有情况下均对训练和验证数据施加了加性高斯噪声。我们发现，尽管……（原文摘要在此处截断）

    arXiv:2509.20191v2 Announce Type: replace-cross  Abstract: Approximating solutions to partial differential equations (PDEs) is fundamental for the modeling of dynamical systems in science and engineering. Physics-informed neural networks (PINNs) are a recent machine learning-based approach, for which many properties and limitations remain unknown.   PINNs are widely accepted as less computationally efficient and accurate than traditional methods for solving PDEs, such as the finite element method. However, PINNs are commonly claimed to show promise in solving inverse problems and handling noisy or incomplete data. We compare the performance of PINNs in solving inverse problems with that of a traditional approach using the finite element method combined with a numerical optimizer. The models are tested on viscosity identification in 1D Burgers' equation and in 2D/3D Taylor-Green Vortex, in all cases with additive Gaussian noise applied to training and validation data.   We find that whi
    
[^192]: 基于算子诱导与正则化符号森林的概率符号回归方程发现方法

    Probabilistic Symbolic Regression for Equation Discovery via Operator-induced and Regularized Symbolic Forests

    [https://arxiv.org/abs/2509.19710](https://arxiv.org/abs/2509.19710)

    该论文提出一种概率符号回归框架，将数学表达式表示为符号树集成，通过树拓扑上的正则化先验控制表达式复杂度，并利用基于奥卡姆窗口的后验摘要刻画多个合理符号模型的不确定性，为方程发现提供了兼具精度、简洁性与不确定性量化的统一解决方案。

    

    符号回归已成为人工智能驱动的科学发现的强大工具，它通过学习可解释的解析表达式，直接从数据中揭示变量间的支配性关系。然而，现有方法往往依赖启发式搜索，在噪声环境下难以平衡预测精度与表达式复杂度，且对符号不确定性的刻画十分有限。能够以统一方式解决这些挑战的概率化方法仍未得到充分探索。我们提出了一种概率符号回归框架，将数学表达式表示为符号树的集成。树拓扑结构上的正则化先验用于控制表达式复杂度，而基于奥卡姆窗口的后验摘要则用于捕捉多个合理符号模型之间的不确定性。鉴于符号回归领域现有的理论研究较为匮乏，我们进一步发展了后验集中性保证。

    arXiv:2509.19710v3 Announce Type: replace-cross  Abstract: Symbolic regression has emerged as a powerful tool for artificial intelligence-driven scientific discovery by learning interpretable analytical expressions that reveal governing relationships directly from data. Existing methods, however, often rely on heuristic search, struggle to balance predictive accuracy with expression complexity in noisy settings, and offer limited characterization of symbolic uncertainty. Probabilistic approaches that address these challenges in a unified manner remain underexplored. We introduce a probabilistic symbolic regression framework that represents mathematical expressions as ensembles of symbolic trees. A regularizing prior over tree topology controls expression complexity, while an Occam's window-based posterior summary captures uncertainty across multiple plausible symbolic models. Given the limited existing theoretical treatment of symbolic regression, we develop posterior concentration gua
    
[^193]: 先移位再学习：在强化学习中实现低秩表示

    Shift Before You Learn: Enabling Low-Rank Representations in Reinforcement Learning

    [https://arxiv.org/abs/2509.05193](https://arxiv.org/abs/2509.05193)

    该论文发现后继测度本身并非低秩，但跳过初始转移的“移位后继测度”自然具有低秩结构，并为其低秩估计提供了有限样本保证，误差由新提出的“谱可恢复性”指标刻画。

    

    低秩结构是许多现代强化学习（RL）算法中常见的隐含假设。例如，无奖励和目标条件的RL方法通常假设后继测度具有低秩表示。在这项工作中，我们首先指出后继测度本身并不近似低秩，从而对这一假设提出质疑。相反，我们证明了低秩结构自然地出现在移位后继测度中，该测度刻画了跳过若干初始转移之后的系统动态。我们为从采样条目中对移位后继测度的低秩近似进行逐条目估计提供了有限样本性能保证。我们的分析表明，近似误差和估计误差主要由一个新引入的量所主导：相应矩阵的谱可恢复性。为了对该参数进行界定，我们推导了一类新的函数……

    arXiv:2509.05193v3 Announce Type: replace  Abstract: Low-rank structure is a common implicit assumption in many modern reinforcement learning (RL) algorithms. For instance, reward-free and goal-conditioned RL methods often presume that the successor measure admits a low-rank representation. In this work, we challenge this assumption by first remarking that the successor measure itself is not approximately low-rank. Instead, we demonstrate that a low-rank structure naturally emerges in the shifted successor measure, which captures the system dynamics after bypassing a few initial transitions. We provide finite-sample performance guarantees for the entry-wise estimation of a low-rank approximation of the shifted successor measure from sampled entries. Our analysis reveals that both the approximation and estimation errors are primarily governed by a newly introduced quantitity: the spectral recoverability of the corresponding matrix. To bound this parameter, we derive a new class of funct
    
[^194]: 使用深度学习的《古兰经》学习者发音错误自动检测与纠正

    Automatic Pronunciation Error Detection and Correction of the Holy Quran's Learners Using Deep Learning

    [https://arxiv.org/abs/2509.00094](https://arxiv.org/abs/2509.00094)

    该论文提出了一套98%自动化的《古兰经》诵读数据构建流程，发布了848小时音频数据集（28.6万条标注语句）以及涵盖Tajweed规则的基准qdat_bench，实现了对《古兰经》学习者发音错误的自动检测与纠正。

    

    评估口语具有挑战性，而量化用于机器学习模型的发音指标更是难上加难。然而，对于《古兰经》而言，得益于穆斯林学者们建立的严谨诵读规则（Tajweed，泰吉维德），这一任务得以实现，使高效评估成为可能。尽管有这一优势，高质量标注数据的稀缺仍是一个重大障碍。在本工作中，我们通过引入以下内容来弥合这些差距：(1) 一套98%自动化的流程，用于生成高质量的《古兰经》数据集——包括从专业诵经师处收集诵读音频、使用我们微调的wav2vec2-BERT模型在停顿点（waqf）进行分割、对片段进行转录，以及通过我们新颖的Tasmeea算法进行转录验证；(2) 848小时音频（28.6万条标注语句）；(3) qdat_bench，一个涵盖音素、标音符号以及Tajweed规则（Ghunnah鼻音、Qalqalah弹音、Madd长音）的真实诵读基准数据集。

    arXiv:2509.00094v2 Announce Type: replace-cross  Abstract: Assessing spoken language is challenging, and quantifying pronunciation metrics for machine learning models is even harder. However, for the Holy Quran, this task is enabled by the rigorous recitation rules (Tajweed) established through the efforts of Muslim scholars, making highly effective assessment possible. Despite this advantage, the scarcity of high-quality annotated data remains a significant barrier. In this work, we bridge these gaps by introducing: (1) A 98% automated pipeline to produce high-quality Quranic datasets -- encompassing collection of recitations from expert reciters, segmentation at pause points (waqf) using our fine-tuned wav2vec2-BERT model, transcription of segments, and transcript verification via our novel Tasmeea algorithm; (2) 848 hours of audio (286K annotated utterances); (3) qdat_bench, a benchmark covering phonemes, diacritization, and Tajweed rules (Ghunnah, Qalqalah, Madd) on real recitation
    
[^195]: 基于自组织映射与合成重放的类增量持续学习

    Class Incremental Continual Learning with Self-Organizing Maps and Synthetic Replay

    [https://arxiv.org/abs/2508.21240](https://arxiv.org/abs/2508.21240)

    提出一种基于自组织映射（SOM）的生成式持续学习框架，通过为每个SOM单元存储分布统计量来生成合成样本进行重放，在不保存原始数据的情况下实现类增量学习。

    

    这项工作提出了一种新颖的生成式持续学习框架，该框架基于自组织映射（SOMs）——一种受大脑启发的自然计算模型，并通过学习到的分布统计信息和编码器-解码器模型对其进行了扩展，用于类增量持续学习。这些扩展的SOM能够以固定容量的统计记忆实现无样本重放，从而无需存储原始数据样本。对于高维输入空间，SOM在编码器-解码器的潜在空间上运行；而对于低维输入，SOM则以独立方式运行。我们的方法为每个SOM单元存储运行中的均值、方差和协方差，然后在未来的学习迭代中据此生成合成样本。对于编码器-解码器方法，生成的样本通过解码器输入，用于后续的重放。在标准类增量基准上的实验结果表明，我们的方法表现……

    arXiv:2508.21240v2 Announce Type: replace  Abstract: This work introduces a novel generative continual learning framework based on self-organizing maps (SOMs), a brain-inspired natural computing model, extended with learned distributional statistics and encoder--decoder models for class incremental continual learning. These extended SOMs enable exemplar-free replay with fixed-capacity statistical memory, eliminating the need to store raw data samples. For high-dimensional input spaces, the SOM operates over the latent space of the encoder--decoder, while for lower-dimensional inputs, the SOM operates in a standalone fashion. Our method stores a running mean, variance, and covariance for each SOM unit, from which synthetic samples are then generated during future learning iterations. For the encoder--decoder method, generated samples are fed through the decoder to be used in subsequent replay. Experimental results on standard class-incremental benchmarks show that our approach performs 
    
[^196]: 注意力即条件反射：经典学习理论对线性Transformer的预测

    Attention as Conditioning: What Classical Learning Theory Predicts About Linear Transformers

    [https://arxiv.org/abs/2508.08289](https://arxiv.org/abs/2508.08289)

    论文揭示线性注意力的状态更新与动物学习理论中的经典模型（Hebbian接近性、Rescorla-Wagner误差校正、刺激痕迹衰减）逐项等价，从而将条件反射现象转化为线性Transformer上下文行为的可检验预测。

    

    注意力被广泛理解为一种联想记忆，但仅凭这一描述无法预测该记忆将如何表现。预测性理论确实存在，但存在于动物学习理论的文献中。我们证明了主要线性注意力家族的状态更新与一个世纪以来动物学习理论中的经典模型逐项完全一致：线性注意力实现了赫布（Hebbian）接近性学习，DeltaNet实现了Rescorla-Wagner误差校正，而RetNet等衰减变体则实现了带有刺激痕迹的接近性学习。这一对应词典将条件反射现象转化为关于线性Transformer上下文内行为的可检验命题，同时将代数推论与实证测量区分开来。在代数层面，它给出了Kamin阻断效应的精确闭式解，并在五个学习率下的仿真验证中误差小于$10^{-7}$。在实证层面，它预测了一种在通用上下文数据上训练后依然存在的分离现象……

    arXiv:2508.08289v3 Announce Type: replace  Abstract: Attention is widely understood as an associative memory, but that description alone does not predict how the memory will behave. Predictive theories do exist, but in the literature on animal learning. We show that the state updates of the major linear-attention families are term-for-term identical with named models from a century of animal learning theory: linear attention implements Hebbian contiguity, DeltaNet implements Rescorla--Wagner error correction, and decay variants such as RetNet implement contiguity with a stimulus trace. This dictionary turns conditioning phenomena into testable statements about the in-context behavior of linear transformers, while distinguishing algebraic consequences from empirical measurements. Algebraically, it yields an exact closed form for Kamin blocking, verified in simulation to $<10^{-7}$ across five learning rates. Empirically, it predicts a dissociation that survives training on generic in-co
    
[^197]: RegCL：面向多感官媒体中视觉定位的紧凑式持续SAM适配

    RegCL: Compact Continual SAM Adaptation for Visual Grounding in Multi-Sensorial Media

    [https://arxiv.org/abs/2507.12297](https://arxiv.org/abs/2507.12297)

    提出RegCL框架，通过增量模型合并将多领域分割知识整合进单个SAM适配器，无需重放数据即可实现多感官媒体中紧凑高效的持续视觉定位适配。

    

    多感官媒体系统，包括AR/VR、远程操作和具身智能，需要视觉定位模块在感知环境和应用领域不断演变的情况下保持可靠性。Segment Anything Model（SAM）为密集视觉分割提供了强大的基础，但其在专业化且动态出现的领域（如医学影像、伪装场景和阴影主导的环境）上性能会下降。现有的持续学习方法通常依赖重放数据或不断增长的领域特定模块，限制了在不断演变的媒体流水线中的紧凑部署。为解决这一问题，我们提出RegCL，一种无需重放的持续适配框架，通过增量模型合并将多领域分割知识整合到单个SAM适配器中。RegCL通过优化合并模型与领域特定模块之间的预测一致性，将轻量级适配模块（如LoRA风格的AugModules）进行合并……（摘要内容在此处被截断）

    arXiv:2507.12297v2 Announce Type: replace  Abstract: Multi-sensorial media systems, including AR/VR, remote operation, and embodied AI, require visual grounding modules that remain reliable as sensing environments and application domains evolve. The Segment Anything Model (SAM) provides a strong foundation for dense visual segmentation, but its performance degrades on specialized and dynamically arriving domains such as medical imagery, camouflaged scenes, and shadow-dominant environments. Existing continual learning methods often rely on replay data or growing domain-specific modules, limiting compact deployment in evolving media pipelines. To address this issue, we propose RegCL, a non-replay continual adaptation framework that consolidates multi-domain segmentation knowledge into a single SAM adapter through incremental model merging. RegCL merges lightweight adaptation modules, e.g., LoRA-style AugModules, by optimizing prediction consistency between the merged model and domain-spe
    
[^198]: Ampere：高效通信与高精度的分割联邦学习

    Ampere: Communication-Efficient and High-Accuracy Split Federated Learning

    [https://arxiv.org/abs/2507.07130](https://arxiv.org/abs/2507.07130)

    Ampere是一种新型分割联邦学习系统，通过基于局部损失的单向块间训练消除了梯度传输，在最小化设备端计算和通信开销的同时，提高了非独立同分布数据下的模型精度。

    

    联邦学习（FL）系统在设备和服务器之间协同训练神经网络，但受限于设备端显著的计算成本。分割联邦学习（SFL）系统通过将网络的一部分层从设备卸载到服务器来缓解这一问题。然而，这样做由于设备和服务器之间频繁交换中间激活值和梯度而引入了巨大的通信开销，并且降低了非独立同分布数据下的模型精度。我们提出了Ampere，一种新型的协同训练系统，它在最小化设备端计算和设备-服务器通信的同时，还能提高模型精度。与SFL采用迭代端到端训练的全局损失不同，Ampere开发了单向块间训练方法，使用局部损失依次训练设备块和服务器块，从而消除了梯度的传输。一个轻量级的辅助网络生成...

    arXiv:2507.07130v2 Announce Type: replace-cross  Abstract: A Federated Learning (FL) system collaboratively trains neural networks across devices and a server but is limited by significant on-device computation costs. Split Federated Learning (SFL) systems mitigate this by offloading a block of layers of the network from the device to a server. However, in doing so, it introduces large communication overheads due to frequent exchanges of intermediate activations and gradients between devices and the server and reduces model accuracy for non-IID data. We propose Ampere, a novel collaborative training system that simultaneously minimizes on-device computation and device-server communication while improving model accuracy. Unlike SFL, which uses a global loss by iterative end-to-end training, Ampere develops unidirectional inter-block training to sequentially train the device and server blocks with a local loss, eliminating the transfer of gradients. A lightweight auxiliary network genera
    
[^199]: 基于大语言模型的序贯决策的元提示词优化

    Meta-Prompt Optimization for LLM-Based Sequential Decision Making

    [https://arxiv.org/abs/2502.00728](https://arxiv.org/abs/2502.00728)

    该论文提出EXPO算法，借鉴对抗性老虎机算法处理非平稳奖励观测的能力，实现基于大语言模型的序贯决策智能体元提示词的自动优化。

    

    大语言模型（LLMs）最近被用作智能体来解决序贯决策任务，例如贝叶斯优化和多臂老虎机（MAB）。这些工作通常通过为LLM提供一个固定的、人工设计的元提示词来完成序贯动作选择。然而，许多先前的研究发现，提示词对LLM的性能有显著影响，这就需要一种方法来自动优化基于LLM的智能体的元提示词。遗憾的是，在基于LLM的序贯决策过程中，奖励观测的非平稳性使得元提示词优化极具挑战性。为了应对这一挑战，我们从对抗性老虎机算法中汲取灵感，这类算法天生能够处理非平稳的奖励观测。在此基础上，我们提出了用于提示词优化的指数权重算法（EXPO），以自动优化……

    arXiv:2502.00728v2 Announce Type: replace  Abstract: Large language models (LLMs) have recently been employed as agents to solve sequential decision-making tasks such as Bayesian optimization and multi-armed bandits (MAB). These works usually adopt an LLM for sequential action selection by providing it with a fixed, manually designed meta-prompt. However, numerous previous works have found that the prompt has a significant impact on the performance of the LLM, which calls for a method to automatically optimize the meta-prompt for LLM-based agents. Unfortunately, the non-stationarity in the reward observations during LLM-based sequential decision-making makes meta-prompt optimization highly challenging. To address this challenge, we draw inspirations from adversarial bandit algorithms, which are inherently capable of handling non-stationary reward observations. Building on this foundation, we propose our EXPonential-weight algorithm for prompt Optimization} (EXPO) to automatically optim
    
[^200]: 偏离正态之路：学习节点移动性的空间密度模型

    Off the Normal Path: Learning Spatial Density Models of Node Mobility

    [https://arxiv.org/abs/2411.10997](https://arxiv.org/abs/2411.10997)

    该论文引入Möbius分布混合模型来学习二维地形上移动节点的稳态空间密度，相比混合密度网络和归一化流等现成方法，提供了更可解释、更简洁且性能相当或更优的模型。

    

    我们研究学习空间密度函数模型的问题，该函数表示在二维地形上移动的移动节点的稳态密度。推导此类模型可以辅助网络设计与优化问题，例如在参数扫描过程中加速密度函数的计算。我们探讨了现成的混合密度网络模型以及两种类型的归一化流在描述圆盘上移动节点密度方面的适用性。我们引入了Möbius分布来保持对称的空间关系。我们的结果表明，Möbius分布的混合为所研究的稳态密度分布提供了可解释且简洁的模型，其性能与替代方法相当或更优。

    arXiv:2411.10997v2 Announce Type: replace-cross  Abstract: We consider the problem of learning models of spatial density functions, representing the steady-state density of mobile nodes moving on a two-dimensional terrain. Deriving such models can assist in network design and optimization problems, e.g., by accelerating the computation of the density function during a parameter sweep. We address the question of applicability of off-the-shelf mixture density network models and of, two varieties of, normalizing flows for the description of mobile node density over a disk. We introduce the use of M\"obius distributions to retain symmetric spatial relations. Our results indicate that mixtures of M\"obius distributions provide interpretable, parsimonious models for the studied steady state density distributions, that match or outperform the alternatives.
    
[^201]: 多模态人工智能中用于去偏放疗靶区勾画的多中心专家混合框架

    Mixture of Multicenter Experts in Multimodal AI for Debiased Radiotherapy Target Delineation

    [https://arxiv.org/abs/2410.00046](https://arxiv.org/abs/2410.00046)

    提出多中心专家混合框架，无需跨机构数据共享即可整合多中心临床专业知识、解决医学AI偏见，在前列腺癌放疗靶区勾画任务中显著提升了模型的泛化能力和适应性。

    

    临床决策反映了受区域患者群体和机构诊疗规范影响的多样化策略。然而，大多数现有的医学人工智能（AI）模型都是在高度普遍的数据模式上进行训练的，这加剧了偏见，并且无法捕捉临床专业知识的广度。受近期专家混合技术的启发，我们提出了多中心专家混合框架，以解决医学领域的AI偏见问题，而无需跨机构共享数据。MoME整合了来自不同临床策略的专业知识，以增强模型在各个医疗中心之间的泛化能力和适应性。我们使用前列腺癌放疗的多模态靶体积勾画模型对该框架进行了验证。通过结合来自每个中心的影像和临床记录进行少样本训练，该模型的表现优于基线方法，尤其是在机构多样性较高的场景中表现突出。

    arXiv:2410.00046v4 Announce Type: replace-cross  Abstract: Clinical decision-making reflects diverse strategies shaped by regional patient populations and institutional protocols. However, most existing medical artificial intelligence (AI) models are trained on highly prevalent data patterns, which reinforces biases and fails to capture the breadth of clinical expertise. Inspired by the recent advances in Mixture of Experts (MoE), we propose a Mixture of Multicenter Experts (MoME) framework to address AI bias in the medical domain without requiring data sharing across institutions. MoME integrates specialized expertise from diverse clinical strategies to enhance model generalizability and adaptability across medical centers. We validate this framework using a multimodal target volume delineation model for prostate cancer radiotherapy. With few-shot training that combines imaging and clinical notes from each center, the model outperformed baselines, particularly in settings with high in
    
[^202]: 重新思考用于语音生成的说话人嵌入：捕捉说话人内部多样性的子中心建模

    Rethinking Speaker Embeddings for Speech Generation: Sub-Center Modeling for Capturing Intra-Speaker Diversity

    [https://arxiv.org/abs/2407.04291](https://arxiv.org/abs/2407.04291)

    该论文提出一种说话人嵌入的子中心建模框架，通过为每位说话人学习多个子中心而非单一原型来保留说话人内部的结构化多样性，从而在零样本语音转换中提升可懂度、音高变化性和自然度，同时保持说话人验证性能。

    

    对语音变化进行建模是实现自然、富有表现力语音生成的关键。说话人嵌入通常被用于为个性化语音系统提供条件控制，但它们通常是为说话人识别任务训练的，在这种任务中，说话人内部的变异性被抑制，而说话人之间的区分度被最大化。这一目标导致表征过于紧凑，可能会丢弃对生成至关重要的变化信息。我们重新审视了这一设计选择，并提出了一种针对说话人嵌入的子中心建模框架。不同于为每位说话人设置单一原型，我们在判别式训练过程中学习多个子中心，使语音片段能够与不同的原型对齐。这一策略在保持区分度的同时，保留了结构化的说话人内部变异性。在零样本语音转换任务中，我们的方法提高了可懂度，增加了音高变化性，获得了更高的自然度评分，并保持了强大的说话人验证性能。

    arXiv:2407.04291v4 Announce Type: replace-cross  Abstract: Modeling speech variation is key to natural, expressive generation. Speaker embeddings are commonly used to condition personalized speech systems, but they are typically trained for speaker recognition, where intra-speaker variability is suppressed and inter-speaker separation is maximized. This objective leads to overly compact representations that may discard variations crucial for generation. We revisit this design choice and propose a sub-center modeling framework for speaker embeddings. Instead of a single prototype per speaker, we learn multiple sub-centers during discriminative training, allowing utterances to align with different prototypes. This strategy preserves structured intra-speaker variability while maintaining discriminability. In zero-shot voice conversion, our method improves intelligibility, increases pitch variability, achieves higher naturalness ratings, and retains strong speaker verification performance.
    
[^203]: 在视觉、语言与控制任务中摊销扩散模型中的难解推理

    Amortizing intractable inference in diffusion models for vision, language, and control

    [https://arxiv.org/abs/2405.20971](https://arxiv.org/abs/2405.20971)

    本文提出“相对轨迹平衡”这一具有渐近正确性保证的无数据学习目标，借助生成流网络视角与深度强化学习技术，训练扩散模型以摊销方式从扩散先验与黑盒约束构成的后验分布中进行精确采样，从而解决视觉、语言与控制任务中的难解后验推断问题。

    

    扩散模型已成为视觉、语言和强化学习中有效的分布估计器，但将其作为先验用于下游任务时会带来一个难解的后验推断问题。本文研究了数据后验分布 p^post(x) ∝ p(x)r(x) 的摊销采样问题，其中模型由扩散生成模型先验 p(x) 和黑盒约束或似然函数 r(x) 组成。我们提出并证明了一个无数据学习目标——相对轨迹平衡的渐近正确性，用于训练能从该后验分布采样的扩散模型；这一问题现有方法只能近似求解或仅在受限情形下求解。相对轨迹平衡源于从生成流网络视角对扩散模型的理解，这使得可以利用深度强化学习技术来改进模式覆盖。

    arXiv:2405.20971v3 Announce Type: replace  Abstract: Diffusion models have emerged as effective distribution estimators in vision, language, and reinforcement learning, but their use as priors in downstream tasks poses an intractable posterior inference problem. This paper studies amortized sampling of the posterior over data, $\mathbf{x}\sim p^{\rm post}(\mathbf{x})\propto p(\mathbf{x})r(\mathbf{x})$, in a model that consists of a diffusion generative model prior $p(\mathbf{x})$ and a black-box constraint or likelihood function $r(\mathbf{x})$. We state and prove the asymptotic correctness of a data-free learning objective, relative trajectory balance, for training a diffusion model that samples from this posterior, a problem that existing methods solve only approximately or in restricted cases. Relative trajectory balance arises from the generative flow network perspective on diffusion models, which allows the use of deep reinforcement learning techniques to improve mode coverage. Ex
    
[^204]: 关于分散推断模型的扩散模型：基准测试和改进随机控制和采样

    On diffusion models for amortized inference: Benchmarking and improving stochastic control and sampling

    [https://arxiv.org/abs/2402.05098](https://arxiv.org/abs/2402.05098)

    本研究探讨了训练扩散模型以从给定分布中采样的问题，并针对随机控制和采样提出了一种新的探索策略，通过基准测试比较了不同推断方法的相对优劣，并对过去的工作提出了质疑。

    

    我们研究了训练扩散模型以从给定的非标准化密度或能量函数分布中采样的问题。我们对几种扩散结构推断方法进行了基准测试，包括基于模拟的变分方法和离策略方法（连续生成流网络）。我们的结果揭示了现有算法的相对优势，同时对过去的研究提出了一些质疑。我们还提出了一种新颖的离策略方法探索策略，基于目标空间中的局部搜索和回放缓冲区的使用，并证明它可以改善各种目标分布上的样本质量。我们研究的采样方法和基准测试的代码已公开在https://github.com/GFNOrg/gfn-diffusion，作为未来在分散推断模型上工作的基础。

    We study the problem of training diffusion models to sample from a distribution with a given unnormalized density or energy function. We benchmark several diffusion-structured inference methods, including simulation-based variational approaches and off-policy methods (continuous generative flow networks). Our results shed light on the relative advantages of existing algorithms while bringing into question some claims from past work. We also propose a novel exploration strategy for off-policy methods, based on local search in the target space with the use of a replay buffer, and show that it improves the quality of samples on a variety of target distributions. Our code for the sampling methods and benchmarks studied is made public at https://github.com/GFNOrg/gfn-diffusion as a base for future work on diffusion models for amortized inference.
    
[^205]: 扩散模型作为即插即用的先验

    Diffusion models as plug-and-play priors

    [https://arxiv.org/abs/2206.09012](https://arxiv.org/abs/2206.09012)

    本文提出将独立训练的扩散模型作为即插即用的先验模块，通过与可微辅助约束结合并在去噪网络上进行迭代微分实现近似推理，从而支持条件生成、图像分割等新任务的应用。

    

    我们考虑在由先验 p(x) 和给定附加信息 y 时对 x 施加的辅助可微约束 c(x,y) 所组成的模型中推断高维数据 x 的问题。在本文中，先验是一个独立训练的去噪扩散生成模型。辅助约束需要具有可微形式，但可以来自多种不同的来源。这种推理的可能性使扩散模型成为即插即用的模块，从而支持在将模型适配到新领域和新任务方面的广泛潜在应用，例如条件生成或图像分割。扩散模型的结构使我们能够通过在固定的去噪网络上进行迭代微分来执行近似推理，其中每一步注入不同量的噪声，并在评估 x 时考虑其多个加噪版本。

    arXiv:2206.09012v4 Announce Type: replace  Abstract: We consider the problem of inferring high-dimensional data $\mathbf{x}$ in a model that consists of a prior $p(\mathbf{x})$ and an auxiliary differentiable constraint $c(\mathbf{x},\mathbf{y})$ on $x$ given some additional information $\mathbf{y}$. In this paper, the prior is an independently trained denoising diffusion generative model. The auxiliary constraint is expected to have a differentiable form, but can come from diverse sources. The possibility of such inference turns diffusion models into plug-and-play modules, thereby allowing a range of potential applications in adapting models to new domains and tasks, such as conditional generation or image segmentation. The structure of diffusion models allows us to perform approximate inference by iterating differentiation through the fixed denoising network enriched with different amounts of noise at each step. Considering many noised versions of $\mathbf{x}$ in evaluation of its fi
    
[^206]: 期望进球模型中的偏见影响了射门能力

    Biases in Expected Goals Models Confound Finishing Ability. (arXiv:2401.09940v1 [cs.LG])

    [http://arxiv.org/abs/2401.09940](http://arxiv.org/abs/2401.09940)

    本研究旨在解决使用期望进球（xG）统计评估射门能力时的限制和偏见。研究发现，持续超出累积xG需要高射门频率。

    

    期望进球（xG）已成为评估足球分析中射门技能的一种常用工具。它涉及将球员的累积xG与实际进球数量进行比较，如果持续表现超出预期，则表明射门能力强。然而，使用xG统计评估足球射门技能仍存在争议，因为球员很难在持续表现中超出累积xG。本文旨在解决使用xG统计评估射门能力时的限制和细微差别。具体而言，我们探讨了三个假设：（1）实际进球和预期进球之间的偏差是一个不足的度量标准，因为射门结果的差异和样本量有限，（2）在累积xG计算中包含所有射门可能不合适，并且（3）xG模型中存在数据相关性引起的偏见，影响了技能测量。我们发现，持续超出累积xG需要高射门频率。

    Expected Goals (xG) has emerged as a popular tool for evaluating finishing skill in soccer analytics. It involves comparing a player's cumulative xG with their actual goal output, where consistent overperformance indicates strong finishing ability. However, the assessment of finishing skill in soccer using xG remains contentious due to players' difficulty in consistently outperforming their cumulative xG. In this paper, we aim to address the limitations and nuances surrounding the evaluation of finishing skill using xG statistics. Specifically, we explore three hypotheses: (1) the deviation between actual and expected goals is an inadequate metric due to the high variance of shot outcomes and limited sample sizes, (2) the inclusion of all shots in cumulative xG calculation may be inappropriate, and (3) xG models contain biases arising from interdependencies in the data that affect skill measurement. We found that sustained overperformance of cumulative xG requires both high shot volume
    
[^207]: 单一生成流网络中的图结构与参数的联合贝叶斯推理

    Joint Bayesian Inference of Graphical Structure and Parameters with a Single Generative Flow Network. (arXiv:2305.19366v1 [cs.LG])

    [http://arxiv.org/abs/2305.19366](http://arxiv.org/abs/2305.19366)

    本文提出了在单一生成流网络中联合建模贝叶斯网络结构和参数的方法，包括非离散样本空间，提高了贝叶斯网络局部概率模型的灵活性。

    

    生成流网络是一类对离散和结构化样本空间进行建模的生成模型。先前的研究已将其应用于推断给定观测数据的贝叶斯网络的有向无环图（DAG）的边缘后验分布。本文基于最近的研究进展，在非离散样本空间上将此框架扩展到联合后验分布的建模，不仅包括贝叶斯网络的结构，还考虑了其条件概率分布的参数。

    Generative Flow Networks (GFlowNets), a class of generative models over discrete and structured sample spaces, have been previously applied to the problem of inferring the marginal posterior distribution over the directed acyclic graph (DAG) of a Bayesian Network, given a dataset of observations. Based on recent advances extending this framework to non-discrete sample spaces, we propose in this paper to approximate the joint posterior over not only the structure of a Bayesian Network, but also the parameters of its conditional probability distributions. We use a single GFlowNet whose sampling policy follows a two-phase process: the DAG is first generated sequentially one edge at a time, and then the corresponding parameters are picked once the full structure is known. Since the parameters are included in the posterior distribution, this leaves more flexibility for the local probability models of the Bayesian Network, making our approach applicable even to non-linear models parametrized
    
[^208]: 利用GFlowNets解决图形组合优化问题

    Let the Flows Tell: Solving Graph Combinatorial Optimization Problems with GFlowNets. (arXiv:2305.17010v1 [cs.LG])

    [http://arxiv.org/abs/2305.17010](http://arxiv.org/abs/2305.17010)

    本文提出了一种名为GFlowNets的机器，可以有效地解决组合优化问题，同时在训练方面进行了优化，结果表明其可以高效地找到高质量的解决方案。

    

    组合优化问题通常是NP难题，因此不适用于精确算法，这使它们成为应用机器学习方法的理想领域。这些问题中高度结构化的限制可能会直接阻碍优化或采样解决方案的空间。另一方面，GFlowNets最近被发现是一种强大的机器，可以顺序地从复合非规范化密度中有效地采样，并具有在CO中分摊此类解决方案搜索过程以及生成不同的解决方案候选项的潜力。在本文中，我们设计了适用于不同组合问题的马尔科夫决策过程（MDP），并提出训练有条件的GFlowNets从解空间中采样的策略。还开发了高效的训练技术来受益于远程信用分配。通过对各种使用合成和实际数据的不同CO任务的广泛实验，我们证明了GFlowNet策略可以有效地找到高质量的解。

    Combinatorial optimization (CO) problems are often NP-hard and thus out of reach for exact algorithms, making them a tempting domain to apply machine learning methods. The highly structured constraints in these problems can hinder either optimization or sampling directly in the solution space. On the other hand, GFlowNets have recently emerged as a powerful machinery to efficiently sample from composite unnormalized densities sequentially and have the potential to amortize such solution-searching processes in CO, as well as generate diverse solution candidates. In this paper, we design Markov decision processes (MDPs) for different combinatorial problems and propose to train conditional GFlowNets to sample from the solution space. Efficient training techniques are also developed to benefit long-range credit assignment. Through extensive experiments on a variety of different CO tasks with synthetic and realistic data, we demonstrate that GFlowNet policies can efficiently find high-quali
    
[^209]: 自动驾驶中基于Transformer的模型及其硬件加速分析：综述 (arXiv:2304.10891v1 [cs.LG])

    Transformer-based models and hardware acceleration analysis in autonomous driving: A survey. (arXiv:2304.10891v1 [cs.LG])

    [http://arxiv.org/abs/2304.10891](http://arxiv.org/abs/2304.10891)

    本文综述了基于Transformer的模型在自动驾驶中的应用，探讨了不同体系结构和运算符的优缺点，重点讨论了针对便携计算平台的硬件加速方案，并对卷积神经网络和Transformer的层进行了对比。

    

    近年来，Transformer架构在各种自动驾驶应用中表现出了很好的性能。另一方面，将其专门用于便携式计算平台的硬件加速已成为实际部署在真实自动汽车中的下一步关键步骤。本综述论文提供了针对自动驾驶任务的基于Transformer的模型的全面概述、基准和分析，例如车道检测、分割、跟踪、规划和决策制定。我们审查了不同的体系结构，用于组织Transformer的输入和输出，例如编码器-解码器和仅编码器结构，并探讨了它们各自的优缺点。此外，我们深入讨论了Transformer相关的运算符及其硬件加速方案，考虑到关键因素，如量化和运行时。我们特别在移动和桌面平台上对卷积神经网络的层与基于Transformer的模型的运算符进行了对比。总的来说，本综述论文为研究人员和从业者提供了系统的指南，以了解基于Transformer的模型及其在自动驾驶中的硬件加速的当前进展和挑战。

    Transformer architectures have exhibited promising performance in various autonomous driving applications in recent years. On the other hand, its dedicated hardware acceleration on portable computational platforms has become the next critical step for practical deployment in real autonomous vehicles. This survey paper provides a comprehensive overview, benchmark, and analysis of Transformer-based models specifically tailored for autonomous driving tasks such as lane detection, segmentation, tracking, planning, and decision-making. We review different architectures for organizing Transformer inputs and outputs, such as encoder-decoder and encoder-only structures, and explore their respective advantages and disadvantages. Furthermore, we discuss Transformer-related operators and their hardware acceleration schemes in depth, taking into account key factors such as quantization and runtime. We specifically illustrate the operator level comparison between layers from convolutional neural ne
    
[^210]: 轨迹平衡：改进了GFlowNets中的信用分配

    Trajectory balance: Improved credit assignment in GFlowNets. (arXiv:2201.13259v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2201.13259](http://arxiv.org/abs/2201.13259)

    GFlowNets使用轨迹平衡作为一种更高效的学习目标，解决了先前学习目标中信用传播效率低下的问题，并且在实验中证明了其在收敛性、生成样本多样性以及鲁棒性方面的优势。

    

    生成流网络（GFlowNets）是一种学习使用动作序列生成组合对象（如图形或字符串）的随机策略的方法，其中许多可能的动作序列可能导致相同的对象。我们发现先前提出的GFlowNets学习目标，即流匹配和详细平衡，类似于时间差分学习，容易在长的动作序列中出现信用传播效率低下的问题。因此，我们提出了一种新的学习目标，即轨迹平衡，作为先前使用目标的更高效的替代方法。我们证明了轨迹平衡目标的任何全局极小值可以定义一个从目标分布精确采样的策略。在四个不同领域的实验中，我们从实证上证明了轨迹平衡目标对于GFlowNet收敛性、生成样本的多样性以及对长动作序列和噪声的鲁棒性的益处。

    Generative flow networks (GFlowNets) are a method for learning a stochastic policy for generating compositional objects, such as graphs or strings, from a given unnormalized density by sequences of actions, where many possible action sequences may lead to the same object. We find previously proposed learning objectives for GFlowNets, flow matching and detailed balance, which are analogous to temporal difference learning, to be prone to inefficient credit propagation across long action sequences. We thus propose a new learning objective for GFlowNets, trajectory balance, as a more efficient alternative to previously used objectives. We prove that any global minimizer of the trajectory balance objective can define a policy that samples exactly from the target distribution. In experiments on four distinct domains, we empirically demonstrate the benefits of the trajectory balance objective for GFlowNet convergence, diversity of generated samples, and robustness to long action sequences and
    

