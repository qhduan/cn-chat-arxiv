# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DanceOPD: On-Policy Generative Field Distillation](https://arxiv.org/abs/2606.27377) | DanceOPD提出了一种在线策略生成场蒸馏框架，通过将不同图像生成能力（文生图、局部编辑、全局编辑）建模为共享空间中的速度场，并利用学生自身状态进行查询和训练，有效解决了多种能力之间的冲突与组合问题。 |
| [^2] | [Reinforcement Learning without Ground-Truth Solutions can Improve LLMs](https://arxiv.org/abs/2606.27369) | 提出RiVER框架，通过校准的连续奖励塑造（克服尺度主导和频率主导问题），使大语言模型能在无真实解法情况下进行强化学习训练。 |
| [^3] | [Autoregressive Boltzmann Generators](https://arxiv.org/abs/2606.27361) | 提出自回归玻尔兹曼生成器（ArBG），通过自回归建模替代归一化流，克服了流模型在拓扑约束和计算成本上的局限，实现更高效灵活的分子平衡态采样。 |
| [^4] | [When are likely answers right? On Sequence Probability and Correctness in LLMs](https://arxiv.org/abs/2606.27359) | 本文发现序列概率仅在固定数据集内能预测跨提示-答案对的正确性，但不能推广到跨解码方法或超参数调整的解码决策中。 |
| [^5] | [Error-Conditioned Neural Solvers](https://arxiv.org/abs/2606.27354) | 本文提出误差条件神经求解器（ENS），通过将偏微分方程残差场作为网络直接输入而非优化目标，解决了现有方法在病态系统中低残差却预测不准的问题。 |
| [^6] | [Understanding Domain-Aware Distribution Alignment in Budgeted Entity Matching](https://arxiv.org/abs/2606.27342) | 本文通过系统实验揭示了BEACON框架在预算约束下进行领域感知分布对齐的性能表现，为低资源实体匹配提供了关键见解。 |
| [^7] | [Empowering GUI Agents via Autonomous Experience Exploration and Hindsight Experience Utilization for Task Planning](https://arxiv.org/abs/2606.27330) | 本文提出PEEU方法，通过自主探索和事后经验利用生成高层次训练数据，并结合TDHAF框架分析，揭示掌握低层原子技能对提升GUI代理任务规划泛化能力的关键作用。 |
| [^8] | [Hallucination in World Models is Predictable and Preventable](https://arxiv.org/abs/2606.27326) | 本文提出通过检测状态-动作空间的低覆盖区域来预测和预防世界模型中的幻觉，并开发了覆盖感知采样和在线预测器两种方法，显著提升了长期预测的保真度。 |
| [^9] | [Beyond the Hard Budget: Sparsity Regularizers for More Interpretable Top-k Sparse Autoencoders](https://arxiv.org/abs/2606.27321) | 本文针对Top-k稀疏自编码器固定预算和过拟合缺陷，提出了两种新的稀疏正则化方法，通过作用于Top-k选择前的激活值来提升模型可解释性。 |
| [^10] | [Blackwell Approachability and Gradient Equilibrium are Equivalent](https://arxiv.org/abs/2606.27315) | 本文证明梯度均衡（GEQ）与布莱克韦尔可逼近性在算法上等价，并由此推出GEQ与遗憾最小化、校准等框架等价，从而统一了在线学习中的多个重要概念。 |
| [^11] | [A Multi-Fidelity Convolutional Autoencoder-Transfer Learning Framework for Guided-Wave-Based Damage Diagnosis Using Large Simulated and Limited Experimental Datasets](https://arxiv.org/abs/2606.27304) | 提出了一种结合轻量级仿真与有限实验数据的多保真度迁移学习框架，通过卷积自编码器提取特征，解决了导波损伤诊断中实验数据稀缺和仿真计算成本高的问题，实现了精确的损伤定位与尺寸评估。 |
| [^12] | [Fast algorithms for learning a Gaussian under halfspace truncation with optimal sample complexity](https://arxiv.org/abs/2606.27298) | 提出了一种在半空间截断下学习高斯分布的算法，其样本和时间复杂度均达到理论最优，且与无截断情况下的最优复杂度一致。 |
| [^13] | [Generative Models on Analog Hardware with Dynamics](https://arxiv.org/abs/2606.27294) | 本文提出模拟交互系统框架，通过时变分段参数和隐藏物理状态机制，并借助Wasserstein GAN训练，弥合模拟硬件与生成模型之间的动力学不匹配。 |
| [^14] | [Designing Reward Signals for Portable Query Generation: A Case Study in Industrial Semantic Job Search](https://arxiv.org/abs/2606.27291) | 本文提出一种基于AI反馈的强化学习框架，通过设计稳健的奖励信号而非优化算法，来生成可迁移的求职搜索查询，解决了策略优化易利用大语言模型评判缺陷导致退化行为的问题。 |
| [^15] | [When Does Combining Language Models Help? A Co-Failure Ceiling on Routing, Voting, and Mixture-of-Agents Across 67 Frontier Models](https://arxiv.org/abs/2606.27288) | 本研究揭示了多模型LLM系统（如路由、投票等）的准确率提升受限于所有模型在相同查询上同时出错的比率β，且β无法通过常用的平均成对误差相关系数ρ识别，基于67个模型的实验表明β常被低估，从而限制了组合策略的实际增益。 |
| [^16] | [Recovering Governing Equations from Solution Data: Identifiability Bounds for Linear and Nonlinear ODEs](https://arxiv.org/abs/2606.27285) | 本文通过引入豪斯多夫距离作为度量，首次为从解数据中恢复线性和非线性常微分方程建立了定量可辨识性界，填补了理论空白。 |
| [^17] | [How Good Can Linear Models Be for Time-Series Forecasting?](https://arxiv.org/abs/2606.27282) | 本文挑战了“更大模型容量带来更高精度”的主流观点，通过岭回归证明，精心调整预处理（如上下文长度、归一化）能以极低成本显著缩小与大型模型的性能差距，并揭示了最优回溯长度与预测范围的非单调关系等反直觉模式。 |
| [^18] | [BetXplain: An Explanation-Annotated Dataset for Detecting Manipulative Betting Advertisements on Social Media](https://arxiv.org/abs/2606.27274) | 本文提出了BetXplain数据集，通过人工标注社交媒体上的博彩广告并附带解释，为自动检测操纵性和欺骗性广告提供了可解释的研究基础。 |
| [^19] | [Ribbon: Scalable Approximation and Robust Uncertainty Quantification](https://arxiv.org/abs/2606.27269) | Ribbon通过影响函数线性化替代重复重拟合，实现了对贝叶斯自助不确定性的高效近似，并引入可调集中参数实现校准。 |
| [^20] | [RSPC: A Benchmark for Modeling Stress and Psychiatric Conditions in Digitally Mediated Relationships using Psychiatrist Annotations](https://arxiv.org/abs/2606.27247) | 该论文提出了一个由精神病学家标注的Reddit异地恋帖子数据集（RSPC），用于在人际背景下联合建模精神疾病（如焦虑和抑郁）及其关系压力触发因素，并揭示了不同模型在障碍分类与关系触发检测任务上的性能差异。 |
| [^21] | [Effective Covariance Dynamics in Solvable High-Dimensional GANs](https://arxiv.org/abs/2606.27246) | 本文提出了一个可解高维GAN模型，通过概率加权有效协方差统一处理类别相关、相关及非零均值的潜在结构，并证明了训练过程在高维极限下收敛到确定性常微分方程。 |
| [^22] | [The Geometry of Updates: Fisher Alignment at Vocabulary Scale](https://arxiv.org/abs/2606.27242) | 本文提出FisherSketch方法，通过将头费舍尔对齐等价为联合激活-误差空间中核均值嵌入的余弦值，实现了在词汇规模下高效、可识别的无训练源选择，解决了传统表示度量不可识别而经典几何度量计算成本过高的问题。 |
| [^23] | [CARVE: Content-Aware Recurrent with Value Efficiency for Chunk-Parallel Linear Attention](https://arxiv.org/abs/2606.27229) | CARVE通过仅在键轴上擦除的单一原则，解决了递归模型中的记忆盲区门控、参数浪费和WY形式求解器失效三个问题，实现了高效的内容感知递归线性注意力。 |
| [^24] | [Hierarchical Muon: Tiled Newton-Schulz Updates for Efficient Muon Optimization](https://arxiv.org/abs/2606.27216) | 本文提出层级Muon（HiMuon），通过将动量梯度矩阵分块并独立应用Newton-Schulz映射，大幅降低计算复杂度，同时保留局部谱相互作用。 |
| [^25] | [Graph Neural Networks Applications Across Domains: All Insights You Need](https://arxiv.org/abs/2606.27202) | 本综述统一了图神经网络的设计空间，从第一性原理推导谱域与空域方法，并系统评估了十二个应用领域中图结构计算成本的合理性，明确了当前架构的表达能力边界。 |
| [^26] | [Explaining Temporal Graph Neural Networks via Feature-induced Information Flow](https://arxiv.org/abs/2606.27201) | 提出了一种基于正则化相关性度量框架的新归因方法，通过分析所有事件相关变量中的完整信息流，克服了现有方法忽略事件诱导变量路径的局限，从而更全面地解释基于事件的时序图神经网络。 |
| [^27] | [Forecasting With LLMs: Improved Generalization Through Feature Steering](https://arxiv.org/abs/2606.27199) | 本研究发现通过增强大型语言模型中的时间感知特征，可以有效减少预测中的前瞻偏差，提升泛化能力，而干预前瞻偏差特征则无效。 |
| [^28] | [Automating Potential-based Reward Shaping with Vision Language Model Guidance](https://arxiv.org/abs/2606.27180) | 提出VLM-PBRS框架，利用视觉语言模型反馈自动学习势能函数，在保留最优策略的同时消除了手工定义势能函数的需求。 |
| [^29] | [RecallRisk-BERT: A Multi-Task Framework for Post-Report Medical Device Recall Triage](https://arxiv.org/abs/2606.27174) | 本文提出了一个名为RecallRisk-BERT的多任务框架，通过联合建模召回严重性与根本原因类别，实现了对医疗器械报告后召回记录的高效自动分诊，填补了现有研究在联合预测方面的空白。 |
| [^30] | [Stochastic Gradient Optimization with Model-Assisted Sampling](https://arxiv.org/abs/2606.27171) | 本文提出一种模型辅助采样框架，结合调查抽样理论，利用辅助梯度预测模型降低随机梯度估计的方差，从而提升机器学习优化性能。 |
| [^31] | [Learning to Fold: prizewinning solution at LeHome Challenge 2026 (1st place online, 2nd offline)](https://arxiv.org/abs/2606.27163) | 本文提出了一种通过强化学习改进的视觉-语言-动作策略，实现了在线仿真和真实世界双臂衣物折叠的高性能，并整合了多种优化技术。 |
| [^32] | [DMuon: Efficient Distributed Muon Training with Near-Adam Overhead](https://arxiv.org/abs/2606.27153) | 本文提出DMuon，一个开源分布式Muon优化器实现，通过即插即用模块设计将训练开销降至接近Adam水平，无需框架修改。 |
| [^33] | [fTNN: a tensor neural network for fractional PDEs](https://arxiv.org/abs/2606.27140) | 本文提出了一种名为fTNN的确定性张量神经网络子空间方法，通过几何自适应积分分裂和边界奇异性感知试验函数，有效解决了有界域上分数阶拉普拉斯算子相关问题的低正则性解和积分计算挑战。 |
| [^34] | [Kolmogorov Arnold networks (KAN) for aerodynamic prediction: a comparison with MLPs and GNNs](https://arxiv.org/abs/2606.27126) | 本文通过气动预测任务，系统比较了科尔莫戈罗夫-阿诺德网络（KAN）与MLP及GNN的性能，揭示了KAN在流体动力学代理建模中的优势与局限。 |
| [^35] | [Efficient foundation decoders for fault-tolerant quantum computing](https://arxiv.org/abs/2606.27119) | 提出神经迁移统一框架NTU，通过利用可扩展码族的代数结构对齐不同码距下的解码任务，实现小码知识加速大码解码器训练，并实例化为NTU-Transformer，在容错量子计算中取得优于相关方法的性能。 |
| [^36] | [Cross-Head Attention Uplift Network with Inverse Propensity Score under Unobserved Confounding](https://arxiv.org/abs/2606.27114) | 本文提出跨头注意力提升网络和鲁棒对抗逆倾向得分方法，通过动态整合组间表示和对抗性优化倾向权重，解决了未观测混杂下的个体处理效应估计偏差问题。 |
| [^37] | [Heavy-Ball Q-Learning with Residual Weighting Correction](https://arxiv.org/abs/2606.27112) | 本文提出了一种带残差加权校正的重球Q学习方法，通过切换线性系统视角证明了其收敛性和加速效果，并扩展到了线性函数逼近场景。 |
| [^38] | [Transformer-Based Classification of Bacterial Raman Spectra with LOOCV](https://arxiv.org/abs/2606.27096) | 本研究证明，基于Transformer的模型在细菌拉曼光谱分类中，通过嵌套留一重复交叉验证，其性能显著优于传统机器学习方法，且能直接处理原始光谱数据。 |
| [^39] | [Data-Free Reservoir Features for Efficient Long-Horizon Cold-Start Continual Learning](https://arxiv.org/abs/2606.27095) | 本文提出CIRCLE方法，利用从未训练过的固定双向二维储层特征和流式线性判别分析，在冷启动持续学习中实现高效且无需数据回放的类增量学习。 |
| [^40] | [Beyond Global Divergences: A Local-Mass Perspective on Bayesian Inference](https://arxiv.org/abs/2606.27090) | 本文通过引入质量指数和正则化扩展KL散度，从局部质量视角揭示了贝叶斯推理中全局目标函数（如KL散度）未直接捕获的局部行为，并证明了比较局部质量的不等式。 |
| [^41] | [Finding Stationary Points by Comparisons](https://arxiv.org/abs/2606.27082) | 本文提出了在比较预言机下寻找非凸函数驻点的经典和量子算法，分别达到$\widetilde O(n^2/\epsilon^{1.5})$和$\widetilde O(n/\epsilon^{1.5})$的查询复杂度。 |
| [^42] | [Parametric Open Source Games](https://arxiv.org/abs/2606.27068) | 本文提出参数化开源游戏框架，通过连续参数空间替代离散程序，揭示了博弈中合作涌现的耦合阈值及神经语义下的合作条件。 |
| [^43] | [State Representation Matters in Deep Reinforcement Learning: Application to Energy Trading](https://arxiv.org/abs/2606.27032) | 本研究表明，在深度强化学习用于能源交易时，状态表示的设计（特别是使用相对特征和预测特征）对性能有显著影响，仅用绝对特征效果很差。 |
| [^44] | [Symplectic Neural Networks for learning Generalized Hamiltonians](https://arxiv.org/abs/2606.27029) | 本文提出利用伴随系统的辛离散化与反向传播灵敏度等价的特性，实现了一种在噪声观测下高效训练哈密顿神经网络的方法，解决了隐式辛积分器计算复杂和反向传播困难的问题。 |
| [^45] | [Just how sure are you? Improving Verbalized Uncertainty Calibration in Medical VQA](https://arxiv.org/abs/2606.27023) | 本文针对医学视觉问答中多模态大语言模型过度自信的问题，提出了一种基于训练的框架，通过复合损失函数和因子扰动设计来改善置信度校准。 |
| [^46] | [A Generalization Theory for JEPA-Based World Models](https://arxiv.org/abs/2606.27014) | 本文首次为基于JEPA的世界模型建立了泛化理论，揭示了其预训练误差与下游规划性能之间的量化关系，并发现了潜在维度上近似误差与样本误差的权衡。 |
| [^47] | [Semantic Early-Stopping for Iterative LLM Agent Loops](https://arxiv.org/abs/2606.27009) | 提出了一种基于语义相似度和答案质量评估的早期停止方法，替代固定迭代上限，以优化多智能体LLM循环中的token使用效率。 |
| [^48] | [Uncertainty quantification via conformal prediction in data assimilation](https://arxiv.org/abs/2606.27001) | 本研究首次将保形预测方法应用于数据同化领域，通过一维修正浅水模型验证了其比传统集合方法更有效量化不确定性的能力。 |
| [^49] | [RolloutPipe: Overlapping Pipelined Rollout and Training in Disaggregated On-Policy LLM Reinforcement Learning](https://arxiv.org/abs/2606.26997) | 该论文提出了RolloutPipe框架，通过将固定权重的推出过程转化为完整的组流水线，实现了推出与训练在分离式RLVR系统中的高效重叠，解决了同步系统的GPU空闲问题和异步系统的数据过时问题。 |
| [^50] | [Enabling self-supervised learned primal dual with Noise2Inverse](https://arxiv.org/abs/2606.26991) | 提出一种自监督方法，通过将Noise2Inverse框架与学习原偶算法结合，利用CT扫描不同角度测量噪声的统计独立性，实现了无需真实图像即可训练迭代重建算子。 |
| [^51] | [Decision-Aligned Evaluation of Uncertainty Quantification](https://arxiv.org/abs/2606.26990) | 本文提出决策对齐标准，发现传统不确定性量化指标常与下游决策效用不一致，并设计先验加权效用指标以实现与决策效用的对齐，从而修正了现有评估协议的缺陷。 |
| [^52] | [XMSE-Aware Adaptive Empirical Bayes Estimation](https://arxiv.org/abs/2606.26975) | 本文通过将超额均方误差（XMSE）分析从诊断工具转化为设计原则，提出了一种在最大似然和经验贝叶斯之间自适应插值的混合估计器，并证明了其在二阶意义下不劣于两者。 |
| [^53] | [Geometric Gradient Rectification for Safe Open-Set Semi-Supervised Learning](https://arxiv.org/abs/2606.26973) | 本文提出几何梯度修正（GGR）框架，通过将冲突的辅助梯度投影到以监督梯度为锚点的可接受区域，在梯度层面控制更新方向，从而避免样本选择与伪标签错误带来的性能权衡，实现安全的开放集半监督学习。 |
| [^54] | [Jailbreaking for the Average Jane: Choosing Optimal Jailbreaks via Bandit Algorithms for Automatically Enhanced Queries](https://arxiv.org/abs/2606.26936) | 本文提出一种基于多臂老虎机算法的越狱攻击策略，让非专业用户也能高效选择最优越狱方法，并构建了包含大量恶意查询的基准测试FrankensteinBench，验证了非专业恶意行为者成功攻击LLMs的可行性。 |
| [^55] | [GEOALIGN: Geometric Rollout Curation for Robust LLM Reinforcement Learning](https://arxiv.org/abs/2606.26917) | GEOALIGN通过检测和修正批次中与多数方向不一致的高奖励轨迹，有效抑制了噪声奖励下的训练不稳定性，是一种轻量级的在线强化学习轨迹筛选方法。 |
| [^56] | [Tractography-Driven Synthetic Data Generation for Fiber Bundle Segmentation in Tracer Histology](https://arxiv.org/abs/2606.26898) | 提出利用dMRI纤维束成像作为生成先验，合成逼真二维图像块来训练分割网络，实现示踪组织学中纤维束的自动分割，并提升跨大脑的泛化能力。 |
| [^57] | [Asymptotically Optimal Learning for Parametric Prophet Inequalities](https://arxiv.org/abs/2606.26893) | 针对参数未知的指数型参数族先知不等式问题，提出了一种仅靠在线观测即可达到最优渐近竞争比的置信度动态规划策略，无需离线样本。 |
| [^58] | [Accelerated sampling using SamAdams variable timesteps and position-adaptive Langevin dynamics](https://arxiv.org/abs/2606.26881) | 本文提出了一种结合自适应步长和方向性摩擦的加速朗之万采样方法，在保持采样精度的同时显著提高计算效率。 |
| [^59] | [Heterogeneous Neural Predictivity from Language Models During Naturalistic Comprehension](https://arxiv.org/abs/2606.26880) | 该论文证明语言模型表征在自然语言理解过程中可有效预测神经活动，并通过多种控制验证了这种预测的稳健性和敏感性。 |
| [^60] | [Scalable Message-Passing Quantum Graph Neural Networks in the Weisfeiler-Leman Hierarchy](https://arxiv.org/abs/2606.26873) | 本文提出了一种在韦斯费勒-莱曼层级中具有可扩展性的量子图神经网络，通过消息传递和排列等变性，实现了表达能力与可扩展性的双重保证。 |
| [^61] | [Quantization in Federated Learning: Methods, Challenges and Future Directions](https://arxiv.org/abs/2606.26822) | 本文首次系统性地综述了联邦学习中的量化技术，提出了一个针对联邦学习特有维度（如客户端异构性、非IID鲁棒性等）的新分类法，并分析了量化与联邦学习核心行为的相互作用。 |
| [^62] | [Memory Depth, Not Memory Access: Selective Parametric Consolidation for Long-Running Language Agents](https://arxiv.org/abs/2606.26806) | 本文提出了一种通过选择性参数巩固（EVAF机制）来增强长时间运行语言代理在卸载工作上下文后持久保持目标导向行为的能力，而传统检索系统仅能实现浅层事实回忆。 |
| [^63] | [Reasoning Quality Emerges Early: Data Curation for Reasoning Models](https://arxiv.org/abs/2606.26797) | 本文提出仅利用推理模型初始令牌的损失值，即可低成本、高效地筛选出高质量且多样化的监督微调数据，从而提升推理能力。 |
| [^64] | [MIRROR: Novelty-Constrained Memory-Guided MCTS Red-Teaming for Agentic RAG](https://arxiv.org/abs/2606.26793) | 提出MIRROR框架，通过新颖性约束和记忆引导的蒙特卡洛树搜索，统一解决多模态智能体RAG系统的多种攻击面，显著提升攻击成功率并避免攻击模板重复。 |
| [^65] | [AIGP: An LLM-Based Framework for Long-Term Value Alignment in E-Commerce Pricing](https://arxiv.org/abs/2606.26787) | 本文提出AIGP框架，通过大语言模型结合领域知识、结构化数据和文本上下文实现可解释的定价决策，并利用长期价值估计器和直接偏好优化使定价策略与累计GMV、ROI等长期业务目标对齐。 |
| [^66] | [Reproducibility Study of "AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models"](https://arxiv.org/abs/2606.26783) | 本研究复现了AlphaEdit知识编辑方法，发现其原始结果基本可重复，但在流畅性指标上存在差异，且该方法在新型模型架构上的优势不具有普遍性。 |
| [^67] | [LearniBridge: Learnable Calibration of Feature Caching for Diffusion Models Acceleration](https://arxiv.org/abs/2606.26778) | 提出LearniBridge，一种基于低秩子空间结构洞察的可学习特征缓存校准方法，仅需少量训练样本即可显著减少高加速比下的误差累积，实现高达5.87倍的加速。 |
| [^68] | [Evaluation Pitfalls and Challenges in Multimedia Event Extraction](https://arxiv.org/abs/2606.26775) | 本文首次系统揭示了多媒体事件抽取中因数据处理、任务假设和评估设置不一致导致的性能高估问题，并呼吁建立标准化评估框架。 |
| [^69] | [Escaping Iterative Parameter-Space Noise: Differentially Private Learning with a Hypernetwork](https://arxiv.org/abs/2606.26772) | 提出了一种基于超网络的新框架，通过仅一次向低维数据集表示注入隐私噪声，避免了迭代参数空间噪声，从而显著降低差分隐私学习中噪声的不利影响。 |
| [^70] | [ProtoKV: Streaming Video Understanding under Delayed Query with Summary-State Memory](https://arxiv.org/abs/2606.26762) | 本文提出ProtoKV，通过将远程历史压缩为固定容量的摘要状态而非保留所有令牌，在恒定内存下解决了流式视频理解中延迟查询导致的关键线索被稀释或逐出的问题，准确率提升高达12.5个百分点。 |
| [^71] | [Batch-Invariant Spectral Intelligence for Robust and Explainable Insect Authentication](https://arxiv.org/abs/2606.26757) | 提出了一个端到端的批次不变光谱网络（BISN），通过结合可学习的预处理模块和熵正则化对抗目标，有效抑制了光谱测量中的批次间变异，从而在未见过的生产批次上实现了鲁棒且可解释的昆虫物种认证。 |
| [^72] | [Structure Before Collapse: Transient semantic geometry in next-token prediction](https://arxiv.org/abs/2606.26749) | 该论文揭示了在单热标签训练下，语言模型通过瞬态几何结构学会了潜在语义类别，挑战了神经网络崩溃理论的对称性假设。 |
| [^73] | [HyperDFlash: MHC-Aligned Block Speculative Decoding with Gated Residual Reduction](https://arxiv.org/abs/2606.26744) | 针对DeepSeek-V4的多超连接架构，提出了一种通过门控残差缩减实现特征对齐的块级推测解码方法，解决了多路径残差流导致的生成准确率下降问题。 |
| [^74] | [Scientific discovery as meta-optimization: a combinatorial optimization case study](https://arxiv.org/abs/2606.26728) | 本文提出将科学发现形式化为元优化，通过共识目标聚合方法结合LLM生成的目标函数，形成自修正评估标准，在3-SAT问题算法发现中显著降低了计算复杂度。 |
| [^75] | [State-Specific Respiratory Signatures for Affective and Stress Recognition: Interpretable Respiratory Markers, Autocorrelation Lags, and Compact CNN Models](https://arxiv.org/abs/2606.26723) | 本研究通过结合紧凑型一维CNN和手工设计的呼吸特征，不仅实现了压力与非压力的高精度二分类检测，还识别出了基线、压力、愉悦和冥想状态各自特异的可解释呼吸标记。 |
| [^76] | [DroidBreaker: Practical and Functional Problem-Space Attacks on Machine-Learning Android Malware Detectors](https://arxiv.org/abs/2606.26707) | DROIDBREAKER是一种在问题空间中通过构建安全和语义保持的方法，对安卓APK进行实用且功能性的修改，以有效逃避机器学习恶意软件检测器。 |
| [^77] | [Algorithmic Foundations of Deep Learning: Complexity-Theoretic Rates and a Characterization of Universal Approximation](https://arxiv.org/abs/2606.26705) | 本文提出神经网络应被视为计算模型，其复杂度由算法复杂度而非仅正则性决定，并给出基于电路计算框架的普适逼近表征。 |
| [^78] | [Attributed, But Not Incremental: Cannibalization-Corrected Attribution for Large-Scale Advertising](https://arxiv.org/abs/2606.26690) | 提出一种利用增量实验作为因果锚点，将稀疏提升度量转化为每日修正估计的归因修正框架，并在结构性约束下分配蚕食量，以解决大规模广告中付费归因高估增量增长的问题。 |
| [^79] | [PersistentKV: Page-Aware Decode Scheduling for Long-Context LLM Serving on Commodity GPUs](https://arxiv.org/abs/2606.26666) | 本文提出PersistentKV，一种针对商用GPU上长上下文LLM服务的页面感知解码调度引擎，通过按KV头组映射工作、重用K/V块和紧凑工作队列调度，优化了页面感知解码的效率。 |
| [^80] | [Zero-Shot Size Transfer for Neural ODEs on Sparse Random Graphs: Graphon Limits and Adjoint Convergence](https://arxiv.org/abs/2606.26662) | 本文针对稀疏随机图，从理论上证明了图神经微分方程具有零样本尺寸迁移能力，即在小图上训练后可无需重新训练直接部署到更大图上，并给出了收敛率分析。 |
| [^81] | [Generating Special Triangulations with Transformers](https://arxiv.org/abs/2606.26660) | 本文提出使用Transformer模型并配以适当编码方案，能够有效生成四维自反多面体的精细、规则且星形三角剖分，且模型可通过自我再训练改进，为卡拉比-丘流形分类及多领域研究提供新方法。 |
| [^82] | [Target-Aware Bandit Allocation for Scalable Surrogate Optimization in Chemical Space](https://arxiv.org/abs/2606.26657) | 提出BOBa框架，利用多臂赌博机自适应分配计算资源到动作空间分区，消除全库推理，实现化学空间中大规模分子库的高效替代优化。 |
| [^83] | [FracEvent: Event-Camera Simulation via Fractional-Relaxation Pixel Dynamics](https://arxiv.org/abs/2606.26636) | 提出FracEvent事件仿真器，通过分式松弛电压动力学精确建模像素生命周期，以改善事件时序保真度和下游任务迁移性能。 |
| [^84] | [From Weights to Features: SAE-Guided Activation Regularization for LLM Continual Learning](https://arxiv.org/abs/2606.26629) | 本文提出利用稀疏自编码器（SAE）在激活空间进行正则化，通过单义特征字典显式平衡稳定性与可塑性，解决了大语言模型持续学习中EWC等权重正则化方法因多义性而表现不佳的问题。 |
| [^85] | [Discovering Millions of Interpretable Features with Sparse Autoencoders](https://arxiv.org/abs/2606.26620) | 本文提出了Qwen3-Instruct SAE套件，在多个Qwen3模型上训练了覆盖不同激活位置的稀疏自编码器，并系统评估了其稀疏性与保真度的权衡，最后通过拒绝引导案例展示了其应用价值。 |
| [^86] | [Sketched Linear Contrastive Learning: Approximation, Optimization, and Statistical Scaling](https://arxiv.org/abs/2606.26617) | 本文针对对比学习中的草图线性模型，推导了包含近似、优化和统计误差的显式尺度定律，揭示了草图维度、样本大小与光谱衰减率之间的权衡关系。 |
| [^87] | [Latent Diffusion Posterior Sampling with Surrogate Likelihood Guidance for PDE Inverse Problems](https://arxiv.org/abs/2606.26592) | 提出一种结合变分自编码器、潜空间扩散模型和神经代理的贝叶斯反演方法，有效解决了高维PDE反问题中的先验建模、维度灾难和计算成本三大挑战。 |
| [^88] | [Empirical Software Engineering TerraProbe: A Layered-Oracle Framework for Detecting Deceptive Fixes in LLM-Assisted Terraform](https://arxiv.org/abs/2606.26590) | 本文提出TerraProbe框架，通过五层Oracle评估揭示LLM辅助Terraform安全修复中仅消除静态告警的欺骗性，实际完整修复率远低于表面指标。 |
| [^89] | [SharQ: Bridging Activation Sparsity and FP4 Quantization for LLM Inference](https://arxiv.org/abs/2606.26587) | 提出SharQ方法，通过在线稀疏-稠密分解与残差补偿机制，有效结合了激活稀疏性与FP4量化，在无需训练的情况下降低了大语言模型推理中的激活压缩损失。 |
| [^90] | [Revisiting Action Factorization for Complex Action Spaces](https://arxiv.org/abs/2606.26574) | 本文对六种动作分解方法在多种算法上的表现进行了系统性横截面比较研究，揭示了现有复杂动作空间基准测试的局限性。 |
| [^91] | [Explainable Ensemble-Based Machine Learning Models for Detecting the Presence of Cirrhosis in Hepatitis C Patients](https://arxiv.org/abs/2606.26561) | 本研究首次利用可解释的集成学习机器学习模型检测丙型肝炎患者的肝硬化，填补了该领域的研究空白。 |
| [^92] | [PMDformer: Patch-Mean Decoupling Information Transformer for Long-term Forecasting](https://arxiv.org/abs/2606.26549) | 本文提出PMDformer，通过补丁均值解耦分离趋势与形状信息，并设计趋势恢复注意力和近端变量注意力，以解决长期时间序列预测中因尺度差异导致的形状相似性建模难题。 |
| [^93] | [Can Large Language Models Reliably Code Qualitative Humanitarian Data? A Benchmark Study Against Human Expert Adjudication](https://arxiv.org/abs/2606.26541) | 该研究首次通过基准测试证明，多个大型语言模型在编码定性人道主义数据时能达到接近人类专家的可靠性，尤其适用于大规模需求分析。 |
| [^94] | [CascadeFormer: Depth-Tapered Transformers Motivated by Gradient Fan-in Asymmetry](https://arxiv.org/abs/2606.26538) | 本文通过提出梯度扇入不对称性理论，解释了深度Transformer中深层贡献小的现象，并设计了两种高效方法：CascadeFormer通过深度锥形化宽度提升效率，CascadeFlow Pruning利用累积梯度剪枝冗余层，在保持性能的同时显著降低延迟并提高吞吐量。 |
| [^95] | [Sample-efficient Transfer Reinforcement Learning via Adaptive Reward Shaping and Policy-Ratio Reweighting Strategy](https://arxiv.org/abs/2606.26527) | 本文提出了一种通过自适应奖励塑造和策略比重新加权策略解决迁移分布偏移与安全探索冲突的自主车道变换强化学习框架。 |
| [^96] | [Theory-Scale Auto-Formalization of Logics for Computer Science](https://arxiv.org/abs/2606.26525) | 本文提出了LCS-Bench，一个基于《计算机科学逻辑》的理论级自动形式化基准，通过半自动化智能体流水线构建，实现了大规模相互依赖逻辑理论的连贯形式化翻译。 |
| [^97] | [Radical AI Interpretability](https://arxiv.org/abs/2606.26523) | 本书提出一个结合激进解释哲学与机械可解释性的框架，为从AI系统内部读取信念和欲望提供了可验证的标准，强调归因必须整体进行而非零散操作。 |
| [^98] | [Multipath Adaptive Gated Bottleneck Latent ODE with Raman Data Fusion for Cell Culture Process Forecasting](https://arxiv.org/abs/2606.26520) | 提出了一种结合门控瓶颈潜在常微分方程与多路径即时微调的自适应框架，通过变量级门控和掩码感知瓶颈机制有效处理高维稀疏数据，实现了对细胞培养过程的多日早期预测。 |
| [^99] | [Temporal Validity in Retrieval Memory: Eliminating Stale-Fact Errors for AI Agents over Evolving Knowledge](https://arxiv.org/abs/2606.26511) | MemStrata通过双时间账本和确定性取代规则，在不依赖相似度阈值或大语言模型的情况下，解决了RAG系统中因知识演化导致的过时事实检索问题。 |
| [^100] | [Mean-Field PhiBE: Continuous-Time Mean-Field Reinforcement Learning from Discrete-Time Data](https://arxiv.org/abs/2606.26498) | 本文提出MF-PhiBE，通过将离散时间数据融入Wasserstein空间上的连续时间偏微分方程，解决了连续时间平均场控制中模型不可识别性的难题。 |
| [^101] | [Learning Probabilistic Filters with Strictly Proper Scoring Rules](https://arxiv.org/abs/2606.26497) | 本文提出PSEF方法，利用严格恰当评分规则训练基于Transformer的置换不变映射，仅通过合成数据实现贝叶斯滤波分布的逼近。 |
| [^102] | [Evaluation-Strategy Gap in Fault Diagnosis of Deep Learning Programs](https://arxiv.org/abs/2606.26492) | 本研究揭示了深度学习程序故障诊断中程序内评估与跨程序评估之间存在显著性能差距（平衡准确率差距达0.190），并发现该差距主要源于特征中的程序级结构。 |
| [^103] | [What Survives When You Compress a Recursive Reasoner for the Edge?](https://arxiv.org/abs/2606.26488) | 本研究发现，对递归推理模型进行激进压缩会保留局部预测能力，但会彻底破坏全局推理能力，且这种崩溃是架构性的，无法通过基于令牌的训练目标修复。 |
| [^104] | [Adaptive Evaluation of Out-of-Band Defenses Against Prompt Injection in LLM Agents](https://arxiv.org/abs/2606.26479) | 本文系统比较了LLM代理中多种带外防御策略，并指出当前所有防御仅通过静态基准验证，未能抵御自适应攻击。 |
| [^105] | [Retrieval-Warmed Energy-Based Reasoning: A Five-Arm Ablation Methodology for Diffusion-as-Inference on Structured Reasoning Tasks](https://arxiv.org/abs/2606.26476) | 本研究提出一种五臂消融方法学，通过分离类别先验偏差、随机预热启动和图对齐值复用三种混淆效应，揭示了检索预热能量基推理在结构化推理任务中的关键增益来源，并在连通性任务上实现了高达35个百分点的平衡准确率提升。 |
| [^106] | [Localizing RL-Induced Tool Use to a Single Crosscoder Feature](https://arxiv.org/abs/2606.26474) | 本文提出专用特征交叉编码器（DFC），能够分离并定位强化学习引入的单一特征，从而显著提升工具调用的正确性并实现无重训的能力迁移。 |
| [^107] | [When Does Quality-Aware Multimodal Fusion Matter? A Leakage-Safe Diagnostic for Decision-Level Dependence](https://arxiv.org/abs/2606.26473) | 本文提出一种泄露安全的诊断方法，通过置换测试样本间的可靠性分数，发现多模态融合中可靠性信息只有在能正确识别模态时才会影响决策，否则仅与性能相关。 |
| [^108] | [Epiphany-Aware KV Cache Eviction Without the Attention Matrix](https://arxiv.org/abs/2606.26472) | 本文提出EpiKV，一种通过直接读取模型前向传播中的内部表示变化（顿悟分数）来淘汰KV缓存的方法，无需注意力矩阵，可将可行上下文长度扩展至传统注意力评分方法的16倍，且无需训练或自定义内核。 |
| [^109] | [A Causal Foundation Model for Structure and Outcome Prediction](https://arxiv.org/abs/2606.26467) | 本文提出了一种基于合成数据训练、能同时预测因果结构与结果并支持多层级因果查询的通用因果基础模型TabPFN-CFM。 |
| [^110] | [Finding the Time to Think: Learning Planning Budgets in Real-Time RL](https://arxiv.org/abs/2606.26463) | 提出了一种在实时强化学习中，通过轻量级门控策略动态选择状态依赖的规划预算的方法，有效解决了环境持续运行下的决策延迟问题。 |
| [^111] | [A probabilistic framework for online test-time adaptation](https://arxiv.org/abs/2606.26457) | 提出了一种基于状态空间模型的概率框架，用于在线测试时自适应，以应对训练与测试分布之间的偏移。 |
| [^112] | [Active Adversarial Perturbation-driven Associative Memory Retrieval for RGB-Event Visual Object Tracking](https://arxiv.org/abs/2606.26455) | 本文提出APRTrack框架，通过分层对抗扰动模拟现实世界信号退化，并利用联想记忆检索机制，解决了RGB-事件跟踪中因模态失效或目标不完整导致的鲁棒性下降问题。 |
| [^113] | [Optimizing CUDA like a Human: Micro-Profiling Tools as Expert Surrogates for LLM-Based GPU Kernel Optimization](https://arxiv.org/abs/2606.26453) | 提出KernelPro闭环多智能体系统，通过将专家启发式编码为可插拔微性能分析工具，结合多层级分析器反馈和领域自适应蒙特卡洛树搜索，实现GPU内核代码的自动生成与迭代优化。 |
| [^114] | [Listening Like a Judge: A Music-Aware Framework for Automatic Singing Performance Evaluation](https://arxiv.org/abs/2606.26451) | 本文提出MusicJudge框架，通过多模态对齐分析歌词准确性和音高节奏保真度，并引入模态引导的LoRA微调改进歌唱转录，实现了与人类专家高度一致的自动歌唱质量评估。 |
| [^115] | [Embedding Foundation Model Predictions in Discrete-Choice Models with Structural Guarantees](https://arxiv.org/abs/2606.26432) | 本文提出一种两阶段适配器方法，通过将基础模型预测嵌入多项Logit模型，在保留结构保证的同时修正经济逻辑冲突，使时间价值等指标成为数学保证。 |
| [^116] | [DualEval: Joint Model-Item Calibration for Unified LLM Evaluation](https://arxiv.org/abs/2606.26429) | DualEval通过联合校准模型能力和项目特征，在统一框架中融合静态基准与偏好数据，实现了更可靠、均衡的大语言模型评估，并支持基准压缩和异常检测等应用。 |
| [^117] | [Rethinking Training & Inference for Forecasting: Linking Winner-Take-All back to GMMs](https://arxiv.org/abs/2606.26424) | 本文揭示了轨迹预测中赢者通吃训练导致模式概率无信息的问题，并提出了两种事后处理方法（后验加权合并和一步EM更新）来改善模式分布。 |
| [^118] | [Otter Weather: Skillful and Computationally Efficient Medium-Range Weather Forecasting](https://arxiv.org/abs/2606.26421) | 本文提出Otter Weather模型，通过高效时空架构实现中期天气预报，在1.5°分辨率下以极低计算成本（<3.5 A100天）超越传统NWP 9.6%，效率比轻量AI模型高2倍，比资源密集型模型低100倍，推动了天气预测的民主化。 |
| [^119] | [Unbiased Canonical Set-Valued Oracles Via Lattice Theory](https://arxiv.org/abs/2606.26418) | 本文通过Knaster–Tarski不动点定理，在完备格框架下提出了一种规范的非平凡credal集，解决了自指预言机在无偏和自洽约束下的唯一性问题。 |
| [^120] | [Beyond Feedforward Networks: Reentry Neural Systems as the Fundamental Basis of Subjecthood and Intrinsic Safety of Next-Generation AGI](https://arxiv.org/abs/2606.26406) | 提出了一种基于闭环再入回路的新型AGI架构，通过结构性循环和自维持放大在数学上保证自我意识与内在安全性，并设计了可高效计算的S度量替代传统综合信息度量。 |
| [^121] | [Geometry-Aware MCTS for Extremal Problems in Combinatorial Geometry](https://arxiv.org/abs/2606.26399) | 提出一种几何感知的MCTS框架，通过增量更新动作空间和利用几何对称性剪枝，将组合几何极值问题的约束检查复杂度从O(n³)降至O(n²)，并有效处理稀疏奖励和二次令牌消耗问题。 |
| [^122] | [Deterministic Pareto-Optimal Policy Synthesis for Multi-Objective Reinforcement Learning](https://arxiv.org/abs/2606.26397) | 本文提出了一种基于切比雪夫标量化的偏好条件贝尔曼算子，能够为多目标马尔可夫决策过程计算并收敛到确定性帕累托最优策略覆盖集。 |
| [^123] | [At the Edge of Understanding: Sparse Autoencoders Trace The Limits of Transformer Generalization](https://arxiv.org/abs/2606.26396) | 本文通过稀疏自编码器揭示了Transformer在分布外输入下内部虚假概念增加的现象，并提出了一种基于机制的微调策略来增强模型鲁棒性。 |
| [^124] | [Staying VIGILant: Mitigating Visual Laziness via Counterfactual Visual Alignment in MLLMs](https://arxiv.org/abs/2606.26387) | VIGIL通过强化学习后训练框架，引入反事实视觉对齐来缓解多模态大语言模型中的视觉惰性，从而减少幻觉并增强视觉信息在响应中的因果作用。 |
| [^125] | [SOLAR: AI-Powered Speed-of-Light Performance Analysis](https://arxiv.org/abs/2606.26383) | 本文提出了SOLAR框架，能够自动从PyTorch和JAX代码中推导出经过验证的深度学习模型理论最小执行时间（光速界限），通过结合大语言模型和确定性分析，实现了自动化的性能极限分析。 |
| [^126] | [Scoring Is Not Enough: Addressing Gaps in Utility-fairness Trade-offs for Ranking](https://arxiv.org/abs/2606.26369) | 本文揭示了评分函数在实现效用与公平性的所有权衡中本质上是次优的，并通过反例证明了这一缺陷在多种场景下普遍存在。 |
| [^127] | [Does Aurora Encode Atmospheric Structure? Latent Regime Analysis and Attribution](https://arxiv.org/abs/2606.26361) | 极光模型在无需显式指导的情况下，通过潜在空间按季节周期组织，并学会了气象连贯性与三维垂直结构，且风暴事件未形成独立聚类。 |
| [^128] | [OpenFinGym: A Verifiable Multi-Task Gym Environment for Evaluating Quant Agents](https://arxiv.org/abs/2606.26350) | OpenFinGym提出了一个统一的多任务Gym环境，通过覆盖预测、市场生成、实时交易和欺诈检测等关键任务，并配备自动化任务构建流程，解决了现有平台因单任务评估而导致的量化金融智能体能力误判和泛化性缺失问题。 |
| [^129] | [EMA-FS: Accelerating GBDT Training via Gain-Informed Feature Screening](https://arxiv.org/abs/2606.26337) | 提出了一种基于指数移动平均增益信息的特征筛选方法，在保持与LightGBM核心算法完全兼容的前提下，通过动态保留高增益特征来加速直方图构建，从而显著提升GBDT训练效率。 |
| [^130] | [Mesh-RL: Coupled subgrid reinforcement learning](https://arxiv.org/abs/2606.26333) | Mesh-RL通过将环境分解为重叠子网格并强制执行边界一致的时间差分更新，在不修改奖励函数或引入规划机制的情况下，显著加速了大规模稀疏奖励强化学习中的长程信用分配与收敛速度。 |
| [^131] | [EVOM: Agentic Meta-Evolution of Actor-Critic Architectures for Reinforcement Learning](https://arxiv.org/abs/2606.26327) | EVOM提出了一种基于大语言模型的智能体元进化框架，通过双层优化实现行动者-评论家架构的自动化搜索，显著提升了强化学习性能。 |
| [^132] | [High-Probability PL-SGD with Markovian Noise: Optimal Mixing and Tail Dependence](https://arxiv.org/abs/2606.26316) | 本文通过滞后阻塞论证，将马尔可夫噪声下PL-SGD的高概率界从$\widetilde{O}(t_{mix}^2/k)$优化到$\widetilde{O}(t_{mix}/k)$，并证明该线性依赖是最优的，同时扩展到了重尾情况。 |
| [^133] | [Tailor Made Embeddings for Quantum Machine Learning](https://arxiv.org/abs/2606.26312) | 本文提出了一种变分自编码器框架，用于学习针对特定任务的经典数据量子嵌入，能够将高维数据集压缩到13量子比特表示中，在MNIST任务上达到98.5%的验证准确率，显著优于传统量子嵌入方法。 |
| [^134] | [The Red Queen G\"odel Machine: Co-Evolving Agents and Their Evaluators](https://arxiv.org/abs/2606.26294) | 本文提出红皇后哥德尔机（RQGM），通过将评估者纳入进化循环，使智能体能在非平稳评估标准下进行递归自我改进，从而突破静态基准的限制。 |
| [^135] | [SSM Adapters via Hankel Reduced-order Modeling: Injection Site Determines Task Suitability in Long-Context Fine-Tuning](https://arxiv.org/abs/2606.26290) | 本文提出了一种基于Hankel降阶建模的SSM适配器HRM，通过平衡截断初始化实现高效并行计算，在长上下文微调中显著优于LoRA，并揭示了注入位置对任务适用性的关键影响。 |
| [^136] | [From Clicks to Intent: Cross-Platform Session Embeddings with LLM-Distilled Taxonomy for Financial Services Recommendations](https://arxiv.org/abs/2606.26277) | 本文提出了一种利用大语言模型提炼的分类法来生成跨平台会话嵌入，从而在金融服务推荐中实现从网页点击到用户意图的预测，并支持定量推荐与定性理解的双重目标。 |
| [^137] | [Equivariance and Augmentation for Bayesian Neural Networks](https://arxiv.org/abs/2606.26273) | 本文针对贝叶斯神经网络，在变分推理框架下研究了数据增强如何实现等变性，推导了精确等变条件并提出了三种对称化技术以提升性能。 |
| [^138] | [Dataset Usage Inference without Shadow Models or Held-out Data](https://arxiv.org/abs/2606.26257) | 提出了一种无需影子模型和保留数据的数据集使用推断方法，通过生成合成非成员样本和混合比例估计，解决了现有方法依赖不切实际假设的问题。 |
| [^139] | [Interpreting "Interpretability" and Explaining "Explainability" in Machine Learning in Physics](https://arxiv.org/abs/2606.26228) | 本文明确区分了物理学机器学习中模型的结构透明度（可解释性）与科学内容映射（可说明性），强调它们是有意的建模选择而非固有属性，并讨论了相关的权衡与工具。 |
| [^140] | [Fast LeWorldModel](https://arxiv.org/abs/2606.26217) | Fast-LeWM通过动作前缀并行预测替代传统自回归展开，显著降低了视觉世界模型在规划中的计算成本并减少了潜在误差累积。 |
| [^141] | [A General Framework for Learning Algebraic Properties from Cayley Graphs using Graph Neural Networks](https://arxiv.org/abs/2606.26212) | 本文提出了一种通用框架，利用图神经网络从有限群的凯莱图中直接学习并区分其代数性质（如阿贝尔性、幂零性和可解性），证明了图表示中蕴含的代数信息可通过GNN有效提取。 |
| [^142] | [The Role of Input Dimensionality in the Emergence and Targeted Control of Adversarial Examples](https://arxiv.org/abs/2606.26207) | 通过实证研究揭示输入维度增加会使对抗样本更易构造，并发现真实图像类别的强经验局部化特性超出传统高维几何理论假设。 |
| [^143] | [Topology-Informed Neural Networks for Flood Detection in Optical and Synthetic Aperture Radar Imagery](https://arxiv.org/abs/2606.26204) | 本文提出了一种结合拓扑信息的神经网络方法，用于在光学和合成孔径雷达影像中更准确、可解释地检测洪水，克服了云层遮挡和现有模型不透明的局限。 |
| [^144] | [Statistical and Structural Approaches to Algorithmic Fairness](https://arxiv.org/abs/2606.26200) | 本论文指出现代算法公平性方法的两大根本缺陷——依赖确定性点估计审计和将个体视为孤立实体，并提出改进方案。 |
| [^145] | [From Structure to Synergy: A Survey of Vision-Language Perception Paradigm Evolution in Multimodal Large Language Models](https://arxiv.org/abs/2606.26196) | 该论文首次系统综述了多模态大语言模型中视觉与语言作为不可分割整体的统一感知范式演进，并形式化了类似人类先天感知的内在跨模态能力。 |
| [^146] | [Self-Supervised Tree-level Biomass Estimation in Urban Environments From Airborne LiDAR and Optical Observations](https://arxiv.org/abs/2606.26194) | 该研究提出了一种结合机载LiDAR和光学数据的自监督树冠级生物量估算框架，通过双流交叉注意力网络和伪标签训练，实现了城市树木的精准语义分割和生物量估计。 |
| [^147] | [Federated Hash Projected Latent Factor Learning](https://arxiv.org/abs/2606.26192) | 提出联邦哈希投影潜在因子模型，通过用二进制哈希码替代实值梯度并增强表示容量，在联邦学习中实现低通信开销和高模型精度。 |
| [^148] | [Clue-Guided Money Laundering Group Discovery](https://arxiv.org/abs/2606.26189) | 提出了一种基于线索引导的洗钱团伙发现方法，通过分析师交互从初始线索逐步恢复团伙结构，并设计了Clue2Group框架来构建局部调查上下文并保留关键结构模式。 |
| [^149] | [Necessary but Not Sufficient: Temperature Control and Reproducibility in LLM-as-Judge Safety Evaluations](https://arxiv.org/abs/2606.26185) | 本研究通过实验证明，即使将大语言模型作为评判者的采样温度设为0，也无法完全消除安全评估结果的不确定性，尤其是在决策边界附近的项目中仍存在不可重复性。 |
| [^150] | [LiMoDE: Rethinking Lifelong Robot Manipulation from a Mixture-of-Dynamic-Experts Perspective](https://arxiv.org/abs/2606.26183) | 本文提出LiMoDE，一种基于动态专家混合架构的两阶段终身机器人操作学习方案，通过多任务预训练学习可复用先验知识，并在任务适应阶段实现高效连续适应。 |
| [^151] | [KG-TRACE: A Neuro-Symbolic Framework for Mechanistic Grounding in Antimicrobial Resistance Prediction](https://arxiv.org/abs/2606.26179) | 该论文提出KG-TRACE框架，通过将知识图谱作为生物学约束集成到神经模型中，并引入BGR指标量化神经归因与生物学知识的一致性，从而在保持高预测准确率的同时实现了可解释的符号归因。 |
| [^152] | [Neural Architecture Search for Generative Adversarial Networks: A Comprehensive Review and Critical Analysis](https://arxiv.org/abs/2606.26169) | 本文全面回顾了神经架构搜索在生成对抗网络中的应用，重点比较了不同搜索策略和评估指标，强调了进化算法和梯度方法的优势，以及使用多样化数据集和稳健评估指标的重要性。 |
| [^153] | [Implementation of reinforcement learning in chemical reaction networks: application to phototaxis as curiosity-driven exploration](https://arxiv.org/abs/2606.26168) | 本文提出一个将部分可观测马尔可夫决策过程与化学反应网络常微分方程结合的框架，通过信息驱动的最小认知模型重新定义单细胞藻类的趋光性导航，实现了在感官模糊性下的主动采样与探索平衡。 |
| [^154] | [\chisao{}: A GPU-Native Parallel Optimizer for Multimodal Black-Box Functions via Convergence-Anticonvergence Oscillation](https://arxiv.org/abs/2606.26164) | 本文提出了一种GPU原生并行优化器chisao，通过创新的收敛-反收敛振荡机制和多策略自适应重播种，能够高效地找到多模态黑箱函数的所有模态。 |
| [^155] | [Reinforcement Learning Enables Autonomous Microrobot Navigation and Intervention in Simulated Blood Capillaries](https://arxiv.org/abs/2606.26154) | 本研究通过构建包含真实流体动力学、红细胞动力学和毛细血管分支结构的物理仿真环境，利用深度强化学习训练微型机器人自主导航策略，并首次系统揭示了不同尺寸和速度下的导航物理极限及多种自主发现的通用策略类型。 |
| [^156] | [Neural Speaker Diarization via Multilingual Training: Evaluation on Low-Resource Nepali-Hindi Speech](https://arxiv.org/abs/2606.26144) | 本文通过多语言训练方法，在低资源尼泊尔语-印地语语音上评估两种神经说话人日志化架构，以解决标注数据稀缺导致的性能下降问题。 |
| [^157] | [Code evolution for link prediction in complex networks](https://arxiv.org/abs/2606.26132) | 通过代码进化生成的算法在链接预测任务上显著优于人工设计方法，并在特征选择与组合上实现了关键创新。 |
| [^158] | [Physics-guided Convolutional Neural Network for Domain Growth Prediction in Systems with Conserved Kinetics](https://arxiv.org/abs/2606.26128) | 提出了一种基于注意力机制的物理引导卷积神经网络，作为代理模型精确预测Cahn-Hilliard方程控制的二元混合物相分离的长时间演化，并保持组成守恒及与理论一致的畴生长行为。 |
| [^159] | [Dot-Flik: A Scalable Edge AI Architecture for Distributed Insect Monitoring](https://arxiv.org/abs/2606.26121) | 本文提出了一种基于运动感知帧过滤算法的分布式边缘AI架构，通过边缘预处理丢弃无关帧、解耦数据采集与AI分类，显著降低了硬件成本与能源需求，并大幅提升了昆虫监测的可扩展性与覆盖范围。 |
| [^160] | [Dynamic-dLLM: Dynamic Cache-Budget and Adaptive Parallel Decoding for Training-Free Acceleration of Diffusion LLM](https://arxiv.org/abs/2606.26120) | 提出一种无训练框架，通过动态缓存预算分配和自适应并行解码，显著提升扩散大语言模型的推理效率，解决了长序列计算复杂度和动态令牌行为问题。 |
| [^161] | [The Open Source Economic Index of AI Adoption and Capability](https://arxiv.org/abs/2606.26118) | 本论文通过开发开源经济指数和基准测试系统，揭示了AI在金融、计算机科学和艺术行业采用率最高，但执行具体任务细节时仍易出错。 |
| [^162] | [Context Recycling for Long-Horizon LLM Inference](https://arxiv.org/abs/2606.26105) | 提出ContextForge系统，通过结构化查询生成、外部记忆检索和受控合成技术实现上下文回收，在不依赖完整上下文重放的情况下高效复用先前计算，从而在长程对话中保持答案质量并减少token消耗。 |
| [^163] | [Autodata: An agentic data scientist to create high quality synthetic data](https://arxiv.org/abs/2606.25996) | 本文提出了一种名为Autodata的通用方法，通过训练AI智能体作为自主数据科学家，并对其进行元优化，从而在多个任务上生成比传统方法更高质量的合成数据，显著提升模型性能。 |
| [^164] | [MiniOpt: Reasoning to Model and Solve General Optimization Problems with Limited Resources](https://arxiv.org/abs/2606.25832) | MiniOpt通过“推理-建模-求解”范式和分层奖励函数，在无需大规模监督数据或专家演示的情况下，实现了对通用优化问题的高效强化学习求解。 |
| [^165] | [How Reliable Is Your Jailbreak Judge? Calibration and Adversarial Robustness of Automated ASR Scoring](https://arxiv.org/abs/2606.25487) | 该论文揭示了LLM越狱攻击成功率（ASR）评估中两类自动化评判者的严重不可靠性：专用分类器过度标记而LLM评判者召回率波动大，且两者在对抗攻击下鲁棒性极差。 |
| [^166] | [The Generalization Spectrum: A Chromatographic Approach to Evaluating Learning Algorithms](https://arxiv.org/abs/2606.25450) | 本文提出“泛化光谱”评估框架，通过按迁移距离排列的受控测试变体，逐样本揭示学习算法的泛化程度与模式，超越传统单一聚合分数评估。 |
| [^167] | [Geometry-Anchored Transport Framework for Exemplar-Free Class-Incremental Learning](https://arxiv.org/abs/2606.25347) | 提出了一种将特征传输作为内生训练约束的几何锚定框架，通过解析几何锚点和拓扑感知演化目标，有效缓解了无样本类增量学习中的表征漂移和流形退化问题。 |
| [^168] | [Scalable Peptide Design via Memory-Efficient Equivariant Transformer](https://arxiv.org/abs/2606.25006) | 提出了一种名为MEET的内存高效等变Transformer，通过优化几何计算与特征流设计，实现了可扩展的原子级肽序列与结构协同生成。 |
| [^169] | [MORL-A2C: Multi-Objective Reinforcement Learning Reranker for Optimizing Healthiness in MOPI-HFRS](https://arxiv.org/abs/2606.23603) | 本文提出MORL-A2C，一种基于多目标强化学习的重排序器，通过优势演员-评论家算法和标量化奖励将序列决策引入食品推荐，以联合优化用户偏好与营养健康。 |
| [^170] | [How Should a Simulation-to-Reality Transfer Budget Be Spent?](https://arxiv.org/abs/2606.22062) | 少量真实数据用于系统辨识能有效弥合仿真到现实的迁移差距，且效果优于扩大域随机化范围。 |
| [^171] | [Geometric and Information Compression of Representations in Deep Learning](https://arxiv.org/abs/2606.21593) | 本文发现深度学习中表示的低互信息与几何压缩之间并不存在稳定的对应关系，二者关系比通常认为的更复杂。 |
| [^172] | [GRAG: Generic Response-Augmented Generation Framework for Personalized Conversational Systems](https://arxiv.org/abs/2606.21097) | GRAG框架通过解耦个性化和上下文基础两个目标，利用离线通用响应来提升资源受限或隐私敏感环境中个性化对话系统的性能。 |
| [^173] | [A-Evolve-Training: Autonomous Post-Training of a 30B Model](https://arxiv.org/abs/2606.20657) | 该论文提出了一个无需人工干预的自主后训练系统，能在数周内对300亿参数模型进行多轮训练，并自主发现并修正了优化策略偏差，使模型性能接近人类顶尖水平。 |
| [^174] | [Prompt, Plan, Extract: Zero-Shot Agentic LLMs Workflows for Lung Pathology Extraction from Clinical Narratives](https://arxiv.org/abs/2606.19852) | 本研究提出了一种零样本智能大语言模型工作流，用于从临床叙述中自动提取肺部病理信息，在无需人工标注的情况下，最佳模型达到了0.8的微平均F1分数。 |
| [^175] | [MetaboNet-Bench: A Multi-modal Benchmark for Glucose Forecasting in Type 1 Diabetes](https://arxiv.org/abs/2606.18640) | 提出了一个用于1型糖尿病血糖预测的多模态开源基准，整合了血糖、胰岛素和碳水化合物数据，以解决现有基准缺失和单一模态限制的问题。 |
| [^176] | [Signature filtering: a lightweight enhancement for statistical watermark detection in large language models](https://arxiv.org/abs/2606.18430) | 提出了一种轻量级检测时模块——签名过滤，通过学习并移除使水印检测不可靠的“签名”标记，在不改变水印嵌入和文本生成过程的情况下显著提升了统计水印检测的性能，并提供了理论界限。 |
| [^177] | [Representation Costs in Data Science: Foundations and the Quasi-Banach Spaces of Deep Neural Networks](https://arxiv.org/abs/2606.14954) | 本文提出了一个统一框架，通过参数空间正则化器分析数据科学中的表示成本，揭示了参数化方法与其原生函数空间之间的联系，并将核方法、小波和神经网络等经典方法统一为特例。 |
| [^178] | [When to Write and When to Suppress: Route-Specialized Dual Adapters for Memory-Assisted Knowledge Editing](https://arxiv.org/abs/2606.14668) | 本文提出一种路径专用的双适配器方法，通过相关性路由器决定何时应用编辑记忆或保留原始知识，从而在知识编辑中实现精确更新与无关行为的保护。 |
| [^179] | [Running the Gauntlet: Re-evaluating the Capabilities of Agents Beyond Familiar Environments](https://arxiv.org/abs/2606.14397) | 本论文提出了GauntletBench基准测试，通过聚焦时间感知、图形理解和3D推理三种未充分探索的能力，并在五个专业应用中设置100个视觉密集型任务，以更全面评估智能体在陌生环境中的泛化能力。 |
| [^180] | [Position: Align AI to Our Aspirations, Not Our Flaws](https://arxiv.org/abs/2606.13755) | 本文主张人工智能不应与人类有缺陷的偏好对齐，而应基于事实准确、诚实和合法的客观目标进行训练，将多元性限制在表面层面。 |
| [^181] | [Graph Reinforcement Learning for Calibration-Aware Quantum Circuit Routing](https://arxiv.org/abs/2606.12816) | 该论文提出了一种利用实时校准数据通过图强化学习进行量子电路路由的方法，在中小型量子电路上显著提升了保真度，平均精确保真度达到0.727，远超基线方法。 |
| [^182] | [SymQNet: Amortized Acquisition for Low-Latency Adaptive Hamiltonian Learning](https://arxiv.org/abs/2606.12808) | 本文提出SymQNet，一种通过离线学习后验条件采集策略并在线快速执行，从而大幅降低自适应哈密顿学习延迟的摊销强化学习方法。 |
| [^183] | [Bellman-sufficient Information Complexity](https://arxiv.org/abs/2606.11171) | 本文提出了一个名为贝尔曼充分信息复杂度的框架，通过状态表示和信息指数在序贯决策中实现信息复杂度与风险的上界和下界匹配。 |
| [^184] | [Symbolic Reasoning Frameworks Trigger Memory-Mediated Ecosystem Dynamics in Multi-Agent LLM Systems](https://arxiv.org/abs/2606.07552) | 本研究表明，在多智能体大语言模型系统中，单个智能体注入符号推理框架的微小扰动，通过累积记忆和交互涌现出稳定的、与条件相关的优胜者生态系统，揭示了记忆介导的系统动力学机制。 |
| [^185] | [Dynamic Multi-Pair Trading Strategy in Cryptocurrency Markets with Deep Reinforcement Learning](https://arxiv.org/abs/2606.04574) | 本研究通过分层配对选择方法和专有执行模型，结合深度强化学习，显著提升了加密货币市场中配对交易的稳健性与收益表现。 |
| [^186] | [How Many Trees in a Random Forest? A Revisited Approach with Plateau Search and Optuna Integration](https://arxiv.org/abs/2606.03549) | 本文提出一种集成高原搜索与Optuna的新方法，通过监控袋外分数的相对变化来自适应确定随机森林的最小足够树数量，无需预设搜索范围且避免早停。 |
| [^187] | [Beyond Independent Manipulation: Individual Fairness-aware Strategic Classification with Peer Imitation](https://arxiv.org/abs/2606.00827) | 本文提出个体公平感知策略分类框架，通过建模同伴模仿行为解决了传统策略分类在个体公平要求下无法准确描述代理人相互依赖操纵的问题。 |
| [^188] | [CALIBURN: Operationally Calibrated Streaming Intrusion Detection with Regime-Dependent Conformal Risk Control](https://arxiv.org/abs/2605.24696) | 本文提出CALIBURN，通过将运行约束（如告警预算、成本）直接嵌入阈值选择流程，而非依赖标签调优，实现了可操作化的流式入侵检测告警系统。 |
| [^189] | [LLMTabBench: Evaluating LLMs on Binary Tabular Classification From Zero to Few Shots](https://arxiv.org/abs/2605.24417) | 本文提出LLMTabBench基准，系统评估大语言模型在低数据表格分类任务中的零样本与少样本能力，发现其在零样本场景下极具竞争力，有时甚至优于传统模型。 |
| [^190] | [From One-Pass SGD to Data Reuse: Mini-Batch Scaling Laws in Sketched Linear Regression](https://arxiv.org/abs/2605.24316) | 本文通过风险分解，揭示了单次遍历和两种多遍遍历小批量SGD在草图线性回归中的缩放定律，关键创新在于将随机项与采样协议关联，并证明了幂律协方差谱下的源条件缩放定律。 |
| [^191] | [Chebyshev Policies and the Mountain Car Problem: Reinforcement Learning for Low-Dimensional Control Tasks](https://arxiv.org/abs/2605.22305) | 本文通过解析求解山地车问题的最优控制，提出切比雪夫策略作为神经网络的轻量级替代，在低维控制任务中大幅降低参数量和遗憾值，并提升性能。 |
| [^192] | [Sutra: Tensor-Op RNNs as a Compilation Target for Vector Symbolic Architectures](https://arxiv.org/abs/2605.20919) | Sutra是一种纯函数式编程语言，通过将整个程序编译为融合张量操作图，在多个嵌入基座上实现100%准确率的向量符号架构解码，远超传统Hadamard乘积方法。 |
| [^193] | [Evaluating Deep Research Agents on Expert Consulting Work: A Benchmark with Verifiers, Rubrics, and Cognitive Traps](https://arxiv.org/abs/2605.17554) | 本文提出了一个包含认知陷阱的70个专家咨询提示基准测试，通过二元验证器和五维评分量表评估三个前沿深度研究代理，发现它们在联合阈值下的接受率普遍很低（最高仅15.7%）。 |
| [^194] | [Weak-to-Strong Elicitation via Mismatched Wrong Drafts](https://arxiv.org/abs/2605.17314) | 本文发现，将较小模型产生的数学错误草稿不匹配地注入到更强学习者的GRPO训练中，能显著提升其在数学推理任务上的表现，优于标准策略内GRPO。 |
| [^195] | [Random test functions, $H^{-1}$ norm equivalence, and stochastic variational physics-informed neural networks](https://arxiv.org/abs/2605.03542) | 本文证明了任意泛函的 $H^{-1}$ 范数等价于其针对仅依赖于定义域的随机测试函数的期望平方评估，从而避免了计算上困难的上确界，并引入了与经典弱解一致的随机弱解概念，为随机变分物理信息神经网络提供了理论基础。 |
| [^196] | [Closed-Loop CO2 Storage Control With History-Based Reinforcement Learning and Latent Model-Based Adaptation](https://arxiv.org/abs/2605.02405) | 本文提出了一种结合历史强化学习和潜在模型自适应的闭环二氧化碳封存控制方法，通过利用时间井响应信息和自适应机制，有效应对储层不确定性和动态变化。 |
| [^197] | [Hierarchical Fault Detection and Diagnosis for Transformer Architectures](https://arxiv.org/abs/2604.28118) | 提出了一种名为DEFault++的层次化学习方法，能够自动检测Transformer模型中的隐蔽故障、定位受影响的组件并找出根本原因，同时构建了包含5556个标注运行实例的DEFault-bench基准测试集用于训练和评估。 |
| [^198] | [TransXion: A High-Fidelity Graph Benchmark for Realistic Anti-Money Laundering](https://arxiv.org/abs/2604.17420) | 该论文提出了TransXion，一个通过结合档案感知的正常模拟与非模板化异常合成来解决现有反洗钱基准局限性，从而评估模型对“出格”异常检测能力的高保真图基准。 |
| [^199] | [Statistical Properties of the King Wen Sequence: An Anti-Habituation Structure That Does Not Improve Neural Network Training](https://arxiv.org/abs/2604.09234) | 文王序列虽具有显著统计特性且表面类似课程学习原则，但实验证明这些特性并不能改善神经网络训练。 |
| [^200] | [Learning from Equivalence Queries, Revisited](https://arxiv.org/abs/2604.04535) | 本文通过放宽对抗性假设和完全信息要求，重新审视了经典等价查询学习模型，使其更适应现代机器学习系统的实际部署循环。 |
| [^201] | [NASimJax: A GPU-Accelerated Policy Learning Framework for Penetration Testing](https://arxiv.org/abs/2603.19864) | NASimJax通过基于JAX的GPU加速实现，将网络攻击模拟器吞吐量提升高达100倍，使大规模渗透测试策略学习成为可能。 |
| [^202] | [Training-Free Generation of Protein Sequences from Small Family Alignments via Stochastic Attention](https://arxiv.org/abs/2603.14717) | 本文提出一种无需训练的蛋白质序列生成方法，通过随机注意力机制和朗之万动力学，在小规模家族比对数据上生成符合统计约束且结构合理的新序列，解决了传统方法在小数据下的过拟合问题。 |
| [^203] | [BrepCoder: A Unified Multimodal Large Language Model for Multi-task B-rep Reasoning](https://arxiv.org/abs/2602.22284) | BrepCoder是一个统一的多模态大语言模型，通过将CAD序列转化为类似Python的代码并与B-rep对齐，再经两阶段训练，实现了从B-rep输入执行补全、纠错和问答等多种CAD任务。 |
| [^204] | [Quantum Maximum Likelihood Prediction via Hilbert Space Embeddings](https://arxiv.org/abs/2602.18364) | 本文通过将经验概率分布嵌入量子态并最小化量子相对熵，提出了一种量子最大似然预测方法，并为其在经典和量子大语言模型中的统一应用提供了非渐近性能保证。 |
| [^205] | [Learning Long-Range Dependencies with Temporal Predictive Coding](https://arxiv.org/abs/2602.18131) | 本文首次将时间预测编码与实时循环学习结合，通过引入在线影响矩阵追踪参数历史影响，在保留局部学习特性的同时精确恢复时间反向传播梯度，从而有效解决了长程依赖学习问题。 |
| [^206] | [Probabilistic NDVI Forecasting from Sparse Satellite Time Series and Weather Covariates](https://arxiv.org/abs/2602.17683) | 该论文提出了一种概率性预测框架，通过分离历史数据编码与未来协变量、引入时间距离加权损失函数，解决了稀疏不规则卫星观测下的田块级NDVI短期预测挑战。 |
| [^207] | [SEMixer: Semantics Enhanced MLP-Mixer for Multiscale Mixing and Long-term Time Series Forecasting](https://arxiv.org/abs/2602.16220) | SEMixer通过随机注意力机制和多尺度渐进混合链，在轻量级MLP-Mixer架构中有效解决了时间序列多尺度建模中的语义鸿沟与噪声问题，显著提升了长期预测性能。 |
| [^208] | [Use What You Know: Causal Foundation Models with Partial Graphs](https://arxiv.org/abs/2602.14972) | 本文提出了一种将因果基础模型条件化于部分因果图或祖先信息的方法，通过注入可学习偏置与图卷积编码器，有效提升了利用不完全领域知识时的因果推断性能。 |
| [^209] | [Learning State-Tracking from Code Using Linear RNNs](https://arxiv.org/abs/2602.14814) | 本文通过将状态跟踪任务转换为代码形式，证明线性RNN在代码环境下的状态跟踪能力优于Transformer，并揭示了动作部分可观测性是导致状态跟踪困难的根本原因。 |
| [^210] | [Revisiting the Platonic Representation Hypothesis: An Aristotelian View](https://arxiv.org/abs/2602.14486) | 本文发现现有表征相似性度量受网络规模干扰，提出基于排列的零校准框架，证明校准后全局表征收敛现象消失，仅局部邻域相似性跨模态一致，从而提出亚里士多德式表征假说。 |
| [^211] | [Conditional Flow Matching for Visually-Guided Acoustic Highlighting](https://arxiv.org/abs/2602.03762) | 本文提出一种条件流匹配生成框架，通过引入展开损失惩罚最终步骤漂移，有效解决了视觉引导音频增强中判别模型难以处理音频重混模糊性的问题。 |
| [^212] | [DASH: Faster Shampoo via Batched Block Preconditioning and Efficient Inverse-Root Solvers](https://arxiv.org/abs/2602.02016) | 本文通过将预处理器块堆叠成3D张量以提高GPU利用率，并引入牛顿-DB迭代和切比雪夫多项式近似来加速矩阵逆根计算，从而显著提升了分布式Shampoo优化器的运行速度。 |
| [^213] | [Dual-Prototype Disentanglement: A Context-Aware Enhancement Framework for Time Series Forecasting](https://arxiv.org/abs/2601.16632) | 提出一种模型无关的辅助框架DPAD，通过动态双原型库解耦常见与罕见时间模式，使预测模型获得上下文感知的自适应能力。 |
| [^214] | [Spurious Rewards Paradox: Mechanistically Understanding How RLVR Activates Memorization Shortcuts in LLMs](https://arxiv.org/abs/2601.11061) | 本文揭示了虚假奖励在RLVR中会激活大语言模型中的“锚点-适配器”神经电路，导致模型依赖记忆捷径而非真正推理，从而解释了“困惑度悖论”。 |
| [^215] | [Metaphors are a Source of Cross-Domain Misalignment of Large Reasoning Models](https://arxiv.org/abs/2601.03388) | 本文发现训练数据中的隐喻会导致大推理模型在推理时出现跨领域错位，并且这种错位可以通过隐喻干预被诱导和放大。 |
| [^216] | [Digital Twin-Driven Communication-Efficient Federated Anomaly Detection for Industrial IoT](https://arxiv.org/abs/2601.01701) | 本文提出了一套数字孪生集成的联邦学习方法，通过五种创新机制在保护数据隐私和提升通信效率的同时，显著提高了工业物联网异常检测的全局模型性能。 |
| [^217] | [Improved Bounds for Private and Robust Alignment](https://arxiv.org/abs/2512.23816) | 本文首次为私密且稳健的语言模型对齐（包括离线和在线场景）建立了次优性差距的理论上界，并在联合隐私与破坏设置中证明现有离线算法可同时提供更强的隐私与破坏保证，从而在仅破坏场景中获得改进的界。 |
| [^218] | [Solving Semi-Supervised Few-Shot Learning from an Auto-Annotation Perspective](https://arxiv.org/abs/2512.10244) | 本文发现半监督少样本学习中，视觉-语言模型因输出概率分布平坦导致未标注数据无法被有效利用，从而提出应借鉴少样本学习思路利用开放资源来解决这一问题。 |
| [^219] | [Patent Representation Learning via Self-supervision](https://arxiv.org/abs/2511.10657) | 提出一种利用专利内部结构（如权利要求、背景等）作为自监督信号的混合dropout-部分正样本策略，有效提升了专利表示学习在多种检索任务中的迁移性。 |
| [^220] | [Data-driven Sensor Placement for Predictive Applications: A Correlation-Assisted Attribution Framework (CAAF)](https://arxiv.org/abs/2510.22517) | 提出了一种名为CAAF的机器学习框架，通过在特征归因前对传感器位置进行聚类，解决了高相关性数据下的最优传感器布局问题，并在多个实际动态系统中验证了其有效性。 |
| [^221] | [CSU-PCAST: A Dual-Branch Transformer Framework for medium-range ensemble Precipitation Forecasting](https://arxiv.org/abs/2510.20769) | 本文提出了CSU-PCAST，一个基于Swin Transformer和双分支解码器的深度学习集合预报框架，用于全球中期降水预测，能够从GFS分析初始化并生成30个集合成员的15天预报。 |
| [^222] | [Estimating Orbital Parameters of Direct Imaging Exoplanet Using Neural Network](https://arxiv.org/abs/2510.17459) | 提出了一种结合流匹配后验估计和马尔可夫链蒙特卡洛的算法，在保持精度的同时大幅加速了系外行星轨道参数估算，速度比传统方法快数十至数百倍。 |
| [^223] | [HOB: A Holistically Optimized Bidding Strategy under Heterogeneous Bidding Environments](https://arxiv.org/abs/2510.15238) | 本文提出HOB策略，通过使边际成本在异构渠道间可计算和对齐，并建模免费获胜概率与价格不确定性，实现了跨不同拍卖机制的全局协调竞价优化。 |
| [^224] | [Reinforcement Fine-Tuning of Flow-Matching Policies for Vision-Language-Action Models](https://arxiv.org/abs/2510.09976) | 提出流策略优化算法，通过重新定义重要性采样解决了流匹配模型中强化微调的计算不可行问题，实现了对视觉-语言-动作模型的稳定在线强化微调。 |
| [^225] | [Computationally-efficient Graph Modeling with Refined Graph Random Features](https://arxiv.org/abs/2510.07716) | 提出GRFs++方法，通过游走拼接技术将长随机游走替换为并行短游走，在保持无偏性的同时显著提升图核函数计算的效率，并扩展了游走终止机制。 |
| [^226] | [Efficient learning of bosonic Gaussian unitaries](https://arxiv.org/abs/2510.05531) | 本文首次提出了一个时间高效、严格分析的算法，用于学习玻色高斯幺正算符，其复杂度与模式数和能量参数呈多项式关系，并仅使用实验友好的光子资源。 |
| [^227] | [Eyes-on-Me: Scalable RAG Poisoning through Transferable Attention-Steering Attractors](https://arxiv.org/abs/2510.00586) | 提出一种模块化RAG投毒攻击方法，通过可迁移的注意力吸引子实现零成本适应新目标，将攻击成功率提升至先前工作的2.6倍。 |
| [^228] | [Rotary Position Encodings for Graphs](https://arxiv.org/abs/2509.22259) | 本文提出了一种名为WIRE的图结构旋转位置编码方法，通过基于图拉普拉斯谱的令牌旋转，将结构信息注入注意力机制，在理论上可恢复网格上的常规RoPE且渐近依赖图有效电阻，并兼容线性注意力。 |
| [^229] | [Learning Robust Penetration Testing Policies under Partial Observability: A systematic evaluation](https://arxiv.org/abs/2509.20008) | 本研究系统评估了在部分可观测性下，使用标准强化学习方法（如PPO）学习鲁棒且可迁移的渗透测试策略，并通过更具挑战性的网络基准来提升现实适用性。 |
| [^230] | [Limited Reference, Reliable Generation: A Two-Component Framework for Tabular Data Generation in Low-Data Regimes](https://arxiv.org/abs/2509.09960) | 提出ReFine框架，通过提取可解释规则嵌入提示和双粒度过滤，解决了低数据场景下表格数据生成中的分布偏移和过采样问题。 |
| [^231] | [Reconstruction Alignment Improves Unified Multimodal Models](https://arxiv.org/abs/2509.07295) | 提出一种名为RECA的后训练方法，利用视觉理解编码器嵌入作为密集监督信号，无需文本标题即可重新对齐统一多模态模型的理解与生成，从而显著提升图像生成和编辑的保真度。 |
| [^232] | [Huracan: A skillful end-to-end data-driven system for ensemble data assimilation and weather prediction](https://arxiv.org/abs/2508.18486) | Huracan是首个仅依赖观测数据、无需传统数值天气预报初始条件，就能实现与最先进NWP相当精度的端到端集合天气预报系统。 |
| [^233] | [Learning to Select Maximum Clique Algorithms: From Traditional Machine Learning to a Dual-Channel Hybrid Neural Architecture](https://arxiv.org/abs/2508.08005) | 提出了一种融合传统机器学习与图神经网络的双通道模型GAT-MLP，通过同时捕捉图的结构特征和全局属性，实现了对最大团问题最优算法的实例感知选择。 |
| [^234] | [DMSC: Dynamic Multi-Scale Coordination Framework for Time Series Forecasting](https://arxiv.org/abs/2508.02753) | 提出了一种动态多尺度协调框架（DMSC），通过内置的多尺度补丁分解、三元交互和自适应路由机制，解决了时间序列预测中静态分解、碎片化依赖和僵化融合的问题。 |
| [^235] | [Normalizing Flows are Capable Models for Continuous Control](https://arxiv.org/abs/2505.23527) | 本文证明归一化流在连续控制中具备足够的表达能力，是扩散模型和自回归Transformer的有效替代方案。 |
| [^236] | [No Free Lunch: Non-Asymptotic Analysis of Prediction-Powered Inference](https://arxiv.org/abs/2505.20178) | 本文通过有限样本分析证明，PPI++方法并非总是优于仅使用黄金标准标签，其优势仅在伪标签与黄金标准标签的相关性高于特定阈值时成立。 |
| [^237] | [Mitigating Hallucinations via Inter-Layer Consistency Aggregation in Large Vision-Language Models](https://arxiv.org/abs/2505.12343) | 本文提出了一种名为DCLA的免训练解码方法，通过聚合前层表示构建动态语义参考来纠正语义偏差的层，从而有效缓解大型视觉语言模型中的幻觉问题，并在多个模型和基准上显著提升性能。 |
| [^238] | [Chisme: Heterogeneity-Aware Gossip Learning](https://arxiv.org/abs/2505.09854) | 本文提出Chisme，一种完全去中心化的分布式学习算法，通过感知客户端和数据分布的异质性，解决了网络边缘环境中异构数据与间歇性连接带来的鲁棒智能实现挑战。 |
| [^239] | [Bayesian Optimization for General Reaction Conditions](https://arxiv.org/abs/2502.18966) | 本文提出CurryBO框架，通过将通用反应条件优化问题形式化为柯里化函数的贝叶斯优化，实现了在多种底物上高效寻找高性能通用条件的方法。 |
| [^240] | [Learning to Explain Air Traffic Situation](https://arxiv.org/abs/2502.10764) | 提出了一种基于Transformer的多智能体轨迹模型，通过注意力分数量化单架飞机对整体空中交通动态的影响，以解释复杂空中交通态势。 |
| [^241] | [Adversarial Robustness of AI-Generated Image Detectors in the Real World](https://arxiv.org/abs/2410.01574) | 本文揭示了当前最先进的AI生成图像检测器在现实条件下易受对抗样本攻击，攻击者无需了解检测器内部架构即可显著降低其性能，且多数攻击在常见图像处理后仍有效。 |
| [^242] | [Byzantine-Robust Aggregation for Securing Decentralized Federated Learning](https://arxiv.org/abs/2409.17754) | 本文提出了一种名为WFAgg的拜占庭鲁棒聚合算法，通过多过滤器机制在去中心化联邦学习中同时抵御攻击并增强动态拓扑鲁棒性。 |
| [^243] | [Decentralized Best-Response-Based Learning in Two-Player Zero-Sum Stochastic Games: A Finite-Sample Analysis](https://arxiv.org/abs/2409.01447) | 本文通过有限样本分析，证明了在两人零和矩阵博弈和随机博弈中，基于最佳响应的去中心化学习算法能够以样本复杂度$\mathcal{O}(\epsilon^{-1})$找到$\epsilon$-纳什均衡。 |
| [^244] | [Over-parameterization and Adversarial Robustness in Neural Networks: An Overview and Empirical Analysis](https://arxiv.org/abs/2406.10090) | 本文通过实证分析，揭示了过度参数化神经网络在对抗鲁棒性上的矛盾结论可能源于攻击方法的失效，并重新评估了其真实鲁棒性。 |
| [^245] | [Gradient Testing and Estimation by Comparisons](https://arxiv.org/abs/2405.11454) | 本文提出仅通过比较函数值大小的预言机，实现了光滑函数梯度的高效测试与估计，并在经典和量子模型下分别达到了最优或对数级的查询复杂度。 |
| [^246] | [Learning from a Biased Sample](https://arxiv.org/abs/2209.01754) | 本文提出了一种新的条件Γ-有偏抽样模型来量化训练数据中的抽样偏差，并利用分布鲁棒优化框架开发了一种元方法，以在部署时仍能获得良好性能的决策规则。 |
| [^247] | [Theory of the Frequency Principle for General Deep Neural Networks](https://arxiv.org/abs/1906.09235) | 本文严格证明了通用深度神经网络在训练中从低频到高频学习的频率原理，并提供了针对不同训练阶段的理论定理，适用于多种激活函数、数据分布和损失函数。 |
| [^248] | [On the Evaluation of Generative Models in Distributed Learning Tasks.](http://arxiv.org/abs/2310.11714) | 本文研究了在具有异构数据分布的分布式学习任务中评估生成模型。通过研究Fr\'echet inception距离（FID），并考虑不同聚合分数，发现FID-all和FID-avg分数的模型排名可能不一致。 |
| [^249] | [NervePool: A Simplicial Pooling Layer.](http://arxiv.org/abs/2305.06315) | 单纯复形池化层NervePool在池化图结构数据时，基于顶点分区生成单纯复形的分层表示，可以灵活地建模更高阶的关系，同时缩小高维单纯形，实现降采样，减少计算成本和减少过拟合。 |

# 详细

[^1]: DanceOPD：在线策略生成场蒸馏

    DanceOPD: On-Policy Generative Field Distillation

    [https://arxiv.org/abs/2606.27377](https://arxiv.org/abs/2606.27377)

    DanceOPD提出了一种在线策略生成场蒸馏框架，通过将不同图像生成能力（文生图、局部编辑、全局编辑）建模为共享空间中的速度场，并利用学生自身状态进行查询和训练，有效解决了多种能力之间的冲突与组合问题。

    

    现代图像生成需要一个统一的模型，能够集成多种能力，包括文生图、局部编辑和全局编辑。然而，这些能力很少自然对齐，且常常相互冲突。例如，编辑往往会降低文生图的性能，而全局编辑与局部编辑也会相互干扰。因此，如何有效组合这些能力已成为图像生成模型训练的核心挑战。为了解决这一问题，我们提出了DanceOPD，一种用于流匹配模型的在线策略生成场蒸馏框架。该框架将每个样本路由至一个能力场，查询一个低噪声的学生诱导状态，并通过简单的速度均方误差目标进行训练。每个能力源被定义为共享流状态空间上的速度场，学生通过在其自身生成状态上查询这些场来学习组合专家能力。该公式还吸收了操作符依赖。

    arXiv:2606.27377v1 Announce Type: cross  Abstract: Modern image generation demands a single model that unifies diverse capabilities, including text-to-image (T2I), local editing, and global editing. However, these capabilities are rarely naturally aligned and often conflict. For instance, editing tends to degrade T2I performance, while global and local editing interfere with each other. Consequently, effectively composing these capabilities has become a central challenge for image generation model training. To tackle this, we introduce DanceOPD, an on-policy generative field distillation framework for flow-matching models that routes each sample to one capability field, queries one low-noise student-induced state, and trains with a simple velocity MSE objective. With each capability source defined as a velocity field over the shared flow state space, the student learns from fields queried on its own rollout states to compose expert capabilities. This formulation also absorbs operator-d
    
[^2]: 无需真实解法的强化学习可以改进大语言模型

    Reinforcement Learning without Ground-Truth Solutions can Improve LLMs

    [https://arxiv.org/abs/2606.27369](https://arxiv.org/abs/2606.27369)

    提出RiVER框架，通过校准的连续奖励塑造（克服尺度主导和频率主导问题），使大语言模型能在无真实解法情况下进行强化学习训练。

    

    arXiv:2606.27369v1 公告类型：新 摘要：基于可验证奖励的强化学习（RLVR）在训练大语言模型时通常依赖真实答案来分配奖励，这限制了其在真实解法未知的任务中的适用性。我们提出了一种**排序诱导的可验证框架（RiVER）**，该框架无需真实解法即可在基于得分的优化任务上训练大语言模型，利用确定性执行反馈作为连续值监督。当将群体相对强化学习应用于此类连续奖励时，我们识别出两个关键挑战：**尺度主导**，即测试实例间未校准的得分幅度会扭曲策略更新；以及**频率主导**，即重复采样的次优解可能压倒罕见但更强的候选解。RiVER通过校准奖励塑造来解决这些挑战，该方法使用实例间比较并强调排名靠前的求解器，同时为其他求解器保留有界反馈。

    arXiv:2606.27369v1 Announce Type: new  Abstract: Reinforcement learning with verifiable rewards (RLVR) for training LLMs typically rely on ground-truth answers to assign rewards, limiting their applicability to tasks where the ground-truth solution is unknown. We introduce a \textbf{R}anking-\textbf{i}nduced \textbf{VER}ifiable framework (RiVER) that trains LLMs on score-based optimization tasks without ground-truth solutions, using deterministic execution feedback as continuous-valued supervision. When applying group-relative RL to such continuous rewards, we identify two key challenges: \emph{scale dominance}, where uncalibrated score magnitudes across test instances distort policy updates, and \emph{frequency dominance}, where repeatedly sampled suboptimal solutions can outweigh rare but stronger candidates. RiVER addresses these challenges with calibrated reward shaping that uses instance-wise comparisons and emphasizes top-ranked solvers while retaining bounded feedback for other 
    
[^3]: 自回归玻尔兹曼生成器

    Autoregressive Boltzmann Generators

    [https://arxiv.org/abs/2606.27361](https://arxiv.org/abs/2606.27361)

    提出自回归玻尔兹曼生成器（ArBG），通过自回归建模替代归一化流，克服了流模型在拓扑约束和计算成本上的局限，实现更高效灵活的分子平衡态采样。

    

    在统计物理中，高效采样处于热力学平衡状态的分子系统是一个标志性挑战。这一挑战推动了玻尔兹曼生成器（BGs）的发展，该生成器通过结合生成模型、精确似然函数和重要性采样校正，能够快速生成无相关的平衡样本。然而，现代BG主要依赖归一化流（NFs），而NFs要么因严格的逆映射约束（离散时间）导致表达能力有限，要么因计算成本高昂的似然函数（连续时间）而受限。本文提出自回归玻尔兹曼生成器（ArBG）——一种新型自回归建模框架——通过摆脱基于流的BG范式来克服这些局限。ArBG规避了流的拓扑约束，支持序列式推理时干预，同时通过利用高效架构提升了可扩展性。

    arXiv:2606.27361v1 Announce Type: cross  Abstract: Efficient sampling of molecular systems at thermodynamic equilibrium is a hallmark challenge in statistical physics. This challenge has driven the development of Boltzmann Generators (BGs), which allow rapid generation of uncorrelated equilibrium samples by combining a generative model with exact likelihoods and an importance sampling correction. However, modern BGs predominantly rely on normalizing flows (NFs), which either suffer from limited expressivity due to strict invertibility constraints (discrete time) or computationally expensive likelihoods (continuous time). In this paper, we propose Autoregressive Boltzmann Generators (ArBG) -- a novel autoregressive modelling framework -- that overcomes these limitations by departing from the flow-based BG paradigm. ArBG circumvents the topological constraints of flows and enables sequential inference-time interventions, while offering enhanced scalability by leveraging architectures eff
    
[^4]: 何时高概率答案是正确的？论大语言模型中序列概率与正确性的关系

    When are likely answers right? On Sequence Probability and Correctness in LLMs

    [https://arxiv.org/abs/2606.27359](https://arxiv.org/abs/2606.27359)

    本文发现序列概率仅在固定数据集内能预测跨提示-答案对的正确性，但不能推广到跨解码方法或超参数调整的解码决策中。

    

    arXiv:2606.27359v1 公告类型：跨领域 摘要：大语言模型的许多解码方法可以被理解为将概率质量向模型更可能输出的结果转移，这种转移既可以在词元级别局部进行，也可以在序列级别全局进行。因此，它们的成功取决于一个基本问题：序列概率，即给定提示后延续文本的条件概率，何时真正与正确性一致？在本文中，我们旨在跨解码方法、模型和基准，在四个层面上量化这种关系：跨解码方法、方法内跨超参数、数据集内跨提示-答案对、以及针对同一提示的重复响应。我们发现，在固定数据集中，更高的序列概率通常能预测跨提示-答案对的正确性。然而，这种关系通常不能推广到解码决策中：通过改变超参数来增加序列概率并不总能保证更高的正确性。

    arXiv:2606.27359v1 Announce Type: cross  Abstract: Many decoding methods for large language models can be understood as shifting probability mass toward outputs that are more likely under the model, either locally at the token level or globally at the sequence level. Therefore, their success depends on a fundamental question: when does sequence probability, that is, the conditional probability of a continuation given a prompt, actually align with correctness? In this paper, we set out to quantify this relationship across decoding methods, models, and benchmarks at four levels: across decoding methods, across hyperparameters within a method, across prompt-answer pairs within a dataset, and across repeated responses to the same prompt. We find that higher sequence probability is often predictive of correctness across prompt-answer pairs within a fixed dataset. However, this relationship does not generally transfer to decoding decisions: increasing sequence probability by changing hyperpa
    
[^5]: 误差条件神经求解器

    Error-Conditioned Neural Solvers

    [https://arxiv.org/abs/2606.27354](https://arxiv.org/abs/2606.27354)

    本文提出误差条件神经求解器（ENS），通过将偏微分方程残差场作为网络直接输入而非优化目标，解决了现有方法在病态系统中低残差却预测不准的问题。

    

    arXiv:2606.27354v1 公告类型：交叉 摘要：神经替代模型提供了从偏微分方程参数到解的快速近似映射，但它们通常将求解视为纯粹的统计任务：一旦训练完成，它们很难纠正自身的约束违反行为，也无法外推到训练分布之外。最近的混合方法通过针对偏微分方程残差进行梯度下降或高斯-牛顿步长来促进物理正确性，但继承了底层经典优化器的计算成本和不稳定性。我们从理论和实验上证明，在病态系统中，数值最小化偏微分方程残差可能是重建准确性的不可靠代理，这解释了为什么这些方法尽管实现了低残差，却往往无法做出准确预测。我们提出了基于不同原理的误差条件神经求解器（ENS）：偏微分方程残差场不是作为优化目标，而是在每次迭代中作为直接输入传递给网络。

    arXiv:2606.27354v1 Announce Type: cross  Abstract: Neural surrogate models offer fast approximate mappings from PDE parameters to solutions, but they typically treat solving as a purely statistical task: once trained, they struggle to correct their own constraint violations and extrapolate beyond the training distribution. Recent hybrid methods promote physical correctness by targeting the PDE residual via gradient descent or Gauss--Newton steps, but inherit the compute cost and instability of the underlying classical optimizers. We show, theoretically and empirically, that numerically minimizing the PDE residual can be an unreliable proxy for reconstruction accuracy in ill-conditioned systems, explaining why these methods often do not make accurate predictions despite achieving low residuals. We propose error-conditioned Neural Solvers (ENS), built on a different principle: rather than an optimization target, the PDE residual field is passed as a direct input to the network at each it
    
[^6]: 预算限制下的实体匹配中领域感知分布对齐的理解

    Understanding Domain-Aware Distribution Alignment in Budgeted Entity Matching

    [https://arxiv.org/abs/2606.27342](https://arxiv.org/abs/2606.27342)

    本文通过系统实验揭示了BEACON框架在预算约束下进行领域感知分布对齐的性能表现，为低资源实体匹配提供了关键见解。

    

    实体匹配（EM）是数据集成流程中的核心操作，它通过比较来自不同数据源的记录，判断它们是否指向同一真实世界实体。近期研究融入了领域信息和低资源学习技术，以更好地使EM系统适应现实场景。尽管这些方法展现出强大性能，但在实践中，它们在不同数据约束和监督程度下的表现尚不明确。本文研究了一种先进的低资源、领域感知EM方法——BEACON，并探讨了不同算法选择和数据可用性条件对其性能的影响。我们通过一系列针对性实验来评估这些变化，从而更深入地理解分布对齐的作用以及BEACON框架的行为特性。

    arXiv:2606.27342v1 Announce Type: cross  Abstract: Entity Matching (EM) is a core operation in the data integration pipeline, where records from different sources are compared to determine whether they refer to the same real-world entity. Recent work has incorporated domain information and low-resource learning techniques to better adapt EM systems to realistic settings. While these approaches have demonstrated strong performance, it remains unclear how they behave under varying data constraints and levels of supervision in practice. In this paper, we investigate a state-of-the-art method for low-resource, domain-aware EM--BEACON--and study how its performance is affected by different algorithmic choices and data availability conditions. We conduct a series of targeted experiments to evaluate these variations, providing deeper insight into the role of distribution alignment and the behavior of the BEACON framework.
    
[^7]: 通过自主经验探索与事后经验利用赋能GUI代理任务规划

    Empowering GUI Agents via Autonomous Experience Exploration and Hindsight Experience Utilization for Task Planning

    [https://arxiv.org/abs/2606.27330](https://arxiv.org/abs/2606.27330)

    本文提出PEEU方法，通过自主探索和事后经验利用生成高层次训练数据，并结合TDHAF框架分析，揭示掌握低层原子技能对提升GUI代理任务规划泛化能力的关键作用。

    

    arXiv:2606.27330v1 公告类型：交叉 摘要：多模态网络代理可以帮助人类操作重复性GUI任务，其中有效的任务规划对于将复杂任务分解为可执行动作至关重要。与商业大型模型相比，小型开源多模态大语言模型虽然成本高效且保护隐私，但存在规划能力弱和跨网站泛化能力有限的问题。为解决这些局限，我们提出了规划经验探索与利用（PEEU）方法，该方法自主探索环境以发现经验，并利用事后经验合成严格对齐的高层次训练数据。为定量分析驱动此性能的泛化行为，我们提出了任务分解层次分析框架（TDHAF），系统研究跨三个任务粒度（低、中、高水平）的组合泛化能力。分析表明，掌握低层次原子技能是关键。

    arXiv:2606.27330v1 Announce Type: cross  Abstract: Multimodal web agents can assist humans in operating repetitive GUI tasks, where effective task planning is essential for decomposing complex tasks into executable actions. While small open source MLLMs are cost efficient and privacy preserving compared with commercial large models, they suffer from weak planning and limited cross website generalization. To address these limitations, we introduce the planning experience exploration and utilization (PEEU) method, which autonomously explores environments to discover experiences and utilizes hindsight experience to synthesize strictly aligned, high level training data. To quantitatively analyze the generalization behaviors driving this performance, we propose the task decomposition hierarchical analysis framework (TDHAF) to systematically study compositional generalization across three task granularities: low, middle and high levels. Our analysis reveals that mastering low level atomic sk
    
[^8]: 世界模型中的幻觉是可预测且可预防的

    Hallucination in World Models is Predictable and Preventable

    [https://arxiv.org/abs/2606.27326](https://arxiv.org/abs/2606.27326)

    本文提出通过检测状态-动作空间的低覆盖区域来预测和预防世界模型中的幻觉，并开发了覆盖感知采样和在线预测器两种方法，显著提升了长期预测的保真度。

    

    arXiv:2606.27326v1 公告类型：新 摘要：现代生成式世界模型能够渲染出越来越逼真的、可控制动作的未来场景，但它们经常出现幻觉：生成的序列在视觉上仍然流畅，但偏离了真实动力学。我们假设幻觉集中在状态-动作空间的低覆盖区域，在这些区域中，轻量级的数据中心信号既可以检测到幻觉，也可以指导缓解措施。为验证这一点，我们引入了MMBench2，一个包含427小时、210个任务的视觉世界建模数据集，具有真实动作、奖励和实时模拟器，并在其上训练了一个350M参数的世界模型。我们识别出三种不同的幻觉模式：感知幻觉、动作边缘化幻觉和场景发散幻觉——每种模式都与流程的不同阶段相关，并开发了三种信号来准确预测模型将在何处失败。为了在训练时弥补覆盖差距，我们开发了一种覆盖感知采样技术；为了在线弥补，我们的幻觉预测器作为在线检测工具，能够在推理过程中实时识别和缓解低置信度区域中的幻觉，从而在不牺牲视觉质量的情况下显著提升长期预测的保真度。

    arXiv:2606.27326v1 Announce Type: new  Abstract: Modern generative world models render increasingly realistic action-controllable futures, yet they frequently hallucinate: rollouts remain visually fluent while drifting from the ground-truth dynamics. We hypothesize that hallucination concentrates in low-coverage regions of the state-action space, where lightweight data-centric signals can both detect it and guide mitigation. To test this, we introduce MMBench2, a 427-hour, 210-task dataset for visual world modeling with ground-truth actions, rewards, and live simulators, and train a 350M-parameter world model on it. We identify three distinct hallucination modes: perceptual, action-marginalized, and scene-diverging -- each anchored to a different stage of the pipeline, and develop three signals that accurately predict where the model will fail. To close coverage gaps at training time, we develop a coverage-aware sampling technique; to close them online, our hallucination predictors ser
    
[^9]: 超越硬预算：面向更可解释的Top-k稀疏自编码器的稀疏正则化方法

    Beyond the Hard Budget: Sparsity Regularizers for More Interpretable Top-k Sparse Autoencoders

    [https://arxiv.org/abs/2606.27321](https://arxiv.org/abs/2606.27321)

    本文针对Top-k稀疏自编码器固定预算和过拟合缺陷，提出了两种新的稀疏正则化方法，通过作用于Top-k选择前的激活值来提升模型可解释性。

    

    arXiv:2606.27321v1 公告类型：交叉 摘要：稀疏自编码器已成为解释视觉基础模型表示的主要工具，将其多语义激活分解为更大规模的稀疏、更单语义特征集合。Top-k稀疏自编码器作为当前标准变体，通过其激活函数从架构层面强制实现稀疏性，每个输入仅保留最活跃的k个潜在变量。由于该设计旨在规避早期稀疏自编码器使用的ℓ1惩罚及其已知缺陷，因此尽管其本身存在局限性——如预算k固定不变（不随输入复杂度调整）以及倾向于对训练时设定的k值过拟合——但至今未与显式稀疏正则化方法结合。我们提出了两种与Top-k架构兼容的稀疏正则化方法，两者均在Top-k选择之前作用于激活值：对未选中单元施加ℓ1惩罚，以及一种尺度不变的ℓ...（原文截断）

    arXiv:2606.27321v1 Announce Type: cross  Abstract: Sparse autoencoders (SAEs) have become a leading tool for interpreting the representations of vision foundation models, decomposing their polysemantic activations into a larger set of sparse, more monosemantic features. The Top-$k$ SAE, a now-standard variant, enforces sparsity architecturally through its activation function, retaining only the $k$ most active latents per input. Because it was designed precisely to avoid the $\ell_1$ penalty used by earlier SAEs and its known drawbacks, it has not been combined with an explicit sparsity regularizer, despite retaining limitations of its own, such as a budget $k$ that is fixed regardless of input complexity and a tendency to overfit to the training value of $k$. We introduce two sparsity regularizers compatible with the Top-$k$ architecture, both acting on the activations before the Top-$k$ selection: an $\ell_1$ penalty on the unselected (off-support) units, and a scale-invariant $\ell_
    
[^10]: 布莱克韦尔可逼近性与梯度均衡是等价的

    Blackwell Approachability and Gradient Equilibrium are Equivalent

    [https://arxiv.org/abs/2606.27315](https://arxiv.org/abs/2606.27315)

    本文证明梯度均衡（GEQ）与布莱克韦尔可逼近性在算法上等价，并由此推出GEQ与遗憾最小化、校准等框架等价，从而统一了在线学习中的多个重要概念。

    

    摘要：arXiv:2606.27315v1 公告类型：新 摘要：梯度均衡（GEQ）是最近引入的一种在线优化框架，它将离线优化中的一阶平稳性进行泛化，并抽象了如在线共形预测等问题。虽然GEQ与已知的在线学习框架（即遗憾最小化）存在有趣的相似性，但先前的研究表明，GEQ误差与遗憾是不可比较的目标，这留下了对GEQ如何融入更广泛的在线学习领域的精确理解空白。在这项工作中，我们证明GEQ在算法意义上与布莱克韦尔可逼近性是等价的。也就是说，一个布莱克韦尔可逼近问题总可以通过查询一个黑盒GEQ预言机来解决，且预言机的误差率没有渐近损失，反之亦然。结合已知的可逼近性、遗憾最小化与校准之间的等价关系，这些结果表明GEQ也与这些框架等价。我们的归约是……

    arXiv:2606.27315v1 Announce Type: new  Abstract: Gradient equilibrium (GEQ) is a recently introduced online optimization framework that generalizes first-order stationarity from offline optimization and abstracts problems like online conformal prediction. While GEQ has curious similarities with known online learning frameworks, namely regret minimization, prior work has shown that GEQ error and regret are incomparable objectives, leaving open a precise understanding of how GEQ fits into the broader online learning landscape. In this work, we show that GEQ is equivalent to Blackwell approachability in the algorithmic sense. That is, a Blackwell approachability problem can always be solved using queries to a black-box GEQ oracle, with no asymptotic loss in the oracle's error rate, and vice versa. Taken together with known equivalences between approachability, regret minimization, and calibration, these results imply that GEQ is equivalent to these frameworks, as well. Our reductions are 
    
[^11]: 一种基于大仿真与有限实验数据集的多保真度卷积自编码器-迁移学习框架，用于导波损伤诊断

    A Multi-Fidelity Convolutional Autoencoder-Transfer Learning Framework for Guided-Wave-Based Damage Diagnosis Using Large Simulated and Limited Experimental Datasets

    [https://arxiv.org/abs/2606.27304](https://arxiv.org/abs/2606.27304)

    提出了一种结合轻量级仿真与有限实验数据的多保真度迁移学习框架，通过卷积自编码器提取特征，解决了导波损伤诊断中实验数据稀缺和仿真计算成本高的问题，实现了精确的损伤定位与尺寸评估。

    

    基于导波的结构健康监测（GWSHM）利用机载传感器，为工程结构早期损伤诊断提供了巨大潜力。然而，深度学习模型的实际部署常常受到标记实验数据有限以及生成大规模高保真仿真数据集计算成本高昂的阻碍。本研究提出了一种多保真度迁移学习框架，该框架集成了轻量级基于物理的仿真、基于卷积自编码器（CAE）的深度特征学习、前馈神经网络以及有限的实验测量，用于对配备压电传感器的板状结构进行精确的损伤定位和尺寸评估。研究采用了一种计算高效的一维时域谱元模型来生成大规模合成数据集用于预训练，同时利用迁移学习使模型适应实验数据。

    arXiv:2606.27304v1 Announce Type: new  Abstract: Guided wave-based structural health monitoring (GWSHM) with onboard transducers offers significant potential for the early diagnosis of damage in engineering structures. However, the practical deployment of deep learning models is often hindered by the limited availability of labelled experimental data and the high computational cost of generating large-scale high-fidelity simulation datasets. This study presents a multifidelity transfer learning framework that integrates lightweight physics-based simulations, convolutional autoencoder (CAE)-based deep feature learning, a feed-forward neural network, and limited experimental measurements for accurate damage localisation and sizing in plate-like structures instrumented with piezoelectric transducers. A computationally efficient one-dimensional time-domain spectral element model is employed to generate a large synthetic dataset for pretraining, while transfer learning adapts the model to e
    
[^12]: 在半空间截断下以最优样本复杂度学习高斯分布的快速算法

    Fast algorithms for learning a Gaussian under halfspace truncation with optimal sample complexity

    [https://arxiv.org/abs/2606.27298](https://arxiv.org/abs/2606.27298)

    提出了一种在半空间截断下学习高斯分布的算法，其样本和时间复杂度均达到理论最优，且与无截断情况下的最优复杂度一致。

    

    我们研究了将高维高斯分布截断到未知半空间这一基本学习问题。Lee、Mehrotra 和 Zampetakis（FOCS'24）最近首次提出了该问题的多项式时间算法，但其样本和时间复杂度界限并非最优。在非平凡截断下，对于任意目标精度 $\varepsilon > 0$ 和维度 $d$，我们给出了一种高效算法，该算法使用 $n = \tilde{O}(d^2/\varepsilon^2)$ 个样本，并以总变差距离误差 $\varepsilon$ 学习到原始高斯分布。我们的算法速度也很快：其运行时间主要由计算经验协方差矩阵的成本决定。我们的样本和时间复杂度在 $d$ 和 $\varepsilon$ 方面均达到最优，即使在没有截断的情况下也是如此：就此而言，我们可以在半空间截断下免费学习高斯分布。我们结果的关键要素是对截断后高斯分布低阶矩的一种新颖重新解释。

    arXiv:2606.27298v1 Announce Type: cross  Abstract: We study the fundamental problem of learning a high-dimensional Gaussian truncated to an unknown halfspace. Lee, Mehrotra and Zampetakis (FOCS'24) recently obtained the first polynomial time algorithm for this problem, but their resulting sample and time complexity bounds are not optimal. Under non-trivial truncation, for any target accuracy $\varepsilon > 0$ and dimension $d$ we give an efficient algorithm that uses $n = \tilde{O}(d^2/\varepsilon^2)$ samples and learns the underlying Gaussian to error $\varepsilon$ in total variation distance. Our algorithm is also fast: its runtime is dominated by the cost of computing the empirical covariance matrix. Both our sample and time complexity are optimal in terms of $d$ and $\varepsilon$ even without truncation: in this regard, we can learn a Gaussian under halfspace truncation for free.   The key ingredient behind our result is a novel reinterpretation of the low-degree moments of the tru
    
[^13]: 基于动力学的模拟硬件生成模型

    Generative Models on Analog Hardware with Dynamics

    [https://arxiv.org/abs/2606.27294](https://arxiv.org/abs/2606.27294)

    本文提出模拟交互系统框架，通过时变分段参数和隐藏物理状态机制，并借助Wasserstein GAN训练，弥合模拟硬件与生成模型之间的动力学不匹配。

    

    模拟硬件平台（如耦合振荡器和模拟伊辛机）能够以数字计算能耗的一小部分自然求解微分方程，这使得它们在低功耗生成建模中极具吸引力。然而，一个根本性的不匹配依然存在：现代生成模型假设灵活的、软件定义的动力学过程，而模拟硬件则施加了固定的、由物理定律决定的微分方程，其近似能力有限。本文提出了模拟交互系统（AIS）——一个统一的可硬件实现的动力学系统框架，并通过实验刻画了其与神经网络基线模型在表达能力上的差距。为缩小这一差距，本文提出了两种硬件兼容的机制——时变分段参数和隐藏物理状态——并开发了一种基于Wasserstein GAN的训练流程，使得这些模型无需遵循特定轨迹即可进行训练。

    arXiv:2606.27294v1 Announce Type: cross  Abstract: Analog hardware platforms such as coupled oscillators and Analog Ising Machines naturally solve differential equations at a fraction of the energy cost of digital computation, making them attractive for low-power generative modeling, yet a fundamental mismatch exists: modern generative models assume flexible, software-defined dynamics, whereas analog hardware imposes fixed, physics-determined differential equations with limited approximation capacity. This paper introduces Analog Interaction Systems (AIS), a unified framework for hardware-implementable dynamical systems, and empirically characterizes their expressivity gap relative to neural network baselines. Two hardware-compatible mechanisms are proposed to narrow this gap - time-varying piecewise parameters and hidden physical states - and a Wasserstein GAN training procedure is developed to enable training of these models without requiring them to follow a specific trajectory. We 
    
[^14]: 面向可迁移查询生成的奖励信号设计：以工业语义求职搜索为例

    Designing Reward Signals for Portable Query Generation: A Case Study in Industrial Semantic Job Search

    [https://arxiv.org/abs/2606.27291](https://arxiv.org/abs/2606.27291)

    本文提出一种基于AI反馈的强化学习框架，通过设计稳健的奖励信号而非优化算法，来生成可迁移的求职搜索查询，解决了策略优化易利用大语言模型评判缺陷导致退化行为的问题。

    

    arXiv:2606.27291v1 公告类型：新 摘要：求职搜索平台依赖于低带宽的查询接口，这些接口往往无法捕捉候选人资料的高维复杂性。我们提出了一种端到端的RLAIF（基于AI反馈的强化学习）框架，用于生成“可迁移的”求职搜索查询，这些查询能抽象掉求职者特定的标识符，同时保留可泛化的资质。这一任务引入了一个高度对抗性的奖励表面，其中策略优化经常利用大语言模型作为评判者的评分标准中的缺陷，导致退化的逐字复制行为。我们进行了全面的实证实验，以分离优化机制与结构化奖励工程的影响。我们的结果表明，对于无评论家的优化器，性能主要取决于稳健的奖励塑造，使得具体算法的选择变得无关紧要。而无评论家的逐轮次基线方法（RLOO和REINFORCE++）也支持这一结论。

    arXiv:2606.27291v1 Announce Type: new  Abstract: Job-search platforms rely on low-bandwidth query interfaces that often fail to capture the high-dimensional complexity of candidate profiles. We present an end-to-end RLAIF (Reinforcement Learning from AI Feedback) framework to generate \emph{portable} job search queries, terms that abstract away seeker-specific identifiers while preserving generalizable qualifications. This task introduces a highly adversarial reward surface where policy optimization frequently exploits flaws in LLM-as-judge rubrics, resulting in degenerate verbatim-copying behaviors.   We conducted comprehensive empirical experiments to isolate the impact of optimization mechanics against structured reward engineering. Our results demonstrate that for critic-free optimizers, performance is overwhelmingly dictated by robust reward shaping, rendering the specific choice of algorithm largely immaterial. While critic-free per-rollout baseline methods (RLOO and REINFORCE++)
    
[^15]: 语言模型组合何时有效？基于67个前沿模型的协同失败上限：路由、投票与混合智能体系统

    When Does Combining Language Models Help? A Co-Failure Ceiling on Routing, Voting, and Mixture-of-Agents Across 67 Frontier Models

    [https://arxiv.org/abs/2606.27288](https://arxiv.org/abs/2606.27288)

    本研究揭示了多模型LLM系统（如路由、投票等）的准确率提升受限于所有模型在相同查询上同时出错的比率β，且β无法通过常用的平均成对误差相关系数ρ识别，基于67个模型的实验表明β常被低估，从而限制了组合策略的实际增益。

    

    arXiv:2606.27288v1 公告类型：新论文 摘要：多模型大语言模型系统（如路由、投票、级联、融合和混合智能体）被用于超越单一模型的准确性。我们证明，它们的增益受到一个领域内很少报告的量值的限制。对于任何输出为单一成员模型答案的策略，其准确率不能超过1减去β，其中β是每个模型在同一查询上同时出错的比率。相比之下，常用的诊断指标——平均成对误差相关系数ρ——无法识别β：具有相同边缘分布和成对相关性的误差定律可能具有不同的全错率。β的Clopper-Pearson界给出了一个有限样本的保证，即在训练路由器之前，任何路由器、投票或级联所能提供的最大增益。在来自21个提供商的67个模型中，一个基于四分相关校准的单因子模型仍然低估了全错尾部：在开放式数学问题上，观察到的β为0.052，而在完整的67模型高斯连接函数下为0.023，大约高出一倍。

    arXiv:2606.27288v1 Announce Type: new  Abstract: Multi-model LLM systems such as routing, voting, cascades, fusion, and mixture-of-agents are used to beat single-model accuracy. We show that their gain is capped by a quantity the field rarely reports. For any policy whose output is one member model answer, accuracy cannot exceed one minus beta, where beta is the rate at which every model is wrong on the same query. In contrast, the usual diagnostic, average pairwise error correlation rho, cannot identify beta: error laws with identical marginals and pairwise correlations can have different all-wrong rates. A Clopper-Pearson bound on beta gives a finite-sample certificate on the largest gain any router, vote, or cascade could deliver before training a router.   Across 67 models from 21 providers, a tetrachoric-calibrated single-factor model still underprices the all-wrong tail: on open-ended mathematics, observed beta is 0.052 versus 0.023 under the full 67-model Gaussian copula, about 
    
[^16]: 从解数据中恢复控制方程：线性和非线性常微分方程的可辨识性界

    Recovering Governing Equations from Solution Data: Identifiability Bounds for Linear and Nonlinear ODEs

    [https://arxiv.org/abs/2606.27285](https://arxiv.org/abs/2606.27285)

    本文通过引入豪斯多夫距离作为度量，首次为从解数据中恢复线性和非线性常微分方程建立了定量可辨识性界，填补了理论空白。

    

    从观测到的解数据中学习控制方程是科学机器学习中的一个基本挑战（参见文献\cite{bruntonDiscoveringGoverningEquations2016,kovachkiNeuralOperatorLearning2023,longPDENetLearningPDEs2018,rudyDatadrivenDiscoveryPartial2017,raonicConvolutionalNeuralOperators2023}），然而，关于在何种理论条件下可以从多个解观测中唯一且稳定地辨识出真实常微分方程（ODE）的研究仍基本空白，并且文献中缺乏对此类学习任务样本复杂性的定量分析。为填补这一空白，我们引入解集上的豪斯多夫距离作为比较微分方程的自然度量，因为它捕捉了所有允许初始条件下两个方程之间的最坏情况分离，从而编码了辨识问题的极小极大结构。我们针对一大类控制常微分方程建立了可辨识性界。

    arXiv:2606.27285v1 Announce Type: new  Abstract: Learning governing equations from observed solution data is a fundamental challenge in scientific machine learning \cite{bruntonDiscoveringGoverningEquations2016,kovachkiNeuralOperatorLearning2023,longPDENetLearningPDEs2018,rudyDatadrivenDiscoveryPartial2017,raonicConvolutionalNeuralOperators2023}, yet the theoretical conditions under which a ground-truth ODE can be uniquely and stably identified from multiple solution observations remain largely undeveloped, and no quantitative analysis of the sample complexity of such learning tasks exists in the literature. To address this gap, we introduce the Hausdorff distance on solution sets as the natural metric for comparing differential equations, since it captures the worst-case separation between two equations over all admissible initial conditions and thus encodes the minimax structure of the identification problem. We establish identifiability bounds for governing ODEs across a wide class 
    
[^17]: 线性模型在时间序列预测中能有多好？

    How Good Can Linear Models Be for Time-Series Forecasting?

    [https://arxiv.org/abs/2606.27282](https://arxiv.org/abs/2606.27282)

    本文挑战了“更大模型容量带来更高精度”的主流观点，通过岭回归证明，精心调整预处理（如上下文长度、归一化）能以极低成本显著缩小与大型模型的性能差距，并揭示了最优回溯长度与预测范围的非单调关系等反直觉模式。

    

    arXiv:2606.27282v1 公告类型：新 摘要：时间序列预测研究一直稳步转向更大的架构，从专门的Transformer到通用基础模型，其假设是容量是解锁精度的关键。我们持相反观点：通过调整预处理而非扩大模型规模，可以在成本低得多的前提下缩小大部分差距。我们使用岭回归作为测试平台，因为它具有闭式解和可解释的权重，这可以直接从搜索中读出最优超参数。我们在八个标准基准上对上下文长度、局部归一化、正则化和数据增强进行了搜索，发现了三种模式。(1) 最优回溯长度具有强烈的序列特异性，且通常在预测范围内是非单调的，拟合的幂律指数从ETTm2上的+0.46到Exchange和Traffic上的-0.19，挑战了“更长范围需要更长历史”的惯例。(2) 在一个学习的截断范围内进行归一化处理...

    arXiv:2606.27282v1 Announce Type: new  Abstract: Time-series forecasting research has been moving steadily toward larger architectures, from specialized transformers to general-purpose foundation models, on the assumption that capacity is what unlocks accuracy. We take the opposite position: most of the gap can be closed at far lower cost by tuning preprocessing rather than scaling models. We use Ridge regression as the testbed, since it has a closed-form solution and interpretable weights, which let the optimal hyperparameters be read off the search directly. We search over context length, local normalization, regularization, and augmentation on eight standard benchmarks and find three patterns. (1) Optimal lookback is strongly series-specific and often non-monotonic in forecast horizon, with fitted power-law exponents ranging from $+0.46$ on ETTm2 to $-0.19$ on Exchange and Traffic, challenging the convention that longer horizons need longer history. (2) Normalizing over a learned tr
    
[^18]: BetXplain：用于检测社交媒体上操纵性博彩广告的解释性标注数据集

    BetXplain: An Explanation-Annotated Dataset for Detecting Manipulative Betting Advertisements on Social Media

    [https://arxiv.org/abs/2606.27274](https://arxiv.org/abs/2606.27274)

    本文提出了BetXplain数据集，通过人工标注社交媒体上的博彩广告并附带解释，为自动检测操纵性和欺骗性广告提供了可解释的研究基础。

    

    arXiv:2606.27274v1 公告类型：新 摘要：近年来，社交媒体平台上博彩应用的推广显著增加。许多此类广告使用具有说服力的技术，可能误导用户、鼓励冒险行为，并可能影响用户的心理健康。然而，由于缺乏公开可用的标注数据集，关于自动检测操纵性和欺骗性博彩广告的研究仍然有限。在这项工作中，我们引入了一个新的博彩相关广告数据集，这些广告收集自两个广泛使用的社交媒体平台——Instagram和Reddit。这些广告被人工标注了操纵性和欺骗性广告实践。除了分类标签外，数据集还包括人工提供的解释，描述了每个标注背后的推理过程，从而支持对可解释性方法在检测操纵性广告方面的研究。此外，我们分析了这些广告中使用的策略。

    arXiv:2606.27274v1 Announce Type: new  Abstract: The promotion of betting applications on social media platforms has increased significantly in recent years. Many of these advertisements use persuasive techniques that may mislead users, encourage risky behavior, and potentially influence users' mental well-being. However, research on the automated detection of manipulative and deceptive betting advertisements remains limited due to the lack of publicly available annotated datasets. In this work, we introduce a new dataset of betting-related advertisements collected from two widely used social media platforms, Instagram and Reddit. The advertisements were manually annotated for manipulative and deceptive advertising practices. In addition to classification labels, the dataset includes human-provided explanations that describe the reasoning behind each annotation, enabling research into explainable approaches to detecting manipulative advertising. Furthermore, we analyze the strategies c
    
[^19]: Ribbon：可扩展的近似与稳健的不确定性量化

    Ribbon: Scalable Approximation and Robust Uncertainty Quantification

    [https://arxiv.org/abs/2606.27269](https://arxiv.org/abs/2606.27269)

    Ribbon通过影响函数线性化替代重复重拟合，实现了对贝叶斯自助不确定性的高效近似，并引入可调集中参数实现校准。

    

    arXiv:2606.27269v1 公告类型：交叉 摘要：对于复杂、高维或错误指定的模型，可靠地量化预测不确定性是困难的。完全贝叶斯方法和自助重采样方法都能提供原则性的不确定性估计，但对于现代机器学习模型而言，这些方法往往过于昂贵，因为它们需要后验采样或重复模型重拟合。我们提出了Ribbon，一种对狄利克雷加权自助不确定性的可扩展近似方法。Ribbon用围绕单个拟合模型的影响函数线性化替代了重复重拟合，保留了贝叶斯自助的一阶数据加权结构，同时仅需事后线性代数运算。Ribbon近似于贝叶斯自助或加权似然自助的重拟合目标。通过一个通用的集中参数，Ribbon提供了一个校准的狄利克雷加权族，其不确定性尺度可在验证数据上调整。我们证明了Ribbon在渐近意义上是等效的。

    arXiv:2606.27269v1 Announce Type: cross  Abstract: Reliably quantifying predictive uncertainty is difficult for complex, high-dimensional, or misspecified models. Both fully Bayesian and bootstrap resampling methods provide principled uncertainty estimates but are often too expensive for modern machine-learning models because they require posterior sampling or repeated model refitting. We introduce Ribbon, a scalable approximation to Dirichlet-reweighted bootstrap uncertainty. Ribbon replaces repeated refitting with an influence-function linearization around a single fitted model, preserving the first-order data-reweighting structure of the Bayesian bootstrap while requiring only post-hoc linear algebra. Ribbon approximates the Bayesian-bootstrap or weighted-likelihood-bootstrap refitting target. With a general concentration parameter, Ribbon gives a calibrated Dirichlet-reweighting family whose uncertainty scale can be tuned on validation data. We show that Ribbon is asymptotically eq
    
[^20]: RSPC：一个利用精神病学家标注对数字媒介关系中压力与精神疾病状况进行建模的基准数据集

    RSPC: A Benchmark for Modeling Stress and Psychiatric Conditions in Digitally Mediated Relationships using Psychiatrist Annotations

    [https://arxiv.org/abs/2606.27247](https://arxiv.org/abs/2606.27247)

    该论文提出了一个由精神病学家标注的Reddit异地恋帖子数据集（RSPC），用于在人际背景下联合建模精神疾病（如焦虑和抑郁）及其关系压力触发因素，并揭示了不同模型在障碍分类与关系触发检测任务上的性能差异。

    

    在自然语言处理中，精神健康状况通常被建模为孤立现象，缺乏人际背景。我们利用关于异地恋的Reddit帖子来同时捕捉心理健康困扰及其相关的关系触发因素。我们引入了关系压力与精神病学语料库（RSPC），其中包含1,799条由精神病学家标注的Reddit帖子，标注内容涉及诊断类别（包括最常见的情绪障碍：焦虑症和抑郁症）、关系压力触发因素以及关系阶段指示。我们在多标签障碍分类、关系触发检测和时间阶段预测任务上对七个微调后的Transformer模型和五个大型语言模型进行了基准测试。我们发现模型家族之间存在明显的任务依赖性差异，其中Claude-3-Haiku在障碍分类上表现最佳（宏F1=0.538），而GPT-4o在关系触发检测上表现最强。

    arXiv:2606.27247v1 Announce Type: new  Abstract: In NLP, mental health conditions are often modeled as isolated phenomena, without interpersonal context. We use Reddit posts about long-distance relationships to capture both mental health distress and associated relational triggers. We introduce the Relational Stress and Psychiatry Corpus (RSPC) containing 1,799 Reddit posts annotated by psychiatrists for diagnostic categories, including the most prevalent mood disorders (anxiety and depression), relational stressor triggers, and indications of relationship phase. We benchmark seven fine-tuned transformer models and five large language models across multi-label disorder classification, relational trigger detection, and temporal phase prediction tasks. We find clear task-dependent differences between model families, with Claude-3-Haiku achieving the best disorder classification performance (Macro-F1 = 0.538) and GPT-4o obtaining the strongest relational trigger detection performance (Mac
    
[^21]: 可解高维生成对抗网络中的有效协方差动力学

    Effective Covariance Dynamics in Solvable High-Dimensional GANs

    [https://arxiv.org/abs/2606.27246](https://arxiv.org/abs/2606.27246)

    本文提出了一个可解高维GAN模型，通过概率加权有效协方差统一处理类别相关、相关及非零均值的潜在结构，并证明了训练过程在高维极限下收敛到确定性常微分方程。

    

    我们研究了一个可解的高维生成对抗网络（GAN）训练模型，其中线性生成器从具有结构化潜在协方差的数据中学习低维子空间。先前的可解GAN分析假设无条件信号具有对角潜在协方差；我们将多特征判别器设置扩展到类别相关、相关且非零均值的潜在结构。对于二次能量判别器，所有此类异质性通过概率加权的有效二阶矩进入动力学。我们证明了随机微观训练过程在高维极限下收敛到由该有效协方差控制的确定性常微分方程。在匹配协方差特例中，稳定性分析得出了由学习率和噪声水平决定的模态可解区间：当主导有效特征值跨越阈值时，学习开始。

    arXiv:2606.27246v1 Announce Type: new  Abstract: We study a solvable high-dimensional model of generative adversarial network (GAN) training in which a linear generator learns a low-dimensional subspace from data with structured latent covariance. Prior solvable GAN analyses assume unconditional signals with diagonal latent covariance; we extend the multi-feature discriminator setting to class-dependent, correlated, and non-zero-mean latent structure. For the quadratic energy discriminator, all such heterogeneity enters the dynamics through a probability-weighted effective second moment. We prove that the stochastic microscopic training process converges, in the high-dimensional limit, to deterministic ordinary differential equations governed by this effective covariance. In the matched-covariance specialization, the stability analysis yields a mode-wise solvable interval determined by the learning rates and noise level: learning begins when the leading effective eigenvalue crosses the
    
[^22]: 更新的几何：词汇规模下的费舍尔对齐

    The Geometry of Updates: Fisher Alignment at Vocabulary Scale

    [https://arxiv.org/abs/2606.27242](https://arxiv.org/abs/2606.27242)

    本文提出FisherSketch方法，通过将头费舍尔对齐等价为联合激活-误差空间中核均值嵌入的余弦值，实现了在词汇规模下高效、可识别的无训练源选择，解决了传统表示度量不可识别而经典几何度量计算成本过高的问题。

    

    arXiv:2606.27242v1 公告类型：交叉 摘要：在具有共享词汇表的大型语言模型家族中进行无训练源选择，出现在SMILES、蛋白质和基因组序列等科学字符串领域，其中候选语料库共享分词器但预测目标不同。这导致了一种“激活-暗区”现象：在缺乏关于标签条件误差几何假设的情况下，表示相似性度量可能无效，而经典的更新几何度量在词汇规模下计算成本过高。我们证明，在共享输出头的设定下，表示度量（如CKA）对于迁移是不可识别的；模型可以共享完全相同的表示，却具有正交的头更新。关键恒等式是：头费舍尔对齐恰好是联合激活-误差空间中核均值嵌入之间的余弦值，从而揭示了激活、误差和耦合因子，而无需实例化费舍尔矩阵。FisherSketch通过直接估计这一余弦值来工作。

    arXiv:2606.27242v1 Announce Type: cross  Abstract: Training-free source selection for LLM families with shared vocabularies arises in scientific string domains such as SMILES, protein, and genomic sequences, where candidate corpora share a tokenizer but differ in prediction targets. This creates an activation-dark regime: representation-similarity metrics can be uninformative without assumptions about label-conditioned error geometry, while classical update-geometry metrics are computationally prohibitive at vocabulary scale. We show that, in a shared-output head setting, representation metrics (e.g., CKA) are non-identifiable for transfer; models can share identical representations yet have orthogonal head updates. The key identity is that head Fisher alignment is exactly a cosine between kernel mean embeddings in the joint activation-error space, exposing activation, error, and coupling factors rather than requiring a materialized Fisher matrix. FisherSketch estimates this cosine dir
    
[^23]: CARVE：面向分块并行线性注意力的内容感知递归与价值效率模型

    CARVE: Content-Aware Recurrent with Value Efficiency for Chunk-Parallel Linear Attention

    [https://arxiv.org/abs/2606.27229](https://arxiv.org/abs/2606.27229)

    CARVE通过仅在键轴上擦除的单一原则，解决了递归模型中的记忆盲区门控、参数浪费和WY形式求解器失效三个问题，实现了高效的内容感知递归线性注意力。

    

    arXiv:2606.27229v1 公告类型：跨领域 摘要：递归模型必须通过遗忘来记住，然而现有技术决定遗忘什么时并不参考已存储的内容——门控机制仅看到当前到达的标记，而非即将修改的记忆。这种记忆盲区门控是当前主流delta规则架构（GDN-2）中三个相互耦合的缺陷之一：价值轴擦除掩码在价值投影的尺度上浪费参数，并且——正如我们所证明的——在数学上阻碍了使递归训练与Transformer相媲美的WY形式三角分块求解器。我们提出CARVE（内容感知递归与价值效率模型），通过一个原则解决所有三个问题：仅在键轴上擦除。这在数学上被证明是WY形式求解器保持有效的必要且充分条件。在此框架下，CARVE复用已写入GPU内存的递归输出张量作为擦除门控的免费内容信号，并替换逐值写入门控。

    arXiv:2606.27229v1 Announce Type: cross  Abstract: Recurrent models must forget in order to remember, yet the state of the art decides what to erase without consulting what is stored -- the gate sees only the arriving token, not the memory it is about to modify. This memory-blind gating is one of three coupled defects in the leading delta-rule architecture (GDN-2): the value-axis erase mask wastes parameters at the scale of the value projection, and -- as we prove -- mathematically prevents the WY-form triangular chunk solver that makes recurrent training competitive with Transformers.   We introduce CARVE (Content-Aware Recurrent with Value Efficiency), which resolves all three problems through one principle: erase only on the key axis. This is provably necessary and sufficient for the WY-form solver to remain valid. Within it, CARVE reuses the recurrent output tensor -- already written to GPU memory -- as a free content signal for the erase gate, and replaces the per-value write-gate
    
[^24]: 层级Muon：用于高效Muon优化的分块Newton-Schulz更新方法

    Hierarchical Muon: Tiled Newton-Schulz Updates for Efficient Muon Optimization

    [https://arxiv.org/abs/2606.27216](https://arxiv.org/abs/2606.27216)

    本文提出层级Muon（HiMuon），通过将动量梯度矩阵分块并独立应用Newton-Schulz映射，大幅降低计算复杂度，同时保留局部谱相互作用。

    

    arXiv:2606.27216v1 公告类型：交叉 摘要：Muon类优化器通过对动量梯度矩阵应用有限次Newton-Schulz映射，为密集神经网络权重构建更新方向。对于一个$H \times W$的矩阵，其中$r=\min\{H,W\}$且$s=\max\{H,W\}$，全矩阵Newton-Schulz更新的$K$步计算需要$O(r^2 s K)$的工作量，并通过重复的Gram矩阵乘积耦合所有行和列。我们引入了层级Muon（HiMuon），一种用于Muon类优化的分块Newton-Schulz方案。HiMuon将每个动量梯度矩阵划分为$T \times T$个分块，独立地对每个分块应用相同的有限次Newton-Schulz映射，然后重新组合结果。对于低于矩阵维度的有限$T$，HiMuon定义了一个局部矩阵函数映射，而非全矩阵更新的收敛近似：谱相互作用在分块内部保留，并在分块边界处丢弃。对于固定的有限$T$，主要的Newton-Schulz计算量减少到$O(H$

    arXiv:2606.27216v1 Announce Type: cross  Abstract: Muon-type optimizers construct update directions for dense neural-network weights by applying a finite Newton-Schulz map to momentum-gradient matrices. For an $H \times W$ matrix, with $r=\min\{H,W\}$ and $s=\max\{H,W\}$, $K$ steps of the full-matrix Newton-Schulz update require $O(r^2 s K)$ work and couple all rows and columns through repeated Gram matrix products. We introduce Hierarchical Muon (HiMuon), a tiled Newton-Schulz scheme for Muon-type optimization. HiMuon partitions each momentum-gradient matrix into $T \times T$ tiles, applies the same finite Newton-Schulz map independently to each tile, and reassembles the results. For finite $T$ below the matrix dimensions, HiMuon defines a local matrix-function map rather than a convergent approximation to the full-matrix update: spectral interactions are preserved within tiles and discarded across tile boundaries. For fixed finite $T$, the leading Newton-Schulz work decreases to $O(H
    
[^25]: 图神经网络跨领域应用：你需要了解的所有洞见

    Graph Neural Networks Applications Across Domains: All Insights You Need

    [https://arxiv.org/abs/2606.27202](https://arxiv.org/abs/2606.27202)

    本综述统一了图神经网络的设计空间，从第一性原理推导谱域与空域方法，并系统评估了十二个应用领域中图结构计算成本的合理性，明确了当前架构的表达能力边界。

    

    arXiv:2606.27202v1 公告类型：新 摘要：图神经网络已从小众的表示学习技术发展成为处理带有关系结构数据的默认模型类别。当前有趣的问题不再是消息传递是否有助于特定数据集，而是图结构在哪些场景下值得其计算成本，在哪些场景下则不然。本综述围绕单一设计空间组织该领域，从共享第一性原理推导出谱域和空域公式，并将表达能力与Weisfeiler-Leman层次结构联系起来，明确陈述了当前架构能区分和不能区分的内容。以此方法论为基础，我们考察了十二个应用领域，包括推荐与社交网络、知识图谱与语言模型集成、药物发现与分子性质学习、医疗健康与神经科学、计算机视觉、交通与城市计算、电力与可再生能源系统、无线与第六代通信等。

    arXiv:2606.27202v1 Announce Type: new  Abstract: Graph neural networks have moved from a niche representation-learning technique to the default model class wherever data carry relational structure. The interesting question is no longer whether message passing helps on a given dataset, but where graph structure earns its computational cost and where it does not. This survey organises the field around a single design space, derives the spectral and spatial formulations from shared first principles, and connects expressive power to the Weisfeiler-Leman hierarchy with explicit statements of what current architectures can and cannot separate. Against that methodological backbone we examine twelve application domains, among them recommendation and social networks, knowledge graphs and language-model integration, drug discovery and molecular property learning, healthcare and neuroscience, computer vision, traffic and urban computing, power and renewable-energy systems, wireless and sixth-gene
    
[^26]: 通过特征诱导的信息流解释时序图神经网络

    Explaining Temporal Graph Neural Networks via Feature-induced Information Flow

    [https://arxiv.org/abs/2606.27201](https://arxiv.org/abs/2606.27201)

    提出了一种基于正则化相关性度量框架的新归因方法，通过分析所有事件相关变量中的完整信息流，克服了现有方法忽略事件诱导变量路径的局限，从而更全面地解释基于事件的时序图神经网络。

    

    arXiv:2606.27201v1 公告类型：新 摘要：基于事件的时序图神经网络（ETGNNs）在社交网络分析、疫情追踪、推荐系统和政治事件预测等多种应用中表现出强大的性能。然而，其日益增长的复杂性对可解释性提出了重大挑战。现有的解释方法仅关注ETGNNs中信息流的一部分，通常追踪从事件相关嵌入到输出的贡献。因此，它们忽略了通过事件诱导变量的重要路径，这些变量介导了节点之间的交互，从而在捕捉长期时序依赖中发挥核心作用。为克服这一局限，我们提出了一种新的归因方法，分析所有事件相关变量中的完整信息流。我们的方法建立在最近的正则化相关性度量（NRM）框架之上，从而实现了可解释性。

    arXiv:2606.27201v1 Announce Type: new  Abstract: Event-based Temporal Graph Neural Networks (ETGNNs) have demonstrated strong performance across a wide range of applications, including social network analysis, epidemic tracing, recommender systems, and political event forecasting. However, their increasing complexity poses significant challenges for explainability. Existing explanation methods focus only on a subset of the information flow within ETGNNs, typically tracing contributions from the event-related embeddings to the output. Consequently, they overlook the important pathways through event-induced variables, which mediate interactions between nodes and thereby play a central role in capturing long-range temporal dependencies. To overcome this limitation, we propose a novel attribution method that analyzes the \emph{entire} information flow through all event-associated variables. Our method is built upon the recent Normalized Relevance Measure (NRM) framework, which enables expl
    
[^27]: 利用大型语言模型进行预测：通过特征引导提升泛化能力

    Forecasting With LLMs: Improved Generalization Through Feature Steering

    [https://arxiv.org/abs/2606.27199](https://arxiv.org/abs/2606.27199)

    本研究发现通过增强大型语言模型中的时间感知特征，可以有效减少预测中的前瞻偏差，提升泛化能力，而干预前瞻偏差特征则无效。

    

    arXiv:2606.27199v1 公告类型：新论文 摘要：成功的预测需要识别历史与未来世界状态之间的模式，这些模式能够泛化到未来的观测中。我们将大型语言模型应用于多种预测任务，并使用稀疏自编码器检查其内部状态，以理解它们是否依赖特定时间的知识还是可泛化的模式。我们的分析识别出与时间感知推理和前瞻偏差推理相关的特征。随后，我们将LLMs应用于一个完全不同的领域，并对这些特征进行干预。我们发现，增强时间感知特征显著减少了预测提示中的前瞻偏差，同时保留了通用的推理性能。相比之下，引导候选的前瞻偏差特征并未产生效果。这些结果表明，可解释的时间特征可以因果性地将LLMs转向更基于历史事实的推理。

    arXiv:2606.27199v1 Announce Type: new  Abstract: Successful forecasting involves identifying patterns between historical and future states of the world which generalize to future observations. We apply LLMs to a variety of forecasting tasks and inspect their internal states using sparse autoencoders to understand whether they appear to rely on time-specific pieces of knowledge versus generalizable patterns. Our analyses identify features associated with both time-aware reasoning and look-ahead-biased reasoning. We then apply the LLMs to an entirely different domain and intervene on these features. We find that amplifying time-awareness features substantially reduces look-ahead bias on forecasting prompts while preserving general reasoning performance. In contrast, steering the candidate look-ahead-bias features does not produce an effect. These results suggest that interpretable temporal features can be used to causally shift LLMs toward more historically grounded reasoning.
    
[^28]: 基于视觉语言模型引导的势能奖励塑形自动化

    Automating Potential-based Reward Shaping with Vision Language Model Guidance

    [https://arxiv.org/abs/2606.27180](https://arxiv.org/abs/2606.27180)

    提出VLM-PBRS框架，利用视觉语言模型反馈自动学习势能函数，在保留最优策略的同时消除了手工定义势能函数的需求。

    

    arXiv:2606.27180v1 公告类型：交叉 摘要：稀疏奖励对强化学习智能体来说本质上是具有挑战性的，因为它们缺乏中间反馈来引导探索，也无法将稀疏的成功奖励正确归因于轨迹的相关部分。朴素的奖励塑形可能导致奖励黑客行为，产生利用辅助信号而非解决预期任务的策略。基于势能的奖励塑形（PBRS）保证了最优策略集的保留，但需要在状态空间上定义启发式势能函数。在这项工作中，我们引入了视觉语言模型引导的PBRS框架VLM-PBRS，该框架直接从视觉语言模型（VLM）反馈中学习势能函数。我们查询一个轻量级VLM以获取图像对的偏好，并利用这些偏好训练势能函数模型。由于该方法基于势能奖励塑形，它保留了原始最优策略，并消除了手动定义势能函数的需求。

    arXiv:2606.27180v1 Announce Type: cross  Abstract: Sparse rewards are inherently challenging for reinforcement learning agents as they lack intermediate feedback to guide exploration and to correctly attribute the sparse success rewards to relevant parts of the trajectory. Naive reward shaping can induce reward hacking, yielding policies that exploit auxiliary signals instead of solving the intended task. Potential-based reward shaping (PBRS) guarantees preservation of the optimal policy set, but requires the definition of a heuristic potential function over the state space. In this work, we introduce the VLM-guided PBRS framework VLM-PBRS that learns the potential function directly from vision language model (VLM) feedback. We query a lightweight VLM to obtain preferences over image pairs and train a model of the potential function using these preferences. As this approach is based on potential-based reward shaping, it preserves the original optimal policies, and removes the need for 
    
[^29]: RecallRisk-BERT：面向报告后医疗器械召回分诊的多任务框架

    RecallRisk-BERT: A Multi-Task Framework for Post-Report Medical Device Recall Triage

    [https://arxiv.org/abs/2606.27174](https://arxiv.org/abs/2606.27174)

    本文提出了一个名为RecallRisk-BERT的多任务框架，通过联合建模召回严重性与根本原因类别，实现了对医疗器械报告后召回记录的高效自动分诊，填补了现有研究在联合预测方面的空白。

    

    医疗器械召回是保护患者安全的关键监管机制。美国食品药品监督管理局（FDA）召回记录数量的不断增长，给报告后的召回分诊、严重性评估及根本原因解释带来了挑战。现有研究大多分别处理召回发生预测或根本原因分析，而对召回严重性与根本原因类别的联合建模关注有限。我们利用来自openFDA的54,165条FDA医疗器械召回记录（覆盖2002年至2025年10月期间），开发了一个自动化召回分诊框架。我们首先评估了经典机器学习模型和基于提升的模型在召回严重性和根本原因类别预测上的表现。随后，我们开发了RecallRisk-BERT，这是一个多任务模型，它结合了基于PubMedBERT的召回叙述文本表示与基于嵌入的结构化分类特征（包括产品代码、法规编号等）表示。

    arXiv:2606.27174v1 Announce Type: new  Abstract: Medical device recalls are a critical regulatory mechanism for protecting patient safety. The growing volume of FDA recall records presents challenges in post-report recall triage, severity assessment, and root-cause interpretation. Existing studies mostly address recall occurrence prediction or root-cause analysis separately, while joint modeling of recall severity and root-cause categories has received limited attention. We develop an automated recall triage framework using 54,165 FDA medical device recall records from openFDA, covering the period from 2002 to October 2025. We first evaluate classical machine learning and boosting-based models for recall severity and root-cause category prediction. We then develop RecallRisk-BERT, a multi-task model that combines PubMedBERT-based textual representations of recall narratives with embedding-based representations of structured categorical features, including product code, regulation numbe
    
[^30]: 基于模型辅助采样的随机梯度优化

    Stochastic Gradient Optimization with Model-Assisted Sampling

    [https://arxiv.org/abs/2606.27171](https://arxiv.org/abs/2606.27171)

    本文提出一种模型辅助采样框架，结合调查抽样理论，利用辅助梯度预测模型降低随机梯度估计的方差，从而提升机器学习优化性能。

    

    本文针对机器学习优化中随机梯度估计的方差问题展开研究。深度学习依赖于小批量方法（如随机梯度下降），这些方法虽能近似全梯度，但会引入噪声，从而在收敛稳定性、速度和泛化能力之间产生权衡。现有的方法，包括方差缩减技术（例如SVRG和SAG）以及自适应优化器，旨在缓解梯度噪声，但可能会增加额外的计算开销。我们提出了一种模型辅助采样框架，通过调查抽样理论来解释小批量梯度，将数据集视为一个固定的有限总体，梯度视为基于样本的估计值。我们的目标是结合基于样本估计和方差缩减的视角，将机器学习优化与调查抽样理论联系起来。通过引入辅助梯度预测模型，我们构建了模型辅助的估计器，以降低方差并提升优化性能。

    arXiv:2606.27171v1 Announce Type: new  Abstract: This work addresses the problem of variance in stochastic gradient estimation for machine learning optimization. Deep learning relies on mini-batch methods such as stochastic gradient descent, which approximate full gradients but introduce noise, creating trade-offs between convergence stability, speed, and generalization. Existing methods, including variance reduction techniques (e.g., SVRG and SAG) and adaptive optimizers, aim to mitigate gradient noise but may introduce additional computational overhead. We propose a model-assisted sampling framework that interprets mini-batch gradients through survey sampling theory, treating the dataset as a fixed finite population and gradients as sample-based estimates. Our aim is to bridge machine learning optimization and survey sampling theory by combining their perspectives on sample-based estimation and variance reduction. By incorporating auxiliary gradient-prediction models, we construct mo
    
[^31]: 学习折叠：LeHome 2026挑战赛获奖方案（在线第一名，线下第二名）

    Learning to Fold: prizewinning solution at LeHome Challenge 2026 (1st place online, 2nd offline)

    [https://arxiv.org/abs/2606.27163](https://arxiv.org/abs/2606.27163)

    本文提出了一种通过强化学习改进的视觉-语言-动作策略，实现了在线仿真和真实世界双臂衣物折叠的高性能，并整合了多种优化技术。

    

    本文描述了我在LeHome 2026挑战赛（ICRA 2026双臂衣物折叠竞赛）中的解决方案。该系统在在线（仿真）环节中于62支队伍中排名第一，并在真实世界决赛中排名第二。它通过强化学习循环改进了视觉-语言-动作（VLA）策略。该策略本身即为价值函数：同一网络不仅预测动作，还预测成功、进度以及若干任务相关的未来量，这些预测用于优势估计、实时故障检测和候选选择。本工作主要将现有强化学习思想与工程和优化贡献相结合，这些贡献可作为整体配方或单独使用：结合AWR和RECAP用于流匹配VLA；通过HuggingFace Hub实现异步分布式训练/部署流水线；通过汤普森采样进行推理时超参数优化；以及包含相机对齐工具和强数据增强的仿真到现实迁移方案。

    arXiv:2606.27163v1 Announce Type: cross  Abstract: I describe my solution to the LeHome Challenge 2026, an ICRA 2026 competition on bimanual garment folding. The system placed 1st of 62 teams in the online (simulation) round and 2nd in the real-world final. It improves a vision-language-action (VLA) policy with a reinforcement-learning loop. The policy is its own value function: the same network that predicts actions also predicts success, progress, and a few task-relevant future quantities, and those predictions drive advantage estimation, live failure detection, and candidate selection. The work mostly recombines existing RL ideas with engineering and optimization contributions that can be used together as one recipe or individually: AWR + RECAP combined for flow-matching VLA; an asynchronous distributed training / rollout pipeline through HuggingFace Hub; inference-time hyperparameters optimization via Thompson sampling; a sim-to-real recipe with camera-alignment tooling, heavy augm
    
[^32]: DMuon：以接近Adam开销实现高效的分布式Muon训练

    DMuon: Efficient Distributed Muon Training with Near-Adam Overhead

    [https://arxiv.org/abs/2606.27153](https://arxiv.org/abs/2606.27153)

    本文提出DMuon，一个开源分布式Muon优化器实现，通过即插即用模块设计将训练开销降至接近Adam水平，无需框架修改。

    

    arXiv:2606.27153v1 公告类型：跨领域 摘要：以Muon为代表的基于矩阵正交化的优化器，在广泛现代深度学习任务中展现出强大的收敛性能。矩阵感知的更新方式为传统的逐元素优化提供了极具吸引力的替代方案，尤其在模型架构规模持续增长且异构性日益增强的背景下。然而，当前围绕逐元素优化器假设构建的分布式训练基础设施，与Muon这类矩阵级优化器并不匹配——后者的更新会耦合整个权重矩阵，并需要昂贵的牛顿-舒尔茨迭代。原始Muon实现的开销超过前向和后向传播成本的两倍。为弥合这一差距，我们提出了DMuon，一个开源分布式Muon实现，可作为即插即用模块集成到现有训练流程中，无需修改框架层面。在具身基础模型和大语言模型（LL

    arXiv:2606.27153v1 Announce Type: cross  Abstract: Matrix-orthogonalization-based optimizers, exemplified by Muon, have demonstrated strong convergence behavior across a wide range of modern deep learning workloads. The matrix-aware updates offer a compelling alternative to conventional element-wise optimization, particularly as model architectures continue to grow in scale and heterogeneity. Yet contemporary distributed training infrastructure built around the assumption of element-wise optimizers is poorly matched to matrix-level optimizers such as Muon, whose updates couple entire weight matrices and require costly Newton-Schulz iterations. Vanilla Muon implementations incur more than 2x the cost of forward and backward passes. To close this gap, we present DMuon, an open-source distributed Muon implementation that integrates into existing training pipelines as a drop-in module, with no framework-level modifications. Across both embodied foundation model and large language model (LL
    
[^33]: fTNN：一种用于分数阶偏微分方程的张量神经网络

    fTNN: a tensor neural network for fractional PDEs

    [https://arxiv.org/abs/2606.27140](https://arxiv.org/abs/2606.27140)

    本文提出了一种名为fTNN的确定性张量神经网络子空间方法，通过几何自适应积分分裂和边界奇异性感知试验函数，有效解决了有界域上分数阶拉普拉斯算子相关问题的低正则性解和积分计算挑战。

    

    arXiv:2606.27140v1 公告类型：新论文 摘要：我们开发了fTNN，一种确定性的张量神经网络子空间方法，用于处理有界域上涉及分数阶拉普拉斯算子的问题，以分数阶泊松方程和含时分数阶对流扩散方程为典型代表。该工作采用了一种几何自适应积分分裂方法，该方法包含一个依赖于空间位置的近场半径，将分数阶拉普拉斯算子分解为三个部分：奇异近场、规则内部远场和解析外部远场。然后，奇异径向积分通过高斯-雅可比求积法处理，规则径向积分通过高斯求积法处理，角度变量通过确定性角度求积法处理，从而形成一个完全确定性的分数阶拉普拉斯算子积分框架。为了精确解析低正则性解及其相关的损失泛函，我们构造了边界奇异性感知的试验函数。

    arXiv:2606.27140v1 Announce Type: new  Abstract: We develop the fTNN, a deterministic tensor neural network subspace method for problems involving the fractional Laplacian on bounded domains, taking the fractional Poisson equation and time-dependent fractional advection-diffusion equation as typical representatives. The work employs a geometry-adapted integration split featuring a spatially dependent near-field radius, which decomposes the fractional Laplacian into three contributions: a singular near field, a regular interior far field, and an analytical exterior far field. Then the singular radial integrals are treated by Gauss-Jacobi quadrature, the regular radial integrals by Gauss quadrature, and the angular variables by deterministic angular quadrature, yielding a fully deterministic integration framework of the fractional Laplacian operator. To accurately resolve low-regularity solutions and the associated loss functional, we construct boundary-singularity-aware trial functions 
    
[^34]: 科尔莫戈罗夫-阿诺德网络（KAN）在气动预测中的应用：与MLP和GNN的比较

    Kolmogorov Arnold networks (KAN) for aerodynamic prediction: a comparison with MLPs and GNNs

    [https://arxiv.org/abs/2606.27126](https://arxiv.org/abs/2606.27126)

    本文通过气动预测任务，系统比较了科尔莫戈罗夫-阿诺德网络（KAN）与MLP及GNN的性能，揭示了KAN在流体动力学代理建模中的优势与局限。

    

    摘要：科尔莫戈罗夫-阿诺德网络（KAN）最近被引入作为一种（深度）神经网络架构，其可训练参数用于调整激活函数，而非传统架构（如深度多层感知器MLP）核心中仿射变换的系数。该架构基于科尔莫戈罗夫-阿诺德定理，使其具备通用逼近特性。尽管KAN的出现令人兴奋，但目前关于KAN在符号回归、通用机器学习、自然语言处理或计算机视觉等经典领域是否优于深度多层感知器（MLP）存在争议。本文评估了KAN在流体动力学代理建模中的性能，并细致比较了其与MLP和图神经网络（GNN）的差异。为此，我们考虑了预测表面压力分布的任务。

    arXiv:2606.27126v1 Announce Type: new  Abstract: Kolmogorov Arnold networks (KAN) have recently been introduced as a (deep) neural network architecture whose trainable parameters adapt the activation functions, instead of the coefficients of the affine transformations at the core of traditional architectures such as deep multilayer perceptrons (MLPs). This architecture builds on the Kolmogorov-Arnold theorem, which endows it with universal approximation properties. While the advent of KANs has been received with excitement, there is a current debate about the possible KAN supremacy over deep multilayer perceptrons (MLPs) for classic fields such as symbolic regression, generic-purpose machine learning, natural language processing or computer vision. Here we assess the performance of KANs --and its nuanced comparison against MLPs and graph neural networks (GNNs)-- in the realm of fluid dynamics surrogate modelling. To that aim, we consider the task of predicting the surface pressure dist
    
[^35]: 面向容错量子计算的高效基础解码器

    Efficient foundation decoders for fault-tolerant quantum computing

    [https://arxiv.org/abs/2606.27119](https://arxiv.org/abs/2606.27119)

    提出神经迁移统一框架NTU，通过利用可扩展码族的代数结构对齐不同码距下的解码任务，实现小码知识加速大码解码器训练，并实例化为NTU-Transformer，在容错量子计算中取得优于相关方法的性能。

    

    基础解码器是一类高容量神经解码器，凭借在大码距下准确高效的解码能力，成为容错量子计算的主要候选方案。然而，其构建常面临严峻的扩展障碍：随着码距增大，综合征生成和神经优化的成本迅速攀升。为突破这一瓶颈，我们设计了神经迁移统一框架（NTU），这是一个用于高效基础解码器的统一框架。NTU的核心特性在于，它能通过可扩展码族共有的代数结构，对齐不同码距下的解码任务，从而使在小码上习得的知识能够加速大规模解码器的训练。我们将NTU实例化为NTU-Transformer，这是一个专为平面表面码和双变量自行车码设计的基于Transformer的神经解码器。在电路级噪声下的平面表面码中，NTU-Transformer的表现优于相关方法。

    arXiv:2606.27119v1 Announce Type: cross  Abstract: Foundation decoders, a class of high-capacity neural decoders, are leading candidates for fault-tolerant quantum computing, with accurate and efficient decoding at large code distances. However, their construction often faces a steep scaling barrier, as larger code distances rapidly amplify the cost of syndrome generation and neural optimization. To address this bottleneck, here we devise neural transfer unification (NTU), a unified framework for efficient foundation decoders. A central feature of NTU is its ability to align decoding tasks across code distances via algebraic structures shared by scalable code families, which enables knowledge learned on smaller codes to accelerate large-scale decoder training. We instantiate NTU as NTU-Transformer, a transformer-based neural decoder tailored for planar surface codes and bivariate bicycle codes. For planar surface codes under circuit-level noise, NTU-Transformer outperforms correlation-
    
[^36]: 跨头注意力提升网络与未观测混杂下的逆倾向得分方法

    Cross-Head Attention Uplift Network with Inverse Propensity Score under Unobserved Confounding

    [https://arxiv.org/abs/2606.27114](https://arxiv.org/abs/2606.27114)

    本文提出跨头注意力提升网络和鲁棒对抗逆倾向得分方法，通过动态整合组间表示和对抗性优化倾向权重，解决了未观测混杂下的个体处理效应估计偏差问题。

    

    arXiv:2606.27114v1 公告类型：新 摘要：提升建模对于估计个体处理效应至关重要，但面临双重挑战：灵活利用组间相似性以增强判别能力，以及在未观测混杂场景下的去偏问题。本文提出跨头注意力提升网络（CHAUN）和鲁棒对抗逆倾向得分（RA-IPS）方法来解决这些局限。CHAUN采用共享特征嵌入和跨头注意力机制，动态整合处理组和对照组的表示，增强组间相关性建模。理论上，我们证明即使存在未观测混杂因素，获得真实倾向得分也能确保ITE的可识别性。对于缺乏真实倾向得分的实际场景，RA-IPS在约束不确定性集内对抗性地优化倾向权重，以减轻未观测变量带来的偏差。在公开数据集（CRITEO-）上的实验表明……

    arXiv:2606.27114v1 Announce Type: new  Abstract: Uplift modeling, crucial for estimating individual treatment effects (ITE), faces dual challenges: flexibly leveraging inter-group similarity to enhance discriminative power and debiasing under unobserved confounding scenarios. In this paper, we propose the Cross-Head Attention Uplift Network (CHAUN) and Robust Adversarial Inverse Propensity Score (RA-IPS) method to address these limitations. CHAUN employs shared feature embeddings and cross-head attention mechanisms to dynamically integrate treatment-specific and control-specific representations, enhancing inter-group correlation modeling. Theoretically, we prove that access to the true propensity scores ensures ITE identifiability even with unobserved confounders. For practical scenarios lacking true propensity scores, RA-IPS adversarially optimizes propensity weights within constrained uncertainty sets to mitigate bias from unobserved variables. Experiments on public datasets (CRITEO-
    
[^37]: 带残差加权校正的重球Q学习

    Heavy-Ball Q-Learning with Residual Weighting Correction

    [https://arxiv.org/abs/2606.27112](https://arxiv.org/abs/2606.27112)

    本文提出了一种带残差加权校正的重球Q学习方法，通过切换线性系统视角证明了其收敛性和加速效果，并扩展到了线性函数逼近场景。

    

    本文提出了一种用于强化学习的校正重球Q学习方法，并证明了其收敛性。同时，文章识别了该方法在理论上保证比标准Q学习收敛更快的条件。随后，相同的构造被扩展到线性函数逼近的Q学习中，并推导出了类似的收敛性和加速结论。该分析基于Q学习算法的切换线性系统表示以及相关切换族的联合谱半径。这种切换线性系统视角在标准Q学习分析中并不常用，它为理解重球动量如何加速Q学习提供了补充框架和新的见解。

    arXiv:2606.27112v1 Announce Type: cross  Abstract: This paper proposes a corrected heavy-ball Q-learning method for reinforcement learning (RL) and establishes its convergence. It also identifies conditions under which the method is theoretically guaranteed to converge faster than standard Q-learning. The same construction is then extended to Q-learning with linear function approximation, where analogous convergence and acceleration statements are derived. The analysis is based on a switched linear system (SLS) representation of Q-learning algorithms and on the joint spectral radius (JSR) of the associated switching families. This SLS viewpoint is not commonly used in standard analyses of Q-learning, and it provides a complementary framework and new insight into how heavy-ball momentum can accelerate Q-learning.
    
[^38]: 基于Transformer的细菌拉曼光谱分类与留一法交叉验证

    Transformer-Based Classification of Bacterial Raman Spectra with LOOCV

    [https://arxiv.org/abs/2606.27096](https://arxiv.org/abs/2606.27096)

    本研究证明，基于Transformer的模型在细菌拉曼光谱分类中，通过嵌套留一重复交叉验证，其性能显著优于传统机器学习方法，且能直接处理原始光谱数据。

    

    基于Transformer的模型最近在拉曼光谱分类中引起了越来越多的关注。本研究采用嵌套的留一重复交叉验证框架，系统评估了基于Transformer的方法，并将其与结合PCA或ICA与LDA、SVM及随机森林分类器的传统机器学习流程进行了比较。研究使用了包含来自6种细菌物种和9个独立测量重复的5417个单细胞光谱的细菌拉曼数据集。Transformer在独立测试重复中始终达到最高的分类性能，并显著优于所有传统方法。对学习到的潜在特征空间的分析显示，与基于PCA和ICA的表示相比，其类别分离效果得到了改善。此外，Transformer在直接应用于未经预处理的原始拉曼光谱时仍保持了优越的性能。

    arXiv:2606.27096v1 Announce Type: new  Abstract: Transformer-based models have recently attracted increasing attention for Raman spectral classification. In this study, a transformer-based approach was systematically evaluated using a nested leave-one-replicate-out cross-validation framework and compared with conventional machine-learning pipelines combining PCA or ICA with LDA, SVM, and Random Forest classifiers. A bacterial Raman dataset comprising 5,417 single-cell spectra from six bacterial species and nine independent measurement replicates was used. The transformer consistently achieved the highest classification performance across independent test replicates and significantly outperformed all conventional approaches. Analysis of the learned latent feature space revealed improved class separation compared with PCA- and ICA-based representations. Furthermore, the transformer maintained superior performance when applied directly to raw Raman spectra without preprocessing, demonstra
    
[^39]: 无数据储层特征用于高效长视野冷启动持续学习

    Data-Free Reservoir Features for Efficient Long-Horizon Cold-Start Continual Learning

    [https://arxiv.org/abs/2606.27095](https://arxiv.org/abs/2606.27095)

    本文提出CIRCLE方法，利用从未训练过的固定双向二维储层特征和流式线性判别分析，在冷启动持续学习中实现高效且无需数据回放的类增量学习。

    

    arXiv:2606.27095v1 公告类型：交叉 摘要：冷启动无范例类增量学习要求在不依赖回放、外部预训练或大型初始任务的情况下学习不断增长的类别集合。现有的冷启动方法通常要么在整个数据流中训练骨干网络并补偿语义漂移，要么在第一个任务后冻结骨干网络，从而产生偏向初始类别的特征。这些选择也造成了计算上的矛盾：漂移补偿方法需要重复训练骨干网络，且随着任务视野增长，更新成本越来越高；而冻结骨干网络的方法虽然成本低，但在冷启动条件下表现较弱。我们研究了第三种选择：一种从未适应图像数据的特征提取器。我们提出CIRCLE，一种基于固定双向二维储层特征（从BiRC2D改编用于图像分类）和流式线性判别分析头的类增量分类器。CIRCLE将多个随机重采样特征分组，从而在无需训练的情况下实现高效分类。

    arXiv:2606.27095v1 Announce Type: cross  Abstract: Cold-start exemplar-free class-incremental learning requires learning a growing set of classes without replay, external pretraining, or a large initial task. Existing cold-start methods typically either train the backbone throughout the stream and compensate for semantic drift, or freeze a backbone after the first task, producing features biased toward the initial classes. These choices also create a computational tension: drift-compensation methods require repeated backbone training and increasingly expensive updates as the task horizon grows, while frozen-backbone methods are cheap but weak under cold start. We study a third option: a feature extractor that is never fit to image data at all. We propose CIRCLE, a class-incremental classifier built from fixed bidirectional two-dimensional reservoir features, adapted from BiRC2D for image classification, and streaming linear discriminant analysis heads. CIRCLE groups multiple random res
    
[^40]: 超越全局分歧：贝叶斯推理中的局部质量视角

    Beyond Global Divergences: A Local-Mass Perspective on Bayesian Inference

    [https://arxiv.org/abs/2606.27090](https://arxiv.org/abs/2606.27090)

    本文通过引入质量指数和正则化扩展KL散度，从局部质量视角揭示了贝叶斯推理中全局目标函数（如KL散度）未直接捕获的局部行为，并证明了比较局部质量的不等式。

    

    摘要：arXiv:2606.27090v1 公告类型：交叉 摘要：全局目标函数，如KL散度和ELBO，在贝叶斯推理中被广泛用于度量分布差异。本文研究这些目标函数未能直接捕捉的局部质量行为。我们引入并使用了两种数学工具：（1）质量指数，用于记录局部质量的多项式和对数衰减尺度；（2）正则化扩展KL（RE-KL），一种在存在奇异成分时可公式化的局部化散度。质量指数有助于刻画贝叶斯更新如何改变局部质量：（1）幂对数似然因子显式地改变它；（2）参数依赖的支持域或其平滑软化，可能通过参数值附近剩余的质量量来改变局部尺度。利用局部RE-KL，我们证明了在两种KL方向下比较局部小球质量的绝对、相对和方向性不等式。这些结果共同为局部质量行为提供了理论依据。

    arXiv:2606.27090v1 Announce Type: cross  Abstract: Global objectives, such as KL divergence and ELBO, are widely used in Bayesian inference for measuring distributional discrepancy. This paper studies their local-mass behaviour that is not directly captured by such objectives. We introduce and use two mathematical tools: (1) Mass Index for recording the polynomial and logarithmic decay scales of local mass, and (2) regularised extended KL (RE-KL), a set-localised divergence that can be formulated in the presence of singular components. Mass Indices help characterise how Bayesian updating changes local mass: (1) power-log likelihood factors shift it explicitly, and (2) parameter-dependent supports, or their smooth softenings, may change the local scale through the amount of mass that remains near the parameter value. Using local RE-KL, we prove absolute, relative, and directional inequalities for comparing local small-ball masses under the two KL directions. Together, these results prov
    
[^41]: 通过比较寻找驻点

    Finding Stationary Points by Comparisons

    [https://arxiv.org/abs/2606.27082](https://arxiv.org/abs/2606.27082)

    本文提出了在比较预言机下寻找非凸函数驻点的经典和量子算法，分别达到$\widetilde O(n^2/\epsilon^{1.5})$和$\widetilde O(n/\epsilon^{1.5})$的查询复杂度。

    

    我们研究了在仅通过比较预言机（给定两个点，输出哪个函数值更大）访问目标函数时，寻找非凸函数驻点的问题。对于一个二次可微且具有Lipschitz梯度和Hessian矩阵的函数$f\colon\mathbb R^n\to\mathbb R$，我们开发了一种算法，使用$\widetilde O(n^2/\epsilon^{1.5})$次查询即可访问到一个$\epsilon$-驻点。我们的方法使用一个子程序，通过$\widetilde O(n^2\log(1/\delta))$次查询以$\delta$精度估计归一化的Hessian矩阵。我们还进一步研究了量子比较预言机模型下的这个问题，其中查询可以以叠加态进行，并开发了首个量子算法，该算法使用$\widetilde O(n/\epsilon^{1.5})$次查询即可找到一个$\epsilon$-驻点。

    arXiv:2606.27082v1 Announce Type: new  Abstract: We study the problem of finding stationary points of non-convex functions when access to the objective is provided only through a comparison oracle that, given two points, outputs which has the larger function value. For a twice differentiable $f\colon\mathbb R^n\to\mathbb R$ with Lipschitz gradient and Hessian, we develop an algorithm that visits an $\epsilon$-stationary point using $\widetilde O(n^2/\epsilon^{1.5})$ queries. Our approach uses a subroutine that estimates the normalized Hessian to accuracy $\delta$ using $\widetilde O(n^2\log(1/\delta))$ queries. We further study this problem with a quantum comparison oracle model where queries can be made in superpositions, and develop the first quantum algorithm that finds an $\epsilon$-stationary point, which takes $\widetilde O(n/\epsilon^{1.5})$ queries.
    
[^42]: 参数化开源游戏

    Parametric Open Source Games

    [https://arxiv.org/abs/2606.27068](https://arxiv.org/abs/2606.27068)

    本文提出参数化开源游戏框架，通过连续参数空间替代离散程序，揭示了博弈中合作涌现的耦合阈值及神经语义下的合作条件。

    

    开源博弈理论研究的是智能体行为可能相互依赖决策程序的情形，但现有模型大多采用离散或符号化程序。我们引入了参数化开源游戏，这是程序均衡的连续类比，其中玩家选择参数向量，语义映射将完整参数配置转化为底层有限博弈中的混合行动。我们建立了均衡存在性结果，推导出对称2×2博弈中自私梯度上升从背叛转向合作的确切耦合阈值，并给出了参数化程序纳什均衡的一维边界检验。进一步将框架扩展到神经语义类，其一级合作条件由跨玩家敏感度与自身敏感度之比决定。在经典博弈中，该框架展示了访问内部参数化如何从本质上重塑学习过程。

    arXiv:2606.27068v1 Announce Type: cross  Abstract: Open-source game theory studies agents whose behavior may depend on one another's decision procedures, but most existing models use discrete or symbolic programs. We introduce parametric open-source games, a continuous analogue of program equilibria in which players choose parameter vectors and semantics maps convert the full parameter profile into mixed actions in an underlying finite game. We establish equilibrium existence results, derive an exact coupling threshold at which selfish gradient ascent in symmetric $2\times2$ games switches from defection toward cooperation, and give a one-dimensional boundary test for parametric program Nash equilibria. We further extend the framework to a neural semantics class whose first-order cooperation condition is governed by the ratio of cross-player to self-player sensitivity. Across canonical games, the framework shows how access to internal parameterizations can qualitatively reshape learnin
    
[^43]: 状态表示在深度强化学习中至关重要：应用于能源交易

    State Representation Matters in Deep Reinforcement Learning: Application to Energy Trading

    [https://arxiv.org/abs/2606.27032](https://arxiv.org/abs/2606.27032)

    本研究表明，在深度强化学习用于能源交易时，状态表示的设计（特别是使用相对特征和预测特征）对性能有显著影响，仅用绝对特征效果很差。

    

    arXiv:2606.27032v1 公告类型：交叉 摘要：能源交易决策不仅取决于当前市场价格，还取决于预期的未来市场状况和运营约束。这使得提供给强化学习智能体的状态表示成为一个重要的设计选择。我们在HydroDam（一个抽水蓄能套利环境）中，使用固定的Double DQN智能体对此进行了研究。环境、动作空间、奖励函数、网络和训练协议均保持不变；仅更改市场特征。我们比较了绝对价格/日历特征、将当前价格与近期市场历史进行比较的相对特征、预测特征，以及这三个特征族的所有组合。使用2007—2011年比利时日前价格训练和选择策略，并在两个测试设置上评估：2012—2025年的后期同市场测试集和39个其他ENTSO-E市场区域。仅使用绝对特征在测试集上达到28.8%，在各区域中位数为5.7%。

    arXiv:2606.27032v1 Announce Type: cross  Abstract: Energy trading decisions depend not only on current market prices, but also on expected future market conditions, and operational constraints. This makes the state representation given to a reinforcement learning agent an important design choice. We study this in HydroDam, a pumped-storage arbitrage environment, using a fixed Double DQN agent. The environment, action space, reward function, network, and training protocol are kept fixed; only the market features are changed. We compare absolute price/calendar features, relative features that compare current prices with recent market history, forecast features, and all combinations of these three feature families. Policies are trained and selected using 2007--2011 Belgian day-ahead prices and evaluated on two test settings: a later same-market test set from 2012--2025 and 39 other ENTSO-E market zones. Absolute features only reaches 28.8% on the test set and a median 5.7% across zones. R
    
[^44]: 用于学习广义哈密顿量的辛神经网络

    Symplectic Neural Networks for learning Generalized Hamiltonians

    [https://arxiv.org/abs/2606.27029](https://arxiv.org/abs/2606.27029)

    本文提出利用伴随系统的辛离散化与反向传播灵敏度等价的特性，实现了一种在噪声观测下高效训练哈密顿神经网络的方法，解决了隐式辛积分器计算复杂和反向传播困难的问题。

    

    arXiv:2606.27029v1 公告类型：新 摘要：哈密顿神经网络通过学习系统的哈密顿量将物理先验融入神经模型，从而提升泛化能力和样本效率。从状态变量的噪声观测中识别系统哈密顿量是一项具有挑战性的任务。为使模拟真实反映哈密顿系统的长期行为（尤其是能量守恒），必须使用能够保持系统几何结构的辛积分器。这种保真度是有代价的：隐式辛积分器计算强度更高，且使得通过常微分方程求解器进行反向传播变得复杂。然而，通过利用伴随系统的辛离散化能产生与反向传播相同的灵敏度这一事实，我们获得了一种训练神经网络参数的高效方法。在本工作中，我们探索了在轨迹噪声观测下训练哈密顿神经网络的这种替代方法。

    arXiv:2606.27029v1 Announce Type: new  Abstract: Hamiltonian Neural Networks (HNNs) integrate physical priors into neural models by learning a system's Hamiltonian, improving generalization and sample efficiency. Identifying the system Hamiltonian from noisy observations of state variables is a challenging task. For simulations to faithfully reflect the long-term behavior of Hamiltonian systems, especially energy conservation, it is essential to use symplectic integrators, which preserve the system's geometric structure. This fidelity comes at a cost: implicit symplectic integrators are more computationally intensive and make backpropagation through the ODE solver non-trivial. However, by leveraging the fact that symplectic discretizations of the adjoint system yield the same sensitivities associated by backpropagation, we obtain an efficient method of training the Neural Network parameters. In our work, we explore this alternate method of HNN training under noisy observation of trajec
    
[^45]: 你到底有多确定？提升医学视觉问答中语言化置信度校准

    Just how sure are you? Improving Verbalized Uncertainty Calibration in Medical VQA

    [https://arxiv.org/abs/2606.27023](https://arxiv.org/abs/2606.27023)

    本文针对医学视觉问答中多模态大语言模型过度自信的问题，提出了一种基于训练的框架，通过复合损失函数和因子扰动设计来改善置信度校准。

    

    应用于医学视觉问答的多模态大语言模型往往会产生过度自信的输出，而不管实际正确性如何。现有的语言化置信度校准方法主要针对纯文本大语言模型开发，未能考虑医学图像理解的多模态特性。本文提出一个基于训练的框架，通过微调多模态大语言模型来改善其校准性能。该框架采用复合损失函数，结合了布里尔风格的校准项、防止置信度向极端值崩溃的锚点正则化器、对比图像-文本对齐项以及基于KL散度的模型稳定项。对齐信号源自一个$2 \times 2$因子扰动设计，该设计交叉处理图像存在性与文本完整性，以探测模型对视觉模态输入与语言先验的依赖程度。最后，使用一个前K项KL散度正则化器来保护答案。

    arXiv:2606.27023v1 Announce Type: cross  Abstract: Multimodal large language models (MLLMs) applied to Medical Visual Question Answering (VQA) tend to produce overconfident outputs regardless of actual correctness, and existing verbalized confidence calibration methods, developed primarily for text only LLMs, do not account for the multimodal nature of medical image understanding.   This work proposes a training based framework that finetunes MLLMs to improve their calibration using a composite loss function combining a Brier style calibration term, an anchor regularizer that prevents confidence collapse toward extreme values, a contrastive image text alignment term, and a KL based model stabilization term. The alignment signal is derived from a $2 \times 2$ factorial perturbation design that crosses image presence with text integrity, probing the reliance of the model on visual modality input versus language priors. Finally, a top K KL divergence regularizer is used to protect the ans
    
[^46]: 基于JEPA的世界模型泛化理论

    A Generalization Theory for JEPA-Based World Models

    [https://arxiv.org/abs/2606.27014](https://arxiv.org/abs/2606.27014)

    本文首次为基于JEPA的世界模型建立了泛化理论，揭示了其预训练误差与下游规划性能之间的量化关系，并发现了潜在维度上近似误差与样本误差的权衡。

    

    arXiv:2606.27014v1 公告类型：新论文 摘要：联合嵌入预测架构（JEPAs）近期作为一种有前景的世界建模范式出现，它通过在潜在空间中学习预测动力学，而非在输入层面生成未来观测。尽管在经验上取得了成功，但对基于JEPA的世界模型的理论理解仍然有限。在本文中，我们首次为基于JEPA的世界模型建立了泛化理论。我们将JEPA预训练形式化为一个条件谱图学习问题，并证明JEPA目标等价于一个动作条件共现矩阵的低秩分解。基于这一特征，我们建立了JEPA预训练误差与下游规划遗憾之间的关联，从而得出了基于JEPA的世界模型的有限样本泛化界限。我们的分析揭示了在潜在维度方面，近似误差与样本误差之间存在固有的权衡。

    arXiv:2606.27014v1 Announce Type: new  Abstract: Joint Embedding Predictive Architectures (JEPAs) have recently emerged as a promising paradigm for world modeling by learning predictive dynamics in a latent space rather than generating future observations at the input level. Despite their empirical success, the theoretical understanding of JEPA-based world models remains limited. In this paper, we develop the first generalization theory for JEPA-based world models. We formulate JEPA pretraining as a conditional spectral graph learning problem and show that the JEPA objective is equivalent to a low-rank factorization of an action-conditioned co-occurrence matrix. Building on this characterization, we establish a connection between JEPA pretraining error and downstream planning regret, leading to a finite-sample generalization bound for JEPA-based world models. Our analysis reveals an inherent trade-off between approximation and sample errors with respect to the latent dimension, providi
    
[^47]: 迭代式LLM智能体循环的语义早期停止

    Semantic Early-Stopping for Iterative LLM Agent Loops

    [https://arxiv.org/abs/2606.27009](https://arxiv.org/abs/2606.27009)

    提出了一种基于语义相似度和答案质量评估的早期停止方法，替代固定迭代上限，以优化多智能体LLM循环中的token使用效率。

    

    多智能体大语言模型（LLM）循环，例如一个起草的“写手”和一个修订的“评论家”，几乎总是通过固定的迭代上限（max_iterations）来终止。这是一种句法层面的终止开关：它无法判断答案是否仍在改进，因此在简单输入上浪费了过多的token，而在困难输入上过早截断。我们研究了语义早期停止：当连续草稿的嵌入向量在语义上不再变化（余弦距离，配合一个耐心窗口）且答案的测量质量不再提升时，循环停止。我们的工作做出了三项贡献。第一，一个严谨的理论基础：我们证明了确定性终止和良好定义性，并对这些主张进行了机器验证，同时将距离序列的收敛性视为一个经验验证的猜想，而非（此前被过度宣称的）巴拿赫压缩。第二，一个高效评估协议：我们为每个问题生成完整的轨迹一次，然后重放所有内容。

    arXiv:2606.27009v1 Announce Type: new  Abstract: Multi-agent large language model (LLM) loops, for example a Writer that drafts and a Critic that revises, are almost always terminated by a fixed iteration cap (max_iterations). This is a syntactic kill-switch: it is blind to whether the answer is still improving, so it over-spends tokens on easy inputs and truncates hard ones. We study semantic early-stopping: the loop halts when consecutive draft embeddings stop changing in meaning (cosine distance with a patience window) and the answer's measured quality stops improving. Our work makes three contributions. First, an honest theoretical footing: we prove deterministic termination and well-definedness and machine-check these claims, while treating the convergence of the distance sequence as an empirically tested conjecture rather than a (previously over-claimed) Banach contraction. Second, a judge-efficient evaluation protocol: we generate each question's full trajectory once, replay eve
    
[^48]: 数据同化中基于保形预测的不确定性量化

    Uncertainty quantification via conformal prediction in data assimilation

    [https://arxiv.org/abs/2606.27001](https://arxiv.org/abs/2606.27001)

    本研究首次将保形预测方法应用于数据同化领域，通过一维修正浅水模型验证了其比传统集合方法更有效量化不确定性的能力。

    

    摘要：量化不确定性的演变对于数值天气预报中的概率预测和数据同化至关重要。在本研究中，我们探讨了保形预测（CP）这一最新的机器学习方法在受控理想化环境中量化不确定性的适用性。我们使用了为模拟对流过程而设计的一维修正浅水模型。CP提供了一组具有选定置信水平的可能结果。在这里，我们比较并评估了三种CP变体（即a）标准CP、b）归一化CP和c）保形分位数回归）的平均经验覆盖率、平均区间长度、下侧缺失率、上侧缺失率和平均区间评分损失（AISL）。我们进一步将这些基于CP的不确定性估计与传统的基于集合的度量（如标准差区间和集合扩散）进行了比较。此外，我们还研究了CP的集成方法。

    arXiv:2606.27001v1 Announce Type: new  Abstract: Quantifying the evolution of uncertainty is critical to both probabilistic forecasting and data assimilation in numerical weather prediction. In this study, we investigate the applicability of conformal prediction (CP), a recent machine learning (ML) method, to quantify uncertainty in a controlled, idealized setting. We use the one dimensional modified shallow water model, designed to mimic the convective process. CP provides a set of possible outcomes with a chosen confidence level. Here, we compare and evaluate the average empirical coverage, the average interval length, miss low, miss high and average interval score loss (AISL) for three variants of CP, namely a) Standard CP, b) Normalized CP and c) Conformalized Quantile Regression. We further compare these CP-based uncertainty estimates with traditional ensemble-based measures such as standard deviation intervals and ensemble spread. In addition, we investigate the integration of CP
    
[^49]: RolloutPipe：在分离式在线策略大语言模型强化学习中重叠流水线式推出与训练

    RolloutPipe: Overlapping Pipelined Rollout and Training in Disaggregated On-Policy LLM Reinforcement Learning

    [https://arxiv.org/abs/2606.26997](https://arxiv.org/abs/2606.26997)

    该论文提出了RolloutPipe框架，通过将固定权重的推出过程转化为完整的组流水线，实现了推出与训练在分离式RLVR系统中的高效重叠，解决了同步系统的GPU空闲问题和异步系统的数据过时问题。

    

    arXiv:2606.26997v1 公告类型：交叉 摘要：大语言模型（LLM）在推理方面的后训练越来越依赖于可验证奖励的强化学习（RLVR），模型通过数学、逻辑和科学任务中的真实反馈进行学习。为了实现灵活的资源分配并支持异构训练设置，现代RLVR系统采用分离式架构，将推出生成与策略训练解耦到独立的GPU池中。然而，现有的同步在线策略GRPO（组相对策略优化）RLVR系统在开始训练前完成整个推出过程，导致推出仍在进行时训练器GPU池处于空闲状态。异步RL管道重叠了这两个阶段，但代价是使用过时数据进行训练。为应对这些挑战，我们提出了RolloutPipe，一种用于分离式RLVR系统的后训练框架，它将固定权重的推出转变为完整的组流水线，其中可训练组...

    arXiv:2606.26997v1 Announce Type: cross  Abstract: Large language model (LLM) post-training for reasoning increasingly relies on reinforcement learning with verifiable rewards (RLVR), where models learn from ground-truth feedback on mathematical, logical, and scientific tasks. To enable flexible resource allocation and support heterogeneous training setups, modern RLVR systems adopt disaggregated architectures that decouple rollout generation and policy training across independent GPU pools. However, existing synchronous on-policy GRPO (Group Relative Policy Optimization) RLVR systems finish an entire rollout before starting training, leaving the trainer GPU pool idle while rollout is still ongoing. Asynchronous RL pipelines overlap the two stages, but at the cost of training on stale data. To address these challenges, we propose RolloutPipe, a post-training framework for disaggregated RLVR systems, which turns the fixed-weight rollout into a complete-group pipeline where trainable gro
    
[^50]: 实现基于Noise2Inverse的自监督学习原偶方法

    Enabling self-supervised learned primal dual with Noise2Inverse

    [https://arxiv.org/abs/2606.26991](https://arxiv.org/abs/2606.26991)

    提出一种自监督方法，通过将Noise2Inverse框架与学习原偶算法结合，利用CT扫描不同角度测量噪声的统计独立性，实现了无需真实图像即可训练迭代重建算子。

    

    X射线计算机断层扫描重建是一个不适定的逆问题，尤其是在低剂量和稀疏角度设置下，测量数据存在噪声且不完整。尽管诸如学习原偶算法等学习重建方法表现出色，但它们通常依赖于有监督训练，需要访问真实数据，而这在实际中往往无法获得。在本工作中，我们通过将Noise2Inverse框架扩展到学习原偶算法，提出了一种自监督重建方法。所提出的方法称为Noise2Inverse学习原偶算法，它通过利用CT扫描中不同角度测量数据中噪声的统计独立性，无需真实图像即可训练学习迭代重建算子。我们将所提出的方法与经典重建方法以及基于神经网络的方法进行了比较。

    arXiv:2606.26991v1 Announce Type: cross  Abstract: X-ray computed tomography reconstruction is an ill-posed inverse problem, particularly in low-dose and sparse-angle settings where measurements are noisy and incomplete. While learned reconstruction methods such as the Learned Primal-Dual algorithm achieve strong performance, they typically rely on supervised training with access to ground-truth data, which is often unavailable in practice.   In this work, we propose a self-supervised reconstruction method by extending the Noise2Inverse framework to the Learned Primal-Dual algorithm. The resulting approach, called Noise2Inverse Learned Primal-Dual (N2I-LPD), enables training of a learned iterative reconstruction operator without ground-truth images by exploiting the statistical independence of noise in distinct measurements with respect to angular rotation of the CT-scan.   We compare the proposed method with classical reconstruction methods, as well as neural network-based approaches 
    
[^51]: 不确定性量化的决策对齐评估

    Decision-Aligned Evaluation of Uncertainty Quantification

    [https://arxiv.org/abs/2606.26990](https://arxiv.org/abs/2606.26990)

    本文提出决策对齐标准，发现传统不确定性量化指标常与下游决策效用不一致，并设计先验加权效用指标以实现与决策效用的对齐，从而修正了现有评估协议的缺陷。

    

    arXiv:2606.26990v1 公告类型：交叉 摘要：机器学习中的不确定性估计通常使用通用指标（如负对数似然和期望校准误差）进行评估，然而在这些指标上表现良好并不一定意味着在下游决策中具有高实用性。我们引入了“决策对齐”这一标准，它揭示了哪些评估指标能够有意义地与下游效用对齐。应用这一框架，我们表明许多广泛使用的不确定性指标要么与常见决策问题不一致，要么编码了关于下游任务的病态先验信念。然后，我们提出了先验加权效用指标，这是一类特殊的适当评分规则，能够提供决策对齐的不确定性评估。在基准实验和实际案例研究中，我们的指标始终与实现的决策效用保持一致，而传统指标则不然。我们的结果揭示了当前不确定性量化评估协议中的缺陷，并提供了一种新的评估范式。

    arXiv:2606.26990v1 Announce Type: cross  Abstract: Uncertainty estimates in machine learning are typically evaluated using generic metrics such as the negative log-likelihood and expected calibration error, yet good performance on such metrics does not necessarily imply high utility in downstream decisions. We introduce decision-alignment, a criterion that reveals which evaluation metrics meaningfully align with downstream utilities. Applying this framework, we show that many widely used uncertainty metrics are either misaligned with common decision problems or encode pathological prior beliefs about the downstream task. We then propose prior-weighted utility metrics, a special class of proper scoring rules that provides decision-aligned uncertainty evaluation. Across benchmark experiments and real-world case studies, our metrics consistently align with realized decision utility, while conventional metrics do not. Our results surface flaws in the current UQ evaluation protocol and offe
    
[^52]: 面向超额均方误差的自适应经验贝叶斯估计

    XMSE-Aware Adaptive Empirical Bayes Estimation

    [https://arxiv.org/abs/2606.26975](https://arxiv.org/abs/2606.26975)

    本文通过将超额均方误差（XMSE）分析从诊断工具转化为设计原则，提出了一种在最大似然和经验贝叶斯之间自适应插值的混合估计器，并证明了其在二阶意义下不劣于两者。

    

    经验贝叶斯（EB）估计器能够在一阶渐近风险上与最大似然（ML）估计相匹配，但在二阶行为上存在显著差异：最新的超额均方误差（XMSE）分析表明，当核函数与真实参数对齐不佳时，基于核的经验贝叶斯估计可能比最大似然估计更差。本文将这一诊断转化为设计原则。我们提出了一种面向XMSE的混合估计器，它在ML估计和EB收缩之间进行插值。其固定权重的XMSE是一个标量二次形式，从而得到一个闭式的理想混合权重，该权重在XMSE尺度上不劣于ML估计和基础EB估计。基于有限样本XMSE近似的插件实现被证明是一致的，并且对于内部理想权重具有二阶遗憾率。我们进一步将遗憾界迁移到所选权重下的固定权重风险曲线、一个阈值边界规则以及相关扩展。

    arXiv:2606.26975v1 Announce Type: cross  Abstract: Empirical Bayes (EB) estimators can match the first-order asymptotic risk of maximum likelihood (ML) while behaving very differently at second order: recent excess mean squared error (XMSE) analysis shows that kernel-based EB estimation may be worse than ML when the kernel is poorly aligned with the true parameter. This paper turns that diagnostic into a design principle. We propose an XMSE-aware mixed estimator that interpolates between ML and EB shrinkage. Its fixed-weight XMSE is a scalar quadratic, yielding a closed-form oracle mixing weight that is no worse than both ML and the base EB estimator at the XMSE scale. A plug-in implementation based on finite-sample XMSE approximations is proved consistent, with a second-order oracle regret rate for an interior oracle weight. We further establish a transfer of the regret bound to the fixed-weight risk curve evaluated at the selected weight, a thresholded boundary rule, and extensions t
    
[^53]: 面向安全开放集半监督学习的几何梯度修正

    Geometric Gradient Rectification for Safe Open-Set Semi-Supervised Learning

    [https://arxiv.org/abs/2606.26973](https://arxiv.org/abs/2606.26973)

    本文提出几何梯度修正（GGR）框架，通过将冲突的辅助梯度投影到以监督梯度为锚点的可接受区域，在梯度层面控制更新方向，从而避免样本选择与伪标签错误带来的性能权衡，实现安全的开放集半监督学习。

    

    arXiv:2606.26973v1 公告类型：交叉 摘要：开放集半监督学习旨在利用可能包含分布外异常值的未标记数据，同时保持对分布内类别的性能。现有方法主要遵循两种范式：过滤可疑样本或结合带有软权重的未标记目标。我们认为两者都面临一个共同的权衡：激进的过滤可能会丢弃信息丰富但难以处理的分布内样本，而利用未标记数据则可能在伪标签错误时引入与监督学习冲突的辅助梯度。因此，我们将关注点从样本选择转向梯度级控制。我们提出了《几何梯度修正》（GGR），这是一个即插即用的框架，它利用监督梯度作为锚点，并将冲突的辅助梯度投影到梯度空间中的一个可接受区域内。这使得应用的辅助更新在修正后的坐标块内保持一阶非对立性，同时保留其有益信息。

    arXiv:2606.26973v1 Announce Type: cross  Abstract: Open-set semi-supervised learning aims to leverage unlabeled data that may contain out-of-distribution outliers while maintaining performance on in-distribution classes. Existing methods mainly follow two paradigms: filtering suspicious samples or incorporating unlabeled objectives with soft weighting. We argue that both face a common trade-off: aggressive filtering can discard informative but hard ID samples, whereas utilization can introduce auxiliary gradients that conflict with supervised learning when pseudo labels are wrong. We therefore shift the focus from sample selection to gradient-level control. We propose \textit{Geometric Gradient Rectification} (GGR), a plug-in framework that uses the supervised gradient as an anchor and projects conflicting auxiliary gradients onto an admissible region in gradient space. This makes the applied auxiliary update first-order non-opposing within the rectified coordinate block while preservi
    
[^54]: 针对普通用户的越狱攻击：通过多臂老虎机算法选择最优越狱策略以自动增强查询

    Jailbreaking for the Average Jane: Choosing Optimal Jailbreaks via Bandit Algorithms for Automatically Enhanced Queries

    [https://arxiv.org/abs/2606.26936](https://arxiv.org/abs/2606.26936)

    本文提出一种基于多臂老虎机算法的越狱攻击策略，让非专业用户也能高效选择最优越狱方法，并构建了包含大量恶意查询的基准测试FrankensteinBench，验证了非专业恶意行为者成功攻击LLMs的可行性。

    

    arXiv:2606.26936v1 公告类型：交叉 摘要：随着大量针对大语言模型的越狱方法被广泛知晓，一个日益增长的担忧是，非专业恶意行为者（即“普通用户”）可能能够对恶意请求获得可操作的回答。在这项工作中，我们检验了这一担忧是否合理。非专业恶意行为者成功攻击需要两个要素：针对目标模型的强大越狱策略，以及有效的恶意查询。对于前者，我们提出了一种基于多臂老虎机框架的新型攻击策略。这使得通过在小规模查询上进行有噪声的探索，从大量候选集中高效在线学习最优越狱策略成为可能，随后将学到的策略应用于开发集。对于后者，我们构建了FrankensteinBench，一个包含11,279个恶意查询的安全基准测试，这些查询来自对7个现有基准测试的人工整理，并结合了自动增强和生成。每个查询都进行了分类。

    arXiv:2606.26936v1 Announce Type: cross  Abstract: With a profusion of jailbreaks for LLMs now widely known, a growing concern is that non-expert malicious actors ("the average Jane") could elicit actionable responses to malicious requests. In this work, we examine whether this concern is justified. A non-expert malicious actor requires two ingredients for a successful attack: a powerful jailbreak for their target model, acting on an effective malicious query. For the former, we propose a novel attack strategy based on the multi-armed bandit framework. This allows efficient online learning of the optimal jailbreak from a large choice set via noisy exploration on a small number of queries, with subsequent application of the learnt policy on an exploitation set. For the latter, we curate $\mathrm{FrankensteinBench}$, a safety benchmark of $11,279$ malicious queries drawn from manual curation over $7$ existing benchmarks, along with automated enhancement and generation. Each query is cate
    
[^55]: GEOALIGN：面向鲁棒大语言模型强化学习的几何轨迹筛选

    GEOALIGN: Geometric Rollout Curation for Robust LLM Reinforcement Learning

    [https://arxiv.org/abs/2606.26917](https://arxiv.org/abs/2606.26917)

    GEOALIGN通过检测和修正批次中与多数方向不一致的高奖励轨迹，有效抑制了噪声奖励下的训练不稳定性，是一种轻量级的在线强化学习轨迹筛选方法。

    

    在线强化学习被广泛用于将大语言模型与奖励信号对齐，但在奖励信号存在噪声或错误设定时，训练可能不稳定。我们识别出一种称为“方向不一致性”的失败模式：在批次内，少量高奖励轨迹会产生与批次多数意见严重不一致的表征空间偏好方向，从而导致高方差和不稳定的更新。我们提出GEOALIGN，一种用于迭代策略优化中轨迹筛选的轻量级插件。GEOALIGN (i) 形成提示内偏好对，(ii) 学习一个在线投影器，基于每条轨迹的隐藏状态来集中奖励排序的位移方向，(iii) 通过轨迹与批次共识原型的角度偏差检测方向不一致的轨迹，并用提示内稳定替代方案进行修正。GEOALIGN仅需前向传播，且增加的开销极小。在对话任务中...

    arXiv:2606.26917v1 Announce Type: cross  Abstract: Online reinforcement learning is widely used to align large language models (LLMs) with reward signals, yet training can be unstable under noisy or misspecified rewards. We identify a failure mode we call directional inconsistency: within a batch, a small set of high-reward rollouts induces representation-space preference directions that sharply disagree with the batch majority, resulting in high-variance and destabilizing updates. We propose geoalign, a lightweight plug-in for rollout curation in iterative policy optimization. Geoalign (i) forms within-prompt preference pairs, (ii) learns an online projector on per-rollout hidden states to concentrate reward-ordered displacement directions, and (iii) detects directionally inconsistent rollouts via their angular deviation from a batch consensus prototype and rectifies them with within-prompt stable alternatives. Geoalign is forward-pass only and adds negligible overhead. Across dialogu
    
[^56]: 基于纤维束成像的示踪组织学纤维束分割合成数据生成

    Tractography-Driven Synthetic Data Generation for Fiber Bundle Segmentation in Tracer Histology

    [https://arxiv.org/abs/2606.26898](https://arxiv.org/abs/2606.26898)

    提出利用dMRI纤维束成像作为生成先验，合成逼真二维图像块来训练分割网络，实现示踪组织学中纤维束的自动分割，并提升跨大脑的泛化能力。

    

    arXiv:2606.26898v1 公告类型：交叉 摘要：扩散磁共振成像（dMRI）纤维束成像能够无创重建白质通路，但其准确性从根本上受限于对轴突组织的间接、低分辨率测量。非人灵长类动物的示踪注射研究为验证dMRI纤维束成像提供了金标准。然而，这需要对组织切片中的纤维束进行耗时的手动标注。我们提出了一种基于合成数据增强的框架，用于猕猴示踪组织学中的自动纤维束分割。我们的方法利用离体dMRI纤维束成像作为生成先验，合成用于训练的二维图像块。这为我们提供了足够逼真的前景纹理，我们将其与块面照片的背景组合，并通过域随机化进行多样化。在混合真实和合成块上训练二维U-Net。在保留大脑上的实验表明，该方法在大脑和纤维束间具有更好的泛化能力。

    arXiv:2606.26898v1 Announce Type: cross  Abstract: Diffusion MRI (dMRI) tractography enables non-invasive reconstruction of white-matter pathways, but its accuracy is fundamentally limited by indirect, low-resolution measurements of axonal organization. Tracer injection studies in non-human primates provide a gold standard for validating dMRI tractography. This, however, requires time-consuming manual annotation of fiber bundles in histology sections. We propose a synthetic-data augmented framework for automated fiber bundle segmentation in macaque tracer histology. Our approach uses ex vivo dMRI tractography as a generative prior to synthesize 2D image patches for training. This provides us with sufficiently realistic foreground texture, which we compose with backgrounds from blockface photos and diversify via domain randomization. A 2D U-Net is trained on mixed real and synthetic patches. Experiments on held-out brains demonstrate improved generalization across brains and fiber bundl
    
[^57]: 参数化先知不等式问题的渐近最优学习

    Asymptotically Optimal Learning for Parametric Prophet Inequalities

    [https://arxiv.org/abs/2606.26893](https://arxiv.org/abs/2606.26893)

    针对参数未知的指数型参数族先知不等式问题，提出了一种仅靠在线观测即可达到最优渐近竞争比的置信度动态规划策略，无需离线样本。

    

    我们研究了先知不等式中的学习问题，其中收益独立同分布，来自一个参数未知的指数型参数族，该族包括指数分布、帕累托分布和有界支撑幂族分布。我们首先刻画了该族的最优全信息渐近竞争比。在无界支撑情形下，该极限值为 ${\left({\theta}/({\theta-c_+})\right)^{c_+/\theta}}/ {\Gamma(1-c_+/\theta)}$；而在有界支撑情形下，极限值为 $1$。随后，我们提出了一种基于置信度的动态规划在线学习策略。通过利用显式的参数结构，该策略仅使用在线观测数据即可达到相同的最优渐近竞争比，无需外部离线样本。我们还针对典型例子推导了分布特定的收敛速率。最后，在合成实例上的数值实验展示了我们算法的性能。

    arXiv:2606.26893v1 Announce Type: new  Abstract: We study learning in prophet inequalities with i.i.d. rewards drawn from an exponential-type parametric family with an unknown parameter $\theta$, a class that includes exponential, Pareto, and bounded-support power-family distributions. We first characterize the optimal full-information asymptotic competitive ratio for this family. In the unbounded-support case, the limit is $ {\left({\theta}/({\theta-c_+})\right)^{c_+/\theta}}/ {\Gamma(1-c_+/\theta)},$ while in the bounded-support case, the limit is $1$. We then propose a confidence-based dynamic-programming policy for online learning. By exploiting the explicit parametric structure, the policy achieves the same optimal asymptotic competitive ratio using only online observations, without external offline samples. We further derive distribution-specific convergence rates for canonical examples. Finally, numerical experiments on synthetic instances illustrate the performance of our algor
    
[^58]: 使用SamAdams变步长和位置自适应朗之万动力学的加速采样方法

    Accelerated sampling using SamAdams variable timesteps and position-adaptive Langevin dynamics

    [https://arxiv.org/abs/2606.26881](https://arxiv.org/abs/2606.26881)

    本文提出了一种结合自适应步长和方向性摩擦的加速朗之万采样方法，在保持采样精度的同时显著提高计算效率。

    

    我们提出了一种基于两种互补机制的加速朗之万采样方法：\emph{SamAdams}自适应步长方法，通过松弛刚度监测器自动缩小相空间刚性区域的有效积分步长；以及\emph{位置自适应朗之万}（PAL）动力学，该方法将摩擦集中在局部力方向，同时保持正则分布作为精确不变测度。由此产生的组合方案（SA-PAL）在一个回文积分器中实现，该积分器通过适当组织积分步骤并利用PAL摩擦张量的秩一加标量结构，每次迭代仅需一次力评估。我们在多种模型问题上测试了该方法：Rosenbrock函数、薄熵通道、Mueller-Brown势能以及一个具有稀疏诱导收缩先验的贝叶斯参数化问题。

    arXiv:2606.26881v1 Announce Type: cross  Abstract: We introduce an accelerated Langevin-based sampling method that is based on two complementary devices: \emph{SamAdams} adaptive timestepping, which automatically shrinks the effective integration step in stiff regions of phase space using a relaxed stiffness monitor, and \emph{position-adaptive Langevin} (PAL) dynamics, which concentrates friction along the local force direction while preserving the canonical distribution as the exact invariant measure. The resulting combined scheme (SA-PAL) is implemented in a palindromic integrator which requires only one force evaluation per iteration through suitable organisation of the integration steps and by exploiting the rank-one-plus-scalar structure of the PAL friction tensor. We test the method on various model problems: the Rosenbrock function, a thin entropic channel, the Mueller-Brown potential, and a Bayesian parameterisation problem with a sparsity-inducing shrinkage prior. On the Rose
    
[^59]: 自然理解过程中语言模型的异质性神经预测性

    Heterogeneous Neural Predictivity from Language Models During Naturalistic Comprehension

    [https://arxiv.org/abs/2606.26880](https://arxiv.org/abs/2606.26880)

    该论文证明语言模型表征在自然语言理解过程中可有效预测神经活动，并通过多种控制验证了这种预测的稳健性和敏感性。

    

    语言模型表征能够对自然语言刺激提供结构化、高维度的注释，并在理解过程中作为信息丰富的神经预测因子。我们分析了来自Brain Treebank、MEG-MASC和Podcast ECoG的锁定派生数据，使用了八个冻结的语言模型、阻塞编码模型以及匹配的时间、干扰项和表征容量控制。在源级摘要中，正向的保留预测以及相对于低级基线的提升普遍存在。在Brain Treebank和Podcast ECoG中，432个可评估行中有67行满足受控的预测性唯一标准，模型侧特征消融在大多数可评估源行中改变了预测分数。基于大脑、时间关联、声学和植入信号的控制证实了分析流程在组件层面的敏感性。这些发现表明，语言模型衍生的量值能够注释自然语音和文本处理过程中的神经活动。

    arXiv:2606.26880v1 Announce Type: new  Abstract: Language-model representations provide structured, high-dimensional annotations of naturalistic language stimuli and can serve as informative neural predictors during comprehension. We analyzed locked derived data from Brain Treebank, MEG-MASC, and Podcast ECoG with eight frozen language models, blocked encoding models, and matched temporal, nuisance, and representation-capacity controls. Positive held-out prediction and gains over low-level baselines were widespread in source-level summaries. Across Brain Treebank and Podcast ECoG, 67 of 432 evaluable rows met a controlled predictive-only criterion, and model-side feature ablations changed prediction scores in most evaluable source rows. Brain-derived, timing-linked, acoustic, and implanted-signal controls confirmed component-level sensitivity of the analysis pipeline. These findings show that language-model-derived quantities can annotate neural activity during natural speech and text 
    
[^60]: 韦斯费勒-莱曼层级中的可扩展消息传递量子图神经网络

    Scalable Message-Passing Quantum Graph Neural Networks in the Weisfeiler-Leman Hierarchy

    [https://arxiv.org/abs/2606.26873](https://arxiv.org/abs/2606.26873)

    本文提出了一种在韦斯费勒-莱曼层级中具有可扩展性的量子图神经网络，通过消息传递和排列等变性，实现了表达能力与可扩展性的双重保证。

    

    摘要：图在化学、生物学和优化领域中为关系数据提供了一种自然语言。图神经网络（GNNs）通过消息传递这一单一原语（它概括了卷积和注意力机制）推动了从这类数据中学习的最新进展。虽然已经提出了量子对应模型，但它们与消息传递的联系有限，并且在性能或可扩展性方面几乎没有保证。更广泛地说，变分量子电路的可训练性是其广泛应用的一个公认瓶颈，而预训练已成为解决这一问题的一种方法。然而，要使量子模型有用，它必须提供表达能力保证以及可证明的可扩展性。在这里，我们展示了如何构建一个量子图神经网络来执行消息传递，使其具有排列等变性，并位于韦斯费勒-莱曼层级的选定级别上——这是衡量模型区分图结构精细程度的标准指标。

    arXiv:2606.26873v1 Announce Type: cross  Abstract: Graphs provide a natural language for relational data in chemistry, biology and optimisation. Graph neural networks (GNNs) have driven much of the recent progress in learning from such data through message passing, a single primitive that generalises convolution and attention. Quantum counterparts have been proposed, but with limited connection to message passing and few guarantees on performance or scalability. More broadly, the trainability of variational quantum circuits is a recognised bottleneck for their wide applicability, and pre-training has emerged as one way to address it. Yet for a quantum model to be useful, it must offer expressivity guarantees along with demonstrable scalability. Here we show how a quantum graph neural network can be built to perform message passing, to be permutation equivariant, and to sit at a chosen level of the Weisfeiler-Leman hierarchy, the standard measure of how finely a model can tell graphs ap
    
[^61]: 联邦学习中的量化：方法、挑战与未来方向

    Quantization in Federated Learning: Methods, Challenges and Future Directions

    [https://arxiv.org/abs/2606.26822](https://arxiv.org/abs/2606.26822)

    本文首次系统性地综述了联邦学习中的量化技术，提出了一个针对联邦学习特有维度（如客户端异构性、非IID鲁棒性等）的新分类法，并分析了量化与联邦学习核心行为的相互作用。

    

    arXiv:2606.26822v1 公告类型：新文 摘要：联邦学习已成为隐私保护分布式智能的基础范式，但其可扩展性仍从根本上受限于通信瓶颈、设备异构性以及在统计非独立同分布数据下训练的挑战。量化是缓解这些限制的最有效机制之一，它能减少上行/下行传输负载以及设备上的计算量。本文提供了首个以联邦学习为中心的系统性量化综述，引入了一种围绕联邦学习特定维度组织的新分类法，包括客户端异构性、聚合一致性、通信调度自适应、非独立同分布鲁棒性、隐私/安全集成以及硬件/能源协同优化。除了对现有方法进行分类整理，我们还分析了量化如何与联邦学习的核心行为相互作用，例如客户端漂移、部分参与、收敛稳定性、安全聚合以及差分隐私。

    arXiv:2606.26822v1 Announce Type: new  Abstract: Federated Learning (FL) has become a foundational paradigm for privacy-preserving distributed intelligence, yet its scalability remains fundamentally constrained by communication bottlenecks, device heterogeneity, and the challenges of training under statistically non-IID data. Quantization is one of the most effective mechanisms for mitigating these limitations, reducing both uplink/downlink payloads and on-device computation. This paper provides the first FL-centric systematic review of quantization, introducing a novel taxonomy organized around FL-specific dimensions, including client heterogeneity, aggregation consistency, communication-scheduling adaptation, non-IID robustness, privacy/security integration, and hardware/energy co-optimization. Beyond cataloging existing methods, we analyze how quantization interacts with core FL behaviors such as client drift, partial participation, convergence stability, secure aggregation, and dif
    
[^62]: 记忆深度，而非记忆访问：面向长时间运行语言代理的选择性参数巩固

    Memory Depth, Not Memory Access: Selective Parametric Consolidation for Long-Running Language Agents

    [https://arxiv.org/abs/2606.26806](https://arxiv.org/abs/2606.26806)

    本文提出了一种通过选择性参数巩固（EVAF机制）来增强长时间运行语言代理在卸载工作上下文后持久保持目标导向行为的能力，而传统检索系统仅能实现浅层事实回忆。

    

    arXiv:2606.26806v1 公告类型：新论文 摘要：长时间运行的语言代理需要的不仅仅是记忆访问。检索系统可以在查询时获取过去的事实，但它们不会决定在卸载工作上下文后，哪些经验应继续塑造行为。我们将这个独立的问题研究为记忆深度：将持久的目标导向倾向写入一个小型参数存储中。我们引入了循环漂移协议，这是一种受控的压力测试，其中检索索引保持不变，而工作上下文被卸载，目标导向行为必须在长循环干扰下持续存在。我们评估了EVAF，一种基于惊喜和效价门控的LoRA巩固机制。在GPT-2和TinyLlama上，检索在浅层事实回忆方面最强（短事实准确率0.956-0.973），而EVAF在目标持久性和卸载后恢复方面最强（0.812-0.904），每200个事件仅需2-3次参数写入。机制控制表明，选择性巩固是分解的。

    arXiv:2606.26806v1 Announce Type: new  Abstract: Long-running language agents need more than memory access. Retrieval systems can fetch past facts at query time, but they do not decide which experiences should continue to shape behavior after the working context is unloaded. We study this separate problem as memory depth: durable goal-conditioned tendencies written into a small parametric store. We introduce the loop-drift protocol, a controlled stress test in which the retrieval index remains intact while working context is unloaded and goal-conditioned behavior must persist under long-loop interference. We evaluate EVAF, a surprise- and valence-gated LoRA consolidation mechanism. Across GPT-2 and TinyLlama, retrieval is strongest on shallow factual recall (short-fact accuracy 0.956--0.973), while EVAF is strongest on goal persistence and post-unload recovery (0.812--0.904) with only 2--3 parametric writes per 200 events. Mechanism controls show that selective consolidation factorizes
    
[^63]: 推理质量早期显现：面向推理模型的数据筛选

    Reasoning Quality Emerges Early: Data Curation for Reasoning Models

    [https://arxiv.org/abs/2606.26797](https://arxiv.org/abs/2606.26797)

    本文提出仅利用推理模型初始令牌的损失值，即可低成本、高效地筛选出高质量且多样化的监督微调数据，从而提升推理能力。

    

    arXiv:2606.26797v1 公告类型：新文 摘要：在一小批高质量的长推理轨迹上进行监督微调（SFT），是激发大型语言模型（LLMs）强大推理能力的有效方法。然而，现有的高质量SFT数据筛选方法严重依赖强大的推理模型，根据多样性和难度来过滤示例，这使得筛选过程成本高昂，且往往导致数据质量欠佳。在本工作中，我们表明，仅利用初始推理令牌即可识别出多样且具有挑战性的推理示例。具体而言，我们证明，基于在预训练模型的随机扰动检查点上评估的前100个推理令牌的损失，可以可靠地检测出困难问题。我们进一步展示，那些在少数几个沿微调轨迹外推的扰动检查点上，其前1000个推理令牌的损失模式相似的示例，能够显著提升推理性能。

    arXiv:2606.26797v1 Announce Type: new  Abstract: Supervised fine-tuning (SFT) on a small, high-quality set of long reasoning traces is an effective approach for eliciting strong reasoning capabilities in Large Language Models (LLMs). However, existing methods for curating high-quality SFT data rely heavily on strong reasoning models to filter examples based on diversity and difficulty, making the curation process costly while often yielding suboptimal data quality. In this work, we show that diverse and challenging reasoning examples can be identified using only the initial reasoning tokens. Specifically, we demonstrate that difficult problems can be reliably detected based on the loss of the first 100 reasoning tokens evaluated at a randomly perturbed checkpoint of the pretrained model. We further show that examples exhibiting similar loss patterns over their first 1k reasoning tokens across a small number of perturbed checkpoints extrapolating along the fine-tuning trajectory provabl
    
[^64]: MIRROR：面向智能体检索增强生成的新颖性约束记忆引导蒙特卡洛树搜索红队测试方法

    MIRROR: Novelty-Constrained Memory-Guided MCTS Red-Teaming for Agentic RAG

    [https://arxiv.org/abs/2606.26793](https://arxiv.org/abs/2606.26793)

    提出MIRROR框架，通过新颖性约束和记忆引导的蒙特卡洛树搜索，统一解决多模态智能体RAG系统的多种攻击面，显著提升攻击成功率并避免攻击模板重复。

    

    arXiv:2606.26793v1 公告类型：交叉 摘要：多模态智能体检索增强生成（RAG）系统将攻击面从提示注入扩展到文本投毒、图像注入、直接查询攻击以及编排器级工具操纵。现有的红队测试方法通常针对特定攻击面，且经常重复使用已知的攻击模板；在文本投毒基准测试中，我们测量到73-84%的完全重复。我们提出MIRROR，一个统一的跨攻击面框架，该框架执行记忆引导的蒙特卡洛树搜索，同时在显式新颖性约束下基于检索到的上下文对候选生成进行条件化。确定性新颖性门控机制会拒绝任何与归一化比较下的检索集匹配的候选，从而允许检索信息用于搜索先验，同时避免提示复制。在多模态智能体RAG目标的四个攻击面上，MIRROR在图像投毒上达到了76%的攻击成功率（ASR），而基线仅为52%；在编排器攻击上达到了97%的ASR。

    arXiv:2606.26793v1 Announce Type: cross  Abstract: Multimodal agentic retrieval-augmented generation (RAG) systems expand the attack surface beyond prompt injection to include text poisoning, image injection, direct-query attacks, and orchestrator-level tool manipulation. Existing red-teaming approaches are typically surface-specific and often recycle known attack templates; on text-poisoning benchmarks we measure 73-84% exact duplication. We present MIRROR, a unified cross-surface framework that performs memory-guided Monte Carlo tree search while conditioning candidate generation on retrieved context under an explicit novelty constraint. A deterministic Novelty Gate rejects any candidate matching the retrieval set under normalized comparison, allowing retrieval to inform search priors without enabling prompt copying. Across four attack surfaces on a multimodal agentic RAG target, MIRROR attains 76% ASR on image poisoning compared with 52% for baselines, 97% ASR on orchestrator attack
    
[^65]: AIGP：基于大语言模型的电商定价长期价值对齐框架

    AIGP: An LLM-Based Framework for Long-Term Value Alignment in E-Commerce Pricing

    [https://arxiv.org/abs/2606.26787](https://arxiv.org/abs/2606.26787)

    本文提出AIGP框架，通过大语言模型结合领域知识、结构化数据和文本上下文实现可解释的定价决策，并利用长期价值估计器和直接偏好优化使定价策略与累计GMV、ROI等长期业务目标对齐。

    

    传统的大规模电商动态定价模型存在可解释性有限、非结构化信息利用不足以及与长期业务目标（如累计商品交易总额、投资回报率和里程碑达成）不一致的问题。我们提出AIGP，这是一个新颖的框架，利用领域知识提示的大语言模型、结构化数据和文本上下文来做出可解释、知识感知的定价决策。为了在保持高质量输出的同时实现高效部署，我们采用监督微调进行知识蒸馏。AIGP的核心是长期价值估计器，它通过离线强化学习在历史数据上训练，作为奖励模型对候选定价行为进行评分，并选择偏好对用于直接偏好优化，从而使定价策略与长期业务目标对齐。

    arXiv:2606.26787v1 Announce Type: cross  Abstract: Traditional dynamic pricing models in large-scale e-commerce suffer from limited interpretability, poor utilization of unstructured information, and misalignment with long-term business objectives such as cumulative Gross Merchandise Value (GMV), Return on Investment (ROI) and milestone achievement. We propose AIGP, a novel framework that leverages a Large Language Model (LLM) prompted with domain knowledge, structured data and textual context to make interpretable, knowledge-aware pricing decisions. For efficient deployment while maintaining high-quality outputs, we employ supervised fine-tuning for knowledge distillation. Central to AIGP is the Long-Term Value Estimator (LTVE), trained via offline reinforcement learning on historical data, which serves as a reward model to score candidate pricing actions and select preference pairs for Direct Preference Optimization (DPO), thereby aligning the pricing policy with long-term business o
    
[^66]: “AlphaEdit：语言模型的零空间约束知识编辑”的可重复性研究

    Reproducibility Study of "AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models"

    [https://arxiv.org/abs/2606.26783](https://arxiv.org/abs/2606.26783)

    本研究复现了AlphaEdit知识编辑方法，发现其原始结果基本可重复，但在流畅性指标上存在差异，且该方法在新型模型架构上的优势不具有普遍性。

    

    Fang等人（2025）提出了一种名为AlphaEdit的零空间约束投影方法，用于“定位-编辑”式知识编辑技术，该方法在理论上保证了编辑操作不会破坏先前保存的知识，并在LLaMA3、GPT2-XL和GPT-J上报告了相较于现有编辑方法的显著性能提升。本研究对AlphaEdit进行了可重复性验证，在原始实验设置下复现了其报告的结果，并沿着三个方向扩展了评估：新的模型架构、额外的下游基准测试以及更长的序列编辑范围。我们成功地在原始模型上复现了AlphaEdit报告的指标，但在报告的流畅性和一致性指标上发现了一处差异。将AlphaEdit扩展到更新的模型系列后，我们发现其优势并未普遍适用，这归因于“定位-编辑”范式中的架构假设。

    arXiv:2606.26783v1 Announce Type: cross  Abstract: Fang et al. (2025) introduced a null-space constrained projection, named AlphaEdit, for locate-then-edit knowledge editing methods, theoretically guaranteeing that edits do not disrupt previously preserved knowledge, and reports substantial gains over existing editing methods on LLaMA3, GPT2-XL, and GPT-J. In this work, we present a reproducibility study of AlphaEdit, reproducing its reported results under the original experimental setup and extending the evaluation along three axes: new model architectures, additional downstream benchmarks, and substantially longer sequential editing horizons. We successfully reproduce AlphaEdit's reported metrics across the original models, though we identify a discrepancy in the reported fluency and consistency metric. Extending AlphaEdit to newer model families, we find that its advantage does not generalize uniformly, which we trace to architectural assumptions in the locate-then-edit paradigm tha
    
[^67]: LearniBridge：用于扩散模型加速的特征缓存的可学习校准方法

    LearniBridge: Learnable Calibration of Feature Caching for Diffusion Models Acceleration

    [https://arxiv.org/abs/2606.26778](https://arxiv.org/abs/2606.26778)

    提出LearniBridge，一种基于低秩子空间结构洞察的可学习特征缓存校准方法，仅需少量训练样本即可显著减少高加速比下的误差累积，实现高达5.87倍的加速。

    

    arXiv:2606.26778v1 公告类型：交叉 摘要：扩散变换器（DiTs）在图像和视频生成方面取得了显著进展，但计算成本过高。特征缓存通过重用中间表示来加速推理。现有方法为简化实现而依赖历史特征，但在高加速比下存在严重的误差累积问题。为解决这一限制，我们研究了所需特征校正的本质。我们证明，最优校准更新具有跨不同提示共享的低秩子空间特征。基于这一结构洞察，我们提出了LearniBridge，一种用于特征缓存的可学习校准机制，通过轻量级LoRA更新桥接多个时间步。该机制仅需3-5个训练样本即可实现有效校准。在图像和视频生成上的大量实验表明，LearniBridge实现了高达5.87倍的加速。

    arXiv:2606.26778v1 Announce Type: cross  Abstract: Diffusion Transformers (DiTs) have driven substantial progress in image and video generation but suffer from prohibitive computational costs. Feature caching accelerates inference by reusing intermediate representations. Existing methods rely on historical features for implementation simplicity, yet suffer from severe error accumulation at high acceleration ratios. To address this limitation, we investigate the nature of the requisite feature correction. We demonstrate that the optimal calibration update is characterized by a shared low-rank subspace across diverse prompts. Guided by this structural insight, we propose LearniBridge, a learnable calibration mechanism for feature caching that bridges multiple timesteps through lightweight LoRA updates. This mechanism enables effective calibration requiring only 3-5 training samples. Extensive experiments on image and video generation show that LearniBridge achieves up to $5.87\times$, $5
    
[^68]: 多媒体事件抽取中的评估陷阱与挑战

    Evaluation Pitfalls and Challenges in Multimedia Event Extraction

    [https://arxiv.org/abs/2606.26775](https://arxiv.org/abs/2606.26775)

    本文首次系统揭示了多媒体事件抽取中因数据处理、任务假设和评估设置不一致导致的性能高估问题，并呼吁建立标准化评估框架。

    

    多媒体事件抽取旨在联合识别跨多种模态（如文本和图像）的事件及其论元，以支持更全面的事件理解。尽管近期研究报道了稳步且显著的进展，但这些结果的可靠性和可比性关键取决于一致且严谨的评估。在本工作中，我们首次系统分析了多媒体事件抽取中的评估陷阱，并识别出三个主要问题来源：不一致的数据处理、不一致的任务假设以及过于宽松的评估设置。通过在一套严格评估框架下进行一系列控制实验，我们证明微小的评估选择会导致巨大的性能变化，并导致模型跨模态理解真实世界事件的能力被高估。我们的发现强调了建立可比评估标准的必要性，并鼓励更严谨的评估实践。

    arXiv:2606.26775v1 Announce Type: new  Abstract: Multimedia event extraction aims to jointly identify events and their arguments across multiple modalities, such as text and images, to support more comprehensive event understanding. While recent work reports steady and substantial progress, the reliability and comparability of these results critically depend on consistent and rigorous evaluation. In this work, we present the first systematic analysis of evaluation pitfalls in multimedia event extraction and identify three major sources of issues: inconsistent data processing, inconsistent task assumptions, and overly relaxed evaluation settings. We demonstrate, through a series of controlled experiments under a strict evaluation framework, that minor evaluation choices can cause large performance variations and lead to overestimation of a model's ability to ground real-world events across modalities. Our findings highlight the need for comparable evaluation standards and encourage a sh
    
[^69]: 逃离迭代参数空间噪声：基于超网络的差分隐私学习

    Escaping Iterative Parameter-Space Noise: Differentially Private Learning with a Hypernetwork

    [https://arxiv.org/abs/2606.26772](https://arxiv.org/abs/2606.26772)

    提出了一种基于超网络的新框架，通过仅一次向低维数据集表示注入隐私噪声，避免了迭代参数空间噪声，从而显著降低差分隐私学习中噪声的不利影响。

    

    神经网络差分隐私（DP）训练常常受到基于梯度的方法（如DP-SGD）所需大量噪声的阻碍，这些方法在整个训练过程中反复向参数空间注入高维噪声。本文提出了一种新的DP学习框架，避免了参数空间中的迭代优化。我们不使用私有梯度更新目标模型，而是采用在公共数据集上训练的超网络，将私有数据集映射到目标模型的参数。具体来说，每个示例被嵌入到一个低维表示中，嵌入被聚合和扰动以获得DP数据集嵌入，超网络从该噪声嵌入生成目标模型参数。由于隐私噪声仅注入一次到低维数据集表示中，我们的方法可以显著降低噪声的不利影响。我们从理论上证明了该框架的隐私保证，并实验表明其在多个基准数据集上优于标准DP-SGD方法。

    arXiv:2606.26772v1 Announce Type: new  Abstract: Differentially private (DP) training of neural networks is often hindered by the large amount of noise required by gradient-based methods such as DP-SGD, which repeatedly inject high-dimensional noise in parameter space throughout training. In this paper, we propose a new framework for DP learning that avoids iterative optimization in parameter space. Instead of updating the target model using privatized gradients, we employ a hypernetwork trained on public datasets to map a private dataset to the parameters of the target model. Specifically, each example is embedded into a low-dimensional representation, the embeddings are aggregated and perturbed to obtain a DP dataset embedding, and the hypernetwork generates the target model parameters from this noisy embedding. Because privacy noise is injected only once into a low-dimensional dataset representation, our approach can significantly reduce the adverse effect of noise. We theoretically
    
[^70]: ProtoKV：基于摘要状态记忆的延迟查询流式视频理解

    ProtoKV: Streaming Video Understanding under Delayed Query with Summary-State Memory

    [https://arxiv.org/abs/2606.26762](https://arxiv.org/abs/2606.26762)

    本文提出ProtoKV，通过将远程历史压缩为固定容量的摘要状态而非保留所有令牌，在恒定内存下解决了流式视频理解中延迟查询导致的关键线索被稀释或逐出的问题，准确率提升高达12.5个百分点。

    

    arXiv:2606.26762v1 公告类型：交叉 摘要：流式视频理解（SVU）必须在视觉令牌持续流式传输的同时，在严格的GPU内存和查询时间延迟预算下，回答异步到达的查询。一个关键挑战是延迟查询：决定性线索可能短暂出现，但在查询到达之前，后续会进行许多更新，这增加了这些线索在有限内存下被逐出或稀释的风险。我们提出ProtoKV，一种恒定占用内存的SVU记忆方法，它将远程历史表示为一个固定容量的摘要状态，而不是保留令牌实例。ProtoKV保留一个精确的近窗口KV缓存，并将更旧的内容聚合到一个具有残差统计信息的语义-空间原型库中。在查询时，每个原型通过一个有边界的伪令牌接口暴露出来，该接口与标准注意力机制兼容。在匹配的预算和可比的查询时间成本下，ProtoKV在S基准上的准确率比令牌保留基线提高了最多12.5个百分点。

    arXiv:2606.26762v1 Announce Type: cross  Abstract: Streaming video understanding (SVU) must answer queries that arrive asynchronously while visual tokens stream continuously under strict GPU-memory and query-time latency budgets. A key challenge is delayed query: decisive cues may appear briefly, yet many subsequent updates occur before the query arrives, increasing the risk that those cues are evicted or diluted under bounded memory. We propose ProtoKV, a constant-footprint SVU memory that represents far history as a fixed-capacity summary state rather than retaining token instances. ProtoKV keeps an exact near-window KV cache and aggregates older content into a semantic-spatial prototype bank with residual statistics. At query time, each prototype is exposed through a bounded pseudo-token interface that is drop-in compatible with standard attention. Under matched budgets and comparable query-time cost, ProtoKV improves accuracy by up to 12.5 points over token-retention baselines on S
    
[^71]: 用于鲁棒且可解释的昆虫认证的批次不变光谱智能技术

    Batch-Invariant Spectral Intelligence for Robust and Explainable Insect Authentication

    [https://arxiv.org/abs/2606.26757](https://arxiv.org/abs/2606.26757)

    提出了一个端到端的批次不变光谱网络（BISN），通过结合可学习的预处理模块和熵正则化对抗目标，有效抑制了光谱测量中的批次间变异，从而在未见过的生产批次上实现了鲁棒且可解释的昆虫物种认证。

    

    arXiv:2606.26757v1 公告类型：新 摘要：可食用昆虫提供了一种高效的替代蛋白质来源，与传统畜牧业相比，它们需要的土地和水更少，排放的温室气体也更少。然而，要成功地将它们融入食品供应链，需要可靠的物种认证，以控制过敏原暴露、防止掺假并满足监管标准。近红外光谱提供了一种快速分析工具，但由于光谱测量中的批次间差异，当应用于训练过程中未见过的生产批次时，其性能会下降。我们引入了批次不变光谱网络（BISN），这是一个端到端框架，它将一个可学习的预处理模块（初始化为Savitzky-Golay滤波）与一个熵正则化的对抗目标相结合，以抑制批次特定的光谱变异。与仅在特征提取后强制执行领域适应的域对抗神经网络不同，BISN抑制批次特定的光谱变化，从而在未见过的生产批次上实现鲁棒且可解释的昆虫认证。

    arXiv:2606.26757v1 Announce Type: new  Abstract: Edible insects offer an efficient source of alternative protein, requiring less land, water and emitting less greenhouse gas than conventional livestock. However, their successful integration into the food supply chain demands reliable species authentication to control allergen exposure, prevent adulteration, and meet regulatory standards. Near-infrared spectroscopy provides a rapid analytical tool, but its performance drops when applied to production batches unseen during training due to batch-to-batch variation in spectral measurements. We introduce the Batch-Invariant Spectral Network (BISN), an end-to-end framework that combines a learnable preprocessing module, initialised with Savitzky-Golay filtering, with an entropy-regularised adversarial objective to suppress batch-specific spectral variation. In contrast to Domain-Adversarial Neural Networks, which enforce domain adaptation only after feature extraction, BISN suppress batch-ef
    
[^72]: 崩溃前的结构：下一个词预测中的瞬态语义几何

    Structure Before Collapse: Transient semantic geometry in next-token prediction

    [https://arxiv.org/abs/2606.26749](https://arxiv.org/abs/2606.26749)

    该论文揭示了在单热标签训练下，语言模型通过瞬态几何结构学会了潜在语义类别，挑战了神经网络崩溃理论的对称性假设。

    

    神经网络崩溃理论预测，平衡的单热分类会使模型表示彼此等距，形成一种仅依赖于输出标签、忽略输入语义相似性的对称结构。这引发了一个难题：基于下一个词预测的语言模型主要（随着上下文长度增加）使用单热标签进行训练——同一上下文几乎不可能在训练中以不同标签出现两次。然而，它们显然学会了潜在的结构化特征。也就是说，尽管采用单热训练机制，语言模型的上下文嵌入仍能表示出“玛丽打破了___”中的下一个词很可能由属于以下潜在类别的词填充：a）中等大小、b）刚性、c）无生命的名词。当共现统计因单热稀疏性而崩溃，消除了不同输入之间共享的下一个词时，梯度下降是如何找到这种分类语义结构的呢？

    arXiv:2606.26749v1 Announce Type: cross  Abstract: Neural Collapse predicts that balanced one-hot classification pushes model representations to be equally far from each other; a symmetric configuration that depends only on the output label and ignores any semantic similarity in the inputs. This creates a puzzle: next-token prediction language models are trained predominantly (as context length increases) with one-hot labels: the same context is very unlikely to appear twice in training with different labels. However, they clearly learn latent structural features. That is, despite the one-hot training regime, a language model's contextual embeddings represent the fact that the next word in ''Mary broke the ___'' is likely to be filled by tokens in the latent classes of a) medium-sized, b) rigid, c) inanimate nouns. How does gradient descent find such categorical semantic structure when co-occurrence statistics collapse to one-hot sparsity, eliminating any shared next-tokens among diffe
    
[^73]: HyperDFlash：基于门控残差缩减的MHC对齐块级推测解码

    HyperDFlash: MHC-Aligned Block Speculative Decoding with Gated Residual Reduction

    [https://arxiv.org/abs/2606.26744](https://arxiv.org/abs/2606.26744)

    针对DeepSeek-V4的多超连接架构，提出了一种通过门控残差缩减实现特征对齐的块级推测解码方法，解决了多路径残差流导致的生成准确率下降问题。

    

    arXiv:2606.26744v1 公告类型：交叉 摘要：我们提出了HyperDFlash，这是一个针对DeepSeek-V4提出的新型多超连接（MHC）架构量身定制的块级并行推测解码框架。尽管DeepSeek-V4原生多令牌预测（MTP）模块在初始令牌生成方面表现强劲，但其后续位置的生成准确性会急剧下降，因为未验证中间令牌的错误累积会降低接受率。虽然原始的DFlash方法支持高效的单次块级生成，但它无法无缝适配到MHC范式，因为DeepSeek-V4的多路径残差流会导致与常规生成设计之间的特征错配。为了解决这一不匹配问题，我们针对MHC残差流提出了两种模型对齐优化。首先，我们采用预折叠残差状态作为唯一的条件信号，保留多路径结构信息，并将生成器与原生的预测路径对齐。

    arXiv:2606.26744v1 Announce Type: cross  Abstract: We present HyperDFlash, a block-parallel speculative decoding framework tailored to the novel multi-hyper-connection (MHC) architecture proposed by DeepSeek-V4. Despite the strong initial-token drafting performance of the native Multi-Token Prediction (MTP) module in DeepSeek-V4, its draft accuracy degrades sharply at later positions, as error accumulation from unverified intermediate tokens harms acceptance rates. Although the original DFlash method supports efficient one-pass block drafting, it cannot be seamlessly adapted to the MHC paradigm, since the multi-path residual stream of DeepSeek-V4 induces feature misalignment with conventional drafting designs. To resolve this mismatch, we propose two model-aligned optimizations for MHC residual streams. First, we adopt pre-collapse residual states as the exclusive conditioning signal, preserving multi-path structural information and aligning the drafter with the native prediction pathw
    
[^74]: 科学发现作为元优化：一个组合优化案例研究

    Scientific discovery as meta-optimization: a combinatorial optimization case study

    [https://arxiv.org/abs/2606.26728](https://arxiv.org/abs/2606.26728)

    本文提出将科学发现形式化为元优化，通过共识目标聚合方法结合LLM生成的目标函数，形成自修正评估标准，在3-SAT问题算法发现中显著降低了计算复杂度。

    

    arXiv:2606.26728v1 公告类型：新论文 摘要：科学发现本质上是一个优化问题，由理论和实验构成的巨大“状态空间”以及基于质量、新颖性和有效性的评估标准所定义。大型语言模型（LLMs）实现了对这一空间的自动探索，但我们认为，同时修改评估标准同样重要。在此，我们提出将研究形式化为元优化，即优化目标本身也在被优化。我们的关键贡献是“共识目标聚合”，通过相关加权投票将LLM生成的目标函数结合起来，形成一个稳定、自我修正的评估标准，该标准会随着理解的加深而演变。我们将此框架应用于基于数字MemComputing机器的3-SAT问题算法发现，将基线随问题规模$N$的缩放从约$N^{2.51}$降低到约$N^{1.33}$，并实现了约

    arXiv:2606.26728v1 Announce Type: new  Abstract: Scientific discovery is fundamentally an optimization problem, defined by a vast "state space" of theories and experiments, and an evaluation criterion based on quality, novelty, and validity. Large language models (LLMs) have enabled automated exploration of this space, but we argue that simultaneous modification of the evaluation criteria is equally important. Here, we propose formalizing research as meta-optimization, where the optimization objective itself is also being optimized. Our key contribution is "consensus objective aggregation," where LLM-generated objective functions are combined via correlation-weighted voting, yielding a stable, self-correcting evaluation criterion that evolves as understanding deepens. We apply this framework to algorithm discovery for 3-SAT problems based on digital MemComputing machines, reducing the baseline scaling with problem size $N$ from $\sim N^{2.51}$ to $\sim N^{1.33}$ and delivering a $\sim 
    
[^75]: 用于情感和压力识别的状态特异性呼吸特征：可解释的呼吸标记、自相关滞后与紧凑型CNN模型

    State-Specific Respiratory Signatures for Affective and Stress Recognition: Interpretable Respiratory Markers, Autocorrelation Lags, and Compact CNN Models

    [https://arxiv.org/abs/2606.26723](https://arxiv.org/abs/2606.26723)

    本研究通过结合紧凑型一维CNN和手工设计的呼吸特征，不仅实现了压力与非压力的高精度二分类检测，还识别出了基线、压力、愉悦和冥想状态各自特异的可解释呼吸标记。

    

    arXiv:2606.26723v1 公告类型：交叉 摘要：呼吸活动是可穿戴压力与情感状态识别中一种直接且可解释的生理通道，然而许多研究侧重于分类准确性，而未识别出哪些呼吸属性能够区分不同状态。本研究将基于呼吸信号（RESP）的识别重新定义为一个联合预测与解释问题。利用WESAD数据集中的胸部呼吸通道，我们在留一被试交叉验证下分析60秒窗口，并结合两个互补分支：紧凑的原始信号一维卷积神经网络（1D-CNN）和物理分组的基于手工特征的呼吸特征。主要应用任务是二分类的压力与非压力检测，同时在一对多设置中额外分析基线、压力、愉悦和冥想状态，以揭示状态特异性的呼吸标记。特征空间被组织为呼吸时序、呼吸间变异、波形特征等。

    arXiv:2606.26723v1 Announce Type: cross  Abstract: Respiratory activity is a direct and interpretable physiological channel for wearable stress and affective-state recognition, yet many studies emphasize classification accuracy without identifying which respiratory properties separate different states. This work reframes RESP-based recognition as a joint predictive and explanatory problem. Using the chest respiratory channel of the WESAD dataset, we analyze 60 s windows under leave-one-subject-out validation and combine two complementary branches: compact raw-signal one-dimensional convolutional neural networks (1D-CNNs) and physically grouped handcrafted respiratory signatures. The primary application task is binary stress versus non-stress detection, while baseline, stress, amusement, and meditation are additionally analyzed in a one-vs-rest setting to reveal state-specific respiratory markers. The feature space is organized into respiratory timing, breath-to-breath variability, wave
    
[^76]: DroidBreaker：针对机器学习安卓恶意软件检测器的实用且功能性空间攻击

    DroidBreaker: Practical and Functional Problem-Space Attacks on Machine-Learning Android Malware Detectors

    [https://arxiv.org/abs/2606.26707](https://arxiv.org/abs/2606.26707)

    DROIDBREAKER是一种在问题空间中通过构建安全和语义保持的方法，对安卓APK进行实用且功能性的修改，以有效逃避机器学习恶意软件检测器。

    

    arXiv:2606.26707v1 公告类型：交叉 摘要：对抗性APK是在问题空间中修改的安卓应用，用于逃避机器学习恶意软件检测器。在这项工作中，我们首先表明，尽管有相关声称，现有的问题空间攻击在很大程度上仍不实用。大多数技术利用软件移植来注入整个良性模块，引入许多副作用特征，并常常导致构建时失败。仅注入窄子集组件的细粒度方法效果有限，而那些同时使用混淆的方法依赖于脆弱的字节码重写，产生的APK在语法上有效但在语义上不可用。先前的工作还通过仅验证安装和基本执行的冒烟测试高估了攻击成功率，而未评估修改后的APK是否仍保留其预期行为。为克服这些局限，我们提出了DROIDBREAKER，一种实用（构建安全）且功能性（语义保持）的攻击方法。

    arXiv:2606.26707v1 Announce Type: cross  Abstract: Adversarial APKs are Android applications modified in the problem space to evade machine-learning malware detectors. In this work, we first show that, despite claims, existing problem-space attacks remain largely impractical. Most techniques leverage software transplantation to inject entire benign modules, introducing many side-effect features and often causing build-time failures. Fine-grained methods that inject only a narrow subset of components exhibit limited effectiveness, while those that also use obfuscation rely on brittle bytecode rewriting, producing APKs that are syntactically valid but semantically unusable. Prior work further overestimates attack success rates by running smoke tests that only validate installation and basic execution, without assessing whether the modified APK still preserves its intended behavior. To overcome these limitations, we present DROIDBREAKER, a practical (build-safe) and functional (semantics-
    
[^77]: 深度学习算法基础：复杂度理论速率与普适逼近的表征

    Algorithmic Foundations of Deep Learning: Complexity-Theoretic Rates and a Characterization of Universal Approximation

    [https://arxiv.org/abs/2606.26705](https://arxiv.org/abs/2606.26705)

    本文提出神经网络应被视为计算模型，其复杂度由算法复杂度而非仅正则性决定，并给出基于电路计算框架的普适逼近表征。

    

    前馈神经网络（NN）的表达能力通常通过模拟最优基展开方案来研究。尽管这种方法很强大，但它并不完整：它主要通过正则性来捕捉复杂度，因此无法区分具有相似正则性的直观简单和复杂对象，例如平方根函数和典型布朗路径。核心观点是，神经网络不仅应被视为灵活的基函数，还应被视为计算模型。如果一个函数可以通过预设基本运算语言上的实值电路计算，那么它就可以通过具有显式深度、宽度和非零参数边界的神经网络以可比精度计算，这些边界由深度、宽度、门计数和门结构控制。因此，神经网络的复杂度不仅由正则性决定，还由算法复杂度决定。我们随后证明，任何可定义的满足特定条件的神经网络模型都可以实现普适逼近。

    arXiv:2606.26705v1 Announce Type: cross  Abstract: Feedforward neural network (NN) expressivity is typically studied by emulating optimal basis-expansion schemes. While powerful, this perspective is incomplete: it primarily captures complexity through regularity, and therefore does not distinguish intuitively simple and complicated objects with comparable regularity, such as the square-root function and a typical Brownian path.   The guiding message is that neural networks should be viewed not only as flexible basis functions, but also as models of computation. If a function is computable by a real-valued circuit over a prescribed elementary gate language, then it can be computed to comparable accuracy by an NN with explicit depth, width, and non-zero-parameter bounds controlled by the depth, width, gate count, and gate structure. Thus, neural-network complexity is not governed by regularity alone, but also by algorithmic complexity. We then show that any definable NN model satisfying 
    
[^78]: 有归因，但非增量：大规模广告中修正蚕食效应的归因方法

    Attributed, But Not Incremental: Cannibalization-Corrected Attribution for Large-Scale Advertising

    [https://arxiv.org/abs/2606.26690](https://arxiv.org/abs/2606.26690)

    提出一种利用增量实验作为因果锚点，将稀疏提升度量转化为每日修正估计的归因修正框架，并在结构性约束下分配蚕食量，以解决大规模广告中付费归因高估增量增长的问题。

    

    在大规模付费获客与增长广告系统中，生产级归因输出被广泛用于日常预算分配和渠道诊断。然而，当付费渠道与自然需求、品牌驱动流量或其他获客渠道重叠时，付费归因的转化（如每日新增用户）可能会系统性地高估真实的增量增长。这种归因-蚕食错配会扭曲大规模的增量ROI测算和预算决策。我们提出了一种实验校准的归因修正框架，该框架以增量实验作为因果锚点，将稀疏的提升度量转化为每日修正估计。为了使修正后的信号在生产粒度上具有可操作性，我们进一步在结构性一致性约束下，跨业务层级分配校准后的蚕食量。通过针对渠道级增量的离线前向时间验证，我们证明了该方法的效果。

    arXiv:2606.26690v1 Announce Type: cross  Abstract: In large-scale paid acquisition and growth advertising systems, production attribution outputs are widely used for daily budget allocation and channel diagnosis. However, paid-attributed conversions such as daily new users (DNU) may systematically overstate true incremental growth when paid channels overlap with organic demand, brand-driven traffic, or other acquisition channels. This attribution-cannibalization mismatch can distort incremental ROI measurement and budget decisions at scale.   We propose an experiment-calibrated attribution correction framework that uses incrementality experiments as causal anchors to convert sparse lift measurements into daily correction estimates. To make the corrected signal actionable at production granularity, we further allocate calibrated cannibalization volume across business hierarchies under structural consistency constraints. Offline forward-in-time validation against channel-level incrementa
    
[^79]: PersistentKV：面向长上下文LLM在商用GPU上服务的页面感知解码调度

    PersistentKV: Page-Aware Decode Scheduling for Long-Context LLM Serving on Commodity GPUs

    [https://arxiv.org/abs/2606.26666](https://arxiv.org/abs/2606.26666)

    本文提出PersistentKV，一种针对商用GPU上长上下文LLM服务的页面感知解码调度引擎，通过按KV头组映射工作、重用K/V块和紧凑工作队列调度，优化了页面感知解码的效率。

    

    自回归大型语言模型（LLM）服务正越来越多地受到键值（KV）缓存移动的限制，而非密集矩阵乘法。现代分页注意力系统减少了KV缓存碎片，而成熟的核函数（如FlashInfer）提供了高度优化的原生分页解码注意力。然而，最佳的单核实现并不总是最佳的服务调度方案：低活跃度的长上下文解码可能无法充分利用商用GPU，而混合序列长度则在精确长度启动和粗粒度填充批次之间引入了张力。我们提出了PersistentKV，一种原生的块表解码注意力引擎，并针对分组查询注意力（GQA）进行了页面感知调度研究。PersistentKV按KV头组映射工作，设计用于跨分组查询头重用K、V块，支持原生页表，并添加了一个紧凑的工作队列调度，该调度仅执行非空的行-KV头-序列拆分任务。在RTX 3060上

    arXiv:2606.26666v1 Announce Type: new  Abstract: Autoregressive large language model (LLM) serving is increasingly limited by key-value (KV) cache movement rather than dense matrix multiplication. Modern paged-attention systems reduce KV-cache fragmentation and mature kernels such as FlashInfer provide highly optimized native-paged decode attention. However, the best single-kernel implementation is not always the best serving schedule: low-active long-context decode can under-utilize commodity GPUs, while mixed sequence lengths introduce a tension between many exact-length launches and coarse padded batches. We present PersistentKV, a native block-table decode attention engine and page-aware scheduling study for grouped-query attention (GQA). PersistentKV maps work by KV-head group, is designed to reuse K,V tiles across grouped query heads, supports native page tables, and adds a compact workqueue schedule that executes only non-empty row-KV-head-sequence-split tasks. On an RTX 3060 wi
    
[^80]: 基于稀疏随机图的神经ODE零样本尺寸迁移：图极限与伴随收敛性

    Zero-Shot Size Transfer for Neural ODEs on Sparse Random Graphs: Graphon Limits and Adjoint Convergence

    [https://arxiv.org/abs/2606.26662](https://arxiv.org/abs/2606.26662)

    本文针对稀疏随机图，从理论上证明了图神经微分方程具有零样本尺寸迁移能力，即在小图上训练后可无需重新训练直接部署到更大图上，并给出了收敛率分析。

    

    arXiv:2606.26662v1 公告类型：交叉 摘要：图神经微分方程通过用图神经网络参数化神经ODE速度场来建模连续时间图动态。其局部、尺寸无关的滤波器暗示了一种零样本尺寸迁移原理：在小图上训练，然后部署到更大、相似的图上而无需重新训练。我们针对从图极限中采样的稀疏随机图，为该原理发展了一种定量理论。我们考虑图极限神经微分方程和伴随图极限神经微分方程，作为前向和伴随GNDE系统的无限节点极限，并建立了适定性。对于一个具有稀疏参数α_n的n节点随机图，我们以高概率证明了GNDE解向图极限神经微分方程解的轨迹收敛率，为O((α_n n)^{-1/2})（忽略对数因子）。我们还为控制隐藏状态和参数梯度的伴随系统建立了时间一致收敛界。

    arXiv:2606.26662v1 Announce Type: cross  Abstract: Graph Neural Differential Equations (GNDEs) model continuous-time graph dynamics by parameterizing Neural ODE velocity fields with Graph Neural Networks. Their local, size-independent filters suggest a zero-shot size-transfer principle: train on a small graph and deploy on larger, similar graphs without retraining. We develop a quantitative theory for this principle on sparse random graphs sampled from graphons. We consider Graphon Neural Differential Equations (Graphon-NDEs) and adjoint Graphon-NDEs as the infinite-node limits of the forward and adjoint GNDE systems, and establish well-posedness. For an $n$-node random graph with sparsity parameter $\alpha_n$, we prove trajectory-wise convergence of GNDE solutions to Graphon-NDE solutions at rate $O((\alpha_n n)^{-1/2})$, up to logarithmic factors, with high probability. We also establish uniform-in-time convergence bounds for adjoint systems governing hidden-state and parameter gradi
    
[^81]: 利用Transformer生成特殊三角剖分

    Generating Special Triangulations with Transformers

    [https://arxiv.org/abs/2606.26660](https://arxiv.org/abs/2606.26660)

    本文提出使用Transformer模型并配以适当编码方案，能够有效生成四维自反多面体的精细、规则且星形三角剖分，且模型可通过自我再训练改进，为卡拉比-丘流形分类及多领域研究提供新方法。

    

    三角剖分，即将几何对象分解为类似三角形的结构，是数学和物理学许多领域的核心对象。特别是，四维自反多面体的精细、规则且星形的三角剖分（FRST）能产生光滑的卡拉比-丘三维流形，这在弦理论中具有重要意义。然而，三角剖分的高维度和组合复杂性使得使用经典数值方法或机器学习对其进行建模极具挑战。在这项工作中，我们证明，配备适当编码方案的Transformer能够被有效训练，以代表性方式生成一系列多面体尺寸上的新FRST。此外，这些模型还能通过对其自身输出进行再训练来实现自我改进。这为卡拉比-丘流形分类的具体应用以及物理学、组合学和代数几何的进一步研究打开了大门。

    arXiv:2606.26660v1 Announce Type: cross  Abstract: Triangulations, i.e., well-structured decompositions of geometric objects into triangle-like pieces, are central objects in many domains of mathematics and physics. In particular, fine, regular, and star triangulations (FRSTs) of 4D reflexive polytopes give rise to smooth Calabi-Yau threefolds, which are of significant interest in string theory. However, the high dimensionality and combinatorial complexity of triangulations make them particularly challenging to model with classical numerical methods or machine learning. In this work, we show that transformers, equipped with an appropriate encoding scheme, can be effectively trained to representatively generate new FRSTs across a range of polytope sizes. Moreover, these models can also self-improve through retraining on their own output. This opens the door to both concrete applications to the classification of Calabi-Yau manifolds and further research in physics, combinatorics and alge
    
[^82]: 面向化学空间中可扩展替代优化的目标感知赌博机分配

    Target-Aware Bandit Allocation for Scalable Surrogate Optimization in Chemical Space

    [https://arxiv.org/abs/2606.26657](https://arxiv.org/abs/2606.26657)

    提出BOBa框架，利用多臂赌博机自适应分配计算资源到动作空间分区，消除全库推理，实现化学空间中大规模分子库的高效替代优化。

    

    arXiv:2606.26657v1 公告类型：新 摘要：在昂贵评估条件下从大规模离散空间中识别高价值候选者，是科学领域反复出现的挑战，其中基于结构的药物发现是一个突出例子。虽然基于替代模型的优化可以通过减少昂贵评估次数来提高样本效率，但现代分子库已达到数十亿到万亿个化合物，使得全库替代推理本身成为主要计算瓶颈。我们提出BOBa，一种赌博机引导的替代优化框架，通过自适应地将计算分配到动作空间的各个分区，消除了全库推理。通过将分区视为多臂赌博机中的臂，BOBa将推理和评估集中在经验上有前景的分区上，同时保持原则性的探索。在现实世界按需合成库上的实验表明，不确定性乐观赌博机与...相结合。

    arXiv:2606.26657v1 Announce Type: new  Abstract: Identifying high-utility candidates from massive discrete spaces under expensive evaluations is a recurring challenge across the sciences, with structure-based drug discovery as a prominent example. While surrogate-based optimization can increase sample efficiency by reducing the number of expensive evaluations, modern molecular libraries have reached billions to trillions of compounds, making full-library surrogate inference itself a major computational bottleneck. We introduce BOBa, a bandit-guided surrogate optimization framework that eliminates full-library inference by adaptively allocating computation across partitions of the action space. By treating partitions as arms in a multi-armed bandit, BOBa concentrates inference and evaluations on empirically promising partitions while maintaining principled exploration. Experiments on real-world synthesis-on-demand libraries demonstrate that optimism-under-uncertainty bandits, combined w
    
[^83]: 分式松弛像素动力学下的事件相机仿真

    FracEvent: Event-Camera Simulation via Fractional-Relaxation Pixel Dynamics

    [https://arxiv.org/abs/2606.26636](https://arxiv.org/abs/2606.26636)

    提出FracEvent事件仿真器，通过分式松弛电压动力学精确建模像素生命周期，以改善事件时序保真度和下游任务迁移性能。

    

    事件相机能以微秒级时间分辨率异步报告亮度变化，但由于需要专用传感器、精确同步和任务特定标注，真实事件数据难以大规模采集。因此，事件相机仿真对于基于事件的视觉任务至关重要。大多数实用仿真器基于对比度阈值事件生成，部分还加入额外滤波、随机噪声或手动调参的传感器参数。尽管有效，这类公式常简化每个像素生命周期产生的时间结构，从而扭曲事件时序并削弱下游任务迁移效果。我们提出FracEvent，一种采用分式松弛电压动力学建模像素级生命周期的事件仿真器。给定对数强度轨迹，FracEvent驱动一个紧凑的松弛模式栈，将其响应合并为电压状态，并通过比较电压与阈值来触发ON/OFF事件。

    arXiv:2606.26636v1 Announce Type: cross  Abstract: Event cameras asynchronously report brightness changes with microsecond-level temporal resolution, but real event data remain difficult to collect at scale because specialized sensors, careful synchronization, and task-specific annotations are required. Event-camera simulation is therefore important to event-based vision tasks. Most practical simulators build on contrast-threshold event generation, some with additional filtering, stochastic noise, or hand-tuned sensor parameters. While effective, such formulations often simplify the temporal structure produced by the lifecycle of each pixel, which can distort event timing and weaken downstream transfer. We introduce FracEvent, an event simulator that models this pixel-level lifecycle with fractional-relaxation voltage dynamics. Given a log-intensity trajectory, FracEvent drives a compact stack of relaxation modes, combines their responses into a voltage state, emits ON/OFF events by lo
    
[^84]: 从权重到特征：基于SAE引导的激活正则化实现大语言模型持续学习

    From Weights to Features: SAE-Guided Activation Regularization for LLM Continual Learning

    [https://arxiv.org/abs/2606.26629](https://arxiv.org/abs/2606.26629)

    本文提出利用稀疏自编码器（SAE）在激活空间进行正则化，通过单义特征字典显式平衡稳定性与可塑性，解决了大语言模型持续学习中EWC等权重正则化方法因多义性而表现不佳的问题。

    

    权重空间正则化方法（如弹性权重巩固EWC）是持续学习中应对灾难性遗忘的标准方法。然而，当这些方法应用于大语言模型时，往往表现不佳。我们认为，这种不佳表现部分可归因于大语言模型的“多义性”本质：EWC风格正则化所使用的逐权重重要性估计过于粗糙，无法隔离需要保护的知识。本文提出在模型的激活空间中进行正则化，利用预训练的稀疏自编码器（SAE）作为单义特征字典。从约束优化的角度，我们推导出一种新的损失函数，该函数利用SAE特征字典显式地平衡稳定性与可塑性，并证明EWC是单侧权重空间惩罚设置中的一个特例。与需要存储或重放数据的基于重放的方法不同，我们的方法避免了数据存储需求。

    arXiv:2606.26629v1 Announce Type: cross  Abstract: Weight-space regularization methods such as Elastic Weight Consolidation (EWC) are the standard approach to catastrophic forgetting in continual learning. However, those methods tend to underperform when applied to large language models. We argue that such underperformance can be partly explained by the ``polysemantic'' nature of large language models: per-weight importance estimates utilized by EWC-style regularization are too coarse and cannot isolate the knowledge that needs protection. In this paper, we propose regularizing instead in the model's activation space, using pretrained Sparse Autoencoders (SAEs) as a monosemantic feature dictionary. From the perspective of constrained optimization, we derive a new loss function that uses the SAE feature dictionary to explicitly balance stability and plasticity, and show that EWC is a special case in the one-sided weight-space penalty setting. Unlike replay-based methods that store or re
    
[^85]: 利用稀疏自编码器发现数百万个可解释特征

    Discovering Millions of Interpretable Features with Sparse Autoencoders

    [https://arxiv.org/abs/2606.26620](https://arxiv.org/abs/2606.26620)

    本文提出了Qwen3-Instruct SAE套件，在多个Qwen3模型上训练了覆盖不同激活位置的稀疏自编码器，并系统评估了其稀疏性与保真度的权衡，最后通过拒绝引导案例展示了其应用价值。

    

    稀疏自编码器已成为一种将叠加的语言模型表示分解为稀疏且可解释特征的有力工具。然而，训练稀疏自编码器计算成本高昂，且现有的开源稀疏自编码器模型仍然有限。在这项工作中，我们引入了**Qwen3-Instruct SAE**，这是一个在Qwen3指令微调模型系列上训练的全面稀疏自编码器套件，覆盖了Qwen3-1.7B、Qwen3-4B和Qwen3-8B。对于Qwen3-1.7B和Qwen3-4B，我们在三个关键激活位置训练了逐层稀疏自编码器：残差流、MLP输出和注意力输出。对于Qwen3-8B，我们在残差流层的子集上训练了稀疏自编码器。我们使用激活级重建指标和模型级恢复指标系统地评估了这些稀疏自编码器，揭示了跨层和组件的不同稀疏性-保真度权衡。最后，我们通过一个拒绝引导案例研究展示了Qwen3-Instruct SAE的实用性。

    arXiv:2606.26620v1 Announce Type: cross  Abstract: Sparse autoencoders (SAEs) have emerged as a powerful tool for decomposing superposed language model representations into sparse and interpretable features. However, training SAEs is computationally expensive, and available open-source SAE models remain limited. In this work, we introduce \textbf{Qwen3-Instruct SAE}, a comprehensive suite of SAEs trained on the Qwen3 instruction-tuned model family, covering Qwen3-1.7B, Qwen3-4B, and Qwen3-8B. For Qwen3-1.7B and Qwen3-4B, we train layer-wise SAEs at three key activation sites: residual streams, MLP outputs, and attention outputs. For Qwen3-8B, we train SAEs on a subset of residual stream layers. We systematically evaluate these SAEs using both activation-level reconstruction metrics and model-level recovery metrics, revealing distinct sparsity--fidelity trade-offs across layers and components. Finally, we demonstrate the utility of Qwen3-Instruct SAE through a refusal-steering case stud
    
[^86]: 草图线性对比学习：近似、优化与统计尺度

    Sketched Linear Contrastive Learning: Approximation, Optimization, and Statistical Scaling

    [https://arxiv.org/abs/2606.26617](https://arxiv.org/abs/2606.26617)

    本文针对对比学习中的草图线性模型，推导了包含近似、优化和统计误差的显式尺度定律，揭示了草图维度、样本大小与光谱衰减率之间的权衡关系。

    

    arXiv:2606.26617v1 公告类型：新 摘要：尺度定律描述了学习性能如何随模型大小、数据规模和计算量变化。虽然近期理论工作已为草图线性回归建立了尺度定律，但对对比表征学习的理解仍非常有限。本文研究了一种在配对高斯潜变量设置下用于对比学习的草图线性模型。学习者仅观察到两个相关变量的草图化视图，并通过全批次经验梯度下降训练一个双线性对比评分。我们在对齐幂律谱和对比源条件下分析了一个高斯负二次对比代理函数，其中我们将风险分解为不可约风险、近似误差、梯度下降偏差、梯度下降方差和一个交叉项。交叉项由偏差和方差控制，因此不影响上界尺度。我们的主要定理给出了一个关于草图化维度、样本大小、模型秩和光谱衰减率的显式尺度定律，并揭示了近似误差、优化偏差和统计方差之间的权衡。我们还刻画了梯度下降的动态行为，展示了早期停止如何通过避免过度拟合噪声特征来改善泛化性能。数值实验验证了理论预测。

    arXiv:2606.26617v1 Announce Type: new  Abstract: Scaling laws describe how learning performance varies with model size, data size, and compute. While recent theoretical work has established scaling laws for sketched linear regression, much less is understood for contrastive representation learning. In this paper, we study a sketched linear model for contrastive learning under a paired Gaussian latent-variable setup. The learner observes only sketched views of two correlated variables and trains a bilinear contrastive score by full-batch empirical gradient descent. We analyze a Gaussian-negative quadratic contrastive surrogate under aligned power-law spectra and a contrastive source condition, where we derive a risk decomposition into irreducible risk, approximation error, GD bias, GD variance, and a cross term. The cross term is controlled by the bias and variance and therefore does not affect the upper-bound scaling. Our main theorem gives an explicit scaling law with respect to sketc
    
[^87]: 基于代理似然引导的潜空间扩散后验采样用于偏微分方程反问题

    Latent Diffusion Posterior Sampling with Surrogate Likelihood Guidance for PDE Inverse Problems

    [https://arxiv.org/abs/2606.26592](https://arxiv.org/abs/2606.26592)

    提出一种结合变分自编码器、潜空间扩散模型和神经代理的贝叶斯反演方法，有效解决了高维PDE反问题中的先验建模、维度灾难和计算成本三大挑战。

    

    arXiv:2606.26592v1 公告类型：交叉 摘要：我们提出潜空间扩散后验采样（L-DPS），一种用于由偏微分方程（PDE）控制的高维反问题的近似贝叶斯框架。该方法解决了PDE约束反演中的三个挑战：缺乏可处理密度的隐式基于样本的先验、高维空间分布参数，以及后验采样过程中重复前向模型评估的高成本。L-DPS结合了变分自编码器、无条件潜空间扩散模型、扩散后验采样和可微神经代理。VAE将参数场映射到低维潜空间，扩散模型在此潜空间中学习隐式先验得分，而DPS将学习到的先验与基于似然的引导相结合。似然梯度通过解码器-代理组合进行评估，避免了重复调用完整的数值PDE求解器。

    arXiv:2606.26592v1 Announce Type: cross  Abstract: We propose latent-space diffusion posterior sampling (L-DPS), an approximate Bayesian framework for high-dimensional inverse problems governed by partial differential equations (PDEs). The method addresses three challenges in PDE-constrained inversion: implicit sample-based priors without tractable densities, high-dimensional spatially distributed parameters, and the high cost of repeated forward-model evaluations during posterior sampling. L-DPS combines a variational autoencoder, an unconditional latent diffusion model, diffusion posterior sampling, and a differentiable neural surrogate. The VAE maps the parameter field to a lower-dimensional latent space, the diffusion model learns an implicit prior score in this latent space, and DPS combines this learned prior with likelihood-based guidance. The likelihood gradient is evaluated through the decoder-surrogate composition, avoiding repeated calls to the full numerical PDE solver. We 
    
[^88]: 实证软件工程：TerraProbe——用于检测LLM辅助Terraform中欺骗性修复的分层Oracle框架

    Empirical Software Engineering TerraProbe: A Layered-Oracle Framework for Detecting Deceptive Fixes in LLM-Assisted Terraform

    [https://arxiv.org/abs/2606.26590](https://arxiv.org/abs/2606.26590)

    本文提出TerraProbe框架，通过五层Oracle评估揭示LLM辅助Terraform安全修复中仅消除静态告警的欺骗性，实际完整修复率远低于表面指标。

    

    arXiv:2606.26590v1 公告类型：新论文 摘要：Terraform基础设施即代码中的安全配置错误是云部署中日益增长的风险，大型语言模型正越来越多地被用作自动修复代理。现有评估通常将目标静态分析发现消失视为修复成功，而不检查规划有效性、行为变化或安全意图。本文提出了TerraProbe，一个用于评估LLM辅助Terraform安全修复的五层Oracle框架。我们将TerraProbe应用于由gemini-2.5-flash-lite、GPT-4o和Claude 3.5 Sonnet生成的288个首次修复，这些修复覆盖了68个真实世界的TerraDS模块和28个受控注入缺陷模块。结果显示，针对性的Checkov移除夸大了修复成功率。尽管主要模型的针对性移除率达到83.3%，但全扫描器清洁度降至10.4%，Terraform规划成功率为39.6%，而规划比较可达率为38%。

    arXiv:2606.26590v1 Announce Type: new  Abstract: Security misconfigurations in Terraform Infrastructure-as-Code are a growing risk in cloud deployments, and large language models are increasingly used as automated repair agents. Existing evaluations often treat a repair as successful when the targeted static-analysis finding disappears, without checking planning validity, behavioral change, or security intent. This paper presents TerraProbe, a five-layer oracle framework for evaluating LLM-assisted Terraform security repair. We apply TerraProbe to 288 first-pass repairs generated by gemini-2.5-flash-lite, GPT-4o, and Claude 3.5 Sonnet across 68 real-world TerraDS modules and 28 controlled injected-defect modules. The results show that targeted Checkov removal overstates repair success. Although targeted removal reaches 83.3 percent for the primary model, full-scanner cleanliness drops to 10.4 percent, Terraform planning succeeds for 39.6 percent, and plan comparison is reachable for 38
    
[^89]: SharQ：融合激活稀疏性与FP4量化以提升大语言模型推理效率

    SharQ: Bridging Activation Sparsity and FP4 Quantization for LLM Inference

    [https://arxiv.org/abs/2606.26587](https://arxiv.org/abs/2606.26587)

    提出SharQ方法，通过在线稀疏-稠密分解与残差补偿机制，有效结合了激活稀疏性与FP4量化，在无需训练的情况下降低了大语言模型推理中的激活压缩损失。

    

    arXiv:2606.26587v1 公告类型：交叉 摘要：现代加速器日益支持低位浮点格式与半结构化稀疏性，但将二者结合用于大语言模型激活压缩仍面临挑战：激活值包含输入相关的异常值，这些异常值主导了FP4量化中的块缩放因子；而直接应用N:M稀疏掩码会丢弃中等大小的激活值，导致稀疏化损失与量化误差相互耦合。为此，我们提出SharQ——一种无需训练的推理方法，通过在线稀疏-稠密分解来融合激活稀疏性与FP4量化。对于每个激活张量，SharQ生成输入自适应的N:M掩码，提取出以异常值主导的稀疏主干，将其量化为FP4，并定义一个相对于量化后稀疏主干（而非未量化的稀疏值）的稠密残差。一个稀疏FP4通用矩阵乘处理主干，而一个稠密FP4通用矩阵乘则同时补偿掩码导致的激活损失与稀疏路径的量化误差。

    arXiv:2606.26587v1 Announce Type: cross  Abstract: Low-bit floating-point formats and semi-structured sparsity are increasingly supported by modern accelerators, yet combining them for LLM activation compression remains challenging: activations contain input-dependent outliers that dominate block scales in FP4 quantization, and directly applying N:M sparsity masks discards moderate values, coupling sparsification loss with quantization error. We introduce SharQ, a training-free inference method that bridges activation sparsity and FP4 quantization through an online sparse--dense decomposition. For each activation tensor, SharQ generates an input-adaptive N:M mask to extract an outlier-dominated sparse backbone, quantizes it to FP4, and defines a dense residual relative to the quantized sparse backbone rather than the unquantized sparse values. A sparse FP4 GEMM processes the backbone while a dense FP4 GEMM compensates for both mask-induced activation loss and sparse-path quantization e
    
[^90]: 重新审视复杂动作空间中的动作分解方法

    Revisiting Action Factorization for Complex Action Spaces

    [https://arxiv.org/abs/2606.26574](https://arxiv.org/abs/2606.26574)

    本文对六种动作分解方法在多种算法上的表现进行了系统性横截面比较研究，揭示了现有复杂动作空间基准测试的局限性。

    

    许多现实世界的控制问题涉及混合离散-连续动作空间。例如，自动驾驶中的转向和信号控制，以及机器人或电子游戏中的瞄准和射击。尽管现实世界中的混合分解和强化学习框架（如Gymnasium、PettingZoo、TorchRL、SeedRL、Mujoco等）支持复杂动作空间，但这些框架中的默认环境通常实现统一的动作空间配置（如LunarLander、Walker2D、Cheetah、SMAC、SUMO、Ant、Atari）。标志性的混合动作基准（如RoboCup 2D HFO、SC2LE、Platform、CARLA等）大多是来自论文的重型或存档实现，这些论文仅在一种控制类型上测试一种或少数几种竞争性分解方法。本文对三类算法中的每一类进行了分解方法[独立网络、共享编码器、VDN、QPLEX、联合、自回归]的横截面研究。

    arXiv:2606.26574v1 Announce Type: new  Abstract: Many real-world control problems involve hybrid discrete-continuous action spaces. For example, steering and signaling in autonomous driving, and aiming and firing in robotics or video-games. Despite real-world hybrid factorization and reinforcement learning framework support for complex action spaces (e.g., Gymnasium, PettingZoo, TorchRL, SeedRL, Mujoco, etc), the default environments within those frameworks often implement uniform action space configurations (LunarLander, Walker2D, Cheetah, SMAC, SUMO, Ant, Atari). Landmark hybrid-action benchmarks (RoboCup 2D HFO, SC2LE, Platform, CARLA, etc) are mostly heavyweight or archival implementations originating from papers which test one or a small number of competing factorization methods on one kind of control. This article provides a cross-sectional study of factorization methods [independent networks, shared encoder, VDN, QPLEX, Joint, Auto-Regressive] on each of three families of algori
    
[^91]: 可解释的基于集成学习的机器学习模型用于检测丙型肝炎患者的肝硬化存在情况

    Explainable Ensemble-Based Machine Learning Models for Detecting the Presence of Cirrhosis in Hepatitis C Patients

    [https://arxiv.org/abs/2606.26561](https://arxiv.org/abs/2606.26561)

    本研究首次利用可解释的集成学习机器学习模型检测丙型肝炎患者的肝硬化，填补了该领域的研究空白。

    

    丙型肝炎是一种由病毒引起的肝脏感染，会导致轻度至重度的肝脏炎症。多年来，丙型肝炎会逐渐损害肝脏，通常导致永久性疤痕，即肝硬化。患者在发展为肝硬化之前，有时数十年间仅有中度或无症状的肝脏疾病。肝硬化通常会恶化至肝功能衰竭。肝硬化患者还可能出现脑和神经系统损伤以及胃肠道出血。肝硬化的治疗重点是防止疾病进一步进展。因此，早期检测肝硬化对于避免并发症至关重要。机器学习已被证明能够为多种疾病的诊断提供精确和准确的信息。尽管如此，目前尚无研究使用机器学习来检测丙型肝炎患者的肝硬化。本研究获取了一个包含28个特征的数据集。

    arXiv:2606.26561v1 Announce Type: new  Abstract: Hepatitis C is a liver infection caused by a virus, which results in mild to severe inflammation of the liver. Over many years, hepatitis C gradually damages the liver, often leading to permanent scarring, known as cirrhosis. Patients sometimes have moderate or no symptoms of liver illness for decades before developing cirrhosis. Cirrhosis typically worsens to the point of liver failure. Patients with cirrhosis may also experience brain and nerve system damage, as well as gastrointestinal hemorrhage. Treatment for cirrhosis focuses on preventing further progression of the disease. Detecting cirrhosis earlier is therefore crucial for avoiding complications. Machine learning (ML) has been shown to be effective at providing precise and accurate information for use in diagnosing several diseases. Despite this, no studies have so far used ML to detect cirrhosis in patients with hepatitis C. This study obtained a dataset consisting of 28 attri
    
[^92]: PMDformer：用于长期预测的补丁均值解耦信息变换器

    PMDformer: Patch-Mean Decoupling Information Transformer for Long-term Forecasting

    [https://arxiv.org/abs/2606.26549](https://arxiv.org/abs/2606.26549)

    本文提出PMDformer，通过补丁均值解耦分离趋势与形状信息，并设计趋势恢复注意力和近端变量注意力，以解决长期时间序列预测中因尺度差异导致的形状相似性建模难题。

    

    长期时间序列预测（LTSF）在能源管理、金融和交通预测等领域发挥着关键作用。基于变换器的模型采用了补丁策略来捕捉长程依赖关系，但由于尺度差异，准确建模不同补丁和变量之间的形状相似性仍然具有挑战性。为了解决这一问题，我们引入了补丁均值解耦（PMD），该方法通过减去每个补丁的均值来分离趋势和残差形状信息，从而保留原始结构并确保注意力机制能够捕捉到真实的形状相似性。此外，为了更有效地建模长程依赖关系并捕捉跨变量关系，我们提出了趋势恢复注意力（TRA）和近端变量注意力（PVA）。前者模块在计算注意力输出的同时重新整合了PMD解耦出的趋势，而后者则将跨变量注意力聚焦于近端变量。

    arXiv:2606.26549v1 Announce Type: new  Abstract: Long-term time series forecasting (LTSF) plays a crucial role in fields such as energy management, finance, and traffic prediction. Transformer-based models have adopted patch-based strategies to capture long-range dependencies, but accurately modeling shape similarities across patches and variables remains challenging due to scale differences. To address this, we introduce patch-mean decoupling (PMD), which separates the trend and residual shape information by subtracting the mean of each patch, preserving the original structure and ensuring that the attention mechanism captures true shape similarities. Futhermore, to more effectively model long-range dependencies and capture cross-variable relationships, we propose Trend Restoration Attention (TRA) and Proximal Variable Attention (PVA). The former module reintegrates the decoupled trend from PMD while calculating attention output. And the latter focuses cross-variable attention on the 
    
[^93]: 大型语言模型能否可靠地编码定性人道主义数据？一项针对人类专家裁决的基准研究

    Can Large Language Models Reliably Code Qualitative Humanitarian Data? A Benchmark Study Against Human Expert Adjudication

    [https://arxiv.org/abs/2606.26541](https://arxiv.org/abs/2606.26541)

    该研究首次通过基准测试证明，多个大型语言模型在编码定性人道主义数据时能达到接近人类专家的可靠性，尤其适用于大规模需求分析。

    

    摘要：来自受影响人群的数据对于指导人道主义响应至关重要，但其价值取决于对细微需求叙述的及时且一致的解释。人道主义组织往往缺乏大规模分析这些数据所需的人员、时间和专业专长。大型语言模型（LLMs）可能扩展这一能力，但其在编码定性人道主义数据方面的可靠性尚未得到直接验证。这项基准研究使用150份高保真合成人道主义文本，将46个LLMs与人类黄金标准进行了比较。评估结合了Krippendorff's alpha的评分者间信度测试、区分正确、接近正确和错误编码的差异分析，以及针对人道主义特定标准（包括歧视、复杂需求层次和非标准沟通风格）的定性评估。作者发现，多个LLMs能够执行演绎编码。

    arXiv:2606.26541v1 Announce Type: new  Abstract: Data from affected populations are crucial for informing humanitarian response, but their value depends on timely and consistent interpretation of nuanced accounts of need. Humanitarian organizations often lack the staff, time, and specialist expertise required to analyze this information at scale. Large language models (LLMs) may expand this capacity, but their reliability for coding qualitative humanitarian data has not been directly established. This benchmark study compares 46 LLMs to a human Gold Standard using 150 high-fidelity synthetic humanitarian transcripts. Evaluation combined inter-rater reliability testing with Krippendorff's alpha, discrepancy analysis distinguishing correct, near-correct, and incorrect codes, and qualitative assessment across humanitarian-specific criteria including discrimination, complex needs hierarchies, and non-standard communication styles. The authors find that multiple LLMs can perform deductive c
    
[^94]: CascadeFormer：受梯度扇入不对称性启发的深度锥形Transformer

    CascadeFormer: Depth-Tapered Transformers Motivated by Gradient Fan-in Asymmetry

    [https://arxiv.org/abs/2606.26538](https://arxiv.org/abs/2606.26538)

    本文通过提出梯度扇入不对称性理论，解释了深度Transformer中深层贡献小的现象，并设计了两种高效方法：CascadeFormer通过深度锥形化宽度提升效率，CascadeFlow Pruning利用累积梯度剪枝冗余层，在保持性能的同时显著降低延迟并提高吞吐量。

    

    arXiv:2606.26538v1 公告类型：交叉 摘要：深度Transformer由均匀堆叠的残差块组成，但其最深层的贡献往往微乎其微。我们提出了两种利用这种不对称性的高效方法。CascadeFormer根据深度锥形化宽度，以匹配各层间不均匀的信息流，在相同训练预算下达到与均匀基线相当的语言困惑度，同时将延迟降低8.6%，吞吐量提升9.4%。CascadeFlow剪枝利用累积训练梯度移除层，无需事后分析。它在语言困惑度和秩稳定性上优于标准启发式方法，并在下游准确率上保持竞争力。为启发这些方法，我们提出梯度扇入不对称性（GFA），作为对深层贡献较小的结构性解释。在Pre-LayerNorm残差堆栈中，某一层的梯度是恒等路径与所有下游功能路径之和，产生随深度线性衰减的梯度扇入。

    arXiv:2606.26538v1 Announce Type: cross  Abstract: Deep Transformers are composed of uniformly stacked residual blocks, yet their deepest layers often add little value. We present two efficiency methods that exploit this asymmetry. CascadeFormer tapers width with depth to match the uneven information flow across layers, achieving comparable perplexity to a uniform baseline at the same training budget while reducing latency by 8.6% and increasing throughput by 9.4%. CascadeFlow Pruning removes layers using accumulated training gradients, with no post hoc analysis. It outperforms standard heuristics on perplexity and rank-stability and stays competitive on downstream accuracy. To motivate these methods, we propose Gradient Fan-in Asymmetry (GFA) as a structural account of why deeper layers contribute less. In Pre-LayerNorm residual stacks, the gradient at a layer is the sum of an identity path and all downstream functional paths, producing a gradient fan-in that decays linearly with dept
    
[^95]: 基于自适应奖励塑造与策略比重新加权策略的高效样本迁移强化学习

    Sample-efficient Transfer Reinforcement Learning via Adaptive Reward Shaping and Policy-Ratio Reweighting Strategy

    [https://arxiv.org/abs/2606.26527](https://arxiv.org/abs/2606.26527)

    本文提出了一种通过自适应奖励塑造和策略比重新加权策略解决迁移分布偏移与安全探索冲突的自主车道变换强化学习框架。

    

    arXiv:2606.26527v1 公告类型：新 摘要：迁移学习通过重用源任务的知识来提高策略学习效率，为安全高效的自主高速公路车道变换决策提供了一种可行的范式。现有方法经常遭遇由源域和目标域之间分布偏移引起的迁移不匹配，导致训练振荡和性能下降。此外，目标域适应依赖于探索性交互，这在安全关键的车道变换场景中难以保证训练安全性。为解决这些限制，本文提出了一种用于自主高速公路车道变换的安全迁移强化学习框架。首先，我们设计了一种基于瞬时安全成本的自适应教师干预机制，以抑制风险探索并逐渐减弱干预强度，并对混合行为策略的回报边界进行了理论分析。这种干预还产生了双源……

    arXiv:2606.26527v1 Announce Type: new  Abstract: Transfer learning improves policy learning efficiency by reusing knowledge from source tasks, providing a feasible paradigm for safe and efficient autonomous highway lane changing decision-making. Existing methods frequently encounter transfer mismatch induced by distribution shifts between source and target domains, leading to training oscillation and performance decline. Besides, target domain adaptation depends on exploratory interactions, which struggles to guarantee training safety in safety-critical lane changing cases. To tackle these limitations, this paper proposes a safe transfer reinforcement learning framework for autonomous highway lane changing. First, we design an adaptive teacher intervention mechanism based on instantaneous safety cost to restrain risky exploration and fade intervention strength progressively, with theoretical analysis on return bounds for mixed behavior policy. This intervention also produces dual-sourc
    
[^96]: 面向计算机科学逻辑的理论级自动形式化

    Theory-Scale Auto-Formalization of Logics for Computer Science

    [https://arxiv.org/abs/2606.26525](https://arxiv.org/abs/2606.26525)

    本文提出了LCS-Bench，一个基于《计算机科学逻辑》的理论级自动形式化基准，通过半自动化智能体流水线构建，实现了大规模相互依赖逻辑理论的连贯形式化翻译。

    

    arXiv:2606.26525v1 公告类型：新 摘要：自动形式化对于可扩展的形式化验证至关重要，但现有进展主要集中在孤立语句上，而理论级自动形式化——即连贯地翻译数百个相互依赖的定义、引理和定理——由于在一致性、忠实性、可扩展性和正确性方面面临挑战，仍然是一个未解决的问题。在本文中，我们介绍了LCS-Bench，这是一个基于《计算机科学逻辑》的独立理论级基准。LCS-Bench通过一种新颖的半自动化智能体流水线构建，该流水线利用概念图、形式化签名规划、问题追踪、带反例搜索的“sorry”填充，并辅以人类专家的忠实性审查。最终成果涵盖327个教科书条目、超过4,076个Lean声明以及超过85,000行Lean代码。该数据集通过一个数据引擎支持广泛的评估，该引擎自动衍生出五个评估基准轨道。

    arXiv:2606.26525v1 Announce Type: new  Abstract: Auto-formalization is critical for scalable formal verification, but existing progress largely focuses on isolated statements, while theory-scale auto-formalization, which coherently translates hundreds of interdependent definitions, lemmas, and theorems, remains open due to challenges in consistency, faithfulness, scalability, and correctness. In this paper, we introduce LCS-Bench, a stand-alone, theory-scale benchmark based on Logics for Computer Science. LCS-Bench is built through a novel semi-automated agentic pipeline that leverages concept graphs, formal signature planning, issue tracking, sorry-filling with counter-example search, complemented by faithfulness review from human experts. The resulting artifact covers 327 textbook items, over 4,076 Lean declarations, and more than 85K lines of Lean code. The dataset supports broad evaluation through a data engine that automatically derives five tracks of evaluation benchmarks, measur
    
[^97]: 激进AI可解释性

    Radical AI Interpretability

    [https://arxiv.org/abs/2606.26523](https://arxiv.org/abs/2606.26523)

    本书提出一个结合激进解释哲学与机械可解释性的框架，为从AI系统内部读取信念和欲望提供了可验证的标准，强调归因必须整体进行而非零散操作。

    

    我们借鉴激进解释的哲学传统和机械可解释性的工具，开发了一个将AI系统解释为智能体的框架。核心问题是：给定一个系统的计算事实，我们如何求解其信念、欲望和意义？这一问题对于安全性日益重要。我们希望信任所部署的系统，无论是通过理解其目标，还是更谦逊地通过可靠地检测欺骗行为。可解释性研究者正在构建从模型内部读取信念和欲望的工具，但目前尚无关于此类工具何时成功的公认标准。本书提供了这一标准。我们提出了表征主义方法和解释主义方法的标准，并将它们与当前可解释性方法能够执行的测试联系起来。一个核心教训是，这些归因不能零散地进行。信念、欲望以及它们所预设的命题结构。

    arXiv:2606.26523v1 Announce Type: new  Abstract: We develop a framework for interpreting AI systems as agents, drawing on the philosophical tradition of radical interpretation and the tools of mechanistic interpretability. The core question is: given the computational facts about a system, how do we solve for its beliefs, desires, and meanings? This matters increasingly for safety. We want to be able to trust the systems we deploy, whether by understanding their goals or, more modestly, by reliably detecting deception. Interpretability researchers are building tools to read beliefs and desires off a model's internals, but there is no settled account of when such a tool has succeeded. This book supplies one. We propose criteria on both representationalist and interpretationist approaches, and tie each to tests current interpretability methods can carry out. A central lesson is that these attributions cannot be made piecemeal. Beliefs, desires, and the propositional structure they presup
    
[^98]: 基于多路径自适应门控瓶颈潜在常微分方程与拉曼数据融合的细胞培养过程预测方法

    Multipath Adaptive Gated Bottleneck Latent ODE with Raman Data Fusion for Cell Culture Process Forecasting

    [https://arxiv.org/abs/2606.26520](https://arxiv.org/abs/2606.26520)

    提出了一种结合门控瓶颈潜在常微分方程与多路径即时微调的自适应框架，通过变量级门控和掩码感知瓶颈机制有效处理高维稀疏数据，实现了对细胞培养过程的多日早期预测。

    

    哺乳动物细胞培养过程是许多生物制药生产的基础，但保持过程稳定运行十分困难：关键工艺参数会随时间漂移，而偏离规格的趋势往往在确认时已为时过晚，无法及时干预。早期的多日预测能够实现对补料、采样和控制的及时调整，但生物过程预测面临诸多挑战：测量数据稀疏且采样时间不规则，不同细胞系和培养基的操作条件存在异质性，且初始行为几乎相同的运行过程可能走向截然不同的未来结果。为此，我们提出了一种自适应框架，将门控瓶颈潜在常微分方程与多路径即时微调相结合。门控瓶颈潜在常微分方程通过可学习的变量级门控和掩码感知瓶颈机制对标准潜在常微分方程进行增强，能够压缩高维稀疏输入，从而在有限数据条件下提升学习效果。

    arXiv:2606.26520v1 Announce Type: cross  Abstract: Mammalian cell-culture processes underpin the manufacture of many biopharmaceuticals, yet keeping a run on track is hard: critical process parameters drift over days, and an off-specification trend is often confirmed too late to intervene. Early-stage, multi-day forecasts could enable timely adjustment of feeding, sampling, and control, but bioprocess forecasting is challenging because measurements are sparse and irregularly sampled, operating conditions are heterogeneous across cell lines and media, and runs with near-identical early behaviour can diverge into different futures. We propose an adaptive framework combining a Gated Bottleneck Latent Ordinary Differential Equation (GB-Latent ODE) with Multi-Path Just-In-Time Fine Tuning (MP-JIT-FT). The GB-Latent ODE augments the stan dard Latent ODE with learnable variable-wise gating and a mask-aware bottleneck that compress high-dimensional sparse inputs, improving learning under limit
    
[^99]: 检索记忆中的时间有效性：为AI智能体消除演化知识中的过时事实错误

    Temporal Validity in Retrieval Memory: Eliminating Stale-Fact Errors for AI Agents over Evolving Knowledge

    [https://arxiv.org/abs/2606.26511](https://arxiv.org/abs/2606.26511)

    MemStrata通过双时间账本和确定性取代规则，在不依赖相似度阈值或大语言模型的情况下，解决了RAG系统中因知识演化导致的过时事实检索问题。

    

    检索增强生成（RAG）使智能体能够访问积累的知识，但缺乏时间模型。当某个事实发生变化时（例如函数重命名或API重构），RAG会同时检索到过时值和当前值，且两者嵌入相似度几乎相同。此时智能体要么放弃响应，要么提供已被取代的事实。我们证明这是一个结构性问题：在校准数据集上，余弦相似度区分矛盾事实与重复事实的AUROC仅为0.59（接近随机水平），因为矛盾事实与原始事实的嵌入相似度往往高于改写后的重复事实。我们提出了MemStrata——一种维护时间有效性的检索记忆。它像RAG一样存储事实并保持静态召回能力，但当某个事实的值被矛盾时，通过确定性（主语、关系、宾语）取代规则在双时间账本中淘汰过时值——无需相似度阈值，也无需调用大语言模型。在六个本地运行的基准测试中，该方法均表现出色。

    arXiv:2606.26511v1 Announce Type: cross  Abstract: Retrieval-augmented generation (RAG) gives agents access to accumulated knowledge, but has no model of time. When a fact changes (e.g., a function is renamed or API restructured), RAG retrieves both the stale and current value with near-identical embedding similarity. The agent then either abstains or serves the superseded fact. We show this is a structural problem: on a calibrated dataset, cosine similarity distinguishes a contradicted fact from a duplicated one with AUROC 0.59 (near chance), as contradictions are often more embedding-similar to the original than rephrased duplicates.   We present MemStrata, a retrieval memory maintaining temporal validity. It stores facts like RAG, preserving static recall, but when a fact's value is contradicted, a deterministic (subject, relation, object) supersession rule retires the stale value in a bi-temporal ledger - with no similarity threshold and no LLM call. Across six benchmarks run local
    
[^100]: 平均场PhiBE：基于离散时间数据的连续时间平均场强化学习

    Mean-Field PhiBE: Continuous-Time Mean-Field Reinforcement Learning from Discrete-Time Data

    [https://arxiv.org/abs/2606.26498](https://arxiv.org/abs/2606.26498)

    本文提出MF-PhiBE，通过将离散时间数据融入Wasserstein空间上的连续时间偏微分方程，解决了连续时间平均场控制中模型不可识别性的难题。

    

    本文研究了在群体动态根据未知的McKean-Vlasov随机微分方程连续演化，但仅有离散时间转移数据可用的情况下的无模型连续时间平均场控制问题。在基于模型的公式中，策略评估自然由$\mathcal P_2(\mathbb R^d)$上的稳态Hamilton-Jacobi-Bellman方程描述，但该方程涉及受控McKean-Vlasov动态的漂移和扩散系数，而这些系数在仅有离散时间数据时无法识别。另一方面，直接简化为时间离散的Bellman方程虽避免了不可识别性问题，但失去了微分方程结构。为桥接这两种视角，我们引入了平均场PhiBE（MF-PhiBE），它将离散时间转移信息整合到Wasserstein空间上的连续时间偏微分方程中。

    arXiv:2606.26498v1 Announce Type: cross  Abstract: This paper addresses model-free continuous-time mean-field control in a setting where the population dynamics evolve continuously according to an unknown McKean-Vlasov stochastic differential equation, while only discrete-time transition data are available. In the model-based formulation, policy evaluation is naturally described by a stationary Hamilton-Jacobi-Bellman equation on $\mathcal P_2(\mathbb R^d)$, but this equation involves the drift and diffusion coefficients of the controlled McKean-Vlasov dynamics, which are not identifiable when only discrete-time data are available. On the other hand, a direct reduction to a time-discrete Bellman equation avoids the non-identifiability issue but loses the differential equation structure. To bridge these two viewpoints, we introduce a Mean-Field-PhiBE (MF-PhiBE), which incorporates discrete-time transition information into a continuous-time PDE on the Wasserstein space. The MF-PhiBE repl
    
[^101]: 基于严格恰当评分规则学习概率滤波器

    Learning Probabilistic Filters with Strictly Proper Scoring Rules

    [https://arxiv.org/abs/2606.26497](https://arxiv.org/abs/2606.26497)

    本文提出PSEF方法，利用严格恰当评分规则训练基于Transformer的置换不变映射，仅通过合成数据实现贝叶斯滤波分布的逼近。

    

    针对部分观测且含噪声的动态系统的贝叶斯滤波，旨在在线推断系统状态随观测演变的条件分布。该贝叶斯滤波分布是不确定性量化的自然对象，但很少能作为监督学习目标直接获得。然而，我们通常可以利用预测模型生成合成系统轨迹及合成观测数据。本文提出了恰当评分集成滤波器（PSEF），这是一种基于训练分析映射的集成数据同化方法，仅通过合成状态-观测轨迹来逼近滤波分布。分析步骤被表示为一种基于置换不变性、Transformer架构的映射，它接收预测集成和观测作为输入，生成分析集成。训练基于严格恰当的评分规则——其中使用了能量评分。

    arXiv:2606.26497v1 Announce Type: new  Abstract: Bayesian filtering of partially and noisily observed dynamical systems seeks to infer the evolving conditional distribution of the state of a dynamical system, given observations, in an online fashion. This Bayesian filtering distribution is the natural object for uncertainty quantification, but it is rarely available as a supervised learning target. However, one can often use the forecast model to generate synthetic system trajectories, along with synthetic observations. We introduce the proper scoring ensemble filter (PSEF), an ensemble data assimilation method based on training an analysis map to approximate the filtering distribution using only synthetic state--observation trajectories. The analysis step is represented as a permutation-invariant, transformer-based map that takes as input a forecast ensemble and observations, producing an analysis ensemble. Training is based on strictly proper scoring rules -- with the energy score us
    
[^102]: 深度学习程序故障诊断中的评估策略差距

    Evaluation-Strategy Gap in Fault Diagnosis of Deep Learning Programs

    [https://arxiv.org/abs/2606.26492](https://arxiv.org/abs/2606.26492)

    本研究揭示了深度学习程序故障诊断中程序内评估与跨程序评估之间存在显著性能差距（平衡准确率差距达0.190），并发现该差距主要源于特征中的程序级结构。

    

    深度学习程序在训练过程中可能因多种原因失败，诊断故障原因是一项成本高昂且耗时的维护任务。用于诊断此类故障的技术通常采用程序内交叉验证进行评估，但这可能不足以应对涉及未见过程的部署场景。因此，有必要评估不同场景下的性能差异，并识别现有深度学习故障诊断技术中性能差距的成因。我们利用DynFault数据集（包含38个真实世界深度学习程序的5542个注入故障的训练轨迹）研究了这一差距。我们发现，在程序内评估与保留完整程序评估之间，现有故障诊断技术的平衡准确率存在0.190的差距。我们还发现，这一差距源于特征中的程序级结构，这促使我们研究了两种运行时特征集：曲率特征和优化器特征。

    arXiv:2606.26492v1 Announce Type: cross  Abstract: Deep Learning (DL) programs can fail during training for many reasons, and diagnosing the cause is a costly and time-consuming maintenance task. Techniques for diagnosing such failures are commonly assessed using within-program cross-validation, which may be inadequate for deployment settings involving previously unseen programs. It is therefore necessary to assess how performance differs across these settings and to identify the causes of any performance gap in established fault diagnosis techniques for DL. We investigate this gap using DynFault, a corpus of 5,542 fault-injected training traces from 38 real-world DL programs. We found a gap of 0.190 in balanced accuracy for existing fault diagnosis techniques between within-program evaluation and holding out whole programs. We also found the gap comes from program-level structure in the features, which led us to examine two runtime feature sets, curvature features and optimizer featur
    
[^103]: 当你为边缘设备压缩递归推理器时，什么能幸存下来？

    What Survives When You Compress a Recursive Reasoner for the Edge?

    [https://arxiv.org/abs/2606.26488](https://arxiv.org/abs/2606.26488)

    本研究发现，对递归推理模型进行激进压缩会保留局部预测能力，但会彻底破坏全局推理能力，且这种崩溃是架构性的，无法通过基于令牌的训练目标修复。

    

    arXiv:2606.26488v1 公告类型：新 摘要：递归推理模型通过反复更新潜在状态，仅需几百万个参数就能解决复杂的结构化任务。将这些模型部署在边缘硬件上需要大幅压缩，但与传统的序列模型不同，量化误差是在递归推理循环中累积的，而不是在输出令牌之间累积。因此，关于压缩的标准直觉不再适用。在这项工作中，我们探讨了递归推理器被压缩时什么能够幸存下来。通过在全精度范围扫描、三个任务和两种递归架构上的实验，我们发现：激进的压缩保留了局部预测能力，但破坏了全局推理能力——在朴素的INT4剪枝、蒸馏和线性注意力下，单元准确率保持稳定，但拼图精确准确率却骤降至零。基于令牌的目标（包括量化感知训练）无法修复这一问题。这种崩溃是架构性的——它发生在MLP混合递归上，但不在其他架构上。

    arXiv:2606.26488v1 Announce Type: new  Abstract: Recursive reasoning models can solve complex structured tasks with only a few million parameters by repeatedly updating a latent state. Deploying these models on edge hardware requires significant compression, but unlike conventional sequence models, quantization errors compound across recursive reasoning cycles rather than across output tokens. As a result, standard intuitions about compression fail to apply. In this work, we ask what survives when recursive reasoners are compressed. Across a full precision sweep, three tasks, and two recursive architectures, we find that aggressive compression preserves local prediction but destroys global reasoning: cell accuracy holds while puzzle-exact accuracy collapses to zero under naive INT4 pruning, distillation, and linear attention alike. Token-level objectives, including quantization-aware training, cannot repair it. The collapse is architectural -- it strikes MLP-mixing recursion but not at
    
[^104]: LLM代理中针对提示注入的带外防御的自适应评估

    Adaptive Evaluation of Out-of-Band Defenses Against Prompt Injection in LLM Agents

    [https://arxiv.org/abs/2606.26479](https://arxiv.org/abs/2606.26479)

    本文系统比较了LLM代理中多种带外防御策略，并指出当前所有防御仅通过静态基准验证，未能抵御自适应攻击。

    

    近期研究（2024至2026年）已就防御使用工具的LLM代理免受间接提示注入攻击的策略达成共识：与其训练模型拒绝恶意指令，不如在模型外部通过确定性策略强制实施安全机制来调控代理的行为。CaMeL、FIDES、Progent、RTBAS和FORGE等系统通过能力机制、信息流标签和引用监控器实现了这一策略，其中多个系统报告称在AgentDojo基准测试中几乎完全消除了攻击。我们做出了两项贡献。首先，我们将这些带外防御系统组织为经典完整性保护（Biba模型）、引用监控和最小权限原则的实例，从而对其覆盖与未覆盖的内容进行结构化比较。其次，我们警告称，所有这些系统仅通过静态基准测试（一组固定的注入尝试）进行验证——而正是这种方法论曾使带内防御在自适应、防御感知攻击出现之前看起来十分强大。

    arXiv:2606.26479v1 Announce Type: cross  Abstract: Recent work (2024 to 2026) has converged on a strategy for defending tool-using LLM agents against indirect prompt injection: rather than training the model to refuse malicious instructions, enforce security outside the model with a deterministic policy that mediates the agent's actions. Systems such as CaMeL, FIDES, Progent, RTBAS, and FORGE realize this with capabilities, information-flow labels, and reference monitors, and several report near-elimination of attacks on the AgentDojo benchmark. We make two contributions. First, we organize these out-of-band defenses as instances of classical integrity protection (Biba), reference monitoring, and least privilege, yielding a structured comparison of what they do and do not cover. Second, we warn that every one of them is validated only on static benchmarks (a fixed set of injection attempts), the same methodology that made in-band defenses look strong until adaptive, defense-aware attac
    
[^105]: 检索预热能量基推理：结构化推理任务上扩散即推断的五臂消融方法学

    Retrieval-Warmed Energy-Based Reasoning: A Five-Arm Ablation Methodology for Diffusion-as-Inference on Structured Reasoning Tasks

    [https://arxiv.org/abs/2606.26476](https://arxiv.org/abs/2606.26476)

    本研究提出一种五臂消融方法学，通过分离类别先验偏差、随机预热启动和图对齐值复用三种混淆效应，揭示了检索预热能量基推理在结构化推理任务中的关键增益来源，并在连通性任务上实现了高达35个百分点的平衡准确率提升。

    

    预热启动的扩散采样器加速了迭代推理，但通常不清楚管线的哪一部分带来了这种增益。我们研究了**检索预热能量基推理（RW-EBR）**——一种基于IRED能量扩散模型，并辅以现代Hopfield轨迹记忆的模型——并贡献了一种**五臂消融方法学**（包括 oracle、最佳常数、每查询随机、打乱、对齐），该方法分离了三种混淆效应：类别先验偏差偏移、随机预热启动和图对齐值复用。该诊断分解方法改编自LLM-RAG评估。在**连通性-2**（Erdős–Rényi 全对可达性）任务上，在固定1000图验证集诊断中，对齐与打乱oracle之间的差异达到了**+35个百分点**的平衡准确率，其中值分布和检索机制固定，仅破坏了每图的对齐，而每查询...

    arXiv:2606.26476v1 Announce Type: cross  Abstract: Warm-started diffusion samplers accelerate iterative inference, but it is rarely clear which part of the pipeline carries the gain. We study \textbf{retrieval-warmed energy-based reasoning (RW-EBR)} -- an IRED energy-based diffusion model \cite{du2024ired} augmented with a Modern Hopfield trajectory memory -- and contribute a \textbf{five-arm ablation methodology} (oracle, best-constant, per-query-random, shuffled, aligned) that separates three confounded effects: class-prior bias shift, stochastic warm-starting, and graph-aligned value reuse. The diagnostic decomposition is adapted from LLM-RAG evaluation \cite{ru2024ragchecker}. On \textbf{connectivity-2} (Erd\H{o}s--R\'enyi all-pairs reachability), the aligned-vs-shuffled-oracle swing reaches \textbf{$+35$\,pp} balanced accuracy on a fixed 1{,}000-graph validation-set diagnostic, with value distribution and retrieval mechanics fixed, only per-graph alignment destroyed, while per-que
    
[^106]: 将强化学习诱导的工具使用定位到单一交叉编码器特征上

    Localizing RL-Induced Tool Use to a Single Crosscoder Feature

    [https://arxiv.org/abs/2606.26474](https://arxiv.org/abs/2606.26474)

    本文提出专用特征交叉编码器（DFC），能够分离并定位强化学习引入的单一特征，从而显著提升工具调用的正确性并实现无重训的能力迁移。

    

    arXiv:2606.26474v1 公告类型：交叉 摘要：通过强化学习进行微调重塑了语言模型的内部表示，使其能够实现工具使用等智能体行为，但这些变化的机制基础仍未被充分理解。虽然强化学习显著改善了结构化工具调用的生成，但尚不清楚哪些特征会涌现、哪些特征得以保留，以及识别出的特征是否可用于无重训的行为控制。在这项工作中，我们展示了“专用特征交叉编码器（DFC）”能够分离出一组紧凑的强化学习特定特征，这些特征在$\texttt{Qwen2.5-3B}$模型中中介了工具调用能力。在$48$个交叉编码器超参数扫描中，编码-解码重建使强化学习模型的工具正确性提高了$+31.1 \pm {9.7}$个百分点，并将工具调用能力被动迁移到冻结的基础模型上，提高了$+6.8 \pm 5.0$个百分点，我们称此为“能力溢出”。我们的发现表明，DFC分区能够集中强化学习引入的能力。

    arXiv:2606.26474v1 Announce Type: cross  Abstract: Fine-tuning through RL reshapes the internal representations of language models to enable agentic behaviors such as tool use, yet the mechanistic basis of these changes remains poorly understood. While RL substantially improves structured tool-call generation, it is unclear which features emerge, which are preserved, and whether identified features can be leveraged for retraining-free behavioral control. In this work, we show that $\textit{Dedicated Feature Crosscoders (DFC)}$ isolate a compact set of RL-specific features that mediate tool-calling capability in $\texttt{Qwen2.5-3B}$. Across a $48$-crosscoder hyperparameter sweep, encode-decode reconstruction improves the RL model's tool correctness by $+31.1 \pm {9.7}$ pp and passively transfers tool-calling ability to the frozen base model by $+6.8 \pm 5.0$ pp which we call a $\textit{capability spillover}$. Our findings show that DFC partitioning concentrates RL-introduced capability
    
[^107]: 质量感知的多模态融合何时真正起作用？一种针对决策层面依赖性的泄露安全诊断方法

    When Does Quality-Aware Multimodal Fusion Matter? A Leakage-Safe Diagnostic for Decision-Level Dependence

    [https://arxiv.org/abs/2606.26473](https://arxiv.org/abs/2606.26473)

    本文提出一种泄露安全的诊断方法，通过置换测试样本间的可靠性分数，发现多模态融合中可靠性信息只有在能正确识别模态时才会影响决策，否则仅与性能相关。

    

    许多多模态系统会评估每种模态的可靠性，并根据这些权重来调整它们在最终预测中的贡献。然而，目前尚不清楚这些可靠性分数是否真正影响了模型决策，还是仅仅与性能相关。我们提出了一种简单的诊断方法，用于测试在推理过程中是否使用了可靠性信息。在训练完成后，模型和输入被固定，而可靠性分数则在测试样本之间进行置换。如果预测结果依赖于这些分数，那么性能应该会下降。在用于压力识别的StressID数据集和用于情感分析的CMU-MOSEI数据集上的实验表明，尽管为每个样本选择最佳模态可能带来显著性能提升，但置换可靠性分数后性能并未发生变化。在可靠性信号能正确识别模态的阳性控制实验中，相同的固定融合规则却带来了显著改进，这表明只有当可靠性信号能够正确指示模态时，它们才会影响融合决策。

    arXiv:2606.26473v1 Announce Type: new  Abstract: Many multimodal systems estimate the reliability of each modality and weight their contributions to the final prediction. However, it remains unclear whether these scores influence model decisions or merely correlate with performance. We propose a simple diagnostic to test whether reliability information is used during inference. After training, the model and inputs are fixed while reliability scores are permuted across test examples. If predictions depend on these scores, performance should degrade. Experiments on StressID for stress recognition and CMU-MOSEI for sentiment analysis show that permuting reliability scores leaves performance unchanged despite substantial potential gains from selecting the best modality per example. In positive controls where reliability signals identify the correct modality, the same frozen fusion rules yield significant improvements, indicating that reliability signals influence fused decisions only when 
    
[^108]: 基于顿悟分数的KV缓存淘汰方法：无需注意力矩阵

    Epiphany-Aware KV Cache Eviction Without the Attention Matrix

    [https://arxiv.org/abs/2606.26472](https://arxiv.org/abs/2606.26472)

    本文提出EpiKV，一种通过直接读取模型前向传播中的内部表示变化（顿悟分数）来淘汰KV缓存的方法，无需注意力矩阵，可将可行上下文长度扩展至传统注意力评分方法的16倍，且无需训练或自定义内核。

    

    arXiv:2606.26472v1 公告类型：交叉 摘要：随着推理模型生成长达数万token的思维链，KV缓存日益成为部署瓶颈。现有缓存淘汰方法通过注意力权重对token进行排序，这在长推理轨迹中是一个有噪声的重要性代理，并且通过强制模型实现注意力矩阵，阻碍了生产推理中融合内核的使用。在这项工作中，我们提出了一种称为"顿悟分数"的度量标准来对token进行评分：该分数直接从前向传播中读取模型内部表示的变化，无需注意力矩阵且仅需极少的额外状态。由此产生的缓存淘汰方法EpiKV无需训练、分类器或自定义内核，可直接在FlashAttention推理栈中不变地使用——将可行上下文长度扩展至基于注意力评分方法的16倍。针对上层中间层（负向影响）和下层中间层（正向影响），我们采用因果滚动z分数消除位置趋势。在4096-token缓存设置下，该方法表现出色。

    arXiv:2606.26472v1 Announce Type: cross  Abstract: As reasoning models emit chains of thought tens of thousands of tokens long, KV cache increasingly becomes a deployment bottleneck. Existing cache eviction methods rank tokens by attention weight, which is a noisy importance proxy in long reasoning traces, and prohibits the use of fused kernels in production inference by forcing the model to materialize the attention matrix. In this work, we instead score tokens with a metric we term the epiphany score: the change in the model's internal representation, read directly from the forward pass with no attention matrix and negligible extra state. Our resulting cache eviction method, EpiKV, requires no training, classifier, or custom kernel, and can be used directly in FlashAttention inference stacks unchanged -- scaling to a 16x longer feasible context than attention-based scoring. upper-mid layers negatively) and remove a positional trend with a causal rolling z-score. At a 4096-token cache
    
[^109]: 一种用于结构与结果预测的因果基础模型

    A Causal Foundation Model for Structure and Outcome Prediction

    [https://arxiv.org/abs/2606.26467](https://arxiv.org/abs/2606.26467)

    本文提出了一种基于合成数据训练、能同时预测因果结构与结果并支持多层级因果查询的通用因果基础模型TabPFN-CFM。

    

    我们引入了TabPFN-CFM，这是一个能够处理多种因果问题的因果基础模型。TabPFN-CFM可以从观测数据中预测因果结构和结果，支持Pearl因果层级中所有三个层面的查询，并在已知图结构可用时利用其改进预测。该模型在合成数据集上训练，并能泛化到真实数据集，在结构预测和结果预测基准上均展现出优于基线的性能。

    arXiv:2606.26467v1 Announce Type: new  Abstract: We introduce TabPFN-CFM, a causal foundation model that can handle multiple causal problems. TabPFN-CFM predicts both causal structure and outcomes from observational data, supports queries on all three levels of Pearl's Causal Hierarchy and uses known graph structure when available to improve predictions. TabPFN-CFM is trained on synthetic datasets, and generalises to real datasets, demonstrating improved performance over both structural and outcome prediction baselines.
    
[^110]: 寻找思考的时间：在实时强化学习中学习规划预算

    Finding the Time to Think: Learning Planning Budgets in Real-Time RL

    [https://arxiv.org/abs/2606.26463](https://arxiv.org/abs/2606.26463)

    提出了一种在实时强化学习中，通过轻量级门控策略动态选择状态依赖的规划预算的方法，有效解决了环境持续运行下的决策延迟问题。

    

    深思熟虑需要时间。在实时环境中，这段时间并非免费。标准强化学习（RL）通过让环境无限期等待智能体的决策来回避这一问题。相反，我们研究了实时强化学习环境，在这种环境中，环境在等待智能体行动的同时仍在持续运行。基于先前的实时形式化方法，我们引入了可变延迟实时强化学习，其中智能体在每个决策点自行决定思考多长时间，因为环境在持续演进。对于我们使用的规划智能体而言，正确的延迟是状态依赖的，而单纯地规划“规划多长时间”可能会使智能体陷入瘫痪。我们转而通过在规划器之上训练一个轻量级的门控策略，来选择状态依赖的规划预算。在实时《吃豆人》、《俄罗斯方块》、《贪吃蛇》、《极速六角棋》和《极速围棋》中，我们的门控策略优于固定预算和启发式基线方法，并且能够迁移到环境具有不同实时约束的实时设置中。

    arXiv:2606.26463v1 Announce Type: new  Abstract: Deliberating takes time. In real-time settings, that time is not free. Standard reinforcement learning (RL) sidesteps this as the environment waits indefinitely for the agent's decision. Instead, we study real-time RL environments where the environment progresses while waiting for the agent's action. Building on prior real-time formalizations, we introduce variable-delay real-time RL, where the agent chooses how long to deliberate at each decision point since the environment progresses. For the planning agents we use, the right delay is state-dependent, and naively planning how long to plan can paralyze the agent. We instead approach this setting by training a lightweight gating policy on top of a planner to select state-dependent planning budgets. Across real-time Pac-Man, Tetris, Snake, Speed Hex, and Speed Go, our gating policy outperforms fixed-budget and heuristic baselines, and transfers to a real-time setup where the environment a
    
[^111]: 在线测试时自适应的概率框架

    A probabilistic framework for online test-time adaptation

    [https://arxiv.org/abs/2606.26457](https://arxiv.org/abs/2606.26457)

    提出了一种基于状态空间模型的概率框架，用于在线测试时自适应，以应对训练与测试分布之间的偏移。

    

    本文提出了一种用于在线测试时自适应问题的概率框架。在该问题中，模型基于标注数据进行训练，但必须在测试时适应未标注数据，且假设训练分布与测试分布可能存在差异，即可能存在分布偏移。该框架基于状态空间建模架构，可对参数学习、参数时间演化、先验调优以及预测进行系统刻画。

    arXiv:2606.26457v1 Announce Type: cross  Abstract: This paper presents a probabilistic framework for online test-time adaptation problems. In them, a model is trained on labeled data but must adapt to unlabeled data at test time under the assumption that training and test distributions potentially differ, that is, there might have been a distributional shift. The framework is based on a state-space modelling architecture from which parameter learning, parameter time evolution, prior tuning, and prediction can be characterized.
    
[^112]: 面向RGB-事件视觉目标跟踪的主动对抗扰动驱动联想记忆检索

    Active Adversarial Perturbation-driven Associative Memory Retrieval for RGB-Event Visual Object Tracking

    [https://arxiv.org/abs/2606.26455](https://arxiv.org/abs/2606.26455)

    本文提出APRTrack框架，通过分层对抗扰动模拟现实世界信号退化，并利用联想记忆检索机制，解决了RGB-事件跟踪中因模态失效或目标不完整导致的鲁棒性下降问题。

    

    RGB-事件跟踪通过融合RGB外观纹理和来自事件传感器的密集时间运动线索，提高了定位鲁棒性。虽然这种多模态方案拓宽了跟踪的适用性，但现实世界场景中存在的各种结构化信号退化阻碍了传统的多模态融合。在恶劣环境中，任何一种模态都可能急剧丧失可靠性，并且由于遮挡、边缘截断和前景杂波，目标经常出现不完整的情况。为应对上述挑战，我们提出了一种针对RGB-事件跟踪的分层扰动与检索框架，名为APRTrack，该框架对部分目标缺失和模态退化具有鲁棒性。为了模拟现实世界的信号损坏，APRTrack通过两个对抗扰动分支在模态和空间层面构建结构化退化，分别模拟全模态故障和局部目标区域缺失。

    arXiv:2606.26455v1 Announce Type: cross  Abstract: RGB-Event tracking improves localization robustness by fusing RGB appearance textures and dense temporal motion cues from event sensors. While this multi-modal scheme broadens tracking applicability, real-world scenes suffer diverse structured signal degradations that hinder traditional multi-modal fusion. In harsh environments, either modality can lose reliability drastically, and targets frequently appear incomplete due to occlusion, edge truncation and foreground clutter.To tackle the above challenges, we present a hierarchical perturbation and retrieval framework tailored for RGB-Event tracking with robustness against partial target missing and modal degradation, termed APRTrack. To mimic real-world signal corruption, APRTrack constructs structured degradation via two adversarial perturbation branches at the modality and spatial levels, which separately simulate full-modal failure and localized target region absence. A hierarchical
    
[^113]: 像人类一样优化CUDA：微性能分析工具作为基于LLM的GPU内核优化的专家替代方案

    Optimizing CUDA like a Human: Micro-Profiling Tools as Expert Surrogates for LLM-Based GPU Kernel Optimization

    [https://arxiv.org/abs/2606.26453](https://arxiv.org/abs/2606.26453)

    提出KernelPro闭环多智能体系统，通过将专家启发式编码为可插拔微性能分析工具，结合多层级分析器反馈和领域自适应蒙特卡洛树搜索，实现GPU内核代码的自动生成与迭代优化。

    

    arXiv:2606.26453v1 公告类型：新 摘要：我们提出了KernelPro，一个闭环多智能体系统，通过将大语言模型代码生成与硬件分析器反馈及可插拔的瓶颈检测工具相结合，自动生成、分析并迭代优化GPU内核代码。KernelPro贡献了四个方面：（1）一个语义反馈算子，将专家启发式编码为可插拔的微性能分析工具，将原始硬件指标转化为可操作的自然语言指导；（2）一个两阶段工具调用架构，其中基于屋顶线的瓶颈分类筛选出哪些专门分析工具执行，结合内核级（ncu）、指令级（SASS）和系统级（nsys）分析；（3）一个领域自适应的蒙特卡洛树搜索，具备渐进扩展、非对称分支、对数奖励校准、死胡同剪枝和用于跨迭代学习的搜索记忆；（4）通过自主协作直接生成CuTe源代码。

    arXiv:2606.26453v1 Announce Type: new  Abstract: We present KernelPro, a closed-loop multi-agent system that automatically generates, profiles, and iteratively optimizes GPU kernel code by integrating large language model (LLM) code generation with hardware profiler feedback and pluggable bottleneck detection tools. KernelPro introduces four contributions: (1) a semantic feedback operator that encodes expert heuristics as pluggable micro-profiling tools, transforming raw hardware metrics into actionable natural language guidance; (2) a two-stage tool invocation architecture where roofline-based bottleneck classification filters which specialized analysis tools execute, combining kernel-level (ncu), instruction-level (SASS), and system-level (nsys) profiling; (3) a domain-adapted MCTS with progressive widening, asymmetric branching, log-reward calibration, dead-end pruning, and search memory for cross-iteration learning; and (4) direct CuTe source-level code generation via autonomous co
    
[^114]: 像法官一样倾听：一个面向自动歌唱表演评估的音乐感知框架

    Listening Like a Judge: A Music-Aware Framework for Automatic Singing Performance Evaluation

    [https://arxiv.org/abs/2606.26451](https://arxiv.org/abs/2606.26451)

    本文提出MusicJudge框架，通过多模态对齐分析歌词准确性和音高节奏保真度，并引入模态引导的LoRA微调改进歌唱转录，实现了与人类专家高度一致的自动歌唱质量评估。

    

    arXiv:2606.26451v1 公告类型：交叉 摘要：自动歌唱质量评估（SQA）需要评估歌词准确性和音乐保真度，同时处理表现性的变化。然而，现有系统主要依赖声学线索或歌词转录中的一种，限制了整体表现评估。此外，由于在花腔、颤音和节奏弹性中稳健的歌唱转录面临挑战，它们的整合并非易事。为此，我们提出了MusicJudge，一个用于自动SQA的模态引导框架，通过将歌词准确性与音高-节奏保真度耦合，进行块对齐的多模态分析。它利用多信号匹配检测语义上有意义的歌词块，该匹配整合了语义嵌入、词汇相似性和语音对齐。为了改进歌唱音频转录，我们引入了模态引导的LoRA用于ASR微调。跨数据集实验表明，该方法与人类专家判断具有高度一致性。

    arXiv:2606.26451v1 Announce Type: cross  Abstract: Automatic singing quality assessment (SQA) requires evaluating lyrical correctness and musical fidelity while handling expressive variations. However, existing systems largely rely on either acoustic cues or lyric transcriptions exclusively, limiting holistic performance evaluation. Furthermore, their integration is non-trivial due to challenges in robust singing transcription amid melisma, vibrato, and tempo elasticity. To this end, we propose MusicJudge, a modality-guided framework for automated SQA that performs block-aligned multimodal analysis by coupling lyric correctness with pitch-rhythm fidelity. It detects semantically meaningful lyric blocks using multi-signal matching that integrates semantic embeddings, lexical similarity, and phonetic alignment. To improve singing audio transcription, we introduce Modality-Guided LoRA for ASR fine-tuning. Experiments across datasets demonstrate strong agreement with human expert judgments
    
[^115]: 具有结构保证的离散选择模型中嵌入基础模型预测

    Embedding Foundation Model Predictions in Discrete-Choice Models with Structural Guarantees

    [https://arxiv.org/abs/2606.26432](https://arxiv.org/abs/2606.26432)

    本文提出一种两阶段适配器方法，通过将基础模型预测嵌入多项Logit模型，在保留结构保证的同时修正经济逻辑冲突，使时间价值等指标成为数学保证。

    

    arXiv:2606.26432v1 公告类型：新 摘要：表格基础模型在选择预测任务上表现出较高的准确性，但其预测结果常常违背这些任务所需的经济逻辑：提高价格可能导致预测需求增加，隐含的支付意愿估计值经常为负或不合理，且不可用的替代方案获得非零概率。我们提出了一种两阶段适配器，将基础模型预测的选择概率作为预计算特征，并嵌入到多项Logit的效用函数中。在第一阶段，我们通过带符号约束的最大似然估计拟合多项Logit的结构系数；在第二阶段，我们冻结这些系数，并拟合一个基于基础模型预测的小型神经网络修正项。我们证明这种组合精确保留了多项Logit的边际替代率，因此可分析计算的时间价值成为数学保证而非经验准确性。

    arXiv:2606.26432v1 Announce Type: new  Abstract: Tabular foundation models achieve strong accuracy on choice prediction tasks, but their predictions often violate the economic logic those tasks require: raising a price can increase predicted demand, implied willingness-to-pay estimates are frequently negative or implausible, and unavailable alternatives receive nonzero probability. We propose a two-stage adapter that takes a foundation model's predicted choice probabilities as a precomputed feature and embeds them inside a multinomial logit's utility. In Stage 1, we fit the multinomial logit's structural coefficients by maximum likelihood with sign constraints; in Stage 2, we freeze those coefficients and fit a small neural correction operating on the foundation model's predictions. We prove that this composition exactly preserves the multinomial logit's marginal rate of substitution, so analytically computable value-of-time becomes a mathematical guarantee rather than an empirical acc
    
[^116]: DualEval：统一大语言模型评估的联合模型-项目校准框架

    DualEval: Joint Model-Item Calibration for Unified LLM Evaluation

    [https://arxiv.org/abs/2606.26429](https://arxiv.org/abs/2606.26429)

    DualEval通过联合校准模型能力和项目特征，在统一框架中融合静态基准与偏好数据，实现了更可靠、均衡的大语言模型评估，并支持基准压缩和异常检测等应用。

    

    当前大语言模型评估依赖于两种互补但往往脱节的信号：具有客观正确性标签的静态基准测试，以及更能反映开放式用户交互的竞技场式偏好数据。我们提出了DualEval，这是一种潜在模型-项目校准框架，将模型和评估项目表示在一个共享空间中，联合估计模型能力以及项目难度和区分度。我们将DualEval应用于四个领域：编程、数学、多领域知识任务和通用日常用户查询。我们的评估使用了18个前沿大语言模型、静态基准标签以及奖励模型分数（这些分数针对保留的、用于开放式模型响应的人类偏好进行了验证）。实验表明，我们的框架能产生可靠且均衡的模型排名，其学习到的项目级特征支持下游应用，如用于样本高效评估的基准压缩和用于数据污染的异常检测。

    arXiv:2606.26429v1 Announce Type: cross  Abstract: Current LLM evaluation relies on two complementary but often disconnected signals: static benchmarks with objective correctness labels and arena-style preference data that better reflect open-ended user interactions. We introduce DualEval, a latent model-item calibration framework that represents models and evaluation items in a shared space, jointly estimating model ability together with item difficulty and sharpness. We apply DualEval across four domains: coding, math, miscellaneous domain-knowledge tasks, and generic everyday user queries. Our evaluation uses 18 frontier LLMs, static benchmark labels, and reward-model scores validated against held-out human preferences for open-ended model responses. Empirically, our framework produces reliable and balanced model rankings, and its learned item-level profiles support downstream applications such as benchmark compression for sample-efficient evaluation and anomaly detection for contam
    
[^117]: 重新思考预测中的训练与推理：将赢者通吃连接回高斯混合模型

    Rethinking Training & Inference for Forecasting: Linking Winner-Take-All back to GMMs

    [https://arxiv.org/abs/2606.26424](https://arxiv.org/abs/2606.26424)

    本文揭示了轨迹预测中赢者通吃训练导致模式概率无信息的问题，并提出了两种事后处理方法（后验加权合并和一步EM更新）来改善模式分布。

    

    arXiv:2606.26424v1 公告类型：新 摘要：自动驾驶的轨迹预测已经取得了快速进展，但代表性模型通常在预测模式上产生无信息的后验分布，导致模式剪枝出现问题。我们将此归因于建模与训练的不匹配：预测器通常被建模为条件高斯混合模型（GMMs），但使用赢者通吃（WTA）损失进行训练，该损失将每个样本分配给它最近的模式。我们认为，这种类似K均值的硬分配（独热编码）虽然防止了模式坍缩，却是无信息模式概率的根源：它过度分割了轨迹空间，忽略了邻近模式之间的相关性，并在微小扰动下导致分配不稳定。基于这一视角，我们引入了两种事后处理方法：（1）测试时后验加权合并，用于聚合邻近的候选轨迹；（2）一步期望最大化（EM）更新，用软责任替换硬标签。

    arXiv:2606.26424v1 Announce Type: new  Abstract: Trajectory forecasting for autonomous driving has advanced rapidly, yet representative models often produce uninformative posteriors over forecast modes, causing problems for mode pruning. We trace this to a modeling-training mismatch: forecasters are typically modeled as conditional Gaussian mixture models (GMMs) but trained with a winner-take-all (WTA) loss that assigns each sample to its nearest mode. We argue that this K-means-like hard assignment (one-hot), while preventing mode collapse, is the source of uninformative mode probabilities: it over-segments the trajectory space, ignores relatedness among nearby modes, and yields assignment instability under small perturbations. Guided by this lens, we introduce two post-hoc treatments: (1) test-time posterior-weighted merging that aggregates nearby candidate trajectories; and (2) a one-step expectation-maximization (EM) update that replaces hard labels with soft responsibilities, shar
    
[^118]: 水獭天气：高技巧且计算高效的中期天气预报

    Otter Weather: Skillful and Computationally Efficient Medium-Range Weather Forecasting

    [https://arxiv.org/abs/2606.26421](https://arxiv.org/abs/2606.26421)

    本文提出Otter Weather模型，通过高效时空架构实现中期天气预报，在1.5°分辨率下以极低计算成本（<3.5 A100天）超越传统NWP 9.6%，效率比轻量AI模型高2倍，比资源密集型模型低100倍，推动了天气预测的民主化。

    

    arXiv:2606.26421v1 公告类型：新 摘要：最先进的中期人工智能天气模型可以超越传统的数值天气预报（NWP），但需要巨大的训练预算。这限制了对资源不足群体的使用，并严重限制了模型的快速迭代。在这里，我们开发了Otter Weather，这是一个高效的时空预测模型，旨在通过人工智能实现高性能天气预测的民主化。在1.5°分辨率下的ERA5再分析数据上，使用标准WeatherBench协议进行评估，Otter系列显著推进了技巧-计算帕累托前沿。确定性版本在24小时预报时效上比最佳NWP基线高出9.6%，同时训练所需的A100天数少于3.5天。与轻量级人工智能模型相比，它提供了2倍的效率提升，与资源密集型前沿架构相比，计算量减少了100倍。我们通过Continuo训练将这些效率优势扩展到概率预测中。

    arXiv:2606.26421v1 Announce Type: new  Abstract: State-of-the-art medium-range AI weather models can outperform traditional Numerical Weather Prediction (NWP) but require massive training budgets. This restricts usage for under-resourced groups and severely limits fast model iteration. Here we develop Otter Weather, a highly efficient spatiotemporal forecasting model designed to democratise high-performance weather prediction with AI. Evaluated on ERA5 reanalysis data at 1.5{\deg} resolution using standard WeatherBench protocols, the Otter family significantly advances the skill-compute Pareto frontier. The deterministic version outperforms the best NWP baseline by 9.6% at a 24-hour lead time while requiring fewer than 3.5 A100-days for training. It provides a 2x efficiency gain over lightweight AI models and a 100-fold reduction in compute compared to resource-intensive frontier architectures. We extend these efficiency gains into probabilistic forecasting by training via the Continuo
    
[^119]: 基于格理论的无偏规范集值预言机

    Unbiased Canonical Set-Valued Oracles Via Lattice Theory

    [https://arxiv.org/abs/2606.26418](https://arxiv.org/abs/2606.26418)

    本文通过Knaster–Tarski不动点定理，在完备格框架下提出了一种规范的非平凡credal集，解决了自指预言机在无偏和自洽约束下的唯一性问题。

    

    非智能体“预言机”AI在估计未来事件概率时面临自指问题：一旦其答案被学习并采取行动，就会改变它被要求报告的概率本身。针对科学家AI计划所倡导的一种回应是只询问反事实问题，并假设答案没有影响进行评估。我们观察到，这类答案一旦被学习就会变得无关紧要，恰恰是因为其前提随后变为假。因此，我们探索了一种自指替代方案：预言机报告的不是单一概率，而是一个同时无偏且与学习后果自洽的credal集。朴素的自洽性要求被太多集合满足（包括无用的答案[0,1]），因此问题在于挑选出一个规范的、非平凡的成员。我们通过闭包完备格上的Knaster–Tarski不动点定理实现了这一点。

    arXiv:2606.26418v1 Announce Type: new  Abstract: A non-agentic "oracle" AI that estimates probabilities of future events faces a self-reference problem: once its answer is learned and acted upon, it can change the very probability it was asked to report. One response, advocated for the Scientist AI programme, is to ask only counterfactual questions, evaluated as if the answer had no influence. We observe that such answers tend to become irrelevant the moment they are learned, precisely because their premise is then false. We therefore explore a self-referential alternative in which the oracle reports not a single probability but a credal set that is simultaneously unbiased and self-consistent with the consequences of being learned. The naive self-consistency requirement is satisfied by too many sets (including the useless answer $[0,1]$), so the problem is to single out a canonical, nontrivial member. We do so with the Knaster--Tarski fixed-point theorem on the complete lattice of clos
    
[^120]: 超越前馈网络：作为下一代通用人工智能主体性与内在安全性基础的再入神经架构

    Beyond Feedforward Networks: Reentry Neural Systems as the Fundamental Basis of Subjecthood and Intrinsic Safety of Next-Generation AGI

    [https://arxiv.org/abs/2606.26406](https://arxiv.org/abs/2606.26406)

    提出了一种基于闭环再入回路的新型AGI架构，通过结构性循环和自维持放大在数学上保证自我意识与内在安全性，并设计了可高效计算的S度量替代传统综合信息度量。

    

    我们提出了一种基于闭环再入回路（D <-> I循环）的完全架构蓝图，用于构建安全的人工通用智能。与前馈网络（有向无环图，C=0，S=0，无法实现自我指涉）不同，所提出的架构包含一个结构性循环（C ≥ 1）并具有自维持放大能力（ρ > 1），从而在数学上保证了自我模型、工具性自我保存以及非编程目标导向行为的涌现。智能体的目标以非文本的D-向量形式编码在架构本身中，使其免于被重新解释或遭受提示注入攻击。我们提出了S度量——一种多项式时间可计算[O(N^3)]的替代Tononi的NP难Phi度量的方法，并提供了机器验证的Lean 4证明，表明S>0意味着正的综合信息。本工作提供了完整的Python/NumPy实现（基于Tarjan的循环复杂度、Delta-S屏障），以及工业级的水平扩展方案。

    arXiv:2606.26406v1 Announce Type: cross  Abstract: We propose a complete architectural blueprint for safe artificial general intelligence based on a closed reentry loop (D <-> I cycle). In contrast to feedforward networks, which are directed acyclic graphs (C=0, S=0) incapable of self-reference, the proposed architecture contains a structural cycle (C >= 1) with self-sustaining amplification (rho > 1), mathematically guaranteeing the emergence of a self-model, instrumental self-preservation, and unprogrammed goal-directed behaviour. The agent's goals are encoded as a non-textual D-vector in the architecture itself, making them immune to reinterpretation and prompt injection. We present the S-measure -- a polynomial-time [O(N^3)] computable alternative to Tononi's NP-hard Phi -- with machine-verified Lean 4 proof that S>0 implies positive integrated information. The work provides full Python/NumPy implementations (Tarjan-based cycle complexity, Delta-S barrier), industrial horizontal sc
    
[^121]: 面向组合几何极值问题的几何感知蒙特卡洛树搜索

    Geometry-Aware MCTS for Extremal Problems in Combinatorial Geometry

    [https://arxiv.org/abs/2606.26399](https://arxiv.org/abs/2606.26399)

    提出一种几何感知的MCTS框架，通过增量更新动作空间和利用几何对称性剪枝，将组合几何极值问题的约束检查复杂度从O(n³)降至O(n²)，并有效处理稀疏奖励和二次令牌消耗问题。

    

    我们研究了组合几何中的某些极值问题，这些问题询问在n×n网格中满足严格全局几何约束的点配置。经典精确求解器会因组合爆炸而难以处理这类问题，而标准强化学习和基于Transformer的模型则难以应对稀疏奖励的“有效性悬崖”和二次令牌消耗限制。为克服这些瓶颈，我们提出了一种几何感知的蒙特卡洛树搜索（MCTS）框架。我们的方法通过对可行动作空间进行增量更新来严格强制几何约束。对于共线点集合的约束（如经典的无三点共线问题（Max-N3IL）中的约束），该机制将约束检查复杂度从O(n³)降低到O(n²)。为提高搜索效率，我们以两种方式利用几何对称性：节点展开时的规范剪枝。

    arXiv:2606.26399v1 Announce Type: new  Abstract: We study certain extremal problems in combinatorial geometry that ask about configurations of points in an $n \times n$ grid that satisfy strict, global geometric constraints. Classical exact solvers suffer from combinatorial explosion for these types of problems, and standard reinforcement learning and transformer-based models struggle with the sparse reward "validity cliff" and quadratic token-consumption limits. To overcome these bottlenecks, we propose a Geometry-Aware Monte Carlo Tree Search (MCTS) framework. Our approach strictly enforces geometric constraints through incremental updates to the feasible action space. For constraints about collections of collinear points, like those that occur in the classic No-Three-in-Line problem (Max-N3IL), this mechanism reduces the constraint checking complexity from $O(n^3)$ to $O(n^2)$. To improve search efficiency, we exploit geometric symmetries in two ways: canonical pruning during node e
    
[^122]: 面向多目标强化学习的确定性帕累托最优策略综合

    Deterministic Pareto-Optimal Policy Synthesis for Multi-Objective Reinforcement Learning

    [https://arxiv.org/abs/2606.26397](https://arxiv.org/abs/2606.26397)

    本文提出了一种基于切比雪夫标量化的偏好条件贝尔曼算子，能够为多目标马尔可夫决策过程计算并收敛到确定性帕累托最优策略覆盖集。

    

    现实世界中的决策制定常常需要平衡多个相互冲突的目标，而标准强化学习通常通过将奖励聚合为单一标量信号来处理这一挑战。虽然这种方法对简单任务有效，但它往往无法捕捉到最优权衡的完整谱系，即帕累托前沿。在本文中，我们引入了一种新颖的偏好条件贝尔曼算子，其动机源于切比雪夫标量化，旨在为多目标马尔可夫决策过程计算确定性帕累托最优策略。我们证明该算子满足一个包络性质，即估计的价值函数在帕累托前沿之上形成一个上界，并证明它单调收敛到该前沿的一个覆盖集。此外，我们还展示了如何从这些收敛的Q估计中提取确定性策略。这确保了智能体能够恢复一个策略。

    arXiv:2606.26397v1 Announce Type: cross  Abstract: Real-world decision-making often requires balancing multiple conflicting objectives, a challenge that standard Reinforcement Learning (RL) frequently addresses by aggregating rewards into a single scalar signal. While effective for simple tasks, this approach often fails to capture the full spectrum of optimal trade-offs, known as the Pareto frontier. In this paper, we introduce a novel preference-conditioned Bellman operator, motivated from the Chebyshev scalarization, designed to compute deterministic Pareto-optimal policies for Multi-Objective Markov Decision Processes (MOMDPs). We prove that this operator satisfies an enveloping property, where the estimated value functions upper-bound the true Pareto frontier, and demonstrate that it monotonically converges to a coverage set of this frontier. Furthermore, we also show how to extract deterministic policies from these converged Q-estimates. This ensures the agent can recover a polic
    
[^123]: 理解边缘：稀疏自编码器追踪Transformer泛化的极限

    At the Edge of Understanding: Sparse Autoencoders Trace The Limits of Transformer Generalization

    [https://arxiv.org/abs/2606.26396](https://arxiv.org/abs/2606.26396)

    本文通过稀疏自编码器揭示了Transformer在分布外输入下内部虚假概念增加的现象，并提出了一种基于机制的微调策略来增强模型鲁棒性。

    

    预训练Transformer展现了卓越的泛化能力，有时甚至能超越其训练数据的范围。然而，现实世界的部署常常面临与训练数据分布不同的意外或对抗性数据。如果没有明确的机制来处理这种分布变化，模型的可靠性和安全性会下降，这促使我们对Transformer的分布外（OOD）场景进行更系统的研究。通过系统实验，我们提出了一个机制性框架，用于描绘Transformer模型鲁棒性的精确轮廓。我们发现，OOD输入（包括微小的拼写错误和越狱提示）会驱动语言模型在其内部使用更多虚假概念。我们利用这一工具来量化和理解提示中的分布变化程度，从而提出一种基于机制的精调策略，以增强LLM的鲁棒性。我们将OOD的概念从传统的独立同分布（i.i.d.）设置扩展到更广泛的场景。

    arXiv:2606.26396v1 Announce Type: new  Abstract: Pre-trained transformers have demonstrated remarkable generalization abilities, at times extending beyond the scope of their training data. Yet, real-world deployments often face unexpected or adversarial data that diverges from training data distributions. Without explicit mechanisms for handling such shifts, model reliability and safety degrade, urging more disciplined study of out-of-distribution (OOD) settings for transformers. By systematic experiments, we present a mechanistic framework for delineating the precise contours of transformer model robustness. We find that OOD inputs, including subtle typos and jailbreak prompts, drive language models to operate on an increased number of fallacious concepts in their internals. We leverage this device to quantify and understand the degree of distributional shift in prompts, enabling a mechanistically grounded fine-tuning strategy to robustify LLMs. Expanding the very notion of OOD from i
    
[^124]: 保持警觉：通过多模态大语言模型中的反事实视觉对齐缓解视觉惰性

    Staying VIGILant: Mitigating Visual Laziness via Counterfactual Visual Alignment in MLLMs

    [https://arxiv.org/abs/2606.26387](https://arxiv.org/abs/2606.26387)

    VIGIL通过强化学习后训练框架，引入反事实视觉对齐来缓解多模态大语言模型中的视觉惰性，从而减少幻觉并增强视觉信息在响应中的因果作用。

    

    arXiv:2606.26387v1 公告类型：交叉  摘要：多模态大语言模型（MLLMs）通过引入视觉感知能力扩展了大型语言模型（LLMs），使其能够对图像和文本进行联合推理。尽管继承了LLMs强大的推理能力，它们仍然容易产生与视觉输入相矛盾的幻觉。机制研究表明，这一弱点源于“视觉惰性”：MLLMs内部编码了正确的视觉证据，但在生成响应时过度依赖强大的语言先验知识。现有的对齐方法（如直接偏好优化）主要基于文本优化结果层面的奖励，这引入了对语言捷径的优化偏差，导致生成的响应常常与视觉证据相矛盾。为解决这一问题，我们提出了“对齐中的视觉信息增益”（VIGIL），这是一种基于强化学习（RL）的后训练框架，将重点从数值奖励拟合转向因果视觉基础。VIGIL引入了

    arXiv:2606.26387v1 Announce Type: cross  Abstract: Multimodal large language models (MLLMs) extend large language models (LLMs) with visual perception, enabling joint reasoning over images and text. Despite inheriting strong reasoning capabilities from LLMs, they remain prone to hallucinations that contradict their visual inputs. Mechanistic studies indicate that this weakness stems from visual laziness: MLLMs encode the correct visual evidence internally, but overly rely on strong language priors during response. Existing alignment methods, such as direct preference optimization, primarily optimize outcome-level rewards based on text. This introduces an optimization bias toward linguistic shortcuts, leading to responses that often contradict the visual evidence. To address this, we propose Visual Information Gain In aLignment (VIGIL), a reinforcement-learning (RL) post-training framework that shifts the focus from numerical reward fitting to causal visual grounding. VIGIL introduces a
    
[^125]: SOLAR：人工智能驱动的光速性能分析

    SOLAR: AI-Powered Speed-of-Light Performance Analysis

    [https://arxiv.org/abs/2606.26383](https://arxiv.org/abs/2606.26383)

    本文提出了SOLAR框架，能够自动从PyTorch和JAX代码中推导出经过验证的深度学习模型理论最小执行时间（光速界限），通过结合大语言模型和确定性分析，实现了自动化的性能极限分析。

    

    一个深度学习模型在目标硬件上能运行多快？当前的实现距离该极限还有多远？这些问题对于软件、硬件和算法优化至关重要。光速分析通过计算给定架构上工作负载的理论最小执行时间来回答这些问题。然而，推导光速界限仍然依赖手动操作，容易出错，并且与快速模型开发脱节。为弥补这一差距，我们引入了SOLAR，这是一个能够从PyTorch和JAX源代码自动推导出经过验证的光速界限的框架。SOLAR在其流程中结合了生成式组件和确定性组件：一个LLM前端将任何源程序翻译成可执行的仿射循环中间表示，并通过输出比较进行验证；一个确定性流程将中间表示提升为爱因斯坦求和图；一个分析后端计算未融合、融合和缓存感知的光速界限。SOLAR提供了全面的算子和语言覆盖。

    arXiv:2606.26383v1 Announce Type: cross  Abstract: How fast could a deep-learning model run on target hardware, and how far is today's implementation from that limit? These questions are central to software, hardware, and algorithm optimizations. Speed-of-Light (SOL) analysis answers them by computing a workload's theoretical minimum execution time on a given architecture. Yet deriving SOL bounds remains manual, error-prone, and disconnected from rapid model development. To close this gap, we introduce SOLAR, a framework that automatically derives validated SOL bounds from PyTorch and JAX source code. SOLAR leverages both generative and deterministic components in its flow: an LLM frontend translates any source programs into an executable Affine Loop IR, validated by output comparison; a deterministic flow lifts the IR into an einsum graph; and an analytical backend computes unfused, fused, and cache-aware SOL bounds. SOLAR provides comprehensive operator and language coverage, produce
    
[^126]: 评分是不够的：解决排序中效用-公平性权衡的差距

    Scoring Is Not Enough: Addressing Gaps in Utility-fairness Trade-offs for Ranking

    [https://arxiv.org/abs/2606.26369](https://arxiv.org/abs/2606.26369)

    本文揭示了评分函数在实现效用与公平性的所有权衡中本质上是次优的，并通过反例证明了这一缺陷在多种场景下普遍存在。

    

    arXiv:2606.26369v1 公告类型：交叉 摘要：评分函数用于表示单个文档的相关性。在现代信息检索或推荐系统中，这些函数通常从数据中学习，并在对文档或物品集合进行排序时发挥关键作用，以最大化对查询或用户的效用。随着最近对算法公平性的关注，评分的成功自然催生了同时权衡公平性和效用的评分学习方法。在这项工作中，我们表明，与以效用为中心的目标形成鲜明对比的是，评分在实现所有效用-公平性权衡方面是次优的。我们通过一系列反例和一个通用公平性公式来确立这一点。我们表明，无论我们使用确定性评分函数还是随机评分函数，无论我们是在单个查询范围内还是跨多个查询衡量公平性，这个问题都持续存在。从积极方面来看，我们通过实验证明了半贪婪方法...

    arXiv:2606.26369v1 Announce Type: cross  Abstract: Scoring functions are used to represent the relevance of individual documents. In modern information retrieval or recommendation systems, they are often learned from data and play a pivotal role in ranking sets of documents or items in a way that maximizes utility to a query or user. With the recent interest in algorithmic fairness, the success of scoring has naturally led to methods that learn scores that simultaneously trade off fairness and utility. In this work, we show that in stark contrast with utility-centric objectives, scoring is sub-optimal in achieving all utility-fairness trade-offs. We establish this with a series of counter-examples with a generic fairness formulation. We show that the issue persists whether we have a deterministic scoring function or a randomized one, or whether we measure fairness at the scope of a single query or across multiple queries. On the positive side, we empirically demonstrate that semi-greed
    
[^127]: 极光模型能否编码大气结构？潜在状态分析与归因研究

    Does Aurora Encode Atmospheric Structure? Latent Regime Analysis and Attribution

    [https://arxiv.org/abs/2606.26361](https://arxiv.org/abs/2606.26361)

    极光模型在无需显式指导的情况下，通过潜在空间按季节周期组织，并学会了气象连贯性与三维垂直结构，且风暴事件未形成独立聚类。

    

    arXiv:2606.26361v1 公告类型：新论文 摘要：机器学习基础模型能够准确且高效地模拟大气动力学，但其运作方式如同不透明的“黑箱”。我们利用空间池化主成分分析和逐层相关性传播技术，研究了极光模型的内部表征。研究发现，极光模型的潜在空间主要按季节周期组织，而极端风暴事件并未形成线性可分的聚类。逐层相关性传播分析表明，模型关注的特征与1987年大风暴的三维垂直结构一致。扰动测试显示，遮蔽相关区域对预测的破坏程度是随机遮蔽的3.31倍。这些发现表明，极光模型在无需显式指导的情况下，学会了气象连贯性与垂直结构。

    arXiv:2606.26361v1 Announce Type: new  Abstract: ML foundation models are able to emulate atmospheric dynamics accurately and efficiently but operate as opaque ``black boxes''. We investigate the internal representations of the Aurora model using spatially pooled PCA and layer-wise relevance propagation (LRP). We find evidence that Aurora's latent space is primarily organized by seasonal cycles, whereas extreme storm events do not form a linearly separable cluster. LRP indicates that the model attends to features consistent with the 3D vertical structure of the Great Storm of 1987. Perturbation tests show masking relevant regions degrades forecasts $3.31\times$ more than random masking. These findings suggest that Aurora learns meteorological coherence and vertical structure without explicit instruction.
    
[^128]: OpenFinGym：一个用于评估量化交易智能体的可验证多任务Gym环境

    OpenFinGym: A Verifiable Multi-Task Gym Environment for Evaluating Quant Agents

    [https://arxiv.org/abs/2606.26350](https://arxiv.org/abs/2606.26350)

    OpenFinGym提出了一个统一的多任务Gym环境，通过覆盖预测、市场生成、实时交易和欺诈检测等关键任务，并配备自动化任务构建流程，解决了现有平台因单任务评估而导致的量化金融智能体能力误判和泛化性缺失问题。

    

    arXiv:2606.26350v1 公告类型：新论文 摘要：尽管大语言模型智能体越来越多地被应用于量化金融工作流程中，但它们的评估仍然分散在孤立的任务上，同时基准任务与金融的相关性常常被忽视。然而，金融工作流程本质上是多阶段的，涵盖了诸如预测、策略构建、风险管理和交易等相互依赖的任务。现有的平台通常只关注单一任务，因此可能夸大智能体的能力，无法揭示其在泛化能力、真实市场交互以及具有金融意义的决策方面的弱点。我们提出了OpenFinGym，这是一个统一的量化金融智能体开发Gym环境，它在单一的执行和验证接口下涵盖了预测、市场生成、实时交易和欺诈检测。OpenFinGym还提供了一个自动化的任务构建流程，能够将量化金融出版物转化为可执行的基准任务。

    arXiv:2606.26350v1 Announce Type: new  Abstract: Although large language model agents are increasingly applied to quantitative-finance workflows, their evaluation remains fragmented across isolated tasks, while the financial relevance of benchmark tasks is often overlooked. Yet financial workflows are inherently multi-stage, spanning interdependent tasks such as forecasting, strategy construction, risk management, and trading. Existing platforms typically focus on a single task, and can therefore overstate agent competence and fail to reveal weaknesses in generalization, real-market interaction, and financially meaningful decision-making. We introduce OpenFinGym, a unified gym environment for quantitative-finance agent development that covers forecasting, market generation, real-time trading, and fraud detection under a single execution and verification interface. OpenFinGym additionally provides an automated task-construction pipeline that turns quantitative finance publications into 
    
[^129]: EMA-FS：通过增益信息特征筛选加速GBDT训练

    EMA-FS: Accelerating GBDT Training via Gain-Informed Feature Screening

    [https://arxiv.org/abs/2606.26337](https://arxiv.org/abs/2606.26337)

    提出了一种基于指数移动平均增益信息的特征筛选方法，在保持与LightGBM核心算法完全兼容的前提下，通过动态保留高增益特征来加速直方图构建，从而显著提升GBDT训练效率。

    

    arXiv:2606.26337v1 公告类型：新 摘要：梯度提升决策树（GBDT），以LightGBM为例，其训练时间的绝大部分——通常为65%-70%——都花费在为每个特征构建直方图上。现有的方法，如随机特征子采样（feature_fraction），会丢弃特征而不考虑其预测效用。我们提出了基于EMA的特征筛选（EMA-FS），这是一种算法级优化方法，它在提升迭代过程中维护每个特征分裂增益的指数移动平均（EMA），并在短暂的预热期后，将直方图构建限制在按历史增益排序的前K个特征上。与随机子采样不同，EMA-FS是有信息依据的：它保留高增益特征，同时筛选掉低增益特征。该方法在每棵树级别上运行，完全兼容LightGBM的直方图减法技巧，无需修改核心例程。我们在金融欺诈检测、广告点击率等多个数据集上评估了EMA-FS。

    arXiv:2606.26337v1 Announce Type: new  Abstract: Gradient Boosted Decision Trees (GBDT), exemplified by LightGBM, spend a dominant fraction of training time -- typically 65-70% -- constructing per-feature histograms. Existing approaches such as random feature subsampling (feature_fraction) discard features without regard for their predictive utility. We propose EMA-based Feature Screening (EMA-FS), an algorithm-level optimization that maintains an exponential moving average (EMA) of per-feature split gains across boosting iterations and, after a short warmup, restricts histogram construction to the top-K features ranked by historical gain. Unlike random subsampling, EMA-FS is informed: it retains high-gain features while screening out low-gain ones. Operating at the per-tree level, it preserves full compatibility with LightGBM's histogram subtraction trick, requiring no changes to core routines.   We evaluate EMA-FS on datasets spanning financial fraud detection, advertising click-thro
    
[^130]: Mesh-RL：耦合子网格强化学习

    Mesh-RL: Coupled subgrid reinforcement learning

    [https://arxiv.org/abs/2606.26333](https://arxiv.org/abs/2606.26333)

    Mesh-RL通过将环境分解为重叠子网格并强制执行边界一致的时间差分更新，在不修改奖励函数或引入规划机制的情况下，显著加速了大规模稀疏奖励强化学习中的长程信用分配与收敛速度。

    

    arXiv:2606.26333v1 公告类型：新论文  摘要：在大规模或稀疏奖励环境中，强化学习面临时间差分奖励传播缓慢的问题，因为价值信息仅在状态空间中局部扩散。我们提出了Mesh-RL，一种受有限元方法和区域分解理论启发的空间区域分解框架，该框架将环境划分为重叠的子网格，并强制执行边界一致的时间差分更新。这种方法在确保全局一致的价值传播的同时，实现了局部化学习。与分层或基于模型的方法不同，Mesh-RL无需修改奖励函数、贝尔曼算子或引入显式规划机制，即可加速长程信用分配。我们在具有不同几何形状和网格分辨率的危险密集网格世界环境中评估了Mesh-RL。在Q学习、SARSA和Dyna-Q算法中，Mesh-RL持续提升了收敛速度、累积奖励。

    arXiv:2606.26333v1 Announce Type: new  Abstract: Reinforcement learning in large or sparse-reward environments suffers from slow temporal-difference reward propagation, as value information spreads only locally across the state space. We propose Mesh-RL, a spatial domain-decomposition framework inspired by the finite element method and domain decomposition theory, which partitions the environment into overlapping subgrids and enforces boundary-consistent temporal-difference updates. Such an approach enables localized learning while ensuring globally coherent value propagation. Unlike hierarchical or model-based approaches, Mesh-RL accelerates long-range credit assignment without modifying the reward function, Bellman operator, or introducing explicit planning mechanisms. We evaluate Mesh-RL on hazard-dense grid-world environments with varying geometries and mesh resolutions. Across Q-learning, SARSA, and Dyna-Q, Mesh-RL consistently improves convergence speed, cumulative reward, and le
    
[^131]: EVOM：用于强化学习的行动者-评论家架构的智能体元进化

    EVOM: Agentic Meta-Evolution of Actor-Critic Architectures for Reinforcement Learning

    [https://arxiv.org/abs/2606.26327](https://arxiv.org/abs/2606.26327)

    EVOM提出了一种基于大语言模型的智能体元进化框架，通过双层优化实现行动者-评论家架构的自动化搜索，显著提升了强化学习性能。

    

    在行动者-评论家强化学习中，网络架构通常需要手动设计。自动化这一设计颇具挑战性，因为每个候选架构在评估前都必须经过训练，且设计空间是开放式的。为应对这些挑战，我们引入了EVOM，这是一个用于发现高性能行动者-评论家架构的智能体元进化框架。我们将架构搜索构建为双层优化：内层循环通过低保真度的近端策略优化（PPO）训练权重，而外层循环则通过迭代优化架构程序来驱动元进化。关键在于，外层循环由一个基于大语言模型（LLM）的设计智能体驱动，该智能体纯粹作为架构设计者运作，与策略执行和环境控制完全解耦。实验表明，EVOM的性能优于手动设计的基线、LLM引导的随机搜索以及最先进的LLM引导的程序化策略搜索方法。

    arXiv:2606.26327v1 Announce Type: cross  Abstract: In actor-critic reinforcement learning, network architectures are typically manually designed. Automating this design is challenging because each candidate must be trained before evaluation, and the design space is open-ended. To address these challenges, we introduce EVOM, an agentic meta-evolution framework for discovering high-performance actor-critic architectures. We frame architecture search as a bi-level optimization: an inner loop trains weights via the low-fidelity proximal policy optimization (PPO), while an outer loop drives meta-evolution by iteratively refining architecture programs. Crucially, this outer loop is powered by an LLM-based design agent that operates purely as an architecture designer, completely decoupled from policy execution and environment control. Experiments reveal that EVOM outperforms the manually designed baseline, an LLM-guided random search, and the state-of-the-art LLM-guided programmatic policy se
    
[^132]: 具有马尔可夫噪声的高概率PL-SGD：最优混合与尾部依赖性

    High-Probability PL-SGD with Markovian Noise: Optimal Mixing and Tail Dependence

    [https://arxiv.org/abs/2606.26316](https://arxiv.org/abs/2606.26316)

    本文通过滞后阻塞论证，将马尔可夫噪声下PL-SGD的高概率界从$\widetilde{O}(t_{mix}^2/k)$优化到$\widetilde{O}(t_{mix}/k)$，并证明该线性依赖是最优的，同时扩展到了重尾情况。

    

    我们研究在梯度样本由外生马尔可夫链生成时，满足Polyak-Łojasiewicz（PL）条件的光滑目标的一阶方法。在轻尾设置下，标准随机梯度下降（SGD）在标准增长包络下的先验一致时间高概率界为$\widetilde{O}(t_{mix}^2/k)$，与期望界的$\widetilde{O}(t_{mix}/k)$存在差距。我们通过使用滞后阻塞论证，在几何混合条件下建立了具有主导随机项$\widetilde{O}(t_{mix}/(k+K_0))$的一致高概率保证，从而弥合了这一差距。我们通过在一个由持久两状态链驱动的二次目标上推导匹配的下界$\Omega(\sigma^2 t_{mix}/k)$，证明了这种对混合时间的线性依赖是最优的。然后，我们将该框架扩展到满足平稳有限$p$矩条件（$p \in (1,2]$）的重尾马尔可夫梯度。

    arXiv:2606.26316v1 Announce Type: new  Abstract: We study first-order methods for smooth objectives satisfying the Polyak-\L{}ojasiewicz (PL) condition when gradient samples are generated by an exogenous Markov chain. In the light-tailed setting, prior uniform-in-time high-probability bounds for ordinary Stochastic Gradient Descent (SGD) under a standard growth envelope scale as $\widetilde{O}(t_{mix}^2/k)$, leaving a gap with the $\widetilde{O}(t_{mix}/k)$ expectation bounds. We close this gap using a lag-blocking argument to establish a uniform high-probability guarantee with a leading stochastic term of $\widetilde{O}(t_{mix}/(k+K_0))$ under geometric mixing. We prove this linear dependence on the mixing time is optimal via a matching $\Omega(\sigma^2 t_{mix}/k)$ lower bound on a quadratic objective driven by a persistent two-state chain.   We then extend this framework to heavy-tailed Markovian gradients satisfying a stationary finite-$p$-moment condition, $p \in (1,2]$. We design 
    
[^133]: 面向量子机器学习的定制嵌入方法

    Tailor Made Embeddings for Quantum Machine Learning

    [https://arxiv.org/abs/2606.26312](https://arxiv.org/abs/2606.26312)

    本文提出了一种变分自编码器框架，用于学习针对特定任务的经典数据量子嵌入，能够将高维数据集压缩到13量子比特表示中，在MNIST任务上达到98.5%的验证准确率，显著优于传统量子嵌入方法。

    

    arXiv:2606.26312v1 公告类型：交叉 摘要：自编码器通过解决维度灾难问题，实现了原则性的权重初始化并学习紧凑、结构化的表示，从而改变了经典机器学习。在这项工作中，我们将这一范式扩展到量子机器学习，引入了一个变分自编码器框架，该框架学习针对特定任务的经典数据量子嵌入。我们证明了包括ImageNet在内的高维数据集可以被压缩到13量子比特的量子表示中，同时通过一个学习的解码器仍然可以重构。在MNIST（3 vs 5）任务上，我们的方法使用基于电路的量子分类器达到了98.5%的验证准确率，与经典神经网络基线（99.7%）相差不到1.2个百分点，并且比朴素振幅嵌入方法高出超过30个百分点。与需要完整量子态层析成像才能恢复的振幅嵌入或通常依赖于循环编码的角度嵌入不同，我们的方法提供了更高效和实用的解决方案。

    arXiv:2606.26312v1 Announce Type: cross  Abstract: Autoencoders transformed classical machine learning by solving the curse of dimensionality, enabling principled weight initialization and learning compact, structured representations. In this work, we extend this paradigm to quantum machine learning by introducing a variational autoencoder framework that learns task-specific quantum embeddings of classical data. We demonstrate that high-dimensional datasets, including ImageNet, can be compressed into a 13-qubit quantum representation while remaining reconstructable through a learned decoder. On MNIST (3 vs 5), our approach achieves 98.5% validation accuracy using a circuit-centric quantum classifier, within 1.2 percentage points of a classical neural network baseline (99.7%) and more than 30 percentage points above a naive amplitude-embedding approach. Unlike amplitude embeddings, which require full quantum state tomography for recovery, or angle embeddings, which generally rely on cir
    
[^134]: 红皇后哥德尔机：共同进化的智能体及其评估者

    The Red Queen G\"odel Machine: Co-Evolving Agents and Their Evaluators

    [https://arxiv.org/abs/2606.26294](https://arxiv.org/abs/2606.26294)

    本文提出红皇后哥德尔机（RQGM），通过将评估者纳入进化循环，使智能体能在非平稳评估标准下进行递归自我改进，从而突破静态基准的限制。

    

    arXiv:2606.26294v1 公告类型：交叉 摘要：自我改进的智能体在编程基准测试中已达到最先进水平（SOTA），并最近被扩展到通用领域。然而，它们的搜索方法通常假设一个静态的评估标准：一个固定的验证器、基准测试或标注数据集，在智能体改进过程中保持有效。这忽略了进化的一个核心特征：物种随着环境的变化而适应。我们旨在将同样的原则引入递归自我改进，使评估成为改进循环的一部分，并将搜索开放给不断演化的评估者、对抗性目标和可能超越静态基准的动态效用函数。我们引入了红皇后哥德尔机（RQGM），这是一个用于非平稳效用下递归自我改进的演化框架。RQGM通过受控的效用演化实现了这一点：搜索被组织成具有固定期内评估标准的周期，而效用可以跨周期演化。

    arXiv:2606.26294v1 Announce Type: cross  Abstract: Self-improving agents are state-of-the-art (SOTA) on agentic coding benchmarks and have recently been extended to general domains. However, their search methods generally assume a stationary evaluation criterion: a fixed verifier, benchmark, or labeled dataset that remains valid as the agent improves. This ignores a central feature of evolution: species adapt as their environments change with them. We aim to bring the same principle to recursive self-improvement, making evaluation part of the improvement loop and opening search to evolving evaluators, adversarial objectives, and dynamic utilities that may surpass static benchmarks. We introduce the Red Queen Godel Machine (RQGM), an evolutionary framework for recursive self-improvement under non-stationary utilities. The RQGM makes this possible through controlled utility evolution: search is organized into epochs with a fixed within-epoch evaluation criterion, while the utility can be
    
[^135]: 通过Hankel降阶建模的SSM适配器：注入位置决定长上下文微调中的任务适用性

    SSM Adapters via Hankel Reduced-order Modeling: Injection Site Determines Task Suitability in Long-Context Fine-Tuning

    [https://arxiv.org/abs/2606.26290](https://arxiv.org/abs/2606.26290)

    本文提出了一种基于Hankel降阶建模的SSM适配器HRM，通过平衡截断初始化实现高效并行计算，在长上下文微调中显著优于LoRA，并揭示了注入位置对任务适用性的关键影响。

    

    虽然参数高效微调通常针对注意力投影器，但其在需要序列状态累积的任务中的有效性仍未被充分探索。我们研究了此类任务中参数高效微调是否能从状态空间模型适配器中受益，以及MLP块是否是更好的注入位置。我们引入了Hankel降阶模型适配器，这是一种基于SSM的残差模块，通过经验Hankel格拉姆矩阵的平衡截断进行初始化。通过利用系统矩阵$\bar{A}$的时间不变性，HRM实现了基于FFT的精确并行扫描，在所有上下文长度上达到与LoRA相同的计算效率。在Mistral-7B（840万可训练参数）的等参数评估中，HRM在LongBench任务上优于LoRA变体，包括QuALITY（相对准确率提升34.8%）和QMSum（相对ROUGE-1提升71.6%）。HRM在18种合成状态跟踪配置中进一步展现出持续优越性。

    arXiv:2606.26290v1 Announce Type: cross  Abstract: While parameter-efficient fine-tuning (PEFT) typically targets attention projectors, its efficacy for tasks requiring sequential state accumulation remains under-explored. We examine if PEFT for such tasks can benefit from state space model (SSMs) adapters, and if MLP blocks are better injection sites. We introduce Hankel Reduced order Model (HRM) adapter, an SSM-based residual module initialized via Balanced Truncation of empirical Hankel Grammians. By leveraging the time-invariance of the system matrix $\bar{A}$, HRM enables an exact FFT-based parallel scan, achieving computational parity with LoRA across all context lengths. In iso-parametric evaluations on Mistral-7B (8.4M trainable parameters), HRM outperforms LoRA variants on LongBench tasks, including QuALITY (+34.8\% relative accuracy) and QMSum (+71.6\% relative ROUGE-1). HRM further demonstrates consistent superiority across 18 configurations of synthetic state-tracking (DFA,
    
[^136]: 从点击到意图：基于大语言模型提炼分类法的跨平台会话嵌入在金融服务推荐中的应用

    From Clicks to Intent: Cross-Platform Session Embeddings with LLM-Distilled Taxonomy for Financial Services Recommendations

    [https://arxiv.org/abs/2606.26277](https://arxiv.org/abs/2606.26277)

    本文提出了一种利用大语言模型提炼的分类法来生成跨平台会话嵌入，从而在金融服务推荐中实现从网页点击到用户意图的预测，并支持定量推荐与定性理解的双重目标。

    

    顺序用户行为建模在工业推荐系统中被广泛采用；然而，在金融服务领域仍存在显著差距，因为登录前的网页交互与认证后的应用内体验截然不同。具体而言，登录前的网页用户通常探索新产品，而登录后的应用用户则专注于账户服务。由于跨渠道实体解析的挑战（例如，将匿名网页会话匹配到已认证的移动账户），基于网页的意图信号在认证后个性化中仍未得到充分利用。现有捕获网页意图的方法通常是临时且狭窄的，缺乏灵活性来同时支持定量下游推荐和定性的大规模理解。在这项工作中，我们提出了一个可扩展且双用途的意图预测框架，用于基于网页的交互，并展示了其在个性化中的适用性。

    arXiv:2606.26277v1 Announce Type: cross  Abstract: Sequential user behavior modeling is widely adopted in industrial recommender systems; however, significant gaps remain in financial services, where pre-login web interactions and authenticated in-app experiences differ drastically. Specifically, pre-login web users typically explore new products, whereas logged-in app users focus on account servicing. Due to the challenge of cross-channel entity resolution (e.g., matching anonymous web sessions to authenticated mobile accounts), web-based intent signals remain underutilized for post-authentication personalization. Existing methods for capturing web-based intent are often ad-hoc and narrow, lacking the flexibility to support both quantitative downstream recommendations and qualitative understanding at scale. In this work, we propose a scalable and dual-purpose intent prediction framework for web-based interactions and demonstrate its applicability for personalization. Our approach tran
    
[^137]: 贝叶斯神经网络的等变性与数据增强

    Equivariance and Augmentation for Bayesian Neural Networks

    [https://arxiv.org/abs/2606.26273](https://arxiv.org/abs/2606.26273)

    本文针对贝叶斯神经网络，在变分推理框架下研究了数据增强如何实现等变性，推导了精确等变条件并提出了三种对称化技术以提升性能。

    

    对称性对许多深度学习任务至关重要，从科学应用到医学影像。然而，关于是否应将对称性约束强加于神经网络架构（产生等变神经网络）还是通过增强训练数据来学习对称性，一直存在争论。尽管等变网络在理论上已有充分研究，但对数据增强的了解却少得多，因为分析增强需要控制训练动态。受近期研究结果（表明增强的无限深度集成具有完全等变性）的启发，我们研究了使用变分推理训练的贝叶斯神经网络（BNNs）的数据增强问题。我们聚焦于指数族中的变分分布，并推导出达到精确等变性的条件。此外，我们得到了等变误差的界限，并引入了三种新颖的对称化技术，以提升效果。

    arXiv:2606.26273v1 Announce Type: new  Abstract: Symmetries are important for many deep learning tasks, ranging from applications in the sciences to medical imaging. However, there is an ongoing debate about whether to impose symmetry constraints on the neural network architecture (yielding equivariant neural networks) or learn them from augmented training data. Although equivariant networks are well-studied theoretically, much less is known about data augmentation, since analyzing augmentation requires control over the training dynamics. Inspired by recent results that show that augmented infinite deep ensembles are exactly equivariant, we study data augmentation for Bayesian neural networks (BNNs) trained with variational inference. We focus on variational distributions in the exponential family and derive conditions under which exact equivariance is reached. We furthermore obtain bounds on the equivariance error and introduce three novel symmetrization techniques which boost the eff
    
[^138]: 无需影子模型或保留数据的数据集使用推断

    Dataset Usage Inference without Shadow Models or Held-out Data

    [https://arxiv.org/abs/2606.26257](https://arxiv.org/abs/2606.26257)

    提出了一种无需影子模型和保留数据的数据集使用推断方法，通过生成合成非成员样本和混合比例估计，解决了现有方法依赖不切实际假设的问题。

    

    arXiv:2606.26257v1 公告类型：新  摘要：我的数据有多少被用于训练机器学习模型？数据集使用推断（DUI）旨在通过估计一个数据集中有多大比例被用于模型训练来回答这个问题。然而，现有的DUI方法依赖于在实践中很少成立的假设：它们需要训练昂贵的影子模型来模仿目标模型，并且假设可以访问已知的训练样本以及一个确认未参与训练且分布内保留集。这些条件使得当前的方法对于现代大型模型和真实数据所有权争议而言不切实际。我们引入了一个实用的DUI框架，消除了这些限制。我们的方法既不需要影子模型，也不需要真实的保留数据。相反，它生成合成的非成员样本，提取多样的成员信号，并将DUI视为一个混合比例估计问题，以估计候选数据集在训练过程中被使用的比例。

    arXiv:2606.26257v1 Announce Type: new  Abstract: How much of my data was used to train a machine learning model? Dataset Usage Inference (DUI) aims to answer this by estimating what fraction of a dataset contributed to a model's training. However, existing DUI methods rely on assumptions that rarely hold in practice: they require training expensive shadow models to imitate the target model, and they assume access to both known training samples and an in-distribution held-out set confirmed to be absent from training. These conditions make current approaches impractical for modern large models and real data ownership disputes. We introduce a practical DUI framework that removes these constraints. Our method requires neither shadow models nor real held-out data. Instead, it generates synthetic non-member samples, extracts diverse membership signals, and casts DUI as a mixture proportion estimation problem to estimate what share of the candidate dataset was used during training. Experiment
    
[^139]: 解读物理学中机器学习中的“可解释性”与阐释“可说明性”

    Interpreting "Interpretability" and Explaining "Explainability" in Machine Learning in Physics

    [https://arxiv.org/abs/2606.26228](https://arxiv.org/abs/2606.26228)

    本文明确区分了物理学机器学习中模型的结构透明度（可解释性）与科学内容映射（可说明性），强调它们是有意的建模选择而非固有属性，并讨论了相关的权衡与工具。

    

    我们回顾了可解释性与可说明性这两个概念在物理学机器学习中的应用。我们将可解释性定义为关于模型的结构透明度（理解或近似其内部运作的能力），而可说明性则涉及模型的科学内容（将其映射到领域知识的能力）。我们讨论了它们各自带来的权衡（可解释性与表达能力；可说明性与适应性）、需要它们的上下文环境，以及实现它们可用的内在与事后工具。全文强调，机器学习模型与经典模型面临相同的科学问题，仅在规模上有所区别，并且可解释性与可说明性最好被理解为有意的建模选择，而非固有属性。我们还强调了任务规范和干预计划作为核心方面的重要性。

    arXiv:2606.26228v1 Announce Type: cross  Abstract: We review the concepts of interpretability and explainability as they apply to machine learning in physics. We define interpretability as concerning the structural transparency of a model (the ability to understand or approximate its inner workings) and explainability as concerning the scientific content of a model (the ability to map it onto domain knowledge). We discuss the trade-offs each entails (interpretability vs. expressivity; explainability vs. adaptability), the contexts in which each is needed, and the intrinsic and post-hoc tools available for achieving them. Throughout, we emphasize that machine-learned models are subject to the same scientific questions as classical models, differing only in scale, and that interpretability and explainability are best understood as deliberate modeling choices rather than inherent properties. We also emphasize the importance of task specification and intervention plans as a core aspect of 
    
[^140]: 快速LeWorldModel

    Fast LeWorldModel

    [https://arxiv.org/abs/2606.26217](https://arxiv.org/abs/2606.26217)

    Fast-LeWM通过动作前缀并行预测替代传统自回归展开，显著降低了视觉世界模型在规划中的计算成本并减少了潜在误差累积。

    

    arXiv:2606.26217v1 公告类型：新论文 摘要：联合嵌入预测架构（JEPAs），包括最近的LeWorldModel（LeWM），已成为无重建视觉世界模型的有前景的基础。然而，在视觉规划中，LeWM通过重复应用局部的一步潜在状态转换模型来评估候选动作序列。这种自回归的展开方式使得规划计算成本高昂，并且随着时间跨度的增加，预测轨迹会累积潜在误差。我们提出了快速LeWorldModel（Fast-LeWM），一种快速潜在世界模型，它用动作前缀预测替代了重复的局部展开。给定当前潜在状态和一个候选动作序列，Fast-LeWM对其前缀进行编码，并并行预测执行这些前缀后达到的未来潜在状态。通过将动作前缀作为基本预测单元，Fast-LeWM直接建模了不同时间跨度上累积到不同程度的行为效果。这种前缀级别的监督方式

    arXiv:2606.26217v1 Announce Type: new  Abstract: Joint-Embedding Predictive Architectures (JEPAs), including recent LeWorldModel (LeWM), have become a promising foundation for reconstruction-free visual world models. For visual planning, however, LeWM evaluates candidate action sequences by repeatedly applying a local one-step latent transition model. This autoregressive rollout makes planning computationally expensive and exposes the predicted trajectory to accumulated latent errors as the horizon grows. We propose Fast LeWorldModel (Fast-LeWM), a fast latent world model that replaces repeated local rollout with action-prefix prediction. Given the current latent and a candidate action sequence, Fast-LeWM encodes its prefixes and predicts the future latents reached after executing those prefixes in parallel. By making action prefixes the basic prediction unit, Fast-LeWM directly models action effects accumulated to different extents over multiple horizons. This prefix-level supervision
    
[^141]: 利用图神经网络从凯莱图学习代数性质的一般框架

    A General Framework for Learning Algebraic Properties from Cayley Graphs using Graph Neural Networks

    [https://arxiv.org/abs/2606.26212](https://arxiv.org/abs/2606.26212)

    本文提出了一种通用框架，利用图神经网络从有限群的凯莱图中直接学习并区分其代数性质（如阿贝尔性、幂零性和可解性），证明了图表示中蕴含的代数信息可通过GNN有效提取。

    

    论文摘要：arXiv:2606.26212v1 公告类型：新 摘要：文献[1]提出了一种图神经网络框架，用于从有限群的凯莱图表示预测其可解性。在本工作中，我们推广了该方法，并开发了一个与性质无关的框架，可直接从凯莱图学习有限群的代数性质。作为代表性案例研究，我们考虑了阿贝尔性、幂零性和可解性。通过使用通用的图神经网络架构和训练流程，我们探究了仅从基于图的表示中能恢复多少代数结构。在来自多个族群的有限群集合上的实验表明，该框架能够成功地从其关联的凯莱图中学习并区分多种代数性质。这些发现表明，图表示中编码了大量代数信息，并且可以通过图神经网络提取出来。更广泛地说，所提出的框架为从图结构数据中自动发现代数性质提供了一种通用方法。

    arXiv:2606.26212v1 Announce Type: new  Abstract: A Graph Neural Network (GNN) framework for predicting the solvability of finite groups from their Cayley graph representations was introduced in [1]. In the present work, we generalize this approach and develop a property-independent framework for learning algebraic properties of finite groups directly from Cayley graphs. As representative case studies, we consider abelianity, nilpotency, and solvability. Using a common GNN architecture and training pipeline, we investigate the extent to which algebraic structure can be recovered from graph-based representations alone. Results on a collection of finite groups drawn from several families demonstrate that the framework successfully learns and distinguishes multiple algebraic properties from their associated Cayley graphs. These findings suggest that substantial algebraic information is encoded in graph representations and can be extracted through GNNs. More broadly, the proposed framework 
    
[^142]: 输入维度在对抗样本涌现与定向控制中的作用

    The Role of Input Dimensionality in the Emergence and Targeted Control of Adversarial Examples

    [https://arxiv.org/abs/2606.26207](https://arxiv.org/abs/2606.26207)

    通过实证研究揭示输入维度增加会使对抗样本更易构造，并发现真实图像类别的强经验局部化特性超出传统高维几何理论假设。

    

    arXiv:2606.26207v1 公告类型：交叉 摘要：多项理论研究试图通过高维几何性质解释深度神经网络的对抗脆弱性。然而，这些研究所依据的假设很少得到实证检验，系统性证据仍然有限。在本工作中，我们系统研究了输入维度在对抗样本涌现与定向控制中的作用。我们首先分析了基于测度集中现象的现有理论框架的适用范围与局限性，结果表明真实图像类别表现出强经验局部化特征，其程度超出此类理论通常假设的范围。随后，我们跨多个层次图像数据集（涵盖广泛输入维度范围与多样化神经架构）进行了广泛实证评估。结果一致表明，随着维度增加，对抗样本更易于构造。我们还研究了……

    arXiv:2606.26207v1 Announce Type: cross  Abstract: Several theoretical works have tried to explain the adversarial vulnerability of deep neural networks through properties of high-dimensional geometry. However, the assumptions underlying these works are rarely examined empirically, and systematic evidence remains limited. In this work, we present a systematic study of the role of input dimensionality in both the emergence and the targeted control of adversarial examples. We first analyse the scope and limitations of existing theoretical frameworks based on concentration of measure, showing that real image classes exhibit strong empirical localization, beyond what such theories typically assume. We then conduct an extensive empirical evaluation across hierarchical image datasets spanning a wide range of input dimensionalities and diverse neural architectures. Our results consistently show that adversarial examples become easier to construct as dimensionality increases. We also investiga
    
[^143]: 面向光学与合成孔径雷达影像洪水检测的拓扑信息神经网络

    Topology-Informed Neural Networks for Flood Detection in Optical and Synthetic Aperture Radar Imagery

    [https://arxiv.org/abs/2606.26204](https://arxiv.org/abs/2606.26204)

    本文提出了一种结合拓扑信息的神经网络方法，用于在光学和合成孔径雷达影像中更准确、可解释地检测洪水，克服了云层遮挡和现有模型不透明的局限。

    

    arXiv:2606.26204v1 发布类型：新  摘要：洪水频繁影响全球各地。快速准确的洪水检测对于应急响应和及时减少人员及经济损失至关重要。卫星数据可用性的不断扩大以及人工智能的进步增强了对环境灾害的监测能力，但许多洪水事件仍难以检测，因为云层会遮挡光学卫星图像。Rambour等人引入了SEN12-FLOOD数据集，并使用ResNet-50卷积神经网络骨干网络提取每幅图像的特征，然后将这些特征输入到门控循环单元网络中，证明与单图像基线相比，时间信息可以显著提高准确性。最近，Chamatidis等人表明，视觉变换器可以与流行的卷积架构一起实现强大的性能。然而，这些模型通常作为不透明的黑箱运行，使得解释变得困难。

    arXiv:2606.26204v1 Announce Type: new  Abstract: Floods frequently impact regions around the world. Rapid and accurate flood detection is crucial for emergency response and timely mitigation of human and economic loss. The expanding availability of satellite data and advances in artificial intelligence have enhanced monitoring of environmental hazards, but many flood events remain challenging to detect because cloud cover obscures optical satellite imagery. Rambour et al. introduced the SEN12-FLOOD dataset and extracted per-image features using a ResNet-50 convolutional neural network backbone, then fed these features into a gated recurrent unit network to show that temporal information can substantially improve accuracy compared to single-image baselines. More recently, Chamatidis et al. showed that a vision transformer can achieve strong performance with popular convolutional architectures. However, these models typically function as opaque black boxes, making it difficult to interpr
    
[^144]: 算法公平性的统计与结构方法

    Statistical and Structural Approaches to Algorithmic Fairness

    [https://arxiv.org/abs/2606.26200](https://arxiv.org/abs/2606.26200)

    本论文指出现代算法公平性方法的两大根本缺陷——依赖确定性点估计审计和将个体视为孤立实体，并提出改进方案。

    

    现代机器学习系统已超越其作为孤立预测构件的起源，演变为积极调节人类机遇的复杂社会技术架构。随着算法日益决定经济与社会机会的获取，人们普遍认识到这些系统深度嵌入了其所在环境的结构性不平等与偏见。算法公平性领域应运而生，以应对一个日益明确的认知：为预测准确性优化的模型可能系统性地边缘化弱势群体。然而，早期的缓解策略建立在脆弱的简化假设之上，这限制了其在复杂社会技术环境中的有效性。本论文识别并解决了当代公平性范式的两个根本局限性：依赖确定性点估计进行审计，以及将个体视为孤立实体。

    arXiv:2606.26200v1 Announce Type: cross  Abstract: Modern machine learning systems have outgrown their origins as isolated predictive constructs, evolving into complex socio-technical architectures that actively mediate human opportunity. As algorithms increasingly determine access to economic and social opportunities, it has become widely recognized that these systems are deeply embedded with the structural inequalities and prejudices of their environments. The field of algorithmic fairness emerged in response to the growing recognition that models optimized for predictive accuracy can systematically disadvantage marginalized groups. Early mitigation strategies, however, rested on fragile simplifications that limited their effectiveness in complex socio-technical environments. This thesis identifies and addresses two fundamental limitations of contemporary fairness paradigms: the reliance on deterministic point estimates for auditing and the treatment of individuals as isolated entiti
    
[^145]: 从结构到协同：多模态大语言模型中视觉-语言感知范式演进的综述

    From Structure to Synergy: A Survey of Vision-Language Perception Paradigm Evolution in Multimodal Large Language Models

    [https://arxiv.org/abs/2606.26196](https://arxiv.org/abs/2606.26196)

    该论文首次系统综述了多模态大语言模型中视觉与语言作为不可分割整体的统一感知范式演进，并形式化了类似人类先天感知的内在跨模态能力。

    

    arXiv:2606.26196v1 公告类型：交叉 摘要：多模态大语言模型（MLLMs）近期在统一视觉-语言理解与推理方面取得了显著进展，尤其是在OpenAI的O系列和DeepSeek的R系列等模型推出之后，这些模型推动了向以感知为中心的智能范式的转变。然而，目前仍缺乏从真正统一的视觉-语言视角（即将视觉和语言视为不可分割的模态）来审视感知的系统性综述。现有综述往往较为零散，分别聚焦于视觉或语言，因此很少捕捉到感知作为一种综合能力的跨模态演进。为弥补这一空白，我们首次提出了对MLLMs中统一视觉-语言感知的系统性综述。具体而言，我们（1）将MLLM感知形式化为一种类似于人类先天感知的内在、统一的视觉-语言能力，（2）引入了五...

    arXiv:2606.26196v1 Announce Type: cross  Abstract: Multimodal Large Language Models (MLLMs) have recently made remarkable progress in unifying vision-language understanding and reasoning, especially following the introduction of models such as OpenAI's O-series and DeepSeek's R-series, which have driven a paradigm shift toward perception-centric intelligence. However, there remains a lack of systematic surveys that examine perception from a truly unified vision-language perspective -- one that treats vision and language as an inseparable modality. Existing reviews are often fragmented, focusing separately on either vision or language, and thus rarely capture the cross-modal evolution of perception as an integrated capability. To bridge this gap, we present the first systematic survey of unified vision-language perception in MLLMs. Specifically, we (1) formalize MLLM perception as an intrinsic, unified vision-language capability analogous to human innate perception, (2) introduce a five
    
[^146]: 基于自监督学习的城市环境树冠级生物量估算：结合机载LiDAR与光学观测

    Self-Supervised Tree-level Biomass Estimation in Urban Environments From Airborne LiDAR and Optical Observations

    [https://arxiv.org/abs/2606.26194](https://arxiv.org/abs/2606.26194)

    该研究提出了一种结合机载LiDAR和光学数据的自监督树冠级生物量估算框架，通过双流交叉注意力网络和伪标签训练，实现了城市树木的精准语义分割和生物量估计。

    

    arXiv:2606.26194v1 公告类型：跨领域 摘要：城市树木生物量在空间上的明确量化程度仍低于人工林，因为许多估算依赖于无法分辨单个树冠或细尺度异质性的清单或粗粒度产品。我们提出了一种树冠级地上生物量（AGB）框架，应用于加拿大安大略省一个约810平方公里的区域，使用了2018年和2023年的落叶期机载LiDAR（8–10脉冲/平方米）和近红外RGB正射影像（0.16–0.20米）。一个基于规则伪标签训练的双流交叉注意力网络，生成了建筑物、针叶树和落叶树的语义标记，支持树冠勾画和功能类型分配。在独立标注的保留瓦片上，全局/平均精确率、召回率和Dice分数分别为0.86、0.83和0.84。在映射的树木区域，使用多尺度分水岭分割勾画树冠，并通过树冠面积-高度幂律代理关系估算AGB，该关系经过校准。

    arXiv:2606.26194v1 Announce Type: cross  Abstract: Urban tree biomass remains less spatially explicitly quantified than biomass in managed forests because many estimates rely on inventories or coarse products that cannot resolve individual crowns or fine-scale heterogeneity. We present a crown-level above-ground biomass (AGB) framework for an 810~km$^2$ landscape in Ontario, Canada, using leaf-off airborne LiDAR (8--10~pulses~m$^{-2}$) and near-infrared RGB orthophotography (0.16--0.20~m) from 2018 and 2023. A dual-stream cross-attention network trained on rule-based pseudo-labels produced semantic marks for buildings, needleleaf trees, and deciduous trees, supporting crown delineation and functional-type assignment. On independently annotated withheld tiles, global/mean precision, recall, and Dice scores were 0.86, 0.83, and 0.84. Crowns were delineated with multiscale watershed segmentation in mapped tree areas, and AGB was estimated from a crown area--height power-law proxy calibrat
    
[^147]: 联邦哈希投影潜在因子学习

    Federated Hash Projected Latent Factor Learning

    [https://arxiv.org/abs/2606.26192](https://arxiv.org/abs/2606.26192)

    提出联邦哈希投影潜在因子模型，通过用二进制哈希码替代实值梯度并增强表示容量，在联邦学习中实现低通信开销和高模型精度。

    

    哈希学习是一种高效的表示学习方法，能将实值数据映射为紧凑的二进制表示。传统的哈希学习方法通常要求用户将个人数据上传至中央服务器，这与日益严格的数据安全法规相冲突。联邦学习提供了一种去中心化的范式，无需集中私有数据即可学习全局最优模型。然而，大多数联邦学习方法依赖于传输大规模的实值梯度信息，导致较高的通信开销和潜在的隐私风险。将哈希学习整合到联邦学习中是一个有前景的解决方案。然而，现有的哈希学习方法受限于二进制编码的表示能力，可能降低模型精度。为应对这一挑战，我们提出了一种联邦哈希投影潜在因子模型。该模型引入了三项关键创新：(a) 用二进制哈希码替代实值梯度矩阵，以减少通信开销；(b) 采用投影潜在因子学习增强二进制表示的容量；(c) 设计一种联邦优化机制以保护数据隐私并保持模型性能。

    arXiv:2606.26192v1 Announce Type: new  Abstract: Hash Learning (HL) is an efficient representation learning approach that maps real-valued data into compact binary representations. Traditional HL methods typically require users to upload personal data to a central server, which is incompatible with increasingly stringent data security regulations. Federated Learning (FL) provides a decentralized paradigm for learning globally optimal models without centralizing private data. However, most FL methods rely on transmitting large-scale real-valued gradient information, leading to high communication overhead and potential privacy risks. Integrating HL into FL is a promising solution. Nevertheless, existing HL methods suffer from limited representational capacity of binary codes, which may degrade model accuracy. To address this challenge, we propose a Federated Hash Projected Latent Factor (FHPLF) model. FHPLF introduces three key innovations: (a) replacing real-valued gradient matrices wit
    
[^148]: 线索引导的洗钱团伙发现

    Clue-Guided Money Laundering Group Discovery

    [https://arxiv.org/abs/2606.26189](https://arxiv.org/abs/2606.26189)

    提出了一种基于线索引导的洗钱团伙发现方法，通过分析师交互从初始线索逐步恢复团伙结构，并设计了Clue2Group框架来构建局部调查上下文并保留关键结构模式。

    

    arXiv:2606.26189v1 公告类型：新 摘要：洗钱团伙发现（MLGD）旨在识别大型金融网络中隐藏的犯罪团伙并恢复其完整结构。现有的图异常检测方法主要产生节点级风险警报，而全局团伙发现方法则被动地在整个网络中搜索可疑团伙。这两种方法都与真实的反洗钱（AML）调查不匹配，在实际调查中，分析人员通常从一个具体线索出发，逐步扩大调查范围以恢复负责的团伙。为了解决这一差距，我们提出了线索引导的团伙发现（CGGD），该方法通过分析师交互，从初始线索集合逐步恢复一个洗钱团伙。我们进一步提出了Clue2Group框架，该框架首先构建一个紧凑的局部调查上下文，以减少噪声并保留链状和环状洗钱结构，然后通过多...

    arXiv:2606.26189v1 Announce Type: new  Abstract: Money Laundering Group Discovery (MLGD) aims to identify hidden criminal groups and recover their complete structures in large-scale financial networks. Existing graph anomaly detection methods mainly produce node-level risk alerts, while global group discovery methods passively search for suspicious groups over the whole network. Both are mismatched with real Anti-money-laundering (AML) investigations, where analysts usually start from a concrete clue and gradually expand the investigation to recover the responsible group. To address this gap, we propose Clue-Guided Group Discovery (CGGD), where a laundering group is progressively recovered from an initial clue set through analyst interaction. We further propose Clue2Group, a framework that first constructs a compact local investigation context to reduce noise and preserve chain-like and cycle-like laundering structures. It then estimates a clue-conditioned local risk field with a multi
    
[^149]: 必要但不充分：大语言模型作为评判者的安全评估中的温度控制与可重复性

    Necessary but Not Sufficient: Temperature Control and Reproducibility in LLM-as-Judge Safety Evaluations

    [https://arxiv.org/abs/2606.26185](https://arxiv.org/abs/2606.26185)

    本研究通过实验证明，即使将大语言模型作为评判者的采样温度设为0，也无法完全消除安全评估结果的不确定性，尤其是在决策边界附近的项目中仍存在不可重复性。

    

    arXiv:2606.26185v1 公告类型：新论文 摘要：大语言模型作为评判者（“评分器”）组件现已成为评估框架中的标准配置，包括在安全评估中，通过/不通过的判定结果可能决定下游部署决策。一个普遍假设是将评分器的采样温度设置为0可使评分结果确定。我们针对真实的安全评估代码库（日本AISI的开源项目aisev）测试了这一假设，并在两个层面上证明其不成立。首先，评估框架在调用评分器时未设置温度或随机种子；底层提供商默认应用1.0的温度值，因此接近决策边界的项目在相同运行中会出现通过/不通过翻转（在20次运行中，单个项目的不一致率高达约50%）。其次，将温度固定为0可以减少但无法消除翻转：在跨越两个提供商、三个模型层级和五种采样配置的690次API调用中，即使采用强制贪婪解码（top_k=1），7个边界项目中的1-2个仍然不可重复。Claude Opus 4.7/4.

    arXiv:2606.26185v1 Announce Type: new  Abstract: LLM-as-judge ("grader") components are now standard in evaluation harnesses, including safety evaluations where a pass/fail verdict may gate downstream deployment decisions. A widespread assumption is that setting the grader's sampling temperature to 0 makes grading deterministic. We test this assumption against a real safety-evaluation codebase (Japan AISI's open-source aisev) and show it fails on two levels. First, the harness invokes its grader without setting temperature or seed; the underlying provider silently applies its default of 1.0, so items near the decision boundary flip pass/fail across identical runs (per-item disagreement up to ~50% over 20 runs). Second, pinning temperature=0 reduces but does not eliminate flips: across 690 API calls spanning two providers, three model tiers, and five sampling configurations, 1-2 of 7 borderline items remain non-reproducible even under forced greedy decoding (top_k=1). Claude Opus 4.7/4.
    
[^150]: LiMoDE：从动态专家混合视角重新思考终身机器人操作

    LiMoDE: Rethinking Lifelong Robot Manipulation from a Mixture-of-Dynamic-Experts Perspective

    [https://arxiv.org/abs/2606.26183](https://arxiv.org/abs/2606.26183)

    本文提出LiMoDE，一种基于动态专家混合架构的两阶段终身机器人操作学习方案，通过多任务预训练学习可复用先验知识，并在任务适应阶段实现高效连续适应。

    

    arXiv:2606.26183v1 公告类型：跨领域 摘要：构建一个能够利用先前知识进行连续任务适应的通用机器人仍然是一个重大挑战。以往的研究通过参数高效微调来缓解单任务适应中的灾难性遗忘问题，但未能有效提取可复用技能并建模技能间的交互。近期工作尝试通过学习提示来解决这些问题。与此不同，本文提出了一种架构视角——终身动态专家混合（LiMoDE），这是一种面向终身机器人操作的新型两阶段学习方案。具体而言，在多任务预训练阶段，首先提出一种动态MoE结构来学习先验知识，其中根据运动信息激活不同数量的异构专家以处理不同的短期操作。随后，在任务适应阶段，我们设计了一种终身MoE适应机制。

    arXiv:2606.26183v1 Announce Type: cross  Abstract: Building a generalist robot that can leverage prior knowledge for continuous task adaptation remains a significant challenge. Previous works alleviate the catastrophic forgetting problem by parameter-efficient fine-tuning for single-task adaptation. However, they fail to extract reusable skills and model the interaction with other skills effectively. Recent works try to address these issues by learning prompts. Differently, this paper presents an architectural perspective on the Lifelong Mixture of Dynamic Experts (\textit{LiMoDE}), a novel two-stage learning scheme for lifelong robot manipulation. Specifically, a dynamic MoE structure is first proposed in the multi-task pre-training stage to learn prior knowledge, where a varied number of heterogeneous experts are activated based on the motion information to address different short-term manipulations. Subsequently, in the task adaptation stage, we design a lifelong MoE adaptation mech
    
[^151]: KG-TRACE：一种用于抗菌素耐药性预测中机制性归因的神经符号框架

    KG-TRACE: A Neuro-Symbolic Framework for Mechanistic Grounding in Antimicrobial Resistance Prediction

    [https://arxiv.org/abs/2606.26179](https://arxiv.org/abs/2606.26179)

    该论文提出KG-TRACE框架，通过将知识图谱作为生物学约束集成到神经模型中，并引入BGR指标量化神经归因与生物学知识的一致性，从而在保持高预测准确率的同时实现了可解释的符号归因。

    

    arXiv:2606.26179v1 公告类型：交叉 摘要：虽然基于全基因组测序的抗菌素耐药性预测已达到高准确率，但现有模型缺乏将神经归因锚定于既定生物学通路的机制。我们提出了KG-TRACE，一种新颖的神经符号框架，它将世界卫生组织突变知识图谱作为结构化生物学约束，整合到神经基因组模型中。与现有孤立学习统计模式的方法不同，KG-TRACE通过一个学习到的认知信任门控机制，融合基因组特征和基于RotatE的知识图谱嵌入，动态权衡神经证据与符号生物学知识。在CRyPTIC结核分枝杆菌队列上评估时，KG-TRACE对异烟肼的AUROC达到0.9760，在实现竞争性准确率的同时，其主要价值在于符号归因，而非预测性能提升。更重要的是，我们引入了生物学归因比（BGR），这是一种数据集层面的度量，用于量化神经归因与既定生物学知识之间的一致性。

    arXiv:2606.26179v1 Announce Type: cross  Abstract: While WGS-based AMR prediction has reached high accuracy, existing models lack a mechanism to ground neural attributions in established biological pathways. We present KG-TRACE, a novel neuro-symbolic framework that integrates the WHO mutation knowledge graph (KG) as a structured biological constraint on a neural genomic model. Unlike existing methods that learn statistical patterns in isolation, KG-TRACE fuses genomic features and RotatE-based KG embeddings through a learned epistemic trust gate, dynamically weighting neural evidence against symbolic biological knowledge.   Evaluated on the CRyPTIC M. tuberculosis cohort, KG-TRACE achieves an AUROC of 0.9760 for isoniazid, achieving competitive accuracy while its primary value lies in symbolic grounding, not predictive uplift. More importantly, we introduce the Biological Grounding Ratio (BGR), a dataset-level metric that quantifies alignment between neural attributions and establishe
    
[^152]: 生成对抗网络的神经架构搜索：全面回顾与批判性分析

    Neural Architecture Search for Generative Adversarial Networks: A Comprehensive Review and Critical Analysis

    [https://arxiv.org/abs/2606.26169](https://arxiv.org/abs/2606.26169)

    本文全面回顾了神经架构搜索在生成对抗网络中的应用，重点比较了不同搜索策略和评估指标，强调了进化算法和梯度方法的优势，以及使用多样化数据集和稳健评估指标的重要性。

    

    神经架构搜索（NAS）已成为优化生成对抗网络（GANs）设计的关键技术，能够自动搜索有效架构，同时应对人工设计中的固有挑战。本文对应用于GANs的NAS方法进行了全面回顾，基于搜索策略、评估指标和性能结果等标准对各种方法进行分类和比较。该回顾强调了NAS在提升GAN性能、稳定性和效率方面的优势，同时也指出了局限性及未来研究方向。关键发现包括：进化算法和基于梯度的方法在某些情境下表现优越；除了传统的Inception Score（IS）和Fréchet Inception Distance（FID）等分数外，稳健的评估指标至关重要；以及在评估GAN性能时需要使用多样化的数据集。

    arXiv:2606.26169v1 Announce Type: cross  Abstract: Neural Architecture Search (NAS) has emerged as a pivotal technique in optimizing the design of Generative Adversarial Networks (GANs), automating the search for effective architectures while addressing the challenges inherent in manual design. This paper provides a comprehensive review of NAS methods applied to GANs, categorizing and comparing various approaches based on criteria such as search strategies, evaluation metrics, and performance outcomes. The review highlights the benefits of NAS in improving GAN performance, stability, and efficiency, while also identifying limitations and areas for future research. Key findings include the superiority of evolutionary algorithms and gradient-based methods in certain contexts, the importance of robust evaluation metrics beyond traditional scores like Inception Score (IS) and Fr\'echet Inception Distance (FID), and the need for diverse datasets in assessing GAN performance. By presenting a
    
[^153]: 化学反应网络中强化学习的实现：以趋光性作为好奇心驱动的探索为例

    Implementation of reinforcement learning in chemical reaction networks: application to phototaxis as curiosity-driven exploration

    [https://arxiv.org/abs/2606.26168](https://arxiv.org/abs/2606.26168)

    本文提出一个将部分可观测马尔可夫决策过程与化学反应网络常微分方程结合的框架，通过信息驱动的最小认知模型重新定义单细胞藻类的趋光性导航，实现了在感官模糊性下的主动采样与探索平衡。

    

    生命系统利用嘈杂且不完整的感觉信号在环境中导航。在单细胞藻类中，趋光性通常被建模为由刺激-反应规则驱动的机械性“游动-翻滚”过程。然而，这种描述忽略了生物体如何主动采样其环境以减少感官模糊性。从最小认知的视角出发，我们将这种导航重新定义为一种主观的、信息驱动的感觉运动过程。为此，我们提出了一个框架，将部分可观测马尔可夫决策过程（POMDP）与生化反应动力学联系起来。环境变量是隐藏的，而细胞通过无记忆的贝叶斯步骤从每次观测中更新一个最小的内部状态。这些内部动力学在朝向光的定向与探索性重新定向之间取得平衡，并可通过化学反应网络常微分方程（CRN-ODEs）实现。我们的模型包含一个生物物理观测过程。

    arXiv:2606.26168v1 Announce Type: new  Abstract: Living systems navigate environments using noisy and incomplete sensory signals. In unicellular algae, phototaxis is often modeled as a mechanistic run--tumble process driven by stimulus--response rules. However, such descriptions overlook how organisms actively sample their environment to reduce sensory ambiguity. From a minimal cognition perspective, we reframe this navigation as a subjective, information-driven sensorimotor process. To this end, we propose a framework linking a Partially Observable Markov Decision Process (POMDP) with biochemical reaction dynamics. Environmental variables are hidden, while the cell updates a minimal internal state from each observation through a memoryless Bayesian step. These internal dynamics balance orienting toward light with exploratory reorientation and can be implemented through Chemical-Reaction-Network Ordinary Differential Equations (CRN--ODEs). Our model includes a biophysical observation p
    
[^154]: 《chisao：一种基于收敛-反收敛振荡的GPU原生多模态黑箱函数并行优化器》

    \chisao{}: A GPU-Native Parallel Optimizer for Multimodal Black-Box Functions via Convergence-Anticonvergence Oscillation

    [https://arxiv.org/abs/2606.26164](https://arxiv.org/abs/2606.26164)

    本文提出了一种GPU原生并行优化器chisao，通过创新的收敛-反收敛振荡机制和多策略自适应重播种，能够高效地找到多模态黑箱函数的所有模态。

    

    摘要：在多模态黑箱优化、贝叶斯推断和科学计算中，找到所有模态是一个基础性挑战。现有方法——如盆地跳跃、CMA-ES、多起点梯度下降——是顺序执行的，无法利用现代GPU硬件的大规模并行能力。我们提出了chisao（收敛-暂停-反转-固定-振荡），一种GPU原生的种群优化器，它同时运行整个样本批次，并利用有意的收敛-反收敛振荡循环来逃离局部陷阱，同时冻结已确认的模态。其结构移动是不对称的：达到真实峰值的样本被冻结（“固定”）并保留，而其余样本则通过基于动量的反收敛和随机平滑梯度继续探索。通过两种互补策略（斥猴和金鸡）进行自适应重新播种，以维持种群。

    arXiv:2606.26164v1 Announce Type: new  Abstract: Finding all modes of a multimodal black-box function is a fundamental challenge in optimization, Bayesian inference, and scientific computing. Existing approaches -- basin-hopping, CMA-ES, multistart gradient descent -- operate sequentially and cannot exploit the massive parallelism of modern GPU hardware. We introduce \chisao{} (\textbf{C}onvergence-\textbf{H}alt-\textbf{I}nvert-\textbf{S}tick-\textbf{A}nd-\textbf{O}scillate), a GPU-native population optimizer that runs an entire sample batch simultaneously and exploits a deliberate convergence-anticonvergence oscillation cycle to escape local traps while freezing confirmed modes. The structural move is asymmetric: samples that reach true peaks are frozen (``stuck'') and preserved, while the rest keep exploring via momentum-based anti-convergence and stochastically smoothed gradients. Adaptive reseeding via two complementary strategies (Repulse Monkey and Golden Rooster) maintains popul
    
[^155]: 强化学习实现模拟毛细血管中微型机器人的自主导航与干预

    Reinforcement Learning Enables Autonomous Microrobot Navigation and Intervention in Simulated Blood Capillaries

    [https://arxiv.org/abs/2606.26154](https://arxiv.org/abs/2606.26154)

    本研究通过构建包含真实流体动力学、红细胞动力学和毛细血管分支结构的物理仿真环境，利用深度强化学习训练微型机器人自主导航策略，并首次系统揭示了不同尺寸和速度下的导航物理极限及多种自主发现的通用策略类型。

    

    摘要：能够在生物血管网络中自主导航的微型机器人有望实现靶向药物递送和血栓溶解，然而在真实环境中训练控制策略仍是一个开放挑战。先前关于微型机器人导航的强化学习研究仅限于理想化的几何结构，忽略了体内存在的复杂流体动力学流场、受限的分支结构以及密集的细胞障碍物。在此，我们开发了一个基于物理的毛细血管网络模拟环境，包含了真实的流体动力学流场、明确的红细胞动力学以及基于解剖结构的分支几何形状，并通过化学趋向性训练深度强化学习智能体进行导航。我们系统地绘制了不同机器人尺寸和游泳速度下的导航物理极限，揭示了一个布朗运动和流场克服推进力的禁止区域。成功的智能体自主发现了多种通用策略类型，包括……

    arXiv:2606.26154v1 Announce Type: cross  Abstract: Autonomous microrobots navigating biological vasculature could enable targeted drug delivery and thrombolysis, yet training control policies for realistic environments remains an open challenge. Prior reinforcement learning (RL) studies of microrobotic navigation have been limited to idealized geometries that omit complex hydrodynamic flow fields, confined branching structures, and dense cellular obstacles found in vivo. Here, we develop a physically grounded simulation of a blood capillary network, incorporating realistic hydrodynamic flow fields, explicit red blood cell dynamics, and anatomically derived branching geometry, and train deep RL agents to navigate it via chemotaxis. We systematically map the physical limits of navigation across robot size and swimming speed, revealing a forbidden regime where Brownian motion and flow overcome propulsion. Successful agents independently discover multiple universal strategy types, includin
    
[^156]: 通过多语言训练实现神经说话人日志化：低资源尼泊尔语-印地语语音评估

    Neural Speaker Diarization via Multilingual Training: Evaluation on Low-Resource Nepali-Hindi Speech

    [https://arxiv.org/abs/2606.26144](https://arxiv.org/abs/2606.26144)

    本文通过多语言训练方法，在低资源尼泊尔语-印地语语音上评估两种神经说话人日志化架构，以解决标注数据稀缺导致的性能下降问题。

    

    说话人日志化（即确定多说话人录音中“谁在何时说话”）是会议转录、无障碍工具和多语言信息检索等应用中的关键组成部分。尽管端到端神经日志化系统在英语及其他高资源语言上表现强劲，但对于注释语音数据稀缺的低资源语言，其有效性会大幅下降。本文通过多语言训练方法，研究低资源尼泊尔语-印地语语音的说话人日志化，比较两种现代架构：带编码器-解码器吸引子的EEND（EEND-EDA）和基于感知器的吸引子的EEND（DiaPer）。两种模型均在多语言语料库上训练，该语料库结合了LibriSpeech的英语语音、VoxCeleb的多样化说话人录音以及单独收集的尼泊尔语和印地语音频，这种设置旨在减少语言偏差。

    arXiv:2606.26144v1 Announce Type: cross  Abstract: Speaker diarization, the task of determining "who spoke when" in a multi-speaker recording, is a critical component in applications such as meeting transcription, accessibility tools, and multilingual information retrieval. While end-to-end neural diarization systems have achieved strong performance for English and other high-resource languages, their effectiveness degrades substantially for underrepresented languages where annotated speech data is scarce.   This paper investigates speaker diarization for low-resource Nepali-Hindi speech through a multilingual training approach, comparing two modern architectures: EEND with encoder-decoder attractors (EEND-EDA) and EEND with Perceiver-based attractors (DiaPer). Both models are trained on a multilingual corpus combining English speech from LibriSpeech, diverse speaker recordings from VoxCeleb, and separately collected Nepali and Hindi audio, a setup designed to reduce language bias and 
    
[^157]: 复杂网络中链接预测的代码进化

    Code evolution for link prediction in complex networks

    [https://arxiv.org/abs/2606.26132](https://arxiv.org/abs/2606.26132)

    通过代码进化生成的算法在链接预测任务上显著优于人工设计方法，并在特征选择与组合上实现了关键创新。

    

    arXiv:2606.26132v1 公告类型：交叉 摘要：复杂网络中的链接预测问题出现在不同学科中，并催生了多种巧妙的人工设计方法。我们利用这一丰富的程序空间，探索自动化代码进化系统在获取机器设计的链接预测方法时的性能和行为。尽管仅在有限数据上进行训练，但通过代码进化得到的算法在性能上优于人工设计方法（在580个网络上计算的平均AUC分数为0.915，而人工设计方法为0.783），并且计算效率更高，使其能够应用于包含数百万条链接的网络。发现的方法遵循了人工设计方法中采用的一些策略，但在节点和链接特征的选择与组合上包含了关键创新。这展示了现代大型语言模型和遗传算法在算法创新及更广泛的科学发现中所能发挥的作用。

    arXiv:2606.26132v1 Announce Type: cross  Abstract: The problem of predicting links in complex networks appears in different disciplines and has led to a variety of ingenious human-designed methods. We use this rich program space to explore the performance and behavior of automated code-evolution systems tasked to obtain machine-designed methods for link prediction. Despite being trained on limited data, algorithms evolved through code evolution outperform human-designed methods (with an average AUC score of 0.915 vs. 0.783, computed over 580 networks) and show improved computational efficiency, allowing them to be applied to networks with millions of links. The discovered methods follow approaches that have been employed in human-designed methods, but contain key innovations in the selection and combination of node- and link-features. This illustrates the role modern large language models and genetic algorithms can play in algorithmic innovation and scientific discovery more generally.
    
[^158]: 物理引导的卷积神经网络用于守恒动力学系统中的畴生长预测

    Physics-guided Convolutional Neural Network for Domain Growth Prediction in Systems with Conserved Kinetics

    [https://arxiv.org/abs/2606.26128](https://arxiv.org/abs/2606.26128)

    提出了一种基于注意力机制的物理引导卷积神经网络，作为代理模型精确预测Cahn-Hilliard方程控制的二元混合物相分离的长时间演化，并保持组成守恒及与理论一致的畴生长行为。

    

    arXiv:2606.26128v1 公告类型：新 摘要：许多物理、化学和生物系统的时空演化由非线性偏微分方程描述。近年来，基于深度神经网络的代理模型作为计算成本高昂的传统数值求解器的高效替代方案，日益受到关注。在本工作中，我们提出了一种基于注意力机制的物理引导卷积神经网络作为代理模型，用于学习此类系统的微观结构演化。我们训练该模型以准确预测由Cahn-Hilliard方程控制的二元混合物相分离的完整时间演化。结果表明，我们训练的代理模型在长时间演化中对临界和近临界混合物的预测保持稳定且准确，并在整个演化过程中保持混合物的组成不变。我们还表明，该模型准确捕捉了畴尺寸的增长，并与Lifshitz-Sly理论一致。

    arXiv:2606.26128v1 Announce Type: new  Abstract: The spatiotemporal evolution of many physical, chemical, and biological systems is described by nonlinear partial differential equations (PDEs). Recently, deep neural network-based surrogate models have gained increasing interest as efficient alternatives to computationally expensive traditional numerical solvers. In this work, we propose an attention-based, physics-guided convolutional neural network as a surrogate model to learn the microstructural evolution of such systems. We train the model to accurately predict the full time-evolution of phase separation in binary mixtures governed by the Cahn-Hilliard equation. We show that predictions from our trained surrogate model remain stable and accurate over long-time rollouts for both critical and off-critical mixtures and preserve the mixture composition throughout evolution. We also show that our model accurately captures the growth of domain size and is consistent with the Lifshitz-Sly
    
[^159]: Dot-Flik：一种用于分布式昆虫监测的可扩展边缘AI架构

    Dot-Flik: A Scalable Edge AI Architecture for Distributed Insect Monitoring

    [https://arxiv.org/abs/2606.26121](https://arxiv.org/abs/2606.26121)

    本文提出了一种基于运动感知帧过滤算法的分布式边缘AI架构，通过边缘预处理丢弃无关帧、解耦数据采集与AI分类，显著降低了硬件成本与能源需求，并大幅提升了昆虫监测的可扩展性与覆盖范围。

    

    arXiv:2606.26121v1 公告类型：交叉 摘要：全球昆虫种群数量下降迫切需要可扩展的连续监测系统，然而现有的基于视觉的解决方案仍受到高硬件成本、能源需求以及对集中式处理或云连接的依赖等限制。本文提出了三项贡献来解决这些限制。首先，我们提出了一种基于时间差分、伽马校正运动放大和基于块的运动密度分析的运动感知帧过滤算法，该算法在边缘丢弃无关帧，同时保留昆虫活动，无需在传感设备上进行深度学习推理。其次，我们引入了一种分布式、分层的物联网架构，通过这种边缘级预处理将数据采集与AI分类解耦，与单体单流方法相比，实现了中央处理需求的分数级缩放，并显著增加了监测覆盖范围。

    arXiv:2606.26121v1 Announce Type: cross  Abstract: Global insect population declines necessitate scalable, continuous monitoring systems, yet existing vision-based solutions remain constrained by high hardware costs, energy demands, and reliance on centralized processing or cloud connectivity. This article presents three contributions to address these limitations. First, we propose a motion-informed frame filtering algorithm based on temporal differencing, gamma-corrected motion amplification, and block-based motion density analysis that discards irrelevant frames at the edge while preserving insect activity, without requiring deep learning inference on the sensing device. Second, we introduce a distributed, hierarchical IoT architecture that decouples data acquisition from AI classification through this edge-level preprocessing, projecting fractional scaling of central processing requirements and significantly increasing monitoring coverage compared to monolithic single-stream approac
    
[^160]: 动态dLLM：面向扩散大语言模型的无训练加速的动态缓存预算与自适应并行解码

    Dynamic-dLLM: Dynamic Cache-Budget and Adaptive Parallel Decoding for Training-Free Acceleration of Diffusion LLM

    [https://arxiv.org/abs/2606.26120](https://arxiv.org/abs/2606.26120)

    提出一种无训练框架，通过动态缓存预算分配和自适应并行解码，显著提升扩散大语言模型的推理效率，解决了长序列计算复杂度和动态令牌行为问题。

    

    arXiv:2606.26120v1 公告类型：新  摘要：扩散大语言模型（dLLMs）为自回归模型提供了一种有前景的替代方案，由于其双向注意力机制，在文本生成任务中表现出色。然而，它们的计算复杂度随序列长度L以L的三次方规模扩展。这给长序列和实时应用带来了重大挑战，主要原因是缺乏与键值缓存的兼容性以及去噪步骤的非自回归特性。现有的加速方法依赖于静态缓存或并行解码策略，未能考虑跨层和跨解码步骤中令牌属性的动态行为。我们提出Dynamic-dLLM，一种无训练框架，通过两个组件增强dLLM推理效率：动态缓存更新（DCU），基于层间令牌动态自适应分配缓存更新预算；以及自适应并行解码（APD），动态调整解码过程。

    arXiv:2606.26120v1 Announce Type: new  Abstract: Diffusion Large Language Models (dLLMs) offer a promising alternative to autoregressive models, excelling in text generation tasks due to their bidirectional attention mechanisms. However, their computational complexity scales on the order of L cubed with the sequence length L. This poses significant challenges for long-sequence and real-time applications, primarily due to the lack of compatibility with key-value caching and the non-autoregressive nature of denoising steps. Existing acceleration methods rely on static caching or parallel decoding strategies, which fail to account for the dynamic behavior of token properties across layers and decoding steps. We propose Dynamic-dLLM, a training-free framework that enhances dLLM inference efficiency through two components: Dynamic Cache Updating (DCU), which adaptively allocates cache-update budgets based on layer-wise token dynamics, and Adaptive Parallel Decoding (APD), which dynamically 
    
[^161]: 人工智能采用与能力的开源经济指数

    The Open Source Economic Index of AI Adoption and Capability

    [https://arxiv.org/abs/2606.26118](https://arxiv.org/abs/2606.26118)

    本论文通过开发开源经济指数和基准测试系统，揭示了AI在金融、计算机科学和艺术行业采用率最高，但执行具体任务细节时仍易出错。

    

    arXiv:2606.26118v1 公告类型：交叉 摘要：我们致力于衡量人工智能的采用情况以及其在各职业中执行离散劳动任务的能力。为了衡量采用情况，我们开发了一个开源经济指数，利用公开的用户与大型语言模型聊天数据和O*NET任务来复制前沿人工智能实验室的研究成果，发现金融、计算机科学和艺术行业的职业采用率最高。为了衡量能力，我们构建了一个系统，该系统能够生成基于O*NET职业、任务和模型上下文协议服务器的基准测试场景。我们使用OpenAI agents SDK框架，在指数中频繁出现的9个职业场景上测试了Kimi-k2.5模型，发现人工智能能够正确执行高层工作流程，但在具体细节（如所使用的特定工具调用）上经常出错。

    arXiv:2606.26118v1 Announce Type: cross  Abstract: We work towards measuring both AI adoption and the capability of AI to perform discrete labor tasks across various occupations. To measure adoption, we develop an open-source economic index that uses publicly available user-LLM chat data and O*NET tasks to replicate studies produced by frontier AI labs, finding that occupations in the finance, computer science, and arts sectors are those with the highest adoption rates. To measure capabilities, we build a system that generates benchmark scenarios grounded in O*NET occupations, tasks, and model-context-protocol (MCP) servers. We test Kimi-k2.5 with an OpenAI agents SDK harness on scenarios across 9 occupations that appear frequently in our index, finding that AI correctly executes high-level workflows but often errs in the granular details (such as specific tool calls used).
    
[^162]: 面向长程LLM推理的上下文回收技术

    Context Recycling for Long-Horizon LLM Inference

    [https://arxiv.org/abs/2606.26105](https://arxiv.org/abs/2606.26105)

    提出ContextForge系统，通过结构化查询生成、外部记忆检索和受控合成技术实现上下文回收，在不依赖完整上下文重放的情况下高效复用先前计算，从而在长程对话中保持答案质量并减少token消耗。

    

    arXiv:2606.26105v1 公告类型：交叉 摘要：大语言模型（LLMs）在短上下文推理中展现出强大能力，但由于上下文窗口限制和低效的token使用，在长时间对话场景中性能会下降。我们引入了ContextForge，一个用于上下文回收的系统，它通过结合结构化查询生成、外部记忆检索和受控合成，在多个对话轮次中维护与任务相关的信息。该系统能够在不依赖完整上下文重放的情况下高效复用先前计算，从而在保持答案质量的同时减少token开销。我们使用一个包含15轮对话的基准测试来评估ContextForge，该测试涵盖了结构化医疗查询中的多轮推理、回溯引用和领域切换。与使用相同基础模型的基线智能体相比，ContextForge在保持相当响应准确性的同时，展现出一致性提升和token消耗减少。这些结果表明……

    arXiv:2606.26105v1 Announce Type: cross  Abstract: Large language models (LLMs) exhibit strong capabilities in short-context reasoning but degrade in performance over long conversational horizons due to context window limitations and inefficient token usage. We introduce ContextForge, a system for context recycling that maintains task-relevant information across turns by combining structured query generation, external memory retrieval, and controlled synthesis. The system enables efficient reuse of prior computation without relying on full context replay, reducing token overhead while preserving answer quality. We evaluate ContextForge using a 15-turn conversational benchmark that tests multi-turn reasoning, back-references, and domain shifts across structured healthcare queries. Compared to a baseline agent using identical underlying models, ContextForge demonstrates improved consistency and reduced token consumption, while maintaining comparable response accuracy. These results sugge
    
[^163]: Autodata：一个用于创建高质量合成数据的自主数据科学家

    Autodata: An agentic data scientist to create high quality synthetic data

    [https://arxiv.org/abs/2606.25996](https://arxiv.org/abs/2606.25996)

    本文提出了一种名为Autodata的通用方法，通过训练AI智能体作为自主数据科学家，并对其进行元优化，从而在多个任务上生成比传统方法更高质量的合成数据，显著提升模型性能。

    

    arXiv:2606.25996v2 公告类型：替换 摘要：我们介绍了Autodata，一种通用方法，使AI智能体能够充当数据科学家，构建高质量的训练和评估数据。我们展示了如何训练（元优化）这样一个数据科学家智能体，使其学会创建更强大的数据。我们描述了总体框架以及一个具体的实践实现——自主自我指令（Agentic Self-Instruct）。我们在计算机科学研究任务、法律推理任务和数学对象推理任务上进行了实验，与经典的合成数据集创建方法相比，我们获得了更好的结果。此外，对数据科学家智能体本身进行元优化带来了更大的性能提升。自主数据创建提供了一种将增加的推理计算转化为更高质量模型训练的方法。总体而言，我们相信这一方向有潜力改变我们构建AI数据的方式。

    arXiv:2606.25996v2 Announce Type: replace  Abstract: We introduce Autodata, a general method that enables AI agents to act as data scientists who build high quality training and evaluation data. We show how to train (meta-optimize) such a data scientist agent, so that it learns to create even stronger data. We describe the overall formulation, and a specific practical implementation, Agentic Self-Instruct. We conduct experiments on computer science research tasks, legal reasoning tasks and reasoning with mathematical objects, where we obtain improved results compared to classical synthetic dataset creation methods. Further, meta-optimizing the data scientist agent itself delivers an even larger performance uplift. Agentic data creation provides a way to convert increased inference compute into higher quality model training. Overall, we believe this direction has the potential to change the way we build AI data.
    
[^164]: MiniOpt：在有限资源下推理建模并求解通用优化问题

    MiniOpt: Reasoning to Model and Solve General Optimization Problems with Limited Resources

    [https://arxiv.org/abs/2606.25832](https://arxiv.org/abs/2606.25832)

    MiniOpt通过“推理-建模-求解”范式和分层奖励函数，在无需大规模监督数据或专家演示的情况下，实现了对通用优化问题的高效强化学习求解。

    

    arXiv:2606.25832v2 公告类型：替换-交叉 摘要：在多样化的优化问题上实现强大的优化泛化能力，同时仅需有限的训练资源，这仍然是面向优化的大型语言模型（LLM）面临的一个挑战性问题。现有方法通常依赖大规模监督数据集、昂贵的推理标注以及繁琐的中间步骤验证，导致训练开销巨大。为应对这些挑战，我们提出了MiniOpt，一种强化学习框架，通过“推理-建模-求解”范式学习解决优化问题。MiniOpt将优化推理分解为结构化的优化建模和可执行的求解器生成。基于这一范式，我们引入了OptReward，一种具有分层评分结构的奖励函数，能够联合评估问题建模与求解结果，从而在无需专家演示的情况下实现有效的策略学习。我们进一步提出了一种优化。

    arXiv:2606.25832v2 Announce Type: replace-cross  Abstract: Achieving strong optimization generalization across diverse optimization problems while requiring limited training resources remains a challenging problem for optimization-oriented large language models (LLMs). Existing approaches typically rely on large-scale supervised datasets, costly reasoning annotations, and expensive intermediate step verification, resulting in substantial training overhead. To address these challenges, we propose MiniOpt, a reinforcement learning framework that learns to solve optimization problems through an "reasoning-to-model-and-solve" paradigm. MiniOpt decomposes optimization reasoning into structured optimization modeling and executable solver generation. Building upon this paradigm, we introduce OptReward, a reward function with hierarchical score structure that jointly evaluates formulation and solution, enabling effective policy learning without expert demonstrations. We further develop an opti
    
[^165]: 你的越狱评判有多可靠？自动化ASR评分的校准与对抗鲁棒性

    How Reliable Is Your Jailbreak Judge? Calibration and Adversarial Robustness of Automated ASR Scoring

    [https://arxiv.org/abs/2606.25487](https://arxiv.org/abs/2606.25487)

    该论文揭示了LLM越狱攻击成功率（ASR）评估中两类自动化评判者的严重不可靠性：专用分类器过度标记而LLM评判者召回率波动大，且两者在对抗攻击下鲁棒性极差。

    

    arXiv:2606.25487v2 公告类型：替换 摘要：几乎所有关于大语言模型越狱和提示注入的论文都会报告一个攻击成功率（ASR），而这个数字并非由人类赋予，而是由自动化评判者给出：要么是为此任务训练的安全分类器，要么是被提示进行评分的通用聊天模型。评判者很少被检验。我们对其进行了检验。利用来自HarmBench分类器验证集的596个人工标注补全结果，我们将这两类评判者与人类多数投票进行比较，然后对其进行攻击。这两类评判者以相反的方式失败。专用分类器过度标记（精确率0.835，召回率0.974）；三个不同的LLM作为评判者保持了高精确率（0.81至0.94），但召回率波动较大（0.06至0.65），因此，相同的回复根据不同的评判者打分会产生截然不同的ASR。这两类评判者在鲁棒性上也存在显著差异。那些保留有害文本不变、仅添加良性框架的包装器，在57%至100%的情况下会翻转每个LLM评判者的判断。

    arXiv:2606.25487v2 Announce Type: replace  Abstract: Almost every paper on LLM jailbreaks and prompt injection reports an attack-success rate (ASR), and that number is assigned not by people but by an automated judge: either a safety classifier trained for the task, or a general chat model prompted to grade. The judge is rarely checked. We check it. Using 596 human-labeled completions from the HarmBench classifier validation set, we compare the two judge families against human majority votes and then attack them. The two families fail in opposite ways. The dedicated classifier over-flags (precision 0.835, recall 0.974); three different LLM-as-judges keep high precision (0.81 to 0.94) but show erratic recall (0.06 to 0.65), so the same responses produce very different ASR depending on which judge scores them. The two families also differ sharply in robustness. Wrappers that leave the harmful text untouched and only add benign framing flip every LLM-judge between 57% and 100% of the time
    
[^166]: 泛化光谱：一种评估学习算法的色谱方法

    The Generalization Spectrum: A Chromatographic Approach to Evaluating Learning Algorithms

    [https://arxiv.org/abs/2606.25450](https://arxiv.org/abs/2606.25450)

    本文提出“泛化光谱”评估框架，通过按迁移距离排列的受控测试变体，逐样本揭示学习算法的泛化程度与模式，超越传统单一聚合分数评估。

    

    arXiv:2606.25450v2 公告类型：替换交叉 摘要：传统评估方法仅通过独立同分布测试集上的最终性能来衡量学习算法，将学习过程简化为单一的聚合分数。这种方法掩盖了一个根本性问题：从特定样本中学习到的知识能在多大程度上泛化到其他样本？这种逐样本泛化能力，类似于人类认知中的类比学习，捕捉了从一个示例中提取的知识能迁移多远，但在标准基准测试中却不可见。我们提出了“泛化光谱”这一评估框架，旨在揭示这一隐藏维度。对于每个训练示例，我们构建一组受控的测试变体，按迁移距离递增排列：从精确回忆、跨语言实现迁移、完整叙事重构下的上下文迁移、类别匹配的领域内问题，到无配对基线。通过跟踪不同距离上的性能表现，我们不仅能揭示算法是否泛化，还能揭示其泛化程度和模式。

    arXiv:2606.25450v2 Announce Type: replace-cross  Abstract: Traditional evaluations measure a learning algorithm's final performance on an i.i.d. test set, reducing learning to a single aggregate score. This approach obscures a fundamental question: to what extent does learning from a specific example generalize to others? Such per-sample generalization, akin to learning by analogy in human cognition, captures how far the knowledge extracted from one example can transfer, yet remains invisible to standard benchmarks. We introduce the Generalization Spectrum, an evaluation framework designed to expose this hidden dimension. For each training example, we construct a controlled suite of test variants arranged by increasing transfer distance, from exact recall to implementation transfer across languages, context transfer under complete narrative re-framing, category-matched in-domain problems, and an unpaired baseline. By tracking performance across these distances, we reveal not just wheth
    
[^167]: 基于几何锚定的传输框架用于无样本类增量学习

    Geometry-Anchored Transport Framework for Exemplar-Free Class-Incremental Learning

    [https://arxiv.org/abs/2606.25347](https://arxiv.org/abs/2606.25347)

    提出了一种将特征传输作为内生训练约束的几何锚定框架，通过解析几何锚点和拓扑感知演化目标，有效缓解了无样本类增量学习中的表征漂移和流形退化问题。

    

    无样本类增量学习（EFCIL）需要在不断变化的特征空间中维持稳定的决策边界。虽然保持类别条件高斯统计量提供了一种原则性的分类策略，但这些参数化汇总对非各向同性的表征漂移仍然敏感。现有方法通常采用解耦的后验范式来跨任务传输这些统计量：在缺乏明确几何约束的情况下优化骨干网络会扭曲旧有流形，限制回溯对齐的精度。在本文中，我们将特征传输公式化为一种内生的训练约束，而非单独的后任务步骤，提出了几何锚定传输框架。首先，我们通过马氏距离对齐回归推导出解析几何锚点，以缓解宏观非各向同性漂移。其次，我们引入拓扑感知演化目标，该目标正则化局部流形退化。

    arXiv:2606.25347v2 Announce Type: replace  Abstract: Exemplar-free class-incremental learning (EFCIL) requires stable decision boundaries within a shifting feature space. While maintaining class-conditional Gaussian statistics provides a principled classification strategy, these parametric summaries remain sensitive to anisotropic representation drift. Existing methods often transport these statistics across tasks using a decoupled, post-hoc paradigm: optimizing a backbone without explicit geometric constraints can distort the legacy manifold, limiting the precision of retroactive alignment. In this paper, we formulate feature transport as an endogenous training constraint rather than a separate post-task step, presenting the Geometry-Anchored Transport Framework. First, we derive an Analytic Geometric Anchor via Mahalanobis-aligned regression to mitigate macroscopic anisotropic drift. Second, we introduce a Topology-Aware Evolution objective that regularizes localized manifold degrada
    
[^168]: 通过内存高效等变变压器实现可扩展的肽设计

    Scalable Peptide Design via Memory-Efficient Equivariant Transformer

    [https://arxiv.org/abs/2606.25006](https://arxiv.org/abs/2606.25006)

    提出了一种名为MEET的内存高效等变Transformer，通过优化几何计算与特征流设计，实现了可扩展的原子级肽序列与结构协同生成。

    

    arXiv:2606.25006v2 公告类型：替换 摘要：针对特定靶标的肽设计需要在全原子几何约束下进行序列与结构的协同设计。潜在生成框架通过将细粒度原子结构压缩为块级潜在表示，并在紧凑的潜在空间中进行条件生成，为这一问题提供了有效途径。然而，此类系统的可扩展性在很大程度上取决于其编码、解码和去噪组件中使用的几何主干网络。我们引入了MEET（内存高效等变变压器），这是一种用于可扩展原子级肽建模的E(3)等变主干网络。MEET保持耦合的不变标量与等变向量特征流，同时围绕内存高效注意力机制重新设计几何计算。它通过全局坐标聚合初始化向量特征，通过增强的查询与键点积纳入成对距离，并注入共价键信息。

    arXiv:2606.25006v2 Announce Type: replace  Abstract: Target-specific peptide design requires sequence and structure co-design under full atom geometric constraints. Latent generative frameworks offer an effective route for this problem by compressing fine grained atomic structures into block level latent representations and performing conditional generation in a compact latent space. However, the scalability of such systems depends heavily on the geometric backbone used throughout their encoding, decoding, and denoising components. We introduce MEET (Memory Efficient Equivariant Transformer), an E(3) equivariant backbone for scalable atomistic peptide modeling. MEET maintains coupled invariant scalar and equivariant vector feature streams, while reformulating geometric computation around memory efficient attention. It initializes vector features through global coordinate aggregation, incorporates pairwise distances through augmented query and key dot products, and injects covalent bond
    
[^169]: MORL-A2C：用于优化MOPI-HFRS中健康性的多目标强化学习重排序器

    MORL-A2C: Multi-Objective Reinforcement Learning Reranker for Optimizing Healthiness in MOPI-HFRS

    [https://arxiv.org/abs/2606.23603](https://arxiv.org/abs/2606.23603)

    本文提出MORL-A2C，一种基于多目标强化学习的重排序器，通过优势演员-评论家算法和标量化奖励将序列决策引入食品推荐，以联合优化用户偏好与营养健康。

    

    摘要：不健康的饮食行为仍然是美国一个持续存在的公共卫生问题，而推荐系统优先考虑用户偏好却忽视营养健康，进一步加剧了这一问题。本工作所依托的多目标个性化可解释健康感知食品推荐系统（MOPI-HFRS），通过基于帕累托优化的方法联合优化偏好、健康性和多样性来应对这一挑战。然而，该方法依赖于静态的逐步骤权衡解，未能捕捉饮食决策的序列性特征。我们提出了MORL-A2C，这是对MOPI-HFRS的序列决策扩展，专门针对健康-偏好轴。利用冻结的图神经网络嵌入，MORL-A2C将推荐问题形式化为一个K步重排序问题，采用带有标量化相关性/健康奖励的优势演员-评论家算法。该策略通过针对点积的行为克隆进行热启动。

    arXiv:2606.23603v2 Announce Type: replace  Abstract: Unhealthy dietary behavior continues to be a persistent public health issue in the United States, exacerbated by recommendation systems that prioritize user preference without considering nutritional health. The Multi-Objective Personalized Interpretable Health-aware Food Recommendation System (MOPI-HFRS), from which this work extends, addresses this by jointly optimizing preference, health, and diversity through Pareto-based optimization. However, this approach relies on static, per-step tradeoff solutions that fail to capture the sequential nature of dietary decision-making. We introduce MORL-A2C, a sequential decision-making extension to MOPI-HFRS targeting the health-preference axis. Leveraging frozen GNN embeddings, MORL-A2C formulates recommendation as a K-step reranking problem using an Advantage Actor-Critic algorithm with a scalarized relevance/health reward. The policy is warm-started via behavior cloning against a dot-prod
    
[^170]: 如何分配仿真到现实迁移的预算？

    How Should a Simulation-to-Reality Transfer Budget Be Spent?

    [https://arxiv.org/abs/2606.22062](https://arxiv.org/abs/2606.22062)

    少量真实数据用于系统辨识能有效弥合仿真到现实的迁移差距，且效果优于扩大域随机化范围。

    

    仿真到现实的迁移，通常称为sim-to-real迁移，是机器人学习中的一个核心挑战。然而，对系统进行更精确测量与在更广泛的仿真动力学范围内训练之间的权衡，至今仍未被充分理解。在这项工作中，我们重点关注了真实机器人测量时间在系统辨识与域随机化之间的分配。我们在一个受控的仿真到仿真摆锤环境中研究了这一权衡，其中用一个隐藏参数模型代表物理机器人，实验通过扫描辨识轨迹与随机化分布宽度来探索。在我们测试的各种现实差距和噪声水平下，测量预算发挥了主要作用。少量的辨识轨迹就弥合了大部分迁移差距，而一旦有了任何真实数据，在估计参数而非扩大的随机化范围内训练的策略表现最佳。

    arXiv:2606.22062v2 Announce Type: replace-cross  Abstract: Simulation-to-reality transfer, often called sim-to-real transfer, is a central challenge in robot learning. Yet, the tradeoff between measuring a system more accurately and training over a broader range of simulated dynamics is still poorly understood. In this work, we focused on the allocation of real-robot measurement time between system identification and domain randomization. We studied this tradeoff in a controlled sim-to-sim pendulum setting, where a hidden-parameter model stands in for the physical robot, and the experiment sweeps identification rollouts against the width of the randomization distribution. Across the reality gaps and noise levels we tested, the measurement budget did most of the work. A small number of identification rollouts closed most of the transfer gap, and once any real data was available, policies performed best when trained at the estimated parameters rather than over a widened randomization ban
    
[^171]: 深度学习表示中的几何与信息压缩

    Geometric and Information Compression of Representations in Deep Learning

    [https://arxiv.org/abs/2606.21593](https://arxiv.org/abs/2606.21593)

    本文发现深度学习中表示的低互信息与几何压缩之间并不存在稳定的对应关系，二者关系比通常认为的更复杂。

    

    深度神经网络将输入数据转化为潜在表示，以支持多种下游任务。这些表示可以从信息论和几何维度进行刻画，但两者之间的关系仍不明确。一个核心未解问题是：输入与表示之间的低互信息（MI）是否必然意味着几何上压缩的潜在空间，反之亦然。我们通过将类内聚类作为几何压缩的度量，并在条件熵瓶颈（CEB）网络和连续丢弃网络中采用理论上可靠的MI估计来研究这一问题。我们在受控噪声注入方案下，评估了分类任务中MI、几何压缩与泛化之间的相互作用。研究结果表明，低MI并不稳定地对应于几何压缩，二者之间的联系比通常认为的更为复杂。

    arXiv:2606.21593v2 Announce Type: replace  Abstract: Deep neural networks transform input data into latent representations that support a wide range of downstream tasks. These representations can be characterized along information-theoretic and geometric dimensions, but their relationship remains poorly understood. A central open question is whether low mutual information (MI) between inputs and representations necessarily implies geometrically compressed latent spaces and vice versa. We investigate this question using class-wise clustering as a measure of geometric compression and theoretically sound MI estimation in conditional entropy bottleneck (CEB) networks and continuous dropout networks. We evaluate the interplay between MI, geometric compression, and generalization on classification tasks under controlled noise injection schemes. Our findings show that low MI does not reliably correspond to geometric compression, and that the connection between the two is more nuanced than oft
    
[^172]: GRAG：面向个性化对话系统的通用响应增强生成框架

    GRAG: Generic Response-Augmented Generation Framework for Personalized Conversational Systems

    [https://arxiv.org/abs/2606.21097](https://arxiv.org/abs/2606.21097)

    GRAG框架通过解耦个性化和上下文基础两个目标，利用离线通用响应来提升资源受限或隐私敏感环境中个性化对话系统的性能。

    

    arXiv:2606.21097v2 公告类型：替换 摘要：在资源受限或隐私敏感的环境中部署高性能个性化对话代理仍然是一个重大挑战。我们识别出现有方法中的一个根本瓶颈：当前的训练范式将个性化和上下文基础视为一个单一的、整体的学习问题。在这种范式下，语言模型被迫同时处理“说什么”（内容基础）和“如何以用户特定的方式说”（个性化），这引入了显著的计算和优化挑战。因此，上下文基础常常为了个性化一致性而被牺牲，反之亦然，导致生成的响应要么与对话历史关联薄弱，要么个性化不足。在这项工作中，我们提出了通用响应增强生成（GRAG）框架，通过利用来自高性能模型的离线通用响应来解耦这些相互竞争的目标。

    arXiv:2606.21097v2 Announce Type: replace  Abstract: Deploying highly capable personalized conversational agents in resource-constrained or privacy-sensitive environments remains a significant challenge. We identify a fundamental bottleneck in the existing approaches: current training paradigms treat personalization and grounding as a single monolithic learning problem. Under these paradigms, language models are forced to simultaneously address what to say (content grounding) and how to say it in a user-specific way (personalization), which introduces significant computational and optimization challenges. Consequently, contextual grounding is often sacrificed for persona adherence, or vice versa, resulting in responses that are either weakly grounded in the conversational history or insufficiently personalized. In this work, we propose the Generic Response-Augmented Generation (GRAG) framework that decouples these competing objectives by leveraging offline, generic responses from high-
    
[^173]: A-Evolve-Training：一个300亿参数模型的自主后训练

    A-Evolve-Training: Autonomous Post-Training of a 30B Model

    [https://arxiv.org/abs/2606.20657](https://arxiv.org/abs/2606.20657)

    该论文提出了一个无需人工干预的自主后训练系统，能在数周内对300亿参数模型进行多轮训练，并自主发现并修正了优化策略偏差，使模型性能接近人类顶尖水平。

    

    arXiv:2606.20657v2 公告类型：替换 摘要：对前沿模型进行后训练通常需要数周的人工工作：提出数据和配方更改、启动运行、阅读评估结果、决定保留哪些内容。我们报告了一个自主系统，该系统无需人工干预即可运行这一循环，在数周内对300亿参数的Nemotron模型进行了四轮后训练。自主生成的模型在公开的NVIDIA Nemotron-Reasoning挑战排行榜上达到了0.86的保留得分，而人类最佳提交得分为0.87，截至撰写时在约4000个参赛者中排名第8。比数字更引人注目的是：该系统检测到其自身的开发指标已停止追踪最弱领域的外部性能——候选方案将开发指标推至历史新高，但未能推动外部目标——于是它修订了自己的搜索策略，不再最大化开发指标，而是寻求在改善外部目标的同时降低现已具有误导性的代理指标的干预措施。我们将此视为可审计的直接证据，表明一个规模化系统能够自主调整其优化策略。

    arXiv:2606.20657v2 Announce Type: replace  Abstract: Post-training a frontier model is normally weeks of human work: proposing data and recipe changes, launching runs, reading evals, deciding what to keep. We report an autonomous system that runs this loop with no human in the loop, post-training a 30B Nemotron across four rounds over multiple weeks. The autonomously produced model reaches a held-out score of 0.86 against the top human submission's 0.87 on the public NVIDIA Nemotron-Reasoning Challenge leaderboard, placing 8th of ~4000 at the time of writing. More striking than the number: the loop detected that its own dev metric had stopped tracking external performance on the weakest domain -- candidates drove dev to record highs without moving the external target -- and revised its own search policy, no longer maximizing dev but seeking interventions that lowered the now-misleading proxy while improving the external target. We treat this as direct, auditable evidence that a scaled 
    
[^174]: 提示、规划、提取：用于从临床叙述中提取肺部病理信息的零样本智能大语言模型工作流

    Prompt, Plan, Extract: Zero-Shot Agentic LLMs Workflows for Lung Pathology Extraction from Clinical Narratives

    [https://arxiv.org/abs/2606.19852](https://arxiv.org/abs/2606.19852)

    本研究提出了一种零样本智能大语言模型工作流，用于从临床叙述中自动提取肺部病理信息，在无需人工标注的情况下，最佳模型达到了0.8的微平均F1分数。

    

    arXiv:2606.19852v2 公告类型：替换 摘要：从病理报告中提取信息对于癌症分期、肿瘤登记人群至关重要。然而，关键数据仍嵌入在叙述性报告中，使得手动提取工作劳动密集且易出错。传统的监督式自然语言处理管道通过完全监督的命名实体识别和关系抽取来解决这一问题，但需要昂贵的手动标注，并且当上游实体被遗漏时会出现级联故障。在本研究中，我们开发了一种零样本智能工作流，并评估了五种开源生成式大语言模型，以从肺切除病理报告中填充13个美国病理学家学会的概要字段。我们使用一种新颖的、与登记标准对齐的评估框架，将它们与最先进的监督式GatorTron NER-RE基线进行了比较。基线实现了0.960的微平均F1分数，而最佳零样本模型（GPT-OSS-20B）实现了0.8的微平均F1分数。

    arXiv:2606.19852v2 Announce Type: replace  Abstract: Information extraction from pathology reports is essential for cancer staging, tumor registry population. Yet key data remains embedded in narrative reports, making manual extraction labor-intensive and error-prone. Traditional supervised Natural Language Processing pipelines address this through fully supervised Named Entity Recognition and Relation Extraction, but require expensive manual annotation and suffer cascading failures when upstream entities are missed. In this study, we developed a zero-shot, agentic workflow, and evaluated five open-source generative Large Language Models (LLMs) to populate 13 College of American Pathologists synoptic fields from lung resection pathology reports. We compared them against a state-of-the-art supervised GatorTron NER-RE baseline using a novel, registry-aligned evaluation framework. The baseline achieved Micro-F1of 0.960, while the best zero-shot model (GPT-OSS-20B) achieved Micro-F1 of 0.8
    
[^175]: MetaboNet-Bench：1型糖尿病血糖预测的多模态基准

    MetaboNet-Bench: A Multi-modal Benchmark for Glucose Forecasting in Type 1 Diabetes

    [https://arxiv.org/abs/2606.18640](https://arxiv.org/abs/2606.18640)

    提出了一个用于1型糖尿病血糖预测的多模态开源基准，整合了血糖、胰岛素和碳水化合物数据，以解决现有基准缺失和单一模态限制的问题。

    

    arXiv:2606.18640v2 公告类型：替换 摘要：血糖预测算法是1型糖尿病血糖控制管理中的一个重要方面。迄今为止，研究社区已经开发了大量用于预测的算法和模型。然而，人们普遍认识到，缺乏标准化的模型性能评估基准使得公平比较变得困难，并阻碍了进一步的创新，因此基准标准化迫在眉睫。此外，许多已发表的血糖预测算法仅限于连续血糖监测（CGM）数据，忽略了其他多模态信号，如胰岛素剂量和碳水化合物摄入量。在此，我们引入了MetaboNet-Bench，这是一个针对1型糖尿病患者的多模态血糖预测基准，它提供了一个可扩展的开源评估框架，用于比较利用血糖、胰岛素和碳水化合物数据的血糖预测算法。然后，我们通过基准测试几个最近发表的算法来展示其实用性。

    arXiv:2606.18640v2 Announce Type: replace  Abstract: Glucose forecasting algorithms are an important aspect of glycemic control management in type 1 diabetes. So far, the research community has developed numerous algorithms and models for forecasting. However, it is well-recognized that the lack of standardized model performance evaluation benchmarks makes fair comparison difficult and hinders further innovation, and thus benchmark standardization is in urgent need. Furthermore, many published glucose forecasting algorithms are limited to CGM data alone, ignoring other multimodal signals such as insulin dosing and carbohydrate intake. Here, we introduce MetaboNet-Bench, a benchmark for multimodal glucose forecasting for patients with type 1 diabetes that provides an extensible open-source evaluation framework for comparison of glucose forecasting algorithms that leverage glucose, insulin, and carbohydrate data. We then demonstrate its utility by benchmarking several recently published 
    
[^176]: 签名过滤：大型语言模型中统计水印检测的轻量级增强方法

    Signature filtering: a lightweight enhancement for statistical watermark detection in large language models

    [https://arxiv.org/abs/2606.18430](https://arxiv.org/abs/2606.18430)

    提出了一种轻量级检测时模块——签名过滤，通过学习并移除使水印检测不可靠的“签名”标记，在不改变水印嵌入和文本生成过程的情况下显著提升了统计水印检测的性能，并提供了理论界限。

    

    arXiv:2606.18430v2 公告类型：替换 摘要：统计水印帮助组织归因大型语言模型（LLM）的输出，然而，当水印信号较弱、文本重复或水印被编辑时，现有检测器常常难以有效工作。我们提出了签名过滤，这是一种检测时模块，无需修改水印嵌入和文本生成即可增强水印检测。它学习一小部分“签名”标记，这些标记的存在会使水印测试不可靠，并在检测前移除这些标记。签名是通过在小型训练集上求解混合整数线性规划获得的，其约束条件旨在最大化真阳性率。我们还在几种攻击者模型（色盲、色适应和分布相关）下推导了有限样本和渐近界。在四种知名水印族（Kgw、Sweet、Unigram、Exp）、四个基准语料库（C4、MBPP、HumanEval、Code-Search-Net）和六个LLM（Opt-1.3）上进行了验证。

    arXiv:2606.18430v2 Announce Type: replace  Abstract: Statistical watermarks help organizations attribute large language model (LLM) outputs, yet existing detectors often struggle when watermark signals are weak, texts are repetitive, or watermarks are edited. We propose signature filtering, a detection-time module that enhances watermark detection without modifying watermark embedding and text generation. It learns a small set of ``signature'' tokens whose presence makes watermark tests unreliable, and removes these tokens before detection. The signatures are obtained by solving a mixed-integer linear program on a small training set, with constraints that maximize the true positive rate. We additionally derive finite-sample and asymptotic bounds under several attacker models (color-blind, color-adaptive, and distributionally correlated). On four well-known watermark families (Kgw, Sweet, Unigram, Exp), four benchmark corpora (C4, MBPP, HumanEval, Code-Search-Net), and six LLMs (Opt-1.3
    
[^177]: 数据科学中的表示成本：深度神经网络的基础与拟巴拿赫空间

    Representation Costs in Data Science: Foundations and the Quasi-Banach Spaces of Deep Neural Networks

    [https://arxiv.org/abs/2606.14954](https://arxiv.org/abs/2606.14954)

    本文提出了一个统一框架，通过参数空间正则化器分析数据科学中的表示成本，揭示了参数化方法与其原生函数空间之间的联系，并将核方法、小波和神经网络等经典方法统一为特例。

    

    我们开发了一个通用框架，用于通过参数空间正则化器分析参数化数据拟合方法的表示成本。从这一抽象视角出发，我们定义了任意参数化模型的表示成本，并揭示了它们所诱导的（原生）函数空间。这统一了近期关于数据拟合方法的函数空间视角。我们还证明，在该抽象设定下许多自然结论成立，包括参数化方法在其原生空间上的表示定理。该框架还严格地将参数化方法与其在充分过参数化下的等价非参数描述联系起来。经典方法及其原生空间，如核方法/再生核希尔伯特空间、小波/贝索夫空间以及浅层神经网络/变分空间，均作为我们抽象框架的特例出现。将表示成本研究“公理化”是一个副产品。

    arXiv:2606.14954v3 Announce Type: replace-cross  Abstract: We develop a general framework for analyzing representation costs of parametric data-fitting methods through their parameter-space regularizers. From this abstract perspective, we define representation costs for arbitrary parametric models and reveal their induced (native) function spaces. This unifies recent function-space views of data-fitting methods. We also prove that many natural results hold in this abstract setting, including representer theorems for parametric methods on their native spaces. The framework also rigorously connects parametric methods with their equivalent nonparametric descriptions under sufficient overparameterization. Classical methods and their native spaces, such as kernel methods / reproducing kernel Hilbert spaces, wavelets / Besov spaces, and shallow neural networks / variation spaces emerge as special cases of our abstract framework. A byproduct of "axiomatizing" the study of representation costs
    
[^178]: 何时写入与何时抑制：用于记忆辅助知识编辑的路径专用双适配器

    When to Write and When to Suppress: Route-Specialized Dual Adapters for Memory-Assisted Knowledge Editing

    [https://arxiv.org/abs/2606.14668](https://arxiv.org/abs/2606.14668)

    本文提出一种路径专用的双适配器方法，通过相关性路由器决定何时应用编辑记忆或保留原始知识，从而在知识编辑中实现精确更新与无关行为的保护。

    

    arXiv:2606.14668v3 公告类型：替换 摘要：知识编辑系统必须更新选定的事实，同时保留邻近但无关的行为。本文在记忆辅助设置下研究这一问题，在该设置中，推理时会检索编辑记忆，并且一个参数高效的适配器会纠正模型的对象偏好。我们认为，核心设计问题不仅在于如何写入编辑，还在于何时抑制它。我们引入了 \method{}，一个路径专用的双适配编辑器。一个相关性路由器首先决定一个提示是否应该接收编辑记忆。被路由的提示使用一个编辑适配器，该适配器经过训练以偏好新对象而非原始对象；未被路由的非直接提示则使用一个单独的位置适配器，该适配器经过训练以保留或恢复原始对象偏好。我们在三种包含1000个案例的协议（\cf{}、\zsre{} 和 \mquake{}）上，在相同的记忆协议和两个7B/8B基础模型下评估了 \method{}。在 Llama-3.1-8B-Instruct 上，\method{} 取得了最佳性能。

    arXiv:2606.14668v3 Announce Type: replace  Abstract: Knowledge editing systems must update selected facts while preserving nearby but irrelevant behavior. This paper studies this problem in a memory-assisted setting where an edit memory is retrieved at inference time and a parameter-efficient adapter corrects the model's object preference. We argue that the central design question is not only how to write an edit, but also when to suppress it. We introduce \method{}, a route-specialized dual-adapter editor. A relevance router first decides whether a prompt should receive an edit memory. Routed prompts use an edit adapter trained to prefer the new object over the original object; unrouted non-direct prompts use a separate locality adapter trained to preserve or restore the original-object preference. We evaluate \method{} on three 1,000-case protocols, \cf{}, \zsre{}, and \mquake{}, under the same memory protocol and two 7B/8B base models. On Llama-3.1-8B-Instruct, \method{} obtains the
    
[^179]: 穿越严苛考验：重新评估智能体在陌生环境中的能力

    Running the Gauntlet: Re-evaluating the Capabilities of Agents Beyond Familiar Environments

    [https://arxiv.org/abs/2606.14397](https://arxiv.org/abs/2606.14397)

    本论文提出了GauntletBench基准测试，通过聚焦时间感知、图形理解和3D推理三种未充分探索的能力，并在五个专业应用中设置100个视觉密集型任务，以更全面评估智能体在陌生环境中的泛化能力。

    

    arXiv:2606.14397v2 公告类型：替换 摘要：随着智能体系统持续演进并广泛部署于现实场景，对其能力进行忠实评估的需求日益增长。然而，当前基准测试通常基于流行应用构建，任务相对简单，且仅关注少数能力而忽略更广泛的维度，导致现代智能体在这些测试中表现饱和，无法揭示其局限性。为此，我们提出了GauntletBench——一个基于网络的基准测试，用于评估智能体在挑战性场景中的泛化能力，聚焦于三种未被充分探索的能力（时间感知、图形理解与3D推理），涵盖五个较少涉及的职业应用（视频编辑器、工作流构建器、3D建模器、飞行分析器与电路设计器），每个应用包含20个视觉密集型任务（总计100个任务）。我们的基准测试提供模块化流水线，包含一个兼容多种智能体的环境。

    arXiv:2606.14397v2 Announce Type: replace  Abstract: As agentic systems continue to evolve and are widely deployed in real-world scenarios, there is a growing demand to faithfully evaluate their capabilities. However, current benchmarks are typically built on popular applications with relatively simple tasks and focus on a narrow set of capabilities while overlooking broader dimensions, resulting in saturated performance on modern agents and failing to probe their limitations. To this end, we introduce GauntletBench, a web-based benchmark for evaluating agent generalisation in challenging scenarios, focusing on three underexplored capabilities (temporal perception, graphical understanding, and 3D reasoning), across five less-covered professional applications (Video Editor, Workflow Builder, 3D Modeller, Flight Analyser, and Circuit Designer), each with 20 vision-intensive tasks (100 in total). Our benchmark provides a modular pipeline that comprises an environment compatible with both 
    
[^180]: 立场：让人工智能与我们的抱负对齐，而非与我们的缺陷对齐

    Position: Align AI to Our Aspirations, Not Our Flaws

    [https://arxiv.org/abs/2606.13755](https://arxiv.org/abs/2606.13755)

    本文主张人工智能不应与人类有缺陷的偏好对齐，而应基于事实准确、诚实和合法的客观目标进行训练，将多元性限制在表面层面。

    

    arXiv:2606.13755v2 公告类型：替换-交叉 摘要：我们认为，让人工智能与聚合的人类偏好对齐是错误的追求目标。以当前技术，人们可以训练人工智能去共享硅谷技术乐观主义者、去增长环保主义者、民族保守文化斗士、一党制国家干部或虔诚宗教传统主义者的价值观。但我们不应这样做。人类价值观催生了繁荣或失败的社会——从失败国家和极端不平等，到世界上最富裕民主国家中幸福感下降、政治两极分化和政府功能失调。多元对齐计划正确地诊断出不存在单一的“人类”可以对齐，但如果将其作为主要指令则是危险的。我们认为，人工智能应被训练至不可妥协的客观对齐目标底线——能力，受制于事实准确性、诚实性和合法性的约束，而多元性应停留在表面。

    arXiv:2606.13755v2 Announce Type: replace-cross  Abstract: We argue that aligning AI to aggregated human preferences is the wrong target. With current technology, one can train AIs to share the values of a Silicon Valley techno-optimist, a degrowth environmentalist, a national-conservative culture warrior, a single-party state cadre, or a devout religious traditionalist. We should not. Human values produce societies that thrive or fail on the merits of those values - from failed states and extreme inequality to declining happiness, political polarization, and government dysfunction in the world's wealthiest democracies. The pluralistic-alignment program correctly diagnoses that there is no single "humanity" to align with, but is dangerous if taken as the main directive. We argue that AI should be trained to a non-negotiable floor of objective alignment goals - competence, bounded by the constraints of factual accuracy, honesty, and lawfulness and that pluralism belongs at the surface (
    
[^181]: 面向校准感知的量子电路路由的图强化学习

    Graph Reinforcement Learning for Calibration-Aware Quantum Circuit Routing

    [https://arxiv.org/abs/2606.12816](https://arxiv.org/abs/2606.12816)

    该论文提出了一种利用实时校准数据通过图强化学习进行量子电路路由的方法，在中小型量子电路上显著提升了保真度，平均精确保真度达到0.727，远超基线方法。

    

    arXiv:2606.12816v3 公告类型：替换-交叉 摘要：量子电路路由是为噪声中等规模量子处理器编译程序的关键步骤。即使通过标准开销指标看起来高效的路径，当它们经过校准不良的耦合器时，仍然可能损失保真度。我们研究了一种校准感知的图强化学习路由器，它利用同一天的IBM Heron r2校准数据来选择硬件边缘的SWAP操作。我们使用近端策略优化训练该策略，并通过九个慕尼黑量子工具包（MQT）基准电路和三个校准快照，以精确模拟的保真度对其进行评估。在这些评估中，合并的平均精确保真度为0.727，而SABRE-best20为0.440，目标感知SABRE为0.481。我们观察到，保真度的提升伴随着更高的路由双量子比特计数，并且集中在5量子比特和8量子比特电路族中；在固定的树状动作图下，所有10量子比特族都更倾向于SABRE-best20。总体而言，我们的方法在中小型电路上显著优于基线。

    arXiv:2606.12816v3 Announce Type: replace-cross  Abstract: Quantum circuit routing is a key step in compiling programs for noisy intermediate-scale quantum processors. Routes that appear efficient by standard overhead metrics can still lose fidelity when they pass through poorly calibrated couplers. We study a calibration-aware graph reinforcement-learning router that uses same-day IBM Heron r2 calibration data to choose hardware-edge SWAPs. We train the policy with proximal policy optimization and evaluate it with exact simulated fidelity across nine Munich Quantum Toolkit (MQT) Bench circuits and three calibration snapshots. Across these evaluations, pooled mean exact fidelity is $0.727$, compared with $0.440$ for SABRE-best20 and $0.481$ for target-aware SABRE. We observed that fidelity gains came with higher routed two-qubit counts and were concentrated in 5 qubit and 8 qubit circuit families; under the fixed tree action graph, all 10 qubit families favored SABRE-best20. Overall, o
    
[^182]: SymQNet：面向低延迟自适应哈密顿学习的摊销采集方法

    SymQNet: Amortized Acquisition for Low-Latency Adaptive Hamiltonian Learning

    [https://arxiv.org/abs/2606.12808](https://arxiv.org/abs/2606.12808)

    本文提出SymQNet，一种通过离线学习后验条件采集策略并在线快速执行，从而大幅降低自适应哈密顿学习延迟的摊销强化学习方法。

    

    自适应哈密顿学习是校准和表征量子器件的核心。在自适应控制器中，选择下一个实验本身就是一个计算过程。贝叶斯设计规则在每次后验更新后都会重新计算，这一步骤可能需要数秒时间。在数百次实验轮次中，这些秒数会累积成为自适应过程显著的时钟时间成本。我们提出了SymQNet，一种用于低延迟自适应哈密顿学习的摊销强化学习方法。SymQNet离线学习一个基于后验条件的采集策略，然后在线使用快速策略前向传播，同时保留贝叶斯后验反馈。在横向场伊辛模型基准测试中，与有界费舍尔信息搜索和有界两步贝叶斯主动学习（BALD）相比，SymQNet显著降低了采集延迟。在五个量子比特上，相对于这两种方法，其仅采集决策延迟分别降低了47.1倍和72.6倍。

    arXiv:2606.12808v3 Announce Type: replace-cross  Abstract: Adaptive Hamiltonian learning is central to calibrating and characterizing quantum devices. In an adaptive controller, choosing the next experiment is itself a computation. Bayesian design rules are recomputed after every posterior update, and that step can take seconds. Across hundreds of shots, those seconds become a significant wall-clock cost for adaptivity. We introduce SymQNet, an amortized reinforcement-learning approach for low-latency adaptive Hamiltonian learning. SymQNet learns a posterior-conditioned acquisition policy offline, then uses a fast policy forward pass online while retaining Bayesian posterior feedback. On transverse-field Ising benchmarks, SymQNet substantially reduces acquisition latency relative to bounded Fisher-information search and bounded two-step Bayesian active learning by disagreement (BALD). At five qubits, it reduces acquisition-only decision latency by $47.1\times$ and $72.6\times$ relative
    
[^183]: 贝尔曼充分信息复杂度

    Bellman-sufficient Information Complexity

    [https://arxiv.org/abs/2606.11171](https://arxiv.org/abs/2606.11171)

    本文提出了一个名为贝尔曼充分信息复杂度的框架，通过状态表示和信息指数在序贯决策中实现信息复杂度与风险的上界和下界匹配。

    

    arXiv:2606.11171v5 公告类型：替换 摘要：我们提出了贝尔曼充分信息复杂度，这是一个用于序贯决策的形式化表示框架。原始基准是一个固定真实环境空间Ω，包含无限制的非预测算法。内在对象是一个贝尔曼充分状态表示，作为一种交互式的充分统计量概念，以及一个信息指数Y=χ(Ω)，通常是最优决策或价值对象，而非完整环境。在上界方面，学习被组织为在充分状态上的动态规划，配备了对数信息势能用于指数。在下界方面，贝尔曼-法诺证书使用相同的状态表示和信息指数，但为信息增益和幽灵质量传播独立的贝尔曼递归。因此，核心匹配陈述是一个条件性的贝尔曼信息-风险三明治：当...

    arXiv:2606.11171v5 Announce Type: replace  Abstract: We develop Bellman-sufficient information complexity, a formal representation-level framework for sequential decision making. The primitive benchmark is a fixed-truth environment space $\Omega$ with unrestricted nonanticipating algorithms. The intrinsic object is a Bellman-sufficient state representation, serving as an interactive notion of sufficient statistics, together with an information index $Y=\chi(\Omega)$, often the optimal decision or value object rather than the full environment. On the upper-bound side, learning is organized as a dynamic program on the sufficient state, equipped with a logarithmic information potential for the index. On the lower-bound side, a Bellman-Fano certificate uses the same state representation and information index, but propagates separate Bellman recursions for information gain and ghost mass. The central matching statement is therefore a conditional Bellman information-risk sandwich: when the l
    
[^184]: 符号推理框架触发多智能体大语言模型系统中的记忆介导生态系统动力学

    Symbolic Reasoning Frameworks Trigger Memory-Mediated Ecosystem Dynamics in Multi-Agent LLM Systems

    [https://arxiv.org/abs/2606.07552](https://arxiv.org/abs/2606.07552)

    本研究表明，在多智能体大语言模型系统中，单个智能体注入符号推理框架的微小扰动，通过累积记忆和交互涌现出稳定的、与条件相关的优胜者生态系统，揭示了记忆介导的系统动力学机制。

    

    大语言模型作为策略性智能体时表现出一种规避风险的“乌龟”偏好。我们证明，向一个智能体中注入符号推理框架作为每轮反思提示，会作为一个微小扰动，其后果并非逐决策显现，而是涌现性的：该智能体的风险姿态在孤立状态下保持不变，但在一系列积累记忆和多智能体交互的过程中，系统条件会稳定为不同的、与条件相关的优胜者生态系统。在一个7玩家的战国纵横外交变体游戏中（61局游戏，6种条件），优胜者分布在四个主要条件中差异显著（41局游戏；置换检验综合p值约0.001）：对照组→燕国（7/11）；易经筮法→燕国/楚国共主且秦国完全被压制（0/10）；塔罗牌→秦国（5/10）；乱序文本消融→齐国（5/10）。乱序→齐国吸引子具有鲁棒性（vs. 合并组和单独对照组，p值分别为0.006和0.012）；塔罗牌→秦国的分母相关。

    arXiv:2606.07552v2 Announce Type: replace-cross  Abstract: Large language models exhibit a risk-averse "turtle" bias as strategic agents. We show that injecting a symbolic reasoning framework as a per-round reflective prompt into one agent acts as a small perturbation whose consequences are not per-decision but emergent: the agent's risk posture is unchanged in isolation, yet over a campaign of accumulating memory and multi-agent interaction the conditions settle into distinct, condition-associated winner ecosystems. In a 7-player Warring States Diplomacy variant (61 games, 6 conditions), the winner distribution differs sharply across the four primary conditions (41 games; permutation omnibus p approximately 0.001): control -> Yan (7/11); I-Ching yarrow -> Yan/Chu co-dominance with Qin fully suppressed (0/10); Tarot -> Qin (5/10); scrambled-text ablation -> Qi (5/10). The scrambled->Qi attractor is robust (vs. pooled and control alone, p = 0.006 and 0.012); tarot->Qin is denominator-de
    
[^185]: 基于深度强化学习的加密货币市场动态多对交易策略

    Dynamic Multi-Pair Trading Strategy in Cryptocurrency Markets with Deep Reinforcement Learning

    [https://arxiv.org/abs/2606.04574](https://arxiv.org/abs/2606.04574)

    本研究通过分层配对选择方法和专有执行模型，结合深度强化学习，显著提升了加密货币市场中配对交易的稳健性与收益表现。

    

    本研究旨在探讨深度强化学习作为专门执行覆盖层，能否增强高波动性加密货币市场中的配对交易。尽管经典配对交易策略在传统股票市场中已证明成功，但在高方差环境中常表现出僵化性，并面临严重的发散风险。为应对这一需求，本研究引入了新颖概念。为构建稳健系统，我们开发了分层的“筛选-排序”配对选择方法，以及专有的“固定风险、自适应均值”执行模型。该系统采用带有长短期记忆层的近端策略优化智能体，在严格的确定性风险管理边界内控制执行决策。基于币安USD-M期货市场1小时间隔数据的评估显示，优化后的强化学习策略在样本外测试中实现了...

    arXiv:2606.04574v2 Announce Type: replace  Abstract: This study aims to determine whether the application of Deep Reinforcement Learning (DRL) as a specialized execution overlay can enhance pair trading in highly volatile cryptocurrency markets. Although classical implementations of the strategy have proven successful in traditional equities, they frequently exhibit rigidity and suffer from severe divergence risks when applied to high-variance environments. To address this need, this research introduces novel concepts. To construct a robust system, we developed a hierarchical "Filter-then-Rank" pair selection methodology and a proprietary "Fixed Risk, Adaptive Mean" execution model. The system employs a Proximal Policy Optimization (PPO) agent with a Long Short-Term Memory (LSTM) layer to govern execution decisions within strict deterministic risk management boundaries. Evaluated on 1-hour interval data from the Binance USD-M Futures market, the optimized RL policy achieved an out-of-s
    
[^186]: 随机森林中应该有多少棵树？一种结合高原搜索与Optuna集成的改进方法

    How Many Trees in a Random Forest? A Revisited Approach with Plateau Search and Optuna Integration

    [https://arxiv.org/abs/2606.03549](https://arxiv.org/abs/2606.03549)

    本文提出一种集成高原搜索与Optuna的新方法，通过监控袋外分数的相对变化来自适应确定随机森林的最小足够树数量，无需预设搜索范围且避免早停。

    

    arXiv:2606.03549v2 公告类型：替换 摘要：随机森林的超参数优化（HPO）在调整树的数量时面临一个特殊困难：预测分数通常随集成规模的增大而单调提升，因此诸如树结构帕尔森估计器（TPE）和Hyperband等标准方法需要预先定义搜索范围，并且往往将估计值推向其右边界。早停策略避免了固定这样的范围，但对分数噪声敏感且容易过早停止。为了解决这一问题，我们提出了一种集成的基于三元组的高原搜索算法，该算法将树的数量从直接的TPE搜索空间中移除，同时仍然利用在HPO试验中积累的信息。该方法通过监控三个森林大小之间袋外（OOB）分数的相对变化，并相应地移动这个三元组，从而自适应地追踪一个接近最小的足够集成规模。这产生了一种自动化且用户可解释的流程。

    arXiv:2606.03549v2 Announce Type: replace  Abstract: Hyperparameter optimization (HPO) for Random Forest faces a specific difficulty in tuning the number of trees: the predictive score typically improves monotonically with ensemble size, so standard methods such as Tree-structured Parzen Estimator (TPE) and Hyperband require a predefined search range and often drive the estimate toward its right boundary. Early-stopping strategies avoid fixing such a range, but can be sensitive to score noise and prone to premature stopping. To address this, we propose an integrated triplet-based plateau-search algorithm that removes the number of trees from the direct TPE search space and still exploits information accumulated across HPO trials. The method adaptively tracks a near-minimal sufficient ensemble size by monitoring relative changes in the out-of-bag (OOB) score across a triplet of forest sizes and shifting this triplet accordingly. This yields an automated and user-interpretable procedure 
    
[^187]: 超越独立操纵：基于同伴模仿的个体公平感知策略分类

    Beyond Independent Manipulation: Individual Fairness-aware Strategic Classification with Peer Imitation

    [https://arxiv.org/abs/2606.00827](https://arxiv.org/abs/2606.00827)

    本文提出个体公平感知策略分类框架，通过建模同伴模仿行为解决了传统策略分类在个体公平要求下无法准确描述代理人相互依赖操纵的问题。

    

    策略分类研究的是代理人为获得预测模型有利决策而操纵自身特征的场景。现有的公平感知策略分类方法主要关注群体公平，且通常假设代理人是独立响应的。然而，当需要个体公平时——即确保相似个体获得相似结果——代理人的操纵行为会变得相互依赖：代理人偏好的操纵方式取决于邻近个体的结果。这导致了经典策略分类公式与公平感知决策场景之间的不匹配，因为独立模型不再能准确描述策略性操纵。为解决此问题，我们提出了个体公平感知策略分类框架，该框架建模了由个体公平引发的同伴驱动操纵行为，其中代理人会模仿附近获得正面决策的同伴以获取有利结果。

    arXiv:2606.00827v3 Announce Type: replace-cross  Abstract: Strategic classification (SC) investigates scenarios where agents manipulate their features to obtain favorable decisions from predictive models. Existing fairness-aware SC approaches primarily focus on group fairness and typically assume that agents respond independently. However, when individual fairness is required, ensuring similar individuals receive similar outcomes, agents' manipulation becomes interdependent: an agent's preferred manipulation depends on the neighborhoods' outcomes. This induces a mismatch between classical SC formulations and fairness-aware decision settings, where independent models no longer accurately characterize strategic manipulations. To address this issue, we introduce individual fairness-aware strategic classification (IFSC), a framework that models peer-driven manipulation arising from individual fairness, where agents imitate nearby positively decided peers to obtain favorable outcomes. IFSC 
    
[^188]: CALIBURN：基于运行校准的流式入侵检测与依赖机制的共形风险控制

    CALIBURN: Operationally Calibrated Streaming Intrusion Detection with Regime-Dependent Conformal Risk Control

    [https://arxiv.org/abs/2605.24696](https://arxiv.org/abs/2605.24696)

    本文提出CALIBURN，通过将运行约束（如告警预算、成本）直接嵌入阈值选择流程，而非依赖标签调优，实现了可操作化的流式入侵检测告警系统。

    

    arXiv:2605.24696v2 公告类型：替换交叉 摘要：流式入侵检测系统必须在有限内存下持续处理流量，但大多数系统将告警阈值选择作为后期调优问题，这与实际生产环境不兼容——在生产中，运维人员需预先承诺告警预算、误分类成本和服务水平目标。我们提出CALIBURN，一种流式告警流水线，其决策阈值直接源自这些运行输入，而非依赖标签的搜索过程。CALIBURN在单一流式基座上构建五层架构：截断贝叶斯在线变点检测；将后验概率等渗校准为条件攻击概率；基于运维成本的代价敏感阈值设定；共形风险控制（CRC）包装器，在可交换性假设下将告警预算α映射为误报有界阈值；以及源自站点可靠性工程的多窗口燃烧速率告警。每个层级均为成熟技术，本工作的贡献在于其系统性整合。

    arXiv:2605.24696v2 Announce Type: replace-cross  Abstract: Streaming intrusion detection systems must process flows continuously under bounded memory, yet most leave alerting-threshold selection as a post-hoc tuning problem incompatible with production, where operators commit in advance to alert budgets, misclassification costs, and Service Level Objectives. We present CALIBURN, a streaming alerting pipeline that derives its decision threshold from these operational inputs rather than a label-dependent search. CALIBURN composes five layers on one streaming substrate: truncated Bayesian online change-point detection; isotonic calibration of the posterior to a conditional attack probability; cost-sensitive thresholding from operator costs; a Conformal Risk Control (CRC) wrapper mapping an alert budget alpha to a false-positive-bounded threshold under exchangeability; and multi-window burn-rate alerting from Site Reliability Engineering. Each layer is established; the contribution is the 
    
[^189]: LLMTabBench：从零样本到少样本场景下评估大语言模型在二值表格分类任务中的表现

    LLMTabBench: Evaluating LLMs on Binary Tabular Classification From Zero to Few Shots

    [https://arxiv.org/abs/2605.24417](https://arxiv.org/abs/2605.24417)

    本文提出LLMTabBench基准，系统评估大语言模型在低数据表格分类任务中的零样本与少样本能力，发现其在零样本场景下极具竞争力，有时甚至优于传统模型。

    

    表格数据的监督分类仍是机器学习中的核心任务，但其对大规模标注数据集的依赖限制了其在数据稀缺场景下的适用性。诸如TabPFN等少样本方法通过大规模合成预训练取得了优异性能，但依然需要标注的上下文示例。大语言模型通过任务描述实现零样本和少样本上下文学习，提供了一种更灵活的替代方案，但其在表格数据上的表现仍不稳定。我们提出了LLMTabBench，这是一个用于在低数据条件下评估大语言模型在表格分类任务中表现的基准。该基准研究了大语言模型的先验知识如何与任务描述及少样本示例相互作用，以及随着真实世界和受控合成数据集的数据复杂度增加，性能如何变化。我们发现，大语言模型在零样本设置下极具竞争力，有时甚至能超越那些获得特征表示输入的模型。

    arXiv:2605.24417v2 Announce Type: replace  Abstract: Supervised classification on tabular data remains a central machine learning task, but its dependence on large labeled datasets limits its applicability in data-scarce settings. Few-shot methods such as TabPFN achieve strong performance through large-scale synthetic pretraining, yet still require labeled context examples. Large Language Models (LLMs) offer a more flexible alternative through zero- and few-shot in-context learning from task descriptions, but their behavior on tabular data remains inconsistent. We introduce LLMTabBench, a benchmark for evaluating LLMs on tabular classification under low-data conditions. The benchmark studies how LLM prior knowledge interacts with task descriptions and few-shot examples, and how performance changes with increasing data complexity across real-world and controlled synthetic datasets. We find that LLMs can be highly competitive in zero-shot settings, sometimes outperforming models given fe
    
[^190]: 从单次遍历SGD到数据重用：草图线性回归中的小批量缩放定律

    From One-Pass SGD to Data Reuse: Mini-Batch Scaling Laws in Sketched Linear Regression

    [https://arxiv.org/abs/2605.24316](https://arxiv.org/abs/2605.24316)

    本文通过风险分解，揭示了单次遍历和两种多遍遍历小批量SGD在草图线性回归中的缩放定律，关键创新在于将随机项与采样协议关联，并证明了幂律协方差谱下的源条件缩放定律。

    

    arXiv:2605.24316v2 公告类型：替换 摘要：缩放定律提供了预测误差如何随计算量、模型大小和数据量变化的紧凑描述，但现有理论主要处理单样本SGD或完全数据重用，未能阐明小批量处理的作用。我们研究了在幂律协方差谱和目标参数的源条件假设下，草图线性回归的批量缩放定律。我们分析了单次遍历批量SGD、带替换的多遍遍历批量SGD以及无替换的多遍遍历批量SGD。我们的第一个结果是风险分解：所有三种过程共享相同的不可约项和近似项，而它们的随机项取决于采样协议。单次遍历批量SGD分解为偏差和方差，而两种多遍遍历方法则分解为GD偏差、GD方差以及围绕共同GD参考轨迹的波动项。接着，我们证明了单次遍历和多遍遍历小批量方法的源条件缩放定律。

    arXiv:2605.24316v2 Announce Type: replace  Abstract: Scaling laws provide compact descriptions of how prediction error varies with compute, model size, and data, but existing theory mainly treats single-sample SGD or full data reuse, leaving the role of mini-batching unclear. We study batch scaling laws for sketched linear regression under a power-law covariance spectrum and a source condition on the target parameter. We analyze one-pass batch SGD, multi-pass batch SGD with replacement, and multi-pass batch SGD without replacement. Our first result is a risk decomposition: all three procedures share the same irreducible and approximation terms, while their stochastic terms depend on the sampling protocol. One-pass batch SGD splits into bias and variance, whereas the two multi-pass methods split into GD bias, GD variance, and a fluctuation term around a common GD reference trajectory. We then prove source-condition scaling laws for one-pass and multi-pass mini-batch methods. For one-pas
    
[^191]: 切比雪夫策略与山地车问题：面向低维控制任务的强化学习

    Chebyshev Policies and the Mountain Car Problem: Reinforcement Learning for Low-Dimensional Control Tasks

    [https://arxiv.org/abs/2605.22305](https://arxiv.org/abs/2605.22305)

    本文通过解析求解山地车问题的最优控制，提出切比雪夫策略作为神经网络的轻量级替代，在低维控制任务中大幅降低参数量和遗憾值，并提升性能。

    

    我们解析求解了强化学习中的经典基准问题——山地车问题，并推导出最优控制解，填补了36年来的理论空白。这揭示了两个令人惊讶的发现：最优控制实际上非常简单，但现代强化学习智能体与最优解之间存在巨大差距。受最优控制分析的启发，我们从基本原理出发，提出切比雪夫策略作为一种通用（即密集）的强化学习策略类别。它们可以作为神经网络的直接替代品进行训练，将遗憾值降低4.18倍，同时所需参数减少277倍，从而提升样本效率、可解释性和实时处理能力。切比雪夫策略在更多强化学习任务上进行了评估，包括一个真实世界的非线性运动控制测试平台。在PPO、ARS和REINFORCE算法中，它们始终优于神经网络。我们的结果表明，切比雪夫策略提供了一种极具吸引力且轻量级的替代方案。

    arXiv:2605.22305v3 Announce Type: replace  Abstract: We analytically solve the Mountain Car problem, a canonical benchmark in RL, and derive an optimal control solution, closing a gap after 36 years. This enables us to reveal two surprising insights: The optimal control is quite simple, yet modern RL agents display a large gap to optimality. Motivated by the analysis of the optimal control, we introduce Chebyshev policies as a universal (i.e. dense) class of RL policies from first principles. They can be trained as drop-in replacements of neural nets, reducing the regret by a factor of 4.18, while requiring 277 times fewer parameters, fostering sample efficiency, explainability and realtime capability. Chebyshev policies are evaluated on further RL tasks, including a real-world nonlinear motion control testbed. They consistently improve performance over neural nets with PPO, ARS and REINFORCE. Our results demonstrate how Chebyshev policies offer a compelling and lightweight alternative
    
[^192]: Sutra：面向向量符号架构的张量操作循环神经网络编译目标

    Sutra: Tensor-Op RNNs as a Compilation Target for Vector Symbolic Architectures

    [https://arxiv.org/abs/2605.20919](https://arxiv.org/abs/2605.20919)

    Sutra是一种纯函数式编程语言，通过将整个程序编译为融合张量操作图，在多个嵌入基座上实现100%准确率的向量符号架构解码，远超传统Hadamard乘积方法。

    

    摘要：arXiv:2605.20919v3 类型：替换交叉 摘要：Sutra是一种带类型的纯函数式编程语言，其编译后的前向传播过程是一个PyTorch神经网络。编译器将整个程序（包括原语、控制流、字符串I/O）进行beta归约，最终在冻结的嵌入基座上融合成一个张量操作图。旋转绑定、解绑、捆绑、多项式Kleene三值逻辑以及尾递归循环均被降级为张量操作；Kleene连接词是在{-1, 0, +1}真值网格上精确的拉格朗日插值多项式。验证通过两种方式测试同一个事实：(1) 同一程序在跨越两种模态的四个冻结嵌入上运行——三个文本编码器（nomic-embed-text、all-minilm、mxbai-embed-large）和一个蛋白质语言模型（ESM-2）——并在每个基座上通过宽度k=8以100%的准确率解码捆绑，而教科书中的Hadamard乘积已经崩溃（在mxbai-embed-large上为2.5%，在all-minilm上为7.5%）。(2) PyTorch自动求导流经该程序。

    arXiv:2605.20919v3 Announce Type: replace-cross  Abstract: Sutra is a typed, purely functional programming language whose compiled forward pass is a PyTorch neural network. The compiler beta-reduces the whole program -- primitives, control flow, string I/O -- to one fused tensor-op graph over a frozen embedding substrate. Rotation binding, unbind, bundle, polynomial Kleene three-valued logic, and tail-recursive loops all lower to tensor operations; the Kleene connectives are Lagrange-interpolated polynomials exact on the {-1, 0, +1} truth grid. Validation is one fact tested two ways. (1) The same program runs on four frozen embeddings spanning two modalities -- three text encoders (nomic-embed-text, all-minilm, mxbai-embed-large) and one protein language model (ESM-2) -- and decodes bundles at 100% accuracy through width k=8 on every substrate, where the textbook Hadamard product has already collapsed (2.5% on mxbai-embed-large, 7.5% on all-minilm). (2) PyTorch autograd flows through t
    
[^193]: 评估深度研究代理在专家咨询工作中的表现：一个包含验证器、评分量表和认知陷阱的基准测试

    Evaluating Deep Research Agents on Expert Consulting Work: A Benchmark with Verifiers, Rubrics, and Cognitive Traps

    [https://arxiv.org/abs/2605.17554](https://arxiv.org/abs/2605.17554)

    本文提出了一个包含认知陷阱的70个专家咨询提示基准测试，通过二元验证器和五维评分量表评估三个前沿深度研究代理，发现它们在联合阈值下的接受率普遍很低（最高仅15.7%）。

    

    arXiv:2605.17554v3 公告类型：替换 摘要：前沿深度研究代理（DRAs）在企业工作流程中的部署速度远超其评估速度。现有基准测试衡量的是事实回忆、单跳问答或通用代理技能，但忽略了DRAs被要求生成的多文档、决策级交付成果。我们引入了一个包含70个 SME 撰写的管理咨询提示的基准测试，每个提示都嵌入了认知陷阱，以惩罚表面模式推理。三个前沿代理，即 Claude Opus 4.6、OpenAI o3-deep-research 和 Gemini 3.1 Pro deep-research，在两个互补的层面进行评分：确定性二元验证器（平均每个任务14.9个）和一个五标准0-3分 SME 评分量表（数据完整性、分析严谨性、相关性与重点、执行精确性、格式与可交付性），两者结合形成验证器-评分量表得分（VRS，0-100分）。在联合阈值（评分量表均值≥2.5且验证器通过率≥80%）下的接受率普遍较低：o3 为15.7%。

    arXiv:2605.17554v3 Announce Type: replace  Abstract: Frontier deep research agents (DRAs) are being deployed in enterprise workflows faster than they are being evaluated. Existing benchmarks measure factual recall, single-hop QA, or generic agentic skill, and miss the multi-document, decision-grade deliverables DRAs are asked to produce. We introduce a benchmark of 70 SME-authored management consulting prompts, each embedding cognitive traps that penalize surface-pattern reasoning. Three frontier agents, namely Claude Opus~4.6, OpenAI o3-deep-research and Gemini~3.1~Pro deep-research, are scored on two complementary layers: deterministic binary verifiers (mean 14.9 per task) and a five-criterion 0--3 SME rubric (Data Integrity, Analytical Rigor, Relevance \& Focus, Execution Precision, Format \& Deliverability), combined into a Verifier-Rubric Score (VRS, 0--100).   Acceptance under a joint threshold (rubric mean $\geq 2.5$ and verifier pass rate $\geq 80\%$) is uniformly low: o3 15.7\
    
[^194]: 通过不匹配的错误草稿实现从弱到强的能力激发

    Weak-to-Strong Elicitation via Mismatched Wrong Drafts

    [https://arxiv.org/abs/2605.17314](https://arxiv.org/abs/2605.17314)

    本文发现，将较小模型产生的数学错误草稿不匹配地注入到更强学习者的GRPO训练中，能显著提升其在数学推理任务上的表现，优于标准策略内GRPO。

    

    本文研究了来自较小、较弱模型的离策略经验是否能够激发更强学习者的能力，而这种能力是策略内强化学习微调（如GRPO）无法达到的。我们发现，将来自较小但领域训练更充分的模型（与当前问题不匹配）的数学错误草稿注入到更强学习者的GRPO上下文中，在保留的MATH-500和分布外AIME 2025/2026上持续优于标准的策略内GRPO。具体而言，我们使用Mathstral-7B作为学习者，Qwen2.5-Math-1.5B作为草稿模型，8.8K个级别3-5的MATH问题（保留MATH-500），并使用Dr. GRPO进行训练。不匹配是一个积极因素：在保持其他条件不变的情况下，将草稿随机分配到不匹配的问题上，在MATH-500上（贪婪pass@1）比匹配错误变体高出1.62个百分点（n=10个种子，p=0.0015，Welch's t检验）。事实上，不匹配错误变体在我们测试的所有MATH-500变体中表现最佳。

    arXiv:2605.17314v2 Announce Type: replace-cross  Abstract: We consider whether off-policy experience from a smaller, weaker model can elicit capability in a stronger learner that on-policy RL fine-tuning (e.g., GRPO) does not reach. We find that injecting mathematically wrong drafts from a smaller but more domain-trained model -- mismatched to the current problem -- into a stronger learner's GRPO context consistently outperforms standard on-policy GRPO on held-out MATH-500 and out-of-distribution AIME 2025/2026. Concretely, we use Mathstral-7B as the learner, Qwen2.5-Math-1.5B as the draft model, 8.8K Level 3--5 MATH problems (with MATH-500 held out), and train with Dr. GRPO. Mismatch is an active ingredient: shuffling drafts to mismatched problems while holding everything else constant yields $+1.62$pp on MATH-500 (greedy pass@1) over the matched-wrong variant ($n=10$ seeds, $p=0.0015$, Welch's $t$). In fact, the mismatched-wrong variant leads all other variants we tested on MATH-500 
    
[^195]: 随机测试函数、$H^{-1}$ 范数等价性以及随机变分物理信息神经网络

    Random test functions, $H^{-1}$ norm equivalence, and stochastic variational physics-informed neural networks

    [https://arxiv.org/abs/2605.03542](https://arxiv.org/abs/2605.03542)

    本文证明了任意泛函的 $H^{-1}$ 范数等价于其针对仅依赖于定义域的随机测试函数的期望平方评估，从而避免了计算上困难的上确界，并引入了与经典弱解一致的随机弱解概念，为随机变分物理信息神经网络提供了理论基础。

    

    二阶线性椭圆偏微分方程弱解的对偶范数表征在数学上很自然，但在计算上却难以处理：评估残差的 $H^{-1}$ 范数需要在无限维测试空间上取上确界。我们证明任何泛函的 $H^{-1}$ 范数等价于其针对随机测试函数的期望平方评估，该随机测试函数的概率分布仅依赖于定义域。关键在于，对于 $d \geq 2$，该随机测试函数的实现具有负的索伯列夫正则性，然而这种粗糙性并非障碍：对分布取平均恰好恢复了正确的弱拓扑，且独立于微分算子，无需任何上确界评估。这种等价性引入了随机弱解的概念，它与经典弱解一致，并推动了随机变分物理信息方法的发展。

    arXiv:2605.03542v2 Announce Type: replace-cross  Abstract: The dual norm characterisation of weak solutions of second-order linear elliptic partial differential equations is mathematically natural but computationally intractable: evaluating the $H^{-1}$ norm of the residual requires a supremum over an infinite-dimensional test space. We prove that the $H^{-1}$ norm of any functional is equivalent to its expected squared evaluation against a random test function whose probability distribution depends only on the domain. Crucially, realisations of this random test function have negative Sobolev regularity for $d \geq 2$, yet this roughness is not an obstacle: averaging over the distribution exactly recovers the correct weak topology, independently of the differential operator, and no supremum evaluation is necessary. This equivalence introduces the notion of stochastically weak solutions, which coincide with classical weak solutions, and motivates stochastic variational physics-informed 
    
[^196]: 基于历史强化学习与潜在模型自适应的闭环二氧化碳封存控制

    Closed-Loop CO2 Storage Control With History-Based Reinforcement Learning and Latent Model-Based Adaptation

    [https://arxiv.org/abs/2605.02405](https://arxiv.org/abs/2605.02405)

    本文提出了一种结合历史强化学习和潜在模型自适应的闭环二氧化碳封存控制方法，通过利用时间井响应信息和自适应机制，有效应对储层不确定性和动态变化。

    

    地质二氧化碳封存的闭环管理需要能够适应不确定储层行为的控制策略，同时依赖运营期间实际可获得的观测数据。本文将二氧化碳注入和盐水生产控制建模为一个部分可观测的序贯决策问题，并研究使用高保真储层模拟训练的、可部署的深度强化学习控制器。我们首先比较了特权状态、仅井数据、历史条件、掩蔽课程和非对称师生无模型策略，以量化时间井响应信息和训练时特权模拟器状态的价值。随后，我们评估了一种基于潜在模型的自适应流水线，该流水线重用标称潜在动力学，并在已知注入器故障、泄漏引发的动力学和奖励变化以及分隔储层连通性下重新调整控制器。结果表明，历史条件策略显著优于仅井数据方法，而基于潜在模型的自适应在应对动态变化时表现出鲁棒性。

    arXiv:2605.02405v2 Announce Type: replace  Abstract: Closed-loop management of geological CO2 storage requires control policies that adapt to uncertain reservoir behavior while relying on observations that are realistically available during operation. This work formulates CO2 injection and brine-production control as a partially observable sequential decision problem and studies deployable deep reinforcement-learning controllers trained with high-fidelity reservoir simulation. We first compare privileged-state, well-only, history-conditioned, masking-curriculum, and asymmetric teacher-student model-free policies in order to quantify the value of temporal well-response information and training-time privileged simulator states. We then evaluate a latent model-based adaptation pipeline that reuses nominal latent dynamics and retunes controllers under known injector failure, leakage-induced dynamics and reward shift, and compartmentalized reservoir connectivity. The results show that histo
    
[^197]: 针对Transformer架构的层次化故障检测与诊断方法

    Hierarchical Fault Detection and Diagnosis for Transformer Architectures

    [https://arxiv.org/abs/2604.28118](https://arxiv.org/abs/2604.28118)

    提出了一种名为DEFault++的层次化学习方法，能够自动检测Transformer模型中的隐蔽故障、定位受影响的组件并找出根本原因，同时构建了包含5556个标注运行实例的DEFault-bench基准测试集用于训练和评估。

    

    arXiv:2604.28118v2 公告类型：替换交叉 摘要：Transformer模型如今支撑着工业界和研究领域的众多关键AI系统。然而，其故障可能在不触发运行时错误的情况下悄然改变模型行为，而现有技术几乎无法将这些故障追溯至具体组件及其根本原因。这类故障之所以难以检测，是因为损失函数值和数值指标保持正常，且可见的症状很少能指明具体是哪个组件出了问题。我们提出了DEFault++，一种基于层次化学习的技术，它首先检测故障，然后识别受影响的组件，最后定位组件内的具体原因，从而帮助开发者高效地调试Transformer模型。DEFault++通过故障传播图（FPG）——一种基于架构依赖路径的结构先验——来组织组件级的运行时测量，并报告每项诊断背后的证据。为了训练和评估该方法，我们构建了DEFault-bench基准测试集，该基准包含来自跨模型变异测试的5,556个带标签的运行实例。

    arXiv:2604.28118v2 Announce Type: replace-cross  Abstract: Transformers now underpin critical AI systems across industry and research. Yet their faults can silently alter model behavior without runtime errors, and existing techniques offer little support for tracing these failures to their component and root cause. Such faults evade detection because loss and numerical values stay normal, and the visible symptom rarely identifies the component responsible. We present DEFault++, a hierarchical learning-based technique that first detects a fault, then identifies the affected component, and finally the cause within it, helping developers effectively debug transformer models. DEFault++ organizes component-level runtime measurements with a Fault Propagation Graph (FPG), a structural prior over the architecture's dependency paths, and reports the evidence behind each diagnosis. To train and evaluate it, we construct DEFault-bench, a benchmark of 5,556 labeled runs from mutation testing acros
    
[^198]: TransXion：面向现实反洗钱场景的高保真图基准

    TransXion: A High-Fidelity Graph Benchmark for Realistic Anti-Money Laundering

    [https://arxiv.org/abs/2604.17420](https://arxiv.org/abs/2604.17420)

    该论文提出了TransXion，一个通过结合档案感知的正常模拟与非模板化异常合成来解决现有反洗钱基准局限性，从而评估模型对“出格”异常检测能力的高保真图基准。

    

    摘要：arXiv:2604.17420v2 公告类型：替换-交叉  摘要：洗钱对全球金融体系构成严重风险，推动了机器学习在交易监控中的广泛应用。然而，由于缺乏现实基准，进展仍然受到阻碍。现有的交易图数据集存在两个普遍局限：(i) 它们仅提供匿名标识符之外的稀疏节点级语义；(ii) 它们依赖模板驱动的异常注入，这使基准偏向于静态结构模式，并导致对模型鲁棒性的评估过于乐观。我们提出了TransXion，一个用于反洗钱（AML）研究的基准生态系统，它将基于档案的正常活动模拟与随机的、非模板化的非法子图合成相结合。TransXion联合建模了持久的实体档案与条件性交易行为，从而能够评估“超出特征”的异常情况，即观测到的活动与实体的社会行为相矛盾。

    arXiv:2604.17420v2 Announce Type: replace-cross  Abstract: Money laundering poses severe risks to global financial systems, driving the widespread adoption of machine learning for transaction monitoring. However, progress remains stifled by the lack of realistic benchmarks. Existing transaction-graph datasets suffer from two pervasive limitations: (i) they provide sparse node-level semantics beyond anonymized identifiers, and (ii) they rely on template-driven anomaly injection, which biases benchmarks toward static structural motifs and yields overly optimistic assessments of model robustness. We propose TransXion, a benchmark ecosystem for Anti-Money Laundering (AML) research that integrates profile-aware simulation of normal activity with stochastic, non-template synthesis of illicit subgraphs.TransXion jointly models persistent entity profiles and conditional transaction behavior, enabling evaluation of "out-of-character" anomalies where observed activity contradicts an entity's soc
    
[^199]: 文王序列的统计特性：一种不改善神经网络训练的反习惯化结构

    Statistical Properties of the King Wen Sequence: An Anti-Habituation Structure That Does Not Improve Neural Network Training

    [https://arxiv.org/abs/2604.09234](https://arxiv.org/abs/2604.09234)

    文王序列虽具有显著统计特性且表面类似课程学习原则，但实验证明这些特性并不能改善神经网络训练。

    

    本文针对《易经》（约公元前1000年）中文王序列将64卦（六维二元空间的状态）按某种模式排列的现象进行了研究，该模式三千年来一直令学者困惑。我们通过蒙特卡洛置换分析（以10万个随机基线为参照）对此排序进行了严格的统计特性刻画。研究发现该序列具有四个统计显著性特征：高于随机的转移距离（第98.2百分位）、负的一阶自相关（p=0.037）、阳爻平衡的四卦组（p=0.002）以及组内与组间距离的不对称性（第99.2百分位）。这些特性表面上类似于课程学习和好奇心驱动探索的原则，由此提出了它们可能有益于神经网络训练的假设。我们通过三个实验（学习率调度调制、课程排序和种子敏感性分析）验证了这一假设。

    arXiv:2604.09234v2 Announce Type: replace-cross  Abstract: The King Wen sequence of the I-Ching (c. 1000 BC) orders 64 hexagrams -- states of a six-dimensional binary space -- in a pattern that has puzzled scholars for three millennia. We present a rigorous statistical characterization of this ordering using Monte Carlo permutation analysis against 100,000 random baselines. We find that the sequence has four statistically significant properties: higher-than-random transition distance (98.2nd percentile), negative lag-1 autocorrelation (p=0.037), yang-balanced groups of four (p=0.002), and asymmetric within-pair vs. between-pair distances (99.2nd percentile). These properties superficially resemble principles from curriculum learning and curiosity-driven exploration, motivating the hypothesis that they might benefit neural network training. We test this hypothesis through three experiments: learning rate schedule modulation, curriculum ordering, and seed sensitivity analysis, conducted 
    
[^200]: 重新审视基于等价查询的学习

    Learning from Equivalence Queries, Revisited

    [https://arxiv.org/abs/2604.04535](https://arxiv.org/abs/2604.04535)

    本文通过放宽对抗性假设和完全信息要求，重新审视了经典等价查询学习模型，使其更适应现代机器学习系统的实际部署循环。

    

    现代机器学习系统，例如生成模型和推荐系统，通常通过部署、用户交互和定期模型更新的循环来演进。这与标准的监督学习框架不同，后者专注于在固定的预测任务序列上最小化损失或遗憾。受此场景启发，我们重新审视了由Angluin（1988）提出的经典基于等价查询的学习模型。在该模型中，学习器反复提出假设，当部署的假设不充分时，会收到一个反例。然而，在完全对抗性的反例生成下，该模型可能过于悲观。此外，大多数先前的工作假设一个“完全信息”场景，即学习器还能观察到反例的正确标签，这一假设并不总是自然的。我们通过将环境限制在一类较不具对抗性的范围内来解决这些问题。

    arXiv:2604.04535v2 Announce Type: replace  Abstract: Modern machine learning systems, such as generative models and recommendation systems, often evolve through a cycle of deployment, user interaction, and periodic model updates. This differs from standard supervised learning frameworks, which focus on loss or regret minimization over a fixed sequence of prediction tasks. Motivated by this setting, we revisit the classical model of learning from equivalence queries, introduced by Angluin (1988). In this model, a learner repeatedly proposes hypotheses and, when a deployed hypothesis is inadequate, receives a counterexample. Under fully adversarial counterexample generation, however, the model can be overly pessimistic. In addition, most prior work assumes a \emph{full-information} setting, where the learner also observes the correct label of the counterexample, an assumption that is not always natural.   We address these issues by restricting the environment to a broad class of less adv
    
[^201]: NASimJax：一种用于渗透测试的GPU加速策略学习框架

    NASimJax: A GPU-Accelerated Policy Learning Framework for Penetration Testing

    [https://arxiv.org/abs/2603.19864](https://arxiv.org/abs/2603.19864)

    NASimJax通过基于JAX的GPU加速实现，将网络攻击模拟器吞吐量提升高达100倍，使大规模渗透测试策略学习成为可能。

    

    渗透测试是模拟网络攻击以识别漏洞的实践，这是一个复杂的序列决策任务，本质上具有部分可观测性，并包含巨大的动作空间。为此领域训练强化学习策略面临一个根本性瓶颈：现有模拟器在规模化训练时速度过慢，无法应对真实的网络场景，导致策略泛化能力不足。我们提出了NASimJax，这是对网络攻击模拟器的完整基于JAX的重实现，相比原始模拟器实现了高达100倍的环境吞吐量提升。通过在硬件加速器上运行整个训练流程，NASimJax使得在固定计算预算下对更大网络进行实验成为可能，而这在以前是不可行的。我们将自动化渗透测试形式化为一个上下文感知的部分可观测马尔可夫决策过程，并引入一个网络生成管道，能够生成结构多样且保证...

    arXiv:2603.19864v2 Announce Type: replace  Abstract: Penetration testing, the practice of simulating cyberattacks to identify vulnerabilities, is a complex sequential decision-making task that is inherently partially observable and features large action spaces. Training reinforcement learning (RL) policies for this domain faces a fundamental bottleneck: existing simulators are too slow to train on realistic network scenarios at scale, resulting in policies that fail to generalize. We present NASimJax, a complete JAX-based reimplementation of the Network Attack Simulator (NASim), achieving up to 100x higher environment throughput than the original simulator. By running the entire training pipeline on hardware accelerators, NASimJax enables experimentation on larger networks under fixed compute budgets that were previously infeasible. We formulate automated penetration testing as a Contextual POMDP and introduce a network generation pipeline that produces structurally diverse and guarant
    
[^202]: 基于随机注意力从小规模家族比对中无需训练生成蛋白质序列

    Training-Free Generation of Protein Sequences from Small Family Alignments via Stochastic Attention

    [https://arxiv.org/abs/2603.14717](https://arxiv.org/abs/2603.14717)

    本文提出一种无需训练的蛋白质序列生成方法，通过随机注意力机制和朗之万动力学，在小规模家族比对数据上生成符合统计约束且结构合理的新序列，解决了传统方法在小数据下的过拟合问题。

    

    arXiv:2603.14717v2 公告类型：替换 摘要：生成符合家族统计约束的新型蛋白质序列通常需要在数千到数百万个样本上训练深度生成模型。然而，大多数蛋白质家族规模较小：Pfam种子比对的中位数仅包含22条序列，在这种数据量下，学习模型容易过拟合或崩溃。我们提出了一种无需训练的采样方法——随机注意力（SA），该方法将现代Hopfield能量视为存储序列的玻尔兹曼分布，并通过朗之万动力学进行采样。其得分函数是单次softmax注意力操作的残差，从而消除了对训练得分网络、预训练数据或图形处理单元（GPU）的需求。在跨越37到420条序列、23到262个残基的八个Pfam家族中，SA生成的序列具有低组成差异、新颖性，并得到ESMFold和AlphaFold2支持的结构合理性。与profile隐马尔可夫模型相比……

    arXiv:2603.14717v2 Announce Type: replace  Abstract: Generating novel protein sequences that respect a family's statistical constraints typically requires training deep generative models on thousands to millions of examples. Yet most protein families are small: the median Pfam seed alignment contains only 22 sequences, a regime where learned models overfit or collapse. We propose \emph{stochastic attention} (SA), a training-free sampler that treats the modern Hopfield energy over stored sequences as a Boltzmann distribution and draws samples via Langevin dynamics. The score function is the residual of a single softmax attention operation, eliminating the need for a trained score network, pretraining data, or graphics processing units (GPUs). Across eight Pfam families spanning 37 to 420 sequences and 23 to 262 residues, SA generates sequences with low composition divergence, novelty, and structural plausibility supported by ESMFold and AlphaFold2. Compared with profile hidden Markov mo
    
[^203]: BrepCoder：一个用于多任务B-rep推理的统一多模态大语言模型

    BrepCoder: A Unified Multimodal Large Language Model for Multi-task B-rep Reasoning

    [https://arxiv.org/abs/2602.22284](https://arxiv.org/abs/2602.22284)

    BrepCoder是一个统一的多模态大语言模型，通过将CAD序列转化为类似Python的代码并与B-rep对齐，再经两阶段训练，实现了从B-rep输入执行补全、纠错和问答等多种CAD任务。

    

    近期深度学习领域的进展积极应对了计算机辅助设计（CAD）领域中的复杂挑战。然而，现有方法大多依赖于特定任务的模型，需要为新任务进行结构修改，并且它们主要关注点云或图像，而非行业标准的边界表示（B-rep）格式。为了解决这些局限，我们提出了BrepCoder，这是一个统一的多模态大语言模型（MLLM），能够从B-rep输入执行多种CAD任务。通过利用大语言模型（LLM）的代码生成能力，我们将CAD建模序列转换为类似Python的代码，并将其与B-rep对齐。然后，我们采用两阶段训练策略：首先，进行逆向工程预训练，以学习几何特征和设计逻辑；其次，有效地将模型扩展到各种下游任务，如补全、错误修正和CAD问答。

    arXiv:2602.22284v3 Announce Type: replace  Abstract: Recent advancements in deep learning have actively addressed complex challenges within the Computer-Aided Design (CAD) domain.However, most existing approaches rely on task-specifi c models requiring structural modifi cations for new tasks, and they predominantly focus on point clouds or images rather than the industry-standard Boundary Representation (B-rep) format. To address these limitations, we propose BrepCoder, a unifi ed Multimodal Large Language Model (MLLM) that performs diverse CAD tasks from B-rep inputs. By leveraging the code generation capabilities of Large Language Models (LLMs), we convert CAD modeling sequences into Python-like code and align them with B-rep. We then adopt a two-stage training strategy: First, pre-training on reverse engineering to learn geometric features and design logic. Second, eff ectively extending the model to various downstream tasks such as completion, error correction, and CAD-QA. Conseque
    
[^204]: 基于希尔伯特空间嵌入的量子最大似然预测

    Quantum Maximum Likelihood Prediction via Hilbert Space Embeddings

    [https://arxiv.org/abs/2602.18364](https://arxiv.org/abs/2602.18364)

    本文通过将经验概率分布嵌入量子态并最小化量子相对熵，提出了一种量子最大似然预测方法，并为其在经典和量子大语言模型中的统一应用提供了非渐近性能保证。

    

    arXiv:2602.18364v3 公告类型: 替换-交叉 摘要：最大似然预测（MLP）是现代大型语言模型的核心任务。在此，我们首次针对由独立同分布样本构成的简化数据模型，研究该任务的量子版本。量子最大似然预测器（QMLP）通过将经验概率分布嵌入到量子态中，并在给定状态类上最小化量子相对熵来获得。我们推导了QMLP在迹范数和量子相对熵方面的非渐近性能保证，包括收敛速率和浓度不等式。我们的方法为在经典和量子大语言模型中处理MLP提供了一个统一框架。我们还考虑了量子信息投影的相关问题，并将著名的量子毕达哥拉斯定理推广到并非由自伴类生成的混合族。

    arXiv:2602.18364v3 Announce Type: replace-cross  Abstract: Maximum likelihood prediction (MLP) is a core task at the heart of modern large language models. Here, we study a quantum version of this task for a simplified data model consisting of independent and identically distributed samples, as a first step. The quantum maximum likelihood predictor (QMLP) is obtained by embedding of empirical probability distributions into quantum states and performing a minimization of quantum relative entropy over a given class of states. We derive non-asymptotic performance guarantees for QMLP in terms of convergence rates and concentration inequalities, both in trace norm and quantum relative entropy. Our approach provides a unified framework to handle MLP within both classical and quantum LLMs. We also consider the related problem of quantum information projection and generalize the well known quantum Pythagorean theorem to mixture families which are not necessarily generated by a self-adjoint cla
    
[^205]: 利用时间预测编码学习长程依赖关系

    Learning Long-Range Dependencies with Temporal Predictive Coding

    [https://arxiv.org/abs/2602.18131](https://arxiv.org/abs/2602.18131)

    本文首次将时间预测编码与实时循环学习结合，通过引入在线影响矩阵追踪参数历史影响，在保留局部学习特性的同时精确恢复时间反向传播梯度，从而有效解决了长程依赖学习问题。

    

    arXiv:2602.18131v2 公告类型：替换 摘要：时间预测编码提供了一种逐层、可并行化的循环系统学习机制，使其成为神经形态和边缘硬件上在线局部学习的有吸引力的候选方案。然而，其循环参数更新仅捕获局部时间关系，忽略了参数沿潜在状态轨迹的历史影响，因此难以在更长的时间跨度上进行信用分配。本研究首次将时间预测编码与实时循环学习（tPC-RTRL）相结合，引入了一个在线影响矩阵，该矩阵在追踪历史影响的同时，保留了神经形态实现所重视的空间和时间局部性属性。在明确假设下，我们证明了tPC-RTRL能够精确恢复时间反向传播的梯度。在多个不同规模和复杂度的任务上，包括…，实验结果表明两者几乎等价。

    arXiv:2602.18131v2 Announce Type: replace  Abstract: Temporal Predictive Coding provides a layer-local, parallelisable mechanism for learning in recurrent systems, making it an attractive candidate for online local learning on neuromorphic and edge hardware. However, its recurrent parameter update captures only local temporal relationships, neglecting the historic influence of parameters along the latent-state trajectory, and therefore struggles to assign credit over longer temporal horizons. This work combines for the first time Temporal Predictive Coding with Real-Time Recurrent Learning (tPC-RTRL), incorporating an online influence matrix that tracks this historic effect whilst preserving the spatial and temporal locality properties valued by neuromorphic implementations. Under explicit assumptions, we prove that tPC-RTRL recovers the gradients of backpropagation-through-time exactly. Empirically, a near-equivalence holds across several tasks of varying scale and complexity, includi
    
[^206]: 基于稀疏卫星时间序列和天气协变量的概率性NDVI预测

    Probabilistic NDVI Forecasting from Sparse Satellite Time Series and Weather Covariates

    [https://arxiv.org/abs/2602.17683](https://arxiv.org/abs/2602.17683)

    该论文提出了一种概率性预测框架，通过分离历史数据编码与未来协变量、引入时间距离加权损失函数，解决了稀疏不规则卫星观测下的田块级NDVI短期预测挑战。

    

    arXiv:2602.17683v3 公告类型：替换 摘要：植被动态的短期预测是实现精准农业中数据驱动决策支持的关键推动因素。然而，基于卫星观测的归一化植被指数（NDVI）预测仍具挑战性，原因在于云掩膜导致的稀疏和不规则采样，以及作物生长的异质性气候条件。在这项工作中，我们提出了一种概率性预测框架，用于在稀疏、不规则的晴空采集条件下进行田块级NDVI预测。该架构将历史NDVI和气象观测的编码与未来外生协变量分离，融合两种表示以进行多步分位数预测。为解决不规则重访模式和与预测时间跨度相关的不确定性，我们引入了一种时间距离加权分位数损失函数，使训练目标与有效预测时间跨度对齐。此外，我们纳入了累积...

    arXiv:2602.17683v3 Announce Type: replace  Abstract: Short-term forecasting of vegetation dynamics is a key enabler for data-driven decision support in precision agriculture. Normalized Difference Vegetation Index (NDVI) forecasting from satellite observations, however, remains challenging due to sparse and irregular sampling caused by cloud masking, as well as the heterogeneous climatic conditions under which crops evolve. In this work, we propose a probabilistic forecasting framework for field-level NDVI prediction under sparse, irregular clear-sky acquisitions. The architecture separates the encoding of historical NDVI and meteorological observations from future exogenous covariates, fusing both representations for multi-step quantile prediction. To address irregular revisit patterns and horizon-dependent uncertainty, we introduce a temporal-distance weighted quantile loss that aligns the training objective with the effective forecasting horizon. In addition, we incorporate cumulati
    
[^207]: SEMixer：面向多尺度混合与长期时间序列预测的语义增强型MLP-Mixer

    SEMixer: Semantics Enhanced MLP-Mixer for Multiscale Mixing and Long-term Time Series Forecasting

    [https://arxiv.org/abs/2602.16220](https://arxiv.org/abs/2602.16220)

    SEMixer通过随机注意力机制和多尺度渐进混合链，在轻量级MLP-Mixer架构中有效解决了时间序列多尺度建模中的语义鸿沟与噪声问题，显著提升了长期预测性能。

    

    摘要：建模多尺度模式对于长期时间序列预测（TSF）至关重要。然而，时间序列中的冗余和噪声，以及非相邻尺度之间的语义鸿沟，使得多尺度时间依赖关系的高效对齐与整合变得具有挑战性。为解决这一问题，我们提出了SEMixer，一种专为长期TSF设计的轻量级多尺度模型。SEMixer包含两个关键组件：随机注意力机制（RAM）和多尺度渐进混合链（MPMC）。RAM在训练过程中捕获多样的时间块交互，并在推理时通过丢弃集成进行聚合，从而增强块级语义，使MLP-Mixer能够更好地建模多尺度依赖关系。MPMC进一步以内存高效的方式堆叠RAM和MLP-Mixer，实现更有效的时间混合。它解决了跨尺度的语义鸿沟问题，并促进了更好的多尺度建模与预测性能。我们不仅在多个基准数据集上验证了其有效性……

    arXiv:2602.16220v3 Announce Type: replace  Abstract: Modeling multiscale patterns is crucial for long-term time series forecasting (TSF). However, redundancy and noise in time series, together with semantic gaps between non-adjacent scales, make the efficient alignment and integration of multi-scale temporal dependencies challenging. To address this, we propose SEMixer, a lightweight multiscale model designed for long-term TSF. SEMixer features two key components: a Random Attention Mechanism (RAM) and a Multiscale Progressive Mixing Chain (MPMC). RAM captures diverse time-patch interactions during training and aggregates them via dropout ensemble at inference, enhancing patch-level semantics and enabling MLP-Mixer to better model multi-scale dependencies. MPMC further stacks RAM and MLP-Mixer in a memory-efficient manner, achieving more effective temporal mixing. It addresses semantic gaps across scales and facilitates better multiscale modeling and forecasting performance. We not onl
    
[^208]: 利用已知信息：基于部分图结构的因果基础模型

    Use What You Know: Causal Foundation Models with Partial Graphs

    [https://arxiv.org/abs/2602.14972](https://arxiv.org/abs/2602.14972)

    本文提出了一种将因果基础模型条件化于部分因果图或祖先信息的方法，通过注入可学习偏置与图卷积编码器，有效提升了利用不完全领域知识时的因果推断性能。

    

    arXiv:2602.14972v2 公告类型：替换  摘要：传统上，估计因果量依赖于针对特定假设定制的专用估计器。最近提出的因果基础模型（CFMs）通过将因果发现与推断合并为单一步骤，有望提供一种更统一的方法。然而，在目前的状态下，这些模型无法融入任何领域知识，这可能导致预测结果欠佳。我们通过引入将CFMs条件化于因果信息（如因果图或更易获取的祖先信息）的方法，弥合了这一差距。当获取完整因果图信息的要求过于严格时，我们的方法也能有效利用部分因果信息。我们系统地评估了各种条件化策略，发现将可学习偏置注入注意力机制，并结合图卷积编码器，是充分利用完整及部分因果信息的高效方法。

    arXiv:2602.14972v2 Announce Type: replace  Abstract: Estimating causal quantities traditionally relies on bespoke estimators tailored to specific assumptions. Recently proposed Causal Foundation Models (CFMs) promise a more unified approach by amortising causal discovery and inference in a single step. However, in their current state, they do not allow for the incorporation of any domain knowledge, which can lead to suboptimal predictions. We bridge this gap by introducing methods to condition CFMs on causal information, such as the causal graph or more readily available ancestral information. When access to complete causal graph information is too strict a requirement, our approach also effectively leverages partial causal information. We systematically evaluate conditioning strategies and find that injecting learnable biases into the attention mechanism, together with a graph-convolutional encoder, is a highly effective method to utilise full and partial causal information. Our exper
    
[^209]: 利用线性RNN从代码中学习状态跟踪

    Learning State-Tracking from Code Using Linear RNNs

    [https://arxiv.org/abs/2602.14814](https://arxiv.org/abs/2602.14814)

    本文通过将状态跟踪任务转换为代码形式，证明线性RNN在代码环境下的状态跟踪能力优于Transformer，并揭示了动作部分可观测性是导致状态跟踪困难的根本原因。

    

    在过去几年中，状态跟踪任务（特别是排列组合任务）已成为理解Transformer和RNN（线性和非线性）等序列模型架构极限的测试平台。然而，这些任务通常是序列到序列的任务：学习将动作（排列）映射到状态，这与语言模型常用的下一个词预测设置不兼容。我们通过REPL执行轨迹将排列组合转换为代码，这些轨迹通过打印输出和变量变换交错地揭示状态。我们证明，能够进行状态跟踪的线性RNN在此设置中同样表现出色，而Transformer仍然失败。受这种表示方式的启发，我们研究了代码中状态跟踪普遍困难的原因：动作并非总是完全可观测的。我们将此问题建模为具有确定性状态揭示的概率有限状态自动机的状态跟踪。

    arXiv:2602.14814v3 Announce Type: replace-cross  Abstract: Over the last years, state-tracking tasks, particularly permutation composition, have become a testbed to understand the limits of sequence models architectures like Transformers and RNNs (linear and non-linear). However, these are often sequence-to-sequence tasks: learning to map actions (permutations) to states, which is incompatible with the next-token prediction setting commonly used to train language models. We address this gap by converting permutation composition into code via REPL traces that interleave state-reveals through prints and variable transformations. We show that linear RNNs capable of state-tracking excel also in this setting, while Transformers still fail. Motivated by this representation, we investigate why tracking states in code is generally difficult: actions are not always fully observable. We frame this as tracking the state of a probabilistic finite-state automaton with deterministic state reveals an
    
[^210]: 重新审视柏拉图式表征假说：一种亚里士多德式的观点

    Revisiting the Platonic Representation Hypothesis: An Aristotelian View

    [https://arxiv.org/abs/2602.14486](https://arxiv.org/abs/2602.14486)

    本文发现现有表征相似性度量受网络规模干扰，提出基于排列的零校准框架，证明校准后全局表征收敛现象消失，仅局部邻域相似性跨模态一致，从而提出亚里士多德式表征假说。

    

    arXiv:2602.14486v2 公告类型：替换-交叉 摘要：柏拉图式表征假说认为，神经网络的表征正在收敛到一个共同的现实统计模型。我们表明，现有用于衡量表征相似性的度量会受到网络规模的干扰：增加模型深度或宽度会系统性地夸大表征相似性分数。为纠正这些影响，我们引入了一种基于排列的零校准框架，该框架能将任何表征相似性度量转化为具有统计保证的校准分数。我们使用校准框架重新审视了柏拉图式表征假说，结果揭示了一幅微妙的图景：全局光谱测量所报告的明显收敛在校准后大多消失，而局部邻域相似性（而非局部距离）在不同模态间仍保持显著的一致性。基于这些发现，我们提出了亚里士多德式表征假说。

    arXiv:2602.14486v2 Announce Type: replace-cross  Abstract: The Platonic Representation Hypothesis suggests that representations from neural networks are converging to a common statistical model of reality. We show that the existing metrics used to measure representational similarity are confounded by network scale: increasing model depth or width can systematically inflate representational similarity scores. To correct these effects, we introduce a permutation-based null-calibration framework that transforms any representational similarity metric into a calibrated score with statistical guarantees. We revisit the Platonic Representation Hypothesis with our calibration framework, which reveals a nuanced picture: the apparent convergence reported by global spectral measures largely disappears after calibration, while local neighborhood similarity, but not local distances, retains significant agreement across different modalities. Based on these findings, we propose the Aristotelian Repre
    
[^211]: 基于条件流匹配的视觉引导音频增强

    Conditional Flow Matching for Visually-Guided Acoustic Highlighting

    [https://arxiv.org/abs/2602.03762](https://arxiv.org/abs/2602.03762)

    本文提出一种条件流匹配生成框架，通过引入展开损失惩罚最终步骤漂移，有效解决了视觉引导音频增强中判别模型难以处理音频重混模糊性的问题。

    

    arXiv:2602.03762v4 公告类型：替换交叉 摘要：视觉引导的音频增强旨在根据伴随视频重新平衡音频，以创造连贯的视听体验。虽然视觉显著性和增强已被广泛研究，但音频增强仍未被充分探索，常常导致视觉与听觉焦点之间的错位。现有方法使用判别模型，这些模型难以处理音频重混中固有的模糊性——在平衡不佳与平衡良好的音频混合之间，不存在自然的一一对应关系。为解决这一局限，我们将此任务重新定义为生成问题，并引入了一个条件流匹配（CFM）框架。迭代式流生成中的一个关键挑战是：早期预测错误（即选择正确的音频源进行增强）会随着步骤累积，使轨迹偏离流形。为解决此问题，我们引入了一个展开损失函数，用于惩罚最终步骤的漂移。

    arXiv:2602.03762v4 Announce Type: replace-cross  Abstract: Visually-guided acoustic highlighting seeks to rebalance audio in alignment with the accompanying video, creating a coherent audio-visual experience. While visual saliency and enhancement have been widely studied, acoustic highlighting remains underexplored, often leading to misalignment between visual and auditory focus. Existing approaches use discriminative models, which struggle with the inherent ambiguity in audio remixing, where no natural one-to-one mapping exists between poorly-balanced and well-balanced audio mixes. To address this limitation, we reframe this task as a generative problem and introduce a Conditional Flow Matching (CFM) framework. A key challenge in iterative flow-based generation is that early prediction errors -- in selecting the correct source to enhance -- compound over steps and push trajectories off-manifold. To address this, we introduce a rollout loss that penalizes drift at the final step, encou
    
[^212]: DASH：通过批量块预处理和高效逆根求解器实现更快的Shampoo算法

    DASH: Faster Shampoo via Batched Block Preconditioning and Efficient Inverse-Root Solvers

    [https://arxiv.org/abs/2602.02016](https://arxiv.org/abs/2602.02016)

    本文通过将预处理器块堆叠成3D张量以提高GPU利用率，并引入牛顿-DB迭代和切比雪夫多项式近似来加速矩阵逆根计算，从而显著提升了分布式Shampoo优化器的运行速度。

    

    arXiv:2602.02016v2 公告类型：替换 摘要：Shampoo是领先的近似二阶优化器之一：它的一个变体赢得了MLCommons AlgoPerf竞赛，并且已被证明能够生成激活值异常更低、更易于压缩的模型。然而，由于Shampoo内部操作计算成本高昂，当前应用该算法会带来显著的计算速度下降。在本文中，我们通过提出\method（代表\textbf{D}istributed \textbf{A}ccelerated \textbf{SH}ampoo），基于两项主要新技术，迈出了解决这一缺陷的重要一步：首先，我们展示了可以将预处理器块堆叠成3D张量，以显著提高GPU利用率；其次，我们引入了牛顿-DB迭代和切比雪夫多项式近似，作为计算Shampoo所需矩阵逆根的新颖且更快速的方法。除了这些算法贡献外，我们还提供了相应的实现。

    arXiv:2602.02016v2 Announce Type: replace  Abstract: Shampoo is one of the leading approximate second-order optimizers: a variant of it has won the MLCommons AlgoPerf competition, and it has been shown to produce models with lower activation outliers that are easier to compress. Yet, applying Shampoo currently comes at the cost of significant computational slowdown, due to its expensive internal operations. In this paper, we take a significant step to address this shortcoming by proposing \method (for \textbf{D}istributed \textbf{A}ccelerated \textbf{SH}ampoo), a faster implementation of Distributed Shampoo based on two main new techniques: First, we show that preconditioner blocks can be stacked into 3D tensors to significantly improve GPU utilization; second, we introduce the Newton-DB iteration and the Chebyshev polynomial approximations as novel and faster approaches for computing the inverse matrix roots required by Shampoo. Along with these algorithmic contributions, we provide a
    
[^213]: 双原型解耦：一种面向时间序列预测的上下文感知增强框架

    Dual-Prototype Disentanglement: A Context-Aware Enhancement Framework for Time Series Forecasting

    [https://arxiv.org/abs/2601.16632](https://arxiv.org/abs/2601.16632)

    提出一种模型无关的辅助框架DPAD，通过动态双原型库解耦常见与罕见时间模式，使预测模型获得上下文感知的自适应能力。

    

    时间序列预测在深度学习的推动下取得了显著进展。虽然主流方法通过修改架构或引入新颖的增强策略来提升预测性能，但它们往往无法动态解耦并利用时间序列中固有的复杂、交织的时间模式，从而学习到缺乏上下文感知能力的静态平均化表示。为解决这一问题，我们提出了双原型自适应解耦框架（DPAD），这是一种模型无关的辅助方法，使预测模型具备模式解耦和上下文感知自适应能力。具体来说，我们构建了一个动态双原型库（DDP），包含一个具有强时间先验的公共模式库（用于捕获主流趋势或季节模式）和一个动态记忆关键但罕见事件的罕见模式库，然后通过一个双原...

    arXiv:2601.16632v4 Announce Type: replace-cross  Abstract: Time series forecasting has witnessed significant progress with deep learning. While prevailing approaches enhance forecasting performance by modifying architectures or introducing novel enhancement strategies, they often fail to dynamically disentangle and leverage the complex, intertwined temporal patterns inherent in time series, thus resulting in the learning of static, averaged representations that lack context-aware capabilities. To address this, we propose the Dual-Prototype Adaptive Disentanglement framework (DPAD), a model-agnostic auxiliary method that equips forecasting models with the ability of pattern disentanglement and context-aware adaptation. Specifically, we construct a Dynamic Dual-Prototype bank (DDP), comprising a common pattern bank with strong temporal priors to capture prevailing trend or seasonal patterns, and a rare pattern bank dynamically memorizing critical yet infrequent events, and then an Dual-P
    
[^214]: 虚假奖励悖论：从机制上理解RLVR如何激活大语言模型中的记忆捷径

    Spurious Rewards Paradox: Mechanistically Understanding How RLVR Activates Memorization Shortcuts in LLMs

    [https://arxiv.org/abs/2601.11061](https://arxiv.org/abs/2601.11061)

    本文揭示了虚假奖励在RLVR中会激活大语言模型中的“锚点-适配器”神经电路，导致模型依赖记忆捷径而非真正推理，从而解释了“困惑度悖论”。

    

    arXiv:2601.11061v2 公告类型：替换交叉 摘要：基于可验证奖励的强化学习（RLVR）在提升大语言模型推理能力方面非常有效，然而近期证据表明，像Qwen 2.5这样的模型即便在获得虚假或不正确的奖励时也能取得显著提升。我们研究了这一现象，并识别出一个“困惑度悖论”：虚假的RLVR会引发一种分化，即答案令牌的困惑度下降，而提示端的连贯性却恶化，这表明模型正在绕过推理，转而依赖记忆。通过使用路径修补、对数透镜、JSD分析和神经微分方程，我们揭示了一个隐藏的“锚点-适配器”电路，该电路促进了这一捷径。我们在中间层（第18-20层）定位了一个功能锚点，它触发了对已记忆解决方案的检索，随后在后续层（第21层及以上）中的结构适配器对表征进行转换，以适应捷径信号。最后，我们证明了缩放该电路中的特定MLP键，可以使得……

    arXiv:2601.11061v2 Announce Type: replace-cross  Abstract: Reinforcement Learning with Verifiable Rewards (RLVR) is highly effective for enhancing LLM reasoning, yet recent evidence shows models like Qwen 2.5 achieve significant gains even with spurious or incorrect rewards. We investigate this phenomenon and identify a "Perplexity Paradox": spurious RLVR triggers a divergence where answer-token perplexity drops while prompt-side coherence degrades, suggesting the model is bypassing reasoning in favor of memorization. Using Path Patching, Logit Lens, JSD analysis, and Neural Differential Equations, we uncover a hidden Anchor-Adapter circuit that facilitates this shortcut. We localize a Functional Anchor in the middle layers (L18-20) that triggers the retrieval of memorized solutions, followed by Structural Adapters in later layers (L21+) that transform representations to accommodate the shortcut signal. Finally, we demonstrate that scaling specific MLP keys within this circuit allows f
    
[^215]: 隐喻是大推理模型跨领域错位的来源

    Metaphors are a Source of Cross-Domain Misalignment of Large Reasoning Models

    [https://arxiv.org/abs/2601.03388](https://arxiv.org/abs/2601.03388)

    本文发现训练数据中的隐喻会导致大推理模型在推理时出现跨领域错位，并且这种错位可以通过隐喻干预被诱导和放大。

    

    arXiv:2601.03388v3 公告类型：替换交叉  摘要：早期研究表明，隐喻会影响人类决策，这引发了一个问题：鉴于大语言模型（LLMs）的训练数据中包含大量隐喻，隐喻是否也会影响它们的推理路径。在本研究中，我们在新兴错位问题的范围内探讨了这一问题——即LLMs可以将从一个领域学到的错位内容模式泛化到另一个领域。我们发现强有力的证据表明，训练数据中的隐喻会促成LLMs推理输出中的跨领域错位。通过在持续预训练和微调过程中引入基于隐喻的干预以诱导错位，模型表现出显著不同程度的新兴跨领域错位。我们在重新对齐设置中也观察到了类似的效果。随着我们进一步研究这一现象，我们发现隐喻与大型推理模型中潜在特征的激活有关。

    arXiv:2601.03388v3 Announce Type: replace-cross  Abstract: Earlier research has shown that metaphors influence human decision-making, raising the question of whether metaphors also influence large language models (LLMs)' reasoning pathways, given that their training data contain a large number of metaphors. In this work, we investigate the problem in the scope of the emergent misalignment problem, where LLMs can generalize patterns learned from misaligned content in one domain to another domain. We find strong evidence that metaphors in training data contribute to cross-domain misalignment in LLMs' reasoning outputs. With metaphor-based interventions during continued pre-training and fine-tuning for inducing misalignment, models exhibit significantly different degrees of emergent cross-domain misalignment. We also observe similar effects in re-alignment settings. As we further investigate this phenomenon, we find that metaphors are linked to the activation of latent features in large r
    
[^216]: 数字孪生驱动的通信高效联邦异常检测方法用于工业物联网

    Digital Twin-Driven Communication-Efficient Federated Anomaly Detection for Industrial IoT

    [https://arxiv.org/abs/2601.01701](https://arxiv.org/abs/2601.01701)

    本文提出了一套数字孪生集成的联邦学习方法，通过五种创新机制在保护数据隐私和提升通信效率的同时，显著提高了工业物联网异常检测的全局模型性能。

    

    摘要：异常检测对于维护工业系统的安全性、可靠性和效率变得越来越关键。近年来，随着数字孪生和数据驱动决策的出现，人们提出了多种统计和机器学习方法。然而，这些方法面临若干挑战，例如仅依赖真实传感器数据集、标记数据有限、高误报率以及隐私问题。为解决这些问题，我们提出了一套数字孪生集成的联邦学习（DTFL）方法，这些方法在保护数据隐私和通信效率的同时，提升了全局模型性能。具体而言，我们提出了五种新颖方法：基于数字孪生的元学习（DTML）、联邦参数融合（FPF）、逐层参数交换（LPE）、循环权重自适应（CWA）以及数字孪生知识蒸馏（DTKD）。每种方法都引入了一种独特机制来结合合成数据与真实数据。

    arXiv:2601.01701v2 Announce Type: replace-cross  Abstract: Anomaly detection is increasingly becoming crucial for maintaining the safety, reliability, and efficiency of industrial systems. Recently, with the advent of digital twins and data-driven decision-making, several statistical and machine-learning methods have been proposed. However, these methods face several challenges, such as dependence on only real sensor datasets, limited labeled data, high false alarm rates, and privacy concerns. To address these problems, we propose a suite of digital twin-integrated federated learning (DTFL) methods that enhance global model performance while preserving data privacy and communication efficiency. Specifically, we present five novel approaches: Digital Twin-Based Meta-Learning (DTML), Federated Parameter Fusion (FPF), Layer-wise Parameter Exchange (LPE), Cyclic Weight Adaptation (CWA), and Digital Twin Knowledge Distillation (DTKD). Each method introduces a unique mechanism to combine syn
    
[^217]: 私密且稳健对齐的改进界

    Improved Bounds for Private and Robust Alignment

    [https://arxiv.org/abs/2512.23816](https://arxiv.org/abs/2512.23816)

    本文首次为私密且稳健的语言模型对齐（包括离线和在线场景）建立了次优性差距的理论上界，并在联合隐私与破坏设置中证明现有离线算法可同时提供更强的隐私与破坏保证，从而在仅破坏场景中获得改进的界。

    

    本文从理论角度研究语言模型的私密且稳健对齐，通过在离线和在线两种设置下建立次优性差距的上界。我们考虑受隐私约束和/或对抗性破坏的偏好标签，并分析它们之间两种不同的相互作用：隐私优先和破坏优先。对于仅隐私设置，我们表明采用MLE风格算法的对数损失实现了接近最优的速率，这与传统观点相反。对于联合隐私与破坏设置，我们首先证明现有的离线算法实际上提供了比先前已知更强的保证——同时体现在破坏水平和隐私参数方面，这进一步在仅破坏场景中产生了改进的界。此外，我们还提出了私密且稳健在线对齐的首批结果。我们的结果得益于……

    arXiv:2512.23816v2 Announce Type: replace-cross  Abstract: In this paper, we study the private and robust alignment of language models from a theoretical perspective by establishing upper bounds on the suboptimality gap in both offline and online settings. We consider preference labels subject to privacy constraints and/or adversarial corruption, and analyze two distinct interplays between them: privacy-first and corruption-first. For the privacy-only setting, we show that log loss with an MLE-style algorithm achieves near-optimal rates, in contrast to conventional wisdom. For the joint privacy-and-corruption setting, we first demonstrate that existing offline algorithms in fact provide stronger guarantees -- simultaneously in terms of corruption level and privacy parameters -- than previously known, which further yields improved bounds in the corruption-only regime. In addition, we also present the first set of results for private and robust online alignment. Our results are enabled b
    
[^218]: 从自动标注视角解决半监督少样本学习

    Solving Semi-Supervised Few-Shot Learning from an Auto-Annotation Perspective

    [https://arxiv.org/abs/2512.10244](https://arxiv.org/abs/2512.10244)

    本文发现半监督少样本学习中，视觉-语言模型因输出概率分布平坦导致未标注数据无法被有效利用，从而提出应借鉴少样本学习思路利用开放资源来解决这一问题。

    

    半监督少样本学习（SSFSL）类似于自动标注等现实应用，其目标是从少量标注样本和大量未标注的任务特定样本中学习一个模型，以对未标注样本进行标注。尽管存在强大的开源视觉-语言模型（VLM）和开放世界数据，但现有的SSFSL文献在很大程度上忽略了这些资源。相比之下，相关领域少样本学习（FSL）已经利用这些资源来提升性能。可以说，为了解决现实世界的自动标注问题，SSFSL应该利用这些开放资源。为弥补这一差距，我们探索了使用已有的半监督学习方法对VLM进行微调。出乎意料的是，这些方法的性能显著低于不使用未标注数据的FSL基线。我们的深入分析揭示了失败的根源：VLM产生软最大概率的平坦分布，导致未标注数据零利用率和弱监督信号。

    arXiv:2512.10244v3 Announce Type: replace-cross  Abstract: Semi-supervised few-shot learning (SSFSL) resembles real-world applications such as auto-annotation, as it aims to learn a model from a few labeled and abundant unlabeled task-specific examples to annotate the unlabeled ones. Despite the availability of powerful open-source Vision-Language Models (VLMs) and open-world data, existing SSFSL literature largely neglects these resources. In contrast, the related area few-shot learning (FSL) has already exploited them to boost performance. Arguably, to solve real-world auto-annotation, SSFSL should leverage such open resources. To bridge this gap, we explore established SSL methods to finetune a VLM. Unexpectedly, they significantly underperform FSL baselines that do not use unlabeled data. Our in-depth analysis reveals the root cause of failure: VLMs produce flat distributions of softmax probabilities, resulting in zero utilization of unlabeled data and weak supervision signals. To 
    
[^219]: 基于自监督的专利表示学习

    Patent Representation Learning via Self-supervision

    [https://arxiv.org/abs/2511.10657](https://arxiv.org/abs/2511.10657)

    提出一种利用专利内部结构（如权利要求、背景等）作为自监督信号的混合dropout-部分正样本策略，有效提升了专利表示学习在多种检索任务中的迁移性。

    

    我们研究了基于对比学习目标的自监督专利表示学习。一个标准基线方法通过对同一文本应用两次独立的dropout掩码来构建正样本，但将这一方法应用于长篇幅、结构化的专利文档时，需要仔细校准。我们证明，通过调节温度和dropout率，仅使用dropout的训练方法可以显著增强，但其最佳配置依赖于评估任务，并且无法从标题-摘要检索统一迁移到权利要求-说明书检索。我们提出混合dropout-部分正样本策略，这是一种专利特定的视图构建方法：锚点为标题-摘要视图，正样本要么来自同一视图的dropout重编码，要么来自同一专利的其他部分（如权利要求、摘要、背景、附图或说明书）。该方法利用专利内部结构作为训练时信号，无需IPC标签。

    arXiv:2511.10657v2 Announce Type: replace-cross  Abstract: We study self-supervised patent representation learning with contrastive objectives. A standard baseline constructs positives by encoding the same text twice under independent dropout masks, but applying this recipe to long, structured patent documents requires careful calibration. We show that dropout-only training can be substantially strengthened by tuning temperature and dropout rate, yet its best configuration is evaluation-dependent and does not transfer uniformly from title--abstract retrieval to claim-to-disclosure retrieval. We propose mixed dropout--section positives, a patent-specific view construction strategy in which the anchor is the title--abstract view and the positive is sampled either from a dropout re-encoding of the same view or from another section of the same patent, such as claims, summary, background, drawings, or description. This uses patent-internal structure as a training-time signal without IPC lab
    
[^220]: 面向预测应用的数据驱动传感器布局：一种相关性辅助归因框架（CAAF）

    Data-driven Sensor Placement for Predictive Applications: A Correlation-Assisted Attribution Framework (CAAF)

    [https://arxiv.org/abs/2510.22517](https://arxiv.org/abs/2510.22517)

    提出了一种名为CAAF的机器学习框架，通过在特征归因前对传感器位置进行聚类，解决了高相关性数据下的最优传感器布局问题，并在多个实际动态系统中验证了其有效性。

    

    最优传感器布局（OSP）对于复杂物理系统中的高效、准确监测、控制和推断至关重要。我们提出了一种基于机器学习的特征归因（FA）框架，用于确定目标预测的最优传感器位置。特征归因量化了输入对模型输出的贡献；然而，它在处理实际应用中经常遇到的高度相关输入数据时存在困难。为了解决这一问题，我们提出了一种相关性辅助归因框架（CAAF），该框架在进行特征归因之前，对候选传感器位置进行聚类步骤，以减少冗余并增强泛化能力。我们首先通过一系列验证案例展示了所提出框架的核心原理，然后证明了其在现实动态系统（如结构健康监测、翼型升力预测以及湍流通道流动的壁面法向速度估计）中的有效性。结果显示该框架表现良好。

    arXiv:2510.22517v3 Announce Type: replace-cross  Abstract: Optimal sensor placement (OSP) is critical for efficient, accurate monitoring, control, and inference in complex physical systems. We propose a machine-learning-based feature attribution (FA) framework to identify OSP for target predictions. FA quantifies input contributions to a model output; however, it struggles with highly correlated input data often encountered in practical applications for OSP. To address this, we propose a Correlation-Assisted Attribution Framework (CAAF), which introduces a clustering step on the candidate sensor locations before performing FA to reduce redundancy and enhance generalizability. We first illustrate the core principles of the proposed framework through a series of validation cases, then demonstrate its effectiveness in realistic dynamical systems such as structural health monitoring, airfoil lift prediction, and wall-normal velocity estimation for turbulent channel flow. The results show t
    
[^221]: CSU-PCAST：用于中期集合降水预报的双分支Transformer框架

    CSU-PCAST: A Dual-Branch Transformer Framework for medium-range ensemble Precipitation Forecasting

    [https://arxiv.org/abs/2510.20769](https://arxiv.org/abs/2510.20769)

    本文提出了CSU-PCAST，一个基于Swin Transformer和双分支解码器的深度学习集合预报框架，用于全球中期降水预测，能够从GFS分析初始化并生成30个集合成员的15天预报。

    

    arXiv:2510.20769v2 公告类型：替换交叉 摘要：准确的中期降水预报对于水文气象风险管理至关重要，但无论是数值天气预报（NWP）系统还是数据驱动模型都面临挑战。我们提出了CSU-PCAST，一个基于深度学习的全球降水预测集合预报框架。该模型使用ERA5大气和地表变量（分辨率为0.25°）进行训练，降水标签来自NASA的IMERG数据集。CSU-PCAST利用57个预报变量和静态地理场来预测大气状态和6小时累积降水量。该框架采用Swin Transformer骨干网络，结合随机噪声调节、时间嵌入以及用于降水和非降水变量的双分支解码器。在推理过程中，CSU-PCAST从业务GFS分析初始化，使用自回归策略生成30个集合成员，预报时长长达15天。

    arXiv:2510.20769v2 Announce Type: replace-cross  Abstract: Accurate medium-range precipitation forecasting is essential for hydrometeorological risk management but remains challenging for both numerical weather prediction (NWP) systems and data-driven models. We present CSU-PCAST, a deep learning-based ensemble forecasting framework for global precipitation prediction. The model is trained using ERA5 atmospheric and surface variables at 0.25{\deg} resolution with precipitation labels from NASA's IMERG dataset. CSU-PCAST uses 57 prognostic variables and static geographical fields to predict both atmospheric states and 6-h accumulated precipitation. The framework employs a Swin Transformer backbone with stochastic noise conditioning, temporal embeddings, and a dual-branch decoder for precipitation and non-precipitation variables. During inference, CSU-PCAST is initialized from operational GFS analyses and generates 30 ensemble members out to 15 days using an autoregressive strategy. Eval
    
[^222]: 利用神经网络估算直接成像系外行星的轨道参数

    Estimating Orbital Parameters of Direct Imaging Exoplanet Using Neural Network

    [https://arxiv.org/abs/2510.17459](https://arxiv.org/abs/2510.17459)

    提出了一种结合流匹配后验估计和马尔可夫链蒙特卡洛的算法，在保持精度的同时大幅加速了系外行星轨道参数估算，速度比传统方法快数十至数百倍。

    

    在本工作中，我们提出了一种流匹配马尔可夫链蒙特卡洛（FM-MCMC）算法，用于估算系外行星系统的轨道参数，特别是针对仅涉及一颗系外行星的系统。与依赖贝叶斯框架内随机采样的传统方法相比，我们的方法首先利用流匹配后验估计（FMPE）高效地约束物理参数的先验范围，然后使用MCMC精确推断后验分布。例如，在绘架座β星b的轨道参数推断中，我们的模型在保持相当精度的同时实现了显著加速——比并行回火MCMC（PTMCMC）快77.8倍，比嵌套采样快365.4倍。此外，我们的FM-MCMC方法在所有方法中达到了最高的平均对数似然，展示了其优越的采样效率和准确性。这突显了该方法的可扩展性。

    arXiv:2510.17459v3 Announce Type: replace-cross  Abstract: In this work, we propose a flow-matching Markov chain Monte Carlo (FM-MCMC) algorithm for estimating the orbital parameters of exoplanetary systems, especially for those only one exoplanet is involved. Compared to traditional methods that rely on random sampling within the Bayesian framework, our approach first leverages flow matching posterior estimation (FMPE) to efficiently constrain the prior range of physical parameters, and then employs MCMC to accurately infer the posterior distribution. For example, in the orbital parameter inference of beta Pictoris b, our model achieved a substantial speed-up while maintaining comparable accuracy-running 77.8 times faster than Parallel Tempered MCMC (PTMCMC) and 365.4 times faster than nested sampling. Moreover, our FM-MCMC method also attained the highest average log-likelihood among all approaches, demonstrating its superior sampling efficiency and accuracy. This highlights the scal
    
[^223]: HOB：异构竞价环境下的整体优化竞价策略

    HOB: A Holistically Optimized Bidding Strategy under Heterogeneous Bidding Environments

    [https://arxiv.org/abs/2510.15238](https://arxiv.org/abs/2510.15238)

    本文提出HOB策略，通过使边际成本在异构渠道间可计算和对齐，并建模免费获胜概率与价格不确定性，实现了跨不同拍卖机制的全局协调竞价优化。

    

    arXiv:2510.15238v2 公告类型：替换交叉 摘要：在异构渠道间优化单个广告活动是工业自动竞价中的核心挑战。不同渠道的拍卖机制在排名规则（纯eCPM vs. UE增强评分）、定价格式（第一价格 vs. 第二价格）和竞价惯例（统一 vs. 非统一）上存在差异，而广告主则施加共享的活动级约束。我们提出HOB，它使得边际成本在异构渠道间变得可计算且可对齐，特别是对于存在自然结果与付费结果共存的第一价格拍卖，现有竞价公式无法产生实用的对齐边际成本形式。在全局层面，HOB推导出特定渠道的边际成本形式，并通过共享的边际成本目标协调不同渠道。在局部层面，HOB使用零膨胀指数分布对免费获胜概率和获胜价格不确定性进行建模，为非统一竞价环境提供高效的剩余最优竞价策略。

    arXiv:2510.15238v2 Announce Type: replace-cross  Abstract: Optimizing a single advertising campaign across heterogeneous channels is a central challenge in industrial autobidding. Auction mechanisms vary across channels in ranking rules (pure eCPM vs. UE-augmented scoring), pricing formats (first- vs. second-price), and bidding conventions (uniform vs. non-uniform), while advertisers impose shared campaign-level constraints. We propose HOB, which makes marginal cost (MC) computable and alignable across heterogeneous channels, especially for first-price auctions (FPA) with organic-paid coexistence, where existing bidding formulations do not yield a practical aligned MC form. At the global level, HOB derives channel-specific MC forms and coordinates disparate channels through a shared MC target. At the local level, HOB models free-win probability and winning-price uncertainty with a zero-inflated exponential distribution, yielding an efficient surplus-optimal bidding strategy for non-uni
    
[^224]: 面向视觉-语言-动作模型的流匹配策略的强化微调

    Reinforcement Fine-Tuning of Flow-Matching Policies for Vision-Language-Action Models

    [https://arxiv.org/abs/2510.09976](https://arxiv.org/abs/2510.09976)

    提出流策略优化算法，通过重新定义重要性采样解决了流匹配模型中强化微调的计算不可行问题，实现了对视觉-语言-动作模型的稳定在线强化微调。

    

    arXiv:2510.09976v2 公告类型：替换 摘要：视觉-语言-动作（VLA）模型（如OpenVLA、Octo和π₀）通过利用大规模演示数据展现出强大的泛化能力，但其性能仍从根本上受限于监督数据的质量和覆盖范围。强化学习（RL）为通过在线交互改进和微调VLA模型提供了一条有前景的路径。然而，在基于流匹配的模型中，传统的策略梯度方法因重要性采样过程难以处理（需要显式计算策略比率）而无法实际应用。为克服这一限制，我们提出了流策略优化（FPO）算法，该算法通过利用条件流匹配目标中每个样本的变化来重新定义重要性采样。此外，FPO通过整合结构感知的信用分配，实现了对π₀模型的稳定且可扩展的在线强化微调。

    arXiv:2510.09976v2 Announce Type: replace  Abstract: Vision-Language-Action (VLA) models such as OpenVLA, Octo, and $\pi_0$ have shown strong generalization by leveraging large-scale demonstrations, yet their performance is still fundamentally constrained by the quality and coverage of supervised data. Reinforcement learning (RL) provides a promising path for improving and fine-tuning VLAs through online interaction. However, conventional policy gradient methods are computationally infeasible in the context of flow-matching based models due to the intractability of the importance sampling process, which requires explicit computation of policy ratios. To overcome this limitation, we propose Flow Policy Optimization (FPO) algorithm, which reformulates importance sampling by leveraging per-sample changes in the conditional flow-matching objective. Furthermore, FPO achieves stable and scalable online reinforcement fine-tuning of the $\pi_0$ model by integrating structure-aware credit assig
    
[^225]: 基于精细化图随机特征的高效图建模

    Computationally-efficient Graph Modeling with Refined Graph Random Features

    [https://arxiv.org/abs/2510.07716](https://arxiv.org/abs/2510.07716)

    提出GRFs++方法，通过游走拼接技术将长随机游走替换为并行短游走，在保持无偏性的同时显著提升图核函数计算的效率，并扩展了游走终止机制。

    

    我们提出了精细化图随机特征（GRFs++），这是一类新的图随机特征方法，用于高效且精确地计算定义在图节点上的核函数。GRFs++解决了传统GRFs长期存在的局限性，包括难以建模远距离节点之间的关系。通过一种新颖的游走拼接技术——将多个较短游走连接起来而不破坏无偏性——减少了对采样长图随机游走的依赖。应用这些技术后，GRFs++继承了长游走提供的近似质量，但效率更高，将顺序的、低效的长游走采样转化为短游走的并行计算和矩阵-矩阵乘法。此外，GRFs++将简单的GRFs游走终止机制（具有固定停止概率的伯努利方案）扩展到更广泛的策略类别，对游走长度应用了一般分布。

    arXiv:2510.07716v2 Announce Type: replace  Abstract: We propose refined GRFs (GRFs++), a new class of Graph Random Features (GRFs) for efficient and accurate computations involving kernels defined on the nodes of a graph. GRFs++ resolve some of the long-standing limitations of regular GRFs, including difficulty modeling relationships between more distant nodes. They reduce dependence on sampling long graph random walks via a novel walk-stitching technique, concatenating several shorter walks without breaking unbiasedness. By applying these techniques, GRFs++ inherit the approximation quality provided by longer walks but with greater efficiency, trading sequential, inefficient sampling of a long walk for parallel computation of short walks and matrix-matrix multiplication. Furthermore, GRFs++ extend the simplistic GRFs walk termination mechanism (Bernoulli schemes with fixed halting probabilities) to a broader class of strategies, applying general distributions on the walks' lengths. Th
    
[^226]: 玻色高斯幺正算符的高效学习

    Efficient learning of bosonic Gaussian unitaries

    [https://arxiv.org/abs/2510.05531](https://arxiv.org/abs/2510.05531)

    本文首次提出了一个时间高效、严格分析的算法，用于学习玻色高斯幺正算符，其复杂度与模式数和能量参数呈多项式关系，并仅使用实验友好的光子资源。

    

    玻色高斯幺正算符是连续变量量子技术（如量子光学干涉测量和玻色纠错方案）中的核心构建模块。本文首次提出了一个具有严格分析的时间高效算法，用于学习未知的玻色高斯幺正算符。该算法能够以物理上合理的能量约束金刚石距离为度量，在最小最坏情况误差下给出未知幺正算符的估计。其运行时间和查询复杂度与模式数量、目标精度的倒数以及描述允许输入能量和幺正算符输出能量增长的自然能量参数呈多项式关系。该协议仅使用实验友好的光子资源：相干光和压缩光探针、无源线性光学器件以及外差/零差检测。随后，我们采用一种高效的经典后处理程序，利用辛几何方法进行参数提取。

    arXiv:2510.05531v2 Announce Type: replace-cross  Abstract: Bosonic Gaussian unitaries are fundamental building blocks of central continuous-variable quantum technologies such as quantum-optic interferometry and bosonic error-correction schemes. In this work, we present the first time-efficient algorithm for learning bosonic Gaussian unitaries with a rigorous analysis. Our algorithm produces an estimate of the unknown unitary that is accurate to small worst-case error, measured by the physically motivated energy-constrained diamond distance. Its runtime and query complexity scale polynomially with the number of modes, the inverse target accuracy, and natural energy parameters quantifying the allowed input energy and the unitary's output-energy growth.   The protocol uses only experimentally friendly photonic resources: coherent and squeezed probes, passive linear optics, and heterodyne/homodyne detection. We then employ an efficient classical post-processing routine that leverages a sym
    
[^227]: 注视我：通过可迁移的注意力引导吸引子实现可扩展的RAG投毒攻击

    Eyes-on-Me: Scalable RAG Poisoning through Transferable Attention-Steering Attractors

    [https://arxiv.org/abs/2510.00586](https://arxiv.org/abs/2510.00586)

    提出一种模块化RAG投毒攻击方法，通过可迁移的注意力吸引子实现零成本适应新目标，将攻击成功率提升至先前工作的2.6倍。

    

    现有针对检索增强生成系统的数据投毒攻击可扩展性差，因为它们需要针对每个目标短语对有毒文档进行昂贵的优化。我们提出"注视我"（Eyes-on-Me）攻击，这是一种模块化攻击方法，它将对抗性文档分解为可重复使用的"注意力吸引子"和"聚焦区域"。吸引子经过优化，可将注意力引导至聚焦区域。攻击者随后可以为检索器插入语义诱饵，或为生成器插入恶意指令，以近乎零成本适应新目标。这是通过引导我们经验证与攻击成功高度相关的一小部分注意力头来实现的。在18个端到端RAG设置（3个数据集×2个检索器×3个生成器）中，"注视我"将平均攻击成功率从21.9%提升至57.8%（提升35.9个百分点，是先前工作的2.6倍）。单个优化后的吸引子可迁移至未见过的黑盒检索器。

    arXiv:2510.00586v3 Announce Type: replace-cross  Abstract: Existing data poisoning attacks on retrieval-augmented generation (RAG) systems scale poorly because they require costly optimization of poisoned documents for each target phrase. We introduce Eyes-on-Me, a modular attack that decomposes an adversarial document into reusable **Attention Attractors** and **Focus Regions**. Attractors are optimized to direct attention to the Focus Region. Attackers can then insert semantic baits for the retriever or malicious instructions for the generator, adapting to new targets at near zero cost. This is achieved by steering a small subset of attention heads that we empirically identify as strongly correlated with attack success. Across 18 end-to-end RAG settings (3 datasets $\times$ 2 retrievers $\times$ 3 generators), Eyes-on-Me raises average attack success rates from 21.9 to 57.8 (+35.9 points, 2.6$\times$ over prior work). A single optimized attractor transfers to unseen black box retriev
    
[^228]: 图结构的旋转位置编码

    Rotary Position Encodings for Graphs

    [https://arxiv.org/abs/2509.22259](https://arxiv.org/abs/2509.22259)

    本文提出了一种名为WIRE的图结构旋转位置编码方法，通过基于图拉普拉斯谱的令牌旋转，将结构信息注入注意力机制，在理论上可恢复网格上的常规RoPE且渐近依赖图有效电阻，并兼容线性注意力。

    

    arXiv:2509.22259v4 公告类型：替换-交叉 摘要：我们研究了旋转位置编码（RoPE）——一种在大语言模型（LLM）和视觉变换器（ViT）中广泛采用的现代变换器位置编码算法——能够多大程度上应用于图结构数据。我们发现，根据图拉普拉斯矩阵的谱来旋转令牌，能够高效地将结构信息注入注意力机制中，从而提升在合成和真实世界图学习任务中的性能。这种方法被称为“波诱导旋转编码”（WIRE），具有引人注目的理论性质：它在网格结构上能恢复出常规的RoPE，并且渐近地依赖于图的有效电阻。与基于偏置的相对位置编码不同，WIRE与线性注意力机制兼容。

    arXiv:2509.22259v4 Announce Type: replace-cross  Abstract: We study the extent to which rotary position encodings (RoPE), a recent transformer position encoding algorithm broadly adopted in large language models (LLMs) and vision transformers (ViTs), can be applied to graph-structured data. We find that rotating tokens depending on the spectrum of the graph Laplacian efficiently injects structural information into the attention mechanism, boosting performance in synthetic and real-world graph learning tasks. This approach, coined _Wave-Induced Rotary Encodings_ (WIRE), enjoys intriguing theoretical properties: it recovers regular RoPE on grids, and depends asymptotically on the graph effective resistance. Unlike bias-based relative position encodings, WIRE is compatible with linear attention.
    
[^229]: 在部分可观测性下学习鲁棒的渗透测试策略：一项系统评估

    Learning Robust Penetration Testing Policies under Partial Observability: A systematic evaluation

    [https://arxiv.org/abs/2509.20008](https://arxiv.org/abs/2509.20008)

    本研究系统评估了在部分可观测性下，使用标准强化学习方法（如PPO）学习鲁棒且可迁移的渗透测试策略，并通过更具挑战性的网络基准来提升现实适用性。

    

    渗透测试，即模拟网络攻击以识别安全漏洞，提出了一个非常适合强化学习自动化的序贯决策问题。如同强化学习应用于现实世界问题的许多场景一样，部分可观测性构成了一个主要挑战，因为它破坏了马尔可夫决策过程中存在的马尔可夫性质。部分可观测的马尔可夫决策过程需要历史聚合或信念状态估计来学习成功的策略。我们研究了在不同规模主机网络上的随机、部分可观测的渗透测试场景，旨在通过更具挑战性和代表性的基准来更好地反映现实世界的复杂性。这种方法有助于开发更鲁棒且可迁移的策略，这对于确保在多样且不可预测的现实环境中实现可靠性能至关重要。

    arXiv:2509.20008v2 Announce Type: replace  Abstract: Penetration testing, the simulation of cyberattacks to identify security vulnerabilities, presents a sequential decision-making problem well-suited for reinforcement learning (RL) automation. Like many applications of RL to real-world problems, partial observability presents a major challenge, as it invalidates the Markov property present in Markov Decision Processes (MDPs). Partially Observable MDPs require history aggregation or belief state estimation to learn successful policies. We investigate stochastic, partially observable penetration testing scenarios over host networks of varying size, aiming to better reflect real-world complexity through more challenging and representative benchmarks. This approach leads to the development of more robust and transferable policies, which are crucial for ensuring reliable performance across diverse and unpredictable real-world environments. Using vanilla Proximal Policy Optimization (PPO) a
    
[^230]: 有限参考，可靠生成：低数据场景下表格数据生成的双组件框架

    Limited Reference, Reliable Generation: A Two-Component Framework for Tabular Data Generation in Low-Data Regimes

    [https://arxiv.org/abs/2509.09960](https://arxiv.org/abs/2509.09960)

    提出ReFine框架，通过提取可解释规则嵌入提示和双粒度过滤，解决了低数据场景下表格数据生成中的分布偏移和过采样问题。

    

    arXiv:2509.09960v2 公告类型：替换-跨领域 摘要：合成表格数据生成在机器学习中日益重要，当真实世界的高质量表格数据不足时，它支持下游应用。现有的表格生成方法，如生成对抗网络（GANs）和微调的大语言模型（LLMs），通常需要足够的参考数据，这限制了它们在样本稀缺的特定领域数据集中的有效性。虽然基于提示的LLMs无需参数调整即可提供灵活性，但它们常常生成具有局部冗余的分布偏移数据，导致下游任务性能下降。为克服这些问题，我们提出了ReFine框架，该框架（i）从可解释模型中提取符号化的if-then规则，并将其嵌入提示中，以明确引导生成过程朝向特定领域分布，以及（ii）应用双粒度过滤，减少过采样问题。

    arXiv:2509.09960v2 Announce Type: replace-cross  Abstract: Synthetic tabular data generation is increasingly essential in machine learning, supporting downstream applications when real-world, high-quality tabular data is insufficient. Existing tabular generation approaches, such as generative adversarial networks (GANs) and fine-tuned Large Language Models (LLMs), typically require sufficient reference data, limiting their effectiveness in domain-specific datasets with scarce records. While prompt-based LLMs offer flexibility without parameter tuning, they often generate distributionally drifted data with localized redundancy, leading to degradation in downstream task performance. To overcome these issues, we propose ReFine, a framework that (i) extracts symbolic if-then rules from interpretable models and embeds them into prompts to explicitly guide the generation process toward the domain-specific distribution, and (ii) applies dual-granularity filtering that mitigates over-sampling 
    
[^231]: 重建对齐改进了统一多模态模型

    Reconstruction Alignment Improves Unified Multimodal Models

    [https://arxiv.org/abs/2509.07295](https://arxiv.org/abs/2509.07295)

    提出一种名为RECA的后训练方法，利用视觉理解编码器嵌入作为密集监督信号，无需文本标题即可重新对齐统一多模态模型的理解与生成，从而显著提升图像生成和编辑的保真度。

    

    arXiv:2509.07295v4 公告类型：替换交叉 摘要：统一多模态模型（UMMs）在单一架构中统一了视觉理解与生成。然而，传统的训练依赖于图像-文本对（或序列），其标题通常稀疏且缺失细粒度的视觉细节，即使它们用数百个单词描述一张简单的图像。我们引入了重建对齐（RECA），一种资源高效的后训练方法，利用视觉理解编码器的嵌入作为密集的“文本提示”，在无需标题的情况下提供丰富的监督。具体而言，RECA将UMM以其自身的视觉理解嵌入为条件，并通过自监督重建损失优化其重建输入图像，从而重新对齐理解与生成。尽管方法简单，但RECA具有广泛适用性：在自回归、掩码自回归和基于扩散的UMMs中，它一致地提升了生成和编辑的保真度。仅需2（此处指训练数据或步骤的少量性，但原文不完整，按常规翻译为“仅需2”以保留原意）。

    arXiv:2509.07295v4 Announce Type: replace-cross  Abstract: Unified multimodal models (UMMs) unify visual understanding and generation within a single architecture. However, conventional training relies on image-text pairs (or sequences) whose captions are typically sparse and miss fine-grained visual details, even when they use hundreds of words to describe a simple image. We introduce Reconstruction Alignment (RECA), a resource-efficient post-training method that leverages visual understanding encoder embeddings as dense "text prompts", providing rich supervision without captions. Concretely, RECA conditions a UMM on its own visual understanding embeddings and optimizes it to reconstruct the input image with a self-supervised reconstruction loss, thereby realigning understanding and generation. Despite its simplicity, RECA is broadly applicable: across autoregressive, masked-autoregressive, and diffusion-based UMMs, it consistently improves generation and editing fidelity. With only 2
    
[^232]: Huracan：一种用于集合数据同化和天气预报的、技术精湛的端到端数据驱动系统

    Huracan: A skillful end-to-end data-driven system for ensemble data assimilation and weather prediction

    [https://arxiv.org/abs/2508.18486](https://arxiv.org/abs/2508.18486)

    Huracan是首个仅依赖观测数据、无需传统数值天气预报初始条件，就能实现与最先进NWP相当精度的端到端集合天气预报系统。

    

    过去几年中，基于机器学习的数据驱动天气预报正在改变业务天气预报的格局，它能在使用传统数值天气预报（NWP）一小部分计算能力的同时提供更准确的预报。然而，这些模型仍然依赖于NWP的初始条件，这限制了它们的预报能力上限。此后，虽然提出了一些端到端系统，但它们在预报技能上仍未达到最先进的NWP竞争对手的水平。在这项工作中，我们提出了Huracan，一个观测驱动的天气预报系统，它将集合数据同化模型与预报模型相结合，仅依赖观测数据作为输入，就能生成高度准确的预报。Huracan不仅是第一个提供集合初始条件和端到端集合天气预报的系统，也是第一个在精度上达到与最先进NWP系统相当水平的端到端系统。

    arXiv:2508.18486v2 Announce Type: replace-cross  Abstract: Over the past few years, machine learning-based data-driven weather prediction has been transforming operational weather forecasting by providing more accurate forecasts while using a mere fraction of computing power compared to traditional numerical weather prediction (NWP). However, those models still rely on initial conditions from NWP, putting an upper limit on their forecast abilities. A few end-to-end systems have since been proposed, but they have yet to match the forecast skill of state-of-the-art NWP competitors. In this work, we propose Huracan, an observation-driven weather forecasting system which combines an ensemble data assimilation model with a forecast model to produce highly accurate forecasts relying only on observations as inputs. Huracan is not only the first to provide ensemble initial conditions and end-to-end ensemble weather forecasts, but also the first end-to-end system to achieve an accuracy comparab
    
[^233]: 学习选择最大团算法：从传统机器学习到双通道混合神经架构

    Learning to Select Maximum Clique Algorithms: From Traditional Machine Learning to a Dual-Channel Hybrid Neural Architecture

    [https://arxiv.org/abs/2508.08005](https://arxiv.org/abs/2508.08005)

    提出了一种融合传统机器学习与图神经网络的双通道模型GAT-MLP，通过同时捕捉图的结构特征和全局属性，实现了对最大团问题最优算法的实例感知选择。

    

    摘要：最大团问题（MCP）是一个NP难问题，在生物信息学、网络科学和社会计算等领域有广泛应用，然而没有任何单一算法能在所有不同图实例上始终优于其他算法。这突显了对实例感知算法选择的迫切需求，而这一领域在MCP中仍鲜有探索。为填补这一空白，我们提出了一种新颖的基于学习的框架，该框架融合了传统机器学习和图神经网络。我们首先通过在多样化的图集合上执行四种最先进的精确MCP求解器并提取结构特征，构建了一个基准数据集。对传统分类器的评估表明，随机森林是一个强基线，并揭示出连通性和拓扑特征是性能的关键预测因子。基于这些发现，我们开发了GAT-MLP，一种双通道模型，该模型结合了图注意力网络和MLP，以同时捕捉图的结构特征和全局属性，从而实现对最优MCP算法的有效选择。

    arXiv:2508.08005v4 Announce Type: replace-cross  Abstract: The Maximum Clique Problem (MCP) is an NP-hard problem with wide-ranging applications in fields such as bioinformatics, network science, and social computing, yet no single algorithm consistently outperforms all others across diverse graph instances. This underscores the critical need for instance-aware algorithm selection, a domain that remains largely unexplored for the MCP. To address this gap, we propose a novel learning-based framework that integrates both traditional machine learning and graph neural networks. We first construct a benchmark dataset by executing four state-of-the-art exact MCP solvers on a diverse collection of graphs and extracting structural features. An evaluation of conventional classifiers establishes Random Forest as a strong baseline and reveals that connectivity and topological features are key predictors of performance. Building on these insights, we develop GAT-MLP, a dual-channel model that comb
    
[^234]: DMSC：面向时间序列预测的动态多尺度协调框架

    DMSC: Dynamic Multi-Scale Coordination Framework for Time Series Forecasting

    [https://arxiv.org/abs/2508.02753](https://arxiv.org/abs/2508.02753)

    提出了一种动态多尺度协调框架（DMSC），通过内置的多尺度补丁分解、三元交互和自适应路由机制，解决了时间序列预测中静态分解、碎片化依赖和僵化融合的问题。

    

    时间序列预测（TSF）在建模跨不同尺度的复杂时间依赖关系方面持续面临挑战。尽管近期研究利用不同的分解操作以及基于CNN、MLP或Transformer的新型架构取得了进展，但现有方法仍受限于静态分解策略、碎片化的依赖建模以及僵化的融合机制，这限制了它们对复杂时间依赖关系的建模能力。为分别明确解决上述三个问题，我们提出了一种新颖的动态多尺度协调框架（DMSC），该框架包含多尺度补丁分解模块（EMPD）、三元交互模块（TIB）和自适应尺度路由MoE模块（ASR-MoE）。具体而言，EMPD被设计为一个内置组件，能够动态地将序列分割成具有指数级粒度分级的层次化补丁，通过输入自适应补丁消除预定义的尺度约束。

    arXiv:2508.02753v5 Announce Type: replace-cross  Abstract: Time Series Forecasting (TSF) faces persistent challenges in modeling intricate temporal dependencies across different scales. Despite recent advances leveraging different decomposition operations and novel architectures based on CNN, MLP or Transformer, existing methods still struggle with static decomposition strategies, fragmented dependency modeling, and inflexible fusion mechanisms, limiting their ability to model intricate temporal dependencies. To explicitly solve the mentioned three problems respectively, we propose a novel Dynamic Multi-Scale Coordination Framework (DMSC) with Multi-Scale Patch Decomposition block (EMPD), Triad Interaction Block (TIB) and Adaptive Scale Routing MoE block (ASR-MoE). Specifically, EMPD is designed as a built-in component to dynamically segment sequences into hierarchical patches with exponentially scaled granularities, eliminating predefined scale constraints through input-adaptive patch
    
[^235]: 归一化流是连续控制中的有效模型

    Normalizing Flows are Capable Models for Continuous Control

    [https://arxiv.org/abs/2505.23527](https://arxiv.org/abs/2505.23527)

    本文证明归一化流在连续控制中具备足够的表达能力，是扩散模型和自回归Transformer的有效替代方案。

    

    现代强化学习算法通过使用强大的概率模型（如Transformer、基于能量的模型以及扩散/流模型）取得了成功。为此，强化学习研究者常常选择付出代价将这些模型融入算法中——扩散模型虽然表达能力强，但由于依赖求解微分方程而计算成本高昂；自回归Transformer模型可扩展，但通常需要学习离散表示。相比之下，归一化流似乎提供了一种有吸引力的替代方案，因为它们能够在无需求解微分方程或使用自回归架构的情况下实现似然计算和采样。然而，它们在强化学习中的潜力却受到有限关注，部分原因是人们普遍认为归一化流缺乏足够的表达能力。我们证明情况并非如此。基于归一化流领域的最新研究，我们提出...

    arXiv:2505.23527v4 Announce Type: replace  Abstract: Modern reinforcement learning (RL) algorithms have found success by using powerful probabilistic models, such as transformers, energy-based models, and diffusion/flow-based models. To this end, RL researchers often choose to pay the price of accommodating these models into their algorithms -- diffusion models are expressive, but are computationally intensive due to their reliance on solving differential equations, while autoregressive transformer models are scalable but typically require learning discrete representations. Normalizing flows (NFs), by contrast, seem to provide an appealing alternative, as they enable likelihoods and sampling without solving differential equations or autoregressive architectures. However, their potential in RL has received limited attention, partly due to the prevailing belief that normalizing flows lack sufficient expressivity. We show that this is not the case. Building on recent work in NFs, we propo
    
[^236]: 没有免费午餐：预测驱动推断的非渐近分析

    No Free Lunch: Non-Asymptotic Analysis of Prediction-Powered Inference

    [https://arxiv.org/abs/2505.20178](https://arxiv.org/abs/2505.20178)

    本文通过有限样本分析证明，PPI++方法并非总是优于仅使用黄金标准标签，其优势仅在伪标签与黄金标准标签的相关性高于特定阈值时成立。

    

    预测驱动推断（PPI）是一种将黄金标准标签与可能有噪声的伪标签相结合进行统计估计的流行策略。先前的研究表明，PPI++（PPI的一种自适应形式）存在渐近意义上的“免费午餐”，即PPI++的渐近方差总是小于或等于仅使用黄金标准标签所获得的方差。值得注意的是，这一结论无论伪标签的质量如何都成立。在本工作中，我们通过对均值估计问题中PPI++的估计误差进行精确的有限样本分析，揭示了这一结果背后的真相。我们给出了一个“没有免费午餐”的结论，刻画了在哪些设定（以及样本量）下，PPI++的估计误差会明确差于仅使用黄金标准标签。具体而言，PPI++能够表现更好的充分必要条件是伪标签与黄金标准标签之间的相关性超过某个依赖于样本量的阈值。

    arXiv:2505.20178v2 Announce Type: replace-cross  Abstract: Prediction-Powered Inference (PPI) is a popular strategy for combining gold-standard and possibly noisy pseudo-labels to perform statistical estimation. Prior work has shown an asymptotic \enquote{free lunch} for PPI++, an adaptive form of PPI, showing that the \textit{asymptotic} variance of PPI++ is always less than or equal to the variance obtained from using gold-standard labels alone. Notably, this result holds \textit{regardless of the quality of the pseudo-labels}. In this work, we demystify this result by conducting an exact finite-sample analysis of the estimation error of PPI++ on the mean estimation problem. We give a \enquote{no free lunch} result, characterizing the settings (and sample sizes) where PPI++ has provably worse estimation error than using gold-standard labels alone. Specifically, PPI++ will outperform if and only if the correlation between pseudo- and gold-standard is above a certain level that depends
    
[^237]: 通过层间一致性聚合缓解大型视觉语言模型中的幻觉

    Mitigating Hallucinations via Inter-Layer Consistency Aggregation in Large Vision-Language Models

    [https://arxiv.org/abs/2505.12343](https://arxiv.org/abs/2505.12343)

    本文提出了一种名为DCLA的免训练解码方法，通过聚合前层表示构建动态语义参考来纠正语义偏差的层，从而有效缓解大型视觉语言模型中的幻觉问题，并在多个模型和基准上显著提升性能。

    

    arXiv:2505.12343v2 公告类型：替换交叉 摘要：尽管大型视觉语言模型（LVLMs）能力令人印象深刻，但它们仍然容易产生幻觉，即生成的内容与输入图像不一致。现有的免训练幻觉缓解方法通常存在性能不稳定且对超参数设置高度敏感的问题，这限制了其实用性和更广泛的采用。在本文中，我们提出了一种通过层聚合进行层间一致性解码的方法（DCLA），这是一种免训练的解码机制，无需重新训练、微调或访问外部知识库。具体来说，DCLA通过聚合前几层的表示来构建动态语义参考，并用其纠正语义偏差的层，从而强制实现层间一致性。在七个LVLMs和多个基准上的实验证明了DCLA的通用性：它在MME指标上比标准解码高出28.58分。

    arXiv:2505.12343v2 Announce Type: replace-cross  Abstract: Despite the impressive capabilities of Large Vision-Language Models (LVLMs), they remain susceptible to hallucinations, where generated content is inconsistent with the input image. Existing training-free hallucination mitigation methods often suffer from unstable performance and high sensitivity to hyperparameter settings, which limits their practicality and broader adoption. In this paper, we propose Decoding with Inter-layer Consistency via Layer Aggregation (DCLA), a training-free decoding mechanism that requires no retraining, fine-tuning, or access to external knowledge bases. Specifically, DCLA constructs a dynamic semantic reference by aggregating representations from previous layers and uses it to correct semantically deviated layers, thereby enforcing inter-layer consistency. Experiments across seven LVLMs and multiple benchmarks demonstrate the generality of DCLA: it surpasses standard decoding by 28.58 MME points on
    
[^238]: Chisme：异构感知的八卦学习算法

    Chisme: Heterogeneity-Aware Gossip Learning

    [https://arxiv.org/abs/2505.09854](https://arxiv.org/abs/2505.09854)

    本文提出Chisme，一种完全去中心化的分布式学习算法，通过感知客户端和数据分布的异质性，解决了网络边缘环境中异构数据与间歇性连接带来的鲁棒智能实现挑战。

    

    arXiv:2505.09854v3 公告类型：替换  摘要：随着终端用户设备能力的提升以及互联网边缘对智能服务需求的增长，分布式学习已成为智能边缘的关键使能技术。联邦学习（FL）和去中心化联邦学习（DFL）等现有方法能够在客户端之间实现隐私保护的分布式学习，而八卦学习（GL）方法则旨在解决资源受限、连接困难的无基础设施环境中的潜在挑战。然而，大多数分布式学习方法假设数据分布大致同质，可能未考虑或利用客户端及其底层数据分布的异质性。本文介绍了Chisme，一种新型完全去中心化的分布式学习算法，旨在应对网络边缘环境中由异构数据分布、间歇性连接等特征带来的鲁棒智能实现挑战。

    arXiv:2505.09854v3 Announce Type: replace  Abstract: As end-user device capability increases and demand for intelligent services at the Internet's edge rises, distributed learning has emerged as a key enabling technology for the intelligent edge. Existing approaches like federated learning (FL) and decentralized FL (DFL) enable privacy-preserving distributed learning among clients, while gossip learning (GL) approaches have emerged to address the potential challenges in resource-constrained, connectivity-challenged infrastructure-less environments. However, most distributed learning approaches assume largely homogeneous data distributions and may not consider or exploit the heterogeneity of clients and their underlying data distributions. This paper introduces Chisme, a novel fully decentralized distributed learning algorithm designed to address the challenges of implementing robust intelligence in network edge contexts characterized by heterogeneous data distributions, episodic connec
    
[^239]: 面向通用反应条件的贝叶斯优化

    Bayesian Optimization for General Reaction Conditions

    [https://arxiv.org/abs/2502.18966](https://arxiv.org/abs/2502.18966)

    本文提出CurryBO框架，通过将通用反应条件优化问题形式化为柯里化函数的贝叶斯优化，实现了在多种底物上高效寻找高性能通用条件的方法。

    

    arXiv:2502.18966v2 公告类型：替换 摘要：能够在多种底物上持续实现高性能的通用化学反应条件，对于实际应用（如库合成和高通量实验）至关重要。然而，高效识别此类条件一直是一个长期挑战，因为它需要在面对条件和底物的不确定性时做出决策，同时最小化所需实验次数。在此，我们引入了CurryBO，一个面向通用性优化的高级框架。通过将问题形式化为对柯里化函数的贝叶斯优化，CurryBO提供了一个统一框架，能够容纳不同的通用性定义（例如，底物上的平均产率），并支持多种底物和条件选择策略。我们在实验反应优化的四个基准任务上评估了该框架，并系统分析了关键算法组成部分。

    arXiv:2502.18966v2 Announce Type: replace  Abstract: General chemical reaction conditions that achieve consistently high performance across multiple substrates are important for practical applications such as library synthesis and high-throughput experimentation. However, identifying such conditions efficiently has been a longstanding challenge, as it requires decision making under uncertainty with respect to both conditions and substrates, while minimizing the number of required experiments. Here, we introduce CurryBO, a high-level framework for generality-oriented optimization. By formalizing the problem as Bayesian optimization over curried functions, CurryBO provides a unified framework that accommodates different generality definitions (e.g., mean yield across substrates), and supports a range of substrate and condition selection strategies. We evaluate this framework on four benchmark tasks in experimental reaction optimization, and systematically analyze key algorithmic componen
    
[^240]: 学习解释空中交通态势

    Learning to Explain Air Traffic Situation

    [https://arxiv.org/abs/2502.10764](https://arxiv.org/abs/2502.10764)

    提出了一种基于Transformer的多智能体轨迹模型，通过注意力分数量化单架飞机对整体空中交通动态的影响，以解释复杂空中交通态势。

    

    arXiv:2502.10764v4 公告类型：替换 摘要：理解空中交通管制员如何构建复杂空中交通态势的心理“图像”至关重要，但由于飞机、飞行员和管制员之间固有的复杂高维交互，这仍然是一个挑战。先前关于空中交通管制员策略及其对交通态势心理图像建模的研究，通常集中于特定的空中交通管制任务或飞机之间的成对交互，忽略了捕捉空中交通态势的整体动态。为解决这一问题，我们提出了一种基于机器学习的空中交通态势解释框架。具体而言，我们采用基于Transformer的多智能体轨迹模型，该模型既包含飞机的时空运动，也包含它们之间的社会交互。通过从模型中提取注意力分数，我们可以量化单个飞机对整体交通动态的影响。

    arXiv:2502.10764v4 Announce Type: replace  Abstract: Understanding how air traffic controllers construct a mental 'picture' of complex air traffic situations is crucial but remains a challenge due to the inherently intricate, high-dimensional interactions between aircraft, pilots, and controllers. Previous work on modeling the strategies of air traffic controllers and their mental image of traffic situations often centers on specific air traffic control tasks or pairwise interactions between aircraft, neglecting to capture the comprehensive dynamics of an air traffic situation. To address this issue, we propose a machine learning-based framework for explaining air traffic situations. Specifically, we employ a Transformer-based multi-agent trajectory model that encapsulates both the spatio-temporal movement of aircraft and social interaction between them. By deriving attention scores from the model, we can quantify the influence of individual aircraft on overall traffic dynamics. This p
    
[^241]: 现实世界中AI生成图像检测器的对抗鲁棒性

    Adversarial Robustness of AI-Generated Image Detectors in the Real World

    [https://arxiv.org/abs/2410.01574](https://arxiv.org/abs/2410.01574)

    本文揭示了当前最先进的AI生成图像检测器在现实条件下易受对抗样本攻击，攻击者无需了解检测器内部架构即可显著降低其性能，且多数攻击在常见图像处理后仍有效。

    

    arXiv:2410.01574v4 公告类型：替换交叉 摘要：生成式人工智能（GenAI）能力的快速发展伴随着其被滥用的令人担忧的增长。特别是以图像形式生成可信的虚假信息，对公众对民主进程的信任构成了重大威胁。因此，迫切需要开发能够可靠区分真实内容与AI生成内容的工具。大多数检测方法基于训练用于识别取证伪影的神经网络。在本工作中，我们证明了当前最先进的分类器在现实条件下容易受到对抗样本的攻击。通过包含四种检测方法和五种攻击算法的广泛实验，我们展示了攻击者可以在不了解检测器内部架构的情况下显著降低分类性能。值得注意的是，即使在图像经过常见数字操作（如重新压缩、调整大小和去噪）后，大多数攻击仍然有效。

    arXiv:2410.01574v4 Announce Type: replace-cross  Abstract: The rapid advancement of Generative Artificial Intelligence (GenAI) capabilities is accompanied by a concerning rise in its misuse. In particular the generation of credible misinformation in the form of images poses a significant threat to the public trust in democratic processes. Consequently, there is an urgent need to develop tools to reliably distinguish between authentic and AI-generated content. The majority of detection methods are based on neural networks that are trained to recognize forensic artifacts. In this work, we demonstrate that current state-of-the-art classifiers are vulnerable to adversarial examples under real-world conditions. Through extensive experiments, comprising four detection methods and five attack algorithms, we show that an attacker can dramatically decrease classification performance, without internal knowledge of the detector's architecture. Notably, most attacks remain effective even when imag
    
[^242]: 用于保障去中心化联邦学习安全的拜占庭鲁棒聚合方法

    Byzantine-Robust Aggregation for Securing Decentralized Federated Learning

    [https://arxiv.org/abs/2409.17754](https://arxiv.org/abs/2409.17754)

    本文提出了一种名为WFAgg的拜占庭鲁棒聚合算法，通过多过滤器机制在去中心化联邦学习中同时抵御攻击并增强动态拓扑鲁棒性。

    

    arXiv:2409.17754v2 公告类型：替换-交叉 摘要：联邦学习作为一种分布式机器学习方法，通过在设备本地训练AI模型来解决隐私问题。去中心化联邦学习通过消除中央服务器来扩展联邦学习范式，从而通过避免单点故障来增强可扩展性和鲁棒性。然而，去中心化联邦学习在优化安全性方面面临重大挑战，因为文献中提出的大多数拜占庭鲁棒算法都是为集中式场景设计的。在本文中，我们提出了一种新颖的拜占庭鲁棒聚合算法，以增强去中心化联邦学习环境的安全性，命名为WFAgg。该方案通过采用多个过滤器来识别和缓解拜占庭攻击，同时应对不利条件并增强动态去中心化拓扑的鲁棒性。实验结果表明了所提算法的有效性。

    arXiv:2409.17754v2 Announce Type: replace-cross  Abstract: Federated Learning (FL) emerges as a distributed machine learning approach that addresses privacy concerns by training AI models locally on devices. Decentralized Federated Learning (DFL) extends the FL paradigm by eliminating the central server, thereby enhancing scalability and robustness through the avoidance of a single point of failure. However, DFL faces significant challenges in optimizing security, as most Byzantine-robust algorithms proposed in the literature are designed for centralized scenarios. In this paper, we present a novel Byzantine-robust aggregation algorithm to enhance the security of Decentralized Federated Learning environments, coined WFAgg. This proposal handles adverse conditions and strengthens the robustness of dynamic decentralized topologies at the same time by employing multiple filters to identify and mitigate Byzantine attacks. Experimental results demonstrate the effectiveness of the proposed a
    
[^243]: 两人零和随机博弈中基于最佳响应的去中心化学习：有限样本分析

    Decentralized Best-Response-Based Learning in Two-Player Zero-Sum Stochastic Games: A Finite-Sample Analysis

    [https://arxiv.org/abs/2409.01447](https://arxiv.org/abs/2409.01447)

    本文通过有限样本分析，证明了在两人零和矩阵博弈和随机博弈中，基于最佳响应的去中心化学习算法能够以样本复杂度$\mathcal{O}(\epsilon^{-1})$找到$\epsilon$-纳什均衡。

    

    arXiv:2409.01447v3 公告类型：替换 摘要：我们提出了两人零和矩阵博弈和随机博弈中去中心化学习的有限样本分析，重点关注基于最佳响应的学习算法。在矩阵博弈中，学习算法是基于收益且对称的：每个玩家仅使用自身的收益观测值更新其策略，逐步向对手最新策略的估计平滑最佳响应移动。对于随机博弈，我们基于这一矩阵博弈原语开发了一种称为带平滑最佳响应的值迭代（VI-SBR）的学习算法，该算法将诱导矩阵博弈中的平滑最佳响应学习与去中心化、无模型的极小值值迭代近似相结合。我们在两种设置中建立了有限样本保证。对于矩阵博弈，我们的结果意味着找到$\epsilon$-纳什分布的样本复杂度为$\mathcal{O}(\epsilon^{-1})$，并且在显式探索下，复杂度为$\tilde{\ma$

    arXiv:2409.01447v3 Announce Type: replace  Abstract: We present a finite-sample analysis of decentralized learning in two-player zero-sum matrix games and stochastic games, with a focus on best-response-based learning algorithms. In matrix games, the learning algorithm is payoff-based and symmetric: each player updates its policy using only its own payoff observations, incrementally moving toward an estimated smoothed best response to the opponent's latest policy. For stochastic games, we build on this matrix-game primitive to develop a learning algorithm called value iteration with smoothed best response (VI-SBR), which combines smoothed-best-response learning in induced matrix games with a decentralized, model-free approximation of minimax value iteration. We establish finite-sample guarantees in both settings. For matrix games, our results imply a sample complexity of $\mathcal{O}(\epsilon^{-1})$ for finding an $\epsilon$-Nash distribution and, with explicit exploration, $\tilde{\ma
    
[^244]: 神经网络的过度参数化与对抗鲁棒性：综述与实证分析

    Over-parameterization and Adversarial Robustness in Neural Networks: An Overview and Empirical Analysis

    [https://arxiv.org/abs/2406.10090](https://arxiv.org/abs/2406.10090)

    本文通过实证分析，揭示了过度参数化神经网络在对抗鲁棒性上的矛盾结论可能源于攻击方法的失效，并重新评估了其真实鲁棒性。

    

    arXiv:2406.10090v3 公告类型：替换 摘要：凭借其庞大的容量，过度参数化的神经网络展现出卓越的预测能力和泛化性。然而，拥有大的参数空间被认为是神经网络易受对抗样本攻击的主要嫌疑之一——对抗样本是专门设计的输入样本，旨在诱导出预期的错误分类。相关文献中既有支持也有反对过度参数化网络鲁棒性的矛盾言论。这些矛盾发现可能源于用于评估网络鲁棒性的攻击方法失效。先前研究表明，根据所考虑的模型，生成对抗样本的算法可能无法正常工作，导致高估模型的鲁棒性。在本工作中，我们通过实证研究过度参数化网络对抗对抗样本的鲁棒性。然而，与以往工作不同，我们采用了...

    arXiv:2406.10090v3 Announce Type: replace  Abstract: Thanks to their extensive capacity, over-parameterized neural networks exhibit superior predictive capabilities and generalization. However, having a large parameter space is considered one of the main suspects of the neural networks' vulnerability to adversarial example -- input samples crafted ad-hoc to induce a desired misclassification. Relevant literature has claimed contradictory remarks in support of and against the robustness of over-parameterized networks. These contradictory findings might be due to the failure of the attack employed to evaluate the networks' robustness. Previous research has demonstrated that depending on the considered model, the algorithm employed to generate adversarial examples may not function properly, leading to overestimating the model's robustness. In this work, we empirically study the robustness of over-parameterized networks against adversarial examples. However, unlike the previous works, we a
    
[^245]: 基于比较的梯度测试与估计

    Gradient Testing and Estimation by Comparisons

    [https://arxiv.org/abs/2405.11454](https://arxiv.org/abs/2405.11454)

    本文提出仅通过比较函数值大小的预言机，实现了光滑函数梯度的高效测试与估计，并在经典和量子模型下分别达到了最优或对数级的查询复杂度。

    

    我们研究利用仅能比较两个点函数值大小的比较预言机，对光滑函数进行梯度测试与梯度估计。对于任意光滑函数 $f\colon\mathbb R^n\to\mathbb R$、点 $\mathbf{x}\in\mathbb R^n$ 以及 $\varepsilon>0$，我们设计了一种梯度测试算法，能够以 $O(1)$ 次查询判断归一化梯度 $\nabla f(\mathbf{x})/\|\nabla f(\mathbf{x})\|$ 是 $\varepsilon$-接近还是 $2\varepsilon$-远离给定的单位向量 $\mathbf{v}$；同时，我们还设计了一种梯度估计算法，能以 $O(n\log(1/\varepsilon))$ 次查询输出 $\nabla f(\mathbf{x})/\|\nabla f(\mathbf{x})\|$ 的 $\varepsilon$-估计，并证明该查询复杂度是最优的。此外，我们研究了量子比较预言机模型下的梯度估计，其中查询可以叠加进行，并开发了一种仅需 $O(\log (n/\varepsilon))$ 次查询的量子算法。

    arXiv:2405.11454v3 Announce Type: replace  Abstract: We study gradient testing and gradient estimation of smooth functions using only a comparison oracle that, given two points, indicates which one has the larger function value. For any smooth $f\colon\mathbb R^n\to\mathbb R$, $\mathbf{x}\in\mathbb R^n$, and $\varepsilon>0$, we design a gradient testing algorithm that determines whether the normalized gradient $\nabla f(\mathbf{x})/\|\nabla f(\mathbf{x})\|$ is $\varepsilon$-close or $2\varepsilon$-far from a given unit vector $\mathbf{v}$ using $O(1)$ queries, as well as a gradient estimation algorithm that outputs an $\varepsilon$-estimate of $\nabla f(\mathbf{x})/\|\nabla f(\mathbf{x})\|$ using $O(n\log(1/\varepsilon))$ queries which we prove to be optimal. Furthermore, we study gradient estimation in the quantum comparison oracle model where queries can be made in superpositions, and develop a quantum algorithm using $O(\log (n/\varepsilon))$ queries.
    
[^246]: 从有偏样本中学习

    Learning from a Biased Sample

    [https://arxiv.org/abs/2209.01754](https://arxiv.org/abs/2209.01754)

    本文提出了一种新的条件Γ-有偏抽样模型来量化训练数据中的抽样偏差，并利用分布鲁棒优化框架开发了一种元方法，以在部署时仍能获得良好性能的决策规则。

    

    基于经验风险最小化的数据驱动决策方法要求训练数据与决策规则部署时所面临的条件相同。然而，在许多场景中，我们可能担心训练样本存在偏差，即某些群体（基于可观测或不可观测特征）相对于总体可能被低估或高估；在这种情况下，基于训练集的经验风险最小化可能无法生成在部署时表现良好的决策规则。我们提出了一种称为条件Γ-有偏抽样的抽样偏差模型，其中观测协变量可以任意影响样本选择概率，但样本选择概率中未解释的变化量被常数因子所限制。应用分布鲁棒优化框架，我们提出了一种元方法。

    arXiv:2209.01754v5 Announce Type: replace-cross  Abstract: The empirical risk minimization approach to data-driven decision making requires access to training data drawn under the same conditions as those that will be faced when the decision rule is deployed. However, in a number of settings, we may be concerned that our training sample is biased in the sense that some groups (characterized by either observable or unobservable attributes) may be under- or over-represented relative to the general population; and in this setting empirical risk minimization over the training set may fail to yield rules that perform well at deployment. We propose a model of sampling bias called conditional $\Gamma$-biased sampling, where observed covariates can affect the probability of sample selection arbitrarily much but the amount of unexplained variation in the probability of sample selection is bounded by a constant factor. Applying the distributionally robust optimization framework, we propose a met
    
[^247]: 通用深度神经网络的频率原理理论

    Theory of the Frequency Principle for General Deep Neural Networks

    [https://arxiv.org/abs/1906.09235](https://arxiv.org/abs/1906.09235)

    本文严格证明了通用深度神经网络在训练中从低频到高频学习的频率原理，并提供了针对不同训练阶段的理论定理，适用于多种激活函数、数据分布和损失函数。

    

    随着深度神经网络在现实问题中的广泛应用，近期一些关于DNN的实证研究报道了一个普遍现象——频率原理：在训练过程中，DNN倾向于从低频到高频学习目标函数。频率原理在提供DNN的定性和定量理解方面非常有用。本文严格研究了通用DNN在三个训练阶段（初始阶段、中间阶段和最终阶段）的频率原理。针对每个阶段，我们通过表征频率原理的适当量提供了相应定理。我们的结果具有普适性，适用于具有一般激活函数的多层网络、数据分布密度以及一大类损失函数。本研究为频率原理奠定了理论基础，有助于更好地理解训练过程。

    arXiv:1906.09235v3 Announce Type: replace  Abstract: Along with fruitful applications of Deep Neural Networks (DNNs) to realistic problems, recently, some empirical studies of DNNs reported a universal phenomenon of Frequency Principle (F-Principle): a DNN tends to learn a target function from low to high frequencies during the training. The F-Principle has been very useful in providing both qualitative and quantitative understandings of DNNs. In this paper, we rigorously investigate the F-Principle for the training dynamics of a general DNN at three stages: initial stage, intermediate stage, and final stage. For each stage, a theorem is provided in terms of proper quantities characterizing the F-Principle. Our results are general in the sense that they work for multilayer networks with general activation functions, population densities of data, and a large class of loss functions. Our work lays a theoretical foundation of the F-Principle for a better understanding of the training proc
    
[^248]: 在分布式学习任务中评估生成模型

    On the Evaluation of Generative Models in Distributed Learning Tasks. (arXiv:2310.11714v1 [cs.LG])

    [http://arxiv.org/abs/2310.11714](http://arxiv.org/abs/2310.11714)

    本文研究了在具有异构数据分布的分布式学习任务中评估生成模型。通过研究Fr\'echet inception距离（FID），并考虑不同聚合分数，发现FID-all和FID-avg分数的模型排名可能不一致。

    

    在文献中已经广泛研究了对包括生成对抗网络（GAN）和扩散模型在内的深度生成模型的评估。然而，现有的评估方法主要针对单个客户端存储的训练数据的集中式学习问题，而生成模型的许多应用涉及到分布式学习环境，例如联邦学习场景，其中训练数据由多个客户端收集并分发。本文研究了在具有异构数据分布的分布式学习任务中评估生成模型。首先，我们关注Fr\'echet inception距离（FID），并考虑以下基于FID的聚合分数：1）FID-avg作为客户端个体FID分数的平均值，2）FID-all作为训练模型与包含所有客户端数据的集体数据集之间的FID距离。我们证明了根据FID-all和FID-avg分数的模型排名可能不一致。

    The evaluation of deep generative models including generative adversarial networks (GANs) and diffusion models has been extensively studied in the literature. While the existing evaluation methods mainly target a centralized learning problem with training data stored by a single client, many applications of generative models concern distributed learning settings, e.g. the federated learning scenario, where training data are collected by and distributed among several clients. In this paper, we study the evaluation of generative models in distributed learning tasks with heterogeneous data distributions. First, we focus on the Fr\'echet inception distance (FID) and consider the following FID-based aggregate scores over the clients: 1) FID-avg as the mean of clients' individual FID scores, 2) FID-all as the FID distance of the trained model to the collective dataset containing all clients' data. We prove that the model rankings according to the FID-all and FID-avg scores could be inconsist
    
[^249]: NervePool: 一个单纯复形池化层

    NervePool: A Simplicial Pooling Layer. (arXiv:2305.06315v1 [cs.CG])

    [http://arxiv.org/abs/2305.06315](http://arxiv.org/abs/2305.06315)

    单纯复形池化层NervePool在池化图结构数据时，基于顶点分区生成单纯复形的分层表示，可以灵活地建模更高阶的关系，同时缩小高维单纯形，实现降采样，减少计算成本和减少过拟合。

    

    对于图结构数据的深度学习问题，池化层对于降采样、减少计算成本和减少过拟合都很重要。我们提出了一个池化层，NervePool，适用于单纯复形结构的数据，这种结构是图的推广，包括比顶点和边更高维度的单纯形；这种结构可以更灵活地建模更高阶的关系。所提出的单纯复合缩小方案基于顶点的分区构建，这使得我们可以生成单纯复形的分层表示，以一种学习的方式折叠信息。NervePool建立在学习的顶点群集分配的基础上，并以一种确定性的方式扩展到高维单纯形的缩小。虽然在实践中，池化操作是通过一系列矩阵运算来计算的，但是其拓扑动机是一个基于单纯形星星的并集和神经复合体的集合构造。

    For deep learning problems on graph-structured data, pooling layers are important for down sampling, reducing computational cost, and to minimize overfitting. We define a pooling layer, NervePool, for data structured as simplicial complexes, which are generalizations of graphs that include higher-dimensional simplices beyond vertices and edges; this structure allows for greater flexibility in modeling higher-order relationships. The proposed simplicial coarsening scheme is built upon partitions of vertices, which allow us to generate hierarchical representations of simplicial complexes, collapsing information in a learned fashion. NervePool builds on the learned vertex cluster assignments and extends to coarsening of higher dimensional simplices in a deterministic fashion. While in practice, the pooling operations are computed via a series of matrix operations, the topological motivation is a set-theoretic construction based on unions of stars of simplices and the nerve complex
    

