# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Test-Time Model Adaptation with Only Forward Passes](https://arxiv.org/abs/2404.01650) | 提出了一种测试时间前向适应（FOA）方法，通过无导数的协方差矩阵适应进化策略仅学习新添加的提示，以在资源有限的设备上实现模型适应。 |
| [^2] | [Efficient Algorithms for Regularized Nonnegative Scale-invariant Low-rank Approximation Models](https://arxiv.org/abs/2403.18517) | 通过研究称为均匀正则化尺度不变的更一般模型，揭示了低秩逼近模型中尺度不变性导致隐式正则化的效果，有助于更好理解正则化函数的作用并指导正则化超参数的选择。 |
| [^3] | [Masked Autoencoders are PDE Learners](https://arxiv.org/abs/2403.17728) | 掩码自动编码器在偏微分方程求解器中表现出色，通过自监督学习跨越PDEs，可以学习用于下游任务的有用潜在表示。 |
| [^4] | [Text clustering with LLM embeddings](https://arxiv.org/abs/2403.15112) | 研究表明，LLM嵌入能够捕捉结构化语言的细微差别，BERT在性能上领先于轻量级选项，增加嵌入维度和摘要技术并不一致地提高聚类效率 |
| [^5] | [Understanding Training-free Diffusion Guidance: Mechanisms and Limitations](https://arxiv.org/abs/2403.12404) | 本文旨在深入理解无需训练的扩散引导的操作机制和基本限制，提供了理论分析支持该方法，同时揭示了其更易受到对抗性梯度影响和较慢收敛的缺点。 |
| [^6] | [Introducing Adaptive Continuous Adversarial Training (ACAT) to Enhance ML Robustness](https://arxiv.org/abs/2403.10461) | 引入了自适应连续对抗训练（ACAT）来持续集成对抗训练样本到模型中，使用实际检测到的对抗数据，增强模型对不断演变的对抗威胁的抵抗能力。 |
| [^7] | [Transferable Reinforcement Learning via Generalized Occupancy Models](https://arxiv.org/abs/2403.06328) | 通过广义占有模型，本研究提出了一种新颖的模型类别，保留了模型化强化学习的通用性，并避免了累积错误的问题。 |
| [^8] | [Decoupled Data Consistency with Diffusion Purification for Image Restoration](https://arxiv.org/abs/2403.06054) | 通过分离反向过程和数据一致性步骤，提出了一种新颖的基于扩散的图像恢复求解器。 |
| [^9] | [Robust Emotion Recognition in Context Debiasing](https://arxiv.org/abs/2403.05963) | 提出了一个反事实情绪推理（CLEF）框架来解决上下文偏差干扰的挑战 |
| [^10] | [Unfamiliar Finetuning Examples Control How Language Models Hallucinate](https://arxiv.org/abs/2403.05612) | 本文研究了大型语言模型如何产生幻觉，并提出通过调整微调示例的监督来控制其对不熟悉输入的预测。作者开发了一种基于RL的方法，更可靠地减轻了长篇生成任务中的幻觉。 |
| [^11] | [Active Statistical Inference](https://arxiv.org/abs/2403.03208) | 主动推断是一种统计推断方法，通过利用机器学习模型确定最有利于标记的数据点来有效利用预算，实现比现有基线更少样本的相同准确性。 |
| [^12] | [Spatio-Temporal Field Neural Networks for Air Quality Inference](https://arxiv.org/abs/2403.02354) | 该研究提出了基于时空场神经网络的新模型和金字塔推断框架，在空气质量推断中取得了最先进的性能。 |
| [^13] | [Learning Topological Representations with Bidirectional Graph Attention Network for Solving Job Shop Scheduling Problem](https://arxiv.org/abs/2402.17606) | 本文提出了拓扑感知的双向图注意力网络（TBGAT），在解决车间作业调度问题中，通过嵌入并发图并利用双向视图嵌入、图注意力聚合等技术，实现了对拓扑结构的更好建模和利用。 |
| [^14] | [Dataset Fairness: Achievable Fairness on Your Data With Utility Guarantees](https://arxiv.org/abs/2402.17106) | 该论文提出了一种针对数据集特性量身定制的近似公平性-准确性权衡曲线计算方法，能够有效减轻训练多个模型的计算负担并提供了严格的统计保证 |
| [^15] | [Pandora's White-Box: Increased Training Data Leakage in Open LLMs](https://arxiv.org/abs/2402.17012) | 本文对开源大型语言模型（LLMs）进行了隐私攻击研究，提出了首个能同时实现高真正率和低误分类率的预训练LLMs会员推理攻击（MIAs），以及展示了在自然环境中可以从微调LLM中提取超过50%的微调数据集。 |
| [^16] | [Double-I Watermark: Protecting Model Copyright for LLM Fine-tuning](https://arxiv.org/abs/2402.14883) | 提出了一种名为“双I水印”的水印方法，通过引入两种backdoor数据范例并利用LLM的学习能力，有效地保护了LLM微调定制模型的版权。 |
| [^17] | [Rethinking Invariance Regularization in Adversarial Training to Improve Robustness-Accuracy Trade-off](https://arxiv.org/abs/2402.14648) | 重新审视了基于表示的不变性正则化方法，提出了Asymmetrically Representation-regularized Adversarial Training (AR-AT)来解决“梯度冲突”和混合分布问题，改善鲁棒性-准确性权衡。 |
| [^18] | [Linear bandits with polylogarithmic minimax regret](https://arxiv.org/abs/2402.12042) | 该研究提出了一种新的线性赌博机算法，解决了线性随机赌博机中最小极小遗憾的多对数缩放问题，通过加权最小二乘估计实现对设计矩阵特征值关系的控制，实现了累积遗憾的对数缩放。 |
| [^19] | [Disentanglement in Implicit Causal Models via Switch Variable](https://arxiv.org/abs/2402.11124) | 该论文通过软干预处理隐式潜在因果表征学习，在 Variational Autoencoder (VAE) 框架中引入了因果机制开关变量。 |
| [^20] | [Physics-based material parameters extraction from perovskite experiments via Bayesian optimization](https://arxiv.org/abs/2402.11101) | 使用贝叶斯优化开发了一个分析平台，可以从钙钛矿实验中提取多个基本材料参数，加速材料发现和半导体优化 |
| [^21] | [Zero-Shot Unsupervised and Text-Based Audio Editing Using DDPM Inversion](https://arxiv.org/abs/2402.10009) | 本文研究了使用DDPM反转进行音频信号的零样本编辑技术，包括基于文本的编辑和无监督发现编辑方向。这些方法在音乐信号中展现了多样的音乐兴趣修改。 |
| [^22] | [SLEB: Streamlining LLMs through Redundancy Verification and Elimination of Transformer Blocks](https://arxiv.org/abs/2402.09025) | SLEB是一种通过消除冗余的Transformer块来优化LLM流程的新方法，它成功加速了LLM的推理过程。 |
| [^23] | [Transition Constrained Bayesian Optimization via Markov Decision Processes](https://arxiv.org/abs/2402.08406) | 本文介绍了一种过渡受限的贝叶斯优化方法，通过马尔可夫决策过程的框架，使用强化学习解决了由于转变约束导致的搜索空间依赖历史的问题，并在化学反应器优化、信息化路径规划、机器校准等领域进行了应用。 |
| [^24] | [NeuRes: Learning Proofs of Propositional Satisfiability](https://arxiv.org/abs/2402.08365) | NeuRes是一种神经符号证明为基础的SAT解析器，能够证明不可满足性并加速找到可满足真值分配的过程。 |
| [^25] | [Nesting Particle Filters for Experimental Design in Dynamical Systems](https://arxiv.org/abs/2402.07868) | 本文提出了一种新颖的方法来解决动态系统中的贝叶斯实验设计问题，利用嵌套粒子滤波器和立体蒙特卡洛方法来进行基于梯度的策略优化，相比于其他方法具有更好的性能。 |
| [^26] | [Decoupling Learning and Decision-Making: Breaking the $\mathcal{O}(\sqrt{T})$ Barrier in Online Resource Allocation with First-Order Methods](https://arxiv.org/abs/2402.07108) | 本文研究了在线线性规划的问题，并提出了一种新的算法框架，解决了一阶方法在线算法实现超过$\mathcal{O}(\sqrt{T})$遗憾的挑战，实现了$\mathcal{O}(T^{1/3})$的遗憾。 |
| [^27] | [Entropy-Regularized Token-Level Policy Optimization for Large Language Models](https://arxiv.org/abs/2402.06700) | 本文提出了一种熵正则化的令牌级策略优化方法（ETPO），用于优化大规模语言模型（LLMs）。该方法能够通过直接与任务特定环境进行交互，并解决在如何分配令牌级学分和最大化奖励之间的冲突问题。 |
| [^28] | [Generalized Preference Optimization: A Unified Approach to Offline Alignment](https://arxiv.org/abs/2402.05749) | 广义偏好优化（GPO）是一种离线损失函数，通过参数化一类凸函数来实现统一的偏好优化视角，并提供了新的算法工具和实证洞见。 |
| [^29] | [Improving Token-Based World Models with Parallel Observation Prediction](https://arxiv.org/abs/2402.05643) | 该论文提出了一种改进基于令牌的世界模型的方法，通过引入并行观测预测机制（POP）来解决想象过程中出现的瓶颈问题。通过在一个新型TBWM代理中应用POP，想象速度提高了15.4倍，在不到12小时的训练时间内在Atari 100K基准测试中取得了超人类的表现。 |
| [^30] | [Principled Preferential Bayesian Optimization](https://arxiv.org/abs/2402.05367) | 本文提出了基于原则的优先贝叶斯优化方法，通过利用偏好反馈构建黑盒函数的置信区间，并开发了一个乐观算法来解决问题。实验证明，该方法在遗憾界限和收敛性上具有显著的性能优势。 |
| [^31] | [A Sober Look at LLMs for Material Discovery: Are They Actually Good for Bayesian Optimization Over Molecules?](https://arxiv.org/abs/2402.05015) | 本文研究了LLMs是否真的有助于加速在分子空间中的正规贝叶斯优化。通过将LLMs视为标准但正规的BO替代模型的固定特征提取器，并利用参数效能来实现。 |
| [^32] | [Generalized Sobolev Transport for Probability Measures on a Graph](https://arxiv.org/abs/2402.04516) | 我们研究了支持在图度量空间上的测度的最优传输问题，提出了一种适用于不同几何结构的图上概率测度传输方法，并引入了超力 Wassestein（OW）的概念，为某些机器学习方法的发展带来了新的机遇。 |
| [^33] | [A Data Centric Approach for Unsupervised Domain Generalization via Retrieval from Web Scale Multimodal Data](https://arxiv.org/abs/2402.04416) | 该论文基于大规模多模态数据检索，提出了一个无监督领域泛化的数据中心方法。在多模态无监督领域泛化问题中，通过构建一个小型的源数据子集，而不是依赖丰富的源数据，来解决目标标签空间数据获取困难的问题。 |
| [^34] | [More Flexible PAC-Bayesian Meta-Learning by Learning Learning Algorithms](https://arxiv.org/abs/2402.04054) | 通过学习学习算法，实现更灵活的PAC-Bayesian元学习，允许更灵活的任务之间的知识转移，提供新的泛化界限，可适用于分析和设计各种元学习机制，并在实际应用中改善了预测质量。 |
| [^35] | [Efficient Solvers for Partial Gromov-Wasserstein](https://arxiv.org/abs/2402.03664) | 本文提出了两个基于Frank-Wolfe算法的新的高效求解器来解决偏差Gromov-Wasserstein问题，并且证明了PGW问题构成了度量测度空间的度量。 |
| [^36] | [Diffusive Gibbs Sampling](https://arxiv.org/abs/2402.03008) | 扩散吉布斯采样是一种创新的采样方法，通过集成扩散模型并应用吉布斯采样，有效地从具有远程和断开模态特征的分布中采样，表现出比其他方法更好的混合性能，并在多种任务中取得显著改进的结果。 |
| [^37] | [InterpretCC: Conditional Computation for Inherently Interpretable Neural Networks](https://arxiv.org/abs/2402.02933) | InterpretCC是一种新的解释性神经网络模型，通过条件计算和稀疏激活特征，在保持性能的同时实现了人类中心的解释能力。该模型适用于需要可信解释、可操作解释和准确预测的人类面向领域。 |
| [^38] | [A Graph is Worth $K$ Words: Euclideanizing Graph using Pure Transformer](https://arxiv.org/abs/2402.02464) | 这篇论文介绍了GraphsGPT，它使用纯Transformer将非欧几里德图形转换为在欧几里德空间中可学习的图形单词，并通过解码器将图形单词重新构建为原始图形，保证了信息的等价性。预训练的GraphsGPT在图形表示学习和图形生成方面取得了突出成果。 |
| [^39] | [Beyond the Limits: A Survey of Techniques to Extend the Context Length in Large Language Models](https://arxiv.org/abs/2402.02244) | 这篇论文综述了近期为扩展大型语言模型中上下文长度而设计的技术和方法，并回顾了包括架构修改在内的多种技术，使得语言模型可以更有效地理解长上下文。 |
| [^40] | [Interpreting Graph Neural Networks with In-Distributed Proxies](https://arxiv.org/abs/2402.02036) | 该论文提出了一种翻译图神经网络中可解释子图的代理图的新方法，解决了训练数据分布与可解释子图集之间的分布偏移问题。 |
| [^41] | [Position Paper: Bayesian Deep Learning in the Age of Large-Scale AI](https://arxiv.org/abs/2402.00809) | 《在大规模人工智能时代的贝叶斯深度学习》这篇立场论文探讨了贝叶斯深度学习在各种不同设置下的优势，并指出了与之相关的挑战和有趣的研究方向。未来的研究重点将放在如何将大规模基础模型与贝叶斯深度学习相结合，以发挥它们的全部潜力。 |
| [^42] | [InstructRetro: Instruction Tuning post Retrieval-Augmented Pretraining](https://arxiv.org/abs/2310.07713) | InstructRetro是目前规模最大的使用检索预训练的LLM，扩展了基础模型Retro 48B，通过指令调优在各种零样例任务上取得显著改进。 |
| [^43] | [A Homogenization Approach for Gradient-Dominated Stochastic Optimization](https://arxiv.org/abs/2308.10630) | 本文介绍了一种基于均匀化方法的梯度主导随机优化方法，通过满足梯度主导性质的随机函数，实现全局收敛。我们提供了样本复杂度分析，并通过方差减少技术提供了增强结果。实验结果表明，该方法在无需立方正则化的情况下达到了最佳样本复杂度。 |
| [^44] | [Towards Global Glacier Mapping with Deep Learning and Open Earth Observation Data.](http://arxiv.org/abs/2401.15113) | 本研究提出了一种使用深度学习和开放地球观测数据进行全球冰川制图的方法，通过新的模型和策略，在多种地形和传感器上实现了较高的准确性。通过添加合成孔径雷达数据，并报告冰川范围的校准置信度，提高了预测的可靠性和可解释性。 |
| [^45] | [A novel hybrid time-varying graph neural network for traffic flow forecasting.](http://arxiv.org/abs/2401.10155) | 本文提出了一种新型的混合时变图神经网络（HTVGNN）用于交通流量预测，解决了现有方法中预定义图和自适应图的学习能力受限的问题。 |
| [^46] | [Code Simulation Challenges for Large Language Models.](http://arxiv.org/abs/2401.09074) | 大型语言模型在模拟计算机代码和算法执行方面遇到挑战，性能随着代码长度的增加而迅速下降。在处理短程序或标准过程时，它们能以低错误率按顺序执行指令，但对于复杂的程序，特别是包含关键路径和冗余指令的程序，模拟效果较差。我们提出了一种逐行模拟代码执行的方法来解决这个问题。 |
| [^47] | [SynHIN: Generating Synthetic Heterogeneous Information Network for Explainable AI.](http://arxiv.org/abs/2401.04133) | 该论文提出了一种生成合成异构信息网络的方法，用于可解释人工智能。该方法通过识别现实世界数据集中的模式，构建合成网络，并确保生成的合成图数据与真实数据接近。这提供了用于节点分类任务的合成异构图数据集。 |
| [^48] | [Do Concept Bottleneck Models Obey Locality?.](http://arxiv.org/abs/2401.01259) | 本文研究了概念瓶颈模型（CBMs）是否能够正确捕捉到概念之间的条件独立程度，通过分析对于概念局部性之外特征的变化如何影响概念的预测。 |
| [^49] | [Automated Model Selection for Tabular Data.](http://arxiv.org/abs/2401.00961) | 本文介绍了一种自动化模型选择算法，用于表格数据的预测。该算法考虑了特征之间的交互，并包含了基于优先级的随机网格搜索和贪婪搜索两种不同的特征选择方法。 |
| [^50] | [FaultFormer: Pretraining Transformers for Adaptable Bearing Fault Classification.](http://arxiv.org/abs/2312.02380) | 本论文提出了一个使用预训练Transformer模型进行适应性轴承故障分类的框架。通过研究不同的标记分割和数据增强策略，该方法在稀缺数据环境中能够达到最先进的准确率，并在微调时改善了性能。 |
| [^51] | [Large Catapults in Momentum Gradient Descent with Warmup: An Empirical Study.](http://arxiv.org/abs/2311.15051) | 本研究通过实验证明，带有大学习率和学习率预热的动量梯度下降显示出大型弹射效应，将迭代朝着比梯度下降发现的更平缓的极小值方向推进。 |
| [^52] | [MIST: Defending Against Membership Inference Attacks Through Membership-Invariant Subspace Training.](http://arxiv.org/abs/2311.00919) | 通过成员不变子空间训练的MIST算法有效防御成员推理攻击，能够识别容易受到攻击的实例并避免过度拟合。 |
| [^53] | [A Wireless AI-Generated Content (AIGC) Provisioning Framework Empowered by Semantic Communication.](http://arxiv.org/abs/2310.17705) | 一种由语义通信增强的无线AI生成内容（AIGC）供应框架，通过使用语义信息而不是所有的二进制位提取和传输内容，以解决在无线网络中提供最优AIGC服务的挑战。 |
| [^54] | [Finite Time Analysis of Constrained Actor Critic and Constrained Natural Actor Critic Algorithms.](http://arxiv.org/abs/2310.16363) | 本文研究了受约束的Actor Critic和受约束的Natural Actor Critic算法的有限时间分析，证明了这些算法能找到性能函数的一阶稳定点，并且具有较低的样本复杂度。 |
| [^55] | [Model-agnostic variable importance for predictive uncertainty: an entropy-based approach.](http://arxiv.org/abs/2310.12842) | 本文提出了一种基于熵的方法，通过扩展现有的解释性方法，可以理解不确定性感知模型中的预测来源和置信度，并利用改编后的特征重要性、部分依赖图和个体条件期望图等方法来测量特征对预测分布的熵和基于真实标签的对数似然的影响。 |
| [^56] | [ACES: generating diverse programming puzzles with autotelic language models and semantic descriptors.](http://arxiv.org/abs/2310.10692) | ACES是一种使用自我目标语言模型和语义描述符生成多样化的编程难题的方法，能够优化有趣的多样性和少样本生成。 |
| [^57] | [The Mixtures and the Neural Critics: On the Pointwise Mutual Information Profiles of Fine Distributions.](http://arxiv.org/abs/2310.10240) | 本文研究了点间互信息的特征，引入了细分布家族来解决现有互信息估计器的局限性，并探究了神经批评家在变分估计器中的行为，以及实验异常值对互信息估计的影响。此外，还介绍了基于模型的贝叶斯估计的方法，适用于具有领域专业知识且需要不确定性量化的问题。 |
| [^58] | [ParFam -- Symbolic Regression Based on Continuous Global Optimization.](http://arxiv.org/abs/2310.05537) | ParFam是一种新的符号回归方法，利用参数化的符号函数族将离散问题转化为连续问题，并结合全局优化器，能够有效解决符号回归问题。 |
| [^59] | [On the Error-Propagation of Inexact Deflation for Principal Component Analysis.](http://arxiv.org/abs/2310.04283) | 该论文研究了主成分分析中不精确消除法的误差传播问题，给出了两个主要结果 |
| [^60] | [DeepHGCN: Toward Deeper Hyperbolic Graph Convolutional Networks.](http://arxiv.org/abs/2310.02027) | DeepHGCN是一个具有深层架构的双曲图卷积网络，通过引入新的双曲特征转换层和正则化技术，实现了计算效率的极大改进和过度平滑问题的显著减轻。 |
| [^61] | [On the Disconnect Between Theory and Practice of Overparametrized Neural Networks.](http://arxiv.org/abs/2310.00137) | 本文研究了神经网络在无穷宽度极限下的行为，并与核方法建立了联系。虽然在合成架构中展示了一些优势，如更快的优化和可靠的不确定性量化，但实际相关的架构需要比深度大很多倍的宽度才能实现这些优势。 |
| [^62] | [Comprehensive Analysis of Network Robustness Evaluation Based on Convolutional Neural Networks with Spatial Pyramid Pooling.](http://arxiv.org/abs/2308.08012) | 本文通过设计具有空间金字塔池化网络的卷积神经网络模型，解决了网络稳健性评估中的性能、捕捉稳健性、可扩展性和可转移性等挑战。 |
| [^63] | [GPLaSDI: Gaussian Process-based Interpretable Latent Space Dynamics Identification through Deep Autoencoder.](http://arxiv.org/abs/2308.05882) | GPLaSDI是一种基于高斯过程的可解释潜空间动力学识别方法，通过深度自动编码器将完全阶数的PDE解映射到潜空间，并使用插值和解决ODE系统进行快速和准确的ROM预测。 |
| [^64] | [Fairness-aware Federated Minimax Optimization with Convergence Guarantee.](http://arxiv.org/abs/2307.04417) | 本文提出了一种名为FFALM的算法，通过施加公平约束和解决极小化极大回归问题，在联邦学习中解决了群体公平性问题。实验证明FFALM在处理严重统计异质性问题时具有良好的效果。 |
| [^65] | [Shifting Attention to Relevance: Towards the Uncertainty Estimation of Large Language Models.](http://arxiv.org/abs/2307.01379) | 本论文研究了大型语言模型（LLMs）自动生成的关键词不平等问题，发现在估计不确定性时，重要的令牌和含有有限语义的句子被同等或更加重视。为了解决这个问题，提出了共同转移关注点来更好地估计不确定性。 |
| [^66] | [Learning Any-View 6DoF Robotic Grasping in Cluttered Scenes via Neural Surface Rendering.](http://arxiv.org/abs/2306.07392) | 通过神经表面渲染，NeuGraspNet能够在混乱场景中有效地从任意视角预测6DoF抓取质量，并能够在遮挡的场景中采样抓取候选项。 |
| [^67] | [Stochastic Collapse: How Gradient Noise Attracts SGD Dynamics Towards Simpler Subnetworks.](http://arxiv.org/abs/2306.04251) | SGD在训练过度表达的网络时，会随机地将动态吸引到更简单的子网络，这种随机吸引性能够提高泛化能力。 |
| [^68] | [Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio synthesis.](http://arxiv.org/abs/2306.00814) | Vocos是一个新的模型，通过直接生成傅里叶谱系数，消除了时域和基于傅里叶变化的神经声码器在高质量音频合成中的差距，并实现了计算效率的大幅提升。 |
| [^69] | [Insights from the Design Space Exploration of Flow-Guided Nanoscale Localization.](http://arxiv.org/abs/2305.18493) | 研究了基于流导向纳米定位的设计空间，考虑了能源和信号衰减等因素，为这一新兴领域提供了有希望的解决方案。 |
| [^70] | [Laplace-Approximated Neural Additive Models: Improving Interpretability with Bayesian Inference.](http://arxiv.org/abs/2305.16905) | 本文提出了拉普拉斯逼近神经加性模型，该模型从贝叶斯角度考虑加性结构，在恢复的特征交互中提供可信区间，提供可处理的边缘似然估计，可用于执行隐式特征选择并对特征对进行排名。 |
| [^71] | [Think Before You Act: Decision Transformers with Internal Working Memory.](http://arxiv.org/abs/2305.16338) | 该论文提出了具有内部工作记忆模块的决策Transformer方法，以解决使用大型语言模型的决策代理在处理新任务上性能低下的问题。所提出的方法改善了训练效率和泛化能力，并进一步增强了转化决策制定代理对新任务的适应性。 |
| [^72] | [Non-Log-Concave and Nonsmooth Sampling via Langevin Monte Carlo Algorithms.](http://arxiv.org/abs/2305.15988) | 本文研究了从非对数凸分布进行近似抽样的问题，并通过 Langevin Monte Carlo 算法解决。此外，研究了两种非光滑情况，这些任务源于贝叶斯推断和图像反问题。数值模拟比较了最常用的 Langevin Monte Carlo 算法的性能。 |
| [^73] | [UP5: Unbiased Foundation Model for Fairness-aware Recommendation.](http://arxiv.org/abs/2305.12090) | 本研究提出了一种新颖的基础模型UP5，它采用反事实公平促进技术来消除大型语言模型中的偏见，从而实现面向公平性的推荐。 |
| [^74] | [Provably Correct Physics-Informed Neural Networks.](http://arxiv.org/abs/2305.10157) | 该论文提出了一种名为$\partial$-CROWN的框架，以保证物理知识神经网络（PINN）具有全局正确性的最坏剩余误差，并证明了该框架在获得有效证书方面的有效性。 |
| [^75] | [Expressive Text-to-Image Generation with Rich Text.](http://arxiv.org/abs/2304.06720) | 本文提出了一种使用富文本编辑器生成表达性文本图像的方法，可以通过局部样式控制、明确的标记重新加权、精确的颜色渲染和详细的区域合成，生成高质量且多样化的图像。 |
| [^76] | [SoftED: Metrics for Soft Evaluation of Time Series Event Detection.](http://arxiv.org/abs/2304.00439) | SoftED metrics 是一种适用于时间序列事件检测的新指标，既包括时间的概念，又包括对相邻检测的时间容忍度，它们能够同时评估事件检测的准确性和其检测是否代表事件。 |
| [^77] | [Random Inverse Problems Over Graphs: Decentralized Online Learning.](http://arxiv.org/abs/2303.11789) | 本文提出了一种基于在线数据流的分布式在线学习算法，将希尔伯特空间中的分布参数估计和再生核希尔伯特空间中的最小均方问题统一起来，并发展了一种新的L2-渐近稳定性理论。该算法在网络图为连通且正向算子序列满足无限维度时空励磁条件的情况下，能够实现均方和几乎必然的强一致估计。 |
| [^78] | [Subset-Based Instance Optimality in Private Estimation.](http://arxiv.org/abs/2303.01262) | 本论文提出了一个新的定义来评估差分隐私估计算法的实例优化。我们的定义要求最优算法与一个最佳的已知数据集并在其较大的子集上进行性能评估相竞争，从而使基准算法比以前的工作更强大。我们还展示了在实值数据集上如何构建能够实现实例优化的隐私算法，并对均值进行了详细分析，证明我们的算法在估计一类广泛的数据集属性时能达到或超过渐近性能。 |
| [^79] | [PARIS: Personalized Activity Recommendation for Improving Sleep Quality.](http://arxiv.org/abs/2110.13745) | 该论文利用机器学习技术，结合可穿戴设备监测的数据，通过时间序列聚类找到与指定主题相关的行为模型并生成相应的睡眠质量活动建议，为提高睡眠质量提供了一种个性化解决方案。 |
| [^80] | [MRCpy: A Library for Minimax Risk Classifiers.](http://arxiv.org/abs/2108.01952) | MRCpy是一种用于实现最小化风险分类器的Python库，它基于鲁棒风险最小化技术，可以利用0-1损失并提供了多种分类方法，其中一些提供了紧密的期望损失界限。 |

# 详细

[^1]: 仅使用前向传播的测试时间模型适应

    Test-Time Model Adaptation with Only Forward Passes

    [https://arxiv.org/abs/2404.01650](https://arxiv.org/abs/2404.01650)

    提出了一种测试时间前向适应（FOA）方法，通过无导数的协方差矩阵适应进化策略仅学习新添加的提示，以在资源有限的设备上实现模型适应。

    

    测试时间适应已被证明在适应给定训练模型到具有潜在分布转移的未见测试样本时是有效的。然而，在现实场景中，模型通常部署在资源有限的设备上，例如FPGA，并且通常被量化和硬编码为不可修改的参数以加速。鉴于此，由于现有方法严重依赖于计算密集型的反向传播进行模型更新，因此这些方法通常是不可行的。为了解决这个问题，我们提出了一种测试时间前向适应（FOA）方法。 在FOA中，我们试图通过一个无导数的协方差矩阵适应进化策略来仅学习新添加的提示（作为模型的输入）。 为了使这种策略在我们的在线无监督设置下稳定工作，我们设计了一种通过衡量测试训练统计差异和模型预测熵的新颖适应函数。此外，我们设计了一种激活移位方案。

    arXiv:2404.01650v1 Announce Type: new  Abstract: Test-time adaptation has proven effective in adapting a given trained model to unseen test samples with potential distribution shifts. However, in real-world scenarios, models are usually deployed on resource-limited devices, e.g., FPGAs, and are often quantized and hard-coded with non-modifiable parameters for acceleration. In light of this, existing methods are often infeasible since they heavily depend on computation-intensive backpropagation for model updating that may be not supported. To address this, we propose a test-time Forward-Only Adaptation (FOA) method. In FOA, we seek to solely learn a newly added prompt (as model's input) via a derivative-free covariance matrix adaptation evolution strategy. To make this strategy work stably under our online unsupervised setting, we devise a novel fitness function by measuring test-training statistic discrepancy and model prediction entropy. Moreover, we design an activation shifting sche
    
[^2]: 针对正则化非负尺度不变低秩逼近模型的高效算法

    Efficient Algorithms for Regularized Nonnegative Scale-invariant Low-rank Approximation Models

    [https://arxiv.org/abs/2403.18517](https://arxiv.org/abs/2403.18517)

    通过研究称为均匀正则化尺度不变的更一般模型，揭示了低秩逼近模型中尺度不变性导致隐式正则化的效果，有助于更好理解正则化函数的作用并指导正则化超参数的选择。

    

    正则化非负低秩逼近，如稀疏的非负矩阵分解或稀疏的非负Tucker分解，是具有增强可解释性的降维模型中的一个重要分支。然而，从实践角度来看，由于这些模型的多因素特性以及缺乏支持这些选择的理论，正则化函数和正则化系数的选择，以及高效算法的设计仍然具有挑战性。本文旨在改进这些问题。通过研究一个称为均匀正则化尺度不变的更一般模型，我们证明低秩逼近模型中固有的尺度不变性导致了隐式正则化，具有意想不到的有益和有害效果。这一发现使我们能够更好地理解低秩逼近模型中正则化函数的作用，指导正则化超参数的选择。

    arXiv:2403.18517v1 Announce Type: new  Abstract: Regularized nonnegative low-rank approximations such as sparse Nonnegative Matrix Factorization or sparse Nonnegative Tucker Decomposition are an important branch of dimensionality reduction models with enhanced interpretability. However, from a practical perspective, the choice of regularizers and regularization coefficients, as well as the design of efficient algorithms, is challenging because of the multifactor nature of these models and the lack of theory to back these choices. This paper aims at improving upon these issues. By studying a more general model called the Homogeneous Regularized Scale-Invariant, we prove that the scale-invariance inherent to low-rank approximation models causes an implicit regularization with both unexpected beneficial and detrimental effects. This observation allows to better understand the effect of regularization functions in low-rank approximation models, to guide the choice of the regularization hyp
    
[^3]: 掩码自动编码器是PDE学习者

    Masked Autoencoders are PDE Learners

    [https://arxiv.org/abs/2403.17728](https://arxiv.org/abs/2403.17728)

    掩码自动编码器在偏微分方程求解器中表现出色，通过自监督学习跨越PDEs，可以学习用于下游任务的有用潜在表示。

    

    神经求解器用于偏微分方程（PDE）具有巨大潜力，但实用性目前受到其泛化能力的限制。 PDE在广泛的尺度上演变并展示出多样化的行为；预测这些现象将需要学习跨越各种输入的表示，这些输入可能涵盖不同的系数、几何图形或方程。作为通向可泛化PDE建模的一步，我们为PDEs调整了掩码预训练。通过自监督学习跨越PDEs，掩码自动编码器可以学习有用的潜在表示，以用于下游任务。特别是，掩码预训练可以改善神经求解器对未见方程的系数回归和时间步骤性能。我们希望掩码预训练能成为一种通用方法，可以在大型、未标记和异构数据集上学习规模化的潜在物理学。

    arXiv:2403.17728v1 Announce Type: new  Abstract: Neural solvers for partial differential equations (PDEs) have great potential, yet their practicality is currently limited by their generalizability. PDEs evolve over broad scales and exhibit diverse behaviors; predicting these phenomena will require learning representations across a wide variety of inputs, which may encompass different coefficients, geometries, or equations. As a step towards generalizable PDE modeling, we adapt masked pretraining for PDEs. Through self-supervised learning across PDEs, masked autoencoders can learn useful latent representations for downstream tasks. In particular, masked pretraining can improve coefficient regression and timestepping performance of neural solvers on unseen equations. We hope that masked pretraining can emerge as a unifying method across large, unlabeled, and heterogeneous datasets to learn latent physics at scale.
    
[^4]: 使用LLM嵌入进行文本聚类

    Text clustering with LLM embeddings

    [https://arxiv.org/abs/2403.15112](https://arxiv.org/abs/2403.15112)

    研究表明，LLM嵌入能够捕捉结构化语言的细微差别，BERT在性能上领先于轻量级选项，增加嵌入维度和摘要技术并不一致地提高聚类效率

    

    文本聚类是组织不断增长的数字内容的重要方法，有助于结构化和发现未分类数据中的隐藏模式。在这项研究中，我们调查了不同文本嵌入（特别是大型语言模型LLMs中使用的）和聚类算法如何影响文本数据集的聚类方式。进行了一系列实验以评估嵌入是如何影响聚类结果的，以及通过摘要进行降维和嵌入大小调整的作用。结果显示，LLM嵌入在捕获结构化语言的细微差别方面表现出色，而BERT在性能上领先于轻量级选项。此外，我们发现增加嵌入维度和摘要技术并不一致地提高聚类效率，这表明这些策略需要仔细分析才能在实际模型中使用。这些结果突出了一种

    arXiv:2403.15112v1 Announce Type: cross  Abstract: Text clustering is an important approach for organising the growing amount of digital content, helping to structure and find hidden patterns in uncategorised data. In this research, we investigated how different textual embeddings - particularly those used in large language models (LLMs) - and clustering algorithms affect how text datasets are clustered. A series of experiments were conducted to assess how embeddings influence clustering results, the role played by dimensionality reduction through summarisation, and embedding size adjustment. Results reveal that LLM embeddings excel at capturing the nuances of structured language, while BERT leads the lightweight options in performance. In addition, we find that increasing embedding dimensionality and summarisation techniques do not uniformly improve clustering efficiency, suggesting that these strategies require careful analysis to use in real-life models. These results highlight a co
    
[^5]: 理解无需训练的扩散引导：机制与局限性

    Understanding Training-free Diffusion Guidance: Mechanisms and Limitations

    [https://arxiv.org/abs/2403.12404](https://arxiv.org/abs/2403.12404)

    本文旨在深入理解无需训练的扩散引导的操作机制和基本限制，提供了理论分析支持该方法，同时揭示了其更易受到对抗性梯度影响和较慢收敛的缺点。

    

    向预先训练的扩散模型添加额外控制已成为越来越流行的研究领域，在计算机视觉、强化学习和科学人工智能等领域有广泛应用。最近，一些研究提出使用在干净图像上预训练的现成网络进行无需训练的扩散引导。这种方法实现了零样本条件生成，适用于通用控制格式，看起来提供了无需训练的扩散引导中的免费午餐。本文旨在对无需训练的引导的运行机制和基本限制进行更深入的理解。我们提供了一项支持无需训练引导的理论分析，从优化的角度区分了它与基于分类器的（或者无分类器的）引导。为了阐明它们的缺点，我们在理论上证明了无需训练方法更容易受到对抗性梯度的影响，并表现出更慢的收敛速度。

    arXiv:2403.12404v1 Announce Type: new  Abstract: Adding additional control to pretrained diffusion models has become an increasingly popular research area, with extensive applications in computer vision, reinforcement learning, and AI for science. Recently, several studies have proposed training-free diffusion guidance by using off-the-shelf networks pretrained on clean images. This approach enables zero-shot conditional generation for universal control formats, which appears to offer a free lunch in diffusion guidance. In this paper, we aim to develop a deeper understanding of the operational mechanisms and fundamental limitations of training-free guidance. We offer a theoretical analysis that supports training-free guidance from the perspective of optimization, distinguishing it from classifier-based (or classifier-free) guidance. To elucidate their drawbacks, we theoretically demonstrate that training-free methods are more susceptible to adversarial gradients and exhibit slower conv
    
[^6]: 引入自适应连续对抗训练（ACAT）以增强机器学习的鲁棒性

    Introducing Adaptive Continuous Adversarial Training (ACAT) to Enhance ML Robustness

    [https://arxiv.org/abs/2403.10461](https://arxiv.org/abs/2403.10461)

    引入了自适应连续对抗训练（ACAT）来持续集成对抗训练样本到模型中，使用实际检测到的对抗数据，增强模型对不断演变的对抗威胁的抵抗能力。

    

    机器学习（ML）易受针对ML模型的对抗攻击影响，这些攻击旨在欺骗ML模型，使其产生错误预测。 对抗训练被发现能提高ML模型对这些攻击的鲁棒性。然而，在网络和网络安全领域，获取标记训练和对抗训练数据是具有挑战性且昂贵的。此外，概念漂移加深了挑战，特别是在诸如网络和网络安全等动态领域中，需要各种模型进行定期重新训练。本文介绍了自适应连续对抗训练（ACAT），以在持续的学习会话期间持续将对抗训练样本整合到模型中，使用实际检测到的对抗数据，以增强模型对不断演变的对抗威胁的抵抗能力。 ACAT是一种自适应的防御机制，利用定期重新训练来有效对抗对抗攻击，同时减轻灾难性后果。

    arXiv:2403.10461v1 Announce Type: new  Abstract: Machine Learning (ML) is susceptible to adversarial attacks that aim to trick ML models, making them produce faulty predictions. Adversarial training was found to increase the robustness of ML models against these attacks. However, in network and cybersecurity, obtaining labeled training and adversarial training data is challenging and costly. Furthermore, concept drift deepens the challenge, particularly in dynamic domains like network and cybersecurity, and requires various models to conduct periodic retraining. This letter introduces Adaptive Continuous Adversarial Training (ACAT) to continuously integrate adversarial training samples into the model during ongoing learning sessions, using real-world detected adversarial data, to enhance model resilience against evolving adversarial threats. ACAT is an adaptive defense mechanism that utilizes periodic retraining to effectively counter adversarial attacks while mitigating catastrophic f
    
[^7]: 通过广义占有模型实现可迁移的强化学习

    Transferable Reinforcement Learning via Generalized Occupancy Models

    [https://arxiv.org/abs/2403.06328](https://arxiv.org/abs/2403.06328)

    通过广义占有模型，本研究提出了一种新颖的模型类别，保留了模型化强化学习的通用性，并避免了累积错误的问题。

    

    智能代理必须是通用的 - 具有快速适应和概括到不同任务的能力。在强化学习（RL）框架内，基于模型的RL算法学习世界的任务不可知动态模型，原则上使它们能够概括到任意奖励。然而，一步模型自然会受到累积错误的影响，使它们在具有长时间跨度和大状态空间的问题上失效。在这项工作中，我们提出了一类新型模型 - 广义占有模型（GOMs），保留了基于模型的RL的通用性，同时避免了累积性错误。GOMs的关键思想是在一个固定数据集的覆盖下，建模给定状态的所有可能长期结果的分布，以及实现给定状态的特定结果的策略。然后，这些模型可以迅速用于为任意新任务选择最优操作，而无需担心累积错误。

    arXiv:2403.06328v1 Announce Type: new  Abstract: Intelligent agents must be generalists - showing the ability to quickly adapt and generalize to varying tasks. Within the framework of reinforcement learning (RL), model-based RL algorithms learn a task-agnostic dynamics model of the world, in principle allowing them to generalize to arbitrary rewards. However, one-step models naturally suffer from compounding errors, making them ineffective for problems with long horizons and large state spaces. In this work, we propose a novel class of models - generalized occupancy models (GOMs) - that retain the generality of model-based RL while avoiding compounding error. The key idea behind GOMs is to model the distribution of all possible long-term outcomes from a given state under the coverage of a stationary dataset, along with a policy that realizes a particular outcome from the given state. These models can then quickly be used to select the optimal action for arbitrary new tasks, without hav
    
[^8]: 具有扩散净化的分离数据一致性的图像恢复

    Decoupled Data Consistency with Diffusion Purification for Image Restoration

    [https://arxiv.org/abs/2403.06054](https://arxiv.org/abs/2403.06054)

    通过分离反向过程和数据一致性步骤，提出了一种新颖的基于扩散的图像恢复求解器。

    

    最近，扩散模型作为一种强大的深度生成先验类别已经引起了人们的关注，由于其出色地建模数据分布的能力，在各种图像恢复任务中表现出色。为了解决图像恢复问题，许多现有技术通过将额外的似然梯度步骤纳入到扩散模型的反向采样过程中来实现数据一致性。然而，这些额外的梯度步骤对于实际应用中存在挑战，因为它们造成了巨大的计算开销，从而增加了推理时间。当使用加速的扩散模型采样器时，这些额外的步骤还会导致额外的困难，因为数据一致性步骤的数量受限于反向采样步骤的数量。在这项工作中，我们提出了一种新颖的基于扩散的图像恢复求解器，通过将反向过程与数据一致性步骤分离来解决这些问题。我们的方法涉及

    arXiv:2403.06054v1 Announce Type: cross  Abstract: Diffusion models have recently gained traction as a powerful class of deep generative priors, excelling in a wide range of image restoration tasks due to their exceptional ability to model data distributions. To solve image restoration problems, many existing techniques achieve data consistency by incorporating additional likelihood gradient steps into the reverse sampling process of diffusion models. However, the additional gradient steps pose a challenge for real-world practical applications as they incur a large computational overhead, thereby increasing inference time. They also present additional difficulties when using accelerated diffusion model samplers, as the number of data consistency steps is limited by the number of reverse sampling steps. In this work, we propose a novel diffusion-based image restoration solver that addresses these issues by decoupling the reverse process from the data consistency steps. Our method involv
    
[^9]: 在上下文去偏的情绪识别中的鲁棒性

    Robust Emotion Recognition in Context Debiasing

    [https://arxiv.org/abs/2403.05963](https://arxiv.org/abs/2403.05963)

    提出了一个反事实情绪推理（CLEF）框架来解决上下文偏差干扰的挑战

    

    上下文感知情绪识别（CAER）最近在无约束环境中推动了情感计算技术的实际应用。 主流的CAER方法总是从不同的上下文和以主体为中心的特征中提取集成表示，以感知目标人物的情绪状态。 尽管有所进展，但最大的挑战仍然是由于上下文偏差的干扰。 有害的偏见迫使模型依赖于背景上下文和情感标签之间的虚假相关性，在可能性估计中造成严重的性能瓶颈，并使有价值的上下文先验混淆。 在本文中，我们提出了一个反事实情绪推理（CLEF）框架来解决上述问题。 具体而言，我们首先制定了一个广义因果图，以解耦CAER中变量之间的因果关系。 遵循因果图，CLEF引入了一个非侵入式的上下文分支来获取

    arXiv:2403.05963v1 Announce Type: cross  Abstract: Context-aware emotion recognition (CAER) has recently boosted the practical applications of affective computing techniques in unconstrained environments. Mainstream CAER methods invariably extract ensemble representations from diverse contexts and subject-centred characteristics to perceive the target person's emotional state. Despite advancements, the biggest challenge remains due to context bias interference. The harmful bias forces the models to rely on spurious correlations between background contexts and emotion labels in likelihood estimation, causing severe performance bottlenecks and confounding valuable context priors. In this paper, we propose a counterfactual emotion inference (CLEF) framework to address the above issue. Specifically, we first formulate a generalized causal graph to decouple the causal relationships among the variables in CAER. Following the causal graph, CLEF introduces a non-invasive context branch to capt
    
[^10]: 不熟悉的微调示例控制语言模型如何产生幻觉

    Unfamiliar Finetuning Examples Control How Language Models Hallucinate

    [https://arxiv.org/abs/2403.05612](https://arxiv.org/abs/2403.05612)

    本文研究了大型语言模型如何产生幻觉，并提出通过调整微调示例的监督来控制其对不熟悉输入的预测。作者开发了一种基于RL的方法，更可靠地减轻了长篇生成任务中的幻觉。

    

    大型语言模型（LLMs）倾向于生成听起来令人信服但事实不正确的响应，特别是当在不熟悉的概念上进行查询时。本文探讨了调整后的LLMs如何产生幻觉的基本机制。我们的调查揭示了一个有趣的模式：随着输入变得更不熟悉，LLMs的输出倾向于默认为"含糊其词"的预测，其形式受微调数据中不熟悉示例监督方式的影响。因此，通过策略性地修改这些示例的监督，我们可以控制LLM对不熟悉输入的预测（例如，教会它们说“我不知道”）。基于这些原则，我们开发了一种RL方法，通过解决奖励模型幻觉带来的挑战，更可靠地减轻长篇生成任务的幻觉。我们通过在MMLU上的多选QA中进行一系列受控实验来验证我们的发现。

    arXiv:2403.05612v1 Announce Type: cross  Abstract: Large language models (LLMs) have a tendency to generate plausible-sounding yet factually incorrect responses, especially when queried on unfamiliar concepts. In this work, we explore the underlying mechanisms that govern how finetuned LLMs hallucinate. Our investigation reveals an interesting pattern: as inputs become more unfamiliar, LLM outputs tend to default towards a ``hedged'' prediction, whose form is determined by how the unfamiliar examples in the finetuning data are supervised. Thus, by strategically modifying these examples' supervision, we can control LLM predictions for unfamiliar inputs (e.g., teach them to say ``I don't know''). Based on these principles, we develop an RL approach that more reliably mitigates hallucinations for long-form generation tasks, by tackling the challenges presented by reward model hallucinations. We validate our findings with a series of controlled experiments in multiple-choice QA on MMLU, as
    
[^11]: 主动统计推断

    Active Statistical Inference

    [https://arxiv.org/abs/2403.03208](https://arxiv.org/abs/2403.03208)

    主动推断是一种统计推断方法，通过利用机器学习模型确定最有利于标记的数据点来有效利用预算，实现比现有基线更少样本的相同准确性。

    

    受主动学习概念启发，我们提出了主动推断——一种利用机器学习辅助数据收集进行统计推断的方法。假设对可收集的标签数量有预算限制，该方法利用机器学习模型确定哪些数据点最有利于标记，从而有效利用预算。其运作方式基于一种简单而强大的直觉：优先收集模型表现出不确定性的数据点的标签，并在模型表现出自信时依赖于其预测。主动推断构建了可证明有效的置信区间和假设检验，同时利用任何黑盒机器学习模型并处理任何数据分布。关键点在于，它能以比依赖于非自适应收集数据的现有基线更少的样本达到相同水平的准确性。这意味着对于相同数量的样本，...

    arXiv:2403.03208v1 Announce Type: cross  Abstract: Inspired by the concept of active learning, we propose active inference$\unicode{x2013}$a methodology for statistical inference with machine-learning-assisted data collection. Assuming a budget on the number of labels that can be collected, the methodology uses a machine learning model to identify which data points would be most beneficial to label, thus effectively utilizing the budget. It operates on a simple yet powerful intuition: prioritize the collection of labels for data points where the model exhibits uncertainty, and rely on the model's predictions where it is confident. Active inference constructs provably valid confidence intervals and hypothesis tests while leveraging any black-box machine learning model and handling any data distribution. The key point is that it achieves the same level of accuracy with far fewer samples than existing baselines relying on non-adaptively-collected data. This means that for the same number 
    
[^12]: 基于时空场神经网络的空气质量推断

    Spatio-Temporal Field Neural Networks for Air Quality Inference

    [https://arxiv.org/abs/2403.02354](https://arxiv.org/abs/2403.02354)

    该研究提出了基于时空场神经网络的新模型和金字塔推断框架，在空气质量推断中取得了最先进的性能。

    

    空气质量推断问题旨在利用来自有限观测站的历史数据推断未知位置的空气质量指数。考虑到观测站高昂的维护成本导致数据稀疏性，良好的推断算法可以有效节约成本并细化数据粒度。尽管时空图神经网络在这个问题上取得了显著进展，但它们对现实的非欧几里得和离散数据结构建模限制了潜力。本文首次尝试通过提出一个新模型，即时空场神经网络，及其对应的新框架，金字塔推断，将两种不同的时空观点，场和图，相结合。大量实验证实我们的模型在中国大陆全国范围内的空气质量推断中实现了最新技术水平，展示了我们提出的模型的优越性。

    arXiv:2403.02354v1 Announce Type: cross  Abstract: The air quality inference problem aims to utilize historical data from a limited number of observation sites to infer the air quality index at an unknown location. Considering the sparsity of data due to the high maintenance cost of the stations, good inference algorithms can effectively save the cost and refine the data granularity. While spatio-temporal graph neural networks have made excellent progress on this problem, their non-Euclidean and discrete data structure modeling of reality limits its potential. In this work, we make the first attempt to combine two different spatio-temporal perspectives, fields and graphs, by proposing a new model, Spatio-Temporal Field Neural Network, and its corresponding new framework, Pyramidal Inference. Extensive experiments validate that our model achieves state-of-the-art performance in nationwide air quality inference in the Chinese Mainland, demonstrating the superiority of our proposed model 
    
[^13]: 使用双向图注意力网络学习拓扑表示解决车间作业调度问题

    Learning Topological Representations with Bidirectional Graph Attention Network for Solving Job Shop Scheduling Problem

    [https://arxiv.org/abs/2402.17606](https://arxiv.org/abs/2402.17606)

    本文提出了拓扑感知的双向图注意力网络（TBGAT），在解决车间作业调度问题中，通过嵌入并发图并利用双向视图嵌入、图注意力聚合等技术，实现了对拓扑结构的更好建模和利用。

    

    现有的基于学习的方法通常使用针对无向图的现成GNN模型解决车间作业调度问题（JSSP），并忽略了并发图（DGs）的丰富而有意义的拓扑结构。本文提出了拓扑感知的双向图注意力网络（TBGAT），这是一种基于注意力机制的新颖GNN架构，用于在本地搜索框架中嵌入DG以解决JSSP。具体而言，TBGAT分别从正向和反向视图嵌入DG，消息通过遵循不同视图的拓扑结构传播，并通过图注意力进行汇总。然后，我们提出一种基于消息传递机制的新操作符，用于计算DG的前向和后向拓扑排序，这些特征用于表征拓扑结构并被我们的模型利用。此外，我们从理论和实验上展示了TBGAT的...

    arXiv:2402.17606v1 Announce Type: cross  Abstract: Existing learning-based methods for solving job shop scheduling problem (JSSP) usually use off-the-shelf GNN models tailored to undirected graphs and neglect the rich and meaningful topological structures of disjunctive graphs (DGs). This paper proposes the topology-aware bidirectional graph attention network (TBGAT), a novel GNN architecture based on the attention mechanism, to embed the DG for solving JSSP in a local search framework. Specifically, TBGAT embeds the DG from a forward and a backward view, respectively, where the messages are propagated by following the different topologies of the views and aggregated via graph attention. Then, we propose a novel operator based on the message-passing mechanism to calculate the forward and backward topological sorts of the DG, which are the features for characterizing the topological structures and exploited by our model. In addition, we theoretically and experimentally show that TBGAT h
    
[^14]: 数据集公平性：在您的数据上实现具有效用保证的公平性

    Dataset Fairness: Achievable Fairness on Your Data With Utility Guarantees

    [https://arxiv.org/abs/2402.17106](https://arxiv.org/abs/2402.17106)

    该论文提出了一种针对数据集特性量身定制的近似公平性-准确性权衡曲线计算方法，能够有效减轻训练多个模型的计算负担并提供了严格的统计保证

    

    在机器学习公平性中，训练能够最小化不同敏感群体之间差异的模型通常会导致准确性下降，这种现象被称为公平性-准确性权衡。这种权衡的严重程度基本取决于数据集的特性，如数据集的不均衡或偏见。因此，在数据集之间使用统一的公平性要求仍然值得怀疑，并且往往会导致效用极低的模型。为了解决这个问题，我们提出了一种针对单个数据集量身定制的近似公平性-准确性权衡曲线的计算效率高的方法，该方法支持严格的统计保证。通过利用You-Only-Train-Once（YOTO）框架，我们的方法减轻了在逼近权衡曲线时需要训练多个模型的计算负担。此外，我们通过在该曲线周围引入置信区间来量化我们近似值的不确定性，

    arXiv:2402.17106v1 Announce Type: cross  Abstract: In machine learning fairness, training models which minimize disparity across different sensitive groups often leads to diminished accuracy, a phenomenon known as the fairness-accuracy trade-off. The severity of this trade-off fundamentally depends on dataset characteristics such as dataset imbalances or biases. Therefore using a uniform fairness requirement across datasets remains questionable and can often lead to models with substantially low utility. To address this, we present a computationally efficient approach to approximate the fairness-accuracy trade-off curve tailored to individual datasets, backed by rigorous statistical guarantees. By utilizing the You-Only-Train-Once (YOTO) framework, our approach mitigates the computational burden of having to train multiple models when approximating the trade-off curve. Moreover, we quantify the uncertainty in our approximation by introducing confidence intervals around this curve, offe
    
[^15]: Pandora's White-Box：开放LLMs中训练数据泄漏的增加

    Pandora's White-Box: Increased Training Data Leakage in Open LLMs

    [https://arxiv.org/abs/2402.17012](https://arxiv.org/abs/2402.17012)

    本文对开源大型语言模型（LLMs）进行了隐私攻击研究，提出了首个能同时实现高真正率和低误分类率的预训练LLMs会员推理攻击（MIAs），以及展示了在自然环境中可以从微调LLM中提取超过50%的微调数据集。

    

    在本文中，我们对开源的大型语言模型（LLMs）遭受的隐私攻击进行了系统研究，其中对手可以访问模型权重、梯度或损失，试图利用它们来了解底层训练数据。我们的主要结果是针对预训练LLMs的第一个会员推理攻击（MIAs），能够同时实现高TPR和低FPR，并展示了在自然环境中可以从微调LLM中提取超过50%的微调数据集。我们考虑了对底层模型的不同访问程度、语言模型的定制化以及攻击者可以使用的资源。在预训练设置中，我们提出了三种新的白盒MIAs：基于梯度范数的攻击、监督神经网络分类器和单步损失比攻击。所有这些都优于现有的黑盒基线，并且我们的.....

    arXiv:2402.17012v1 Announce Type: cross  Abstract: In this paper we undertake a systematic study of privacy attacks against open source Large Language Models (LLMs), where an adversary has access to either the model weights, gradients, or losses, and tries to exploit them to learn something about the underlying training data. Our headline results are the first membership inference attacks (MIAs) against pre-trained LLMs that are able to simultaneously achieve high TPRs and low FPRs, and a pipeline showing that over $50\%$ (!) of the fine-tuning dataset can be extracted from a fine-tuned LLM in natural settings. We consider varying degrees of access to the underlying model, customization of the language model, and resources available to the attacker. In the pre-trained setting, we propose three new white-box MIAs: an attack based on the gradient norm, a supervised neural network classifier, and a single step loss ratio attack. All outperform existing black-box baselines, and our supervi
    
[^16]: 双I水印：保护LLM微调模型版权

    Double-I Watermark: Protecting Model Copyright for LLM Fine-tuning

    [https://arxiv.org/abs/2402.14883](https://arxiv.org/abs/2402.14883)

    提出了一种名为“双I水印”的水印方法，通过引入两种backdoor数据范例并利用LLM的学习能力，有效地保护了LLM微调定制模型的版权。

    

    为了支持各种应用，业主经常通过LLM所有者或云服务器提供的API对预训练的LLM进行微调，以获取定制模型。然而，这一过程存在着模型被滥用的风险，可能会给业主带来严重的经济后果。因此，在LLM微调过程中保护这些定制模型的版权已成为紧迫的实际需求，但现有的解决方案有限。为了解决这一紧迫问题，我们提出了一种名为“双I水印”的新型水印方法。具体地，基于指导微调数据，引入了两种backdoor数据范例，分别在指令和输入中触发。通过利用LLM的学习能力将定制的后门样本纳入数据集，所提出的方法有效地注入了特定的水印。

    arXiv:2402.14883v1 Announce Type: cross  Abstract: To support various applications, business owners often seek the customized models that are obtained by fine-tuning a pre-trained LLM through the API provided by LLM owners or cloud servers. However, this process carries a substantial risk of model misuse, potentially resulting in severe economic consequences for business owners. Thus, safeguarding the copyright of these customized models during LLM fine-tuning has become an urgent practical requirement, but there are limited existing solutions to provide such protection. To tackle this pressing issue, we propose a novel watermarking approach named "Double-I watermark". Specifically, based on the instruct-tuning data, two types of backdoor data paradigms are introduced with trigger in the instruction and the input, respectively. By leveraging LLM's learning capability to incorporate customized backdoor samples into the dataset, the proposed approach effectively injects specific watermar
    
[^17]: 在对抗训练中重新思考不变性正则化以改善鲁棒性-准确性权衡

    Rethinking Invariance Regularization in Adversarial Training to Improve Robustness-Accuracy Trade-off

    [https://arxiv.org/abs/2402.14648](https://arxiv.org/abs/2402.14648)

    重新审视了基于表示的不变性正则化方法，提出了Asymmetrically Representation-regularized Adversarial Training (AR-AT)来解决“梯度冲突”和混合分布问题，改善鲁棒性-准确性权衡。

    

    尽管对抗训练一直是抵抗对抗性样本（AEs）的最先进方法，但它们存在鲁棒性-准确性权衡问题。在这项研究中，我们重新审视基于表示的不变性正则化，学习具有辨别性却对抗性不变的表示，旨在缓解这种权衡。我们在经验上确定了妨碍不变性正则化的两个关键问题：（1）不变性损失和分类目标之间的“梯度冲突”，表明存在“崩溃解”，以及（2）由于干净和对抗性输入的分布发散而出现的混合分布问题。为了解决这些问题，我们提出了一种不对称表示正则化的对抗训练（AR-AT），该方法结合了一个停止梯度操作和一个预测器来避免“崩溃解”，灵感来自最近的非对比自监督学习。

    arXiv:2402.14648v1 Announce Type: cross  Abstract: Although adversarial training has been the state-of-the-art approach to defend against adversarial examples (AEs), they suffer from a robustness-accuracy trade-off. In this work, we revisit representation-based invariance regularization to learn discriminative yet adversarially invariant representations, aiming to mitigate this trade-off. We empirically identify two key issues hindering invariance regularization: (1) a "gradient conflict" between invariance loss and classification objectives, indicating the existence of "collapsing solutions," and (2) the mixture distribution problem arising from diverged distributions of clean and adversarial inputs. To address these issues, we propose Asymmetrically Representation-regularized Adversarial Training (AR-AT), which incorporates a stop-gradient operation and a pre-dictor in the invariance loss to avoid "collapsing solutions," inspired by a recent non-contrastive self-supervised learning a
    
[^18]: 具有多对数极小极小遗憾的线性赌博机

    Linear bandits with polylogarithmic minimax regret

    [https://arxiv.org/abs/2402.12042](https://arxiv.org/abs/2402.12042)

    该研究提出了一种新的线性赌博机算法，解决了线性随机赌博机中最小极小遗憾的多对数缩放问题，通过加权最小二乘估计实现对设计矩阵特征值关系的控制，实现了累积遗憾的对数缩放。

    

    我们研究了一种线性随机赌博机的噪声模型，对于该模型，当我们选择越来越接近未知向量的单位球上的动作时，亚高斯噪声参数以线性方式消失。我们针对这个问题引入了一种算法，其在时间长度$T$的情况下呈对数$^3（T）$的最小遗憾缩放，与典型赌博机算法的平方根遗憾缩放形成鲜明对比。我们的策略基于加权最小二乘估计，通过几何论证实现了设计矩阵$V_t$在每个时间步骤$t$处的特征值关系$\lambda_{\min} ( V_t ) = \Omega (\sqrt{\lambda_{\max}(V_t ) })$，这些几何论证与噪声模型无关，并可能具有独立的兴趣。这使我们能够严格控制每个时间步骤的期望遗憾为$O(\frac1{t})$的数量级，从而导致累积遗憾的对数缩放。

    arXiv:2402.12042v1 Announce Type: cross  Abstract: We study a noise model for linear stochastic bandits for which the subgaussian noise parameter vanishes linearly as we select actions on the unit sphere closer and closer to the unknown vector. We introduce an algorithm for this problem that exhibits a minimax regret scaling as $\log^3(T)$ in the time horizon $T$, in stark contrast the square root scaling of this regret for typical bandit algorithms. Our strategy, based on weighted least-squares estimation, achieves the eigenvalue relation $\lambda_{\min} ( V_t ) = \Omega (\sqrt{\lambda_{\max}(V_t ) })$ for the design matrix $V_t$ at each time step $t$ through geometrical arguments that are independent of the noise model and might be of independent interest. This allows us to tightly control the expected regret in each time step to be of the order $O(\frac1{t})$, leading to the logarithmic scaling of the cumulative regret.
    
[^19]: 通过开关变量在隐式因果模型中解开纠缠

    Disentanglement in Implicit Causal Models via Switch Variable

    [https://arxiv.org/abs/2402.11124](https://arxiv.org/abs/2402.11124)

    该论文通过软干预处理隐式潜在因果表征学习，在 Variational Autoencoder (VAE) 框架中引入了因果机制开关变量。

    

    从观测和干预数据中学习因果表征，在没有已知的地面真实图结构的情况下，需要隐式潜在因果表征学习。隐式学习因果机制通常涉及两类干预数据：硬干预和软干预。在现实世界场景中，软干预通常比硬干预更现实，因为后者需要完全受控的环境。与直接强制改变因果变量的硬干预不同，软干预通过影响因果机制间接地产生影响。本文通过软干预在变分自动编码器（VAE）框架中处理隐式潜在因果表征学习。我们的方法通过使用一个旨在在不同因果机制之间切换的因果机制开关变量来建模软干预效果。在我们的实验中，我们始终保持

    arXiv:2402.11124v1 Announce Type: new  Abstract: Learning causal representations from observational and interventional data in the absence of known ground-truth graph structures necessitates implicit latent causal representation learning. Implicitly learning causal mechanisms typically involves two categories of interventional data: hard and soft interventions. In real-world scenarios, soft interventions are often more realistic than hard interventions, as the latter require fully controlled environments. Unlike hard interventions, which directly force changes in a causal variable, soft interventions exert influence indirectly by affecting the causal mechanism. In this paper, we tackle implicit latent causal representation learning in a Variational Autoencoder (VAE) framework through soft interventions. Our approach models soft interventions effects by employing a causal mechanism switch variable designed to toggle between different causal mechanisms. In our experiments, we consistentl
    
[^20]: 通过贝叶斯优化从钙钛矿实验中提取基于物理的材料参数

    Physics-based material parameters extraction from perovskite experiments via Bayesian optimization

    [https://arxiv.org/abs/2402.11101](https://arxiv.org/abs/2402.11101)

    使用贝叶斯优化开发了一个分析平台，可以从钙钛矿实验中提取多个基本材料参数，加速材料发现和半导体优化

    

    从定量实验分析中提取材料参数的能力对于合理设计和理论进步至关重要。然而，随着理论模型的复杂性和材料参数数量的增加，这种分析的难度显着增加。在这里，我们使用贝叶斯优化开发了一个分析平台，可以从瞬态光致发光实验中提取一个有机金属钙钛矿半导体的8个基本材料参数，基于一个包括载流子漂移扩散和动态缺陷占据的复杂全物理模型。热降解的一个示例研究表明，掺杂浓度和载流子迁移率的变化主导，而缺陷能级几乎保持不变。这个平台可以方便地应用于其他实验或实验组合，加速材料发现和半导体优化。

    arXiv:2402.11101v1 Announce Type: cross  Abstract: The ability to extract material parameters from quantitative experimental analysis is essential for rational design and theory advancement. However, the difficulty of this analysis increases significantly with the complexity of the theoretical model and the number of material parameters. Here we use Bayesian optimization to develop an analysis platform that can extract up to 8 fundamental material parameters of an organometallic perovskite semiconductor from a transient photoluminescence experiment, based on a complex full physics model that includes drift-diffusion of carriers and dynamic defect occupation. An example study of thermal degradation reveals that changes in doping concentration and carrier mobility dominate, while the defect energy level remains nearly unchanged. This platform can be conveniently applied to other experiments or to combinations of experiments, accelerating materials discovery and optimization of semiconduc
    
[^21]: 使用DDPM反转进行零样本无监督和基于文本的音频编辑

    Zero-Shot Unsupervised and Text-Based Audio Editing Using DDPM Inversion

    [https://arxiv.org/abs/2402.10009](https://arxiv.org/abs/2402.10009)

    本文研究了使用DDPM反转进行音频信号的零样本编辑技术，包括基于文本的编辑和无监督发现编辑方向。这些方法在音乐信号中展现了多样的音乐兴趣修改。

    

    使用大型预训练模型进行零样本编辑已经在图像领域取得了迅猛的发展，但在音频领域尚未出现。本文中，我们探索了两种基于DDPM反转的音频信号零样本编辑技术。第一种是从图像领域采用的方法，允许基于文本进行编辑。第二种是一种新颖的方法，可以在无监督情况下发现语义上有意义的编辑方向。当应用于音乐信号时，这种方法可以展现出一系列具有音乐兴趣的修改，从控制特定乐器的参与到对旋律进行即兴演奏。示例可以在我们的例子页面中找到：https://hilamanor.github.io/AudioEditing/ ，代码可以在 https://github.com/hilamanor/AudioEditing/ 找到。

    arXiv:2402.10009v1 Announce Type: cross  Abstract: Editing signals using large pre-trained models, in a zero-shot manner, has recently seen rapid advancements in the image domain. However, this wave has yet to reach the audio domain. In this paper, we explore two zero-shot editing techniques for audio signals, which use DDPM inversion on pre-trained diffusion models. The first, adopted from the image domain, allows text-based editing. The second, is a novel approach for discovering semantically meaningful editing directions without supervision. When applied to music signals, this method exposes a range of musically interesting modifications, from controlling the participation of specific instruments to improvisations on the melody. Samples can be found on our examples page in https://hilamanor.github.io/AudioEditing/ and code can be found in https://github.com/hilamanor/AudioEditing/ .
    
[^22]: SLEB: 通过冗余验证和消除Transformer块优化LLM的流程

    SLEB: Streamlining LLMs through Redundancy Verification and Elimination of Transformer Blocks

    [https://arxiv.org/abs/2402.09025](https://arxiv.org/abs/2402.09025)

    SLEB是一种通过消除冗余的Transformer块来优化LLM流程的新方法，它成功加速了LLM的推理过程。

    

    大型语言模型（LLM）在各种自然语言处理任务中证明了其高效性。然而，它们庞大的参数数量给实际部署带来了重大挑战。精简，一种旨在减小LLM大小和复杂度的技术，通过从网络中删除冗余组件提供了潜在解决方案。尽管精简有希望，但现有方法往往难以实现显著的端到端LLM推理加速。本文中，我们引入了SLEB，一种通过消除冗余的Transformer块来优化LLM流程的新方法。我们选择Transformer块作为精简的基本单位，因为LLM在相邻块的输出之间具有块级别的冗余和高相似性。这个选择使我们能够有效地增强LLM的处理速度。我们的实验证明，SLEB成功加速了LLM的推理过程。

    arXiv:2402.09025v1 Announce Type: new Abstract: Large language models (LLMs) have proven to be highly effective across various natural language processing tasks. However, their large number of parameters poses significant challenges for practical deployment. Pruning, a technique aimed at reducing the size and complexity of LLMs, offers a potential solution by removing redundant components from the network. Despite the promise of pruning, existing methods often struggle to achieve substantial end-to-end LLM inference speedup. In this paper, we introduce SLEB, a novel approach designed to streamline LLMs by eliminating redundant transformer blocks. We choose the transformer block as the fundamental unit for pruning, because LLMs exhibit block-level redundancy with high similarity between the outputs of neighboring blocks. This choice allows us to effectively enhance the processing speed of LLMs. Our experimental results demonstrate that SLEB successfully accelerates LLM inference without
    
[^23]: 过渡受限的贝叶斯优化在马尔可夫决策过程中的应用

    Transition Constrained Bayesian Optimization via Markov Decision Processes

    [https://arxiv.org/abs/2402.08406](https://arxiv.org/abs/2402.08406)

    本文介绍了一种过渡受限的贝叶斯优化方法，通过马尔可夫决策过程的框架，使用强化学习解决了由于转变约束导致的搜索空间依赖历史的问题，并在化学反应器优化、信息化路径规划、机器校准等领域进行了应用。

    

    贝叶斯优化是一种优化黑盒函数的方法。传统上，它关注的是可以任意查询搜索空间的情况。然而，许多现实生活中的问题并不具备这种灵活性；特别是，下一个查询的搜索空间可能取决于先前的查询。物理科学领域的例子中存在一些挑战，如局部移动限制、特定变量的单调性要求以及转变影响测量精度。总之，这些过渡约束需要一种规划方法。本文通过马尔可夫决策过程的框架扩展了贝叶斯优化，通过强化学习迭代地解决我们目标的一个可行线性化，从而获得能够提前规划长时间跨度的策略。得到的策略可能是依赖历史的和非马尔可夫的。我们展示了在化学反应器优化、信息化路径规划、机器校准等方面的应用。

    Bayesian optimization is a methodology to optimize black-box functions. Traditionally, it focuses on the setting where you can arbitrarily query the search space. However, many real-life problems do not offer this flexibility; in particular, the search space of the next query may depend on previous ones. Example challenges arise in the physical sciences in the form of local movement constraints, required monotonicity in certain variables, and transitions influencing the accuracy of measurements. Altogether, such transition constraints necessitate a form of planning. This work extends Bayesian optimization via the framework of Markov Decision Processes, iteratively solving a tractable linearization of our objective using reinforcement learning to obtain a policy that plans ahead over long horizons. The resulting policy is potentially history-dependent and non-Markovian. We showcase applications in chemical reactor optimization, informative path planning, machine calibration, and other s
    
[^24]: NeuRes: 学习命题可满足性的证明

    NeuRes: Learning Proofs of Propositional Satisfiability

    [https://arxiv.org/abs/2402.08365](https://arxiv.org/abs/2402.08365)

    NeuRes是一种神经符号证明为基础的SAT解析器，能够证明不可满足性并加速找到可满足真值分配的过程。

    

    我们介绍了一种神经符号证明为基础的SAT解析器NeuRes。与其他神经SAT解算法不同，NeuRes能够证明不可满足性，而不仅仅是预测它。NeuRes通过采用命题推理来证明不可满足性并加速在不可满足和可满足公式中找到满足真值分配的过程。为了实现这一点，我们提出了一种新颖的架构，它结合了图神经网络和指针网络的元素，从动态图结构中自动选择节点对，这对于生成解析证明是至关重要的。我们使用与NeuroSAT相同的随机公式分布编制了一个包含教师证明和真值分配的数据集，对我们的模型进行训练和评估。在实验证明中，我们展示了NeuRes在不同分布上比NeuroSAT解决更多的测试公式，并且需要更少的数据。

    We introduce NeuRes, a neuro-symbolic proof-based SAT solver. Unlike other neural SAT solving methods, NeuRes is capable of proving unsatisfiability as opposed to merely predicting it. By design, NeuRes operates in a certificate-driven fashion by employing propositional resolution to prove unsatisfiability and to accelerate the process of finding satisfying truth assignments in case of unsat and sat formulas, respectively. To realize this, we propose a novel architecture that adapts elements from Graph Neural Networks and Pointer Networks to autoregressively select pairs of nodes from a dynamic graph structure, which is essential to the generation of resolution proofs. Our model is trained and evaluated on a dataset of teacher proofs and truth assignments that we compiled with the same random formula distribution used by NeuroSAT. In our experiments, we show that NeuRes solves more test formulas than NeuroSAT by a rather wide margin on different distributions while being much more data
    
[^25]: 动态系统中的实验设计的嵌套粒子滤波器

    Nesting Particle Filters for Experimental Design in Dynamical Systems

    [https://arxiv.org/abs/2402.07868](https://arxiv.org/abs/2402.07868)

    本文提出了一种新颖的方法来解决动态系统中的贝叶斯实验设计问题，利用嵌套粒子滤波器和立体蒙特卡洛方法来进行基于梯度的策略优化，相比于其他方法具有更好的性能。

    

    本文提出了一种新颖的贝叶斯实验设计方法，用于非交换数据，并将其形式化为风险敏感的策略优化。我们开发了内外SMC^2算法，使用嵌套顺序蒙特卡洛（SMC）估计器来预测期望的信息增益，并将其嵌入到粒子马尔可夫链蒙特卡洛（pMCMC）框架中进行基于梯度的策略优化。与最近依赖于偏估计器来摊销先前学习设计策略的成本的方法相比，我们的方法具有更好的性能。在一组动态系统的数值验证中展示了我们方法的有效性。

    In this paper, we propose a novel approach to Bayesian Experimental Design (BED) for non-exchangeable data that formulates it as risk-sensitive policy optimization. We develop the Inside-Out SMC^2 algorithm that uses a nested sequential Monte Carlo (SMC) estimator of the expected information gain and embeds it into a particle Markov chain Monte Carlo (pMCMC) framework to perform gradient-based policy optimization. This is in contrast to recent approaches that rely on biased estimators of the expected information gain (EIG) to amortize the cost of experiments by learning a design policy in advance. Numerical validation on a set of dynamical systems showcases the efficacy of our method in comparison to other state-of-the-art strategies.
    
[^26]: 解耦学习和决策：用一阶方法突破在线资源分配中的$\mathcal{O}(\sqrt{T})$障碍

    Decoupling Learning and Decision-Making: Breaking the $\mathcal{O}(\sqrt{T})$ Barrier in Online Resource Allocation with First-Order Methods

    [https://arxiv.org/abs/2402.07108](https://arxiv.org/abs/2402.07108)

    本文研究了在线线性规划的问题，并提出了一种新的算法框架，解决了一阶方法在线算法实现超过$\mathcal{O}(\sqrt{T})$遗憾的挑战，实现了$\mathcal{O}(T^{1/3})$的遗憾。

    

    在线线性规划在收益管理和资源分配之间起着重要作用，最近的研究集中在开发有效的一阶在线学习算法。尽管一阶方法在实证上取得了成功，但它们通常只能实现$\mathcal{O}(\sqrt{T})$的遗憾，与最先进的基于线性规划(LP)的在线算法所保证的$\mathcal{O}(\log T)$界限相比是次优的。本文确定了关于在线线性规划的几个重要事实，揭示了一阶方法在线算法实现超过$\mathcal{O}(\sqrt{T})$遗憾的挑战。为了解决这个挑战，我们引入了一个新的算法框架，将学习与决策分离。更重要的是，我们首次展示了一阶方法在这个新框架下可以达到$\mathcal{O}(T^{1/3})$的遗憾。最后，我们进行了数值实验，验证了我们的理论发现。

    Online linear programming plays an important role in both revenue management and resource allocation, and recent research has focused on developing efficient first-order online learning algorithms. Despite the empirical success of first-order methods, they typically achieve a regret no better than $\mathcal{O}(\sqrt{T})$, which is suboptimal compared to the $\mathcal{O}(\log T)$ bound guaranteed by the state-of-the-art linear programming (LP)-based online algorithms. This paper establishes several important facts about online linear programming, which unveils the challenge for first-order-method-based online algorithms to achieve beyond $\mathcal{O}(\sqrt{T})$ regret. To address the challenge, we introduce a new algorithmic framework that decouples learning from decision-making. More importantly, for the first time, we show that first-order methods can attain regret $\mathcal{O}(T^{1/3})$ with this new framework. Lastly, we conduct numerical experiments to validate our theoretical find
    
[^27]: 熵正则化的令牌级策略优化用于大规模语言模型

    Entropy-Regularized Token-Level Policy Optimization for Large Language Models

    [https://arxiv.org/abs/2402.06700](https://arxiv.org/abs/2402.06700)

    本文提出了一种熵正则化的令牌级策略优化方法（ETPO），用于优化大规模语言模型（LLMs）。该方法能够通过直接与任务特定环境进行交互，并解决在如何分配令牌级学分和最大化奖励之间的冲突问题。

    

    大规模语言模型（LLMs）在交互式决策任务中表现出了智能代理的潜力。传统方法通常依赖于精心设计的提示、高质量的示例或额外的奖励模型进行上下文学习、监督微调或RLHF。强化学习（RL）提供了一种动态的解决方案，使LLMs能够通过直接与任务特定环境进行交互来克服这些依赖关系。尽管如此，它面临着重重困难：1）由于巨大的动作空间需要探索而产生的不稳定性；2）基于动作级奖励信号分配令牌级学分的挑战，导致最大化奖励和准确建模语料库数据之间的冲突。为了应对这些挑战，我们引入了熵正则化的令牌级策略优化（ETPO），这是一种专为在令牌级优化LLMs而设计的熵增强强化学习方法。ETPO的核心是我们的一种新颖的逐令牌软Bellman更新算法，

    Large Language Models (LLMs) have shown promise as intelligent agents in interactive decision-making tasks. Traditional approaches often depend on meticulously designed prompts, high-quality examples, or additional reward models for in-context learning, supervised fine-tuning, or RLHF. Reinforcement learning (RL) presents a dynamic alternative for LLMs to overcome these dependencies by engaging directly with task-specific environments. Nonetheless, it faces significant hurdles: 1) instability stemming from the exponentially vast action space requiring exploration; 2) challenges in assigning token-level credit based on action-level reward signals, resulting in discord between maximizing rewards and accurately modeling corpus data. In response to these challenges, we introduce Entropy-Regularized Token-level Policy Optimization (ETPO), an entropy-augmented RL method tailored for optimizing LLMs at the token level. At the heart of ETPO is our novel per-token soft Bellman update, designed 
    
[^28]: 广义偏好优化：离线对齐的统一方法

    Generalized Preference Optimization: A Unified Approach to Offline Alignment

    [https://arxiv.org/abs/2402.05749](https://arxiv.org/abs/2402.05749)

    广义偏好优化（GPO）是一种离线损失函数，通过参数化一类凸函数来实现统一的偏好优化视角，并提供了新的算法工具和实证洞见。

    

    离线偏好优化允许直接从离线数据中对大型模型进行微调，并在最近的对齐实践中证明了其有效性。我们提出了广义偏好优化（GPO），这是一类通过一般的凸函数参数化的离线损失函数。GPO提供了对偏好优化的统一视角，涵盖了现有算法（DPO、IPO和SLiC）作为特殊情况，同时自然引入了新的变体。GPO框架还揭示了离线算法如何通过定义损失的凸函数来实施正则化。我们的分析和实验揭示了离线正则化和规范的RLHF公式所意图的KL散度正则化之间的联系和微妙差异。总的来说，我们的结果为对齐实践者提供了新的算法工具和实证洞见。

    Offline preference optimization allows fine-tuning large models directly from offline data, and has proved effective in recent alignment practices. We propose generalized preference optimization (GPO), a family of offline losses parameterized by a general class of convex functions. GPO enables a unified view over preference optimization, encompassing existing algorithms such as DPO, IPO and SLiC as special cases, while naturally introducing new variants. The GPO framework also sheds light on how offline algorithms enforce regularization, through the design of the convex function that defines the loss. Our analysis and experiments reveal the connections and subtle differences between the offline regularization and the KL divergence regularization intended by the canonical RLHF formulation. In all, our results present new algorithmic toolkits and empirical insights to alignment practitioners.
    
[^29]: 通过并行观测预测改进基于令牌的世界模型

    Improving Token-Based World Models with Parallel Observation Prediction

    [https://arxiv.org/abs/2402.05643](https://arxiv.org/abs/2402.05643)

    该论文提出了一种改进基于令牌的世界模型的方法，通过引入并行观测预测机制（POP）来解决想象过程中出现的瓶颈问题。通过在一个新型TBWM代理中应用POP，想象速度提高了15.4倍，在不到12小时的训练时间内在Atari 100K基准测试中取得了超人类的表现。

    

    受到将Transformer应用于离散符号序列的成功启发，最近提出了基于令牌的世界模型（TBWMs）作为高效样本方法。在TBWMs中，世界模型将代理经验作为一种类似语言的令牌序列进行消耗，其中每个观测构成一个子序列。然而，在想象过程中，通过令牌逐个生成下一个观测的串行方式导致了严重的瓶颈问题，导致训练时间长、GPU利用率低和表示能力有限。为了解决这个瓶颈问题，我们设计了一种新颖的并行观测预测（POP）机制。POP通过一种针对我们的强化学习环境设计的新型前向模式来扩充了保持网络（RetNet）。我们将POP集成到一种名为REM（保持环境模型）的新型TBWM代理中，展示了比以前的TBWMs快15.4倍的想象能力。REM在Atari 100K基准测试的26个游戏中的12个游戏中达到超越人类水平的性能，并且在不到12小时的训练时间内完成训练。

    Motivated by the success of Transformers when applied to sequences of discrete symbols, token-based world models (TBWMs) were recently proposed as sample-efficient methods. In TBWMs, the world model consumes agent experience as a language-like sequence of tokens, where each observation constitutes a sub-sequence. However, during imagination, the sequential token-by-token generation of next observations results in a severe bottleneck, leading to long training times, poor GPU utilization, and limited representations. To resolve this bottleneck, we devise a novel Parallel Observation Prediction (POP) mechanism. POP augments a Retentive Network (RetNet) with a novel forward mode tailored to our reinforcement learning setting. We incorporate POP in a novel TBWM agent named REM (Retentive Environment Model), showcasing a 15.4x faster imagination compared to prior TBWMs. REM attains superhuman performance on 12 out of 26 games of the Atari 100K benchmark, while training in less than 12 hours.
    
[^30]: 基于原则的优先贝叶斯优化

    Principled Preferential Bayesian Optimization

    [https://arxiv.org/abs/2402.05367](https://arxiv.org/abs/2402.05367)

    本文提出了基于原则的优先贝叶斯优化方法，通过利用偏好反馈构建黑盒函数的置信区间，并开发了一个乐观算法来解决问题。实验证明，该方法在遗憾界限和收敛性上具有显著的性能优势。

    

    本文研究了优先贝叶斯优化（BO）问题，其中我们希望仅凭偏好反馈来优化黑盒函数的一对候选解。受到似然比思想的启发，我们使用仅凭偏好反馈构建黑盒函数的置信区间。然后，我们开发了一种乐观算法和高效的计算方法来解决这个问题，它在累积遗憾上具有信息论的界限，这在优先BO中是首次。这个界限进一步允许我们设计一个方案来报告最佳解的估计值，并保证收敛速率。从高斯过程、标准测试函数和一个热舒适度优化问题的实验结果都表明，我们的方法相对于现有的最先进的启发式算法来说，稳定地达到更好或有竞争力的性能。而现有启发式算法在遗憾界限或收敛性上没有理论保证。

    We study the problem of preferential Bayesian optimization (BO), where we aim to optimize a black-box function with only preference feedback over a pair of candidate solutions. Inspired by the likelihood ratio idea, we construct a confidence set of the black-box function using only the preference feedback. An optimistic algorithm with an efficient computational method is then developed to solve the problem, which enjoys an information-theoretic bound on the cumulative regret, a first-of-its-kind for preferential BO. This bound further allows us to design a scheme to report an estimated best solution, with a guaranteed convergence rate. Experimental results on sampled instances from Gaussian processes, standard test functions, and a thermal comfort optimization problem all show that our method stably achieves better or competitive performance as compared to the existing state-of-the-art heuristics, which, however, do not have theoretical guarantees on regret bounds or convergence.
    
[^31]: 对于材料发现来说，对LLM的拜占庭优化是否真的有利？一个冷静的观察

    A Sober Look at LLMs for Material Discovery: Are They Actually Good for Bayesian Optimization Over Molecules?

    [https://arxiv.org/abs/2402.05015](https://arxiv.org/abs/2402.05015)

    本文研究了LLMs是否真的有助于加速在分子空间中的正规贝叶斯优化。通过将LLMs视为标准但正规的BO替代模型的固定特征提取器，并利用参数效能来实现。

    

    自动化是当代材料发现的重要基石。贝叶斯优化是这种工作流程的重要组成部分，它使科学家能够将先前的领域知识应用到对大规模分子空间的高效探索中。尽管这样的先前知识可以采用多种形式，但关于大型语言模型（LLMs）中所包含的辅助科学知识已经引起了很大的轰动。然而，迄今为止的研究仅探索了基于启发式材料搜索的LLMs。实际上，最近的研究通过从点估计的非贝叶斯LLMs中获得不确定性估计，这是BO的一个重要组成部分。在这项工作中，我们研究了LLMs是否真的有助于加速在分子空间中的正规贝叶斯优化。我们对这个问题采取了冷静、客观的立场。这是通过仔细地（i）将LLMs视为标准但正规的BO替代模型的固定特征提取器，以及（ii）利用参数效能来实现的。

    Automation is one of the cornerstones of contemporary material discovery. Bayesian optimization (BO) is an essential part of such workflows, enabling scientists to leverage prior domain knowledge into efficient exploration of a large molecular space. While such prior knowledge can take many forms, there has been significant fanfare around the ancillary scientific knowledge encapsulated in large language models (LLMs). However, existing work thus far has only explored LLMs for heuristic materials searches. Indeed, recent work obtains the uncertainty estimate -- an integral part of BO -- from point-estimated, non-Bayesian LLMs. In this work, we study the question of whether LLMs are actually useful to accelerate principled Bayesian optimization in the molecular space. We take a sober, dispassionate stance in answering this question. This is done by carefully (i) viewing LLMs as fixed feature extractors for standard but principled BO surrogate models and by (ii) leveraging parameter-effic
    
[^32]: 图上概率测度的广义 Sobolev 传输

    Generalized Sobolev Transport for Probability Measures on a Graph

    [https://arxiv.org/abs/2402.04516](https://arxiv.org/abs/2402.04516)

    我们研究了支持在图度量空间上的测度的最优传输问题，提出了一种适用于不同几何结构的图上概率测度传输方法，并引入了超力 Wassestein（OW）的概念，为某些机器学习方法的发展带来了新的机遇。

    

    我们研究了支持在图度量空间上的测度的最优传输（OT）问题。最近，Le 等人（2022）利用图结构提出了一种 OT 的变体，称为 Sobolev 传输（ST），它提供了一种闭式表达式用于快速计算。然而，ST 的定义中实质上与 $L^p$ 几何结构耦合在一起，这使得在其他先验结构中利用 ST 变得非常困难。相反，经典的 OT 通过修改底层成本函数具有适应各种几何结构的灵活性。一个重要的例子是超力 Wassestein（OW），它通过利用\emph{Orlicz 几何结构}超越了 $L^p$ 结构。与使用标准 $p$-阶 Wassestein 相比，OW 显著提高了某些机器学习方法的性能。然而，由于其两层优化 formulation，OW 在其计算上带来了新的挑战。在这项工作中，我们利用了一类特定的凸函数。

    We study the optimal transport (OT) problem for measures supported on a graph metric space. Recently, Le et al. (2022) leverage the graph structure and propose a variant of OT, namely Sobolev transport (ST), which yields a closed-form expression for a fast computation. However, ST is essentially coupled with the $L^p$ geometric structure within its definition which makes it nontrivial to utilize ST for other prior structures. In contrast, the classic OT has the flexibility to adapt to various geometric structures by modifying the underlying cost function. An important instance is the Orlicz-Wasserstein (OW) which moves beyond the $L^p$ structure by leveraging the \emph{Orlicz geometric structure}. Comparing to the usage of standard $p$-order Wasserstein, OW remarkably helps to advance certain machine learning approaches. Nevertheless, OW brings up a new challenge on its computation due to its two-level optimization formulation. In this work, we leverage a specific class of convex funct
    
[^33]: 基于大规模多模态数据检索的无监督领域泛化的数据中心方法

    A Data Centric Approach for Unsupervised Domain Generalization via Retrieval from Web Scale Multimodal Data

    [https://arxiv.org/abs/2402.04416](https://arxiv.org/abs/2402.04416)

    该论文基于大规模多模态数据检索，提出了一个无监督领域泛化的数据中心方法。在多模态无监督领域泛化问题中，通过构建一个小型的源数据子集，而不是依赖丰富的源数据，来解决目标标签空间数据获取困难的问题。

    

    领域泛化(DG)是一个重要的问题，它通过利用一个或多个源领域在共享标签空间的假设下学习一个能够推广到未见测试领域的模型。然而，大多数DG方法假设可以访问丰富的目标标签空间中的源数据，这个要求在许多现实应用中太过严格，因为获取与目标任务相同的标签空间费用高昂。为了解决这个问题，我们处理了无监督领域泛化(UDG)问题的多模态版本，该问题使用一个大型的任务无关的未标记的源数据集，例如LAION-2B在微调期间。我们的框架不显式地假设源数据集与目标任务之间存在任何关系。相反，它只依赖于源数据集可以在联合视觉-语言空间中高效搜索的前提。针对这种多模态UDG设置，我们提出了一种新的方法来构建一个小型（小于100K）的源数据子集。

    Domain generalization (DG) is an important problem that learns a model that can generalize to unseen test domains leveraging one or more source domains, under the assumption of shared label spaces. However, most DG methods assume access to abundant source data in the target label space, a requirement that proves overly stringent for numerous real-world applications, where acquiring the same label space as the target task is prohibitively expensive. For this setting, we tackle the multimodal version of the unsupervised domain generalization (UDG) problem, which uses a large task-agnostic unlabeled source dataset, such as LAION-2B during finetuning. Our framework does not explicitly assume any relationship between the source dataset and target task. Instead, it relies only on the premise that the source dataset can be efficiently searched in a joint vision-language space. For this multimodal UDG setting, we propose a novel method to build a small ($<$100K) subset of the source data in th
    
[^34]: 通过学习学习算法，实现更灵活的PAC-Bayesian元学习

    More Flexible PAC-Bayesian Meta-Learning by Learning Learning Algorithms

    [https://arxiv.org/abs/2402.04054](https://arxiv.org/abs/2402.04054)

    通过学习学习算法，实现更灵活的PAC-Bayesian元学习，允许更灵活的任务之间的知识转移，提供新的泛化界限，可适用于分析和设计各种元学习机制，并在实际应用中改善了预测质量。

    

    我们引入了一种使用PAC-Bayesian理论研究元学习方法的新框架。与之前的工作相比，其主要优势在于它允许在任务之间的知识转移中更具灵活性。以往的方法只能通过学习模型的先验分布间接发生。相比之下，我们证明的新的泛化界限更直接地表达了元学习的过程，即学习适用于将来任务的学习算法。我们的框架的灵活性使其适用于分析各种元学习机制甚至设计新的机制。除了我们的理论贡献外，我们还在实际元学习机制中展示了我们的框架提高了预测质量。

    We introduce a new framework for studying meta-learning methods using PAC-Bayesian theory. Its main advantage over previous work is that it allows for more flexibility in how the transfer of knowledge between tasks is realized. For previous approaches, this could only happen indirectly, by means of learning prior distributions over models. In contrast, the new generalization bounds that we prove express the process of meta-learning much more directly as learning the learning algorithm that should be used for future tasks. The flexibility of our framework makes it suitable to analyze a wide range of meta-learning mechanisms and even design new mechanisms. Other than our theoretical contributions we also show empirically that our framework improves the prediction quality in practical meta-learning mechanisms.
    
[^35]: 高效求解偏差Gromov-Wasserstein问题

    Efficient Solvers for Partial Gromov-Wasserstein

    [https://arxiv.org/abs/2402.03664](https://arxiv.org/abs/2402.03664)

    本文提出了两个基于Frank-Wolfe算法的新的高效求解器来解决偏差Gromov-Wasserstein问题，并且证明了PGW问题构成了度量测度空间的度量。

    

    偏差Gromov-Wasserstein（PGW）问题可以比较具有不均匀质量的度量空间中的测度，从而实现这些空间之间的不平衡和部分匹配。本文证明了PGW问题可以转化为Gromov-Wasserstein问题的一个变种，类似于把偏差最优运输问题转化为最优运输问题。这个转化导致了两个新的求解器，基于Frank-Wolfe算法，数学和计算上等价，提供了高效的PGW问题解决方案。我们进一步证明了PGW问题构成了度量测度空间的度量。最后，我们通过与现有基线方法在形状匹配和正样本未标记学习问题上的计算时间和性能比较，验证了我们提出的求解器的有效性。

    The partial Gromov-Wasserstein (PGW) problem facilitates the comparison of measures with unequal masses residing in potentially distinct metric spaces, thereby enabling unbalanced and partial matching across these spaces. In this paper, we demonstrate that the PGW problem can be transformed into a variant of the Gromov-Wasserstein problem, akin to the conversion of the partial optimal transport problem into an optimal transport problem. This transformation leads to two new solvers, mathematically and computationally equivalent, based on the Frank-Wolfe algorithm, that provide efficient solutions to the PGW problem. We further establish that the PGW problem constitutes a metric for metric measure spaces. Finally, we validate the effectiveness of our proposed solvers in terms of computation time and performance on shape-matching and positive-unlabeled learning problems, comparing them against existing baselines.
    
[^36]: 扩散吉布斯采样

    Diffusive Gibbs Sampling

    [https://arxiv.org/abs/2402.03008](https://arxiv.org/abs/2402.03008)

    扩散吉布斯采样是一种创新的采样方法，通过集成扩散模型并应用吉布斯采样，有效地从具有远程和断开模态特征的分布中采样，表现出比其他方法更好的混合性能，并在多种任务中取得显著改进的结果。

    

    传统马尔可夫链蒙特卡洛（MCMC）方法在多模态分布的混合不足方面存在着挑战，特别是在贝叶斯推断和分子动力学等实际应用中。针对这个问题，我们提出了一种创新的采样方法——扩散吉布斯采样（DiGS），用于有效采样具有远程和断开模态特征的分布。DiGS集成了扩散模型的最新发展，利用高斯卷积创建一个辅助噪声分布，以在原始空间中连接孤立的模态，并应用吉布斯采样从两个空间中交替抽取样本。我们的方法在采样多模态分布方面表现出比并行温度法等最先进方法更好的混合性能。我们证明我们的采样器在各种任务中取得了显著改进的结果，包括高斯混合模型、贝叶斯神经网络和分子动力学。

    The inadequate mixing of conventional Markov Chain Monte Carlo (MCMC) methods for multi-modal distributions presents a significant challenge in practical applications such as Bayesian inference and molecular dynamics. Addressing this, we propose Diffusive Gibbs Sampling (DiGS), an innovative family of sampling methods designed for effective sampling from distributions characterized by distant and disconnected modes. DiGS integrates recent developments in diffusion models, leveraging Gaussian convolution to create an auxiliary noisy distribution that bridges isolated modes in the original space and applying Gibbs sampling to alternately draw samples from both spaces. Our approach exhibits a better mixing property for sampling multi-modal distributions than state-of-the-art methods such as parallel tempering. We demonstrate that our sampler attains substantially improved results across various tasks, including mixtures of Gaussians, Bayesian neural networks and molecular dynamics.
    
[^37]: InterpretCC: 适于解释的神经网络的条件计算

    InterpretCC: Conditional Computation for Inherently Interpretable Neural Networks

    [https://arxiv.org/abs/2402.02933](https://arxiv.org/abs/2402.02933)

    InterpretCC是一种新的解释性神经网络模型，通过条件计算和稀疏激活特征，在保持性能的同时实现了人类中心的解释能力。该模型适用于需要可信解释、可操作解释和准确预测的人类面向领域。

    

    神经网络的真实世界解释性在三个方面之间存在权衡：1）需要人类信任解释的近似（例如事后方法）；2）削弱了解释的可理解性（例如自动识别的特征掩码）；3）削弱了模型性能（例如决策树）。这些缺点对于面向人类的领域（如教育、医疗保健或自然语言）是不可接受的，这些领域需要可信的解释、可操作的解释和准确的预测。在这项工作中，我们提出了InterpretCC（可解释的条件计算），这是一种可解释性的设计神经网络系列，通过在预测之前自适应和稀疏地激活特征，确保人类中心的可解释性，同时保持与最先进模型相当的性能。我们将这个思想扩展为可解释的专家混合模型，允许人们离散地指定兴趣话题。

    Real-world interpretability for neural networks is a tradeoff between three concerns: 1) it requires humans to trust the explanation approximation (e.g. post-hoc approaches), 2) it compromises the understandability of the explanation (e.g. automatically identified feature masks), and 3) it compromises the model performance (e.g. decision trees). These shortcomings are unacceptable for human-facing domains, like education, healthcare, or natural language, which require trustworthy explanations, actionable interpretations, and accurate predictions. In this work, we present InterpretCC (interpretable conditional computation), a family of interpretable-by-design neural networks that guarantee human-centric interpretability while maintaining comparable performance to state-of-the-art models by adaptively and sparsely activating features before prediction. We extend this idea into an interpretable mixture-of-experts model, that allows humans to specify topics of interest, discretely separate
    
[^38]: 一张图值千言：使用纯Transformer将图形欧拉化

    A Graph is Worth $K$ Words: Euclideanizing Graph using Pure Transformer

    [https://arxiv.org/abs/2402.02464](https://arxiv.org/abs/2402.02464)

    这篇论文介绍了GraphsGPT，它使用纯Transformer将非欧几里德图形转换为在欧几里德空间中可学习的图形单词，并通过解码器将图形单词重新构建为原始图形，保证了信息的等价性。预训练的GraphsGPT在图形表示学习和图形生成方面取得了突出成果。

    

    我们能否将非欧几里德图形建模为纯语言甚至欧几里德向量，同时保留其固有信息？非欧几里德性质一直是图形建模中的长期挑战。尽管最近的GNN和Graphformer努力将图形编码为欧几里德向量，但从向量中恢复出原始图形仍然是一个挑战。我们引入了GraphsGPT，它具有一个将非欧几里德图形转换为在欧几里德空间中可学习图形单词的Graph2Seq编码器，以及一个从图形单词重构原始图形以确保信息等价性的GraphGPT解码器。我们在100M个分子上预训练了GraphsGPT，并得到一些有趣的发现：(1) 预训练的Graph2Seq在图形表示学习方面表现出色，在8/9个图形分类和回归任务上取得了最新成果。(2) 预训练的GraphGPT作为一个强大的图形生成器，其能够进行无条件和有条件的图形生成。(3) Graph2Seq+Gr

    Can we model non-Euclidean graphs as pure language or even Euclidean vectors while retaining their inherent information? The non-Euclidean property have posed a long term challenge in graph modeling. Despite recent GNN and Graphformer efforts encoding graphs as Euclidean vectors, recovering original graph from the vectors remains a challenge. We introduce GraphsGPT, featuring a Graph2Seq encoder that transforms non-Euclidean graphs into learnable graph words in a Euclidean space, along with a GraphGPT decoder that reconstructs the original graph from graph words to ensure information equivalence. We pretrain GraphsGPT on 100M molecules and yield some interesting findings: (1) Pretrained Graph2Seq excels in graph representation learning, achieving state-of-the-art results on 8/9 graph classification and regression tasks. (2) Pretrained GraphGPT serves as a strong graph generator, demonstrated by its ability to perform both unconditional and conditional graph generation. (3) Graph2Seq+Gr
    
[^39]: 超越极限：扩展大型语言模型中上下文长度的技术综述

    Beyond the Limits: A Survey of Techniques to Extend the Context Length in Large Language Models

    [https://arxiv.org/abs/2402.02244](https://arxiv.org/abs/2402.02244)

    这篇论文综述了近期为扩展大型语言模型中上下文长度而设计的技术和方法，并回顾了包括架构修改在内的多种技术，使得语言模型可以更有效地理解长上下文。

    

    近期，大型语言模型（LLMs）展现出了令人惊异的能力，包括理解上下文、进行逻辑推理和生成响应。然而，这是以严格的计算和内存要求为代价的，限制了它们有效支持长输入序列的能力。本综述全面回顾了最近为扩展LLMs序列长度而设计的技术和方法，从而增强其对长上下文理解的能力。具体而言，我们回顾和分类了各种技术，包括修改位置编码和修改注意机制等架构修改，旨在增强对更长序列的处理，同时避免计算需求的成比例增加。本研究探讨的多样方法可以在LLMs的不同阶段（即训练、微调和推理）中利用。这使得LLMs可以有效地处理长序列并提升对长上下文的理解能力。

    Recently, large language models (LLMs) have shown remarkable capabilities including understanding context, engaging in logical reasoning, and generating responses. However, this is achieved at the expense of stringent computational and memory requirements, hindering their ability to effectively support long input sequences. This survey provides an inclusive review of the recent techniques and methods devised to extend the sequence length in LLMs, thereby enhancing their capacity for long-context understanding. In particular, we review and categorize a wide range of techniques including architectural modifications, such as modified positional encoding and altered attention mechanisms, which are designed to enhance the processing of longer sequences while avoiding a proportional increase in computational requirements. The diverse methodologies investigated in this study can be leveraged across different phases of LLMs, i.e., training, fine-tuning and inference. This enables LLMs to effic
    
[^40]: 用分布式代理解释图神经网络

    Interpreting Graph Neural Networks with In-Distributed Proxies

    [https://arxiv.org/abs/2402.02036](https://arxiv.org/abs/2402.02036)

    该论文提出了一种翻译图神经网络中可解释子图的代理图的新方法，解决了训练数据分布与可解释子图集之间的分布偏移问题。

    

    图神经网络（GNN）已成为图数据处理的重要组成部分，在关键领域广泛应用。在高风险应用中部署GNN的不断增长需求需要用户在决策过程中能够解释其原因。解释GNN的流行范式是通过比较它们与原始图的标签来识别可解释的子图。由于训练集中原始图与可解释子图集之间存在显著的分布偏移，导致无法准确预测子图的标签，这是一个具有挑战性的任务。为了解决这个问题，在本文中，我们提出了一种新的方法，用于生成与训练数据分布相符的可解释子图的代理图。我们引入了一个使用图生成器生成代理图的参数化方法。基于信息论设计了一个新的训练目标，以确保代理图不仅遵循训练数据的分布，而且便于解释。

    Graph Neural Networks (GNNs) have become a building block in graph data processing, with wide applications in critical domains. The growing needs to deploy GNNs in high-stakes applications necessitate explainability for users in the decision-making processes. A popular paradigm for the explainability of GNNs is to identify explainable subgraphs by comparing their labels with the ones of original graphs. This task is challenging due to the substantial distributional shift from the original graphs in the training set to the set of explainable subgraphs, which prevents accurate prediction of labels with the subgraphs. To address it, in this paper, we propose a novel method that generates proxy graphs for explainable subgraphs that are in the distribution of training data. We introduce a parametric method that employs graph generators to produce proxy graphs. A new training objective based on information theory is designed to ensure that proxy graphs not only adhere to the distribution of 
    
[^41]: 《在大规模人工智能时代的贝叶斯深度学习》的立场论文

    Position Paper: Bayesian Deep Learning in the Age of Large-Scale AI

    [https://arxiv.org/abs/2402.00809](https://arxiv.org/abs/2402.00809)

    《在大规模人工智能时代的贝叶斯深度学习》这篇立场论文探讨了贝叶斯深度学习在各种不同设置下的优势，并指出了与之相关的挑战和有趣的研究方向。未来的研究重点将放在如何将大规模基础模型与贝叶斯深度学习相结合，以发挥它们的全部潜力。

    

    在当前的深度学习研究领域中，人们主要关注在涉及大规模图像和语言数据集的监督任务中实现高预测准确性。然而，更广泛的视角揭示了许多被忽视的度量标准、任务和数据类型，如不确定性、主动和持续学习以及科学数据，这些方面需要关注。贝叶斯深度学习（BDL）是一条有前景的道路，可以在这些不同的设置中提供优势。本文认为BDL可以提升深度学习的能力。它重新审视了BDL的优势、承认了现有的挑战，并重点介绍了一些旨在解决这些障碍的有趣的研究方向。展望未来，讨论集中在可能的方式上，将大规模基础模型与BDL相结合，以充分发挥它们的潜力。

    In the current landscape of deep learning research, there is a predominant emphasis on achieving high predictive accuracy in supervised tasks involving large image and language datasets. However, a broader perspective reveals a multitude of overlooked metrics, tasks, and data types, such as uncertainty, active and continual learning, and scientific data, that demand attention. Bayesian deep learning (BDL) constitutes a promising avenue, offering advantages across these diverse settings. This paper posits that BDL can elevate the capabilities of deep learning. It revisits the strengths of BDL, acknowledges existing challenges, and highlights some exciting research avenues aimed at addressing these obstacles. Looking ahead, the discussion focuses on possible ways to combine large-scale foundation models with BDL to unlock their full potential.
    
[^42]: InstructRetro: 检索增强的预训练中指令调优

    InstructRetro: Instruction Tuning post Retrieval-Augmented Pretraining

    [https://arxiv.org/abs/2310.07713](https://arxiv.org/abs/2310.07713)

    InstructRetro是目前规模最大的使用检索预训练的LLM，扩展了基础模型Retro 48B，通过指令调优在各种零样例任务上取得显著改进。

    

    使用检索增强技术对自回归大型语言模型（LLM）进行预训练可以提高困惑度和事实准确性。然而，现有的预训练检索增强LLM的规模仍然有限（如Retro具有75亿个参数），这限制了指令调优和零样例泛化的效果。本文介绍了Retro 48B，这是目前规模最大的使用检索预训练的LLM。具体来说，我们使用检索技术从1.2万亿个标记中继续预训练一个43B的GPT模型，并借助Retro方法将其扩展到4800亿个参数。值得注意的是，所得到的基础模型Retro 48B在困惑度方面显著优于仅使用1.2万亿个标记进行训练的43B GPT模型，且只增加了2.58%的GPU使用时间，展示了该方法的显著扩展潜力。在对Retro进行指令调优后，InstructRetro在各种零样例任务上表现出显著的改进。

    Pretraining auto-regressive large language models (LLMs) with retrieval demonstrates better perplexity and factual accuracy by leveraging external databases. However, the size of existing pretrained retrieval-augmented LLM is still limited (e.g., Retro has 7.5B parameters), which limits the effectiveness of instruction tuning and zero-shot generalization. In this work, we introduce Retro 48B, the largest LLM pretrained with retrieval. Specifically, we continue to pretrain a 43B GPT model on additional 100 billion tokens using the Retro augmentation method by retrieving from 1.2 trillion tokens. Notably, the obtained foundation model, Retro 48B, largely outperforms the counterpart GPT 43B trained on 1.2T tokens in terms of perplexity with only 2.58% additional GPU hours, demonstrating the significant scaling potential of the method. After instruction tuning on Retro, InstructRetro demonstrates significant improvement over the instruction tuned GPT on a wide range of zero-shot tasks. Spe
    
[^43]: 一个基于均匀化方法的梯度主导随机优化方法

    A Homogenization Approach for Gradient-Dominated Stochastic Optimization

    [https://arxiv.org/abs/2308.10630](https://arxiv.org/abs/2308.10630)

    本文介绍了一种基于均匀化方法的梯度主导随机优化方法，通过满足梯度主导性质的随机函数，实现全局收敛。我们提供了样本复杂度分析，并通过方差减少技术提供了增强结果。实验结果表明，该方法在无需立方正则化的情况下达到了最佳样本复杂度。

    

    梯度主导性质是一种比强凸性条件更弱但足以确保全局收敛的条件，即使在非凸优化中也可以应用广泛。本文提出了一种基于最近提出的均匀化方法的梯度主导随机二阶下降方法（SHSODM），用于满足梯度主导性质的随机函数。从理论上讲，我们提供了其样本复杂度分析，并通过结合方差减少技术提供了进一步的增强结果。我们的发现表明，SHSODM与其他梯度主导随机优化的二阶方法相比，可以达到已知的最佳样本复杂度，而无需立方正则化。从经验上讲，由于均匀化方法仅依赖于每次迭代中解极值特征向量问题，而不是牛顿类型的系统，所以我们的方法具有更低的计算成本。

    Gradient dominance property is a condition weaker than strong convexity, yet sufficiently ensures global convergence even in non-convex optimization. This property finds wide applications in machine learning, reinforcement learning (RL), and operations management. In this paper, we propose the stochastic homogeneous second-order descent method (SHSODM) for stochastic functions enjoying gradient dominance property based on a recently proposed homogenization approach. Theoretically, we provide its sample complexity analysis, and further present an enhanced result by incorporating variance reduction techniques. Our findings show that SHSODM matches the best-known sample complexity achieved by other second-order methods for gradient-dominated stochastic optimization but without cubic regularization. Empirically, since the homogenization approach only relies on solving extremal eigenvector problem at each iteration instead of Newton-type system, our methods gain the advantage of cheaper com
    
[^44]: 使用深度学习和开放地球观测数据实现全球冰川制图

    Towards Global Glacier Mapping with Deep Learning and Open Earth Observation Data. (arXiv:2401.15113v1 [cs.CV])

    [http://arxiv.org/abs/2401.15113](http://arxiv.org/abs/2401.15113)

    本研究提出了一种使用深度学习和开放地球观测数据进行全球冰川制图的方法，通过新的模型和策略，在多种地形和传感器上实现了较高的准确性。通过添加合成孔径雷达数据，并报告冰川范围的校准置信度，提高了预测的可靠性和可解释性。

    

    准确的全球冰川制图对于理解气候变化的影响至关重要。这个过程受到冰川多样性、难以分类的碎石和大数据处理的挑战。本文提出了Glacier-VisionTransformer-U-Net (GlaViTU)，一个卷积-Transformer深度学习模型，并提出了五种利用开放卫星影像进行多时相全球冰川制图的策略。空间、时间和跨传感器的泛化性能评估表明，我们的最佳策略在大多数情况下实现了IoU（交并比）> 0.85，并且在以冰雪为主的地区增加到了> 0.90，而在高山亚洲等碎石丰富的区域则降至> 0.75。此外，添加合成孔径雷达数据，即回波和干涉相干度，可以提高所有可用地区的准确性。报告冰川范围的校准置信度使预测更可靠和可解释。我们还发布了一个基准数据集。

    Accurate global glacier mapping is critical for understanding climate change impacts. It is challenged by glacier diversity, difficult-to-classify debris and big data processing. Here we propose Glacier-VisionTransformer-U-Net (GlaViTU), a convolutional-transformer deep learning model, and five strategies for multitemporal global-scale glacier mapping using open satellite imagery. Assessing the spatial, temporal and cross-sensor generalisation shows that our best strategy achieves intersection over union >0.85 on previously unobserved images in most cases, which drops to >0.75 for debris-rich areas such as High-Mountain Asia and increases to >0.90 for regions dominated by clean ice. Additionally, adding synthetic aperture radar data, namely, backscatter and interferometric coherence, increases the accuracy in all regions where available. The calibrated confidence for glacier extents is reported making the predictions more reliable and interpretable. We also release a benchmark dataset 
    
[^45]: 一种新型的混合时变图神经网络用于交通流量预测

    A novel hybrid time-varying graph neural network for traffic flow forecasting. (arXiv:2401.10155v1 [cs.LG])

    [http://arxiv.org/abs/2401.10155](http://arxiv.org/abs/2401.10155)

    本文提出了一种新型的混合时变图神经网络（HTVGNN）用于交通流量预测，解决了现有方法中预定义图和自适应图的学习能力受限的问题。

    

    实时准确的交通流量预测是确保智能交通系统高效运行的基础。在现有基于图神经网络（GNN）的交通流量预测方法中，通常使用预定义的图来描述城市道路网络中不同交通节点的空间相关性。然而，预定义图描述空间相关性的能力受限于先前的知识和图生成方法。尽管基于数据驱动学习的时变图能部分克服预定义图的缺点，但现有自适应图的学习能力有限。例如，时变图不能充分捕捉交通流量数据中固有的空间相关性。为了解决这些问题，我们提出了一种用于交通流量预测的混合时变图神经网络（HTVGNN）。

    Real-time and accurate traffic flow prediction is the foundation for ensuring the efficient operation of intelligent transportation systems.In existing traffic flow prediction methods based on graph neural networks (GNNs), pre-defined graphs were usually used to describe the spatial correlations of different traffic nodes in urban road networks. However, the ability of pre-defined graphs used to describe spatial correlation was limited by prior knowledge and graph generation methods. Although time-varying graphs based on data-driven learning can partially overcome the drawbacks of pre-defined graphs, the learning ability of existing adaptive graphs was limited. For example, time-varying graphs cannot adequately capture the inherent spatial correlations in traffic flow data.In order to solve these problems, we have proposed a hybrid time-varying graph neural network (HTVGNN) for traffic flow prediction.
    
[^46]: 大型语言模型中的代码模拟挑战

    Code Simulation Challenges for Large Language Models. (arXiv:2401.09074v1 [cs.LG])

    [http://arxiv.org/abs/2401.09074](http://arxiv.org/abs/2401.09074)

    大型语言模型在模拟计算机代码和算法执行方面遇到挑战，性能随着代码长度的增加而迅速下降。在处理短程序或标准过程时，它们能以低错误率按顺序执行指令，但对于复杂的程序，特别是包含关键路径和冗余指令的程序，模拟效果较差。我们提出了一种逐行模拟代码执行的方法来解决这个问题。

    

    我们调查了大型语言模型（LLMs）在模拟计算机代码和算法执行方面的能力。我们首先研究了直线程序，并展示了当前LLMs在处理这样简单的程序时表现出的性能较差——性能随着代码长度的增加而迅速下降。接着，我们研究了LLMs在模拟包含关键路径和冗余指令的程序方面的能力。我们还通过排序算法和嵌套循环超越了直线程序的模拟，并展示了程序的计算复杂性直接影响LLMs模拟其执行的能力。我们观察到LLMs只有在处理短程序或标准过程时才能以低错误率按顺序执行指令。LLMs的代码模拟与它们的模式识别和记忆能力存在矛盾：在记忆对任务有害的情况下，我们提出了一种新的提示方法，逐行模拟代码的执行。

    We investigate the extent to which Large Language Models (LLMs) can simulate the execution of computer code and algorithms. We begin by looking straight line programs, and show that current LLMs demonstrate poor performance even with such simple programs -- performance rapidly degrades with the length of code. We then investigate the ability of LLMs to simulate programs that contain critical paths and redundant instructions. We also go beyond straight line program simulation with sorting algorithms and nested loops, and we show the computational complexity of a routine directly affects the ability of an LLM to simulate its execution. We observe that LLMs execute instructions sequentially and with a low error margin only for short programs or standard procedures. LLMs' code simulation is in tension with their pattern recognition and memorisation capabilities: on tasks where memorisation is detrimental, we propose a novel prompting method to simulate code execution line by line. Empirica
    
[^47]: SynHIN: 生成用于可解释人工智能的合成异构信息网络

    SynHIN: Generating Synthetic Heterogeneous Information Network for Explainable AI. (arXiv:2401.04133v1 [cs.LG])

    [http://arxiv.org/abs/2401.04133](http://arxiv.org/abs/2401.04133)

    该论文提出了一种生成合成异构信息网络的方法，用于可解释人工智能。该方法通过识别现实世界数据集中的模式，构建合成网络，并确保生成的合成图数据与真实数据接近。这提供了用于节点分类任务的合成异构图数据集。

    

    图神经网络在各个领域有着优秀的表现，从检测电子商务垃圾邮件到社交网络分类问题。然而，缺乏公共图数据集阻碍了研究进展，尤其是在异构信息网络（HIN）方面。由于图神经网络解释模型的进展，对于公平的HIN比较而言，需要数据集的需求越来越大。为此，我们提出了SynHIN，一种生成合成异构信息网络的独特方法。SynHIN识别现实世界数据集中的模式，总结图统计数据，并构建合成网络。我们的方法利用In-Cluster和Out-Cluster Merge模块从主要的模式集群构建合成HIN。在In/Out-Cluster合并和符合真实数据集约束的后修剪过程后，我们确保合成的图统计数据与参考数据接近。SynHIN生成用于节点分类任务的合成异构图数据集，使用主要的模式作为输入。

    Graph Neural Networks (GNNs) excel in various domains, from detecting e-commerce spam to social network classification problems. However, the lack of public graph datasets hampers research progress, particularly in heterogeneous information networks (HIN). The demand for datasets for fair HIN comparisons is growing due to advancements in GNN interpretation models. In response, we propose SynHIN, a unique method for generating synthetic heterogeneous information networks. SynHIN identifies motifs in real-world datasets, summarizes graph statistics, and constructs a synthetic network. Our approach utilizes In-Cluster and Out-Cluster Merge modules to build the synthetic HIN from primary motif clusters. After In/Our-Cluster mergers and a post-pruning process fitting the real dataset constraints, we ensure the synthetic graph statistics align closely with the reference one. SynHIN generates a synthetic heterogeneous graph dataset for node classification tasks, using the primary motif as the
    
[^48]: 概念瓶颈模型是否遵循局部性？

    Do Concept Bottleneck Models Obey Locality?. (arXiv:2401.01259v1 [cs.LG])

    [http://arxiv.org/abs/2401.01259](http://arxiv.org/abs/2401.01259)

    本文研究了概念瓶颈模型（CBMs）是否能够正确捕捉到概念之间的条件独立程度，通过分析对于概念局部性之外特征的变化如何影响概念的预测。

    

    概念基础学习通过解释其预测结果使用人可理解的概念，改善了深度学习模型的可解释性。在这种范式下训练的深度学习模型严重依赖于神经网络能够学习独立于其他概念的给定概念的存在或不存在。然而，最近的研究强烈暗示这种假设可能在概念瓶颈模型（CBMs）这一典型的基于概念的可解释架构中不能成立。本文中，我们研究了当这些概念既在空间上（通过它们的值完全由固定子集的特征定义）又在语义上（通过它们的值仅与预定义的固定子集的概念相关联）定位时，CBMs是否正确捕捉到概念之间的条件独立程度。为了理解局部性，我们分析了概念之外的特征变化对概念预测的影响。

    Concept-based learning improves a deep learning model's interpretability by explaining its predictions via human-understandable concepts. Deep learning models trained under this paradigm heavily rely on the assumption that neural networks can learn to predict the presence or absence of a given concept independently of other concepts. Recent work, however, strongly suggests that this assumption may fail to hold in Concept Bottleneck Models (CBMs), a quintessential family of concept-based interpretable architectures. In this paper, we investigate whether CBMs correctly capture the degree of conditional independence across concepts when such concepts are localised both spatially, by having their values entirely defined by a fixed subset of features, and semantically, by having their values correlated with only a fixed subset of predefined concepts. To understand locality, we analyse how changes to features outside of a concept's spatial or semantic locality impact concept predictions. Our
    
[^49]: 自动化模型选择算法用于表格数据

    Automated Model Selection for Tabular Data. (arXiv:2401.00961v1 [cs.LG])

    [http://arxiv.org/abs/2401.00961](http://arxiv.org/abs/2401.00961)

    本文介绍了一种自动化模型选择算法，用于表格数据的预测。该算法考虑了特征之间的交互，并包含了基于优先级的随机网格搜索和贪婪搜索两种不同的特征选择方法。

    

    表格数据中的结构化数据包含独特且离散的特征，并且这些特征对目标的重要性各不相同。单个特征的组合可能比简单的单个特征贡献更具预测性和意义。R的混合效应线性模型库允许用户在模型设计中提供这种交互式特征组合。然而，鉴于有许多特征和可能的交互选择，模型选择变得非常困难。我们的目标是通过保持计算成本较小，自动化表格数据预测中的模型选择过程，并同时考虑特征之间的交互。该框架包括两种不同的特征选择方法：基于优先级的随机网格搜索和贪婪搜索方法。基于优先级的方法利用先验概率来引导搜索，高效地探索特征组合。贪婪方法通过迭代地添加特征构建解决方案。

    Structured data in the form of tabular datasets contain features that are distinct and discrete, with varying individual and relative importances to the target. Combinations of one or more features may be more predictive and meaningful than simple individual feature contributions. R's mixed effect linear models library allows users to provide such interactive feature combinations in the model design. However, given many features and possible interactions to select from, model selection becomes an exponentially difficult task. We aim to automate the model selection process for predictions on tabular datasets incorporating feature interactions while keeping computational costs small. The framework includes two distinct approaches for feature selection: a Priority-based Random Grid Search and a Greedy Search method. The Priority-based approach efficiently explores feature combinations using prior probabilities to guide the search. The Greedy method builds the solution iteratively by addin
    
[^50]: FaultFormer: 预训练Transformer用于适应性轴承故障分类

    FaultFormer: Pretraining Transformers for Adaptable Bearing Fault Classification. (arXiv:2312.02380v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2312.02380](http://arxiv.org/abs/2312.02380)

    本论文提出了一个使用预训练Transformer模型进行适应性轴承故障分类的框架。通过研究不同的标记分割和数据增强策略，该方法在稀缺数据环境中能够达到最先进的准确率，并在微调时改善了性能。

    

    全球消费的增长推动了深度学习在智能制造和机器健康监测方面的重要应用。特别是，振动数据提供了丰富可靠的信息，能够对机器健康和预测性维护进行有意义的洞察。在本工作中，我们提出了基于Transformer模型的轴承故障识别的预训练和微调框架。我们研究了不同的标记分割和数据增强策略，以提高性能并达到最先进的准确率。此外，我们展示了针对振动信号的掩码自监督预训练及其在低数据环境、任务适应和数据集适应中的应用。预训练能够提升在稀缺未见训练样本上的10类轴承分类性能。当在预训练分布之外的故障类别上进行微调时，Transformer模型也受益于预训练。最后，我们展示了预训练的Transformer模型。

    The growth of global consumption has motivated important applications of deep learning to smart manufacturing and machine health monitoring. In particular, vibration data offers a rich and reliable source to provide meaningful insights into machine health and predictive maintenance. In this work, we present pretraining and fine-tuning frameworks for identifying bearing faults based on transformer models. In particular, we investigate different tokenization and data augmentation strategies to improve performance and reach state of the art accuracies. Furthermore, we demonstrate masked self-supervised pretraining for vibration signals and its application to low-data regimes, task adaptation, and dataset adaptation. Pretraining is able to improve performance on 10-way bearing classification on scarce, unseen training samples. Transformer models also benefit from pretraining when fine-tuning on fault classes outside of the pretraining distribution. Lastly, pretrained transformers are shown
    
[^51]: 带预热的动量梯度下降的大型弹射概念研究

    Large Catapults in Momentum Gradient Descent with Warmup: An Empirical Study. (arXiv:2311.15051v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2311.15051](http://arxiv.org/abs/2311.15051)

    本研究通过实验证明，带有大学习率和学习率预热的动量梯度下降显示出大型弹射效应，将迭代朝着比梯度下降发现的更平缓的极小值方向推进。

    

    尽管动量梯度下降在现代深度学习中被广泛使用，但对其对训练轨迹的影响的具体理解仍然难以捉摸。本研究通过实验证明，带有大学习率和学习率预热的动量梯度下降显示出大型弹射效应，将迭代朝着比梯度下降发现的更平缓的极小值方向推进。然后我们提供了实证证据和理论直觉，表明大型弹射效应是由于动量“放大”了自稳定效应（Damian等，2023）。

    Although gradient descent with momentum is widely used in modern deep learning, a concrete understanding of its effects on the training trajectory still remains elusive. In this work, we empirically show that momentum gradient descent with a large learning rate and learning rate warmup displays large catapults, driving the iterates towards flatter minima than those found by gradient descent. We then provide empirical evidence and theoretical intuition that the large catapult is caused by momentum "amplifying" the self-stabilization effect (Damian et al., 2023).B.1
    
[^52]: MIST: 通过成员不变子空间训练对抗成员推理攻击

    MIST: Defending Against Membership Inference Attacks Through Membership-Invariant Subspace Training. (arXiv:2311.00919v1 [cs.CR])

    [http://arxiv.org/abs/2311.00919](http://arxiv.org/abs/2311.00919)

    通过成员不变子空间训练的MIST算法有效防御成员推理攻击，能够识别容易受到攻击的实例并避免过度拟合。

    

    在成员推理（MI）攻击中，对手试图确定一个实例是否被用来训练一个机器学习（ML）模型。MI攻击是在使用私有数据训练ML模型时的一个主要隐私问题。文献中的大多数MI攻击利用了ML模型被训练得很好以适应训练数据的特点，因此在训练实例上具有非常低的损失。因此，大多数对抗MI攻击的防御方法试图使模型在训练数据上的拟合程度降低。然而，这样做通常会导致较低的准确率。我们观察到训练实例对MI攻击具有不同程度的脆弱性。大多数实例即使不包含在训练中也会有低的损失。对于这些实例，模型可以很好地适应它们而不用担心MI攻击。有效的防御只需要（可能是隐式地）识别出容易受到MI攻击的实例，并避免过度拟合。一个主要的挑战是如何在高效的训练过程中实现这样的效果。

    In Member Inference (MI) attacks, the adversary try to determine whether an instance is used to train a machine learning (ML) model. MI attacks are a major privacy concern when using private data to train ML models. Most MI attacks in the literature take advantage of the fact that ML models are trained to fit the training data well, and thus have very low loss on training instances. Most defenses against MI attacks therefore try to make the model fit the training data less well. Doing so, however, generally results in lower accuracy. We observe that training instances have different degrees of vulnerability to MI attacks. Most instances will have low loss even when not included in training. For these instances, the model can fit them well without concerns of MI attacks. An effective defense only needs to (possibly implicitly) identify instances that are vulnerable to MI attacks and avoids overfitting them. A major challenge is how to achieve such an effect in an efficient training proc
    
[^53]: 一种由语义通信增强的无线AI生成内容（AIGC）供应框架

    A Wireless AI-Generated Content (AIGC) Provisioning Framework Empowered by Semantic Communication. (arXiv:2310.17705v1 [cs.NI])

    [http://arxiv.org/abs/2310.17705](http://arxiv.org/abs/2310.17705)

    一种由语义通信增强的无线AI生成内容（AIGC）供应框架，通过使用语义信息而不是所有的二进制位提取和传输内容，以解决在无线网络中提供最优AIGC服务的挑战。

    

    近期，生成式AI应用通过创建多样化且高质量的AI生成内容（AIGC）来满足广大用户群体的需求。随着移动设备的普及和移动流量的快速增长，通过无线通信网络提供对高质量AIGC服务的无处不在的访问已成为AIGC产品的未来方向。然而，在不稳定的信道、有限的带宽资源和分布不均匀的计算资源的无线网络中提供最优的AIGC服务是具有挑战性的。为了解决这些挑战，我们提出了一个由语义通信（SemCom）增强的AIGC（SemAIGC）生成和传输框架，其中只需提取和传输内容的语义信息而不是所有的二进制位。具体而言，SemAIGC在语义编码器和解码器中集成了基于扩散的模型，以实现高效的内容生成和灵活调整计算工作负载的目的。

    Generative AI applications are recently catering to a vast user base by creating diverse and high-quality AI-generated content (AIGC). With the proliferation of mobile devices and rapid growth of mobile traffic, providing ubiquitous access to high-quality AIGC services via wireless communication networks is becoming the future direction for AIGC products. However, it is challenging to provide optimal AIGC services in wireless networks with unstable channels, limited bandwidth resources, and unevenly distributed computational resources. To tackle these challenges, we propose a semantic communication (SemCom)-empowered AIGC (SemAIGC) generation and transmission framework, where only semantic information of the content rather than all the binary bits should be extracted and transmitted by using SemCom. Specifically, SemAIGC integrates diffusion-based models within the semantic encoder and decoder for efficient content generation and flexible adjustment of the computing workload of both tr
    
[^54]: 受约束的Actor Critic和受约束的Natural Actor Critic算法的有限时间分析

    Finite Time Analysis of Constrained Actor Critic and Constrained Natural Actor Critic Algorithms. (arXiv:2310.16363v1 [cs.LG])

    [http://arxiv.org/abs/2310.16363](http://arxiv.org/abs/2310.16363)

    本文研究了受约束的Actor Critic和受约束的Natural Actor Critic算法的有限时间分析，证明了这些算法能找到性能函数的一阶稳定点，并且具有较低的样本复杂度。

    

    Actor Critic方法在广泛的强化学习任务中找到了巨大的应用，特别是当状态-动作空间很大的时候。本文考虑使用函数逼近的actor critic和natural actor critic算法来处理涉及不等式约束的马尔可夫决策过程（C-MDP），并在非 i.i.d（马尔可夫）环境中进行了非渐近分析。我们考虑长期平均成本准则，其中目标和约束函数都是某些规定成本函数的适当策略依赖的长期平均。我们使用拉格朗日乘子法处理不等式约束。我们证明这些算法保证能找到性能（拉格朗日）函数$L(\theta,\gamma)$的一阶稳定点（即$\Vert \nabla L(\theta,\gamma)\Vert_2^2 \leq \epsilon$），并且其样本复杂度为$\mathcal{\tilde{O}}(\epsilon^{-2.5})$。

    Actor Critic methods have found immense applications on a wide range of Reinforcement Learning tasks especially when the state-action space is large. In this paper, we consider actor critic and natural actor critic algorithms with function approximation for constrained Markov decision processes (C-MDP) involving inequality constraints and carry out a non-asymptotic analysis for both of these algorithms in a non-i.i.d (Markovian) setting. We consider the long-run average cost criterion where both the objective and the constraint functions are suitable policy-dependent long-run averages of certain prescribed cost functions. We handle the inequality constraints using the Lagrange multiplier method. We prove that these algorithms are guaranteed to find a first-order stationary point (i.e., $\Vert \nabla L(\theta,\gamma)\Vert_2^2 \leq \epsilon$) of the performance (Lagrange) function $L(\theta,\gamma)$, with a sample complexity of $\mathcal{\tilde{O}}(\epsilon^{-2.5})$ in the case of both C
    
[^55]: 针对预测不确定性的模型无关变量重要性：一种基于熵的方法

    Model-agnostic variable importance for predictive uncertainty: an entropy-based approach. (arXiv:2310.12842v1 [stat.ML])

    [http://arxiv.org/abs/2310.12842](http://arxiv.org/abs/2310.12842)

    本文提出了一种基于熵的方法，通过扩展现有的解释性方法，可以理解不确定性感知模型中的预测来源和置信度，并利用改编后的特征重要性、部分依赖图和个体条件期望图等方法来测量特征对预测分布的熵和基于真实标签的对数似然的影响。

    

    为了相信机器学习算法的预测结果，必须理解导致这些预测的因素。对于概率和不确定性感知的模型来说，不仅需要理解预测本身的原因，还要理解模型对这些预测的置信度。本文展示了如何将现有的解释性方法扩展到不确定性感知的模型，并如何利用这些扩展来理解模型预测分布中的不确定性来源。特别是通过改编排列特征重要性、部分依赖图和个体条件期望图，我们证明可以获得对模型行为的新见解，并且可以使用这些方法来衡量特征对预测分布的熵和基于该分布的真实标签的对数似然的影响。通过使用两个数据集的实验，我们验证了所提方法的有效性。

    In order to trust the predictions of a machine learning algorithm, it is necessary to understand the factors that contribute to those predictions. In the case of probabilistic and uncertainty-aware models, it is necessary to understand not only the reasons for the predictions themselves, but also the model's level of confidence in those predictions. In this paper, we show how existing methods in explainability can be extended to uncertainty-aware models and how such extensions can be used to understand the sources of uncertainty in a model's predictive distribution. In particular, by adapting permutation feature importance, partial dependence plots, and individual conditional expectation plots, we demonstrate that novel insights into model behaviour may be obtained and that these methods can be used to measure the impact of features on both the entropy of the predictive distribution and the log-likelihood of the ground truth labels under that distribution. With experiments using both s
    
[^56]: ACES: 使用自我目标语言模型和语义描述符生成多样的编程难题

    ACES: generating diverse programming puzzles with autotelic language models and semantic descriptors. (arXiv:2310.10692v1 [cs.LG])

    [http://arxiv.org/abs/2310.10692](http://arxiv.org/abs/2310.10692)

    ACES是一种使用自我目标语言模型和语义描述符生成多样化的编程难题的方法，能够优化有趣的多样性和少样本生成。

    

    寻找和选择新颖有趣的问题是好奇心、科学和创新的核心。在Python编程难题的无限空间中，我们研究了自动问题生成。现有的生成模型通常旨在建模参考分布，没有明确的多样性优化。其他方法在有限的手工编码表示空间或不可解释的学习嵌入空间中明确优化多样性，这些嵌入空间可能与人类对有趣变化的感知不符。通过ACES（自我目标代码探索与语义描述符），我们引入了一种新的自我目标生成方法，利用大型语言模型（LLM）生成语义描述符，直接优化有趣的多样性，以及少样本生成。每个难题都标记有10个维度，每个维度捕捉了解决它所需的编程技能。ACES生成并追求新颖可行的目标。

    Finding and selecting new and interesting problems to solve is at the heart of curiosity, science and innovation. We here study automated problem generation in the context of the open-ended space of python programming puzzles. Existing generative models often aim at modeling a reference distribution without any explicit diversity optimization. Other methods explicitly optimizing for diversity do so either in limited hand-coded representation spaces or in uninterpretable learned embedding spaces that may not align with human perceptions of interesting variations. With ACES (Autotelic Code Exploration via Semantic descriptors), we introduce a new autotelic generation method that leverages semantic descriptors produced by a large language model (LLM) to directly optimize for interesting diversity, as well as few-shot-based generation. Each puzzle is labeled along 10 dimensions, each capturing a programming skill required to solve it. ACES generates and pursues novel and feasible goals to 
    
[^57]: 混合物与神经批评家：关于精细分布的点间互信息的研究

    The Mixtures and the Neural Critics: On the Pointwise Mutual Information Profiles of Fine Distributions. (arXiv:2310.10240v1 [stat.ML])

    [http://arxiv.org/abs/2310.10240](http://arxiv.org/abs/2310.10240)

    本文研究了点间互信息的特征，引入了细分布家族来解决现有互信息估计器的局限性，并探究了神经批评家在变分估计器中的行为，以及实验异常值对互信息估计的影响。此外，还介绍了基于模型的贝叶斯估计的方法，适用于具有领域专业知识且需要不确定性量化的问题。

    

    互信息量化了两个随机变量之间的依赖关系，并且在微分同胚下保持不变。在本文中，我们探讨了点间互信息的特征，这是互信息的推广形式，保持了这种不变性。我们在解析上描述了多元正态分布的特征，并引入了细分布家族，通过蒙特卡洛方法可以准确地逼近这种特征。然后，我们展示了如何利用细分布来研究现有互信息估计器的局限性，调查在变分估计器中使用的神经批评家的行为，并了解实验异常值对互信息估计的影响。最后，我们展示了如何利用细分布来获得基于模型的贝叶斯估计的互信息，适用于具有可用领域专业知识且需要不确定性量化的问题。

    Mutual information quantifies the dependence between two random variables and remains invariant under diffeomorphisms. In this paper, we explore the pointwise mutual information profile, an extension of mutual information that maintains this invariance. We analytically describe the profiles of multivariate normal distributions and introduce the family of fine distributions, for which the profile can be accurately approximated using Monte Carlo methods. We then show how fine distributions can be used to study the limitations of existing mutual information estimators, investigate the behavior of neural critics used in variational estimators, and understand the effect of experimental outliers on mutual information estimation. Finally, we show how fine distributions can be used to obtain model-based Bayesian estimates of mutual information, suitable for problems with available domain expertise in which uncertainty quantification is necessary.
    
[^58]: ParFam - 基于连续全局优化的符号回归

    ParFam -- Symbolic Regression Based on Continuous Global Optimization. (arXiv:2310.05537v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2310.05537](http://arxiv.org/abs/2310.05537)

    ParFam是一种新的符号回归方法，利用参数化的符号函数族将离散问题转化为连续问题，并结合全局优化器，能够有效解决符号回归问题。

    

    符号回归（SR）问题在许多不同的应用中出现，比如从给定数据中识别物理定律或推导描述金融市场行为的数学方程。目前存在多种解决SR问题的方法，通常基于遗传编程。然而，这些方法通常非常复杂，需要大量超参数调整和计算资源。本文介绍了我们提出的新方法ParFam，它利用适合的符号函数的参数化族将离散的符号回归问题转化为连续问题，相比当前最先进的方法，这种方法的设置更加直观。结合强大的全局优化器，这种方法可以有效地解决SR问题。此外，它可以轻松扩展到更高级的算法，例如添加深度神经网络以找到适合的参数化族。我们证明了这种方法的性能。

    The problem of symbolic regression (SR) arises in many different applications, such as identifying physical laws or deriving mathematical equations describing the behavior of financial markets from given data. Various methods exist to address the problem of SR, often based on genetic programming. However, these methods are usually quite complicated and require a lot of hyperparameter tuning and computational resources. In this paper, we present our new method ParFam that utilizes parametric families of suitable symbolic functions to translate the discrete symbolic regression problem into a continuous one, resulting in a more straightforward setup compared to current state-of-the-art methods. In combination with a powerful global optimizer, this approach results in an effective method to tackle the problem of SR. Furthermore, it can be easily extended to more advanced algorithms, e.g., by adding a deep neural network to find good-fitting parametric families. We prove the performance of 
    
[^59]: 关于不精确消除法在主成分分析中的误差传播

    On the Error-Propagation of Inexact Deflation for Principal Component Analysis. (arXiv:2310.04283v1 [cs.LG])

    [http://arxiv.org/abs/2310.04283](http://arxiv.org/abs/2310.04283)

    该论文研究了主成分分析中不精确消除法的误差传播问题，给出了两个主要结果

    

    主成分分析（PCA）是数据分析中常用的工具，尤其是在高维数据情况下。PCA旨在找到由所谓“主成分”所张成的子空间，这些主成分最能解释数据集的方差。消除法是一种常用的元算法，用于发现这样的子空间，它从最重要的主成分开始顺序地找到每个主成分，直到找到较不重要的主成分。然而，由于其顺序性质，由于不完全估计主成分引入的数值误差 - 例如，由于此过程中的数值近似 - 会随着消除的进行而传播。据我们所知，这是第一篇在数学上对不精确消除法的误差传播进行了特性化的工作，这是本文的关键贡献。我们提供了两个主要结果：$ i）$当用于查找主要特征向量的子例程是泛型的时候，以及$ ii）$

    Principal Component Analysis (PCA) is a popular tool in data analysis, especially when the data is high-dimensional. PCA aims to find subspaces, spanned by the so-called \textit{principal components}, that best explain the variance in the dataset. The deflation method is a popular meta-algorithm -used to discover such subspaces -- that sequentially finds individual principal components, starting from the most important one and working its way towards the less important ones. However, due to its sequential nature, the numerical error introduced by not estimating principal components exactly -- e.g., due to numerical approximations through this process -- propagates, as deflation proceeds. To the best of our knowledge, this is the first work that mathematically characterizes the error propagation of the inexact deflation method, and this is the key contribution of this paper. We provide two main results: $i)$ when the sub-routine for finding the leading eigenvector is generic, and $ii)
    
[^60]: DeepHGCN：朝着更深的双曲图卷积网络

    DeepHGCN: Toward Deeper Hyperbolic Graph Convolutional Networks. (arXiv:2310.02027v1 [cs.LG])

    [http://arxiv.org/abs/2310.02027](http://arxiv.org/abs/2310.02027)

    DeepHGCN是一个具有深层架构的双曲图卷积网络，通过引入新的双曲特征转换层和正则化技术，实现了计算效率的极大改进和过度平滑问题的显著减轻。

    

    双曲图卷积网络（HGCN）在提取分层图信息方面展示了巨大潜力。然而，由于昂贵的双曲操作和随着深度增加的过度平滑问题，现有的HGCN受限于浅层架构。尽管在GCNs中已经应用了一些方法来减轻过度平滑问题，但是开发双曲治疗方法面临着不同的挑战，因为操作必须经过精心设计以适应双曲性质。解决以上挑战，本文提出了DeepHGCN，这是第一个具有显著提高计算效率和大大减轻过度平滑效果的深层多层HGCN架构。DeepHGCN具有两个深层HGCN的关键因素：（1）一种新颖的双曲特征转换层，能够实现快速而准确的线性映射；（2）通过有效的双曲残差连接和权重和特征的正则化技术来促进。

    Hyperbolic graph convolutional networks (HGCN) have demonstrated significant potential in extracting information from hierarchical graphs. However, existing HGCNs are limited to shallow architectures, due to the expensive hyperbolic operations and the over-smoothing issue as depth increases. Although in GCNs, treatments have been applied to alleviate over-smoothing, developing a hyperbolic therapy presents distinct challenges since operations should be carefully designed to fit the hyperbolic nature. Addressing the above challenges, in this work, we propose DeepHGCN, the first deep multi-layer HGCN architecture with dramatically improved computational efficiency and substantially alleviated over-smoothing effect. DeepHGCN presents two key enablers of deep HGCNs: (1) a novel hyperbolic feature transformation layer that enables fast and accurate linear maps; and (2) Techniques such as hyperbolic residual connections and regularization for both weights and features facilitated by an effic
    
[^61]: 关于过参数化神经网络理论与实践的脱节

    On the Disconnect Between Theory and Practice of Overparametrized Neural Networks. (arXiv:2310.00137v1 [cs.LG])

    [http://arxiv.org/abs/2310.00137](http://arxiv.org/abs/2310.00137)

    本文研究了神经网络在无穷宽度极限下的行为，并与核方法建立了联系。虽然在合成架构中展示了一些优势，如更快的优化和可靠的不确定性量化，但实际相关的架构需要比深度大很多倍的宽度才能实现这些优势。

    

    神经网络（NNs）的无穷宽度极限作为分析大规模、过参数化网络行为的理论框架已经引起了重要关注。通过接近无限宽度，NNs可以有效地收敛到一个具有由神经切线核(NTK)特征化的线性模型。这建立了NNs和核方法之间的联系，后者是被充分理解的。基于这种联系，已经假设并在合成架构中从理论上和算法上验证了一些优势。这些优势包括更快的优化、可靠的不确定性量化和改进的持续学习能力。然而，目前量化向核心领域收敛速度的结果表明，利用这些优势需要比深度大几个数量级的架构。这个假设引发了对实际相关架构是否表现如预测的担忧。

    The infinite-width limit of neural networks (NNs) has garnered significant attention as a theoretical framework for analyzing the behavior of large-scale, overparametrized networks. By approaching infinite width, NNs effectively converge to a linear model with features characterized by the neural tangent kernel (NTK). This establishes a connection between NNs and kernel methods, the latter of which are well understood. Based on this link, theoretical benefits and algorithmic improvements have been hypothesized and empirically demonstrated in synthetic architectures. These advantages include faster optimization, reliable uncertainty quantification and improved continual learning. However, current results quantifying the rate of convergence to the kernel regime suggest that exploiting these benefits requires architectures that are orders of magnitude wider than they are deep. This assumption raises concerns that practically relevant architectures do not exhibit behavior as predicted via 
    
[^62]: 基于具有空间金字塔池化的卷积神经网络的网络稳健性评估的综合分析

    Comprehensive Analysis of Network Robustness Evaluation Based on Convolutional Neural Networks with Spatial Pyramid Pooling. (arXiv:2308.08012v1 [cs.CV])

    [http://arxiv.org/abs/2308.08012](http://arxiv.org/abs/2308.08012)

    本文通过设计具有空间金字塔池化网络的卷积神经网络模型，解决了网络稳健性评估中的性能、捕捉稳健性、可扩展性和可转移性等挑战。

    

    连通性稳健性是理解、优化和修复复杂网络的关键方面，传统上通过耗时且常常不切实际的模拟来评估。幸运的是，机器学习为解决这一挑战提供了一条新的途径。然而，仍然存在一些关键问题尚未解决，包括在更一般的边缘删除场景中的性能，通过攻击曲线捕捉稳健性而不是直接训练稳健性，预测任务的可扩展性以及预测能力的可转移性。本文通过设计具有空间金字塔池化网络的卷积神经网络模型(CNN)，调整现有的评估指标，重新设计攻击模式，引入适当的过滤规则，并将稳健性的价值作为训练数据加以解决这些挑战。结果表明，所提出的CNN框架在解决高计算挑战方面具有全面性。

    Connectivity robustness, a crucial aspect for understanding, optimizing, and repairing complex networks, has traditionally been evaluated through time-consuming and often impractical simulations. Fortunately, machine learning provides a new avenue for addressing this challenge. However, several key issues remain unresolved, including the performance in more general edge removal scenarios, capturing robustness through attack curves instead of directly training for robustness, scalability of predictive tasks, and transferability of predictive capabilities. In this paper, we address these challenges by designing a convolutional neural networks (CNN) model with spatial pyramid pooling networks (SPP-net), adapting existing evaluation metrics, redesigning the attack modes, introducing appropriate filtering rules, and incorporating the value of robustness as training data. The results demonstrate the thoroughness of the proposed CNN framework in addressing the challenges of high computational
    
[^63]: GPLaSDI: 基于高斯过程的可解释潜空间动力学识别方法通过深度自动编码器

    GPLaSDI: Gaussian Process-based Interpretable Latent Space Dynamics Identification through Deep Autoencoder. (arXiv:2308.05882v1 [cs.CE])

    [http://arxiv.org/abs/2308.05882](http://arxiv.org/abs/2308.05882)

    GPLaSDI是一种基于高斯过程的可解释潜空间动力学识别方法，通过深度自动编码器将完全阶数的PDE解映射到潜空间，并使用插值和解决ODE系统进行快速和准确的ROM预测。

    

    数值求解偏微分方程(PDEs)可能具有挑战性且计算成本高。这导致了减少阶数模型(ROMs)的发展，其精确性高于完全阶数模型(FOMs)但计算速度更快。最近，机器学习的进展实现了非线性投影方法的创建，例如潜空间动力学识别(LaSDI)。LaSDI使用自动编码器将完全阶数的PDE解映射到潜空间，并学习潜空间动力学的ODE系统。通过在减少的潜空间中插值和解决ODE系统，可以通过将预测的潜空间动力学输入解码器来进行快速且准确的ROM预测。在本文中，我们介绍了一种基于高斯过程(GP)的新型LaSDI框架，用于潜空间ODE插值。使用GP带来两个重要优势。首先，它能够量化ROM预测的不确定性。其次，利用这个预测。

    Numerically solving partial differential equations (PDEs) can be challenging and computationally expensive. This has led to the development of reduced-order models (ROMs) that are accurate but faster than full order models (FOMs). Recently, machine learning advances have enabled the creation of non-linear projection methods, such as Latent Space Dynamics Identification (LaSDI). LaSDI maps full-order PDE solutions to a latent space using autoencoders and learns the system of ODEs governing the latent space dynamics. By interpolating and solving the ODE system in the reduced latent space, fast and accurate ROM predictions can be made by feeding the predicted latent space dynamics into the decoder. In this paper, we introduce GPLaSDI, a novel LaSDI-based framework that relies on Gaussian process (GP) for latent space ODE interpolations. Using GPs offers two significant advantages. First, it enables the quantification of uncertainty over the ROM predictions. Second, leveraging this predict
    
[^64]: 具有收敛保证的公正感知联邦极小化优化

    Fairness-aware Federated Minimax Optimization with Convergence Guarantee. (arXiv:2307.04417v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2307.04417](http://arxiv.org/abs/2307.04417)

    本文提出了一种名为FFALM的算法，通过施加公平约束和解决极小化极大回归问题，在联邦学习中解决了群体公平性问题。实验证明FFALM在处理严重统计异质性问题时具有良好的效果。

    

    由于其保护隐私的特性，联邦学习 (FL) 吸引了相当多的关注。然而，管理用户数据的自由度不足可能导致群体公平性问题，即模型偏向于敏感因素诸如种族或性别。为了解决这个问题，本文提出了一种新颖的算法，名为带有增广拉格朗日方法的公平联邦平均法 (FFALM)，专门用于解决FL中的群体公平问题。具体来说，我们对训练目标施加了公平约束，并解决了受约束优化问题的极小化极大回归。然后，我们推导了FFALM的收敛速率的理论上界。通过在CelebA和UTKFace数据集中充分考虑严重统计异质性，实证结果表明了FFALM 在提高公平性方面的有效性。

    Federated learning (FL) has garnered considerable attention due to its privacy-preserving feature. Nonetheless, the lack of freedom in managing user data can lead to group fairness issues, where models are biased towards sensitive factors such as race or gender. To tackle this issue, this paper proposes a novel algorithm, fair federated averaging with augmented Lagrangian method (FFALM), designed explicitly to address group fairness issues in FL. Specifically, we impose a fairness constraint on the training objective and solve the minimax reformulation of the constrained optimization problem. Then, we derive the theoretical upper bound for the convergence rate of FFALM. The effectiveness of FFALM in improving fairness is shown empirically on CelebA and UTKFace datasets in the presence of severe statistical heterogeneity.
    
[^65]: 将关注点转移到相关性上: 探索大型语言模型的不确定性估计

    Shifting Attention to Relevance: Towards the Uncertainty Estimation of Large Language Models. (arXiv:2307.01379v1 [cs.CL])

    [http://arxiv.org/abs/2307.01379](http://arxiv.org/abs/2307.01379)

    本论文研究了大型语言模型（LLMs）自动生成的关键词不平等问题，发现在估计不确定性时，重要的令牌和含有有限语义的句子被同等或更加重视。为了解决这个问题，提出了共同转移关注点来更好地估计不确定性。

    

    虽然大型语言模型（LLMs）在自然语言生成方面表现出了巨大的潜力，但是对于模型生成的不确定性的特征化仍然具有挑战性，即用户何时可以信任模型的输出。我们的研究基于一些启发性的事实，即在自回归的LLMs中，令牌在反映生成的含义方面是不平等的，即一些令牌比其他令牌更相关（或更具代表性），然而在估计不确定性时所有的令牌被等值对待。这是由于语言冗余，其中大部分情况下，只需要几个关键词就足以传达一个长句的含义。我们将这些不平等称为生成的不平等，并研究它们如何影响不确定性的估计。我们的结果揭示，相当数量的令牌和包含有限语义的句子，在估计不确定性时被同等或甚至更加重视。为了解决由生成的不平等引起的这些偏差，我们提出了共同转移关注点来更好地估计不确定性。

    Although Large Language Models (LLMs) have shown great potential in Natural Language Generation, it is still challenging to characterize the uncertainty of model generations, i.e., when users could trust model outputs. Our research is derived from the heuristic facts that tokens are created unequally in reflecting the meaning of generations by auto-regressive LLMs, i.e., some tokens are more relevant (or representative) than others, yet all the tokens are equally valued when estimating uncertainty. It is because of the linguistic redundancy where mostly a few keywords are sufficient to convey the meaning of a long sentence. We name these inequalities as generative inequalities and investigate how they affect uncertainty estimation. Our results reveal that considerable tokens and sentences containing limited semantics are weighted equally or even heavily when estimating uncertainty. To tackle these biases posed by generative inequalities, we propose to jointly Shifting Attention to more
    
[^66]: 通过神经表面渲染，在混乱场景中学习任意视角的6DoF机器人抓取

    Learning Any-View 6DoF Robotic Grasping in Cluttered Scenes via Neural Surface Rendering. (arXiv:2306.07392v1 [cs.RO])

    [http://arxiv.org/abs/2306.07392](http://arxiv.org/abs/2306.07392)

    通过神经表面渲染，NeuGraspNet能够在混乱场景中有效地从任意视角预测6DoF抓取质量，并能够在遮挡的场景中采样抓取候选项。

    

    机器人操作在智能辅助等各种应用领域中至关重要。其中一个主要挑战是在杂乱的环境中从任何视角有效地抓取对象，而不需要额外的场景探索。我们引入了NeuGraspNet，一种新颖的6DoF抓取检测方法，利用了神经体积表示和表面渲染的最新进展。我们的方法学习了全局（场景级别）和局部（抓取级别）神经表面表示，使得即使在场景的未见部分，也能有效地预测6DoF抓取质量。此外，我们将抓取重新解释为一个局部的神经表面渲染问题，使得模型能够编码机器人末端执行器和对象表面几何之间的交互。NeuGraspNet在单个视角上运行，并且可以在遮挡的场景中采样抓取候选项，表现出优于现有隐式和半隐式基线模型的性能。

    Robotic manipulation is critical for admitting robotic agents to various application domains, like intelligent assistance. A major challenge therein is the effective 6DoF grasping of objects in cluttered environments from any viewpoint without requiring additional scene exploration. We introduce $\textit{NeuGraspNet}$, a novel method for 6DoF grasp detection that leverages recent advances in neural volumetric representations and surface rendering. Our approach learns both global (scene-level) and local (grasp-level) neural surface representations, enabling effective and fully implicit 6DoF grasp quality prediction, even in unseen parts of the scene. Further, we reinterpret grasping as a local neural surface rendering problem, allowing the model to encode the interaction between the robot's end-effector and the object's surface geometry. NeuGraspNet operates on single viewpoints and can sample grasp candidates in occluded scenes, outperforming existing implicit and semi-implicit baselin
    
[^67]: 随机坍缩：如何利用梯度噪声使SGD动态趋向更简单的子网络

    Stochastic Collapse: How Gradient Noise Attracts SGD Dynamics Towards Simpler Subnetworks. (arXiv:2306.04251v1 [cs.LG])

    [http://arxiv.org/abs/2306.04251](http://arxiv.org/abs/2306.04251)

    SGD在训练过度表达的网络时，会随机地将动态吸引到更简单的子网络，这种随机吸引性能够提高泛化能力。

    

    本文揭示了随机梯度下降（SGD）的一个强烈隐式偏好，它将过度表达的网络驱动到更简单的子网络，从而大大减少了独立参数的数量，并提高了泛化能力。为了揭示这个偏好，我们识别了不变集，或者说是SGD未修改的参数空间的子集。我们专注于两类不变集，它们对应于现代架构中常见的更简单的子网络。我们的分析揭示了SGD在这些简单不变集方面具有随机吸引性的特性。我们根据损失景观在不变集周围的曲率和随机梯度引入的噪声之间的竞争建立了一种随机吸引性的充分条件。值得注意的是，我们发现增加噪声水平会增强吸引力，导致与鞍点或训练损失的局部极大值相关的吸引不变集的出现。

    In this work, we reveal a strong implicit bias of stochastic gradient descent (SGD) that drives overly expressive networks to much simpler subnetworks, thereby dramatically reducing the number of independent parameters, and improving generalization. To reveal this bias, we identify invariant sets, or subsets of parameter space that remain unmodified by SGD. We focus on two classes of invariant sets that correspond to simpler subnetworks and commonly appear in modern architectures. Our analysis uncovers that SGD exhibits a property of stochastic attractivity towards these simpler invariant sets. We establish a sufficient condition for stochastic attractivity based on a competition between the loss landscape's curvature around the invariant set and the noise introduced by stochastic gradients. Remarkably, we find that an increased level of noise strengthens attractivity, leading to the emergence of attractive invariant sets associated with saddle-points or local maxima of the train loss.
    
[^68]: Vocos：消除时域和基于傅里叶变化的神经声码器在高质量音频合成中的差距

    Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio synthesis. (arXiv:2306.00814v2 [cs.SD] UPDATED)

    [http://arxiv.org/abs/2306.00814](http://arxiv.org/abs/2306.00814)

    Vocos是一个新的模型，通过直接生成傅里叶谱系数，消除了时域和基于傅里叶变化的神经声码器在高质量音频合成中的差距，并实现了计算效率的大幅提升。

    

    近期神经声码器的发展主要由在时域中操作的生成对抗网络（GAN）推动。虽然有效，但这种方法忽视了时频表示提供的归纳偏差，从而导致冗余和计算密集的上采样操作。基于傅里叶变换的时频表示是一个有吸引力的替代方案，它与人的听觉感知更加准确，并且通过其计算得到了经过充分验证的快速算法的好处。然而，直接重建复值谱图在历史上一直存在问题，主要是由于相位恢复问题。本研究旨在通过提出Vocos，一个直接生成傅里叶谱系数的新模型，来消除这个差距。我们的评估结果表明，Vocos不仅与音频质量的最新技术水平相匹配，而且在计算效率上实现了数量级的提升，与传统方法相比速度提高了一个数量级。

    Recent advancements in neural vocoding are predominantly driven by Generative Adversarial Networks (GANs) operating in the time-domain. While effective, this approach neglects the inductive bias offered by time-frequency representations, resulting in reduntant and computionally-intensive upsampling operations. Fourier-based time-frequency representation is an appealing alternative, aligning more accurately with human auditory perception, and benefitting from well-established fast algorithms for its computation. Nevertheless, direct reconstruction of complex-valued spectrograms has been historically problematic, primarily due to phase recovery issues. This study seeks to close this gap by presenting Vocos, a new model that directly generates Fourier spectral coefficients. Vocos not only matches the state-of-the-art in audio quality, as demonstrated in our evaluations, but it also substantially improves computational efficiency, achieving an order of magnitude increase in speed compared 
    
[^69]: 基于流导向纳米定位的设计空间探索的见解

    Insights from the Design Space Exploration of Flow-Guided Nanoscale Localization. (arXiv:2305.18493v1 [cs.NI])

    [http://arxiv.org/abs/2305.18493](http://arxiv.org/abs/2305.18493)

    研究了基于流导向纳米定位的设计空间，考虑了能源和信号衰减等因素，为这一新兴领域提供了有希望的解决方案。

    

    具有太赫兹无线通信能力的纳米设备为在人类血液中进行流导向定位提供了基础。此类定位使得将所感受到的事件的位置与事件本身进行匹配成为可能，从而实现了精准医疗方面的早期和精准诊断、降低成本和侵入性。流导向定位仍处于原始阶段，只有少数论文涉及此问题。尽管如此，所提出解决方案的性能评估仍然以非标准化的方式进行，通常只考虑单一的性能指标，并忽略了在这种规模（例如，纳米器件的能量受限）和对于这种具有挑战性的环境（例如，体内太赫兹传播的严重衰减）下相关的各个方面。因此，这些评估具有低水平的真实性，并且无法以客观的方式进行比较。为了解决这个问题，我们考虑了传输能量消耗和信号衰减，对流导向纳米定位的设计空间进行了探索。我们的分析考虑了各种性能指标（例如能量消耗和定位精度）和挑战（例如身体运动和血压），导致我们可以为这个新兴领域提供有希望的解决方案。

    Nanodevices with Terahertz (THz)-based wireless communication capabilities are providing a primer for flow-guided localization within the human bloodstreams. Such localization is allowing for assigning the locations of sensed events with the events themselves, providing benefits in precision medicine along the lines of early and precise diagnostics, and reduced costs and invasiveness. Flow-guided localization is still in a rudimentary phase, with only a handful of works targeting the problem. Nonetheless, the performance assessments of the proposed solutions are already carried out in a non-standardized way, usually along a single performance metric, and ignoring various aspects that are relevant at such a scale (e.g., nanodevices' limited energy) and for such a challenging environment (e.g., extreme attenuation of in-body THz propagation). As such, these assessments feature low levels of realism and cannot be compared in an objective way. Toward addressing this issue, we account for t
    
[^70]: 拉普拉斯逼近神经加性模型：贝叶斯推理提高解释性

    Laplace-Approximated Neural Additive Models: Improving Interpretability with Bayesian Inference. (arXiv:2305.16905v1 [stat.ML])

    [http://arxiv.org/abs/2305.16905](http://arxiv.org/abs/2305.16905)

    本文提出了拉普拉斯逼近神经加性模型，该模型从贝叶斯角度考虑加性结构，在恢复的特征交互中提供可信区间，提供可处理的边缘似然估计，可用于执行隐式特征选择并对特征对进行排名。

    

    深度神经网络（DNN）在许多领域取得了成功应用，但它们的黑盒性质阻碍了解释性。神经加性模型（NAM）解决了这个问题，将网络分为加性子网络，从而使输入特征和预测之间的交互变得明显。在本文中，我们从贝叶斯角度考虑加性结构，并开发了一个实用的拉普拉斯逼近方法。这种方法在以下三个方面提高了可解释性：a）它通过估计子网络的函数空间不确定性为恢复的特征交互提供可信区间；b）它提供可处理的边缘似然估计，可用于通过经验贝叶斯过程执行特征的隐式选择；c）它可用于对特征对进行排名，作为精细调整的交互模型候选。我们在几个基准数据集上实证表明，我们提出的拉普拉斯逼近神经加性模型（LA-NAM）提高了NAM模型的可解释性，并进一步揭示了学习到的子网络的交互结构。

    Deep neural networks (DNNs) have found successful applications in many fields, but their black-box nature hinders interpretability. This is addressed by the neural additive model (NAM), in which the network is divided into additive sub-networks, thus making apparent the interaction between input features and predictions. In this paper, we approach the additive structure from a Bayesian perspective and develop a practical Laplace approximation. This enhances interpretability in three primary ways: a) It provides credible intervals for the recovered feature interactions by estimating function-space uncertainty of the sub-networks; b) it yields a tractable estimate of the marginal likelihood, which can be used to perform an implicit selection of features through an empirical Bayes procedure; and c) it can be used to rank feature pairs as candidates for second-order interactions in fine-tuned interaction models. We show empirically that our proposed Laplace-approximated NAM (LA-NAM) improv
    
[^71]: 深思熟虑：具有内部工作记忆的决策Transformer

    Think Before You Act: Decision Transformers with Internal Working Memory. (arXiv:2305.16338v1 [cs.LG])

    [http://arxiv.org/abs/2305.16338](http://arxiv.org/abs/2305.16338)

    该论文提出了具有内部工作记忆模块的决策Transformer方法，以解决使用大型语言模型的决策代理在处理新任务上性能低下的问题。所提出的方法改善了训练效率和泛化能力，并进一步增强了转化决策制定代理对新任务的适应性。

    

    基于大型语言模型（LLM）的决策制定代理已经展示了跨越多个任务的泛化能力。然而，它们的性能依赖于大规模的数据和计算。我们认为，这种低效性源于遗忘现象，即模型通过参数记忆其行为，在训练过程中。因此，新任务的训练可能会降低模型在先前任务上的性能。与LLM的隐式记忆机制不同，人脑利用分布式存储器存储记忆，以有效地管理和组织多种技能，减轻了遗忘现象。因此，我们建议使用内部工作记忆模块来存储、融合和检索不同下游任务的信息。评估结果表明，所提出的方法改善了Atari游戏和元世界物体操作任务的训练效率和泛化能力。此外，我们证明了记忆微调进一步增强了转化决策制定代理对新任务的适应性。

    Large language model (LLM)-based decision-making agents have shown the ability to generalize across multiple tasks. However, their performance relies on massive data and compute. We argue that this inefficiency stems from the forgetting phenomenon, in which a model memorizes its behaviors in parameters throughout training. As a result, training on a new task may deteriorate the model's performance on previous tasks. In contrast to LLMs' implicit memory mechanism, the human brain utilizes distributed memory storage, which helps manage and organize multiple skills efficiently, mitigating the forgetting phenomenon. Thus inspired, we propose an internal working memory module to store, blend, and retrieve information for different downstream tasks. Evaluation results show that the proposed method improves training efficiency and generalization in both Atari games and meta-world object manipulation tasks. Moreover, we demonstrate that memory fine-tuning further enhances the adaptability of t
    
[^72]: 非对数凸和非光滑采样的 Langevin Monte Carlo 算法研究

    Non-Log-Concave and Nonsmooth Sampling via Langevin Monte Carlo Algorithms. (arXiv:2305.15988v1 [stat.ML])

    [http://arxiv.org/abs/2305.15988](http://arxiv.org/abs/2305.15988)

    本文研究了从非对数凸分布进行近似抽样的问题，并通过 Langevin Monte Carlo 算法解决。此外，研究了两种非光滑情况，这些任务源于贝叶斯推断和图像反问题。数值模拟比较了最常用的 Langevin Monte Carlo 算法的性能。

    

    本文研究了从非对数凸分布（例如高斯混合分布）进行近似抽样的问题。我们通过离散过度阻尼 Langevin 扩散所导出的马尔可夫链蒙特卡罗（MCMC）方法来解决这个问题，这些方法通常称为 Langevin Monte Carlo 算法。此外，我们还研究了两种非光滑情况，其中已经开发了大量的近端 MCMC 方法：(i) 考虑到非光滑的先验和高斯混合似然；(ii) 拉普拉斯混合分布。这样的非光滑和非对数凸采样任务源于广泛的贝叶斯推断和图像反问题，如图像反褶积中。我们进行了数值模拟以比较最常用的 Langevin Monte Carlo 算法的性能。

    We study the problem of approximate sampling from non-log-concave distributions, e.g., Gaussian mixtures, which is often challenging even in low dimensions due to their multimodality. We focus on performing this task via Markov chain Monte Carlo (MCMC) methods derived from discretizations of the overdamped Langevin diffusions, which are commonly known as Langevin Monte Carlo algorithms. Furthermore, we are also interested in two nonsmooth cases for which a large class of proximal MCMC methods have been developed: (i) a nonsmooth prior is considered with a Gaussian mixture likelihood; (ii) a Laplacian mixture distribution. Such nonsmooth and non-log-concave sampling tasks arise from a wide range of applications to Bayesian inference and imaging inverse problems such as image deconvolution. We perform numerical simulations to compare the performance of most commonly used Langevin Monte Carlo algorithms.
    
[^73]: UP5: 面向公平性推荐的无偏基础模型

    UP5: Unbiased Foundation Model for Fairness-aware Recommendation. (arXiv:2305.12090v1 [cs.IR])

    [http://arxiv.org/abs/2305.12090](http://arxiv.org/abs/2305.12090)

    本研究提出了一种新颖的基础模型UP5，它采用反事实公平促进技术来消除大型语言模型中的偏见，从而实现面向公平性的推荐。

    

    基于大型语言模型（LLM）等基础模型的最新进展，已将它们推到了推荐系统（RS）的前沿。此外，RS中的公平性很关键，因为许多用户将其用于决策和需求履行。然而，目前尚缺乏对推荐基础模型展示公平性水平和公平处理不同用户群组的适当方法的理解。本文侧重于用户方面的不公平问题，并通过彻底检查表明，LLMs中存在不公平性，导致不公平的推荐结果。为了消除LLM中的偏差以实现面向公平性的推荐，我们引入了一种基于反事实公平促进技术的新型无偏P5（UP5）基础模型。CFP包括两个子模块：个性化前缀提示和Prompt混合，从而增强了个体敏感属性的公平性。

    Recent advancements in foundation models such as large language models (LLM) have propelled them to the forefront of recommender systems (RS). Moreover, fairness in RS is critical since many users apply it for decision-making and demand fulfillment. However, at present, there is a lack of understanding regarding the level of fairness exhibited by recommendation foundation models and the appropriate methods for equitably treating different groups of users in foundation models. In this paper, we focus on user-side unfairness problem and show through a thorough examination that there is unfairness involved in LLMs that lead to unfair recommendation results. To eliminate bias from LLM for fairness-aware recommendation, we introduce a novel Unbiased P5 (UP5) foundation model based on Counterfactually-Fair-Prompting (CFP) techniques. CFP includes two sub-modules: a personalized prefix prompt that enhances fairness with respect to individual sensitive attributes, and a Prompt Mixture that int
    
[^74]: 证明正确性的物理知识神经网络

    Provably Correct Physics-Informed Neural Networks. (arXiv:2305.10157v1 [cs.LG])

    [http://arxiv.org/abs/2305.10157](http://arxiv.org/abs/2305.10157)

    该论文提出了一种名为$\partial$-CROWN的框架，以保证物理知识神经网络（PINN）具有全局正确性的最坏剩余误差，并证明了该框架在获得有效证书方面的有效性。

    

    最近的研究提供了有希望的证据表明，物理知识神经网络（PINN）可以高效地解决偏微分方程（PDE）。然而，以往的研究未能保证PINN在整个时空域内的最坏剩余误差，这是类似于数字求解器的公差的一种度量，而是集中于在一组输入上通过点对点比较来得到解决方案和求解器得到解决方案的结果。在实际应用中，不能认为在一组有限点上的测试就足以使得部署成立，因为在另一组点上性能可能大不相同。为了解决这个问题，我们建立了对整个输入域的PINN基于公差的正确性条件。为了验证它们的有效程度，我们介绍了$\partial$-CROWN：一个通用的、高效的、可扩展的、后训练框架，用于限制PINN的剩余误差。我们演示了它在获得紧密证书方面的有效性。

    Recent work provides promising evidence that Physics-informed neural networks (PINN) can efficiently solve partial differential equations (PDE). However, previous works have failed to provide guarantees on the worst-case residual error of a PINN across the spatio-temporal domain - a measure akin to the tolerance of numerical solvers - focusing instead on point-wise comparisons between their solution and the ones obtained by a solver on a set of inputs. In real-world applications, one cannot consider tests on a finite set of points to be sufficient grounds for deployment, as the performance could be substantially worse on a different set. To alleviate this issue, we establish tolerance-based correctness conditions for PINNs over the entire input domain. To verify the extent to which they hold, we introduce $\partial$-CROWN: a general, efficient and scalable post-training framework to bound PINN residual errors. We demonstrate its effectiveness in obtaining tight certificates by applying
    
[^75]: 富文本生成表达性文本图像

    Expressive Text-to-Image Generation with Rich Text. (arXiv:2304.06720v1 [cs.CV])

    [http://arxiv.org/abs/2304.06720](http://arxiv.org/abs/2304.06720)

    本文提出了一种使用富文本编辑器生成表达性文本图像的方法，可以通过局部样式控制、明确的标记重新加权、精确的颜色渲染和详细的区域合成，生成高质量且多样化的图像。

    

    纯文本已经成为文字到图像合成的流行界面。但是，它的定制选项有限，阻碍了用户精确描述所需的输出。为了解决这些挑战，我们提出使用支持字体样式、大小、颜色和脚注等格式的富文本编辑器。我们从富文本中提取每个字的属性，以启用局部样式控制、明确的标记重新加权、精确的颜色渲染和详细的区域合成。我们通过基于区域的扩散过程实现了这些功能。我们的实验表明，我们的方法可以比现有的文本到图像方法更好地生成高质量和多样化的图像。

    Plain text has become a prevalent interface for text-to-image synthesis. However, its limited customization options hinder users from accurately describing desired outputs. For example, plain text makes it hard to specify continuous quantities, such as the precise RGB color value or importance of each word. Furthermore, creating detailed text prompts for complex scenes is tedious for humans to write and challenging for text encoders to interpret. To address these challenges, we propose using a rich-text editor supporting formats such as font style, size, color, and footnote. We extract each word's attributes from rich text to enable local style control, explicit token reweighting, precise color rendering, and detailed region synthesis. We achieve these capabilities through a region-based diffusion process. We first obtain each word's region based on cross-attention maps of a vanilla diffusion process using plain text. For each region, we enforce its text attributes by creating region-s
    
[^76]: SoftED: 用于时间序列事件检测的软评估指标

    SoftED: Metrics for Soft Evaluation of Time Series Event Detection. (arXiv:2304.00439v1 [cs.LG])

    [http://arxiv.org/abs/2304.00439](http://arxiv.org/abs/2304.00439)

    SoftED metrics 是一种适用于时间序列事件检测的新指标，既包括时间的概念，又包括对相邻检测的时间容忍度，它们能够同时评估事件检测的准确性和其检测是否代表事件。

    

    时间序列事件检测方法通常通过标准的分类指标进行评估，这些指标仅关注检测准确性。然而，事件检测的不准确往往是由于前后相关事件在相邻检测中的反应产生的。这些检测对于触发必要的行动或帮助减轻不良后果非常有价值。在这种情况下，现有的指标对于事件检测来说是不充分和不适当的。因此，需要一种指标，既包括时间的概念，又包括对相邻检测的时间容忍度。本文介绍了一种新的指标集合“SoftED metrics”，旨在软评估事件检测方法。它们可以评估检测的准确性以及其检测是否代表事件。通过将事件和代表性检测相结合，并在36\%以上的实验中加入时间容忍度，提高了事件检测的评估效果。

    Time series event detection methods are evaluated mainly by standard classification metrics that focus solely on detection accuracy. However, inaccuracy in detecting an event can often result from its preceding or delayed effects reflected in neighboring detections. These detections are valuable to trigger necessary actions or help mitigate unwelcome consequences. In this context, current metrics are insufficient and inadequate for the context of event detection. There is a demand for metrics that incorporate both the concept of time and temporal tolerance for neighboring detections. This paper introduces SoftED metrics, a new set of metrics designed for soft evaluating event detection methods. They enable the evaluation of both detection accuracy and the degree to which their detections represent events. They improved event detection evaluation by associating events and their representative detections, incorporating temporal tolerance in over 36\% of experiments compared to the usual 
    
[^77]: 图上随机逆问题：分布式在线学习

    Random Inverse Problems Over Graphs: Decentralized Online Learning. (arXiv:2303.11789v1 [cs.LG])

    [http://arxiv.org/abs/2303.11789](http://arxiv.org/abs/2303.11789)

    本文提出了一种基于在线数据流的分布式在线学习算法，将希尔伯特空间中的分布参数估计和再生核希尔伯特空间中的最小均方问题统一起来，并发展了一种新的L2-渐近稳定性理论。该算法在网络图为连通且正向算子序列满足无限维度时空励磁条件的情况下，能够实现均方和几乎必然的强一致估计。

    

    我们建立了一个随机逆问题的框架，该问题具有实时的图上观测，并提出了一种基于在线数据流的分布式在线学习算法，将希尔伯特空间中的分布参数估计和再生核希尔伯特空间中的最小均方问题统一起来。我们将算法收敛性转化为带有L2有界鞅差分项的希尔伯特空间中随机时变差分方程的渐近稳定性，并发展了L2-渐近稳定性理论。结果表明，如果网络图是连通的，并且正向算子序列满足无限维度时空励磁条件，则所有节点的估计均为均方和几乎必然的强一致的。通过将RKHS中的分布式学习问题等效地转化为图上随机逆问题，我们提出了一种基于无中心节点的RKHS分布式在线学习算法。

    We establish a framework of random inverse problems with real-time observations over graphs, and present a decentralized online learning algorithm based on online data streams, which unifies the distributed parameter estimation in Hilbert space and the least mean square problem in reproducing kernel Hilbert space (RKHS-LMS). We transform the algorithm convergence into the asymptotic stability of randomly time-varying difference equations in Hilbert space with L2-bounded martingale difference terms and develop the L2 -asymptotic stability theory. It is shown that if the network graph is connected and the sequence of forward operators satisfies the infinitedimensional spatio-temporal persistence of excitation condition, then the estimates of all nodes are mean square and almost surely strongly consistent. By equivalently transferring the distributed learning problem in RKHS to the random inverse problem over graphs, we propose a decentralized online learning algorithm in RKHS based on no
    
[^78]: 隐私估计中基于子集的实例优化

    Subset-Based Instance Optimality in Private Estimation. (arXiv:2303.01262v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2303.01262](http://arxiv.org/abs/2303.01262)

    本论文提出了一个新的定义来评估差分隐私估计算法的实例优化。我们的定义要求最优算法与一个最佳的已知数据集并在其较大的子集上进行性能评估相竞争，从而使基准算法比以前的工作更强大。我们还展示了在实值数据集上如何构建能够实现实例优化的隐私算法，并对均值进行了详细分析，证明我们的算法在估计一类广泛的数据集属性时能达到或超过渐近性能。

    

    我们提出了一个针对差分隐私估计算法的实例优化的新定义。我们的定义要求最优算法在每个数据集$D$上都与最佳的已知$D$并以最坏情况性能对$D$的大子集进行评估的隐私基准算法同时竞争。也就是说，基准算法在潜在的极端点被添加到$D$时可能表现不好；它只需要处理删除已经存在的一小部分真实数据点。这使得我们的基准算法比之前提出的基准算法显著更强大。尽管如此，我们仍然展示了对于实值数据集，如何构造能够实现我们的实例优化概念的隐私算法，以估计包括均值、分位数和$\ell_p$-范数最小化器在内的广泛类别的数据集属性。特别是对于均值，我们提供了一种详细分析，并展示了我们的算法同时匹配或超过了渐近的p

    We propose a new definition of instance optimality for differentially private estimation algorithms. Our definition requires an optimal algorithm to compete, simultaneously for every dataset $D$, with the best private benchmark algorithm that (a) knows $D$ in advance and (b) is evaluated by its worst-case performance on large subsets of $D$. That is, the benchmark algorithm need not perform well when potentially extreme points are added to $D$; it only has to handle the removal of a small number of real data points that already exist. This makes our benchmark significantly stronger than those proposed in prior work. We nevertheless show, for real-valued datasets, how to construct private algorithms that achieve our notion of instance optimality when estimating a broad class of dataset properties, including means, quantiles, and $\ell_p$-norm minimizers. For means in particular, we provide a detailed analysis and show that our algorithm simultaneously matches or exceeds the asymptotic p
    
[^79]: PARIS：用于改善睡眠质量的个性化活动推荐系统

    PARIS: Personalized Activity Recommendation for Improving Sleep Quality. (arXiv:2110.13745v1 [cs.LG] CROSS LISTED)

    [http://arxiv.org/abs/2110.13745](http://arxiv.org/abs/2110.13745)

    该论文利用机器学习技术，结合可穿戴设备监测的数据，通过时间序列聚类找到与指定主题相关的行为模型并生成相应的睡眠质量活动建议，为提高睡眠质量提供了一种个性化解决方案。

    

    睡眠质量对人们的身体和心理健康有深远影响。睡眠不足的人更容易报告身体和心理困扰、活动受限、焦虑和疼痛。此外，过去几年中，活动监测和健康跟踪的应用和设备方兴未艾。从这些可穿戴设备收集到的信号可用于研究和改善睡眠质量。本文利用实体活动和睡眠质量之间的关系，利用机器学习技术找到协助人们改善睡眠的方法。对活动数据进行时间序列聚类，我们找到与特定主题最显着的行为模式相关的簇中心。然后为每个行为模式中的每个簇生成有助于良好睡眠质量的活动建议。这些活动建议供应给每位用户的个性化活动推荐系统。

    The quality of sleep has a deep impact on people's physical and mental health. People with insufficient sleep are more likely to report physical and mental distress, activity limitation, anxiety, and pain. Moreover, in the past few years, there has been an explosion of applications and devices for activity monitoring and health tracking. Signals collected from these wearable devices can be used to study and improve sleep quality. In this paper, we utilize the relationship between physical activity and sleep quality to find ways of assisting people improve their sleep using machine learning techniques. People usually have several behavior modes that their bio-functions can be divided into. Performing time series clustering on activity data, we find cluster centers that would correlate to the most evident behavior modes for a specific subject. Activity recipes are then generated for good sleep quality for each behavior mode within each cluster. These activity recipes are supplied to an a
    
[^80]: MRCpy：一种用于最小化风险分类器的库

    MRCpy: A Library for Minimax Risk Classifiers. (arXiv:2108.01952v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2108.01952](http://arxiv.org/abs/2108.01952)

    MRCpy是一种用于实现最小化风险分类器的Python库，它基于鲁棒风险最小化技术，可以利用0-1损失并提供了多种分类方法，其中一些提供了紧密的期望损失界限。

    

    目前现有的监督分类库都是基于经验风险最小化和使用代理损失技术的。本文介绍MRCpy库，该库实现了基于鲁棒风险最小化的最小化风险分类器（MRC），并可利用0-1损失。这种技术产生了许多分类方法，可以提供紧密的期望损失界限。MRCpy为不同变量的MRC提供了统一的接口，并遵循流行Python库的标准。此外，MRCpy还提供了实现一些流行技术的功能，这些技术可以看作是MRC，例如L1正则化逻辑回归，0-1对抗性和最大熵机。此外，MRCpy还实现了最近的特征映射，如傅里叶，ReLU和阈值特征。该库采用面向对象的方法设计，方便协作者和用户。

    Existing libraries for supervised classification implement techniques that are based on empirical risk minimization and utilize surrogate losses. We present MRCpy library that implements minimax risk classifiers (MRCs) that are based on robust risk minimization and can utilize 0-1-loss. Such techniques give rise to a manifold of classification methods that can provide tight bounds on the expected loss. MRCpy provides a unified interface for different variants of MRCs and follows the standards of popular Python libraries. The presented library also provides implementation for popular techniques that can be seen as MRCs such as L1-regularized logistic regression, zero-one adversarial, and maximum entropy machines. In addition, MRCpy implements recent feature mappings such as Fourier, ReLU, and threshold features. The library is designed with an object-oriented approach that facilitates collaborators and users.
    

