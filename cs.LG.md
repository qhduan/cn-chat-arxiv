# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Hierarchical State Space Models for Continuous Sequence-to-Sequence Modeling](https://arxiv.org/abs/2402.10211) | 分层状态空间模型（HiSS）是一种针对连续序列到序列建模的技术，它利用堆叠的结构化状态空间模型来进行预测。 |
| [^2] | [Self-Play Fine-Tuning of Diffusion Models for Text-to-Image Generation](https://arxiv.org/abs/2402.10210) | 本文介绍了一种创新的技术，称为自我对抗微调扩散模型（SPIN-Diffusion），通过扩散模型与其先前版本的竞争，实现了逐步自我改进过程。 |
| [^3] | [Recovering the Pre-Fine-Tuning Weights of Generative Models](https://arxiv.org/abs/2402.10208) | 该论文提出了一种恢复生成模型预微调权重的方法，通过少量低秩微调模型可以恢复准确的预微调权重，利用这个新漏洞攻击大规模模型。 |
| [^4] | [Rewards-in-Context: Multi-objective Alignment of Foundation Models with Dynamic Preference Adjustment](https://arxiv.org/abs/2402.10207) | 本文介绍了Rewards-in-Context（RiC）方法，该方法通过多个奖励条件控制基础模型的响应，并应用有监督的微调进行对齐。它具有简单性和适应性，并支持在推理时动态调整用户偏好。 |
| [^5] | [Ising on the Graph: Task-specific Graph Subsampling via the Ising Model](https://arxiv.org/abs/2402.10206) | 该论文提出了一种基于伊辛模型的图子抽样方法，可以针对特定任务在图结构上进行减小，并通过学习伊辛模型的外部磁场来实现。该方法的多功能性在图像分割、三维形状稀疏化和稀疏逼近矩阵求逆等应用中得到展示。 |
| [^6] | [Bridging Associative Memory and Probabilistic Modeling](https://arxiv.org/abs/2402.10202) | 基于联想记忆的能量函数可以被视为概率建模的对数似然函数，这篇论文构建了两者之间的桥梁，提出了新的基于能量的模型，并展示了两种新的联想记忆模型，可灵活适应上下文数据集。 |
| [^7] | [Unlocking the Potential of Transformers in Time Series Forecasting with Sharpness-Aware Minimization and Channel-Wise Attention](https://arxiv.org/abs/2402.10198) | 本文研究了Transformer在时间序列预测中的局限性，发现其注意力机制是泛化能力不足的原因。在此基础上，提出了一个浅层轻量级的Transformer模型SAMformer，通过锐度感知优化避免了陷入坏的局部最小值，并在常用时间序列数据集上超过了当前最先进的模型TSMixer。 |
| [^8] | [BitDelta: Your Fine-Tune May Only Be Worth One Bit](https://arxiv.org/abs/2402.10193) | BitDelta研究探讨了大型语言模型在微调过程中的信息冗余性，并提出了一种名为BitDelta的方法，可以将微调过程中添加的信息量化为一个比特，同时保持性能。这一发现对于多租户模型的服务和存储有重要意义，并可以显著降低GPU内存需求。 |
| [^9] | [Multi-Excitation Projective Simulation with a Many-Body Physics Inspired Inductive Bias](https://arxiv.org/abs/2402.10192) | 该论文引入了多激发投影模拟（mePS），通过在超图上多个粒子的随机游走，解决了投影模拟（PS）无法模拟同时结合多个概念的思维的问题。 |
| [^10] | [FedAnchor: Enhancing Federated Semi-Supervised Learning with Label Contrastive Loss for Unlabeled Clients](https://arxiv.org/abs/2402.10191) | 本文介绍了一种名为FedAnchor的创新方法，通过引入anchor head和标签对比损失，增强了无标签客户端的联邦半监督学习。 |
| [^11] | [Uncertainty Decomposition and Quantification for In-Context Learning of Large Language Models](https://arxiv.org/abs/2402.10189) | 本文研究了大型语言模型（LLM）上下文学习中的不确定性，并提出了一种新的方法来量化这种不确定性，包括演示产生的不确定性和模型配置的模糊性。 |
| [^12] | [Self-consistent Validation for Machine Learning Electronic Structure](https://arxiv.org/abs/2402.10186) | 提出了一种自洽验证机器学习电子结构的技术，该技术通过将机器学习与自洽场方法结合起来，实现了低验证成本和强解释能力，并能通过主动学习来探索模型的能力。 |
| [^13] | [Rethinking Information Structures in RLHF: Reward Generalization from a Graph Theory Perspective](https://arxiv.org/abs/2402.10184) | 本研究通过设计奖励建模过程中的数据集信息结构，从图论的视角提出了RLHF中奖励泛化的问题，以解决多样的环境、低成本标注和可靠的对齐性能间的不兼容性。 |
| [^14] | [Large Scale Constrained Clustering With Reinforcement Learning](https://arxiv.org/abs/2402.10177) | 本文介绍了一种使用强化学习解决大规模受限制聚类问题的方法，该方法训练一个代理器生成既可行又接近最优解的解决方案，以提高资源分配和使用的效率。 |
| [^15] | [OpenMathInstruct-1: A 1.8 Million Math Instruction Tuning Dataset](https://arxiv.org/abs/2402.10176) | OpenMathInstruct-1是一个包含180万个数学问题和解决方法对的数据集，通过合成开源LLM的代码解释器解决方案来构建，填补了目前开源LLM在数学技能方面与闭源LLM之间的差距。 |
| [^16] | [DeepSRGM -- Sequence Classification and Ranking in Indian Classical Music with Deep Learning](https://arxiv.org/abs/2402.10168) | DeepSRGM是一种基于深度学习的Raga识别方法，通过使用LSTM-RNN学习音乐数据中的时间序列，达到了88.1%和97%的准确率，在Raga识别任务中取得了最新技术的地位。 |
| [^17] | [Random features and polynomial rules](https://arxiv.org/abs/2402.10164) | 本论文分析了随机特征模型在具有高斯数据的一般监督学习问题中的泛化性能，并将随机特征模型映射到等效的多项式模型，得到了与严格界限和数值实验一致的结果。 |
| [^18] | [$f$-MICL: Understanding and Generalizing InfoNCE-based Contrastive Learning](https://arxiv.org/abs/2402.10150) | 本文提出了一种名为$f$-MICL的方法，用于理解和推广基于InfoNCE的对比学习。通过使用$f$-divergences将基于KL的互信息推广为$f$-Mutual Information in Contrastive Learning ($f$-MICL)，我们回答了超越基于KL的目标函数以及设计更好相似度函数的问题。 |
| [^19] | [A chaotic maps-based privacy-preserving distributed deep learning for incomplete and Non-IID datasets](https://arxiv.org/abs/2402.10145) | 本研究提出了一种基于混沌映射的保护隐私的分布式深度学习方法，在处理不完整和非独立同分布数据集的情况下，通过差分隐私和基于混沌加密的隐私保护层提高了深度神经网络的性能。 |
| [^20] | [Tracking Changing Probabilities via Dynamic Learners](https://arxiv.org/abs/2402.10142) | 该论文介绍了通过动态学习器追踪概率变化的方法，通过输出候选项目及其概率来预测离散项目序列中下一个可能出现的项目。 |
| [^21] | [Benchmarking federated strategies in Peer-to-Peer Federated learning for biomedical data](https://arxiv.org/abs/2402.10135) | 这项研究对生物医学数据的对等联邦学习进行了基准测试，并测试了各种聚合策略，包括加权平均聚合，以确定最强大的策略。 |
| [^22] | [Is Continual Learning Ready for Real-world Challenges?](https://arxiv.org/abs/2402.10130) | 本文研究了连续学习在现实世界场景中的应用，发现当前的评估方法与实际挑战不匹配，现有解决方案无法有效解决复杂的现实世界环境下的问题。 |
| [^23] | [GES: Generalized Exponential Splatting for Efficient Radiance Field Rendering](https://arxiv.org/abs/2402.10128) | 本研究引入了GES（广义指数喷洒），一种利用广义指数函数来建模3D场景的新表示方法，可以显著提高3D重建和生成的效率。相比于传统的高斯喷洒方法，GES所需的粒子数量更少，能更准确地表示具有锐利边缘的信号。 |
| [^24] | [Nonlinear spiked covariance matrices and signal propagation in deep neural networks](https://arxiv.org/abs/2402.10127) | 该论文研究了非线性尖峰协方差矩阵与深度神经网络中的信号传播。通过对尖峰特征结构的定量描述，揭示了输入数据中的低维信号结构如何经过神经网络的隐藏层传播。此外，研究了一种表示学习的简单情境，其中权重矩阵发展出一个秩为一的信号分量。 |
| [^25] | [Reusing Softmax Hardware Unit for GELU Computation in Transformers](https://arxiv.org/abs/2402.10118) | 本文提出了一种在Transformer中重用Softmax硬件单元进行GELU计算的方法，实验证明这种方法不会降低NLP应用的准确性。 |
| [^26] | [Generating Visual Stimuli from EEG Recordings using Transformer-encoder based EEG encoder and GAN](https://arxiv.org/abs/2402.10115) | 本研究使用基于Transformer编码器的EEG编码器和GAN网络，通过合成图像从EEG信号中恢复出各种对象类别的图像，同时结合对抗损失和感知损失，提高生成图像的质量。 |
| [^27] | [Selective Reflection-Tuning: Student-Selected Data Recycling for LLM Instruction-Tuning](https://arxiv.org/abs/2402.10110) | 本文介绍了一种名为选择性反射调节的新方法，该方法通过教师LLM的反射和自省与学生LLM的数据选择能力相结合，自动优化现有的指令调节数据，从而实现了高效的指令调节和卓越性能的LLM。 |
| [^28] | [Towards Reducing Diagnostic Errors with Interpretable Risk Prediction](https://arxiv.org/abs/2402.10109) | 本研究提出了一种使用LLMs方法来识别病人电子病历数据中指示特定诊断风险增加或减少的证据的方法，旨在通过增加证据的获取与减少诊断错误来降低诊断错误。模型使用神经加性模型进行预测，以证据为后盾，并给出个体化风险估计，特别针对诊断延迟和来自不完整鉴别的错误进行优化。 |
| [^29] | [Deep Learning Based Situation Awareness for Multiple Missiles Evasion](https://arxiv.org/abs/2402.10101) | 该研究提出了一种基于深度学习的决策支持工具，用于帮助无人机操作员在多个导弹威胁下进行决策，通过学习高保真度模拟来评估各种策略的风险，并建议最安全的行动方针。 |
| [^30] | [Tuning In: Analysis of Audio Classifier Performance in Clinical Settings with Limited Data](https://arxiv.org/abs/2402.10100) | 本研究评估了在临床设置中使用深度学习模型进行音频分类的效果，并发现在微调之前，预训练模型在大数据集上的性能对临床数据的影响较好。研究结果表明，CNN模型可以在小数据集环境中与转换模型相媲美或超越。 |
| [^31] | [Parameter-tuning-free data entry error unlearning with adaptive selective synaptic dampening](https://arxiv.org/abs/2402.10098) | 本研究提出了一种不需要参数调整的数据输入错误unlearning方法，通过自适应选择性突触抑制（ASSD）提高模型性能，并展示了其在不同模型上的性能表现。 |
| [^32] | [Adaptive Federated Learning in Heterogeneous Wireless Networks with Independent Sampling](https://arxiv.org/abs/2402.10097) | 这项研究提出了一种适用于异构无线网络的自适应联邦学习方法，其中包括了独立客户端采样和带宽分配方案，以提高训练效率和适应数据和系统的异构特性。 |
| [^33] | [Classification Diffusion Models](https://arxiv.org/abs/2402.10095) | 提出了一种分类扩散模型（CDMs），该模型采用了去噪扩散模型（DDM）的形式，并利用一个分类器来预测加在干净信号上的噪声量，取得了在图像、视频和音频生成方面的最先进结果。 |
| [^34] | [MIM-Refiner: A Contrastive Learning Boost from Intermediate Pre-Trained Representations](https://arxiv.org/abs/2402.10093) | MIM-Refiner是一种对比学习提升方法，通过利用MIM模型中的中间层表示和多个对比头，能够将MIM模型的特征从次优的状态提升到最先进的状态，并在ImageNet-1K数据集上取得了新的最先进结果。 |
| [^35] | [Workflow Optimization for Parallel Split Learning](https://arxiv.org/abs/2402.10092) | 本文提出了一种并行分割学习的工作流优化方法，旨在最小化训练时间，通过将问题分解成客户-辅助器分配和调度决策的联合问题进行求解。 |
| [^36] | [Text-Based Product Matching -- Semi-Supervised Clustering Approach](https://arxiv.org/abs/2402.10091) | 本文介绍了一种利用半监督聚类方法进行产品匹配的新思路，并通过实验证明了无监督匹配与少量注释样本的产品链接可以成为主导的监督策略的替代方法。 |
| [^37] | [PICS: Pipeline for Image Captioning and Search](https://arxiv.org/abs/2402.10090) | PICS是一种用于图像描述和搜索的流水线，它利用了大型语言模型的进展来自动化图像描述的过程，并通过集成情感分析来增强元数据，从而提高了大规模图像库的搜索效率和访问性。 |
| [^38] | [Hierarchical hybrid modeling for flexible tool use](https://arxiv.org/abs/2402.10088) | 本研究基于主动推理计算框架，提出了一个分层混合模型，通过组合离散和连续模型以实现灵活工具使用，控制和规划。在非平凡任务中验证了该模型的有效性和可扩展性。 |
| [^39] | [Decentralized Covert Routing in Heterogeneous Networks Using Reinforcement Learning](https://arxiv.org/abs/2402.10087) | 本文提出了一个基于强化学习的新型隐秘路由算法，在异构网络中实现了分散的隐秘路由通信，仅使用本地反馈信息确定每个节点的下一跳和通信模式。在数值模拟中表明，该策略与最优集中式路由方案相比，性能损耗可忽略。 |
| [^40] | [Explainable AI for Safe and Trustworthy Autonomous Driving: A Systematic Review](https://arxiv.org/abs/2402.10086) | 可解释的AI技术对于解决自动驾驶中的安全问题和信任问题至关重要。本文通过系统文献综述的方式，分析了可解释的AI方法在满足自动驾驶要求方面的关键贡献，并提出了可解释的设计、可解释的替代模型、可解释的监控、辅助技术和解释的可视化等五个方面的应用。 |
| [^41] | [Develop End-to-End Anomaly Detection System](https://arxiv.org/abs/2402.10085) | 本文提出了一个端到端的异常检测模型开发流程，可以解决网络中异常检测的挑战，并通过引入新的预测模型"Lachesis"来展示其有效性。 |
| [^42] | [FedRDF: A Robust and Dynamic Aggregation Function against Poisoning Attacks in Federated Learning](https://arxiv.org/abs/2402.10082) | FedRDF提出了一种新颖的鲁棒聚合机制，利用傅里叶变换来有效处理联邦学习中的复杂攻击，而不需要先验知识。 |
| [^43] | [Review of the Learning-based Camera and Lidar Simulation Methods for Autonomous Driving Systems](https://arxiv.org/abs/2402.10079) | 本文综述了自主驾驶系统中基于学习的相机和激光雷达仿真方法的最新研究现状。 |
| [^44] | [EventF2S: Asynchronous and Sparse Spiking AER Framework using Neuromorphic-Friendly Algorithm](https://arxiv.org/abs/2402.10078) | 这项研究提出了一种基于神经形态的AER-SNN对象识别解决方案，集成了异步处理、神经形态兼容性和稀疏尖峰的特性，在资源受限的应用中具有重要意义。 |
| [^45] | [Towards a large-scale fused and labeled dataset of human pose while interacting with robots in shared urban areas](https://arxiv.org/abs/2402.10077) | 本研究通过融合和标记两个数据集，MOT17和NCLT，填补了共享城市区域中人机交互的人体姿势数据集的空白。 |
| [^46] | [QUICK: Quantization-aware Interleaving and Conflict-free Kernel for efficient LLM inference](https://arxiv.org/abs/2402.10076) | QUICK是一组针对量化大语言模型（LLMs）的高效推理的优化CUDA内核。通过解决共享内存冲突问题和交错量化权重矩阵，QUICK实现了显著的速度提升和吞吐量增益。 |
| [^47] | [GraphCBAL: Class-Balanced Active Learning for Graph Neural Networks via Reinforcement Learning](https://arxiv.org/abs/2402.10074) | 本文提出了一种通过强化学习对图神经网络进行类平衡主动学习的框架GraphCBAL，该框架能够学习一种最佳策略，选择类平衡和信息丰富的节点进行注释，以最大化GNNs性能。 |
| [^48] | [Deep Joint Source-Channel Coding for Efficient and Reliable Cross-Technology Communication](https://arxiv.org/abs/2402.10072) | 本文提出了一个深度联合源信道编码(DJSCC)方案，用于实现高效可靠的跨技术通信(CTC)。该方案利用神经网络构建编码器和解码器，同时实现信息压缩和语义含义稳健性。 |
| [^49] | [Learning fast changing slow in spiking neural networks](https://arxiv.org/abs/2402.10069) | 通过近端策略优化实现的生物学可行方法在脉冲神经网络中减轻了强化学习所面临的数据稀缺性和噪声引入的困难。 |
| [^50] | [LLM-based policy generation for intent-based management of applications](https://arxiv.org/abs/2402.10067) | 这项研究提出了基于LLM的策略生成方法，用于实现自动化的应用意图管理。通过生成逐步分解意图所需的动作，并将其映射到API，实现了闭控制循环来自动化策略执行。 |
| [^51] | [NYCTALE: Neuro-Evidence Transformer for Adaptive and Personalized Lung Nodule Invasiveness Prediction](https://arxiv.org/abs/2402.10066) | NYCTALE是一种神经启发的Transformer架构，用于自适应个性化肺结节侵袭性预测。与传统的CT基深度学习模型不同，NYCTALE仅在累积足够数量的证据时才进行预测。 |
| [^52] | [How Much Does Each Datapoint Leak Your Privacy? Quantifying the Per-datum Membership Leakage](https://arxiv.org/abs/2402.10065) | 本论文研究了每个数据点的成员推断攻击，量化了每个数据点的成员泄露，并评估了两种隐私防御措施的效果。 |
| [^53] | [Navigating the Maize: Cyclic and conditional computational graphs for molecular simulation](https://arxiv.org/abs/2402.10064) | 该论文介绍了一种用于分子模拟的循环和条件计算图的工作流管理器，通过并行化和通信实现任意图结构的执行，具有很高的实用性和效果。 |
| [^54] | [Balancing the Causal Effects in Class-Incremental Learning](https://arxiv.org/abs/2402.10063) | 论文揭示了类增量学习中新旧数据之间的不平衡因果效应，并提出了一种平衡因果效应的方法来缓解这个问题。 |
| [^55] | [Optimal Parameter and Neuron Pruning for Out-of-Distribution Detection](https://arxiv.org/abs/2402.10062) | 提出了一种用于识别未知分布的最优参数和神经元剪枝方法（OPNP），通过评估模型参数和神经元的敏感性来解决OOD检测的问题。 |
| [^56] | [How Flawed is ECE? An Analysis via Logit Smoothing](https://arxiv.org/abs/2402.10046) | 本研究通过分析对数平滑，探讨了ECE的缺陷以及对现有结果的影响，并提出了一种新的连续、易于估计的误差测度LS-ECE。通过实验发现，LS-ECE与分箱ECE非常接近。 |
| [^57] | [Short-Form Videos and Mental Health: A Knowledge-Guided Multimodal Neural Topic Model](https://arxiv.org/abs/2402.10045) | 这项研究针对短视频对观众心理健康的抑郁影响问题，开发了一种基于医学知识的多模态神经主题模型，以预测其影响并采取相应的干预措施。 |
| [^58] | [How to validate average calibration for machine learning regression tasks ?](https://arxiv.org/abs/2402.10043) | 本文提出了两种验证机器学习回归任务平均校准性的方法，将校准误差与平均绝对误差之间的差值和将平均平方z-分数与1进行比较。研究发现，前者对不确定性分布敏感，而后者在该方面提供了最可靠的方法。 |
| [^59] | [RS-DPO: A Hybrid Rejection Sampling and Direct Preference Optimization Method for Alignment of Large Language Models](https://arxiv.org/abs/2402.10038) | 本研究提出了一种名为RS-DPO的方法，它将拒绝采样和直接优化偏好结合起来，用于对齐大型语言模型。通过开发一个经过监督微调的策略模型，并从该模型中直接采样响应，RS-DPO能够有效解决基于近端策略优化的不稳定性和高计算成本的问题。通过识别对比样本对，RS-DPO能够更好地进行RLHF。 |
| [^60] | [Predictive Linear Online Tracking for Unknown Targets](https://arxiv.org/abs/2402.10036) | 本文提出了一种名为预测性线性在线追踪（PLOT）的算法，用于在线追踪未知目标。该算法使用具有指数遗忘的递归最小二乘法来学习目标的时变动态模型，并在递推视线控制的框架下使用所学模型进行最优策略。与先前的工作不同，我们的理论结果适用于非平稳目标。 |
| [^61] | [Diffusion Models Meet Contextual Bandits with Large Action Spaces](https://arxiv.org/abs/2402.10028) | 本文设计了一种利用预训练扩散模型的扩散汤普森采样方法，用于在大动作空间下进行高效的情境强化学习探索。实证评估结果表明了该方法的优越性能。 |
| [^62] | [Self-Augmented In-Context Learning for Unsupervised Word Translation](https://arxiv.org/abs/2402.10024) | 通过自学习上下文增强方法，本论文提出一种无监督词汇翻译的方法，在零样本提示的大型语言模型上取得了显著的改进，超过了传统基于映射的方法。 |
| [^63] | [Zero-Shot Unsupervised and Text-Based Audio Editing Using DDPM Inversion](https://arxiv.org/abs/2402.10009) | 本文研究了使用DDPM反转进行音频信号的零样本编辑技术，包括基于文本的编辑和无监督发现编辑方向。这些方法在音乐信号中展现了多样的音乐兴趣修改。 |
| [^64] | [ML-ASPA: A Contemplation of Machine Learning-based Acoustic Signal Processing Analysis for Sounds, & Strains Emerging Technology](https://arxiv.org/abs/2402.10005) | 本文研究了机器学习在声学信号处理分析中的应用，通过数据驱动的方法，揭示了复杂声学现象的模型。 |
| [^65] | [Privacy Attacks in Decentralized Learning](https://arxiv.org/abs/2402.10001) | 该论文介绍了分布式学习中的隐私攻击，针对分布式梯度下降（D-GD）提出了首个攻击方法，能够使用户重建其邻域之外其他用户的私有数据，并验证了这种攻击的有效性。 |
| [^66] | [Risk-Sensitive Soft Actor-Critic for Robust Deep Reinforcement Learning under Distribution Shifts](https://arxiv.org/abs/2402.09992) | 本论文研究了在运营研究领域中，深度强化学习算法在面对分布偏移时的鲁棒性。通过推导出一种风险敏感的深度强化学习算法，并通过数值证据验证其有效性，填补了这一领域实际性能研究的空白。 |
| [^67] | [TIAViz: A Browser-based Visualization Tool for Computational Pathology Models](https://arxiv.org/abs/2402.09990) | TIAViz是一种基于浏览器的计算病理学模型可视化工具，可以灵活、交互式地显示图表、热图、分割、标注和其他信息在整个切片图像上。 |
| [^68] | [Symmetry-Breaking Augmentations for Ad Hoc Teamwork](https://arxiv.org/abs/2402.09984) | 本研究提出了一种称为对称破缺增强的方法，通过增加训练队友的行为多样性来提高人工智能代理与新队友合作的性能。实验证明了该方法的有效性。 |
| [^69] | [Data Augmentation and Transfer Learning Approaches Applied to Facial Expressions Recognition](https://arxiv.org/abs/2402.09982) | 本文提出了一种改进面部表情识别的新型数据增强技术，并应用迁移学习方法，通过使用预训练卷积神经网络在增强的数据集上进行微调，实现了高达85%的平均准确度。 |
| [^70] | [Deep learning for the design of non-Hermitian topolectrical circuits](https://arxiv.org/abs/2402.09978) | 用深度学习设计非厄米拓电路，通过多层感知器和卷积神经网络预测非厄米哈密顿量的本征值，利用DenseNet算法设计高维拓扑电路，证明了深度学习网络在捕捉全局拓扑特性方面的有效性。 |
| [^71] | [Fast Vocabulary Transfer for Language Model Compression](https://arxiv.org/abs/2402.09977) | 提出了一种基于词汇转移的语言模型压缩方法，通过与其他压缩技术结合使用，显著减小模型大小和推理时间，同时性能略有妥协。 |
| [^72] | [Accelerating Parallel Sampling of Diffusion Models](https://arxiv.org/abs/2402.09970) | 本文提出了一种并行化自回归过程来加速扩散模型的采样的方法，并引入了ParaTAA，一种通用的并行采样算法，可以显著减少推理步骤。 |
| [^73] | [Hierarchy Representation of Data in Machine Learnings](https://arxiv.org/abs/2402.09965) | 该论文提出了一种用于可视化目标间层次关系的方法，对于模型改进具有潜在的益处。 |
| [^74] | [Why are Sensitive Functions Hard for Transformers?](https://arxiv.org/abs/2402.09963) | 本文证明了在Transformer架构下，损失函数的空间受到输入敏感性的限制，从而解释了Transformer对敏感函数的困难。这一理论统一了关于Transformer学习能力和偏见的广泛观察。 |
| [^75] | [Enhancing Courier Scheduling in Crowdsourced Last-Mile Delivery through Dynamic Shift Extensions: A Deep Reinforcement Learning Approach](https://arxiv.org/abs/2402.09961) | 本研究提出了一种通过对离线计划进行动态调整来优化众包最后一公里配送中快递员调度的方法。研究采用了深度强化学习算法，旨在最大化众包平台的利润。 |
| [^76] | [On Designing Features for Condition Monitoring of Rotating Machines](https://arxiv.org/abs/2402.09957) | 本文提出了一种新的算法，通过直方图理论设计出适用于不同时间序列传感器数据的输入特征抽取方法，为机械状态识别提供了一种统一的特征提取过程。 |
| [^77] | [Crafting a Good Prompt or Providing Exemplary Dialogues? A Study of In-Context Learning for Persona-based Dialogue Generation](https://arxiv.org/abs/2402.09954) | 本研究通过对大型语言模型在基于角色生成对话方面进行实验，发现调整提示指令可以最直接有效且经济地提高生成质量，并且随机检索示范会取得最佳结果，而查询相同上下文的示范检索效果最差。即使破坏了示范中的多回合关联和单回合语义，对话生成仍然有效。 |
| [^78] | [Multi-Word Tokenization for Sequence Compression](https://arxiv.org/abs/2402.09949) | 本论文介绍了一种名为MWT的多词标记器，通过将频繁出现的多词表达式表示为单个标记，突破了词边界的限制，从而实现更紧凑和高效的标记化，提高了性能并加速推理过程。 |
| [^79] | [Neural 5G Indoor Localization with IMU Supervision](https://arxiv.org/abs/2402.09948) | 本论文研究了带有IMU监督的神经网络5G室内定位问题，提出了基于IMU的伪标签和实用算法，并展示了与完全监督方法相当的性能。 |
| [^80] | [Explaining Probabilistic Models with Distributional Values](https://arxiv.org/abs/2402.09947) | 本文介绍了一种用于解释概率模型的方法，通过引入分布值来解决当前方法在解释模型输出时的不匹配问题，并通过案例研究展示了该方法提供的详细和有洞察力的解释。 |
| [^81] | [FedLion: Faster Adaptive Federated Optimization with Fewer Communication](https://arxiv.org/abs/2402.09941) | FedLion是一种自适应联邦优化算法，通过引入集中式自适应算法Lion的关键元素，实现了更快的收敛速度和更少的通信成本。经过广泛评估，FedLion优于之前的最先进自适应算法，并通过使用有符号梯度在本地训练中减少数据传输要求。 |
| [^82] | [Generative AI in the Construction Industry: A State-of-the-art Analysis](https://arxiv.org/abs/2402.09939) | 本研究通过分析提供了建筑行业中生成式AI的最新状态、机遇和挑战。同时，提出了一个帮助建筑公司构建定制化生成式AI解决方案的框架。 |
| [^83] | [BUSTER: a "BUSiness Transaction Entity Recognition" dataset](https://arxiv.org/abs/2402.09916) | BUSTER是一个商业交易实体识别的数据集，其中包含了3779份手动标注的金融交易文档，并建立了几个基准模型。最佳模型还用于自动标注6196份文档，并作为额外的银标准数据集发布。 |
| [^84] | [DE-COP: Detecting Copyrighted Content in Language Models Training Data](https://arxiv.org/abs/2402.09910) | DE-COP是一种用于检测语言模型训练数据中版权内容的方法，通过对语言模型进行多项选择探测，可以识别出模型训练文本中可能包含的版权内容。该方法在模型的逻辑可用时比之前的方法提高了9.6%的检测性能，并在完全黑盒模型上实现了72%的准确率。 |
| [^85] | [Generative Representational Instruction Tuning](https://arxiv.org/abs/2402.09906) | 本研究引入了生成表示指令调整（GRIT）方法，通过指令区分生成和嵌入任务，训练一个大型语言模型同时处理这两种任务。与其他模型相比，我们的GritLM 7B在文本嵌入基准测试上达到最新的技术水平，并在多种生成任务中表现出色。通过进一步扩大规模，我们的GritLM 8x7B成为最佳的生成语言模型之一，同时仍然是最好的嵌入模型之一。GRIT的统一也大大提高了RAG在长文档上的速度。 |
| [^86] | [Revisiting Recurrent Reinforcement Learning with Memory Monoids](https://arxiv.org/abs/2402.09900) | 这篇论文重新审视了使用内存单子的循环强化学习方法。通过定义新颖的内存单子框架并提出一种新的批处理方法，改进了样本效率、增加了回报并简化了实现过程。 |
| [^87] | [COVIDHealth: A Benchmark Twitter Dataset and Machine Learning based Web Application for Classifying COVID-19 Discussions](https://arxiv.org/abs/2402.09897) | 本研究开发了一个基于机器学习的Web应用程序，用于自动分类社交媒体上的COVID-19相关讨论。通过Twitter数据集的标记和多种特征提取方法的应用，实现了关于COVID-19的健康风险、预防、症状、传播和治疗等方面的分类。 |
| [^88] | [Predictors from causal features do not generalize better to new domains](https://arxiv.org/abs/2402.09891) | 因果特征不能更好地推广到新领域，预测器使用所有特征的效果更好。 |
| [^89] | [Explaining Kernel Clustering via Decision Trees](https://arxiv.org/abs/2402.09881) | 这项工作探讨了可解释的核聚类方法，提出了使用决策树近似核k-means聚类分区的算法，并通过合适的特征选择实现了解释性和近似保证的平衡。 |
| [^90] | [Characterizing Accuracy Trade-offs of EEG Applications on Embedded HMPs](https://arxiv.org/abs/2402.09867) | 该论文研究了在嵌入式多核平台上，采用电池供电的可穿戴设备分析脑电图（EEG）记录的应用。研究发现，通过调整近似方法，可以在有限的能量预算内实现更好的性能和能量收益。 |
| [^91] | [Recommendations for Baselines and Benchmarking Approximate Gaussian Processes](https://arxiv.org/abs/2402.09849) | 对于基准线和基准测试近似高斯过程的研究，我们提出了对比方法的建议，并开发了一种训练程序，该程序不需要用户选择，并且证明这是一个符合要求的强大基准。 |
| [^92] | [A Deep Learning Approach to Radar-based QPE](https://arxiv.org/abs/2402.09846) | 本研究提出了一种基于深度学习的雷达降水估计方法，利用台湾地区的雷达数据进行量化降水估计，并与具体降水位置相关联。与传统的基于Z-R关系的方法不同，该方法利用机器学习算法自动检测天气系统的演变和移动，并将其与特定地形属性的位置相结合。评估结果显示该方法在台北地区的降水估计上具有良好的效果。 |
| [^93] | [LAPDoc: Layout-Aware Prompting for Documents](https://arxiv.org/abs/2402.09841) | 本文研究了通过使用布局增强来使用纯文本LLM进行文档特定任务的可能性。 |
| [^94] | [Performative Reinforcement Learning in Gradually Shifting Environments](https://arxiv.org/abs/2402.09838) | 这项研究提出了一种在渐变环境中进行强化学习的框架，可以模拟部署策略对环境的影响，并提出了一种新的算法MDRR来应对这种情况。 |
| [^95] | [All in One and One for All: A Simple yet Effective Method towards Cross-domain Graph Pretraining](https://arxiv.org/abs/2402.09834) | 本研究提出了一种简单而有效的跨领域图预训练方法，通过一体化和多功能性，使得大型语言模型在各个领域具备了超强的泛化能力。 |
| [^96] | [Utilizing GANs for Fraud Detection: Model Training with Synthetic Transaction Data](https://arxiv.org/abs/2402.09830) | 本论文研究了利用生成对抗网络（GAN）进行欺诈检测的应用，比较了其与传统方法的优势。通过构建对抗性验证图的集合，有效防止了由机器人或自动系统引起的欺诈，并确保交易中的用户是真实的。 |
| [^97] | [Diffusion Models for Audio Restoration](https://arxiv.org/abs/2402.09821) | 本文介绍了基于扩散模型的音频恢复算法，重点关注语音增强和音乐恢复任务。 |
| [^98] | [Enhancing Cybersecurity Resilience in Finance with Deep Learning for Advanced Threat Detection](https://arxiv.org/abs/2402.09820) | 这项研究提出使用深度学习来增强金融行业的网络安全韧性，并实现高级威胁检测。目前的网络威胁检测方法往往基于规则和传统的机器学习方法，无法适用大规模数据应用，并且无法有效检测未知威胁。 |
| [^99] | [Two trust region type algorithms for solving nonconvex-strongly concave minimax problems](https://arxiv.org/abs/2402.09807) | 本文提出了两种信赖域类算法，用于解决非凸强凹最小最大问题，并可以在迭代次数为$\mathcal{O}(\epsilon^{-1.5})$内找到二阶稳定点。 |
| [^100] | [Criterion collapse and loss distribution control](https://arxiv.org/abs/2402.09802) | 该论文研究了"准则崩溃"的概念，即优化一个度量指标意味着另一个度量指标的最优性。研究结果发现，对于损失的伯努利分布，CVaR和DRO的结果远超出现有研究，同时发现了一些特定条件下，单调准则如倾斜ERM无法避免崩溃，而非单调的替代方案可以。 |
| [^101] | [Closed-form Filtering for Non-linear Systems](https://arxiv.org/abs/2402.09796) | 提出了一种基于高斯PSD模型的新型滤波器，可以在转换和观测都是高斯PSD模型时以闭式形式高效地进行滤波，并且提出的估计器具有强大的理论保证，适应转换概率的正则性。 |
| [^102] | [Examining Pathological Bias in a Generative Adversarial Network Discriminator: A Case Study on a StyleGAN3 Model](https://arxiv.org/abs/2402.09786) | 这项研究发现了StyleGAN3模型中判别器的病态偏见，它在图像和面部质量上的得分分层影响了不同性别、种族和其他类别的图像。 |
| [^103] | [MC-DBN: A Deep Belief Network-Based Model for Modality Completion](https://arxiv.org/abs/2402.09782) | MC-DBN是一种基于深度信念网络的模态补全模型，利用完整数据的隐式特征来弥补附加不完整数据的差距，提高预测准确性。 |
| [^104] | [TinyCL: An Efficient Hardware Architecture for Continual Learning on Autonomous Systems](https://arxiv.org/abs/2402.09780) | TinyCL是一种用于自主系统持续学习的高效硬件架构，在CL中支持前向和反向传播，并通过滑动窗口的连续学习策略来减少内存访问。 |
| [^105] | [From Variability to Stability: Advancing RecSys Benchmarking Practices](https://arxiv.org/abs/2402.09766) | 本论文提出了一种新的基准测试方法，通过使用多样化的开放数据集，并在多个度量指标上评估多种协同过滤算法，来研究数据集特征对算法性能的影响。这一方法填补了推荐系统算法比较中的不足之处，推进了评估实践。 |
| [^106] | [A Framework For Gait-Based User Demography Estimation Using Inertial Sensors](https://arxiv.org/abs/2402.09761) | 这项研究提出了一种利用深度学习和层次相关传播（LRP）的框架，用于识别人类步态模式并估计用户的人口统计信息，如年龄和性别。 |
| [^107] | [Robust SVD Made Easy: A fast and reliable algorithm for large-scale data analysis](https://arxiv.org/abs/2402.09754) | 本研究提出了一种名为球形单位正则化SVD的高效算法，用于鲁棒的SVD逼近，该算法不受异常值干扰，计算可伸缩，并能提供准确的奇异向量逼近。相比竞争算法，该算法仅使用标准降秩SVD算法两次应用于适当缩放的数据，具有显著的计算速度优势。 |
| [^108] | [Model Compression and Efficient Inference for Large Language Models: A Survey](https://arxiv.org/abs/2402.09748) | 这项综述研究了大规模语言模型的压缩和高效推理方法，包括量化、修剪、蒸馏、紧凑架构设计和动态网络等方面。大模型的突出特点是压缩后需要微调或重新训练，并且相关的成本很高。 |
| [^109] | [Less is more: Ensemble Learning for Retinal Disease Recognition Under Limited Resources](https://arxiv.org/abs/2402.09747) | 本研究通过使用集成学习方法，在有限资源条件下，提高了视网膜疾病识别的性能，克服了视网膜OCT图像获取和标签过程的挑战。 |
| [^110] | [QuRating: Selecting High-Quality Data for Training Language Models](https://arxiv.org/abs/2402.09739) | QuRating是一种选择高质量数据用于训练语言模型的方法，它能够捕捉人类直观感知的文本的抽象特征。在实验中发现，平衡质量和多样性是很重要的。 |
| [^111] | [DFORM: Diffeomorphic vector field alignment for assessing dynamics across learned models](https://arxiv.org/abs/2402.09735) | 本文提出了DFORM框架，用于评估学习模型的动态特性。DFORM通过学习非线性坐标变换，在学习模型之间提供连续的、最大一对一映射，从而比较它们的动态特性。这扩展了平滑轨道和拓扑的概念，并解决了模型动态对比中的困难。 |
| [^112] | [DOF: Accelerating High-order Differential Operators with Forward Propagation](https://arxiv.org/abs/2402.09730) | DOF是一种高效的计算框架，用于加速高阶微分算子的计算，通过前向传播的方式，不丢失精度，在效率和内存消耗上有着明显的改进。 |
| [^113] | [Best Arm Identification for Prompt Learning under a Limited Budget](https://arxiv.org/abs/2402.09723) | 这项工作提出了一种在提示学习中考虑有限预算约束的方法，通过建立提示学习和多臂赌博机中固定预算最佳臂识别之间的联系，提出了一个通用框架TRIPLE，通过利用聚类和嵌入思想实现了两个增强方法。 |
| [^114] | [Persuading a Learning Agent](https://arxiv.org/abs/2402.09721) | 在一个重复的贝叶斯说服问题中，即使没有承诺能力，委托人可以通过使用上下文无遗憾学习算法来实现与经典无学习模型中具有承诺的委托人的最优效用无限接近的效果；在代理人使用上下文无交换遗憾学习算法的情况下，委托人无法获得比具有承诺的无学习模型中的最优效用更高的效用。 |
| [^115] | [DPBalance: Efficient and Fair Privacy Budget Scheduling for Federated Learning as a Service](https://arxiv.org/abs/2402.09715) | 本文提出了DPBalance，一种新颖的隐私预算调度机制，用于联邦学习作为一项服务（FLaaS）。该机制在效率和公平性之间进行了优化，通过综合考虑数据分析师级别的主导份额和FL特定的性能指标，实现了隐私预算的精确调度。 |
| [^116] | [Node Duplication Improves Cold-start Link Prediction](https://arxiv.org/abs/2402.09711) | 本文研究了在链路预测中改进GNN在低度节点上的性能，提出了一种名为NodeDup的增强技术，通过复制低度节点并创建链接来提高性能。 |
| [^117] | [Preserving Data Privacy for ML-driven Applications in Open Radio Access Networks](https://arxiv.org/abs/2402.09710) | 本文研究了在5G开放无线接入网络（O-RAN）中共享数据库场景下的数据隐私问题，并提出了一种基于洗牌的可学习加密技术来保护机器学习模型的数据隐私。 |
| [^118] | [Sparse and Faithful Explanations Without Sparse Models](https://arxiv.org/abs/2402.09702) | 引入了稀疏解释值(SEV)，用于衡量机器学习模型的决策稀疏性。即使模型不是稀疏的，许多机器学习模型在SEV的衡量下仍具有低决策稀疏性。 |
| [^119] | [Combining Evidence Across Filtrations](https://arxiv.org/abs/2402.09698) | 这篇论文研究了合并使用不同过滤器计算的e进程的方法，探讨了其在顺序推理中的应用。 |
| [^120] | [Reward Poisoning Attack Against Offline Reinforcement Learning](https://arxiv.org/abs/2402.09695) | 这项研究针对深度神经网络函数逼近的一般离线强化学习中的奖励污染攻击问题，提出了一种名为“策略对比攻击”的攻击策略。通过使低性能策略看起来像是高性能的，同时使高性能策略看起来像是低性能的，我们证明了这种攻击有效性。 |
| [^121] | [Robust Learning-Augmented Dictionaries](https://arxiv.org/abs/2402.09687) | 我们提出了一个学习增强的数据结构，通过预测访问频率增强跳表，实现了最佳一致性和鲁棒性。实验证明，与其他数据结构相比，RobustSL在合成数据集和真实数据集上表现出色。 |
| [^122] | [HyperMagNet: A Magnetic Laplacian based Hypergraph Neural Network](https://arxiv.org/abs/2402.09676) | HyperMagNet是一种基于磁度拉普拉斯的超图神经网络，通过将超图表示为非可逆的马尔可夫链并构建磁度拉普拉斯矩阵作为输入，它在节点分类任务中表现出优越性。 |
| [^123] | [PAL: Proxy-Guided Black-Box Attack on Large Language Models](https://arxiv.org/abs/2402.09674) | PAL是第一个黑盒查询攻击大型语言模型的优化算法，通过代理模型引导优化过程，并使用复杂的损失函数，取得了较高的攻击成功率。 |
| [^124] | [Exploiting Alpha Transparency In Language And Vision-Based AI Systems](https://arxiv.org/abs/2402.09671) | 这项研究揭示了利用PNG图像文件格式中的alpha透明层欺骗AI视觉系统的新漏洞，对现有的和实际应用的视觉系统提出了挑战。 |
| [^125] | [How to Train Data-Efficient LLMs](https://arxiv.org/abs/2402.09668) | 本文研究了如何训练数据高效的LLM模型，提出了Ask-LLM和Density两种优秀的数据选择方法。 |
| [^126] | [User Modeling and User Profiling: A Comprehensive Survey](https://arxiv.org/abs/2402.09660) | 这篇综述论文介绍了用户建模与用户画像研究的现状、发展和未来方向。该研究主要关注在人工智能应用中构建准确的用户表示，包括利用大量数据进行建模以及采用深度学习和图数据技术等先进方法。 |
| [^127] | [Digital versus Analog Transmissions for Federated Learning over Wireless Networks](https://arxiv.org/abs/2402.09657) | 本文比较了数字和模拟传输在无线联邦学习中的效果，发现它们的本质区别在于通信和计算是否同时设计，数字方案分离了通信设计和具体任务，而模拟通信可以同时处理大规模设备的传输。 |
| [^128] | [Practitioners' Challenges and Perceptions of CI Build Failure Predictions at Atlassian](https://arxiv.org/abs/2402.09651) | Atlassian的研究调查了CI构建失败对软件开发过程和团队的影响，并研究了将CI构建预测工具集成到Bitbucket环境中所涉及的挑战和期望。 |
| [^129] | [Foul prediction with estimated poses from soccer broadcast video](https://arxiv.org/abs/2402.09650) | 通过整合视频数据、边界框位置、图像细节和姿势信息，我们提出了一种创新的深度学习方法来预测足球犯规。实验结果表明，我们的模型在各方面表现出色。 |
| [^130] | [Multi-Fidelity Methods for Optimization: A Survey](https://arxiv.org/abs/2402.09638) | 多模态优化(MFO)是一种以高模态准确性和计算效率平衡的成本效益策略，通过多模态替代模型、忠诚度管理策略和优化技术来解决复杂计算挑战的方法。MFO在机器学习、工程设计优化和科学发现等多个关键领域有广泛应用的潜力。 |
| [^131] | [MiMiC: Minimally Modified Counterfactuals in the Representation Space](https://arxiv.org/abs/2402.09631) | 提出了一种新颖的对抗事实生成方法，利用闭式解决方案在表示空间中生成富有表达力的对抗事实，以减轻语言模型中的不良行为，该方法在地球移动问题方面提供理论上的保证，并对表示空间的几何组织进行改进。 |
| [^132] | [Smart Information Exchange for Unsupervised Federated Learning via Reinforcement Learning](https://arxiv.org/abs/2402.09629) | 本研究通过强化学习方法，提出了一种解决无监督联邦学习中数据交换问题的智能信息交换方法，该方法通过创建一个最优图形来选择数据传输的链接，以提高收敛速度和提高异常设备鲁棒性。 |
| [^133] | [Conformalized Adaptive Forecasting of Heterogeneous Trajectories](https://arxiv.org/abs/2402.09623) | 本研究提出了一种新的符合性方法，通过结合在线符合性预测技术和解决回归中异方差性的方法，生成了同时预测边界，并能够可靠地覆盖新随机轨迹的整个路径。这种方法不仅有精确的有限样本保证，而且往往比之前的方法具有更丰富的预测结果。 |
| [^134] | [API Pack: A Massive Multilingual Dataset for API Call Generation](https://arxiv.org/abs/2402.09615) | 这个论文介绍了一个名为API Pack的大规模多语言数据集，旨在提高大型语言模型的API调用生成能力，通过实验证明了其在生成未见过的API调用方面的高准确率，并实现了跨语言的API调用生成 |
| [^135] | [Towards Privacy-Aware Sign Language Translation at Scale](https://arxiv.org/abs/2402.09611) | 本研究提出了一种两阶段框架，用于实现规模化隐私感知手语翻译。我们利用自监督视频预训练和有监督微调的方法，在数据稀缺和隐私风险的情况下实现了最先进的手语翻译性能。 |
| [^136] | [Exact, Fast and Expressive Poisson Point Processes via Squared Neural Families](https://arxiv.org/abs/2402.09608) | 该论文介绍了使用平方神经网络族的精确、快速和表达性泊松点过程。通过利用两层神经网络的平方范数来参数化强度函数，可以获得更灵活和高效的方法。该方法在计算积分强度函数时具有封闭形式和二次时间复杂度，并且相比于传统方法更节约内存和时间。通过解决凸优化问题，可以获得对强度函数最终层的参数化重参数化的最大似然估计和最大后验估计。 |
| [^137] | [Scalable Graph Self-Supervised Learning](https://arxiv.org/abs/2402.09603) | 该论文提出了一种通过体积最大化项减少图自监督学习预训练损失函数计算成本的方法。实验证明，采用节点或维度采样可以降低损失计算的成本。 |
| [^138] | [Low-Rank Graph Contrastive Learning for Node Classification](https://arxiv.org/abs/2402.09600) | 本研究提出了一种新颖且鲁棒的低秩图对比学习（LR-GCL）算法，应用于转导节点分类任务。该算法通过低秩正规化的对比学习训练一个编码器，并使用生成的特征进行线性转导分类。 |
| [^139] | [MCMC-driven learning](https://arxiv.org/abs/2402.09598) | 这篇论文旨在统一解决MCMC和机器学习交叉领域的各种问题，包括黑盒变分推断、自适应MCMC、正规流构建和传输辅助MCMC、替代似然MCMC、大数据的MCMC核心集构建等，并提出一个通用的框架。 |
| [^140] | [Pulmonologists-Level lung cancer detection based on standard blood test results and smoking status using an explainable machine learning approach](https://arxiv.org/abs/2402.09596) | 本研究开发了一个基于动态集成选择（DES）的机器学习模型，通过利用标准血液样本分析和吸烟史数据进行肺癌的肺病学水平检测，在丹麦南部地区的大量患有风险的人群中进行了验证。模型在肺病专家提供的诊断预测方面取得了良好的效果。（摘要总结） |
| [^141] | [Reconstructing the Geometry of Random Geometric Graphs](https://arxiv.org/abs/2402.09591) | 该论文通过在底层空间中采样的图来有效地重构随机几何图的几何形状。该方法基于流形假设，即底层空间是低维流形，并且连接概率是嵌入在$\mathbb{R}^N$中的流形中点之间欧几里德距离的严格递减函数。 |
| [^142] | [MLTCP: Congestion Control for DNN Training](https://arxiv.org/abs/2402.09589) | MLTCP是一种用于加速共享GPU集群中的DNN训练作业的拥塞控制技术，通过在每个训练迭代发送的字节数进行缩放，使不同作业的流能够高效利用网络极大地加快训练作业的完成时间。 |
| [^143] | [WERank: Towards Rank Degradation Prevention for Self-Supervised Learning Using Weight Regularization](https://arxiv.org/abs/2402.09586) | 本文提出了一个新的网络权重正则化方法WERank，用于防止自监督学习中的维度坍塌问题。通过实验证明了该方法的有效性，并在图像自监督学习中进行了验证。 |
| [^144] | [Complexity Reduction in Machine Learning-Based Wireless Positioning: Minimum Description Features](https://arxiv.org/abs/2402.09580) | 本文设计了一种定位神经网络（P-NN），通过最小描述特征降低了基于深度学习的无线定位中的复杂度，并开发了一种新的方法来自适应地选择特征空间的大小。 |
| [^145] | [Changes by Butterflies: Farsighted Forecasting with Group Reservoir Transformer](https://arxiv.org/abs/2402.09573) | 我们提出了一种群体储备转换器的架构，通过解决历史序列的挑战和初始条件的敏感性，实现更准确、更稳健地预测长期事件。在多元时间序列中，我们的模型相比最先进的DNN模型表现出更高的准确率，最高可减少89.43%的误差。 |
| [^146] | [Distribution-Free Rates in Neyman-Pearson Classification](https://arxiv.org/abs/2402.09560) | 该论文提供了一个关于Neyman-Pearson分类中无分布率的完整特征，通过简单的几何条件，即三点分离条件，刻画了硬分类器和简单分类器之间的二分条件。 |
| [^147] | [Bidirectional Generative Pre-training for Improving Time Series Representation Learning](https://arxiv.org/abs/2402.09558) | 这项论文提出了一种名为BiTimelyGPT的模型，通过双向的预训练任务在时间序列数据上学习表示，展示了优越的性能，可用于神经功能预测、疾病诊断和生理病征识别。 |
| [^148] | [Dataset Clustering for Improved Offline Policy Learning](https://arxiv.org/abs/2402.09550) | 本文研究了一种名为多行为的数据集特征，提出了一种行为感知的深层聚类方法，将多行为数据集划分为若干单一行为子集，从而提高了离线策略学习的性能。 |
| [^149] | [Layerwise Proximal Replay: A Proximal Point Method for Online Continual Learning](https://arxiv.org/abs/2402.09542) | 这项工作针对在线连续学习中经验回放造成的优化不稳定问题进行了改进，提出了一种逐层近端回放（LPR）方法，通过优化几何的修改来平衡新数据和回放数据的学习，从而改善了回放式在线连续学习方法的准确性。 |
| [^150] | [Why Does Differential Privacy with Large Epsilon Defend Against Practical Membership Inference Attacks?](https://arxiv.org/abs/2402.09540) | 本论文研究了为什么具有较大ε的差分隐私可以防御实际成员推理攻击，因为实际攻击者可能缺乏准确的私有数据知识，并且在实际应用中，数据集可能相对容易被防御。 |
| [^151] | [The Manifold Density Function: An Intrinsic Method for the Validation of Manifold Learning](https://arxiv.org/abs/2402.09529) | 我们提出了一种流形密度函数的内在方法，用于验证流形学习技术，能够适应各种黎曼流形，并证明了其收敛性和鲁棒性。 |
| [^152] | [Guided Quantum Compression for Higgs Identification](https://arxiv.org/abs/2402.09524) | Higgs鉴别的引导量子压缩模型将预处理和量子分类算法统一为可训练模型，解决了量子机器学习中使用自动编码器导致分类性能降低的问题，能够有效鉴别LHC中的希格斯玻色子。 |
| [^153] | [Instruction Tuning for Secure Code Generation](https://arxiv.org/abs/2402.09497) | 现代语言模型在编程中得到广泛应用，指令调优是一个增强其实用性的关键过程。然而，现有的方案忽视了生成代码的安全性。本文提出了SafeCoder，通过安全微调和标准指令调优相结合，来优化安全性和实用性。 |
| [^154] | [On the Potential of Network-Based Features for Fraud Detection](https://arxiv.org/abs/2402.09495) | 本文研究了基于网络特征在欺诈检测中的潜力，通过使用个性化的PageRank算法来捕捉欺诈的社会动态。实验结果表明，集成PPR可以提高模型的预测能力并提供独特有价值的信息。 |
| [^155] | [PMGDA: A Preference-based Multiple Gradient Descent Algorithm](https://arxiv.org/abs/2402.09492) | PMGDA是一个基于偏好的多梯度下降算法，可以 efficiently 在多目标机器学习应用中找到与决策者偏好完全匹配的帕累托最优解。 |
| [^156] | [Intelligent Agricultural Greenhouse Control System Based on Internet of Things and Machine Learning](https://arxiv.org/abs/2402.09488) | 这项研究提出了一种基于物联网和机器学习的智能农业温室控制系统，通过监测和调控温室内环境条件，提高作物生长效率和产量，减少资源浪费。 |
| [^157] | [UMOEA/D: A Multiobjective Evolutionary Algorithm for Uniform Pareto Objectives based on Decomposition](https://arxiv.org/abs/2402.09486) | 该论文提出了一种多目标进化算法UMOEA/D来构建均匀分布的Pareto目标，以解决先前多目标优化方法中有限多样性的问题。 |
| [^158] | [Oracle-Efficient Differentially Private Learning with Public Data](https://arxiv.org/abs/2402.09483) | 这项研究提出了一种具有公共数据的计算高效算法，可以在满足差分隐私条件的情况下学习私有数据，以提高学习算法性能。 |
| [^159] | [Data Reconstruction Attacks and Defenses: A Systematic Evaluation](https://arxiv.org/abs/2402.09478) | 本研究提出了一种在联合学习环境中的强力重构攻击，可以重构中间特征，并且对大部分先前的方法表现更好。实证研究表明，在防御机制中，梯度修剪是对抗最先进攻击最有效的策略。 |
| [^160] | [PANORAMIA: Privacy Auditing of Machine Learning Models without Retraining](https://arxiv.org/abs/2402.09477) | PANORAMIA是一种无需重新训练的机器学习模型隐私审计方案，通过使用生成的“非成员”数据进行成员推断攻击，可以量化大规模ML模型的隐私泄露，而无需控制训练过程或重新训练模型，只需要访问训练数据的子集。 |
| [^161] | [Deciphering Heartbeat Signatures: A Vision Transformer Approach to Explainable Atrial Fibrillation Detection from ECG Signals](https://arxiv.org/abs/2402.09474) | 本研究使用视觉变压器方法解读心率信号，提高心脏疾病检测模型的解释性和可靠性。 |
| [^162] | [One-for-many Counterfactual Explanations by Column Generation](https://arxiv.org/abs/2402.09473) | 本文提出了一个列生成框架，用于解决一对多反事实解释的问题。框架通过限制每个解释中可集体改变的特征数量，旨在尽可能少地使用解释来解释所有实例。相比于现有的混合整数规划方法，该框架在可扩展性、计算性能和解决方案质量方面具有优势。 |
| [^163] | [Machine Learning for Stochastic Parametrisation](https://arxiv.org/abs/2402.09471) | 本文介绍了使用机器学习技术进行随机参数化的方法，旨在改善天气和气候预测模型的准确性和速度。 |
| [^164] | [Rolling Diffusion Models](https://arxiv.org/abs/2402.09470) | 本文介绍了一种滚动扩散模型，用于处理时间数据，通过滑动窗口去噪并根据帧在序列中的时间先后分配不同的噪声量，更好地捕捉到复杂的时间动态。通过实验证明，在视频预测和混沌流体动力学预测任务中，该模型优于传统扩散方法。 |
| [^165] | [Fourier Circuits in Neural Networks: Unlocking the Potential of Large Language Models in Mathematical Reasoning and Modular Arithmetic](https://arxiv.org/abs/2402.09469) | 本研究探索了神经网络和Transformer在数学推理和模运算中的潜力。我们分析了单隐藏层神经网络和单层Transformer在解决复杂代数学习任务中的特征。阐明了边缘最大化原则对单隐藏层神经网络的影响。 |
| [^166] | [Optimal Thresholding Linear Bandit](https://arxiv.org/abs/2402.09467) | 本论文研究了具有固定置信度的随机线性赌博机的ε-阈值赌博机问题，并提出了一种在渐近意义上是最优的算法。 |
| [^167] | [Different Algorithms (Might) Uncover Different Patterns: A Brain-Age Prediction Case Study](https://arxiv.org/abs/2402.09464) | 本文研究了在脑电图研究中不同算法是否能一致揭示出脑龄预测的假设，发现虽然大多数模型揭示了相似的发现，但也存在差异。 |
| [^168] | [A Novel Approach to WaveNet Architecture for RF Signal Separation with Learnable Dilation and Data Augmentation](https://arxiv.org/abs/2402.09461) | 本文提出了一种新的WaveNet架构的适应方法，引入了可学习的扩张参数，显著提高了在密集RF频谱中的信号分离。该方法通过改进模型架构和创新的数据增强策略，成功提高了模型对复杂信号源的识别能力，实现了显著的性能改进，并在竞赛中取得了第一名的成绩。 |
| [^169] | [Unsupervised learning based end-to-end delayless generative fixed-filter active noise control](https://arxiv.org/abs/2402.09460) | 本文提出了一种无监督学习的端到端无延迟生成固定滤波主动噪声控制方法，通过将协处理器和实时控制器集成到一个可微分的ANC系统中，不仅省略了标注过程，而且在噪声降低性能方面表现更好。 |
| [^170] | [Custom IMU-Based Wearable System for Robust 2.4 GHz Wireless Human Body Parts Orientation Tracking and 3D Movement Visualization on an Avatar](https://arxiv.org/abs/2402.09459) | 这项研究的目标是通过构建可负担的定制IMU无线可穿戴系统，在人体运动分析中实现对身体部位定向跟踪和3D运动可视化。 |
| [^171] | [Optimistic Thompson Sampling for No-Regret Learning in Unknown Games](https://arxiv.org/abs/2402.09456) | 该论文提出了一种在未知博弈中进行无遗憾学习的乐观的汤普森抽样方法，通过利用对手的行动和奖励结构信息，显著减少了实验预算，成功地缓解了多机构问题。此外，研究还引入了乐观-无遗憾框架，将现有算法与提出的方法相结合。 |
| [^172] | [Improving EEG Signal Classification Accuracy Using Wasserstein Generative Adversarial Networks](https://arxiv.org/abs/2402.09453) | 该论文提出了一种通过使用Wasserstein生成对抗网络(WGAN)来提高EEG信号分类准确性的实际解决方案。 WGAN在BCI2000数据集上进行训练，并通过改进的平均准确率和测量得分证明了生成的EEG信号的质量。 |
| [^173] | [Data Distribution Dynamics in Real-World WiFi-Based Patient Activity Monitoring for Home Healthcare](https://arxiv.org/abs/2402.09452) | 本文研究了在家庭医疗场景中使用WiFi信号进行日常活动监测的应用，通过在不同环境中部署系统和分析数据变化，指导了稳健、上下文感知的WiFi感知系统的开发，提高了老年护理的生活质量。 |
| [^174] | [Guiding Masked Representation Learning to Capture Spatio-Temporal Relationship of Electrocardiogram](https://arxiv.org/abs/2402.09450) | 本研究提出了一种叫做ST-MEM的模型，通过重构遮蔽的心电图数据来学习时空特征，该模型在心律失常分类任务中优于其他自监督学习方法。 |
| [^175] | [A Comparative Study of Conventional and Tripolar EEG for High-Performance Reach-to-Grasp BCI Systems](https://arxiv.org/abs/2402.09448) | 比较传统EEG与三极EEG在高性能到颤抓握BCI系统中的有效性，包括信噪比、空间分辨率、ERPs和小波时频分析。 |
| [^176] | [Wavelet Analysis of Noninvasive EEG Signals Discriminates Complex and Natural Grasp Types](https://arxiv.org/abs/2402.09447) | 该研究使用小波分析技术对非侵入性脑电图信号进行解码，成功区分复杂和自然的抓握类型，并且证明了小波特征在基于脑电图的抓握区分中的有效性。 |
| [^177] | [iMove: Exploring Bio-impedance Sensing for Fitness Activity Recognition](https://arxiv.org/abs/2402.09445) | 通过传感器融合和对比学习，研究证明生物阻抗传感技术可以改进基于IMU的健身追踪，提高分类模型的精度。 |
| [^178] | [Review of algorithms for predicting fatigue using EEG](https://arxiv.org/abs/2402.09443) | 该研究综述了使用 EEG 信号进行疲劳预测的机器学习算法，并评估了不同算法在基于 EEG 数据预测个体疲劳水平方面的效果。 |
| [^179] | [Deep-Learning Channel Estimation for IRS-Assisted Integrated Sensing and Communication System](https://arxiv.org/abs/2402.09441) | 本文针对智能反射表面辅助集成感知与通信系统中的信道估计问题，提出了一个基于深度学习的三阶段方法。该方法通过解耦问题，分别估计直接感知和通信信道、反射通信信道和反射感知信道，以应对智能反射表面的信号处理能力不足和感知与通信信号之间的互相干扰。 |
| [^180] | [Extreme Learning Machine-based Channel Estimation in IRS-Assisted Multi-User ISAC System](https://arxiv.org/abs/2402.09440) | 本文提出了一种基于极限学习机的智能反射面辅助多用户ISAC系统的信道估计方法，该方法通过将估计问题分解成子问题来解决了感知和通信信号干扰以及被动式IRS缺乏信号处理能力的挑战。该方法可以在保持低成本需求的情况下实现对SAC信道和下行通信信道的准确估计。 |
| [^181] | [Deep-Learning-Based Channel Estimation for IRS-Assisted ISAC System](https://arxiv.org/abs/2402.09439) | 本文提出了一种基于深度学习的框架，在IRS辅助的ISAC系统中解决了信道估计问题。通过设计两种不同的神经网络架构，该方法在不同的信道环境下实现了优越性能。 |
| [^182] | [Subject-Independent Deep Architecture for EEG-based Motor Imagery Classification](https://arxiv.org/abs/2402.09438) | 本研究提出了一种基于无主题深度架构的方法，用于EEG信号的运动想象分类。该方法通过无监督和半监督的方式进行训练，能够在有限的标记样本情况下独立地对不同受试者进行分类。具体而言，通过无监督学习获得潜在特征，然后使用监督学习进行分类。 |
| [^183] | [Disentangling Imperfect: A Wavelet-Infused Multilevel Heterogeneous Network for Human Activity Recognition in Flawed Wearable Sensor Data](https://arxiv.org/abs/2402.09434) | 该论文提出了一种融合小波的多层异构网络（MHNN）用于处理不完美的穿戴式传感器数据。研究团队通过多层离散小波分解提取了多分辨率特征，实现了对不同频率信号的区分，以抑制噪音。 |
| [^184] | [Electrical Behavior Association Mining for Household ShortTerm Energy Consumption Forecasting](https://arxiv.org/abs/2402.09433) | 本文提出了一种基于电气行为关联挖掘的家庭短期能耗预测方法，通过概率关联模型和卷积神经网络门控循环单元的结合，实现了显著的准确性提升。 |
| [^185] | [An Enhanced Analysis of Traffic Intelligence in Smart Cities Using Sustainable Deep Radial Function](https://arxiv.org/abs/2402.09432) | 本论文通过利用深度径向基函数网络，提出了一种新的策略来增强智能城市交通智能。深度RBF网络能够从交通数据中提取有价值的见解，并实现更精确的预测和决策。 |
| [^186] | [DoorINet: A Deep-Learning Inertial Framework for Door-Mounted IoT Applications](https://arxiv.org/abs/2402.09427) | DoorINet是一种用于门贴式物联网应用的深度学习惯性框架，无需使用磁力计即可计算航向角度。 |
| [^187] | [Graph Koopman Autoencoder for Predictive Covert Communication Against UAV Surveillance](https://arxiv.org/abs/2402.09426) | 本论文提出了一种结合了图神经网络（GNN）和Koopman理论的新框架，用于在无人机监视的情况下实现地面的低概率检测（LPD）通信 |
| [^188] | [Epilepsy Seizure Detection and Prediction using an Approximate Spiking Convolutional Transformer](https://arxiv.org/abs/2402.09424) | 本文介绍了一种名为Spiking Conformer的神经形态脉冲卷积变换器，用于从头皮长期脑电图（EEG）记录中检测和预测癫痫发作片段。通过利用基于脉冲的加法操作和近似脉冲神经元层，该模型显著降低了计算成本，同时保持准确性。 |
| [^189] | [EEG Based Generative Depression Discriminator](https://arxiv.org/abs/2402.09421) | 本文通过构建一个生成检测网络，利用脑电图信号学习与抑郁症相关的脑活动，并根据脑活动重新生成目标电极信号，从而实现了对不同类别脑电信号的分类判断。 |
| [^190] | [Multidimensional Gabor-Like Filters Derived from Gaussian Functions on Logarithmic Frequency Axes](https://arxiv.org/abs/2402.09419) | 本文提出了一种新的类小波函数，通过对数频率轴上的高斯函数进行傅里叶逆变换，得到类似于Gabor滤波器的多维滤波器，它可以表示不同大小的定向短时信号振荡，并包含固有的低通滤波器。 |
| [^191] | [Deep Manifold Transformation for Protein Representation Learning](https://arxiv.org/abs/2402.09416) | 提出了一种深度流形转换方法，用于优化蛋白质表示学习，通过应用流形学习策略和新的损失函数来提高学到的嵌入的质量和适应性。 |
| [^192] | [Mitigating Reward Hacking via Information-Theoretic Reward Modeling](https://arxiv.org/abs/2402.09345) | 本文提出了一种名为InfoRM的奖励建模框架，通过引入变分信息瓶颈目标和模型复杂度调节机制，解决了奖励作弊问题，并利用集成聚类偏差得分（ICDS）来检测奖励过度优化。 |
| [^193] | [EcoVal: An Efficient Data Valuation Framework for Machine Learning](https://arxiv.org/abs/2402.09288) | EcoVal是一种高效的机器学习数据估值框架，通过估计每个数据的内在和外在价值，实现了快速实用地估算机器学习模型数据的价值。 |
| [^194] | [Multi-Hierarchical Surrogate Learning for Structural Dynamics of Automotive Crashworthiness Using Graph Convolutional Neural Networks](https://arxiv.org/abs/2402.09234) | 该论文提出了使用图卷积神经网络的多层次代理学习框架，用于汽车碰撞安全结构动力学研究。该框架能够通过创建一系列适应不同计算环境和准确度要求的代理模型，从而提高碰撞仿真的效率和精确度。 |
| [^195] | [Rapid Adoption, Hidden Risks: The Dual Impact of Large Language Model Customization](https://arxiv.org/abs/2402.09179) | 本文介绍了针对不可信定制语言模型的指令后门攻击，通过在定制语言模型中设计带有后门指令的提示，实现攻击者预期的结果。攻击包括三个级别，不需要对后端语言模型进行任何修改。 |
| [^196] | [Exploring the Adversarial Capabilities of Large Language Models](https://arxiv.org/abs/2402.09132) | 本研究探索了大型语言模型的对抗能力，并发现其能够成功地制造对抗性示例以愚弄安全措施，特别是在仇恨言论检测方面具有重大影响。 |
| [^197] | [Towards Robust Model-Based Reinforcement Learning Against Adversarial Corruption](https://arxiv.org/abs/2402.08991) | 本研究通过引入对抗性健壮的乐观MLE（CR-OMLE）算法，解决了模型驱动强化学习中对抗性破坏的挑战，实现了对转移模型的健壮估计。 |
| [^198] | [Correction to "Wasserstein distance estimates for the distributions of numerical approximations to ergodic stochastic differential equations"](https://arxiv.org/abs/2402.08711) | 修正了《对于数值逼近遍历SDE的分布的Wasserstein距离估计》中的错误局部误差估计，提出了一种方法来分析数值离散遍历SDE的Wasserstein-2距离的非渐近保证，并解决了实践中维度依赖性的问题。 |
| [^199] | [Homomorphism Counts for Graph Neural Networks: All About That Basis](https://arxiv.org/abs/2402.08595) | 本研究展示了基于图神经网络的同态计数对于增强其表达能力的重要性，并提出了一种更细致的方法来融合目标模式的同态计数。这种方法比现有方法更具表达力且没有额外的计算复杂度开销。 |
| [^200] | [Denoising Diffusion Restoration Tackles Forward and Inverse Problems for the Laplace Operator](https://arxiv.org/abs/2402.08563) | 本论文提出了一种新的方法，通过使用去噪扩散恢复模型（DDRM）解决了拉普拉斯算子的反向和正向问题，对于泊松方程的解和参数恢复有着显著的改善。 |
| [^201] | [BASE TTS: Lessons from building a billion-parameter Text-to-Speech model on 100K hours of data](https://arxiv.org/abs/2402.08093) | 基于10万小时数据的10亿参数文本到语音模型BASE TTS在语音自然度上达到了最新技术水平，并且能够展现自然的韵律。 |
| [^202] | [Learning Neural Contracting Dynamics: Extended Linearization and Global Guarantees](https://arxiv.org/abs/2402.08090) | 本论文提出了扩展线性化收缩动力学（ELCD），是第一个具有全局收缩性保证的神经网络动力系统，通过参数化非线性向量场的扩展线性化实现。通过在数据空间和潜在空间之间训练微分同胚，并在潜在空间中强制收缩性，ELCD能在面对不确定性时保持全局稳定性和鲁棒性。 |
| [^203] | [Policy Improvement using Language Feedback Models](https://arxiv.org/abs/2402.07876) | 本文介绍了一种使用语言反馈模型（LFMs）改进政策的方法，通过识别期望的行为并进行模仿学习，我们在任务完成率、泛化性能和人类可解释性方面取得了显著改进。 |
| [^204] | [Generalizing across Temporal Domains with Koopman Operators](https://arxiv.org/abs/2402.07834) | 本研究在时间领域泛化问题中提出了库普曼算子的应用，通过对齐条件分布来减小泛化界限。通过使用库普曼算子，我们可以有效地处理时变分布，从而解决时间领域泛化问题。 |
| [^205] | [Universal link predictor by In-context Learning](https://arxiv.org/abs/2402.07738) | 这项工作介绍了一种基于上下文学习的通用链接预测器(UniLP)，它将启发式方法的广泛适用性和参数模型的模式学习能力相结合，实现了自主学习目标图中的链接模式并具有跨不同图的泛化能力。 |
| [^206] | [MAGNETO: Edge AI for Human Activity Recognition -- Privacy and Personalization](https://arxiv.org/abs/2402.07180) | 本文提出了一种名为MAGNETO的边缘AI平台，通过从云端推向边缘进行增量人体活动学习，避免了云端与边缘设备之间的数据传输，实现了数据隐私保护、低延迟处理和高度个性化。 |
| [^207] | [Natural Language Reinforcement Learning](https://arxiv.org/abs/2402.07157) | 本研究将自然语言表示和强化学习原则相结合，提出了自然语言强化学习（NLRL）框架，解决了强化学习在样本效率低、解释性不足和缺乏监督信号等方面的限制问题，通过实验验证了其有效性和可解释性。 |
| [^208] | [Explain Variance of Prediction in Variational Time Series Models for Clinical Deterioration Prediction](https://arxiv.org/abs/2402.06808) | 本文提出了使用delta方法确定性地近似预测的变异性的方法，并采用SHAP方法来归因于变异的贡献。该方法适用于临床恶化预测中的变分时间序列模型，可以在提高预测精度的同时提供解释性。 |
| [^209] | [Sequential Flow Matching for Generative Modeling](https://arxiv.org/abs/2402.06461) | 本文提出了一种称为SeqRF的新方法，用于通过直线化概率流来减小全局截断误差，并以此加速取样和提高综合质量。 |
| [^210] | [Multiscale Modelling with Physics-informed Neural Network: from Large-scale Dynamics to Small-scale Predictions in Complex Systems](https://arxiv.org/abs/2402.05067) | 本文提出了利用物理信息神经网络进行多尺度建模的方法，通过解耦大尺度和小尺度动力学，并在正交基函数空间中近似小尺度系统。实验结果表明该方法在处理液体动力学问题以及更复杂的情况下具有较高的有效性和适用性。 |
| [^211] | [L4Q: Parameter Efficient Quantization-Aware Training on Large Language Models via LoRA-wise LSQ](https://arxiv.org/abs/2402.04902) | L4Q是一种参数高效的量化感知训练算法，通过基于LoRA的学习的量化步长，解决了大型语言模型中量化训练的挑战。 |
| [^212] | [Open-Vocabulary Calibration for Vision-Language Models](https://arxiv.org/abs/2402.04655) | 本文研究了视觉语言模型中的开放词汇校准问题，在提示学习的背景下发现现有的校准方法不足以解决该问题。为此，提出了一种称为 Distance-Aware Ca 的简单而有效的方法来解决问题。 |
| [^213] | [On Computational Limits of Modern Hopfield Models: A Fine-Grained Complexity Analysis](https://arxiv.org/abs/2402.04520) | 通过细粒度复杂性分析，我们研究了现代Hopfield模型的记忆检索计算限制，发现了一种基于模式范数的相变行为，并且建立了有效变体的上界条件。使用低秩逼近的方法，我们提供了有效构造的示例，同时证明了计算时间下界、记忆检索误差界和指数记忆容量。 |
| [^214] | [Intersectional Two-sided Fairness in Recommendation](https://arxiv.org/abs/2402.02816) | 本文针对推荐系统中的交叉双边公平性问题，提出了一种名为交叉双边公平推荐（ITFR）的新方法，通过利用锐度感知损失感知劣势群体，使用协作损失平衡开发不同交叉群体的一致区分能力，并利用预测得分归一化来公平对待不同交叉群体中的正例。实验证明该方法在提高交叉双边公平性方面取得了显著效果。 |
| [^215] | [Robust Multi-Task Learning with Excess Risks](https://arxiv.org/abs/2402.02009) | 提出了一种具有过多风险的多任务学习（ExcessMTL）方法，根据任务到收敛的距离来更新任务权重，以克服存在标签噪声时现有方法的限制。 |
| [^216] | [Adversarial Quantum Machine Learning: An Information-Theoretic Generalization Analysis](https://arxiv.org/abs/2402.00176) | 本文研究了对抗性训练的量子分类器的泛化特性，并提出了新颖的信息论上界。 |
| [^217] | [SWEA: Changing Factual Knowledge in Large Language Models via Subject Word Embedding Altering](https://arxiv.org/abs/2401.17809) | 提出了一种主题词嵌入修改框架（SWEA），通过在推理阶段修改主题的表示来编辑知识，保护模型的原始权重，避免不可逆的损害和额外的推理开销。 |
| [^218] | [OntoMedRec: Logically-Pretrained Model-Agnostic Ontology Encoders for Medication Recommendation](https://arxiv.org/abs/2401.15814) | OntoMedRec是一种基于本体编码器的逻辑预训练、模型无关的医学推荐方法，通过解决医学本体数据稀缺问题，提高了各种模型在EHR数据集和少量药物的入院情况下的性能。 |
| [^219] | [Minimally Supervised Learning using Topological Projections in Self-Organizing Maps](https://arxiv.org/abs/2401.06923) | 这篇论文介绍了一种基于自组织映射的拓扑投影半监督学习方法，可以有效利用大量无标签数据集中的信息，显著降低进行参数预测所需的标记数据点数量。 |
| [^220] | [Enhancing Neural Theorem Proving through Data Augmentation and Dynamic Sampling Method](https://arxiv.org/abs/2312.14188) | 本论文提出了一种名为DS-Prover的动态抽样方法，用于增强神经定理证明的能力。该方法通过动态确定应用于扩展当前目标的策略数量，并调整探索和开发之间的平衡，从而使证明搜索过程更加高效。此外，作者还通过增加训练数据集，将简化和重写策略与多个前提进行分解。 |
| [^221] | [Protect Your Score: Contact Tracing With Differential Privacy Guarantees](https://arxiv.org/abs/2312.11581) | 这篇论文提出了具有差分隐私保障的接触追踪算法，以解决隐私问题限制接触追踪的部署。该算法在多种情景下展现了卓越性能，并通过在发布每个风险分数时保护个体健康状况的隐私。 |
| [^222] | [Learning from Emergence: A Study on Proactively Inhibiting the Monosemantic Neurons of Artificial Neural Networks](https://arxiv.org/abs/2312.11560) | 本文研究了积极抑制人工神经网络中的单意义神经元，这对于提高性能具有重要意义，并提出了一种基于自发现的方法来实现抑制。 |
| [^223] | [GINN-LP: A Growing Interpretable Neural Network for Discovering Multivariate Laurent Polynomial Equations](https://arxiv.org/abs/2312.10913) | GINN-LP是一种可解释的神经网络，用于发现多元Laurent多项式方程的形式和系数。它采用了一种名为“幂项逼近块”的新型可解释性神经网络块，并通过神经网络增长策略和稀疏正则化来优化方程的表示。 |
| [^224] | [Personalized Path Recourse for Reinforcement Learning Agents](https://arxiv.org/abs/2312.08724) | 该论文介绍了一种针对增强学习代理的个性化路径补救方法，该方法通过编辑动作路径来实现期望目标，同时保持与代理的原始路径相似度高，并且个性化适应代理的行为模式。这种方法适用于纠正或改进动作或数据序列以实现预定目标。 |
| [^225] | [Connectivity Oracles for Predictable Vertex Failures](https://arxiv.org/abs/2312.08489) | 论文研究了在预测算法范式下设计支持顶点失败的连通性预测器的问题，并提出了一种数据结构，能够以预处理时间和查询时间的多项式关系来处理失败顶点集合。 |
| [^226] | [Extrapolatable Transformer Pre-training for Ultra Long Time-Series Forecasting](https://arxiv.org/abs/2312.00817) | 提出了一种名为TimelyGPT的可推广的Transformer预训练模型，该模型通过可推广的位置嵌入和循环注意力以及时间卷积模块有效地捕捉超长时间序列数据中的全局和局部时间依赖关系。 |
| [^227] | [Knowledge Transfer from Vision Foundation Models for Efficient Training of Small Task-specific Models](https://arxiv.org/abs/2311.18237) | 本文提出了一个简单的任务导向的知识迁移方法，用于高效训练小型任务特定模型。实验结果表明，该方法在多个目标任务上表现出了更好的性能，并且还展示了高达9倍的性能提升。 |
| [^228] | [AutArch: An AI-assisted workflow for object detection and automated recording in archaeological catalogues](https://arxiv.org/abs/2311.17978) | 这篇论文介绍了AutArch，一种用于考古目录中物体检测和自动化记录的人工智能辅助工作流程，并提出了一种新的数据收集方法，通过自动化从遗留资源中提取数据，解决了现有记录质量和标准不一致的挑战。 |
| [^229] | [ASI: Accuracy-Stability Index for Evaluating Deep Learning Models](https://arxiv.org/abs/2311.15332) | 该论文引入了准确性-稳定性指数（ASI），它是一种综合考虑准确度和稳定性的定量评估深度学习模型的指标。实验结果展示了ASI的应用，提供了一个用于可视化ASI、平均准确度和变异系数的3D曲面模型。这项研究解决了深度学习模型定量基准评估指标的重要问题，并提供了一种准确评估深度学习模型准确性和稳定性的新方法。 |
| [^230] | [Empirical Comparison between Cross-Validation and Mutation-Validation in Model Selection](https://arxiv.org/abs/2311.14079) | 本研究通过对比基准和实际数据集，实证比较了突变验证（MV）和交叉验证（CV）在模型选择中的表现。结果发现，MV和CV在选择模型的泛化性能方面基本等效，但MV在选择简单模型和计算成本方面具有优势。 |
| [^231] | [Analyzing the Evolution and Maintenance of ML Models on Hugging Face](https://arxiv.org/abs/2311.13380) | 本文通过仓库挖掘和文本分析的方式，对Hugging Face上的机器学习模型的演化和维护进行了研究。研究发现了Hugging Face的整体增长和受欢迎程度，揭示了ML领域、框架使用、作者分组等方面的趋势，同时也探讨了开发者社区中普遍存在的主题和见解以及模型的维护状态和演化情况。 |
| [^232] | [Moderating Model Marketplaces: Platform Governance Puzzles for AI Intermediaries](https://arxiv.org/abs/2311.12573) | 本论文研究了模型市场的调节问题，分析了AI中介平台面临的平台治理挑战，并总结了业界的相关实践，包括许可、访问和使用限制、自动内容调节以及公开政策制定。 |
| [^233] | [Random Linear Projections Loss for Hyperplane-Based Optimization in Neural Networks](https://arxiv.org/abs/2311.12356) | 本研究引入了一种名为随机线性投影（RLP）损失的新方法，通过利用数据中的几何关系来提高神经网络训练效率。实证评估表明，使用RLP损失训练的神经网络优于传统损失函数训练的网络，在更少的数据样本下实现更好的性能，并且对于添加噪声表现更强鲁棒性。 |
| [^234] | [Dual input stream transformer for vertical drift correction in eye-tracking reading data](https://arxiv.org/abs/2311.06095) | 这篇论文介绍了一种名为Dual Input Stream Transformer（DIST）的转换器，用于解决眼动阅读数据中由于垂直漂移而产生的注视点分配问题。通过与经典方法进行比较，我们展示了DIST在不同数据集上的高准确性，并通过将多个DIST模型实例组合成一个集成模型进一步提高了准确率。这项研究对于解决阅读研究中手动分配文本行的瓶颈具有重要意义。 |
| [^235] | [Raising the ClaSS of Streaming Time Series Segmentation](https://arxiv.org/abs/2310.20431) | ClaSS是一种新颖、高效且高精度的流式时间序列分割算法，通过自监督时间序列分类评估同质性，并应用统计测试检测显著的变化点。 |
| [^236] | [The Emergence of Reproducibility and Consistency in Diffusion Models](https://arxiv.org/abs/2310.05264) | 该论文研究了扩散模型中的一致模型可重复性现象，实验证实了无论模型框架、模型架构或训练过程如何，不同的扩散模型都能够一致地达到相同的数据分布和评分函数。此外，研究发现扩散模型在学习过程中受训练数据规模的影响，表现出两种不同的训练模式：记忆化模式和泛化模式。 |
| [^237] | [Fleet Learning via Policy Merging](https://arxiv.org/abs/2310.01362) | 本文研究了通过策略合并解决机器人群体学习中的数据存储和传输问题，并提出了一种基于循环神经网络的分布式学习方法。该方法能够在Meta-World环境中将50个任务的策略行为整合，并在大多数训练任务上表现良好。 |
| [^238] | [Enhancing the Hierarchical Environment Design via Generative Trajectory Modeling](https://arxiv.org/abs/2310.00301) | 本文通过引入层次MDP框架，提出了一种在资源约束下增强环境设计的方法，通过上层教师智能体生成适当的训练环境，以促进学生智能体的学习能力发展。 |
| [^239] | [Outlier-Insensitive Kalman Filtering: Theory and Applications](https://arxiv.org/abs/2309.09505) | 本文提出了一种异常不敏感的卡尔曼滤波算法，通过对标准更新步骤的短小迭代过程，减轻了离群值对滤波性能的有害影响。通过将每个潜在的离群值建模为具有未知方差的正态过程，并应用在线估计算法，该方法在滤波场景中表现出竞争性能且对离群值具有鲁棒性。 |
| [^240] | [Rate-Optimal Policy Optimization for Linear Markov Decision Processes](https://arxiv.org/abs/2308.14642) | 本文中，我们研究了在线周期性线性马尔可夫决策过程中的遗憾最小化问题，并提出了一种与周期数K成比率最优的遗憾收敛率O(√K)。这是首个针对带有乐观反馈的随机设置使用基于策略优化的方法并建立与K最优收敛速率的研究，也是首个针对具有全信息反馈的对抗设置并建立与K最优速率的研究，目前尚未找到具有最优速率保证的算法。 |
| [^241] | [Implicit Graph Neural Diffusion Networks: Convergence, Generalization, and Over-Smoothing](https://arxiv.org/abs/2308.03306) | 这篇论文介绍了隐式图神经扩散网络的设计框架，并解决了其收敛性、泛化性和过度平滑问题。这个框架允许学习度量和图扩散强度，同时提出了一个新的模型来避免过度平滑问题。 |
| [^242] | [Fourier-Mixed Window Attention: Accelerating Informer for Long Sequence Time-Series Forecasting](https://arxiv.org/abs/2307.00493) | 本文提出了一种名为FWin的快速本地全局窗口注意力方法，用于加速长序列时间序列预测的Informer方法。通过实验证明，该方法可以提高预测准确性并加速推断速度，同时在非线性回归模型中表现出与Softmax全注意力相媲美甚至更优的效果。 |
| [^243] | [Hyp-OW: Exploiting Hierarchical Structure Learning with Hyperbolic Distance Enhances Open World Object Detection](https://arxiv.org/abs/2306.14291) | Hyp-OW是一种利用超几何距离的层次结构学习增强开放世界目标检测的方法，通过超类正则化器学习和建模已知项目的层次表示，通过基于相似度距离的重新标记模块有效地检测未知对象。 |
| [^244] | [DistriBlock: Identifying adversarial audio samples by leveraging characteristics of the output distribution](https://arxiv.org/abs/2305.17000) | DistriBlock提出了一种能够识别对抗性音频样本的有效检测策略，通过利用输出分布的特征，包括中位数、最大值和最小值、熵以及与后续时间步骤的分布之间的散度，应用二元分类器进行预测。这项研究证明了DistriBlock在识别对抗性音频样本方面的有效性。 |
| [^245] | [Online Algorithms for Hierarchical Inference in Deep Learning applications at the Edge](https://arxiv.org/abs/2304.00891) | 这项研究提出了一种在线算法，用于解决边缘深度学习应用中资源受限的边缘设备的层次推断问题，以在保证推断准确性的同时实现低延迟、带宽节省和能量效率的好处。 |
| [^246] | [Zeroth-Order Optimization Meets Human Feedback: Provable Learning via Ranking Oracles](https://arxiv.org/abs/2303.03751) | 零阶优化算法ZO-RankSGD解决了一个新兴的优化挑战，即只能通过排名预测来评估黑盒目标函数。该算法利用一种新颖的随机估计器来确定下降方向，并保证收敛到一个稳定点。此外，该算法还可用于增强学习中的策略优化问题，特别是当只有对于回报排名的排名预测时。 |
| [^247] | [cGAN-Based High Dimensional IMU Sensor Data Generation for Enhanced Human Activity Recognition in Therapeutic Activities](https://arxiv.org/abs/2302.07998) | 本论文开发了一种基于cGAN的TheraGAN网络，用于生成与康复活动相关的高维IMU传感器数据。通过引入简单活动，简化了生成过程。该方法能够帮助解决传统活动识别分类器中训练数据不足的问题。 |
| [^248] | [Learning Complex Teamwork Tasks Using a Given Sub-task Decomposition](https://arxiv.org/abs/2302.04944) | 通过使用专家提供的任务分解为更简单的多智能体子任务，并将其转移到目标任务中进行集体调整，我们的方法可以有效地学习复杂的多智能体任务，并在解决复杂目标任务所需的时间步数上实现了显著的减少。 |
| [^249] | [On the Convergence of Modified Policy Iteration in Risk Sensitive Exponential Cost Markov Decision Processes](https://arxiv.org/abs/2302.03811) | 这项研究证明了在有限状态和动作空间的情况下，修改的策略迭代算法（MPI）在风险敏感问题中的收敛性，并提供了与已有结果不同的证明方法。 |
| [^250] | [A Latent Space Correlation-Aware Autoencoder for Anomaly Detection in Skewed Data](https://arxiv.org/abs/2301.00462) | 这项工作提出了一种潜变量空间相关感知自编码器，用于解决传感器数据偏斜且非高斯性的异常检测问题。 |
| [^251] | [SimCS: Simulation for Domain Incremental Online Continual Segmentation](https://arxiv.org/abs/2211.16234) | 本论文提出了一个新的方法SimCS，用于解决在线领域增量继续分割的问题。与现有方法相比，SimCS在资源有限的情况下，对不同领域的密集标记图像进行连续训练，无需任务边界信息，展现了更好的性能。 |
| [^252] | [When Less is More: On the Value of "Co-training" for Semi-Supervised Software Defect Predictors](https://arxiv.org/abs/2211.05920) | 该论文研究了半监督软件缺陷预测器中的"共同训练"方法的价值，并发现这种方法可以利用少量标记数据取得与使用全部数据相媲美的预测结果。 |
| [^253] | [FedMT: Federated Learning with Mixed-type Labels](https://arxiv.org/abs/2210.02042) | 本文提出了一种概念新颖的联邦学习设置，即具有混合类型标签的联邦学习，在其中不同的中心可以使用不同的标签准则。为了有效地训练具有混合类型标签的模型，作者提出了一种理论指导和模型无关的方法。 |
| [^254] | [PixTrack: Precise 6DoF Object Pose Tracking using NeRF Templates and Feature-metric Alignment](https://arxiv.org/abs/2209.03910) | PixTrack是一种基于视觉的物体姿态跟踪框架，使用NeRF模板和特征度量对齐方法，能够精确跟踪物体的6DoF姿态，而且无需数据注释或轨迹平滑。方法具有高度精确、鲁棒且无抖动的特点，同时计算效率高，可用于多目标跟踪。 |
| [^255] | [Differentially Private Graph Learning via Sensitivity-Bounded Personalized PageRank](https://arxiv.org/abs/2207.06944) | 本论文提出了一种敏感性有界的个性化PageRank算法，能够保护用户隐私。该算法在保持准确性的同时，实现了差分隐私图学习的几种工具。 |
| [^256] | [Indiscriminate Data Poisoning Attacks on Neural Networks](https://arxiv.org/abs/2204.09092) | 本研究关注对神经网络的任意数据污染攻击，利用二阶信息进行优化设计出了有效的攻击方法，并通过大量实验证明了对深度神经网络的影响。 |
| [^257] | [Fast and explainable clustering based on sorting](https://arxiv.org/abs/2202.01456) | CLASSIX是一种快速可解释的聚类算法，它通过排序后的数据的贪婪聚合和群组合并来进行聚类。该算法具有与最先进的聚类算法相媲美的性能，并且具有线性空间复杂性和近线性时间复杂性。 |
| [^258] | [ED2: Environment Dynamics Decomposition World Models for Continuous Control](https://arxiv.org/abs/2112.02817) | 提出了一种环境动力学分解世界模型构建框架ED2，能够通过发现子动力学并进行分解预测，更准确地构建世界模型。 |
| [^259] | [Structure by Architecture: Structured Representations without Regularization](https://arxiv.org/abs/2006.07796) | 我们提出了一种自我监督的结构化表示学习方法，使用无需正则化的自动编码器架构。通过依赖潜变量的独立性进行采样，我们避免了重构质量和生成性能之间的权衡。我们的模型能够学习出一种有序的结构化表示，改善了生成、解缠和外推等多个下游任务的性能。 |
| [^260] | [Enhancing Efficiency and Robustness in Support Vector Regression with HawkEye Loss.](http://arxiv.org/abs/2401.16785) | 通过引入名为HawkEye损失函数的新的对称损失函数，本文解决了支持向量回归在处理离群值和噪声时遇到的挑战，并提供了增强的泛化性能和鲁棒性。 |
| [^261] | [Are self-explanations from Large Language Models faithful?.](http://arxiv.org/abs/2401.07927) | 大型语言模型的自我解释是否可靠是一个重要的AI安全考虑因素，我们提出使用自洽性检测作为评估其可靠性和解释能力的方法。 |
| [^262] | [Zero-Shot Position Debiasing for Large Language Models.](http://arxiv.org/abs/2401.01218) | 本文提出了一种零样本位置去偏方法（ZOE）来降低大语言模型（LLMs）的位置偏差问题，该方法利用预训练的LLMs的无监督响应进行去偏。实验证实ZOE在多个数据集和任务中均表现出优异的性能。 |
| [^263] | [Speak Like a Native: Prompting Large Language Models in a Native Style.](http://arxiv.org/abs/2311.13538) | 本文提出了一种名为AlignedCoT的新颖有效方法，通过将上下文示例与大型语言模型（LLMs）的母语风格对齐，提高了LLMs的推理能力和性能。 |
| [^264] | [Better Fair than Sorry: Adversarial Missing Data Imputation for Fair GNNs.](http://arxiv.org/abs/2311.01591) | 该论文提出了一种针对公平GNN的对抗性缺失数据填充模型，以解决现有公平GNN的假设问题。实验证明此模型的有效性。 |
| [^265] | [Bayesian Multistate Bennett Acceptance Ratio Methods.](http://arxiv.org/abs/2310.20699) | 贝叶斯多状态Bennett接受比率方法（BayesMBAR）是多状态Bennett接受比率（MBAR）方法的贝叶斯推广。通过整合采样配置和先验分布，BayesMBAR计算了自由能的后验分布，并提供更准确的不确定性估计。 |
| [^266] | [Break it, Imitate it, Fix it: Robustness by Generating Human-Like Attacks.](http://arxiv.org/abs/2310.16955) | 本研究提出了一个对抗训练框架，使用有限的人类对手示例生成更有用的大规模对抗示例，有效提高了自然语言处理系统对于人类对手的鲁棒性。 |
| [^267] | [Absolute Policy Optimization.](http://arxiv.org/abs/2310.13230) | 这篇论文提出了绝对策略优化（APO）的方法，通过优化一个新颖的目标函数，在保证性能下界的同时，实现了连续控制任务和Atari游戏中的令人瞩目的结果。 |
| [^268] | [Causal Similarity-Based Hierarchical Bayesian Models.](http://arxiv.org/abs/2310.12595) | 本文提出了一种基于因果相似性的分层贝叶斯模型，通过学习如何从具有相似因果机制的训练任务中汇集数据来提高机器学习算法对新任务的泛化能力。 |
| [^269] | [NeuroCUT: A Neural Approach for Robust Graph Partitioning.](http://arxiv.org/abs/2310.11787) | NeuroCUT是一种神经方法，用于解决鲁棒的图分区问题。它通过两个关键创新，即对图拓扑和分区计数具有归纳性，以及利用强化学习基础，能够从数据中学习启发式方法。 |
| [^270] | [ByteStack-ID: Integrated Stacked Model Leveraging Payload Byte Frequency for Grayscale Image-based Network Intrusion Detection.](http://arxiv.org/abs/2310.09298) | ByteStack-ID是一种基于灰度图像和负载字节频率的集成堆叠模型，用于数据包级入侵检测。它能迅速准确地识别网络流量中的各种攻击类型，并与传统方法有所不同。 |
| [^271] | [Initialization Bias of Fourier Neural Operator: Revisiting the Edge of Chaos.](http://arxiv.org/abs/2310.06379) | 本文研究了Fourier神经操作符(FNO)的初始化偏差，提出了一种FNO版本的He初始化方案，通过模式截断和密集连接网络相似的特点，解决了训练不稳定的负初始化偏差问题。 |
| [^272] | [Unlabeled Out-Of-Domain Data Improves Generalization.](http://arxiv.org/abs/2310.00027) | 这个论文提出了一种新的框架，可以将无标记的域外数据纳入半监督分类问题，从而改善泛化能力。该框架结合了分布鲁棒优化与自监督训练，并利用了高效的多项式时间算法。在理论上，该框架在高斯混合分类问题中得到了验证。 |
| [^273] | [Differential 2D Copula Approximating Transforms via Sobolev Training: 2-Cats Networks.](http://arxiv.org/abs/2309.16391) | 本文介绍了一种通过Sobolev训练的2-Cats网络，它能够非参数地逼近任何二维Copula，并且在估计输出方面优于现有技术。 |
| [^274] | [Revisiting LARS for Large Batch Training Generalization of Neural Networks.](http://arxiv.org/abs/2309.14053) | 本文通过对大批量训练技术的研究，提出了一种新的算法TVLARS，该算法利用可配置的函数替代了热身阶段，以实现对于神经网络的稳健训练。实验证明，在大多数情况下，TVLARS比LARS和LAMB都有更好的性能表现，特别是在自监督学习方面。 |
| [^275] | [MEDL-U: Uncertainty-aware 3D Automatic Annotation based on Evidential Deep Learning.](http://arxiv.org/abs/2309.09599) | 本文提出了一种基于证据深度学习的方法，用于解决3D对象检测中伪标签的模糊性问题，并生成准确的伪标签和量化伪标签的不确定性。 |
| [^276] | [Monitoring Urban Changes in Mariupol/Ukraine in 2022/23.](http://arxiv.org/abs/2309.08607) | 本文研究证明使用历史数据进行迁移学习是解决城市变化监测问题的可行方案，通过使用合成孔径雷达和光学多光谱观测数据，成功监测了乌克兰马里乌波尔市在俄乌冲突开始阶段的相关城市变化。 |
| [^277] | [ConR: Contrastive Regularizer for Deep Imbalanced Regression.](http://arxiv.org/abs/2309.06651) | ConR是一种对比正则化器，通过建模全局和局部标签相似性，防止少数样本的特征被折叠到其多数邻居中，有效地处理深度不平衡回归问题。 |
| [^278] | [Broadband Ground Motion Synthesis via Generative Adversarial Neural Operators: Development and Validation.](http://arxiv.org/abs/2309.03447) | 本论文提出了一种使用生成对抗神经算子的数据驱动地面运动合成模型，可以根据不同参数生成三分量加速度时间历史。通过使用神经算子架构，模型训练不受数据采样频率影响。研究结果表明，该模型在验证和应用实例中表现出色，并可用于生成日本地震动数据。 |
| [^279] | [Interactive and Concentrated Differential Privacy for Bandits.](http://arxiv.org/abs/2309.00557) | 本文研究了在交互学习和推荐系统中隐私保护的Bandit问题，并引入了集中差分隐私的概念。通过提供关于有限臂和线性Bandit问题遗憾的下界，我们揭示了不同隐私预算下的难度区域，并发现集中差分隐私可以比全局差分隐私更有效地保护隐私，我们提出了两种相应的算法。 |
| [^280] | [Escaping the Sample Trap: Fast and Accurate Epistemic Uncertainty Estimation with Pairwise-Distance Estimators.](http://arxiv.org/abs/2308.13498) | 本文介绍了使用配对距离估计器对集成模型进行认识不确定性估计的新方法，相比于常用的深度学习方法，该方法能够更快速、更准确地在更大的空间和更高维度上估计认识不确定性。 |
| [^281] | [Normalization Is All You Need: Understanding Layer-Normalized Federated Learning under Extreme Label Shift.](http://arxiv.org/abs/2308.09565) | 本论文揭示了层归一化和联邦学习中的标签偏移问题之间的深刻联系，通过在联邦学习中应用特征归一化，使得对严重倾斜的数据集进行加速全局训练，从而在极端标签偏移下获得显著改进。 |
| [^282] | [MDB: Interactively Querying Datasets and Models.](http://arxiv.org/abs/2308.06686) | MDB是一个调试框架，用于互动查询数据集和模型。它通过集成函数式编程与关系代数，能够快速迭代和优化查询，发现和描述错误和模型行为。实验证明，MDB比其他工具能够实现更快的查询速度加快和查询长度缩短。 |
| [^283] | [A/B Testing and Best-arm Identification for Linear Bandits with Robustness to Non-stationarity.](http://arxiv.org/abs/2307.15154) | 本文研究了在非稳态环境中的线性赌博机的最佳臂识别问题，提出了一种具有鲁棒性的算法来解决。该算法通过在每个时间步从一个G-最优设计中随机选择臂来实现最佳臂的鲁棒识别。 |
| [^284] | [Privacy Amplification via Importance Sampling.](http://arxiv.org/abs/2307.10187) | 通过重要性采样进行隐私放大，可以同时增强隐私保护和提高效用。我们提供了一个一般的结果来量化选择概率权重对隐私放大的影响，并展示了异质采样概率可以在保持子采样大小不变的情况下获得更好的隐私和效用。 |
| [^285] | [Programmable Synthetic Tabular Data Generation.](http://arxiv.org/abs/2307.03577) | 这项工作介绍了ProgSyn，第一个可编程的合成表格数据生成算法，它允许对生成的数据进行全面的自定义，并且通过预训练和微调生成模型来确保高质量的数据和遵守自定义规范。 |
| [^286] | [Stabilized Neural Differential Equations for Learning Constrained Dynamics.](http://arxiv.org/abs/2306.09739) | 本文提出了一种稳定神经微分方程（SNDEs）的方法，可以强制使用任意流形约束。该方法通过添加稳定项使约束流形成为渐进稳定的，并且在实验中表现优于现有方法。 |
| [^287] | [Multiscale Flow for Robust and Optimal Cosmological Analysis.](http://arxiv.org/abs/2306.04689) | 用多尺度流进行二维宇宙学数据的生成和建模，可识别不同尺度的信息并显著胜过现有方法。 |
| [^288] | [Improved Stability and Generalization Analysis of the Decentralized SGD Algorithm.](http://arxiv.org/abs/2306.02939) | 本文提出了新的算法稳定性理论来改进分布式SGD算法的泛化性能分析，推翻了现有技术对通信图负面影响的观点，并展示了D-SGD在凸设置中与经典SGD算法泛化界相同。 |
| [^289] | [Clarify Confused Nodes Through Separated Learning.](http://arxiv.org/abs/2306.02285) | 本文提出了使用邻域混淆度量来分离学习解决图神经网络中混淆节点的问题。这种方法可以更可靠地区分异质节点和同质节点，并改善性能。 |
| [^290] | [vFedSec: Efficient Secure Aggregation for Vertical Federated Learning via Secure Layer.](http://arxiv.org/abs/2305.16794) | vFedSec提出了一个用于垂直联邦学习的新型Secure Layer，旨在使用最先进的安全模块，实现安全高效的联合训练。实验结果表明，该方法在保护数据隐私效果显著，不会影响训练性能。 |
| [^291] | [Improving selective classification performance of deep neural networks through post-hoc logit normalization and temperature scaling.](http://arxiv.org/abs/2305.15508) | 本文提出了一种$p$-NormSoftmax的事后置信度估计器来提高深度神经网络的选择分类性能。 |
| [^292] | [Learning Structured Components: Towards Modular and Interpretable Multivariate Time Series Forecasting.](http://arxiv.org/abs/2305.13036) | 本文提出了一个名为SCNN的模块化和解释性的预测框架，旨在单独对空间-时间模式的每个成分进行建模。SCNN使用预定义的MTS生成过程，将MTS数据分解为结构化和异构成分，然后分别推断这些成分的演化，能够实现比现有先进模型更高的性能。 |
| [^293] | [Self-Correcting Bayesian Optimization through Bayesian Active Learning.](http://arxiv.org/abs/2304.11005) | 该论文提出了SAL和SCoreBO两种方法，用于提高高斯过程模型的超参数选择和贝叶斯优化的表现。 |
| [^294] | [Microseismic source imaging using physics-informed neural networks with hard constraints.](http://arxiv.org/abs/2304.04315) | 本论文提出一种使用物理知识约束的神经网络（PINNs）进行直接微震成像的方法，能够生成聚焦的源图像，即使只有极少的记录。数值实验表明，该方法可以产生可靠且精确的结果。 |
| [^295] | [Training and Deploying Spiking NN Applications to the Mixed-Signal Neuromorphic Chip Dynap-SE2 with Rockpool.](http://arxiv.org/abs/2303.12167) | 本文介绍了一种通过优化网络参数和注入对抗性参数噪声，将SNN应用程序离线训练和部署到Dynap-SE2混合信号神经形态处理器的新方法。优化后的网络表现出很强的鲁棒性，对于硬件约束的真实世界应用程序有很大的潜力。 |
| [^296] | [Inverse Solvability and Security with Applications to Federated Learning.](http://arxiv.org/abs/2211.14115) | 介绍了逆可解性和安全性的概念，以及其在联邦学习中的应用。论文提供了模型示例，展示了如何通过增加用户数量来增加可解性和安全性。 |

# 详细

[^1]: 针对连续序列到序列建模的分层状态空间模型

    Hierarchical State Space Models for Continuous Sequence-to-Sequence Modeling

    [https://arxiv.org/abs/2402.10211](https://arxiv.org/abs/2402.10211)

    分层状态空间模型（HiSS）是一种针对连续序列到序列建模的技术，它利用堆叠的结构化状态空间模型来进行预测。

    

    arXiv:2402.10211v1 公告类型：新的 摘要：从原始感知数据的序列推理是从医疗设备到机器人领域中普遍存在的问题。这些问题常常涉及使用长序列的原始传感器数据（例如磁力计，压阻器）来预测理想的物理量序列（例如力量，惯性测量）。虽然经典方法对于局部线性预测问题非常有效，但在使用实际传感器时往往表现不佳。这些传感器通常是非线性的，受到外界变量（例如振动）的影响，并且表现出数据相关漂移。对于许多问题来说，预测任务受到稀缺标记数据集的限制，因为获取地面真实标签需要昂贵的设备。在这项工作中，我们提出了分层状态空间模型（HiSS），这是一种概念上简单、全新的连续顺序预测技术。HiSS将结构化的状态空间模型堆叠在一起，以创建一个暂定的预测模型。

    arXiv:2402.10211v1 Announce Type: new  Abstract: Reasoning from sequences of raw sensory data is a ubiquitous problem across fields ranging from medical devices to robotics. These problems often involve using long sequences of raw sensor data (e.g. magnetometers, piezoresistors) to predict sequences of desirable physical quantities (e.g. force, inertial measurements). While classical approaches are powerful for locally-linear prediction problems, they often fall short when using real-world sensors. These sensors are typically non-linear, are affected by extraneous variables (e.g. vibration), and exhibit data-dependent drift. For many problems, the prediction task is exacerbated by small labeled datasets since obtaining ground-truth labels requires expensive equipment. In this work, we present Hierarchical State-Space Models (HiSS), a conceptually simple, new technique for continuous sequential prediction. HiSS stacks structured state-space models on top of each other to create a tempor
    
[^2]: 自我对抗微调扩散模型用于文本到图像生成

    Self-Play Fine-Tuning of Diffusion Models for Text-to-Image Generation

    [https://arxiv.org/abs/2402.10210](https://arxiv.org/abs/2402.10210)

    本文介绍了一种创新的技术，称为自我对抗微调扩散模型（SPIN-Diffusion），通过扩散模型与其先前版本的竞争，实现了逐步自我改进过程。

    

    微调扩散模型在生成人工智能领域仍然是一个未被充分探索的前沿，尤其是与在大型语言模型（LLMs）微调方面取得的显著进展相比。尽管现在的先进扩散模型如稳定扩散（SD）和SDXL依赖于监督微调，但它们的性能在观察到一定数量的数据后必然会达到瓶颈。最近，强化学习（RL）被应用于通过人类偏好数据对扩散模型进行微调，但每个文本提示需要至少两个图像（“获胜者”和“失败者”图像）。本文介绍了一种创新的技术，称为自我对抗微调扩散模型（SPIN-Diffusion），其中扩散模型与其先前版本进行竞争，促进了一个迭代的自我改进过程。我们的方法提供了一种替代传统监督微调和RL策略的选择。

    arXiv:2402.10210v1 Announce Type: cross  Abstract: Fine-tuning Diffusion Models remains an underexplored frontier in generative artificial intelligence (GenAI), especially when compared with the remarkable progress made in fine-tuning Large Language Models (LLMs). While cutting-edge diffusion models such as Stable Diffusion (SD) and SDXL rely on supervised fine-tuning, their performance inevitably plateaus after seeing a certain volume of data. Recently, reinforcement learning (RL) has been employed to fine-tune diffusion models with human preference data, but it requires at least two images ("winner" and "loser" images) for each text prompt. In this paper, we introduce an innovative technique called self-play fine-tuning for diffusion models (SPIN-Diffusion), where the diffusion model engages in competition with its earlier versions, facilitating an iterative self-improvement process. Our approach offers an alternative to conventional supervised fine-tuning and RL strategies, signific
    
[^3]: 恢复生成模型的预微调权重

    Recovering the Pre-Fine-Tuning Weights of Generative Models

    [https://arxiv.org/abs/2402.10208](https://arxiv.org/abs/2402.10208)

    该论文提出了一种恢复生成模型预微调权重的方法，通过少量低秩微调模型可以恢复准确的预微调权重，利用这个新漏洞攻击大规模模型。

    

    在生成建模中，主流模式包括两个步骤：i) 在大规模但不安全的数据集上进行预训练，ii) 通过微调将预训练模型与人类价值观对齐。这种做法被认为是安全的，因为目前没有一种方法可以恢复不安全的预微调模型权重。本文证明了这种假设通常是错误的。具体而言，我们提出了一种称为谱反调的方法，可以使用少量低秩（LoRA）微调模型恢复预微调模型的权重。与先前试图恢复预微调能力的攻击不同，我们的方法旨在恢复精确的预微调权重。我们的方法利用了这个新的对大规模模型的漏洞，例如个性化的稳定扩散和对齐的Mistral模型。

    arXiv:2402.10208v1 Announce Type: cross  Abstract: The dominant paradigm in generative modeling consists of two steps: i) pre-training on a large-scale but unsafe dataset, ii) aligning the pre-trained model with human values via fine-tuning. This practice is considered safe, as no current method can recover the unsafe, pre-fine-tuning model weights. In this paper, we demonstrate that this assumption is often false. Concretely, we present Spectral DeTuning, a method that can recover the weights of the pre-fine-tuning model using a few low-rank (LoRA) fine-tuned models. In contrast to previous attacks that attempt to recover pre-fine-tuning capabilities, our method aims to recover the exact pre-fine-tuning weights. Our approach exploits this new vulnerability against large-scale models such as a personalized Stable Diffusion and an aligned Mistral.
    
[^4]: 基于上下文的奖励：基于动态偏好调整的多目标基础模型对齐

    Rewards-in-Context: Multi-objective Alignment of Foundation Models with Dynamic Preference Adjustment

    [https://arxiv.org/abs/2402.10207](https://arxiv.org/abs/2402.10207)

    本文介绍了Rewards-in-Context（RiC）方法，该方法通过多个奖励条件控制基础模型的响应，并应用有监督的微调进行对齐。它具有简单性和适应性，并支持在推理时动态调整用户偏好。

    

    我们考虑了基于人类偏好的基础模型多目标对齐问题，这是实现有益和无害的人工智能系统的关键步骤。然而，使用强化学习（RL）对大型基础模型进行微调通常是昂贵且不稳定的，并且人类偏好的多维度、异质性和冲突性进一步复杂化了对齐过程。在本文中，我们引入了Rewards-in-Context（RiC）方法，它使得基础模型的响应取决于其提示上下文中的多个奖励，并应用有监督的微调来进行对齐。RiC的显著特点是简单性和适应性，因为它只需要对单个基础模型进行有监督的微调，并支持在推理时动态调整用户偏好。受到抽象的凸优化问题的解析解的启发，我们提出了一种动态推理时调整方法。

    arXiv:2402.10207v1 Announce Type: cross  Abstract: We consider the problem of multi-objective alignment of foundation models with human preferences, which is a critical step towards helpful and harmless AI systems. However, it is generally costly and unstable to fine-tune large foundation models using reinforcement learning (RL), and the multi-dimensionality, heterogeneity, and conflicting nature of human preferences further complicate the alignment process. In this paper, we introduce Rewards-in-Context (RiC), which conditions the response of a foundation model on multiple rewards in its prompt context and applies supervised fine-tuning for alignment. The salient features of RiC are simplicity and adaptivity, as it only requires supervised fine-tuning of a single foundation model and supports dynamic adjustment for user preferences during inference time. Inspired by the analytical solution of an abstracted convex optimization problem, our dynamic inference-time adjustment method appro
    
[^5]: 异构图上基于伊辛模型的特定任务图子抽样

    Ising on the Graph: Task-specific Graph Subsampling via the Ising Model

    [https://arxiv.org/abs/2402.10206](https://arxiv.org/abs/2402.10206)

    该论文提出了一种基于伊辛模型的图子抽样方法，可以针对特定任务在图结构上进行减小，并通过学习伊辛模型的外部磁场来实现。该方法的多功能性在图像分割、三维形状稀疏化和稀疏逼近矩阵求逆等应用中得到展示。

    

    减少图的大小同时保持其整体结构是一个具有许多应用的重要问题。通常，减小图的方法要么删除边缘（稀疏化），要么合并节点（粗化），而没有特定的下游任务。在本文中，我们提出了一种使用在节点或边上定义的伊辛模型对图结构进行子抽样的方法，并使用图神经网络学习伊辛模型的外部磁场。我们的方法是任务特定的，因为它可以端到端地学习如何为特定的下游任务减小图的大小。所使用的任务损失函数甚至不需要可微分性。我们在三个不同的应用上展示了我们方法的多功能性：图像分割、三维形状稀疏化和稀疏逼近矩阵求逆。

    arXiv:2402.10206v1 Announce Type: cross  Abstract: Reducing a graph while preserving its overall structure is an important problem with many applications. Typically, the reduction approaches either remove edges (sparsification) or merge nodes (coarsening) in an unsupervised way with no specific downstream task in mind. In this paper, we present an approach for subsampling graph structures using an Ising model defined on either the nodes or edges and learning the external magnetic field of the Ising model using a graph neural network. Our approach is task-specific as it can learn how to reduce a graph for a specific downstream task in an end-to-end fashion. The utilized loss function of the task does not even have to be differentiable. We showcase the versatility of our approach on three distinct applications: image segmentation, 3D shape sparsification, and sparse approximate matrix inverse determination.
    
[^6]: 构建联想记忆与概率建模之间的桥梁

    Bridging Associative Memory and Probabilistic Modeling

    [https://arxiv.org/abs/2402.10202](https://arxiv.org/abs/2402.10202)

    基于联想记忆的能量函数可以被视为概率建模的对数似然函数，这篇论文构建了两者之间的桥梁，提出了新的基于能量的模型，并展示了两种新的联想记忆模型，可灵活适应上下文数据集。

    

    arXiv:2402.10202v1 公告类型：新的 摘要：联想记忆和概率建模是人工智能中两个基本的主题。第一个研究设计用于去噪、完成和检索数据的递归神经网络，而第二个研究学习和从概率分布中采样。基于联想记忆的能量函数可以被视为概率建模的负对数似然函数的观察，我们在两个之间建立了一座桥梁，使得想法能在两个方向上有益的流动。我们展示了四个例子：首先，我们提出了新的以能量为基础的模型，这些模型可以灵活地适应新的上下文数据集，这种方法称为“上下文学习能量函数”。其次，我们提出了两种新的联想记忆模型：一种是根据训练数据的需要动态创建新的记忆，使用贝叶斯非参数方法，另一种是明确计算比例记忆分配，使用e作为概率函数分配记忆。

    arXiv:2402.10202v1 Announce Type: new  Abstract: Associative memory and probabilistic modeling are two fundamental topics in artificial intelligence. The first studies recurrent neural networks designed to denoise, complete and retrieve data, whereas the second studies learning and sampling from probability distributions. Based on the observation that associative memory's energy functions can be seen as probabilistic modeling's negative log likelihoods, we build a bridge between the two that enables useful flow of ideas in both directions. We showcase four examples: First, we propose new energy-based models that flexibly adapt their energy functions to new in-context datasets, an approach we term \textit{in-context learning of energy functions}. Second, we propose two new associative memory models: one that dynamically creates new memories as necessitated by the training data using Bayesian nonparametrics, and another that explicitly computes proportional memory assignments using the e
    
[^7]: 使用锐度感知最小化和通道注意力解锁Transformer在时间序列预测中的潜力

    Unlocking the Potential of Transformers in Time Series Forecasting with Sharpness-Aware Minimization and Channel-Wise Attention

    [https://arxiv.org/abs/2402.10198](https://arxiv.org/abs/2402.10198)

    本文研究了Transformer在时间序列预测中的局限性，发现其注意力机制是泛化能力不足的原因。在此基础上，提出了一个浅层轻量级的Transformer模型SAMformer，通过锐度感知优化避免了陷入坏的局部最小值，并在常用时间序列数据集上超过了当前最先进的模型TSMixer。

    

    Transformer架构在自然语言处理和计算机视觉中取得了突破性的性能，但在多元长期预测方面，它们仍然不如更简单的线性基线。为了更好地理解这一现象，我们首先研究了一个玩具线性预测问题，展示了尽管Transformer具有高表达能力，但它们无法收敛到真正的解决方案。我们进一步确定Transformer的注意力是造成其低泛化能力的原因。基于这一认识，我们提出了一个浅层轻量级的Transformer模型，在锐度感知优化的情况下成功避免了坏的局部最小值。我们通过实验证明，这个结果适用于所有常用的实际多元时间序列数据集。特别是，相比当前最先进的模型TSMixer，SAMformer的平均性能提高了14.33%，并且参数数量减少了约4倍。

    arXiv:2402.10198v1 Announce Type: new  Abstract: Transformer-based architectures achieved breakthrough performance in natural language processing and computer vision, yet they remain inferior to simpler linear baselines in multivariate long-term forecasting. To better understand this phenomenon, we start by studying a toy linear forecasting problem for which we show that transformers are incapable of converging to their true solution despite their high expressive power. We further identify the attention of transformers as being responsible for this low generalization capacity. Building upon this insight, we propose a shallow lightweight transformer model that successfully escapes bad local minima when optimized with sharpness-aware optimization. We empirically demonstrate that this result extends to all commonly used real-world multivariate time series datasets. In particular, SAMformer surpasses the current state-of-the-art model TSMixer by 14.33% on average, while having ~4 times few
    
[^8]: BitDelta：你的微调可能只有一个比特的价值

    BitDelta: Your Fine-Tune May Only Be Worth One Bit

    [https://arxiv.org/abs/2402.10193](https://arxiv.org/abs/2402.10193)

    BitDelta研究探讨了大型语言模型在微调过程中的信息冗余性，并提出了一种名为BitDelta的方法，可以将微调过程中添加的信息量化为一个比特，同时保持性能。这一发现对于多租户模型的服务和存储有重要意义，并可以显著降低GPU内存需求。

    

    大型语言模型（LLMs）通常在两个阶段进行训练：在大规模互联网数据集上进行预训练，然后在下游任务上进行微调。由于预训练的高计算需求，直觉上认为微调对模型的信息添加较少，因此更具有可压缩性。我们通过将微调模型的权重分解为预训练组件和额外的增量来探究这一假设。我们引入了一种简单的方法——BitDelta，成功地将这个增量量化为1比特而不影响性能。这一有趣的发现不仅突显了微调过程中添加的信息的潜在冗余性，而且对于多租户模型的服务和存储也具有重要影响。通过使用一个高精度的基础模型以及多个1比特的增量，BitDelta大大降低了GPU内存需求。

    arXiv:2402.10193v1 Announce Type: cross  Abstract: Large Language Models (LLMs) are typically trained in two phases: pre-training on large internet-scale datasets, and fine-tuning for downstream tasks. Given the higher computational demand of pre-training, it's intuitive to assume that fine-tuning adds less new information to the model, and is thus more compressible. We explore this assumption by decomposing the weights of fine-tuned models into their pre-trained components and an additional delta. We introduce a simple method, BitDelta, which successfully quantizes this delta down to 1 bit without compromising performance. This interesting finding not only highlights the potential redundancy of information added during fine-tuning, but also has significant implications for the multi-tenant serving and multi-tenant storage of fine-tuned models. By enabling the use of a single high-precision base model accompanied by multiple 1-bit deltas, BitDelta dramatically reduces GPU memory requir
    
[^9]: 借鉴多体物理的归纳偏置的多激发投影模拟

    Multi-Excitation Projective Simulation with a Many-Body Physics Inspired Inductive Bias

    [https://arxiv.org/abs/2402.10192](https://arxiv.org/abs/2402.10192)

    该论文引入了多激发投影模拟（mePS），通过在超图上多个粒子的随机游走，解决了投影模拟（PS）无法模拟同时结合多个概念的思维的问题。

    

    随着深度学习的进步，依赖于机器学习的应用正在越来越多地融入日常生活。然而，大多数深度学习模型具有不透明的、类似于神谕般的特性，使得解释和理解它们的决策变得困难。这个问题导致了被称为可解释人工智能（XAI）的领域的发展。该领域中的一种方法称为投影模拟（PS），将思维过程建模为一个在具有概念附加的顶点的图上的粒子的随机游走。虽然这种描述具有各种好处，包括量化的可能性，但不能自然地用来模拟同时结合多个概念的思维。为了克服这个限制，我们引入了一种称为多激发投影模拟（mePS）的推广，它将思维过程视为超图上多个粒子的随机游走。

    arXiv:2402.10192v1 Announce Type: cross  Abstract: With the impressive progress of deep learning, applications relying on machine learning are increasingly being integrated into daily life. However, most deep learning models have an opaque, oracle-like nature making it difficult to interpret and understand their decisions. This problem led to the development of the field known as eXplainable Artificial Intelligence (XAI). One method in this field known as Projective Simulation (PS) models a chain-of-thought as a random walk of a particle on a graph with vertices that have concepts attached to them. While this description has various benefits, including the possibility of quantization, it cannot be naturally used to model thoughts that combine several concepts simultaneously. To overcome this limitation, we introduce Multi-Excitation Projective Simulation (mePS), a generalization that considers a chain-of-thought to be a random walk of several particles on a hypergraph. A definition for
    
[^10]: FedAnchor:通过标签对比损失增强无标签客户端的联邦半监督学习

    FedAnchor: Enhancing Federated Semi-Supervised Learning with Label Contrastive Loss for Unlabeled Clients

    [https://arxiv.org/abs/2402.10191](https://arxiv.org/abs/2402.10191)

    本文介绍了一种名为FedAnchor的创新方法，通过引入anchor head和标签对比损失，增强了无标签客户端的联邦半监督学习。

    

    本文提出了一种创新的联邦半监督学习方法FedAnchor，引入了一种独特的双头结构，称为anchor head，并与仅在服务器上标记的锚定数据训练的分类头相配。锚定头通过一种新设计的标签对比损失进行增强。

    arXiv:2402.10191v1 Announce Type: new  Abstract: Federated learning (FL) is a distributed learning paradigm that facilitates collaborative training of a shared global model across devices while keeping data localized. The deployment of FL in numerous real-world applications faces delays, primarily due to the prevalent reliance on supervised tasks. Generating detailed labels at edge devices, if feasible, is demanding, given resource constraints and the imperative for continuous data updates. In addressing these challenges, solutions such as federated semi-supervised learning (FSSL), which relies on unlabeled clients' data and a limited amount of labeled data on the server, become pivotal. In this paper, we propose FedAnchor, an innovative FSSL method that introduces a unique double-head structure, called anchor head, paired with the classification head trained exclusively on labeled anchor data on the server. The anchor head is empowered with a newly designed label contrastive loss base
    
[^11]: 大型语言模型的上下文学习中的不确定性分解和量化

    Uncertainty Decomposition and Quantification for In-Context Learning of Large Language Models

    [https://arxiv.org/abs/2402.10189](https://arxiv.org/abs/2402.10189)

    本文研究了大型语言模型（LLM）上下文学习中的不确定性，并提出了一种新的方法来量化这种不确定性，包括演示产生的不确定性和模型配置的模糊性。

    

    上下文学习已经成为大型语言模型（LLM）的突破性能力，并通过在提示中提供一些与任务相关的演示来彻底改变了各个领域。然而，LLM响应中的可信问题，如幻觉，也被积极讨论。现有工作致力于量化LLM响应中的不确定性，但往往忽视LLM的复杂性和上下文学习的独特性。在这项工作中，我们深入研究了与上下文学习相关的LLM预测不确定性，并强调这种不确定性可能来自于提供的演示（aleatoric不确定性）和与模型配置相关的模糊性（epistemic不确定性）。我们提出了一种新的公式和相应的估计方法来量化这两种类型的不确定性。该方法为理解上下文学习里的预测提供了一种无监督的方式。

    arXiv:2402.10189v1 Announce Type: new  Abstract: In-context learning has emerged as a groundbreaking ability of Large Language Models (LLMs) and revolutionized various fields by providing a few task-relevant demonstrations in the prompt. However, trustworthy issues with LLM's response, such as hallucination, have also been actively discussed. Existing works have been devoted to quantifying the uncertainty in LLM's response, but they often overlook the complex nature of LLMs and the uniqueness of in-context learning. In this work, we delve into the predictive uncertainty of LLMs associated with in-context learning, highlighting that such uncertainties may stem from both the provided demonstrations (aleatoric uncertainty) and ambiguities tied to the model's configurations (epistemic uncertainty). We propose a novel formulation and corresponding estimation method to quantify both types of uncertainties. The proposed method offers an unsupervised way to understand the prediction of in-cont
    
[^12]: 自洽验证机器学习电子结构

    Self-consistent Validation for Machine Learning Electronic Structure

    [https://arxiv.org/abs/2402.10186](https://arxiv.org/abs/2402.10186)

    提出了一种自洽验证机器学习电子结构的技术，该技术通过将机器学习与自洽场方法结合起来，实现了低验证成本和强解释能力，并能通过主动学习来探索模型的能力。

    

    机器学习已经成为有效解决电子结构问题的重要方法。尽管其潜力巨大，但模型在未见数据上的泛化能力缺乏保证，这限制了其在实际场景中的应用。为了解决这个问题，提出了一种技术来估计预测的准确性。该方法将机器学习与自洽场方法结合起来，既能实现低验证成本，又能解释性强。这反过来使得能够通过主动学习来探索模型的能力，并为将其集成到实际研究中赋予信心。

    arXiv:2402.10186v1 Announce Type: new  Abstract: Machine learning has emerged as a significant approach to efficiently tackle electronic structure problems. Despite its potential, there is less guarantee for the model to generalize to unseen data that hinders its application in real-world scenarios. To address this issue, a technique has been proposed to estimate the accuracy of the predictions. This method integrates machine learning with self-consistent field methods to achieve both low validation cost and interpret-ability. This, in turn, enables exploration of the model's ability with active learning and instills confidence in its integration into real-world studies.
    
[^13]: 重塑RLHF中的信息结构：基于图论的奖励泛化视角

    Rethinking Information Structures in RLHF: Reward Generalization from a Graph Theory Perspective

    [https://arxiv.org/abs/2402.10184](https://arxiv.org/abs/2402.10184)

    本研究通过设计奖励建模过程中的数据集信息结构，从图论的视角提出了RLHF中奖励泛化的问题，以解决多样的环境、低成本标注和可靠的对齐性能间的不兼容性。

    

    在强化学习从人类反馈中（RLHF）存在一个三难问题：高度多样的环境、低标注成本和可靠的对齐性能之间的不兼容性。本文旨在通过设计奖励建模过程中的数据集信息结构来缓解这种不兼容性。具体而言，我们重新审视了RLHF过程，并提出了一个理论框架将其描绘为文本分布上的自动编码过程。我们的框架形式化了RLHF目标，即确保人类偏好与大型语言模型（LLM）行为之间的分布一致性。基于这个框架，我们系统地研究了RLHF奖励建模阶段中信息结构的性能影响。为了进一步理解奖励建模阶段中的奖励泛化，我们引入了一种基于随机图论的方法来建模语义空间中的泛化。其中的关键见解是...

    arXiv:2402.10184v1 Announce Type: cross  Abstract: There is a trilemma in reinforcement learning from human feedback (RLHF): the incompatibility between highly diverse contexts, low labeling cost, and reliable alignment performance. Here we aim to mitigate such incompatibility through the design of dataset information structures during reward modeling. Specifically, we first reexamine the RLHF process and propose a theoretical framework portraying it as an autoencoding process over text distributions. Our framework formalizes the RLHF objective of ensuring distributional consistency between human preference and large language model (LLM) behavior. Building on this framework, we then systematically investigate the performance impact of information structure in the reward modeling stage of RLHF. To further understand reward generalization in the reward modeling stage, we introduce a new method based on random graph theory that models generalization in the semantic space. A key insight of
    
[^14]: 大规模受限制聚类与强化学习

    Large Scale Constrained Clustering With Reinforcement Learning

    [https://arxiv.org/abs/2402.10177](https://arxiv.org/abs/2402.10177)

    本文介绍了一种使用强化学习解决大规模受限制聚类问题的方法，该方法训练一个代理器生成既可行又接近最优解的解决方案，以提高资源分配和使用的效率。

    

    给定一个网络，将资源分配在聚类级别而不是在每个节点上，可以增强资源分配和使用的效率。本文研究了在最小化聚类内部距离和最大化分配给聚类的节点数量的同时，确保聚类内部没有两个节点的距离超过阈值的全连接不相交聚类问题。尽管可以使用二进制线性模型轻松地形成问题，但在处理大规模实例时，传统组合优化求解器很难应对。我们提出了一种通过强化学习解决这个受限聚类问题的方法。我们的方法涉及训练一个代理器生成既可行又接近最优解的解决方案。代理器学习特定于该任务所遇到的实例的问题启发式算法。在结果部分，我们展示了我们的算法即使在大规模情况下也能找到接近最优解的解决方案。

    arXiv:2402.10177v1 Announce Type: cross  Abstract: Given a network, allocating resources at clusters level, rather than at each node, enhances efficiency in resource allocation and usage. In this paper, we study the problem of finding fully connected disjoint clusters to minimize the intra-cluster distances and maximize the number of nodes assigned to the clusters, while also ensuring that no two nodes within a cluster exceed a threshold distance. While the problem can easily be formulated using a binary linear model, traditional combinatorial optimization solvers struggle when dealing with large-scale instances. We propose an approach to solve this constrained clustering problem via reinforcement learning. Our method involves training an agent to generate both feasible and (near) optimal solutions. The agent learns problem-specific heuristics, tailored to the instances encountered in this task. In the results section, we show that our algorithm finds near optimal solutions, even for l
    
[^15]: OpenMathInstruct-1: 一个拥有180万个数学教学调优数据集

    OpenMathInstruct-1: A 1.8 Million Math Instruction Tuning Dataset

    [https://arxiv.org/abs/2402.10176](https://arxiv.org/abs/2402.10176)

    OpenMathInstruct-1是一个包含180万个数学问题和解决方法对的数据集，通过合成开源LLM的代码解释器解决方案来构建，填补了目前开源LLM在数学技能方面与闭源LLM之间的差距。

    

    最近的研究表明，利用合成生成的数据集来训练大规模语言模型（LLM）具有巨大潜力，尤其是为了获得特定的技能。目前的大规模数学教学调优数据集，如MetaMathQA和MAmmoTH，是使用来自商业限制许可的闭源LLM的输出构建的。限制在这些数据生成流程中使用开源LLM的一个关键原因是目前最好的闭源LLM（如GPT-4）和最好的开源LLM之间在数学技能上存在很大差距。基于开源LLM的最近进展，我们提出了新颖的提示方式和一些强力缩放，构建了OpenMathInstruct-1，一个拥有180万个问题-解决方法对的数学教学调优数据集。该数据集是通过使用GSM8K和MATH这两个流行的数学推理基准的代码解释器解决方案进行合成构建的。

    arXiv:2402.10176v1 Announce Type: cross  Abstract: Recent work has shown the immense potential of synthetically generated datasets for training large language models (LLMs), especially for acquiring targeted skills. Current large-scale math instruction tuning datasets such as MetaMathQA (Yu et al., 2024) and MAmmoTH (Yue et al., 2024) are constructed using outputs from closed-source LLMs with commercially restrictive licenses. A key reason limiting the use of open-source LLMs in these data generation pipelines has been the wide gap between the mathematical skills of the best closed-source LLMs, such as GPT-4, and the best open-source LLMs. Building on the recent progress in open-source LLMs, our proposed prompting novelty, and some brute-force scaling, we construct OpenMathInstruct-1, a math instruction tuning dataset with 1.8M problem-solution pairs. The dataset is constructed by synthesizing code-interpreter solutions for GSM8K and MATH, two popular math reasoning benchmarks, using t
    
[^16]: DeepSRGM -- 基于深度学习的印度古典音乐中的序列分类和排序

    DeepSRGM -- Sequence Classification and Ranking in Indian Classical Music with Deep Learning

    [https://arxiv.org/abs/2402.10168](https://arxiv.org/abs/2402.10168)

    DeepSRGM是一种基于深度学习的Raga识别方法，通过使用LSTM-RNN学习音乐数据中的时间序列，达到了88.1%和97%的准确率，在Raga识别任务中取得了最新技术的地位。

    

    《arXiv:2402.10168v1 公告类型: 交叉》 摘要：印度古典音乐(ICM)的一个重要方面是Raga，它作为作曲和即兴演奏的旋律框架。Raga的识别是ICM中一项重要的音乐信息检索任务，它可以帮助从音乐推荐到组织大型音乐收藏等多种下游应用。在这项工作中，我们提出了一种基于深度学习的Raga识别方法。我们的方法采用有效的预处理，使用基于长短期记忆(LSTM)的递归神经网络(RNN)学习音乐数据中的时间序列。我们在采样自原始音频的较小序列上进行网络的训练和测试，而最终的推理则是在整个音频上进行的。我们的方法在Comp Music Carnatic数据集和其10个Raga子集上的推理过程中分别达到了88.1%和97%的准确率，使其成为Raga识别任务的最新技术。我们的方法还使序列排序成为可能。

    arXiv:2402.10168v1 Announce Type: cross  Abstract: A vital aspect of Indian Classical Music (ICM) is Raga, which serves as a melodic framework for compositions and improvisations alike. Raga Recognition is an important music information retrieval task in ICM as it can aid numerous downstream applications ranging from music recommendations to organizing huge music collections. In this work, we propose a deep learning based approach to Raga recognition. Our approach employs efficient pre possessing and learns temporal sequences in music data using Long Short Term Memory based Recurrent Neural Networks (LSTM-RNN). We train and test the network on smaller sequences sampled from the original audio while the final inference is performed on the audio as a whole. Our method achieves an accuracy of 88.1% and 97 % during inference on the Comp Music Carnatic dataset and its 10 Raga subset respectively making it the state-of-the-art for the Raga recognition task. Our approach also enables sequence
    
[^17]: 随机特征和多项式规则

    Random features and polynomial rules

    [https://arxiv.org/abs/2402.10164](https://arxiv.org/abs/2402.10164)

    本论文分析了随机特征模型在具有高斯数据的一般监督学习问题中的泛化性能，并将随机特征模型映射到等效的多项式模型，得到了与严格界限和数值实验一致的结果。

    

    随机特征模型在深度学习理论中起着重要的作用，描述了神经网络接近无限宽度极限时的行为。在本工作中，我们对具有高斯数据的一般监督学习问题的随机特征模型的泛化性能进行了详细分析。我们利用无序系统的统计力学工具将随机特征模型映射到等效的多项式模型，并绘制了平均泛化曲线作为问题的两个主要控制参数的函数：随机特征的数量N和训练集的大小P，假设它们都按照输入维度D的幂进行缩放。我们的结果扩展了N，P和D之间比例缩放的情况。它们与特定学习任务已知的严格界限一致，并与数值实验定量一致。

    arXiv:2402.10164v1 Announce Type: cross  Abstract: Random features models play a distinguished role in the theory of deep learning, describing the behavior of neural networks close to their infinite-width limit. In this work, we present a thorough analysis of the generalization performance of random features models for generic supervised learning problems with Gaussian data. Our approach, built with tools from the statistical mechanics of disordered systems, maps the random features model to an equivalent polynomial model, and allows us to plot average generalization curves as functions of the two main control parameters of the problem: the number of random features $N$ and the size $P$ of the training set, both assumed to scale as powers in the input dimension $D$. Our results extend the case of proportional scaling between $N$, $P$ and $D$. They are in accordance with rigorous bounds known for certain particular learning tasks and are in quantitative agreement with numerical experime
    
[^18]: $f$-MICL: Understanding and Generalizing InfoNCE-based Contrastive Learning

    $f$-MICL: Understanding and Generalizing InfoNCE-based Contrastive Learning

    [https://arxiv.org/abs/2402.10150](https://arxiv.org/abs/2402.10150)

    本文提出了一种名为$f$-MICL的方法，用于理解和推广基于InfoNCE的对比学习。通过使用$f$-divergences将基于KL的互信息推广为$f$-Mutual Information in Contrastive Learning ($f$-MICL)，我们回答了超越基于KL的目标函数以及设计更好相似度函数的问题。

    

    在自监督对比学习中，一种被广泛采用的目标函数是InfoNCE，它使用启发式的余弦相似度进行表示比较，并且与最大化基于KL的互信息密切相关。本文旨在回答两个有趣的问题：(1)我们能否超越基于KL的目标？(2)除了流行的余弦相似度，我们能否设计出更好的相似度函数？我们通过将基于KL的互信息推广为$f$-Mutual Information in Contrastive Learning ($f$-MICL)，使用$f$-divergences来回答这两个问题。针对第一个问题，我们提供了一系列$f$-MICL目标函数，它们具有与InfoNCE相似的良好特性（如对齐和均匀性），同时在性能上达到类似甚至更优的效果。对于第二个问题，假设联合特征分布与高斯核成比例。

    arXiv:2402.10150v1 Announce Type: new  Abstract: In self-supervised contrastive learning, a widely-adopted objective function is InfoNCE, which uses the heuristic cosine similarity for the representation comparison, and is closely related to maximizing the Kullback-Leibler (KL)-based mutual information. In this paper, we aim at answering two intriguing questions: (1) Can we go beyond the KL-based objective? (2) Besides the popular cosine similarity, can we design a better similarity function? We provide answers to both questions by generalizing the KL-based mutual information to the $f$-Mutual Information in Contrastive Learning ($f$-MICL) using the $f$-divergences. To answer the first question, we provide a wide range of $f$-MICL objectives which share the nice properties of InfoNCE (e.g., alignment and uniformity), and meanwhile result in similar or even superior performance. For the second question, assuming that the joint feature distribution is proportional to the Gaussian kernel,
    
[^19]: 基于混沌映射的保护隐私的分布式深度学习方法，针对不完整和非独立同分布数据集

    A chaotic maps-based privacy-preserving distributed deep learning for incomplete and Non-IID datasets

    [https://arxiv.org/abs/2402.10145](https://arxiv.org/abs/2402.10145)

    本研究提出了一种基于混沌映射的保护隐私的分布式深度学习方法，在处理不完整和非独立同分布数据集的情况下，通过差分隐私和基于混沌加密的隐私保护层提高了深度神经网络的性能。

    

    集中式学习是一种机器学习方法，使得多个具有敏感数据但希望共享知识的参与者能够训练深度学习模型而不损害数据隐私。本研究使用了一种安全的集中式学习方法，并提出了一种解决非独立同分布数据挑战的方法。此外，将差分隐私与基于混沌加密的隐私保护层进行了比较。实验评估了应用差分隐私的集中式深度学习模型在独立同分布和非独立同分布数据上的性能。在每个实验中，集中式学习过程都提高了深度神经网络的平均性能指标，即使在非独立同分布数据的情况下也是如此。

    arXiv:2402.10145v1 Announce Type: new  Abstract: Federated Learning is a machine learning approach that enables the training of a deep learning model among several participants with sensitive data that wish to share their own knowledge without compromising the privacy of their data. In this research, the authors employ a secured Federated Learning method with an additional layer of privacy and proposes a method for addressing the non-IID challenge. Moreover, differential privacy is compared with chaotic-based encryption as layer of privacy. The experimental approach assesses the performance of the federated deep learning model with differential privacy using both IID and non-IID data. In each experiment, the Federated Learning process improves the average performance metrics of the deep neural network, even in the case of non-IID data.
    
[^20]: 通过动态学习器追踪概率变化

    Tracking Changing Probabilities via Dynamic Learners

    [https://arxiv.org/abs/2402.10142](https://arxiv.org/abs/2402.10142)

    该论文介绍了通过动态学习器追踪概率变化的方法，通过输出候选项目及其概率来预测离散项目序列中下一个可能出现的项目。

    

    考虑一个预测器，即一个学习器，其输入是一系列离散项目。预测器的任务是在每个时间点进行概率多类别预测，即通过输出有零个或多个候选项目及其概率来预测接下来可能发生的项目，然后揭示实际项目并从中学习。为了输出概率，预测器会跟踪其所见项目的比例。预测器具有恒定（有限）的空间，我们寻求高效的预测和更新技术：流是无界的，项目的集合对预测器是未知的，它们的总数也可能无限增长。此外，存在非平稳性：项目的潜在频率可能会不时发生显著变化。例如，新项目可能开始出现，一些当前频繁出现的项目可能再次停止出现。由于有空间限制，预测器只需要提供概率。

    arXiv:2402.10142v1 Announce Type: cross  Abstract: Consider a predictor, a learner, whose input is a stream of discrete items. The predictor's task, at every time point, is probabilistic multiclass prediction, i.e., to predict which item may occur next by outputting zero or more candidate items, each with a probability, after which the actual item is revealed and the predictor learns from this observation. To output probabilities, the predictor keeps track of the proportions of the items it has seen. The predictor has constant (limited) space and we seek efficient prediction and update techniques: The stream is unbounded, the set of items is unknown to the predictor and their totality can also grow unbounded. Moreover, there is non-stationarity: the underlying frequencies of items may change, substantially, from time to time. For instance, new items may start appearing and a few currently frequent items may cease to occur again. The predictor, being space-bounded, need only provide pro
    
[^21]: 用于生物医学数据的对等联邦学习的策略基准测试

    Benchmarking federated strategies in Peer-to-Peer Federated learning for biomedical data

    [https://arxiv.org/abs/2402.10135](https://arxiv.org/abs/2402.10135)

    这项研究对生物医学数据的对等联邦学习进行了基准测试，并测试了各种聚合策略，包括加权平均聚合，以确定最强大的策略。

    

    数据保护和隐私要求的不断增加引起了对分布式人工智能的巨大研究兴趣，尤其是对联邦学习的研究，它是一种新兴的机器学习方法，允许构建一个模型，该模型由持有自己私有数据的多个参与者之间建立。在最初的联邦学习提案中，架构是集中式的，聚合是通过联邦平均化来完成的，意味着一个中央服务器将使用最直接的平均策略来协调联邦。本研究专注于在对等环境中测试不同的联邦策略。作者提出了各种联邦学习的聚合策略，包括加权平均聚合，使用不同的因素和基于参与者贡献的策略。使用不同大小的数据对这些策略进行测试，以确定最强大的策略。

    arXiv:2402.10135v1 Announce Type: cross  Abstract: The increasing requirements for data protection and privacy has attracted a huge research interest on distributed artificial intelligence and specifically on federated learning, an emerging machine learning approach that allows the construction of a model between several participants who hold their own private data. In the initial proposal of federated learning the architecture was centralised and the aggregation was done with federated averaging, meaning that a central server will orchestrate the federation using the most straightforward averaging strategy. This research is focused on testing different federated strategies in a peer-to-peer environment. The authors propose various aggregation strategies for federated learning, including weighted averaging aggregation, using different factors and strategies based on participant contribution. The strategies are tested with varying data sizes to identify the most robust ones. This resear
    
[^22]: 连续学习是否适应现实挑战？

    Is Continual Learning Ready for Real-world Challenges?

    [https://arxiv.org/abs/2402.10130](https://arxiv.org/abs/2402.10130)

    本文研究了连续学习在现实世界场景中的应用，发现当前的评估方法与实际挑战不匹配，现有解决方案无法有效解决复杂的现实世界环境下的问题。

    

    尽管连续学习在学术界有着悠久而良好的历史，但其在实际应用中的应用仍然相对有限。本文认为这种差距是由于当前评估方法与连续学习的实际挑战不匹配，导致现有解决方案无法有效应对现实世界环境的复杂性。通过使用全新的三维语义分割基准测试OCL-3DSS，我们验证了自己的假设并评估了过去的进展。我们通过利用更加现实的协议来研究文献中的各种连续学习方案，这些方案需要在线和持续学习以应对动态的现实世界场景（例如机器人和三维视觉应用）。结果令人沮丧：所有考虑的方法表现不佳，明显偏离联合离线训练的上限。这对现有方法在实际应用中的适用性提出了问题。

    arXiv:2402.10130v1 Announce Type: cross  Abstract: Despite continual learning's long and well-established academic history, its application in real-world scenarios remains rather limited. This paper contends that this gap is attributable to a misalignment between the actual challenges of continual learning and the evaluation protocols in use, rendering proposed solutions ineffective for addressing the complexities of real-world setups. We validate our hypothesis and assess progress to date, using a new 3D semantic segmentation benchmark, OCL-3DSS. We investigate various continual learning schemes from the literature by utilizing more realistic protocols that necessitate online and continual learning for dynamic, real-world scenarios (eg., in robotics and 3D vision applications). The outcomes are sobering: all considered methods perform poorly, significantly deviating from the upper bound of joint offline training. This raises questions about the applicability of existing methods in rea
    
[^23]: GES：用于高效辐射场渲染的广义指数喷洒

    GES: Generalized Exponential Splatting for Efficient Radiance Field Rendering

    [https://arxiv.org/abs/2402.10128](https://arxiv.org/abs/2402.10128)

    本研究引入了GES（广义指数喷洒），一种利用广义指数函数来建模3D场景的新表示方法，可以显著提高3D重建和生成的效率。相比于传统的高斯喷洒方法，GES所需的粒子数量更少，能更准确地表示具有锐利边缘的信号。

    

    arXiv:2402.10128v1 公告类型：跨领域 摘要：3D高斯喷洒技术的进展显著加快了3D重建和生成的速度。然而，这可能需要大量高斯函数，导致占用大量内存。本文介绍了GES（广义指数喷洒），这是一种新的表示方法，它利用广义指数函数（GEF）来建模3D场景，只需要很少的粒子来表示一个场景，因此在效率上明显优于基于高斯的喷洒方法，并且具有高斯-based utilities的即插即用替换能力。GES在理论上和实证上都得到了验证，在原则上的1D设置和逼真的3D场景中。实验证明，GEF比高斯函数更准确地表示具有锐利边缘的信号，这在高斯函数中是具有挑战性的，因为它们具有低通特性。我们的实证分析表明，GEF在拟合自然发生的信号（例如正方形、三角形）方面优于高斯函数。

    arXiv:2402.10128v1 Announce Type: cross  Abstract: Advancements in 3D Gaussian Splatting have significantly accelerated 3D reconstruction and generation. However, it may require a large number of Gaussians, which creates a substantial memory footprint. This paper introduces GES (Generalized Exponential Splatting), a novel representation that employs Generalized Exponential Function (GEF) to model 3D scenes, requiring far fewer particles to represent a scene and thus significantly outperforming Gaussian Splatting methods in efficiency with a plug-and-play replacement ability for Gaussian-based utilities. GES is validated theoretically and empirically in both principled 1D setup and realistic 3D scenes.   It is shown to represent signals with sharp edges more accurately, which are typically challenging for Gaussians due to their inherent low-pass characteristics. Our empirical analysis demonstrates that GEF outperforms Gaussians in fitting natural-occurring signals (e.g. squares, triangl
    
[^24]: 非线性尖峰协方差矩阵与深度神经网络中的信号传播

    Nonlinear spiked covariance matrices and signal propagation in deep neural networks

    [https://arxiv.org/abs/2402.10127](https://arxiv.org/abs/2402.10127)

    该论文研究了非线性尖峰协方差矩阵与深度神经网络中的信号传播。通过对尖峰特征结构的定量描述，揭示了输入数据中的低维信号结构如何经过神经网络的隐藏层传播。此外，研究了一种表示学习的简单情境，其中权重矩阵发展出一个秩为一的信号分量。

    

    许多最近的研究都研究了由前馈神经网络的非线性特征映射定义的共轭核（CK）的特征值谱。然而，现有的结果只能建立经验特征值分布的弱收敛性，并没有提供对通常捕捉学习问题的低维信号结构的“尖峰”特征值和特征向量的精确定量描述。在这项工作中，我们对非线性版本的尖峰协方差模型（包括CK作为特例）进行了这些信号特征值和特征向量的表征。利用这个一般结果，我们定量描述了输入数据中的尖峰特征结构如何通过具有随机权重的神经网络的隐藏层传播。作为第二个应用，我们研究了表示学习的一个简单情境，其中权重矩阵在训练过程中发展出一个秩为一的信号分量。

    arXiv:2402.10127v1 Announce Type: cross  Abstract: Many recent works have studied the eigenvalue spectrum of the Conjugate Kernel (CK) defined by the nonlinear feature map of a feedforward neural network. However, existing results only establish weak convergence of the empirical eigenvalue distribution, and fall short of providing precise quantitative characterizations of the ''spike'' eigenvalues and eigenvectors that often capture the low-dimensional signal structure of the learning problem. In this work, we characterize these signal eigenvalues and eigenvectors for a nonlinear version of the spiked covariance model, including the CK as a special case. Using this general result, we give a quantitative description of how spiked eigenstructure in the input data propagates through the hidden layers of a neural network with random weights. As a second application, we study a simple regime of representation learning where the weight matrix develops a rank-one signal component over trainin
    
[^25]: 在Transformer中重用Softmax硬件单元进行GELU计算

    Reusing Softmax Hardware Unit for GELU Computation in Transformers

    [https://arxiv.org/abs/2402.10118](https://arxiv.org/abs/2402.10118)

    本文提出了一种在Transformer中重用Softmax硬件单元进行GELU计算的方法，实验证明这种方法不会降低NLP应用的准确性。

    

    Transformers大大提高了自然语言处理（NLP）和计算机视觉应用的性能。Transformer的计算涉及矩阵乘法和非线性激活函数，如softmax和GELU（高斯误差线性单元），这些函数可以直接在硬件中加速。目前，每个函数的计算都是分开完成的，很少能够重复使用硬件。为了解决这个问题，本文将GELU计算映射到softmax运算符上。这样，已经设计用于softmax的高效硬件单元也可以用于计算GELU。GELU的计算可以充分利用softmax的向量化特性，同时并行产生多个GELU的结果。实验结果表明，通过预先存在并逐步修改的softmax硬件单元计算GELU（a）不会降低代表性NLP应用的准确性，（b）全部

    arXiv:2402.10118v1 Announce Type: cross  Abstract: Transformers have improved drastically the performance of natural language processing (NLP) and computer vision applications. The computation of transformers involves matrix multiplications and non-linear activation functions such as softmax and GELU (Gaussion Error Linear Unit) that are accelerated directly in hardware. Currently, function evaluation is done separately for each function and rarely allows for hardware reuse. To mitigate this problem, in this work, we map the computation of GELU to a softmax operator. In this way, the efficient hardware units designed already for softmax can be reused for computing GELU as well. Computation of GELU can enjoy the inherent vectorized nature of softmax and produce in parallel multiple GELU outcomes. Experimental results show that computing GELU via a pre-existing and incrementally modified softmax hardware unit (a) does not reduce the accuracy of representative NLP applications and (b) all
    
[^26]: 使用基于Transformer编码器的EEG编码器和GAN从EEG记录中生成视觉刺激

    Generating Visual Stimuli from EEG Recordings using Transformer-encoder based EEG encoder and GAN

    [https://arxiv.org/abs/2402.10115](https://arxiv.org/abs/2402.10115)

    本研究使用基于Transformer编码器的EEG编码器和GAN网络，通过合成图像从EEG信号中恢复出各种对象类别的图像，同时结合对抗损失和感知损失，提高生成图像的质量。

    

    在这项研究中，我们解决了感知性脑解码领域的一个现代研究挑战，即使用对抗式深度学习框架从EEG信号中合成图像。具体目标是利用主体观看图像时获得的EEG记录重新创建属于各种对象类别的图像。为了实现这一目标，我们使用基于Transformer编码器的EEG编码器生成EEG编码，然后将其作为GAN网络的生成器组件的输入。除了对抗损失之外，我们还采用了感知损失来提高生成图像的质量。

    arXiv:2402.10115v1 Announce Type: new  Abstract: In this study, we tackle a modern research challenge within the field of perceptual brain decoding, which revolves around synthesizing images from EEG signals using an adversarial deep learning framework. The specific objective is to recreate images belonging to various object categories by leveraging EEG recordings obtained while subjects view those images. To achieve this, we employ a Transformer-encoder based EEG encoder to produce EEG encodings, which serve as inputs to the generator component of the GAN network. Alongside the adversarial loss, we also incorporate perceptual loss to enhance the quality of the generated images.
    
[^27]: 选择性反射调节：LLM指令调节的学生选择数据回收

    Selective Reflection-Tuning: Student-Selected Data Recycling for LLM Instruction-Tuning

    [https://arxiv.org/abs/2402.10110](https://arxiv.org/abs/2402.10110)

    本文介绍了一种名为选择性反射调节的新方法，该方法通过教师LLM的反射和自省与学生LLM的数据选择能力相结合，自动优化现有的指令调节数据，从而实现了高效的指令调节和卓越性能的LLM。

    

    指令调节对于大型语言模型（LLM）来说非常关键，以实现更好的指令跟踪和任务适应能力，但其成功在很大程度上取决于训练数据的质量。许多最近的方法都致力于改进数据质量，但往往忽视了数据与正在微调的学生模型的兼容性。本文介绍了一种新的范式——选择性反射调节，通过结合教师LLM的反射和自省，以自动优化现有的指令调节数据。这种师生合作产生了高质量且与学生LLM兼容的指令响应对，从而实现了高效的指令调节和卓越性能的LLM。选择性反射调节是一种数据增强和合成方法，通常能改善LLM微调和自我优化，而无需额外的计算资源。

    arXiv:2402.10110v1 Announce Type: cross  Abstract: Instruction tuning is critical to large language models (LLMs) for achieving better instruction following and task adaptation capabilities but its success heavily relies on the training data quality. Many recent methods focus on improving the data quality but often overlook the compatibility of the data with the student model being finetuned. This paper introduces Selective Reflection-Tuning, a novel paradigm that synergizes a teacher LLM's reflection and introspection for improving existing data quality with the data selection capability of the student LLM, to automatically refine existing instruction-tuning data. This teacher-student collaboration produces high-quality and student-compatible instruction-response pairs, resulting in sample-efficient instruction tuning and LLMs of superior performance. Selective Reflection-Tuning is a data augmentation and synthesis that generally improves LLM finetuning and self-improvement without co
    
[^28]: 用可解释的风险预测方法降低诊断错误

    Towards Reducing Diagnostic Errors with Interpretable Risk Prediction

    [https://arxiv.org/abs/2402.10109](https://arxiv.org/abs/2402.10109)

    本研究提出了一种使用LLMs方法来识别病人电子病历数据中指示特定诊断风险增加或减少的证据的方法，旨在通过增加证据的获取与减少诊断错误来降低诊断错误。模型使用神经加性模型进行预测，以证据为后盾，并给出个体化风险估计，特别针对诊断延迟和来自不完整鉴别的错误进行优化。

    

    许多诊断错误发生是因为临床医生无法轻易获取病人电子病历中的相关信息。本研究提出了一种使用LLMs方法来识别病人电子病历数据中指示特定诊断风险增加或减少的证据的方法，最终目标是增加证据的获取与减少诊断错误。我们提出了一种神经加性模型来进行带有个体化风险估计的以证据为后盾的预测，在临床医生仍然不确定的时间点上，旨在特别减轻诊断延迟和源于不完整鉴别的错误。为了训练这样一个模型，需要推断出事件性的“真实”诊断的时间粒度细致的回顾性标签。我们使用LLMs来保证输入文本是在可以进行自信的诊断之前的。我们使用LLMs来检索初始的证据池，然后进行细化。

    arXiv:2402.10109v1 Announce Type: new  Abstract: Many diagnostic errors occur because clinicians cannot easily access relevant information in patient Electronic Health Records (EHRs). In this work we propose a method to use LLMs to identify pieces of evidence in patient EHR data that indicate increased or decreased risk of specific diagnoses; our ultimate aim is to increase access to evidence and reduce diagnostic errors. In particular, we propose a Neural Additive Model to make predictions backed by evidence with individualized risk estimates at time-points where clinicians are still uncertain, aiming to specifically mitigate delays in diagnosis and errors stemming from an incomplete differential. To train such a model, it is necessary to infer temporally fine-grained retrospective labels of eventual "true" diagnoses. We do so with LLMs, to ensure that the input text is from before a confident diagnosis can be made. We use an LLM to retrieve an initial pool of evidence, but then refin
    
[^29]: 基于深度学习的多导弹避让情况感知

    Deep Learning Based Situation Awareness for Multiple Missiles Evasion

    [https://arxiv.org/abs/2402.10101](https://arxiv.org/abs/2402.10101)

    该研究提出了一种基于深度学习的决策支持工具，用于帮助无人机操作员在多个导弹威胁下进行决策，通过学习高保真度模拟来评估各种策略的风险，并建议最安全的行动方针。

    

    随着空对空导弹的有效射程增加，人类操作员难以保持无人机所需的情况感知能力。在本研究中，我们提出了一种决策支持工具，以帮助无人机操作员在视线外（BVR）空战情景中评估不同选择的风险，并根据这些选择做出决策。早期的工作侧重于单一导弹的威胁，而在本研究中，我们将这些想法拓展到多个导弹威胁上。所提出的方法使用深度神经网络（DNN）通过高保真度模拟学习，为操作员提供一组不同策略的结果估计。我们的结果表明，所提出的系统可以处理多个来袭导弹，评估一系列选项，并推荐风险最小的行动方针。

    arXiv:2402.10101v1 Announce Type: cross  Abstract: As the effective range of air-to-air missiles increases, it becomes harder for human operators to maintain the situational awareness needed to keep a UAV safe. In this work, we propose a decision support tool to help UAV operators in Beyond Visual Range (BVR) air combat scenarios assess the risks of different options and make decisions based on those. Earlier work focused on the threat posed by a single missile, and in this work, we extend the ideas to several missile threats. The proposed method uses Deep Neural Networks (DNN) to learn from high-fidelity simulations to provide the operator with an outcome estimate for a set of different strategies. Our results demonstrate that the proposed system can manage multiple incoming missiles, evaluate a family of options, and recommend the least risky course of action.
    
[^30]: 调谐：在临床设置中使用有限数据的音频分类器性能分析

    Tuning In: Analysis of Audio Classifier Performance in Clinical Settings with Limited Data

    [https://arxiv.org/abs/2402.10100](https://arxiv.org/abs/2402.10100)

    本研究评估了在临床设置中使用深度学习模型进行音频分类的效果，并发现在微调之前，预训练模型在大数据集上的性能对临床数据的影响较好。研究结果表明，CNN模型可以在小数据集环境中与转换模型相媲美或超越。

    

    本研究评估了在临床设置中使用深度学习模型进行音频分类的效果，限制条件是以反映实际世界数据收集的小数据集为基础。我们分析了包括DenseNet和ConvNeXt在内的CNN模型，以及ViT、SWIN和AST等转换模型，并将它们与诸如YAMNet和VGGish的预训练音频模型进行比较。我们的方法强调了在特定临床数据上微调之前，在大数据集上进行预训练的好处。我们从卒中患者中新收集了两个前所未有的患者音频数据集。我们研究了各种预处理技术，发现基于它们从预训练中学习到的先验知识，RGB和灰度谱图转换对模型性能产生了不同的影响。我们的研究结果表明，CNN模型在小数据集环境中可以与转换模型相媲美或超越，其中DenseNet-Contrastive和AST模型表现突出。本研究强调了...

    arXiv:2402.10100v1 Announce Type: cross  Abstract: This study assesses deep learning models for audio classification in a clinical setting with the constraint of small datasets reflecting real-world prospective data collection. We analyze CNNs, including DenseNet and ConvNeXt, alongside transformer models like ViT, SWIN, and AST, and compare them against pre-trained audio models such as YAMNet and VGGish. Our method highlights the benefits of pre-training on large datasets before fine-tuning on specific clinical data. We prospectively collected two first-of-their-kind patient audio datasets from stroke patients. We investigated various preprocessing techniques, finding that RGB and grayscale spectrogram transformations affect model performance differently based on the priors they learn from pre-training. Our findings indicate CNNs can match or exceed transformer models in small dataset contexts, with DenseNet-Contrastive and AST models showing notable performance. This study highlights
    
[^31]: 不需要参数调整的数据输入错误unlearning方法与自适应选择性突触抑制

    Parameter-tuning-free data entry error unlearning with adaptive selective synaptic dampening

    [https://arxiv.org/abs/2402.10098](https://arxiv.org/abs/2402.10098)

    本研究提出了一种不需要参数调整的数据输入错误unlearning方法，通过自适应选择性突触抑制（ASSD）提高模型性能，并展示了其在不同模型上的性能表现。

    

    数据输入是机器学习流程的基本组成部分，但经常会导致引入标签错误。当模型在包含这种错误的数据集上进行训练时，其性能会降低。因此，需要有效地去除错误数据的影响以提高模型性能，而无需完全重新训练模型。虽然在已知错误条目的正确标签的情况下存在模型编辑方法，但我们专注于无法知道错误数据的正确标签的数据输入错误情况。我们的贡献有两个方面。首先，我们引入了对选择性突触抑制unlearning方法的扩展，该方法不需要参数调整，使unlearning方法对从业人员更易于使用。我们展示了这一扩展自适应选择性突触抑制（ASSD）在不同的ResNet18和Visio模型上的性能。

    arXiv:2402.10098v1 Announce Type: new  Abstract: Data entry constitutes a fundamental component of the machine learning pipeline, yet it frequently results in the introduction of labelling errors. When a model has been trained on a dataset containing such errors its performance is reduced. This leads to the challenge of efficiently unlearning the influence of the erroneous data to improve the model performance without needing to completely retrain the model. While model editing methods exist for cases in which the correct label for a wrong entry is known, we focus on the case of data entry errors where we do not know the correct labels for the erroneous data. Our contribution is twofold. First, we introduce an extension to the selective synaptic dampening unlearning method that removes the need for parameter tuning, making unlearning accessible to practitioners. We demonstrate the performance of this extension, adaptive selective synaptic dampening (ASSD), on various ResNet18 and Visio
    
[^32]: 异构无线网络中具有独立采样的自适应联邦学习

    Adaptive Federated Learning in Heterogeneous Wireless Networks with Independent Sampling

    [https://arxiv.org/abs/2402.10097](https://arxiv.org/abs/2402.10097)

    这项研究提出了一种适用于异构无线网络的自适应联邦学习方法，其中包括了独立客户端采样和带宽分配方案，以提高训练效率和适应数据和系统的异构特性。

    

    联邦学习算法通常通过对客户端进行随机子集采样来解决迟到者问题并提高通信效率。然而，最近的研究在联合系统和数据异构设计方面存在一些限制，可能与实际的异构无线网络不一致。在本文中，我们提倡一种新的独立客户端采样策略，以最小化联邦学习的实际训练时间，同时考虑通信和计算中的数据异构性和系统异构性。我们首先推导了带有独立客户端采样的非凸损失函数的新收敛界限，然后提出了一种自适应带宽分配方案。此外，我们还提出了一种基于收敛轮数上界和每轮预期训练时间的高效独立客户端采样算法，以最小化联邦学习的实际训练时间，同时考虑数据异构性和系统异构性。

    arXiv:2402.10097v1 Announce Type: new  Abstract: Federated Learning (FL) algorithms commonly sample a random subset of clients to address the straggler issue and improve communication efficiency. While recent works have proposed various client sampling methods, they have limitations in joint system and data heterogeneity design, which may not align with practical heterogeneous wireless networks. In this work, we advocate a new independent client sampling strategy to minimize the wall-clock training time of FL, while considering data heterogeneity and system heterogeneity in both communication and computation. We first derive a new convergence bound for non-convex loss functions with independent client sampling and then propose an adaptive bandwidth allocation scheme. Furthermore, we propose an efficient independent client sampling algorithm based on the upper bounds on the convergence rounds and the expected per-round training time, to minimize the wall-clock time of FL, while consider
    
[^33]: 分类扩散模型

    Classification Diffusion Models

    [https://arxiv.org/abs/2402.10095](https://arxiv.org/abs/2402.10095)

    提出了一种分类扩散模型（CDMs），该模型采用了去噪扩散模型（DDM）的形式，并利用一个分类器来预测加在干净信号上的噪声量，取得了在图像、视频和音频生成方面的最先进结果。

    

    arXiv：2402.10095v1 公告类型：新的 摘要：一种学习数据分布的突出方法家族依赖于密度比估计（DRE），其中模型被训练来$\textit{分类}$数据样本和来自某个参考分布的样本。这些技术在简单的低维环境中取得了成功，但在复杂的高维数据（如图像）中无法取得良好的结果。学习分布的另一种方法家族是去噪扩散模型（DDM），其中模型被训练来$\textit{去噪}$数据样本。这些方法在图像、视频和音频生成方面取得了最先进的结果。在这项工作中，我们提出了$\textit{分类扩散模型}$（CDMs），这是一种生成技术，它采用了DDM的去噪基本形式，同时利用一个分类器来预测加在干净信号上的噪声量，类似于DRE方法。我们的方法基于这样一个观察，即MSE最优化的d

    arXiv:2402.10095v1 Announce Type: new  Abstract: A prominent family of methods for learning data distributions relies on density ratio estimation (DRE), where a model is trained to $\textit{classify}$ between data samples and samples from some reference distribution. These techniques are successful in simple low-dimensional settings but fail to achieve good results on complex high-dimensional data, like images. A different family of methods for learning distributions is that of denoising diffusion models (DDMs), in which a model is trained to $\textit{denoise}$ data samples. These approaches achieve state-of-the-art results in image, video, and audio generation. In this work, we present $\textit{Classification Diffusion Models}$ (CDMs), a generative technique that adopts the denoising-based formalism of DDMs while making use of a classifier that predicts the amount of noise added to a clean signal, similarly to DRE methods. Our approach is based on the observation that an MSE-optimal d
    
[^34]: MIM-Refiner：一种从中间预训练表示中获得对比学习提升的方法

    MIM-Refiner: A Contrastive Learning Boost from Intermediate Pre-Trained Representations

    [https://arxiv.org/abs/2402.10093](https://arxiv.org/abs/2402.10093)

    MIM-Refiner是一种对比学习提升方法，通过利用MIM模型中的中间层表示和多个对比头，能够将MIM模型的特征从次优的状态提升到最先进的状态，并在ImageNet-1K数据集上取得了新的最先进结果。

    

    我们引入了MIM-Refiner，这是一种用于预训练MIM模型的对比学习提升方法。MIM-Refiner的动机在于MIM模型中的最佳表示通常位于中间层。因此，MIM-Refiner利用连接到不同中间层的多个对比头。在每个头中，修改后的最近邻目标帮助构建相应的语义聚类。此过程短而有效，在几个epochs内，我们将MIM模型的特征从次优的状态提升到最先进的状态。使用data2vec 2.0在ImageNet-1K上预训练的ViT-H经过改进后，在线性探测和低样本分类方面取得了新的最先进结果（分别为84.7%和64.2%），超过了在ImageNet-1K上预训练的其他模型的表现。

    arXiv:2402.10093v1 Announce Type: cross  Abstract: We introduce MIM (Masked Image Modeling)-Refiner, a contrastive learning boost for pre-trained MIM models. The motivation behind MIM-Refiner is rooted in the insight that optimal representations within MIM models generally reside in intermediate layers. Accordingly, MIM-Refiner leverages multiple contrastive heads that are connected to diverse intermediate layers. In each head, a modified nearest neighbor objective helps to construct respective semantic clusters.   The refinement process is short but effective. Within a few epochs, we refine the features of MIM models from subpar to state-of-the-art, off-the-shelf features. Refining a ViT-H, pre-trained with data2vec 2.0 on ImageNet-1K, achieves new state-of-the-art results in linear probing (84.7%) and low-shot classification among models that are pre-trained on ImageNet-1K. In ImageNet-1K 1-shot classification, MIM-Refiner sets a new state-of-the-art of 64.2%, outperforming larger mo
    
[^35]: 并行分割学习的工作流优化

    Workflow Optimization for Parallel Split Learning

    [https://arxiv.org/abs/2402.10092](https://arxiv.org/abs/2402.10092)

    本文提出了一种并行分割学习的工作流优化方法，旨在最小化训练时间，通过将问题分解成客户-辅助器分配和调度决策的联合问题进行求解。

    

    分割学习（SL）最近被提出作为一种让资源受限设备训练多参数神经网络（NNs）并参与联邦学习（FL）的方法。SL将NN模型分割成部分，并允许客户端（设备）将最大部分作为处理任务卸载给计算能力强大的辅助器。在并行SL中，多个辅助器可以处理一个或多个客户端的模型部分，从而大大减少了所有客户端的训练时间（makespan）。本文关注该操作的工作流编排，特别是在高度异构系统中的关键性问题。我们将客户-辅助器分配和调度决策的联合问题形式化，目标是最小化训练时间（makespan），并证明该问题是NP难问题。我们提出了一种基于问题分解的解决方法，利用其固有特性

    arXiv:2402.10092v1 Announce Type: cross  Abstract: Split learning (SL) has been recently proposed as a way to enable resource-constrained devices to train multi-parameter neural networks (NNs) and participate in federated learning (FL). In a nutshell, SL splits the NN model into parts, and allows clients (devices) to offload the largest part as a processing task to a computationally powerful helper. In parallel SL, multiple helpers can process model parts of one or more clients, thus, considerably reducing the maximum training time over all clients (makespan). In this paper, we focus on orchestrating the workflow of this operation, which is critical in highly heterogeneous systems, as our experiments show. In particular, we formulate the joint problem of client-helper assignments and scheduling decisions with the goal of minimizing the training makespan, and we prove that it is NP-hard. We propose a solution method based on the decomposition of the problem by leveraging its inherent sy
    
[^36]: 基于文本的产品匹配--半监督聚类方法

    Text-Based Product Matching -- Semi-Supervised Clustering Approach

    [https://arxiv.org/abs/2402.10091](https://arxiv.org/abs/2402.10091)

    本文介绍了一种利用半监督聚类方法进行产品匹配的新思路，并通过实验证明了无监督匹配与少量注释样本的产品链接可以成为主导的监督策略的替代方法。

    

    在电子商务的许多任务中，匹配多个产品提供中相同的产品是一个关键要素，如比较产品供应、动态价格优化和选择为客户个性化定制的产品组合。它对应于众所周知的实体匹配的机器学习任务，具有其自身的特殊性，如无处不在的非结构化数据或不准确和不一致的产品描述。本文旨在提出一种利用半监督聚类方法进行产品匹配的新思路。我们通过在真实数据集上使用主要是文本特征和模糊字符串匹配的IDEC算法进行实验，以及更多标准方法作为参考，来研究该方法的性能。鼓舞人心的结果显示，无监督匹配结合少量注释样本的产品链接可能是一种可能的替代方法，而不是主导的监督策略。

    arXiv:2402.10091v1 Announce Type: cross  Abstract: Matching identical products present in multiple product feeds constitutes a crucial element of many tasks of e-commerce, such as comparing product offerings, dynamic price optimization, and selecting the assortment personalized for the client. It corresponds to the well-known machine learning task of entity matching, with its own specificity, like omnipresent unstructured data or inaccurate and inconsistent product descriptions. This paper aims to present a new philosophy to product matching utilizing a semi-supervised clustering approach. We study the properties of this method by experimenting with the IDEC algorithm on the real-world dataset using predominantly textual features and fuzzy string matching, with more standard approaches as a point of reference. Encouraging results show that unsupervised matching, enriched with a small annotated sample of product links, could be a possible alternative to the dominant supervised strategy,
    
[^37]: PICS: 图像描述和搜索的流水线

    PICS: Pipeline for Image Captioning and Search

    [https://arxiv.org/abs/2402.10090](https://arxiv.org/abs/2402.10090)

    PICS是一种用于图像描述和搜索的流水线，它利用了大型语言模型的进展来自动化图像描述的过程，并通过集成情感分析来增强元数据，从而提高了大规模图像库的搜索效率和访问性。

    

    数字图像的增长使得高效分类和检索的先进系统成为必需，这在数据库管理和信息检索中提出了重大挑战。本文介绍了PICS（图像描述和搜索的流水线），这是一种旨在解决组织大规模图像库中固有复杂性的新方法。PICS利用了大型语言模型（LLM）的进展，自动化图像描述的过程，提供一个超越传统手动注释方法的解决方案。该方法基于这样的认识，即有意义的，人工智能生成的描述能够显著改善大型数据库中图像的可搜索性和可访问性。通过将情感分析整合到流水线中，PICS进一步丰富了元数据，实现了超越基本描述符的细致搜索。这种方法不仅简化了管理大规模数据库的任务，同时提高了图像的检索效率。

    arXiv:2402.10090v1 Announce Type: cross  Abstract: The growing volume of digital images necessitates advanced systems for efficient categorization and retrieval, presenting a significant challenge in database management and information retrieval. This paper introduces PICS (Pipeline for Image Captioning and Search), a novel approach designed to address the complexities inherent in organizing large-scale image repositories. PICS leverages the advancements in Large Language Models (LLMs) to automate the process of image captioning, offering a solution that transcends traditional manual annotation methods. The approach is rooted in the understanding that meaningful, AI-generated captions can significantly enhance the searchability and accessibility of images in large databases. By integrating sentiment analysis into the pipeline, PICS further enriches the metadata, enabling nuanced searches that extend beyond basic descriptors. This methodology not only simplifies the task of managing vas
    
[^38]: 分层混合建模用于灵活工具使用

    Hierarchical hybrid modeling for flexible tool use

    [https://arxiv.org/abs/2402.10088](https://arxiv.org/abs/2402.10088)

    本研究基于主动推理计算框架，提出了一个分层混合模型，通过组合离散和连续模型以实现灵活工具使用，控制和规划。在非平凡任务中验证了该模型的有效性和可扩展性。

    

    在最近提出的主动推理计算框架中，离散模型可以与连续模型相结合，以在不断变化的环境中进行决策。从另一个角度来看，简单的代理可以组合在一起，以更好地捕捉世界的因果关系。我们如何将这两个特点结合起来实现高效的目标导向行为？我们提出了一个架构，由多个混合 - 连续和离散 - 单元组成，复制代理的配置，由高级离散模型控制，实现动态规划和同步行为。每个层次内部的进一步分解可以以分层方式表示与self相关的其他代理和对象。我们在一个非平凡的任务上评估了这种分层混合模型：在拾取一个移动工具后到达一个移动物体。这项研究扩展了以推理为控制的先前工作，并提出了一种替代方案。

    arXiv:2402.10088v1 Announce Type: cross  Abstract: In a recent computational framework called active inference, discrete models can be linked to their continuous counterparts to perform decision-making in changing environments. From another perspective, simple agents can be combined to better capture the causal relationships of the world. How can we use these two features together to achieve efficient goal-directed behavior? We present an architecture composed of several hybrid -- continuous and discrete -- units replicating the agent's configuration, controlled by a high-level discrete model that achieves dynamic planning and synchronized behavior. Additional factorizations within each level allow to represent hierarchically other agents and objects in relation to the self. We evaluate this hierarchical hybrid model on a non-trivial task: reaching a moving object after having picked a moving tool. This study extends past work on control as inference and proposes an alternative directi
    
[^39]: 异构网络中使用强化学习进行分散隐秘路由

    Decentralized Covert Routing in Heterogeneous Networks Using Reinforcement Learning

    [https://arxiv.org/abs/2402.10087](https://arxiv.org/abs/2402.10087)

    本文提出了一个基于强化学习的新型隐秘路由算法，在异构网络中实现了分散的隐秘路由通信，仅使用本地反馈信息确定每个节点的下一跳和通信模式。在数值模拟中表明，该策略与最优集中式路由方案相比，性能损耗可忽略。

    

    本文研究了在异构网络中进行隐秘路由通信，其中源节点通过中继节点将机密数据传输到目标节点，每个传输节点都会明智地在多个通信模式中选择一种模式。我们开发了一种基于强化学习的新型隐秘路由算法，该算法仅根据从其相邻节点接收到的本地反馈信息，使每个节点确定其下一个跳和模式。通过数值模拟，我们展示了所提出的隐秘路由策略与最优集中式路由方案相比仅有可忽略的性能损耗。

    arXiv:2402.10087v1 Announce Type: cross  Abstract: This letter investigates covert routing communications in a heterogeneous network where a source transmits confidential data to a destination with the aid of relaying nodes where each transmitter judiciously chooses one modality among multiple communication modalities. We develop a novel reinforcement learning-based covert routing algorithm that finds a route from the source to the destination where each node identifies its next hop and modality only based on the local feedback information received from its neighboring nodes. We show based on numerical simulations that the proposed covert routing strategy has only negligible performance loss compared to the optimal centralized routing scheme.
    
[^40]: 可解释的人工智能在安全可信的自动驾驶中的应用：一项系统性评述

    Explainable AI for Safe and Trustworthy Autonomous Driving: A Systematic Review

    [https://arxiv.org/abs/2402.10086](https://arxiv.org/abs/2402.10086)

    可解释的AI技术对于解决自动驾驶中的安全问题和信任问题至关重要。本文通过系统文献综述的方式，分析了可解释的AI方法在满足自动驾驶要求方面的关键贡献，并提出了可解释的设计、可解释的替代模型、可解释的监控、辅助技术和解释的可视化等五个方面的应用。

    

    鉴于其在感知和规划任务中相对传统方法具有更优异的性能，人工智能（AI）对于自动驾驶（AD）的应用显示出了很大的潜力。然而，难以理解的AI系统加剧了对AD安全保证的现有挑战。缓解这一挑战的一种方法是利用可解释的AI（XAI）技术。为此，我们首次提出了关于可解释方法在安全可信的AD中的全面系统文献综述。我们首先分析了在AD背景下AI的要求，重点关注数据、模型和机构这三个关键方面。我们发现XAI对于满足这些要求是至关重要的。基于此，我们解释了AI中解释的来源，并描述了一种XAI的分类学。然后，我们确定了XAI在安全可信的AD中的五个主要贡献，包括可解释的设计、可解释的替代模型、可解释的监控，辅助...

    arXiv:2402.10086v1 Announce Type: cross  Abstract: Artificial Intelligence (AI) shows promising applications for the perception and planning tasks in autonomous driving (AD) due to its superior performance compared to conventional methods. However, inscrutable AI systems exacerbate the existing challenge of safety assurance of AD. One way to mitigate this challenge is to utilize explainable AI (XAI) techniques. To this end, we present the first comprehensive systematic literature review of explainable methods for safe and trustworthy AD. We begin by analyzing the requirements for AI in the context of AD, focusing on three key aspects: data, model, and agency. We find that XAI is fundamental to meeting these requirements. Based on this, we explain the sources of explanations in AI and describe a taxonomy of XAI. We then identify five key contributions of XAI for safe and trustworthy AI in AD, which are interpretable design, interpretable surrogate models, interpretable monitoring, auxil
    
[^41]: 开发端到端异常检测系统

    Develop End-to-End Anomaly Detection System

    [https://arxiv.org/abs/2402.10085](https://arxiv.org/abs/2402.10085)

    本文提出了一个端到端的异常检测模型开发流程，可以解决网络中异常检测的挑战，并通过引入新的预测模型"Lachesis"来展示其有效性。

    

    异常检测在确保网络鲁棒性方面起着至关重要的作用。然而，在考虑了恶意和非恶意事件都可能引起异常情况的情况下，实施智能报警系统变得具有挑战性，难以确定异常模式。计算机网络领域中缺乏带标签的数据进一步加剧了这个问题，在处理现实场景时阻碍了开发具有鲁棒性模型的发展。为了解决这个问题，本文提出了一个端到端的异常检测模型开发流程。该框架可以接受用户反馈，并实现持续的用户中心模型性能评估和优化。我们通过引入并基准测试一个名为"Lachesis"的新预测模型来展示该框架的有效性，在一个真实的网络问题上进行了实验。实验结果证明了这个框架的鲁棒性。

    arXiv:2402.10085v1 Announce Type: cross  Abstract: Anomaly detection plays a crucial role in ensuring network robustness. However, implementing intelligent alerting systems becomes a challenge when considering scenarios in which anomalies can be caused by both malicious and non-malicious events, leading to the difficulty of determining anomaly patterns. The lack of labeled data in the computer networking domain further exacerbates this issue, impeding the development of robust models capable of handling real-world scenarios. To address this challenge, in this paper, we propose an end-to-end anomaly detection model development pipeline. This framework makes it possible to consume user feedback and enable continuous user-centric model performance evaluation and optimization. We demonstrate the efficacy of the framework by way of introducing and bench-marking a new forecasting model -- named \emph{Lachesis} -- on a real-world networking problem. Experiments have demonstrated the robustnes
    
[^42]: FedRDF: 一种针对联邦学习中毒化攻击的强大和动态聚合函数

    FedRDF: A Robust and Dynamic Aggregation Function against Poisoning Attacks in Federated Learning

    [https://arxiv.org/abs/2402.10082](https://arxiv.org/abs/2402.10082)

    FedRDF提出了一种新颖的鲁棒聚合机制，利用傅里叶变换来有效处理联邦学习中的复杂攻击，而不需要先验知识。

    

    联邦学习（FL）是一种有前景的方法，可以解决集中式机器学习（ML）部署所带来的典型隐私问题。尽管FL具有众所周知的优点，但它容易受到安全攻击，如拜占庭行为和毒化攻击，这些攻击会严重影响模型性能和收敛。现有方法对于缓解复杂攻击（如中值、修剪均值或Krum聚合函数）的效果仅部分证明了在特定攻击情况下的有效性。我们的研究引入了一种新颖的鲁棒聚合机制，利用傅里叶变换（FT）来有效处理复杂攻击，而不需要对攻击者数量有先验知识。利用这种数据技术，FL客户端生成的权重被投影到频域以确定其密度函数，选择具有最高频率的密度函数。

    arXiv:2402.10082v1 Announce Type: new  Abstract: Federated Learning (FL) represents a promising approach to typical privacy concerns associated with centralized Machine Learning (ML) deployments. Despite its well-known advantages, FL is vulnerable to security attacks such as Byzantine behaviors and poisoning attacks, which can significantly degrade model performance and hinder convergence. The effectiveness of existing approaches to mitigate complex attacks, such as median, trimmed mean, or Krum aggregation functions, has been only partially demonstrated in the case of specific attacks. Our study introduces a novel robust aggregation mechanism utilizing the Fourier Transform (FT), which is able to effectively handling sophisticated attacks without prior knowledge of the number of attackers. Employing this data technique, weights generated by FL clients are projected into the frequency domain to ascertain their density function, selecting the one exhibiting the highest frequency. Conseq
    
[^43]: 自主驾驶系统中基于学习的相机和激光雷达仿真方法的综述

    Review of the Learning-based Camera and Lidar Simulation Methods for Autonomous Driving Systems

    [https://arxiv.org/abs/2402.10079](https://arxiv.org/abs/2402.10079)

    本文综述了自主驾驶系统中基于学习的相机和激光雷达仿真方法的最新研究现状。

    

    感知传感器，尤其是相机和激光雷达，是自主驾驶系统(Autonomous Driving Systems，ADS)的关键元素，使其能够理解周围环境以做出明智的驾驶和控制决策。因此，开发逼真的相机和激光雷达模拟方法，也称为相机和激光雷达模型，对于有效进行基于仿真的ADS测试至关重要。此外，基于深度学习的感知模型的兴起，促进了感知传感器模型作为合成各种训练数据集的有价值工具的普及。传统传感器仿真方法依赖于计算密集型的基于物理的算法，特别是在复杂系统如ADS中。因此，目前的潜力在于基于学习的模型，受到深度生成模型在合成高维数据方面取得成功的推动。本文综述了基于学习的传感器仿真方法的最新研究现状。

    arXiv:2402.10079v1 Announce Type: cross  Abstract: Perception sensors, particularly camera and Lidar, are key elements of Autonomous Driving Systems (ADS) that enable them to comprehend their surroundings for informed driving and control decisions. Therefore, developing realistic camera and Lidar simulation methods, also known as camera and Lidar models, is of paramount importance to effectively conduct simulation-based testing for ADS. Moreover, the rise of deep learning-based perception models has propelled the prevalence of perception sensor models as valuable tools for synthesising diverse training datasets. The traditional sensor simulation methods rely on computationally expensive physics-based algorithms, specifically in complex systems such as ADS. Hence, the current potential resides in learning-based models, driven by the success of deep generative models in synthesising high-dimensional data. This paper reviews the current state-of-the-art in learning-based sensor simulation
    
[^44]: EventF2S：使用神经形态友好算法的异步和稀疏尖峰AER框架

    EventF2S: Asynchronous and Sparse Spiking AER Framework using Neuromorphic-Friendly Algorithm

    [https://arxiv.org/abs/2402.10078](https://arxiv.org/abs/2402.10078)

    这项研究提出了一种基于神经形态的AER-SNN对象识别解决方案，集成了异步处理、神经形态兼容性和稀疏尖峰的特性，在资源受限的应用中具有重要意义。

    

    生物启发式的地址事件表示（AER）传感器因其低功耗、高稀疏性和高时间分辨率而受到广泛关注。尖峰神经网络（SNN）已成为AER数据处理的固有选择。然而，AER-SNN范式的集成尚未充分探索异步处理、神经形态兼容性和稀疏尖峰，这是资源受限应用的关键需求。为了填补这一差距，我们引入了一种受大脑启发的AER-SNN对象识别解决方案，其中包括一个与第一尖峰识别网络集成的数据编码器。受到视觉皮层中神经元功能的启发，我们设计的解决方案是异步的并且与神经形态硬件兼容。此外，我们采用了去噪和第一尖峰编码的原理，以实现最佳的尖峰信号传递，显著减少了能耗。

    arXiv:2402.10078v1 Announce Type: cross  Abstract: Bio-inspired Address Event Representation (AER) sensors have attracted significant popularity owing to their low power consumption, high sparsity, and high temporal resolution. Spiking Neural Network (SNN) has become the inherent choice for AER data processing. However, the integration of the AER-SNN paradigm has not adequately explored asynchronous processing, neuromorphic compatibility, and sparse spiking, which are the key requirements of resource-constrained applications. To address this gap, we introduce a brain-inspired AER-SNN object recognition solution, which includes a data encoder integrated with a First-To-Spike recognition network. Being fascinated by the functionality of neurons in the visual cortex, we designed the solution to be asynchronous and compatible with neuromorphic hardware. Furthermore, we have adapted the principle of denoising and First-To-Spike coding to achieve optimal spike signaling, significantly reduci
    
[^45]: 在共享城市区域中与机器人互动的人体姿势大规模融合和标记数据集的研究

    Towards a large-scale fused and labeled dataset of human pose while interacting with robots in shared urban areas

    [https://arxiv.org/abs/2402.10077](https://arxiv.org/abs/2402.10077)

    本研究通过融合和标记两个数据集，MOT17和NCLT，填补了共享城市区域中人机交互的人体姿势数据集的空白。

    

    在过去的十年中，自主配送机器人（ADR）在回应不断增长的电子商务需求的同时，也改变了传统的配送方法。然而，ADR在共享城市区域中安全与行人共同交往的准备问题仍然存在一个悬而未决的问题。我们认为，在这样的环境中，理解ADR与行人的相互作用存在重要的研究空白。人体姿势估计是各种下游应用（包括姿势预测和具有社会意识的机器人路径规划）的重要基石。然而，缺乏一个捕获共享城市区域中人机交互的丰富且标记的数据集，限制了这个目标的实现。本文通过重新利用两个数据集MOT17和NCLT，分别聚焦于行人跟踪和同时定位与地图构建（SLAM），填补了这一空白。结果得到的独特数据集包含数千个真实室内和户外场景的数据。

    arXiv:2402.10077v1 Announce Type: cross  Abstract: Over the last decade, Autonomous Delivery Robots (ADRs) have transformed conventional delivery methods, responding to the growing e-commerce demand. However, the readiness of ADRs to navigate safely among pedestrians in shared urban areas remains an open question. We contend that there are crucial research gaps in understanding their interactions with pedestrians in such environments. Human Pose Estimation is a vital stepping stone for various downstream applications, including pose prediction and socially aware robot path-planning. Yet, the absence of an enriched and pose-labeled dataset capturing human-robot interactions in shared urban areas hinders this objective. In this paper, we bridge this gap by repurposing, fusing, and labeling two datasets, MOT17 and NCLT, focused on pedestrian tracking and Simultaneous Localization and Mapping (SLAM), respectively. The resulting unique dataset represents thousands of real-world indoor and o
    
[^46]: QUICK：针对高效LLM推理的量化感知交错和无冲突内核

    QUICK: Quantization-aware Interleaving and Conflict-free Kernel for efficient LLM inference

    [https://arxiv.org/abs/2402.10076](https://arxiv.org/abs/2402.10076)

    QUICK是一组针对量化大语言模型（LLMs）的高效推理的优化CUDA内核。通过解决共享内存冲突问题和交错量化权重矩阵，QUICK实现了显著的速度提升和吞吐量增益。

    

    我们介绍了QUICK，一组用于高效推理量化大语言模型（LLMs）的优化CUDA内核。QUICK解决了现有混合精度矩阵乘法内核的共享内存冲突问题。我们的方法在离线情况下交错LLMs的量化权重矩阵，从而跳过解量化后的共享内存写回。我们在较大批次上展示了与AutoAWQ现有内核相比多达1.91倍的加速效果，并在各种NVIDIA GPU设备上的代表性LLM模型上获得了多达1.94倍的吞吐量增益。

    arXiv:2402.10076v1 Announce Type: cross  Abstract: We introduce QUICK, a group of novel optimized CUDA kernels for the efficient inference of quantized Large Language Models (LLMs). QUICK addresses the shared memory bank-conflict problem of state-of-the-art mixed precision matrix multiplication kernels. Our method interleaves the quantized weight matrices of LLMs offline to skip the shared memory write-back after the dequantization. We demonstrate up to 1.91x speedup over existing kernels of AutoAWQ on larger batches and up to 1.94x throughput gain on representative LLM models on various NVIDIA GPU devices.
    
[^47]: GraphCBAL: 通过强化学习对图神经网络进行类平衡主动学习

    GraphCBAL: Class-Balanced Active Learning for Graph Neural Networks via Reinforcement Learning

    [https://arxiv.org/abs/2402.10074](https://arxiv.org/abs/2402.10074)

    本文提出了一种通过强化学习对图神经网络进行类平衡主动学习的框架GraphCBAL，该框架能够学习一种最佳策略，选择类平衡和信息丰富的节点进行注释，以最大化GNNs性能。

    

    最近，图神经网络（GNNs）已经取得了显著的成功。GNNs的主动学习旨在从未标记的数据中查询有价值的样本进行注释，以最大限度地降低成本并提高GNNs的性能。然而，对于GNNs中的强化主动学习，现有的大多数方法可能导致高度不平衡的类分布，尤其是在高度倾斜的类别场景下。这进一步对分类性能产生负面影响。为了解决这个问题，本文提出了一种新颖的增强类平衡主动学习框架GraphCBAL，用于GNNs。它学习一种最佳策略，以获取类平衡和信息丰富的节点进行注释，从而最大化选择的标记节点训练的GNNs的性能。GraphCBAL设计了类平衡感知状态和奖励函数，实现模型性能和类平衡之间的折衷。我们进一步改进了GraphCBAL，得到GraphCBAL++。

    arXiv:2402.10074v1 Announce Type: new  Abstract: Graph neural networks (GNNs) have recently demonstrated significant success. Active learning for GNNs aims to query the valuable samples from the unlabeled data for annotation to maximize the GNNs' performance at a low cost. However, most existing methods for reinforced active learning in GNNs may lead to a highly imbalanced class distribution, especially in highly skewed class scenarios. This further adversely affects the classification performance. To tackle this issue, in this paper, we propose a novel reinforced class-balanced active learning framework for GNNs, namely, GraphCBAL. It learns an optimal policy to acquire class-balanced and informative nodes for annotation, maximizing the performance of GNNs trained with selected labeled nodes. GraphCBAL designs class-balance-aware states, as well as a reward function that achieves trade-off between model performance and class balance. We further upgrade GraphCBAL to GraphCBAL++ by intr
    
[^48]: 深度联合源信道编码以实现高效可靠的异构技术通信

    Deep Joint Source-Channel Coding for Efficient and Reliable Cross-Technology Communication

    [https://arxiv.org/abs/2402.10072](https://arxiv.org/abs/2402.10072)

    本文提出了一个深度联合源信道编码(DJSCC)方案，用于实现高效可靠的跨技术通信(CTC)。该方案利用神经网络构建编码器和解码器，同时实现信息压缩和语义含义稳健性。

    

    跨技术通信(CTC)是一种有效实现不兼容无线技术之间直接通信而不需要硬件修改的有望技术。然而，由于其低效性和不可靠性，在实际应用中尚未得到广泛采用。为了解决这个问题，本文提出了一种深度联合源信道编码(DJSCC)方案，以实现高效可靠的CTC。该方案在发送端和接收端分别建立基于神经网络的编码器和解码器，同时实现两个关键任务：1)将信息压缩到仅保留其基本语义含义的程度; 2)确保在跨不兼容技术传输语义含义时的稳健性。该方案将现有的CTC编码算法作为领域知识，引导编码器-解码器对学习CTC的特性。

    arXiv:2402.10072v1 Announce Type: cross  Abstract: Cross-technology communication (CTC) is a promising technique that enables direct communications among incompatible wireless technologies without needing hardware modification. However, it has not been widely adopted in real-world applications due to its inefficiency and unreliability. To address this issue, this paper proposes a deep joint source-channel coding (DJSCC) scheme to enable efficient and reliable CTC. The proposed scheme builds a neural-network-based encoder and decoder at the sender side and the receiver side, respectively, to achieve two critical tasks simultaneously: 1) compressing the messages to the point where only their essential semantic meanings are preserved; 2) ensuring the robustness of the semantic meanings when they are transmitted across incompatible technologies. The scheme incorporates existing CTC coding algorithms as domain knowledge to guide the encoder-decoder pair to learn the characteristics of CTC l
    
[^49]: 学习快速变化的慢性在脉冲神经网络中

    Learning fast changing slow in spiking neural networks

    [https://arxiv.org/abs/2402.10069](https://arxiv.org/abs/2402.10069)

    通过近端策略优化实现的生物学可行方法在脉冲神经网络中减轻了强化学习所面临的数据稀缺性和噪声引入的困难。

    

    强化学习在现实问题中面临着很大的挑战，主要源于与环境的有限交互导致的可用数据的稀缺性。 RL通常需要大量的数据来进行有效的学习，这使得复杂性进一步增加，尤其是在使用循环脉冲网络实现RL时，脉冲引入的固有噪声增加了难度。终身学习机器在本质上必须解决可塑性-稳定性悖论。在获得新知识和保持稳定之间取得平衡对于人工智能代理至关重要。在这个背景下，我们从机器学习技术中汲取灵感，并引入了一种生物可行的近端策略优化实现，认为它显著减轻了此挑战。我们的方法带来了两个重要的进展：首先，能够...

    arXiv:2402.10069v1 Announce Type: cross  Abstract: Reinforcement learning (RL) faces substantial challenges when applied to real-life problems, primarily stemming from the scarcity of available data due to limited interactions with the environment. This limitation is exacerbated by the fact that RL often demands a considerable volume of data for effective learning. The complexity escalates further when implementing RL in recurrent spiking networks, where inherent noise introduced by spikes adds a layer of difficulty. Life-long learning machines must inherently resolve the plasticity-stability paradox. Striking a balance between acquiring new knowledge and maintaining stability is crucial for artificial agents. In this context, we take inspiration from machine learning technology and introduce a biologically plausible implementation of proximal policy optimization, arguing that it significantly alleviates this challenge. Our approach yields two notable advancements: first, the ability t
    
[^50]: 基于LLM的应用意图管理的策略生成

    LLM-based policy generation for intent-based management of applications

    [https://arxiv.org/abs/2402.10067](https://arxiv.org/abs/2402.10067)

    这项研究提出了基于LLM的策略生成方法，用于实现自动化的应用意图管理。通过生成逐步分解意图所需的动作，并将其映射到API，实现了闭控制循环来自动化策略执行。

    

    自动化管理需要将高级用户请求，例如意图，分解成系统可以理解和执行的抽象。这是具有挑战性的，因为即使是一个简单的意图也需要执行一系列有序的步骤。而识别和适应这些步骤（随着条件的变化）的任务需要一种无法事先完全定义的分解方法。为了解决这些挑战并支持自动化的意图分解和执行，我们探索了大型语言模型（LLM）的少样本能力。我们提出了一个管道，通过生成所需的动作，使用基于策略的抽象逐步分解意图。这使我们能够通过创建用于意图部署的闭控制循环来自动化策略执行。为此，我们生成并将策略映射到API，并形成执行所需的监控、分析、计划和执行的应用管理循环。

    arXiv:2402.10067v1 Announce Type: cross  Abstract: Automated management requires decomposing high-level user requests, such as intents, to an abstraction that the system can understand and execute. This is challenging because even a simple intent requires performing a number of ordered steps. And the task of identifying and adapting these steps (as conditions change) requires a decomposition approach that cannot be exactly pre-defined beforehand. To tackle these challenges and support automated intent decomposition and execution, we explore the few-shot capability of Large Language Models (LLMs). We propose a pipeline that progressively decomposes intents by generating the required actions using a policy-based abstraction. This allows us to automate the policy execution by creating a closed control loop for the intent deployment. To do so, we generate and map the policies to APIs and form application management loops that perform the necessary monitoring, analysis, planning and executi
    
[^51]: NYCTALE: 用于自适应个性化肺结节侵袭性预测的神经证据转换器

    NYCTALE: Neuro-Evidence Transformer for Adaptive and Personalized Lung Nodule Invasiveness Prediction

    [https://arxiv.org/abs/2402.10066](https://arxiv.org/abs/2402.10066)

    NYCTALE是一种神经启发的Transformer架构，用于自适应个性化肺结节侵袭性预测。与传统的CT基深度学习模型不同，NYCTALE仅在累积足够数量的证据时才进行预测。

    

    从灵长类动物大脑引发的证据累积过程中获得灵感，借鉴了认知心理学和神经科学的模型，本文介绍了NYCTALE框架，一种神经启发的基于证据累积的Transformer架构。提出的神经启发的NYCTALE在个性化医学领域，特别是肺癌诊断方面提供了一种创新的途径。作为自然界中的一种小型猫头鹰，Nyctales以其夜间行为而闻名，主要在夜晚进行捕猎。NYCTALE以类似警惕的方式运作，即以基于证据的方式处理数据并动态/自适应地进行预测。与传统的基于计算机断层扫描（CT）的深度学习（DL）模型不同，NYCTALE仅在累积足够数量的证据时才进行预测。换句话说，对于每个人，只处理所有或预定义的CT切片的子集，而不是全部切片。

    arXiv:2402.10066v1 Announce Type: cross  Abstract: Drawing inspiration from the primate brain's intriguing evidence accumulation process, and guided by models from cognitive psychology and neuroscience, the paper introduces the NYCTALE framework, a neuro-inspired and evidence accumulation-based Transformer architecture. The proposed neuro-inspired NYCTALE offers a novel pathway in the domain of Personalized Medicine (PM) for lung cancer diagnosis. In nature, Nyctales are small owls known for their nocturnal behavior, hunting primarily during the darkness of night. The NYCTALE operates in a similarly vigilant manner, i.e., processing data in an evidence-based fashion and making predictions dynamically/adaptively. Distinct from conventional Computed Tomography (CT)-based Deep Learning (DL) models, the NYCTALE performs predictions only when sufficient amount of evidence is accumulated. In other words, instead of processing all or a pre-defined subset of CT slices, for each person, slices 
    
[^52]: 每个数据点泄露您隐私的程度有多大？量化每个数据点的成员泄露

    How Much Does Each Datapoint Leak Your Privacy? Quantifying the Per-datum Membership Leakage

    [https://arxiv.org/abs/2402.10065](https://arxiv.org/abs/2402.10065)

    本论文研究了每个数据点的成员推断攻击，量化了每个数据点的成员泄露，并评估了两种隐私防御措施的效果。

    

    我们研究了每个数据点的成员推断攻击（MIAs），其中攻击者旨在推断出一个固定目标数据是否已包含在算法的输入数据集中，从而侵犯隐私。首先，我们定义每个数据点的成员泄露为最优对手辨识它的优势。然后，我们量化了经验均值的每个数据点的成员泄露，并表明它取决于目标数据点和数据生成分布之间的马氏距离。我们进一步评估了两种隐私防御措施的效果，即添加高斯噪声和子采样。我们准确地量化了它们都如何降低每个数据点的成员泄露。我们的分析建立在一个结合了似然比检验的Edgeworth展开和Lindeberg-Feller中心极限定理的新型证明技术上。我们的分析连接了现有的似然比和标量乘积攻击，并对这些攻击进行了论证。

    arXiv:2402.10065v1 Announce Type: new  Abstract: We study the per-datum Membership Inference Attacks (MIAs), where an attacker aims to infer whether a fixed target datum has been included in the input dataset of an algorithm and thus, violates privacy. First, we define the membership leakage of a datum as the advantage of the optimal adversary targeting to identify it. Then, we quantify the per-datum membership leakage for the empirical mean, and show that it depends on the Mahalanobis distance between the target datum and the data-generating distribution. We further assess the effect of two privacy defences, i.e. adding Gaussian noise and sub-sampling. We quantify exactly how both of them decrease the per-datum membership leakage. Our analysis builds on a novel proof technique that combines an Edgeworth expansion of the likelihood ratio test and a Lindeberg-Feller central limit theorem. Our analysis connects the existing likelihood ratio and scalar product attacks, and also justifies 
    
[^53]: 导航玉米：分子模拟的循环和条件计算图

    Navigating the Maize: Cyclic and conditional computational graphs for molecular simulation

    [https://arxiv.org/abs/2402.10064](https://arxiv.org/abs/2402.10064)

    该论文介绍了一种用于分子模拟的循环和条件计算图的工作流管理器，通过并行化和通信实现任意图结构的执行，具有很高的实用性和效果。

    

    许多计算化学和分子模拟工作流程可以表示为计算图。这种抽象有助于模块化和潜在地重用现有组件，并提供并行化和易于复制。现有工具将计算表示为有向无环图(DAG)，从而通过并行化并发分支来实现高效执行。然而，这些系统通常无法表示循环和条件工作流程。因此，我们开发了Maize，一种基于流程编程原理的、用于循环和条件图的工作流管理器。通过在单独的进程中同时运行图中的每个节点，并在任何时间通过专用的节点间通道进行通信，可以执行任意的图结构。我们通过在计算药物设计中进行动态主动学习任务来展示工具的有效性，其中涉及使用小分子 gen

    arXiv:2402.10064v1 Announce Type: cross  Abstract: Many computational chemistry and molecular simulation workflows can be expressed as graphs. This abstraction is useful to modularize and potentially reuse existing components, as well as provide parallelization and ease reproducibility. Existing tools represent the computation as a directed acyclic graph (DAG), thus allowing efficient execution by parallelization of concurrent branches. These systems can, however, generally not express cyclic and conditional workflows. We therefore developed Maize, a workflow manager for cyclic and conditional graphs based on the principles of flow-based programming. By running each node of the graph concurrently in separate processes and allowing communication at any time through dedicated inter-node channels, arbitrary graph structures can be executed. We demonstrate the effectiveness of the tool on a dynamic active learning task in computational drug design, involving the use of a small molecule gen
    
[^54]: 平衡类增量学习中的因果效应

    Balancing the Causal Effects in Class-Incremental Learning

    [https://arxiv.org/abs/2402.10063](https://arxiv.org/abs/2402.10063)

    论文揭示了类增量学习中新旧数据之间的不平衡因果效应，并提出了一种平衡因果效应的方法来缓解这个问题。

    

    类增量学习（CIL）是实现通用人工智能的一个实际且具有挑战性的问题。最近，预训练模型（PTMs）在视觉和自然语言处理任务中取得了突破。尽管最近的研究显示了PTMs顺序学习的潜力，但大量工作表明了缓解PTMs灾难性遗忘的必要性。通过一个初步研究和CIL的因果分析，我们揭示了问题的关键在于新旧数据之间的不平衡因果效应。具体而言，新数据促使模型适应新类别，同时阻碍了对旧类别的适应。同样，旧数据促使模型适应旧类别，同时阻碍了对新类别的适应。换句话说，从因果的角度来看，新旧类别之间的适应过程存在冲突。为了缓解这个问题，我们提出了“平衡因果效应”（BCE）的方法。

    arXiv:2402.10063v1 Announce Type: new  Abstract: Class-Incremental Learning (CIL) is a practical and challenging problem for achieving general artificial intelligence. Recently, Pre-Trained Models (PTMs) have led to breakthroughs in both visual and natural language processing tasks. Despite recent studies showing PTMs' potential ability to learn sequentially, a plethora of work indicates the necessity of alleviating the catastrophic forgetting of PTMs. Through a pilot study and a causal analysis of CIL, we reveal that the crux lies in the imbalanced causal effects between new and old data. Specifically, the new data encourage models to adapt to new classes while hindering the adaptation of old classes. Similarly, the old data encourages models to adapt to old classes while hindering the adaptation of new classes. In other words, the adaptation process between new and old classes conflicts from the causal perspective. To alleviate this problem, we propose Balancing the Causal Effects (B
    
[^55]: 用于识别未知分布的最优参数和神经元剪枝方法

    Optimal Parameter and Neuron Pruning for Out-of-Distribution Detection

    [https://arxiv.org/abs/2402.10062](https://arxiv.org/abs/2402.10062)

    提出了一种用于识别未知分布的最优参数和神经元剪枝方法（OPNP），通过评估模型参数和神经元的敏感性来解决OOD检测的问题。

    

    对于在现实场景中部署的机器学习模型，识别未知分布（OOD）样本的能力是不可或缺且具有挑战性的。大多数已有的OOD检测方法关注于探索高级训练技巧或训练无关的技巧，以防止模型对未知样本产生过于自信的置信度分数。基于训练的方法需要昂贵的训练成本，并且依赖于并非始终可用的OOD样本，而大多数基于训练无关的方法无法有效利用训练数据的先验信息。在这项工作中，我们提出了一种名为OPNP（Optimal Parameter and Neuron Pruning）的方法，旨在识别并删除导致过度拟合的参数和神经元。主要方法分为两个步骤。在第一步中，我们通过对所有训练样本进行梯度平均来评估模型参数和神经元的敏感性。

    arXiv:2402.10062v1 Announce Type: new  Abstract: For a machine learning model deployed in real world scenarios, the ability of detecting out-of-distribution (OOD) samples is indispensable and challenging. Most existing OOD detection methods focused on exploring advanced training skills or training-free tricks to prevent the model from yielding overconfident confidence score for unknown samples. The training-based methods require expensive training cost and rely on OOD samples which are not always available, while most training-free methods can not efficiently utilize the prior information from the training data. In this work, we propose an \textbf{O}ptimal \textbf{P}arameter and \textbf{N}euron \textbf{P}runing (\textbf{OPNP}) approach, which aims to identify and remove those parameters and neurons that lead to over-fitting. The main method is divided into two steps. In the first step, we evaluate the sensitivity of the model parameters and neurons by averaging gradients over all train
    
[^56]: ECE有多大的缺陷？通过对数平滑的分析

    How Flawed is ECE? An Analysis via Logit Smoothing

    [https://arxiv.org/abs/2402.10046](https://arxiv.org/abs/2402.10046)

    本研究通过分析对数平滑，探讨了ECE的缺陷以及对现有结果的影响，并提出了一种新的连续、易于估计的误差测度LS-ECE。通过实验发现，LS-ECE与分箱ECE非常接近。

    

    简而言之，如果一个模型的预测准确率与置信度匹配，那么这个模型就是校准的。在现有文献中，衡量校准性最常见的方法是期望校准误差（ECE）。然而，最近的研究指出了ECE的缺点，例如它在预测者空间中是不连续的。本研究探讨了这些问题有多本质，并分析了它们对现有结果的影响。为此，我们完全描述了ECE对波兰空间上的一般概率测度的不连续性。然后，我们利用这些不连续性提出了一种新的连续、易于估计的误差测度，称为Logit-Smoothed ECE（LS-ECE）。通过比较预训练图像分类模型的ECE和LS-ECE，我们在初步实验中发现，分箱ECE与LS-ECE非常接近，表明理论方面是相符的。

    arXiv:2402.10046v1 Announce Type: new  Abstract: Informally, a model is calibrated if its predictions are correct with a probability that matches the confidence of the prediction. By far the most common method in the literature for measuring calibration is the expected calibration error (ECE). Recent work, however, has pointed out drawbacks of ECE, such as the fact that it is discontinuous in the space of predictors. In this work, we ask: how fundamental are these issues, and what are their impacts on existing results? Towards this end, we completely characterize the discontinuities of ECE with respect to general probability measures on Polish spaces. We then use the nature of these discontinuities to motivate a novel continuous, easily estimated miscalibration metric, which we term Logit-Smoothed ECE (LS-ECE). By comparing the ECE and LS-ECE of pre-trained image classification models, we show in initial experiments that binned ECE closely tracks LS-ECE, indicating that the theoretical
    
[^57]: 短视频和心理健康：基于知识导向的多模态神经主题模型

    Short-Form Videos and Mental Health: A Knowledge-Guided Multimodal Neural Topic Model

    [https://arxiv.org/abs/2402.10045](https://arxiv.org/abs/2402.10045)

    这项研究针对短视频对观众心理健康的抑郁影响问题，开发了一种基于医学知识的多模态神经主题模型，以预测其影响并采取相应的干预措施。

    

    短视频正试图重新塑造整个社交媒体景观，然而专家们对其对观众的抑郁影响感到极度担忧，这一点已由医学研究证明。为了防止广泛影响，各平台渴望预测这些视频对观众心理健康的影响，从而采取干预措施，比如修订推荐算法和显示观众慎重选择。然而，现有的预测方法缺乏与抑郁症的临床证实的外部环境因素相关的医学知识。为了考虑这样的医学知识，我们采用了一种新兴的方法论学科——种子神经主题模型（NTMs）。然而，现有的种子NTMs存在单一来源主题、未知主题来源、模糊的种子监督和次优的收敛等局限性。为了解决这些挑战，我们开发了一种新颖的基于知识指导的多模态神经主题模型（Knowledg...（待补充）

    arXiv:2402.10045v1 Announce Type: cross  Abstract: While short-form videos head to reshape the entire social media landscape, experts are exceedingly worried about their depressive impacts on viewers, as evidenced by medical studies. To prevent widespread consequences, platforms are eager to predict these videos' impact on viewers' mental health. Subsequently, they can take intervention measures, such as revising recommendation algorithms and displaying viewer discretion. Nevertheless, applicable predictive methods lack relevance to well-established medical knowledge, which outlines clinically proven external and environmental factors of depression. To account for such medical knowledge, we resort to an emergent methodological discipline, seeded Neural Topic Models (NTMs). However, existing seeded NTMs suffer from the limitations of single-origin topics, unknown topic sources, unclear seed supervision, and suboptimal convergence. To address those challenges, we develop a novel Knowledg
    
[^58]: 如何验证机器学习回归任务的平均校准性？

    How to validate average calibration for machine learning regression tasks ?

    [https://arxiv.org/abs/2402.10043](https://arxiv.org/abs/2402.10043)

    本文提出了两种验证机器学习回归任务平均校准性的方法，将校准误差与平均绝对误差之间的差值和将平均平方z-分数与1进行比较。研究发现，前者对不确定性分布敏感，而后者在该方面提供了最可靠的方法。

    

    机器学习回归任务的平均校准性可以通过两种方式进行测试。一种方式是将校准误差（CE）估计为平均绝对误差（MSE）与平均方差（MV）或平均平方不确定性之间的差值。另一种方式是将平均平方z-分数或缩放误差（ZMS）与1进行比较。两种方法可能得出不同的结论，正如来自最近的机器学习不确定性量化文献中的数据集集合所示。研究表明，CE对不确定性分布非常敏感，特别是对于离群不确定性的存在，因此无法可靠地用于校准测试。相比之下，ZMS统计量不具有这种敏感性问题，在这种情况下提供了最可靠的方法。文章还讨论了对条件校准验证的影响。

    arXiv:2402.10043v1 Announce Type: cross  Abstract: Average calibration of the uncertainties of machine learning regression tasks can be tested in two ways. One way is to estimate the calibration error (CE) as the difference between the mean absolute error (MSE) and the mean variance (MV) or mean squared uncertainty. The alternative is to compare the mean squared z-scores or scaled errors (ZMS) to 1. Both approaches might lead to different conclusion, as illustrated on an ensemble of datasets from the recent machine learning uncertainty quantification literature. It is shown here that the CE is very sensitive to the distribution of uncertainties, and notably to the presence of outlying uncertainties, and that it cannot be used reliably for calibration testing. By contrast, the ZMS statistic does not present this sensitivity issue and offers the most reliable approach in this context. Implications for the validation of conditional calibration are discussed.
    
[^59]: RS-DPO：一种用于对齐大型语言模型的混合拒绝采样和直接优化偏好的方法

    RS-DPO: A Hybrid Rejection Sampling and Direct Preference Optimization Method for Alignment of Large Language Models

    [https://arxiv.org/abs/2402.10038](https://arxiv.org/abs/2402.10038)

    本研究提出了一种名为RS-DPO的方法，它将拒绝采样和直接优化偏好结合起来，用于对齐大型语言模型。通过开发一个经过监督微调的策略模型，并从该模型中直接采样响应，RS-DPO能够有效解决基于近端策略优化的不稳定性和高计算成本的问题。通过识别对比样本对，RS-DPO能够更好地进行RLHF。

    

    强化学习从人类反馈中学习（RLHF）已被广泛应用于将大型语言模型与用户意图对齐。然而，基于近端策略优化（PPO）的RLHF有时不稳定，需要显著的超参数微调，并且在对齐过程中计算成本高昂。最近，提出了直接优化偏好（DPO）来解决这些挑战。然而，DPO依赖于从人类标注者和替代LLM生成的对比回复，而不是策略模型，限制了RLHF的效果。本文通过系统地结合拒绝采样（RS）和DPO来解决这两个挑战。我们提出的方法RS-DPO，首先开发出一个经过监督微调的策略模型（SFT）。然后直接从SFT模型中采样每个提示的k个响应。RS-DPO基于其相似度识别对比样本对。

    arXiv:2402.10038v1 Announce Type: cross  Abstract: Reinforcement learning from human feedback (RLHF) has been extensively employed to align large language models with user intent. However, proximal policy optimization (PPO) based RLHF is occasionally unstable requiring significant hyperparameter finetuning, and computationally expensive to maximize the estimated reward during alignment. Recently, direct preference optimization (DPO) is proposed to address those challenges. However, DPO relies on contrastive responses generated from human annotator and alternative LLM, instead of the policy model, limiting the effectiveness of the RLHF. In this paper, we addresses both challenges by systematically combining rejection sampling (RS) and DPO. Our proposed method, RS-DPO, initiates with the development of a supervised fine-tuned policy model (SFT). A varied set of k responses per prompt are sampled directly from the SFT model. RS-DPO identifies pairs of contrastive samples based on their re
    
[^60]: 预测性线性在线追踪未知目标

    Predictive Linear Online Tracking for Unknown Targets

    [https://arxiv.org/abs/2402.10036](https://arxiv.org/abs/2402.10036)

    本文提出了一种名为预测性线性在线追踪（PLOT）的算法，用于在线追踪未知目标。该算法使用具有指数遗忘的递归最小二乘法来学习目标的时变动态模型，并在递推视线控制的框架下使用所学模型进行最优策略。与先前的工作不同，我们的理论结果适用于非平稳目标。

    

    本文研究了在线线性控制系统中的追踪问题，目标是跟随一个移动的目标。与经典的追踪控制不同，目标是未知的、非平稳的，并且它的状态逐步揭示，因此适合在线非随机控制的框架。我们考虑了二次成本的情况，并提出了一种新算法，称为预测性线性在线追踪（PLOT）。该算法使用具有指数遗忘的递归最小二乘法来学习目标的时变动态模型。所学模型在递推视线控制的框架下用于优化策略。我们证明了PLOT的动态遗憾与$\mathcal{O}(\sqrt{TV_T})$成比例，其中$V_T$是目标动力学的总变化量，$T$是时间长度。与先前的工作不同，我们的理论结果适用于非平稳目标。我们在一个真实的四旋翼机上实现了PLOT，并提供了开源代码。

    arXiv:2402.10036v1 Announce Type: cross  Abstract: In this paper, we study the problem of online tracking in linear control systems, where the objective is to follow a moving target. Unlike classical tracking control, the target is unknown, non-stationary, and its state is revealed sequentially, thus, fitting the framework of online non-stochastic control. We consider the case of quadratic costs and propose a new algorithm, called predictive linear online tracking (PLOT). The algorithm uses recursive least squares with exponential forgetting to learn a time-varying dynamic model of the target. The learned model is used in the optimal policy under the framework of receding horizon control. We show the dynamic regret of PLOT scales with $\mathcal{O}(\sqrt{TV_T})$, where $V_T$ is the total variation of the target dynamics and $T$ is the time horizon. Unlike prior work, our theoretical results hold for non-stationary targets. We implement PLOT on a real quadrotor and provide open-source so
    
[^61]: 扩散模型与大动作空间情境强化学习的结合

    Diffusion Models Meet Contextual Bandits with Large Action Spaces

    [https://arxiv.org/abs/2402.10028](https://arxiv.org/abs/2402.10028)

    本文设计了一种利用预训练扩散模型的扩散汤普森采样方法，用于在大动作空间下进行高效的情境强化学习探索。实证评估结果表明了该方法的优越性能。

    

    由于动作空间较大，有效的探索是情境强化学习中的一个关键挑战。本文通过利用预训练的扩散模型来捕捉动作之间的相关性，设计了扩散汤普森采样（dTS）方法，实现了高效的探索。我们为dTS方法提供了理论和算法基础，并通过实证评估展示了它的优越性能。

    arXiv:2402.10028v1 Announce Type: cross  Abstract: Efficient exploration is a key challenge in contextual bandits due to the large size of their action space, where uninformed exploration can result in computational and statistical inefficiencies. Fortunately, the rewards of actions are often correlated and this can be leveraged to explore them efficiently. In this work, we capture such correlations using pre-trained diffusion models; upon which we design diffusion Thompson sampling (dTS). Both theoretical and algorithmic foundations are developed for dTS, and empirical evaluation also shows its favorable performance.
    
[^62]: 自学习上下文增强对于无监督词汇翻译的研究

    Self-Augmented In-Context Learning for Unsupervised Word Translation

    [https://arxiv.org/abs/2402.10024](https://arxiv.org/abs/2402.10024)

    通过自学习上下文增强方法，本论文提出一种无监督词汇翻译的方法，在零样本提示的大型语言模型上取得了显著的改进，超过了传统基于映射的方法。

    

    近期的研究表明，尽管大型语言模型在一些小规模的设置中展示出了较强的词汇翻译和双语词典诱导(BLI)的能力，但在无监督的情况下，即没有种子翻译对可用的情况下，尤其是对于资源较少的语言，它们仍然无法达到“传统”的基于映射的方法的性能。为了解决这个挑战，我们提出了一种自学习上下文增强方法 (SAIL) 来进行无监督的BLI：从零样本提示开始，SAIL通过迭代地从LLM中引出一组高置信度的词汇翻译对，然后在ICL的方式下再次应用于同一个LLM中。我们的方法在两个广泛的BLI基准测试中，跨越多种语言对，在零样本提示的LLM上取得了显著的改进，也在各个方面优于基于映射的基线。除了达到最先进的无监督

    arXiv:2402.10024v1 Announce Type: cross  Abstract: Recent work has shown that, while large language models (LLMs) demonstrate strong word translation or bilingual lexicon induction (BLI) capabilities in few-shot setups, they still cannot match the performance of 'traditional' mapping-based approaches in the unsupervised scenario where no seed translation pairs are available, especially for lower-resource languages. To address this challenge with LLMs, we propose self-augmented in-context learning (SAIL) for unsupervised BLI: starting from a zero-shot prompt, SAIL iteratively induces a set of high-confidence word translation pairs for in-context learning (ICL) from an LLM, which it then reapplies to the same LLM in the ICL fashion. Our method shows substantial gains over zero-shot prompting of LLMs on two established BLI benchmarks spanning a wide range of language pairs, also outperforming mapping-based baselines across the board. In addition to achieving state-of-the-art unsupervised 
    
[^63]: 使用DDPM反转进行零样本无监督和基于文本的音频编辑

    Zero-Shot Unsupervised and Text-Based Audio Editing Using DDPM Inversion

    [https://arxiv.org/abs/2402.10009](https://arxiv.org/abs/2402.10009)

    本文研究了使用DDPM反转进行音频信号的零样本编辑技术，包括基于文本的编辑和无监督发现编辑方向。这些方法在音乐信号中展现了多样的音乐兴趣修改。

    

    使用大型预训练模型进行零样本编辑已经在图像领域取得了迅猛的发展，但在音频领域尚未出现。本文中，我们探索了两种基于DDPM反转的音频信号零样本编辑技术。第一种是从图像领域采用的方法，允许基于文本进行编辑。第二种是一种新颖的方法，可以在无监督情况下发现语义上有意义的编辑方向。当应用于音乐信号时，这种方法可以展现出一系列具有音乐兴趣的修改，从控制特定乐器的参与到对旋律进行即兴演奏。示例可以在我们的例子页面中找到：https://hilamanor.github.io/AudioEditing/ ，代码可以在 https://github.com/hilamanor/AudioEditing/ 找到。

    arXiv:2402.10009v1 Announce Type: cross  Abstract: Editing signals using large pre-trained models, in a zero-shot manner, has recently seen rapid advancements in the image domain. However, this wave has yet to reach the audio domain. In this paper, we explore two zero-shot editing techniques for audio signals, which use DDPM inversion on pre-trained diffusion models. The first, adopted from the image domain, allows text-based editing. The second, is a novel approach for discovering semantically meaningful editing directions without supervision. When applied to music signals, this method exposes a range of musically interesting modifications, from controlling the participation of specific instruments to improvisations on the melody. Samples can be found on our examples page in https://hilamanor.github.io/AudioEditing/ and code can be found in https://github.com/hilamanor/AudioEditing/ .
    
[^64]: ML-ASPA: 机器学习在声学信号处理分析中的思考

    ML-ASPA: A Contemplation of Machine Learning-based Acoustic Signal Processing Analysis for Sounds, & Strains Emerging Technology

    [https://arxiv.org/abs/2402.10005](https://arxiv.org/abs/2402.10005)

    本文研究了机器学习在声学信号处理分析中的应用，通过数据驱动的方法，揭示了复杂声学现象的模型。

    

    声学数据在推动科学和工程理解方面起着基本的基石作用，涉及生物学、通信学以及海洋和地球科学等多个学科。本文详细探讨了声学领域中最近的进展和变革潜力，特别关注机器学习（ML）和深度学习。与传统的声学和信号处理相比，ML采用数据驱动的方法，揭示了特征与期望标签或动作之间以及特征之间的复杂关系，给定充足的训练数据。将ML应用于大量的训练数据集有助于发现能够解释人类语音和混响等复杂声学现象的模型。

    arXiv:2402.10005v1 Announce Type: cross  Abstract: Acoustic data serves as a fundamental cornerstone in advancing scientific and engineering understanding across diverse disciplines, spanning biology, communications, and ocean and Earth science. This inquiry meticulously explores recent advancements and transformative potential within the domain of acoustics, specifically focusing on machine learning (ML) and deep learning. ML, comprising an extensive array of statistical techniques, proves indispensable for autonomously discerning and leveraging patterns within data. In contrast to traditional acoustics and signal processing, ML adopts a data-driven approach, unveiling intricate relationships between features and desired labels or actions, as well as among features themselves, given ample training data. The application of ML to expansive sets of training data facilitates the discovery of models elucidating complex acoustic phenomena such as human speech and reverberation. The dynamic 
    
[^65]: 分布式学习中的隐私攻击

    Privacy Attacks in Decentralized Learning

    [https://arxiv.org/abs/2402.10001](https://arxiv.org/abs/2402.10001)

    该论文介绍了分布式学习中的隐私攻击，针对分布式梯度下降（D-GD）提出了首个攻击方法，能够使用户重建其邻域之外其他用户的私有数据，并验证了这种攻击的有效性。

    

    分布式梯度下降（D-GD）允许一组用户在网络图中通过迭代平均本地模型更新与其邻居合作学习而无需共享数据。非邻居节点之间的直接通信的缺失可能导致用户无法推断出关于其他用户数据的精确信息。在这项工作中，我们提出了首个针对D-GD的攻击，使一个用户（或一组用户）能够重建其邻域之外其他用户的私有数据。我们的方法基于对传闻平均协议的重建攻击，然后将其扩展以处理D-GD提出的额外挑战。我们在真实图和数据集上验证了我们攻击的有效性，结果显示单个或少数攻击者所威胁到的用户数量通常是令人惊讶的大。我们对一些方案进行了经验性的研究。

    arXiv:2402.10001v1 Announce Type: new  Abstract: Decentralized Gradient Descent (D-GD) allows a set of users to perform collaborative learning without sharing their data by iteratively averaging local model updates with their neighbors in a network graph. The absence of direct communication between non-neighbor nodes might lead to the belief that users cannot infer precise information about the data of others. In this work, we demonstrate the opposite, by proposing the first attack against D-GD that enables a user (or set of users) to reconstruct the private data of other users outside their immediate neighborhood. Our approach is based on a reconstruction attack against the gossip averaging protocol, which we then extend to handle the additional challenges raised by D-GD. We validate the effectiveness of our attack on real graphs and datasets, showing that the number of users compromised by a single or a handful of attackers is often surprisingly large. We empirically investigate some
    
[^66]: 风险敏感的软策演员-评论家算法在分布偏移下的鲁棒深度强化学习中的应用

    Risk-Sensitive Soft Actor-Critic for Robust Deep Reinforcement Learning under Distribution Shifts

    [https://arxiv.org/abs/2402.09992](https://arxiv.org/abs/2402.09992)

    本论文研究了在运营研究领域中，深度强化学习算法在面对分布偏移时的鲁棒性。通过推导出一种风险敏感的深度强化学习算法，并通过数值证据验证其有效性，填补了这一领域实际性能研究的空白。

    

    我们研究了在运营研究领域的上下文多阶段随机组合优化问题中，深度强化学习算法对于分布偏移的鲁棒性。在这个背景下，风险敏感算法可以学习到鲁棒的策略。尽管这个领域对强化学习社区非常重要，但大部分研究都着重于理论结果而不是实际性能。本文的目标是填补这一空白，通过正式推导出一种新颖的风险敏感的深度强化学习算法，并提供其有效性的数值证据。具体地，我们通过导出相应的Q值的Bellman方程的版本，引入了离散式的软策演员-评论家算法来进行基于熵风险度量的策略学习。我们建立了一个相应的策略改进结果并推导出一个实际的算法。我们还引入了一个典型的上下文多阶段环境。

    arXiv:2402.09992v1 Announce Type: new  Abstract: We study the robustness of deep reinforcement learning algorithms against distribution shifts within contextual multi-stage stochastic combinatorial optimization problems from the operations research domain. In this context, risk-sensitive algorithms promise to learn robust policies. While this field is of general interest to the reinforcement learning community, most studies up-to-date focus on theoretical results rather than real-world performance. With this work, we aim to bridge this gap by formally deriving a novel risk-sensitive deep reinforcement learning algorithm while providing numerical evidence for its efficacy. Specifically, we introduce discrete Soft Actor-Critic for the entropic risk measure by deriving a version of the Bellman equation for the respective Q-values. We establish a corresponding policy improvement result and infer a practical algorithm. We introduce an environment that represents typical contextual multi-sta
    
[^67]: TIAViz：一种基于浏览器的计算病理学模型可视化工具

    TIAViz: A Browser-based Visualization Tool for Computational Pathology Models

    [https://arxiv.org/abs/2402.09990](https://arxiv.org/abs/2402.09990)

    TIAViz是一种基于浏览器的计算病理学模型可视化工具，可以灵活、交互式地显示图表、热图、分割、标注和其他信息在整个切片图像上。

    

    数字病理学在现代医疗系统中获得了显著的关注。这种从光学显微镜到数字图像的转变带来了提高诊断效率和将人工智能工具整合到病理学家工作流程中的潜力。其中一个关键方面是可视化。在数字病理学中开发机器学习（ML）模型的过程中，拥有灵活、开放的可视化工具以可视化模型的输出、预测以及用于训练或测试模型的底层注释和图像至关重要。我们介绍了TIAViz，它是一个基于Python的可视化工具，内置于TIAToolbox中，允许在整个切片图像上灵活、交互式、完全可缩放地叠加各种信息，包括图表、热图、分割、标注和其他WSI。用户界面基于浏览器，可以在本地、远程计算机上或服务器上使用，提供公开访问。

    arXiv:2402.09990v1 Announce Type: cross  Abstract: Digital pathology has gained significant traction in modern healthcare systems. This shift from optical microscopes to digital imagery brings with it the potential for improved diagnosis, efficiency, and the integration of AI tools into the pathologists workflow. A critical aspect of this is visualization. Throughout the development of a machine learning (ML) model in digital pathology, it is crucial to have flexible, openly available tools to visualize models, from their outputs and predictions to the underlying annotations and images used to train or test a model. We introduce TIAViz, a Python-based visualization tool built into TIAToolbox which allows flexible, interactive, fully zoomable overlay of a wide variety of information onto whole slide images, including graphs, heatmaps, segmentations, annotations and other WSIs. The UI is browser-based, allowing use either locally, on a remote machine, or on a server to provide publicly a
    
[^68]: 对于临时团队合作的对称破缺增强

    Symmetry-Breaking Augmentations for Ad Hoc Teamwork

    [https://arxiv.org/abs/2402.09984](https://arxiv.org/abs/2402.09984)

    本研究提出了一种称为对称破缺增强的方法，通过增加训练队友的行为多样性来提高人工智能代理与新队友合作的性能。实验证明了该方法的有效性。

    

    在许多协作环境中，人工智能（AI）代理必须能够适应使用未知或先前未观察到的策略的新队友。对于AI代理来说，这通常对人类来说很简单，但却是一项具有挑战性的任务。例如，如果一个AI代理在训练集中学会了与只在一侧道路上行驶的其他车辆并行驶，那么即使这些车辆的行为只是在左右对称上进行了翻转，它也可能难以适应与相反方向上行驶的驾驶员进行协调。为了解决这个问题，我们引入了对称破缺增强（SBA），通过应用对称翻转操作来增加训练队友的行为多样性。通过学习对增强后的队友的最佳响应，我们的代理能够接触到更广泛的行为约定，从而提高与新队友合作时的性能。我们在两个设置中进行了实验验证，并证明了我们的方法的有效性。

    arXiv:2402.09984v1 Announce Type: cross  Abstract: In many collaborative settings, artificial intelligence (AI) agents must be able to adapt to new teammates that use unknown or previously unobserved strategies. While often simple for humans, this can be challenging for AI agents. For example, if an AI agent learns to drive alongside others (a training set) that only drive on one side of the road, it may struggle to adapt this experience to coordinate with drivers on the opposite side, even if their behaviours are simply flipped along the left-right symmetry. To address this we introduce symmetry-breaking augmentations (SBA), which increases diversity in the behaviour of training teammates by applying a symmetry-flipping operation. By learning a best-response to the augmented set of teammates, our agent is exposed to a wider range of behavioural conventions, improving performance when deployed with novel teammates. We demonstrate this experimentally in two settings, and show that our a
    
[^69]: 数据增强和迁移学习应用于面部表情识别

    Data Augmentation and Transfer Learning Approaches Applied to Facial Expressions Recognition

    [https://arxiv.org/abs/2402.09982](https://arxiv.org/abs/2402.09982)

    本文提出了一种改进面部表情识别的新型数据增强技术，并应用迁移学习方法，通过使用预训练卷积神经网络在增强的数据集上进行微调，实现了高达85%的平均准确度。

    

    面部表情是我们在理解一个人的心理状态时首先关注的事物。因此，能够自动识别面部表情是一个非常有趣的研究领域。在这篇论文中，由于可用训练数据集的规模较小，我们提出了一种改进识别任务性能的新型数据增强技术。我们应用几何变换，并从头构建了能够为每种情绪类型生成新的合成图像的GAN模型。因此，在增强的数据集上，我们使用不同架构的预训练卷积神经网络进行微调。为了衡量模型的泛化能力，我们采用了额外数据库协议方法，即我们在经过增强的训练数据集上训练模型，然后在两个不同的数据库上进行测试。这些技术的组合使得可以达到平均准确度约为85%的数值。

    arXiv:2402.09982v1 Announce Type: cross  Abstract: The face expression is the first thing we pay attention to when we want to understand a person's state of mind. Thus, the ability to recognize facial expressions in an automatic way is a very interesting research field. In this paper, because the small size of available training datasets, we propose a novel data augmentation technique that improves the performances in the recognition task. We apply geometrical transformations and build from scratch GAN models able to generate new synthetic images for each emotion type. Thus, on the augmented datasets we fine tune pretrained convolutional neural networks with different architectures. To measure the generalization ability of the models, we apply extra-database protocol approach, namely we train models on the augmented versions of training dataset and test them on two different databases. The combination of these techniques allows to reach average accuracy values of the order of 85\% for 
    
[^70]: 用于非厄米拓电路设计的深度学习

    Deep learning for the design of non-Hermitian topolectrical circuits

    [https://arxiv.org/abs/2402.09978](https://arxiv.org/abs/2402.09978)

    用深度学习设计非厄米拓电路，通过多层感知器和卷积神经网络预测非厄米哈密顿量的本征值，利用DenseNet算法设计高维拓扑电路，证明了深度学习网络在捕捉全局拓扑特性方面的有效性。

    

    非厄米拓扑相相对于其厄米对应物具有一些出色的特性，例如传统的体-边对应的破裂以及非厄米拓扑边模态。我们在深度学习领域引入了几种基于多层感知器(MLP)和卷积神经网络(CNN)的算法，用于预测非厄米哈密顿量的本征值。随后，我们使用周期性电路的最小模块作为一个单元来构建高维电路数据特征。进一步地，我们使用DenseNet算法，它是一种利用层之间的密集连接的卷积神经网络，来设计非厄米拓电陈电路，因为DenseNet算法更适合处理高维数据。我们的结果证明了深度学习网络在捕捉全局拓扑特性方面的有效性。

    arXiv:2402.09978v1 Announce Type: cross  Abstract: Non-Hermitian topological phases can produce some remarkable properties, compared with their Hermitian counterpart, such as the breakdown of conventional bulk-boundary correspondence and the non-Hermitian topological edge mode. Here, we introduce several algorithms with multi-layer perceptron (MLP), and convolutional neural network (CNN) in the field of deep learning, to predict the winding of eigenvalues non-Hermitian Hamiltonians. Subsequently, we use the smallest module of the periodic circuit as one unit to construct high-dimensional circuit data features. Further, we use the Dense Convolutional Network (DenseNet), a type of convolutional neural network that utilizes dense connections between layers to design a non-Hermitian topolectrical Chern circuit, as the DenseNet algorithm is more suitable for processing high-dimensional data. Our results demonstrate the effectiveness of the deep learning network in capturing the global topol
    
[^71]: 语言模型压缩的快速词汇转移方法

    Fast Vocabulary Transfer for Language Model Compression

    [https://arxiv.org/abs/2402.09977](https://arxiv.org/abs/2402.09977)

    提出了一种基于词汇转移的语言模型压缩方法，通过与其他压缩技术结合使用，显著减小模型大小和推理时间，同时性能略有妥协。

    

    实际业务应用需要在语言模型性能和大小之间做出权衡。我们提出了一种基于词汇转移的模型压缩方法。我们在不同垂直领域和下游任务中评估了该方法。我们的结果表明，词汇转移可以与其他压缩技术有效结合使用，显著减小模型大小和推理时间，同时在性能上略有妥协。

    arXiv:2402.09977v1 Announce Type: cross  Abstract: Real-world business applications require a trade-off between language model performance and size. We propose a new method for model compression that relies on vocabulary transfer. We evaluate the method on various vertical domains and downstream tasks. Our results indicate that vocabulary transfer can be effectively used in combination with other compression techniques, yielding a significant reduction in model size and inference time while marginally compromising on performance.
    
[^72]: 加速并行采样扩散模型

    Accelerating Parallel Sampling of Diffusion Models

    [https://arxiv.org/abs/2402.09970](https://arxiv.org/abs/2402.09970)

    本文提出了一种并行化自回归过程来加速扩散模型的采样的方法，并引入了ParaTAA，一种通用的并行采样算法，可以显著减少推理步骤。

    

    扩散模型已经成为图像生成的最先进生成模型。然而，由于其采样过程中固有的自回归性质，从扩散模型中进行采样通常耗时。在本文中，我们提出了一种新的方法，通过并行化自回归过程来加速扩散模型的采样。具体而言，我们将采样过程重新构建为通过固定点迭代解决三角非线性方程组的过程。通过这种创新的公式，我们探索了一些系统化的技术，进一步减少了求解过程所需的迭代步骤。应用这些技术，我们引入了ParaTAA，一种通用的、无需训练的并行采样算法，可以利用额外的计算和内存资源来增加采样速度。我们的实验表明，ParaTAA可以减少常见的顺序采样所需的推理步骤。

    arXiv:2402.09970v1 Announce Type: new  Abstract: Diffusion models have emerged as state-of-the-art generative models for image generation. However, sampling from diffusion models is usually time-consuming due to the inherent autoregressive nature of their sampling process. In this work, we propose a novel approach that accelerates the sampling of diffusion models by parallelizing the autoregressive process. Specifically, we reformulate the sampling process as solving a system of triangular nonlinear equations through fixed-point iteration. With this innovative formulation, we explore several systematic techniques to further reduce the iteration steps required by the solving process. Applying these techniques, we introduce ParaTAA, a universal and training-free parallel sampling algorithm that can leverage extra computational and memory resources to increase the sampling speed. Our experiments demonstrate that ParaTAA can decrease the inference steps required by common sequential sampli
    
[^73]: 机器学习中数据的层次化表示

    Hierarchy Representation of Data in Machine Learnings

    [https://arxiv.org/abs/2402.09965](https://arxiv.org/abs/2402.09965)

    该论文提出了一种用于可视化目标间层次关系的方法，对于模型改进具有潜在的益处。

    

    当存在多个数据点的模型具有明确的判断结果时，大多数模型可能展示出一种关系，即如果它们正确判断一个目标，则它们也会正确判断另一个目标。相反，如果大多数模型错误地判断一个目标，它们可能也会错误地判断另一个目标。我们提出了一种可视化目标之间层次关系的方法。这些信息有望对模型改进有益。

    arXiv:2402.09965v1 Announce Type: cross  Abstract: When there are models with clear-cut judgment results for several data points, it is possible that most models exhibit a relationship where if they correctly judge one target, they also correctly judge another target. Conversely, if most models incorrectly judge one target, they may also incorrectly judge another target. We propose a method for visualizing this hierarchy among targets. This information is expected to be beneficial for model improvement.
    
[^74]: 为什么Transformer对敏感函数困难?

    Why are Sensitive Functions Hard for Transformers?

    [https://arxiv.org/abs/2402.09963](https://arxiv.org/abs/2402.09963)

    本文证明了在Transformer架构下，损失函数的空间受到输入敏感性的限制，从而解释了Transformer对敏感函数的困难。这一理论统一了关于Transformer学习能力和偏见的广泛观察。

    

    经验研究发现，Transformer存在一系列的学习偏见和限制，如在学习计算简单形式语言（如PARITY）时的持久困难，以及对低阶函数的偏好。然而，现有的表达能力理论要么过度预测，要么低估了实际的学习能力。我们证明，在Transformer架构下，损失函数的空间受到输入敏感性的限制：输出对输入字符串的多个部分敏感的Transformer存在于参数空间中的孤立点，导致泛化中的低敏感性偏差。我们理论上和实证上证明了该理论统一了关于Transformer学习能力和偏见的广泛观察，如它们对低敏感性和低阶的泛化偏差，以及在长度泛化上的困难。

    arXiv:2402.09963v1 Announce Type: new  Abstract: Empirical studies have identified a range of learnability biases and limitations of transformers, such as a persistent difficulty in learning to compute simple formal languages such as PARITY, and a bias towards low-degree functions. However, theoretical understanding remains limited, with existing expressiveness theory either overpredicting or underpredicting realistic learning abilities. We prove that, under the transformer architecture, the loss landscape is constrained by the input-space sensitivity: Transformers whose output is sensitive to many parts of the input string inhabit isolated points in parameter space, leading to a low-sensitivity bias in generalization. We show theoretically and empirically that this theory unifies a broad array of empirical observations about the learning abilities and biases of transformers, such as their generalization bias towards low sensitivity and low degree, and difficulty in length generalizati
    
[^75]: 通过动态扩展班次的方式提升众包最后一公里配送中的快递员调度：一种深度强化学习方法

    Enhancing Courier Scheduling in Crowdsourced Last-Mile Delivery through Dynamic Shift Extensions: A Deep Reinforcement Learning Approach

    [https://arxiv.org/abs/2402.09961](https://arxiv.org/abs/2402.09961)

    本研究提出了一种通过对离线计划进行动态调整来优化众包最后一公里配送中快递员调度的方法。研究采用了深度强化学习算法，旨在最大化众包平台的利润。

    

    众包配送平台面临着匹配快递员和顾客订单的复杂调度挑战。我们考虑了两种类型的众包快递员，即承诺型和偶发型快递员，每种类型的快递员有不同的补偿方案。众包配送平台通常根据预测的需求为承诺型快递员安排班次。因此，平台可能会在计划周期之前为承诺型快递员制定离线计划。然而，由于需求的不可预测性，有时需要对离线计划进行在线调整。在这个研究中，我们关注通过为承诺型快递员提供班次扩展来动态调整离线计划的问题。这个问题被建模为一个顺序决策过程。目标是通过确定快递员的班次扩展和请求分配给快递员的方式来最大化平台利润。为了解决这个模型，使用了深度Q网络方法。

    arXiv:2402.09961v1 Announce Type: new  Abstract: Crowdsourced delivery platforms face complex scheduling challenges to match couriers and customer orders. We consider two types of crowdsourced couriers, namely, committed and occasional couriers, each with different compensation schemes. Crowdsourced delivery platforms usually schedule committed courier shifts based on predicted demand. Therefore, platforms may devise an offline schedule for committed couriers before the planning period. However, due to the unpredictability of demand, there are instances where it becomes necessary to make online adjustments to the offline schedule. In this study, we focus on the problem of dynamically adjusting the offline schedule through shift extensions for committed couriers. This problem is modeled as a sequential decision process. The objective is to maximize platform profit by determining the shift extensions of couriers and the assignments of requests to couriers. To solve the model, a Deep Q-Ne
    
[^76]: 设计旋转机械状态监测特征的研究

    On Designing Features for Condition Monitoring of Rotating Machines

    [https://arxiv.org/abs/2402.09957](https://arxiv.org/abs/2402.09957)

    本文提出了一种新的算法，通过直方图理论设计出适用于不同时间序列传感器数据的输入特征抽取方法，为机械状态识别提供了一种统一的特征提取过程。

    

    有关使用一维原始传感器数据进行旋转机械故障识别，已经提出了各种设计输入特征的方法。现有方法复杂，依赖经验方法，并且可能因使用的条件监测数据而不同。因此，本文提出了一种新的算法，用于设计适用于不同时间序列传感器数据的输入特征抽取方法。这种新的特征设计/抽取方法是通过直方图理论获得的。所提出的算法提取具有鉴别性的输入特征，适用于简单分类器和深度神经网络分类器。设计的输入特征作为输入传递给端到端训练的分类器，用于单一框架下的机械状态识别。通过三个实时数据集验证了该方案：a) 声学数据集，b) CWRU振动数据集。

    arXiv:2402.09957v1 Announce Type: new  Abstract: Various methods for designing input features have been proposed for fault recognition in rotating machines using one-dimensional raw sensor data. The available methods are complex, rely on empirical approaches, and may differ depending on the condition monitoring data used. Therefore, this article proposes a novel algorithm to design input features that unifies the feature extraction process for different time-series sensor data. This new insight for designing/extracting input features is obtained through the lens of histogram theory. The proposed algorithm extracts discriminative input features, which are suitable for a simple classifier to deep neural network-based classifiers. The designed input features are given as input to the classifier with end-to-end training in a single framework for machine conditions recognition. The proposed scheme has been validated through three real-time datasets: a) acoustic dataset, b) CWRU vibration da
    
[^77]: 制定良好提示还是提供出色的对话？关于基于上下文学习的角色生成对话的研究

    Crafting a Good Prompt or Providing Exemplary Dialogues? A Study of In-Context Learning for Persona-based Dialogue Generation

    [https://arxiv.org/abs/2402.09954](https://arxiv.org/abs/2402.09954)

    本研究通过对大型语言模型在基于角色生成对话方面进行实验，发现调整提示指令可以最直接有效且经济地提高生成质量，并且随机检索示范会取得最佳结果，而查询相同上下文的示范检索效果最差。即使破坏了示范中的多回合关联和单回合语义，对话生成仍然有效。

    

    先前关于上下文学习（ICL）的研究主要侧重于分类、机器翻译、文本到表格等任务，而对于ICL能否改进生成类似人类对话的研究很少。我们的工作通过在高质量的真实人类对话数据集上进行广泛的实验，系统地研究了大型语言模型（LLMs）在基于角色生成对话方面的ICL能力。根据实验结果，我们得出三个结论：1）调整提示指令是提高生成质量最直接、有效和经济的方法；2）随机检索示范可以取得最佳的结果，可能是因为具有更多样化和有效信息的原因；与查询相同上下文的示范检索结果最差；3）即使破坏了示范中的多回合关联和单回合语义，对话生成仍然可以实现较好的效果。

    arXiv:2402.09954v1 Announce Type: new  Abstract: Previous in-context learning (ICL) research has focused on tasks such as classification, machine translation, text2table, etc., while studies on whether ICL can improve human-like dialogue generation are scarce. Our work fills this gap by systematically investigating the ICL capabilities of large language models (LLMs) in persona-based dialogue generation, conducting extensive experiments on high-quality real human Chinese dialogue datasets. From experimental results, we draw three conclusions: 1) adjusting prompt instructions is the most direct, effective, and economical way to improve generation quality; 2) randomly retrieving demonstrations (demos) achieves the best results, possibly due to the greater diversity and the amount of effective information; counter-intuitively, retrieving demos with a context identical to the query performs the worst; 3) even when we destroy the multi-turn associations and single-turn semantics in the demo
    
[^78]: 多词标记化用于序列压缩

    Multi-Word Tokenization for Sequence Compression

    [https://arxiv.org/abs/2402.09949](https://arxiv.org/abs/2402.09949)

    本论文介绍了一种名为MWT的多词标记器，通过将频繁出现的多词表达式表示为单个标记，突破了词边界的限制，从而实现更紧凑和高效的标记化，提高了性能并加速推理过程。

    

    大型语言模型在建模各种任务方面取得了极大成功。然而，这也意味着计算成本的大幅增加，限制了其在工业界的广泛应用。本论文介绍了一种名为MWT的多词标记器，通过将频繁出现的多词表达式表示为单个标记，突破了词边界的限制。MWT产生了更紧凑和高效的标记化结果，带来两个好处：（1）在固定序列长度和预算的情况下，提高了性能，因为能够更全面地覆盖输入数据；（2）由于能够减少序列长度而对性能几乎没有影响，从而实现更快速和更轻量的推理过程。我们的结果表明，MWT在较短的序列长度下更为稳健，从而可以通过早期序列截断实现重大加速。

    arXiv:2402.09949v1 Announce Type: new  Abstract: Large Language Models have proven highly successful at modelling a variety of tasks. However, this comes at a steep computational cost that hinders wider industrial uptake. In this pa005 per, we present MWT: a Multi-Word Tokenizer that goes beyond word boundaries by representing frequent multi-word expressions as single tokens. MWTs produce a more compact and efficient tokenization that yields two benefits: (1) Increase in performance due to a greater coverage of input data given a fixed sequence length and budget; (2) Faster and lighter inference due to the ability to reduce the sequence length with negligible drops in performance. Our results show that MWT is more robust across shorter sequence lengths, thus allowing for major speedups via early sequence truncation.
    
[^79]: 带有IMU监督的神经网络5G室内定位

    Neural 5G Indoor Localization with IMU Supervision

    [https://arxiv.org/abs/2402.09948](https://arxiv.org/abs/2402.09948)

    本论文研究了带有IMU监督的神经网络5G室内定位问题，提出了基于IMU的伪标签和实用算法，并展示了与完全监督方法相当的性能。

    

    无线电信号非常适合用于用户定位，因为它们无处不在，可以在黑暗中运行并保持隐私。许多先前的工作都是通过全面监督学习信道状态信息（CSI）和位置之间的映射关系。然而，这种方法依赖于昂贵的位置标签。在这项工作中，我们通过在部署过程中使用从惯性测量单元（IMU）计算出的伪标签来放宽这一要求。我们提出了IMU双积分和定位系统训练的实用算法。我们展示了在5G测量的模拟和具有挑战性的实际数据上具有分米级精度。我们的IMU监督方法表现出与完全监督方法类似的性能，但需要更少的部署工作。

    arXiv:2402.09948v1 Announce Type: cross  Abstract: Radio signals are well suited for user localization because they are ubiquitous, can operate in the dark and maintain privacy. Many prior works learn mappings between channel state information (CSI) and position fully-supervised. However, that approach relies on position labels which are very expensive to acquire. In this work, this requirement is relaxed by using pseudo-labels during deployment, which are calculated from an inertial measurement unit (IMU). We propose practical algorithms for IMU double integration and training of the localization system. We show decimeter-level accuracy on simulated and challenging real data of 5G measurements. Our IMU-supervised method performs similarly to fully-supervised, but requires much less effort to deploy.
    
[^80]: 用分布值解释概率模型

    Explaining Probabilistic Models with Distributional Values

    [https://arxiv.org/abs/2402.09947](https://arxiv.org/abs/2402.09947)

    本文介绍了一种用于解释概率模型的方法，通过引入分布值来解决当前方法在解释模型输出时的不匹配问题，并通过案例研究展示了该方法提供的详细和有洞察力的解释。

    

    一个重要的可解释机器学习分支基于合作博弈理论。然而，研究表明博弈理论解释可能会误导或难以解释。我们认为通常存在着一个重要的不匹配，即人们希望解释的内容（例如分类器的输出）与当前的方法（例如SHAP）所解释的内容（例如类别的概率）之间。本文通过推广合作博弈和价值算子，来解决概率模型的这种差距。我们引入了分布值，这是一种随机变量，用于追踪模型输出的变化（例如预测类别的反转），并推导出了在具有高斯、伯努利和分类支付的博弈中的分布值的解析表达式。我们进一步建立了几个特性，并通过对视觉和语言模型的案例研究展示了我们的框架提供了细粒度和有洞察力的解释。

    arXiv:2402.09947v1 Announce Type: new  Abstract: A large branch of explainable machine learning is grounded in cooperative game theory. However, research indicates that game-theoretic explanations may mislead or be hard to interpret. We argue that often there is a critical mismatch between what one wishes to explain (e.g. the output of a classifier) and what current methods such as SHAP explain (e.g. the scalar probability of a class). This paper addresses such gap for probabilistic models by generalising cooperative games and value operators. We introduce the distributional values, random variables that track changes in the model output (e.g. flipping of the predicted class) and derive their analytic expressions for games with Gaussian, Bernoulli and Categorical payoffs. We further establish several characterising properties, and show that our framework provides fine-grained and insightful explanations with case studies on vision and language models.
    
[^81]: FedLion: 更快的自适应联邦优化算法，通信更少

    FedLion: Faster Adaptive Federated Optimization with Fewer Communication

    [https://arxiv.org/abs/2402.09941](https://arxiv.org/abs/2402.09941)

    FedLion是一种自适应联邦优化算法，通过引入集中式自适应算法Lion的关键元素，实现了更快的收敛速度和更少的通信成本。经过广泛评估，FedLion优于之前的最先进自适应算法，并通过使用有符号梯度在本地训练中减少数据传输要求。

    

    在联邦学习（FL）中，一种跨分布式数据训练机器学习模型的框架中，像FedAvg这样的知名算法往往具有较慢的收敛速度，在训练过程中导致高通信成本。为了解决这个挑战，我们引入了FedLion，一种自适应联邦优化算法，无缝地将最近提出的集中式自适应算法Lion（Chen et al. 2023）的关键元素融入到FL框架中。通过对两个广泛采用的FL基准进行全面评估，我们证明了FedLion优于之前的最先进自适应算法，包括FAFED（Wu et al. 2023）和FedDA。此外，由于在本地训练中使用了有符号梯度，与现有的自适应算法相比，FedLion在上行通信过程中大大降低了数据传输要求，进一步降低了通信成本。

    arXiv:2402.09941v1 Announce Type: cross  Abstract: In Federated Learning (FL), a framework to train machine learning models across distributed data, well-known algorithms like FedAvg tend to have slow convergence rates, resulting in high communication costs during training. To address this challenge, we introduce FedLion, an adaptive federated optimization algorithm that seamlessly incorporates key elements from the recently proposed centralized adaptive algorithm, Lion (Chen et al. 2o23), into the FL framework. Through comprehensive evaluations on two widely adopted FL benchmarks, we demonstrate that FedLion outperforms previous state-of-the-art adaptive algorithms, including FAFED (Wu et al. 2023) and FedDA. Moreover, thanks to the use of signed gradients in local training, FedLion substantially reduces data transmission requirements during uplink communication when compared to existing adaptive algorithms, further reducing communication costs. Last but not least, this work also incl
    
[^82]: 建筑行业中的生成式人工智能：一项最新分析

    Generative AI in the Construction Industry: A State-of-the-art Analysis

    [https://arxiv.org/abs/2402.09939](https://arxiv.org/abs/2402.09939)

    本研究通过分析提供了建筑行业中生成式AI的最新状态、机遇和挑战。同时，提出了一个帮助建筑公司构建定制化生成式AI解决方案的框架。

    

    建筑行业是全球经济中至关重要的一个部门，但在设计、规划、采购、检查和维护等各个环节中面临着许多生产力挑战。生成式人工智能（AI）可以基于某些输入或先前的知识创造新颖且逼真的数据或内容，如文本、图像、视频或代码，为解决这些挑战提供了创新和颠覆性的解决方案。然而，在关于建筑行业中生成式AI的当前状态、机遇和挑战的文献中存在着空白。本研究旨在通过提供建筑领域生成式AI的最新分析来填补这一空缺，研究目标包括：（1）对建筑行业现有和新兴的生成式AI机遇和挑战进行回顾和分类；（2）提出一个框架，帮助建筑公司利用自己的数据和需求构建定制化的生成式AI解决方案。

    arXiv:2402.09939v1 Announce Type: new  Abstract: The construction industry is a vital sector of the global economy, but it faces many productivity challenges in various processes, such as design, planning, procurement, inspection, and maintenance. Generative artificial intelligence (AI), which can create novel and realistic data or content, such as text, image, video, or code, based on some input or prior knowledge, offers innovative and disruptive solutions to address these challenges. However, there is a gap in the literature on the current state, opportunities, and challenges of generative AI in the construction industry. This study aims to fill this gap by providing a state-of-the-art analysis of generative AI in construction, with three objectives: (1) to review and categorize the existing and emerging generative AI opportunities and challenges in the construction industry; (2) to propose a framework for construction firms to build customized generative AI solutions using their ow
    
[^83]: BUSTER:一份“商业交易实体识别”数据集

    BUSTER: a "BUSiness Transaction Entity Recognition" dataset

    [https://arxiv.org/abs/2402.09916](https://arxiv.org/abs/2402.09916)

    BUSTER是一个商业交易实体识别的数据集，其中包含了3779份手动标注的金融交易文档，并建立了几个基准模型。最佳模型还用于自动标注6196份文档，并作为额外的银标准数据集发布。

    

    尽管自然语言处理在过去几年取得了重大突破，但将这些进展转化为实际的商业案例仍然具有挑战性。其中一个原因在于流行的基准数据与实际数据之间的差异。缺乏监督、类别不平衡、噪声数据和长文档经常影响金融、法律和健康等垂直领域的实际问题。为了支持面向行业的研究，我们提供了一个名为BUSTER的“商业交易实体识别”数据集。该数据集包含了3779份手动标注的金融交易文档。我们建立了几个基准模型，利用了通用和领域特定的语言模型。表现最佳的模型还被用于自动标注6196份文档，我们将其作为额外的银标准数据集发布给BUSTER。

    arXiv:2402.09916v1 Announce Type: new  Abstract: Albeit Natural Language Processing has seen major breakthroughs in the last few years, transferring such advances into real-world business cases can be challenging. One of the reasons resides in the displacement between popular benchmarks and actual data. Lack of supervision, unbalanced classes, noisy data and long documents often affect real problems in vertical domains such as finance, law and health. To support industry-oriented research, we present BUSTER, a BUSiness Transaction Entity Recognition dataset. The dataset consists of 3779 manually annotated documents on financial transactions. We establish several baselines exploiting both general-purpose and domain-specific language models. The best performing model is also used to automatically annotate 6196 documents, which we release as an additional silver corpus to BUSTER.
    
[^84]: 在语言模型训练数据中检测版权内容的方法：DE-COP

    DE-COP: Detecting Copyrighted Content in Language Models Training Data

    [https://arxiv.org/abs/2402.09910](https://arxiv.org/abs/2402.09910)

    DE-COP是一种用于检测语言模型训练数据中版权内容的方法，通过对语言模型进行多项选择探测，可以识别出模型训练文本中可能包含的版权内容。该方法在模型的逻辑可用时比之前的方法提高了9.6%的检测性能，并在完全黑盒模型上实现了72%的准确率。

    

    在考虑到训练数据通常是保密的情况下，我们如何检测语言模型的训练过程中是否使用了版权内容？我们的动机是基于一个语言模型很可能能够识别出其训练文本中的独文摘录。我们提出了一种称为DE-COP的方法，用于确定是否在训练中包含了一段版权内容。DE-COP的核心方法是通过多项选择问题对语言模型进行探测，选择项包括独文本和它们的释义。我们构建了一个基准数据集BookTection，其中包含了在模型训练截止日期之前和之后出版的165本书的摘录以及它们的释义。实验证明，DE-COP在模型的逻辑可用时，检测性能（AUC）超过之前的最佳方法9.6%。此外，DE-COP在完全黑盒模型上检测可疑书籍的平均准确率达到72%，而之前的方法只有$。

    arXiv:2402.09910v1 Announce Type: new  Abstract: How can we detect if copyrighted content was used in the training process of a language model, considering that the training data is typically undisclosed? We are motivated by the premise that a language model is likely to identify verbatim excerpts from its training text. We propose DE-COP, a method to determine whether a piece of copyrighted content was included in training. DE-COP's core approach is to probe an LLM with multiple-choice questions, whose options include both verbatim text and their paraphrases. We construct BookTection, a benchmark with excerpts from 165 books published prior and subsequent to a model's training cutoff, along with their paraphrases. Our experiments show that DE-COP surpasses the prior best method by 9.6% in detection performance (AUC) on models with logits available. Moreover, DE-COP also achieves an average accuracy of 72% for detecting suspect books on fully black-box models where prior methods give $
    
[^85]: 生成表示指令调整

    Generative Representational Instruction Tuning

    [https://arxiv.org/abs/2402.09906](https://arxiv.org/abs/2402.09906)

    本研究引入了生成表示指令调整（GRIT）方法，通过指令区分生成和嵌入任务，训练一个大型语言模型同时处理这两种任务。与其他模型相比，我们的GritLM 7B在文本嵌入基准测试上达到最新的技术水平，并在多种生成任务中表现出色。通过进一步扩大规模，我们的GritLM 8x7B成为最佳的生成语言模型之一，同时仍然是最好的嵌入模型之一。GRIT的统一也大大提高了RAG在长文档上的速度。

    

    所有基于文本的语言问题都可以归结为生成或嵌入。目前的模型只能在其中一种任务上表现良好。我们介绍了生成表示指令调整（GRIT）方法，通过指令来区分生成和嵌入任务，从而训练一个大型语言模型同时处理这两种任务。与其他开放模型相比，我们的GritLM 7B在大规模文本嵌入基准测试（MTEB）上取得了最新的技术水平，并在多种生成任务中超过了同等规模的所有模型。通过进一步扩大规模，GritLM 8x7B在尝试的所有开放生成语言模型中表现最佳，同时仍然是最好的嵌入模型之一。值得注意的是，我们发现GRIT可以与仅在生成或嵌入数据上训练的模型相媲美，因此我们可以在不损失性能的情况下统一两者。除此之外，通过GRIT的统一可以将RAG（检索增强生成）在长文档上的速度提高60%以上。

    arXiv:2402.09906v1 Announce Type: cross  Abstract: All text-based language problems can be reduced to either generation or embedding. Current models only perform well at one or the other. We introduce generative representational instruction tuning (GRIT) whereby a large language model is trained to handle both generative and embedding tasks by distinguishing between them through instructions. Compared to other open models, our resulting GritLM 7B sets a new state of the art on the Massive Text Embedding Benchmark (MTEB) and outperforms all models up to its size on a range of generative tasks. By scaling up further, GritLM 8x7B outperforms all open generative language models that we tried while still being among the best embedding models. Notably, we find that GRIT matches training on only generative or embedding data, thus we can unify both at no performance loss. Among other benefits, the unification via GRIT speeds up Retrieval-Augmented Generation (RAG) by > 60% for long documents, 
    
[^86]: 重新审视带有内存单子的循环强化学习

    Revisiting Recurrent Reinforcement Learning with Memory Monoids

    [https://arxiv.org/abs/2402.09900](https://arxiv.org/abs/2402.09900)

    这篇论文重新审视了使用内存单子的循环强化学习方法。通过定义新颖的内存单子框架并提出一种新的批处理方法，改进了样本效率、增加了回报并简化了实现过程。

    

    在强化学习中，像RNN和transformers这样的记忆模型通过将轨迹映射到潜在的马尔可夫状态来处理部分可观察的马尔可夫决策过程（POMDPs）。这些模型对于长序列的规模化处理能力并不特别好，尤其是与一类新兴的记忆模型（有时称为线性循环模型）相比。我们发现这些模型的循环更新是一个单子，因此我们正式定义了一个新颖的内存单子框架。我们重新审视了循环强化学习中的传统批处理方法，突出了理论和实证上的不足之处。利用内存单子的特性，我们提出了一种新的批处理方法，改进了样本效率，增加了回报，并简化了循环丢失函数在强化学习中的实施。

    arXiv:2402.09900v1 Announce Type: cross  Abstract: In RL, memory models such as RNNs and transformers address Partially Observable Markov Decision Processes (POMDPs) by mapping trajectories to latent Markov states. Neither model scales particularly well to long sequences, especially compared to an emerging class of memory models sometimes called linear recurrent models. We discover that the recurrent update of these models is a monoid, leading us to formally define a novel memory monoid framework. We revisit the traditional approach to batching in recurrent RL, highlighting both theoretical and empirical deficiencies. Leveraging the properties of memory monoids, we propose a new batching method that improves sample efficiency, increases the return, and simplifies the implementation of recurrent loss functions in RL.
    
[^87]: COVIDHealth：一个用于分类COVID-19讨论的基准Twitter数据集和基于机器学习的Web应用程序

    COVIDHealth: A Benchmark Twitter Dataset and Machine Learning based Web Application for Classifying COVID-19 Discussions

    [https://arxiv.org/abs/2402.09897](https://arxiv.org/abs/2402.09897)

    本研究开发了一个基于机器学习的Web应用程序，用于自动分类社交媒体上的COVID-19相关讨论。通过Twitter数据集的标记和多种特征提取方法的应用，实现了关于COVID-19的健康风险、预防、症状、传播和治疗等方面的分类。

    

    COVID-19疫情对身体和心理健康产生了不利影响。在这次疫情中，许多研究致力于从社交媒体中获取与健康相关的观点。本研究的主要目标是开发一个基于机器学习的Web应用程序，用于自动分类社交媒体上的COVID-19相关讨论。为了实现这一目标，我们使用Twitter API收集了数据，并对共6667条推文进行了五个不同类别的标记：健康风险、预防、症状、传播和治疗。我们使用多种特征提取方法提取特征，并将其应用于包括决策树、随机森林、随机梯度下降、Adaboost、K-最近邻居、逻辑回归和线性支持向量机在内的七种传统机器学习算法进行分类。

    arXiv:2402.09897v1 Announce Type: new  Abstract: The COVID-19 pandemic has had adverse effects on both physical and mental health. During this pandemic, numerous studies have focused on gaining insights into health-related perspectives from social media. In this study, our primary objective is to develop a machine learning-based web application for automatically classifying COVID-19-related discussions on social media. To achieve this, we label COVID-19-related Twitter data, provide benchmark classification results, and develop a web application. We collected data using the Twitter API and labeled a total of 6,667 tweets into five different classes: health risks, prevention, symptoms, transmission, and treatment. We extracted features using various feature extraction methods and applied them to seven different traditional machine learning algorithms, including Decision Tree, Random Forest, Stochastic Gradient Descent, Adaboost, K-Nearest Neighbour, Logistic Regression, and Linear SVC. 
    
[^88]: 预测因果特征不能更好地推广到新领域

    Predictors from causal features do not generalize better to new domains

    [https://arxiv.org/abs/2402.09891](https://arxiv.org/abs/2402.09891)

    因果特征不能更好地推广到新领域，预测器使用所有特征的效果更好。

    

    我们研究了在不同领域中，基于因果特征训练的机器学习模型的泛化效果。我们考虑了涵盖健康、就业、教育、社会福利和政治等应用的16个表格数据集的预测任务。每个数据集都有多个领域，我们可以测试一个在一个领域训练的模型在另一个领域的表现。对于每个预测任务，我们选择对预测目标有因果影响的特征。我们的目标是测试基于因果特征训练的模型是否在不同领域中更好地泛化。我们发现，无论是否具有因果关系，使用所有可用特征的预测器都比使用因果特征的预测器在领域内外的准确性更高。而且，即使是从一个领域到另一个领域的准确性绝对下降对于因果预测器来说也不比使用所有特征的模型更好。如果目标是在新领域中泛化，实践中使用所有特征的预测器效果更好。

    arXiv:2402.09891v1 Announce Type: new  Abstract: We study how well machine learning models trained on causal features generalize across domains. We consider 16 prediction tasks on tabular datasets covering applications in health, employment, education, social benefits, and politics. Each dataset comes with multiple domains, allowing us to test how well a model trained in one domain performs in another. For each prediction task, we select features that have a causal influence on the target of prediction. Our goal is to test the hypothesis that models trained on causal features generalize better across domains. Without exception, we find that predictors using all available features, regardless of causality, have better in-domain and out-of-domain accuracy than predictors using causal features. Moreover, even the absolute drop in accuracy from one domain to the other is no better for causal predictors than for models that use all features. If the goal is to generalize to new domains, prac
    
[^89]: 通过决策树解释核聚类

    Explaining Kernel Clustering via Decision Trees

    [https://arxiv.org/abs/2402.09881](https://arxiv.org/abs/2402.09881)

    这项工作探讨了可解释的核聚类方法，提出了使用决策树近似核k-means聚类分区的算法，并通过合适的特征选择实现了解释性和近似保证的平衡。

    

    尽管可解释和可解释的机器学习越来越受欢迎，但关于固有可解释聚类方法的工作仍然非常有限。最近，解释经典k-means算法的兴趣激增，导致了使用轴对齐决策树近似k-means聚类的高效算法。然而，可解释的k-means变种在实践中的适用性有限，通常需要更灵活的聚类方法才能获得有用的数据分区。在本研究中，我们研究了可解释的核聚类，并提出了构建决策树来近似kernel k-means引导分区的算法，kernel k-means是k-means的非线性扩展。我们进一步借鉴了关于可解释k-means的先前工作，并展示了如何通过合适的特征选择在不损失解释能力的情况下保持近似保证。

    arXiv:2402.09881v1 Announce Type: new  Abstract: Despite the growing popularity of explainable and interpretable machine learning, there is still surprisingly limited work on inherently interpretable clustering methods. Recently, there has been a surge of interest in explaining the classic k-means algorithm, leading to efficient algorithms that approximate k-means clusters using axis-aligned decision trees. However, interpretable variants of k-means have limited applicability in practice, where more flexible clustering methods are often needed to obtain useful partitions of the data. In this work, we investigate interpretable kernel clustering, and propose algorithms that construct decision trees to approximate the partitions induced by kernel k-means, a nonlinear extension of k-means. We further build on previous work on explainable k-means and demonstrate how a suitable choice of features allows preserving interpretability without sacrificing approximation guarantees on the interpret
    
[^90]: 在嵌入式多核平台上表征 EEG 应用的准确性权衡

    Characterizing Accuracy Trade-offs of EEG Applications on Embedded HMPs

    [https://arxiv.org/abs/2402.09867](https://arxiv.org/abs/2402.09867)

    该论文研究了在嵌入式多核平台上，采用电池供电的可穿戴设备分析脑电图（EEG）记录的应用。研究发现，通过调整近似方法，可以在有限的能量预算内实现更好的性能和能量收益。

    

    使用电池供电的可穿戴设备分析脑电图（EEG）记录，以监测脑活动和神经系统疾病。这些应用需要长时间连续处理以生成可行的结果。然而，可穿戴设备由于实际使用案例中的小尺寸而受限于有限的能量和计算资源。在限制的能量预算内，嵌入式异构多核平台（HMPs）可以提供更好的性能。可以进一步利用 EEG 应用程序流程的错误韧性来最大化 HMPs 的性能和能量收益。然而，在嵌入式 HMPs 上规范调整近似需要对准确性-性能-功耗权衡空间进行彻底探索。在这项工作中，我们对三种 EEG 应用（包括癫痫发作检测、睡眠阶段分类和压力检测）的错误韧性进行了表征。

    arXiv:2402.09867v1 Announce Type: cross  Abstract: Electroencephalography (EEG) recordings are analyzed using battery-powered wearable devices to monitor brain activities and neurological disorders. These applications require long and continuous processing to generate feasible results. However, wearable devices are constrained with limited energy and computation resources, owing to their small sizes for practical use cases. Embedded heterogeneous multi-core platforms (HMPs) can provide better performance within limited energy budgets for EEG applications. Error resilience of the EEG application pipeline can be exploited further to maximize the performance and energy gains with HMPs. However, disciplined tuning of approximation on embedded HMPs requires a thorough exploration of the accuracy-performance-power trade-off space. In this work, we characterize the error resilience of three EEG applications, including Epileptic Seizure Detection, Sleep Stage Classification, and Stress Detecti
    
[^91]: 对于基准线和基准测试近似高斯过程的建议

    Recommendations for Baselines and Benchmarking Approximate Gaussian Processes

    [https://arxiv.org/abs/2402.09849](https://arxiv.org/abs/2402.09849)

    对于基准线和基准测试近似高斯过程的研究，我们提出了对比方法的建议，并开发了一种训练程序，该程序不需要用户选择，并且证明这是一个符合要求的强大基准。

    

    Gaussian processes (GPs)是机器学习工具箱中成熟且广泛使用的组件。它们具有自动超参数选择的优点，可以实现无需用户干预的训练。然而，在许多现实情况下，通常需要使用近似方法，而这些方法通常需要调整。我们认为，这种调整要求使得评估变得复杂，这导致缺乏对在哪种情况下使用哪种方法的明确建议。为了解决这个问题，我们提出了对比GP近似方法的建议，基于用户对方法的期望的规范。此外，我们开发了一种训练程序，用于Titsias [2009]的变分方法，该方法不需要用户选择，并且证明这是符合我们规范的一个强大基准。我们得出结论，按照我们的建议进行基准测试可以更清晰地了解当前领域的状态，并发现……

    arXiv:2402.09849v1 Announce Type: new  Abstract: Gaussian processes (GPs) are a mature and widely-used component of the ML toolbox. One of their desirable qualities is automatic hyperparameter selection, which allows for training without user intervention. However, in many realistic settings, approximations are typically needed, which typically do require tuning. We argue that this requirement for tuning complicates evaluation, which has led to a lack of a clear recommendations on which method should be used in which situation. To address this, we make recommendations for comparing GP approximations based on a specification of what a user should expect from a method. In addition, we develop a training procedure for the variational method of Titsias [2009] that leaves no choices to the user, and show that this is a strong baseline that meets our specification. We conclude that benchmarking according to our suggestions gives a clearer view of the current state of the field, and uncovers 
    
[^92]: 基于深度学习的雷达降水估计方法

    A Deep Learning Approach to Radar-based QPE

    [https://arxiv.org/abs/2402.09846](https://arxiv.org/abs/2402.09846)

    本研究提出了一种基于深度学习的雷达降水估计方法，利用台湾地区的雷达数据进行量化降水估计，并与具体降水位置相关联。与传统的基于Z-R关系的方法不同，该方法利用机器学习算法自动检测天气系统的演变和移动，并将其与特定地形属性的位置相结合。评估结果显示该方法在台北地区的降水估计上具有良好的效果。

    

    本研究提出了一种基于量化降水估计和使用多个传感器的定量降水估计和分离（QPESUMS）马赛克雷达数据集的体积到点的框架。利用台湾地区的格网化雷达回波时间序列构建了一个机器学习算法模型，用于天气站的降水估计。该模型从输入数据体积中提取空间和时间特征，然后将这些特征与具体位置的降水关联起来。与基于Z-R关系的降水估计方法不同，我们利用机器学习算法自动检测天气系统的演变和移动，并将这些模式与具有特定地形属性的位置关联起来。具体来说，我们使用2013年台北地区45个天气站的小时降水数据对该框架进行了评估。

    arXiv:2402.09846v1 Announce Type: cross  Abstract: In this study, we propose a volume-to-point framework for quantitative precipitation estimation (QPE) based on the Quantitative Precipitation Estimation and Segregation Using Multiple Sensor (QPESUMS) Mosaic Radar data set. With a data volume consisting of the time series of gridded radar reflectivities over the Taiwan area, we used machine learning algorithms to establish a statistical model for QPE in weather stations. The model extracts spatial and temporal features from the input data volume and then associates these features with the location-specific precipitations. In contrast to QPE methods based on the Z-R relation, we leverage the machine learning algorithms to automatically detect the evolution and movement of weather systems and associate these patterns to a location with specific topographic attributes. Specifically, we evaluated this framework with the hourly precipitation data of 45 weather stations in Taipei during 2013
    
[^93]: LAPDoc：面向文档的布局感知提示

    LAPDoc: Layout-Aware Prompting for Documents

    [https://arxiv.org/abs/2402.09841](https://arxiv.org/abs/2402.09841)

    本文研究了通过使用布局增强来使用纯文本LLM进行文档特定任务的可能性。

    

    最近，使用大量纯文本数据训练大型语言模型(LLM)取得了重大突破，在许多领域和任务中实现了强大的泛化能力，包括文档特定任务。相比之下，训练针对文档理解的多模态变压器体系结构，专门设计用于将文本输入与相应的文档布局融合。这需要单独的微调步骤，需要额外的训练数据。目前，尚没有具有与LLM相当泛化能力的文档变压器可用。这引发了一个问题，即在文档理解任务中应该选择哪种类型的模型。在本文中，我们通过使用布局增强来调查使用纯文本LLM用于文档特定任务的可能性。我们探索了添加修改和基于规则的方法，以在纯文本LLM提示中添加布局信息。

    arXiv:2402.09841v1 Announce Type: new  Abstract: Recent advances in training large language models (LLMs) using massive amounts of solely textual data lead to strong generalization across many domains and tasks, including document-specific tasks. Opposed to that there is a trend to train multi-modal transformer architectures tailored for document understanding that are designed specifically to fuse textual inputs with the corresponding document layout. This involves a separate fine-tuning step for which additional training data is required. At present, no document transformers with comparable generalization to LLMs are available That raises the question which type of model is to be preferred for document understanding tasks. In this paper we investigate the possibility to use purely text-based LLMs for document-specific tasks by using layout enrichment. We explore drop-in modifications and rule-based methods to enrich purely textual LLM prompts with layout information. In our experimen
    
[^94]: 渐变环境中的表演性强化学习

    Performative Reinforcement Learning in Gradually Shifting Environments

    [https://arxiv.org/abs/2402.09838](https://arxiv.org/abs/2402.09838)

    这项研究提出了一种在渐变环境中进行强化学习的框架，可以模拟部署策略对环境的影响，并提出了一种新的算法MDRR来应对这种情况。

    

    当强化学习（RL）代理在实践中部署时，它们可能会影响环境并改变其动态。当前的研究试图形式化建模这种现象，并在这些模型中分析学习算法。为此，我们提出了一个框架，其中当前的环境取决于部署策略及其先前的动态。这是Performative RL（PRL）[Mandal et al., 2023]的一种泛化。与PRL不同，我们的框架允许对环境逐渐调整到部署策略的情景进行建模。我们将表演性预测文献中的两种算法适应到我们的设置，并提出了一种新的算法称为混合延迟重复训练（MDRR）。我们给出了这些算法收敛的条件，并使用三个指标进行比较：重训练次数，逼近保证和每次部署的样本数。与之前的方法不同，MDRR结合了样本

    arXiv:2402.09838v1 Announce Type: new  Abstract: When Reinforcement Learning (RL) agents are deployed in practice, they might impact their environment and change its dynamics. Ongoing research attempts to formally model this phenomenon and to analyze learning algorithms in these models. To this end, we propose a framework where the current environment depends on the deployed policy as well as its previous dynamics. This is a generalization of Performative RL (PRL) [Mandal et al., 2023]. Unlike PRL, our framework allows to model scenarios where the environment gradually adjusts to a deployed policy. We adapt two algorithms from the performative prediction literature to our setting and propose a novel algorithm called Mixed Delayed Repeated Retraining (MDRR). We provide conditions under which these algorithms converge and compare them using three metrics: number of retrainings, approximation guarantee, and number of samples per deployment. Unlike previous approaches, MDRR combines sample
    
[^95]: 一体化与多功能性：一种简单而有效的跨领域图预训练方法

    All in One and One for All: A Simple yet Effective Method towards Cross-domain Graph Pretraining

    [https://arxiv.org/abs/2402.09834](https://arxiv.org/abs/2402.09834)

    本研究提出了一种简单而有效的跨领域图预训练方法，通过一体化和多功能性，使得大型语言模型在各个领域具备了超强的泛化能力。

    

    大型语言模型（LLMs）已经在计算机视觉（CV）和自然语言处理（NLP）领域取得了重大突破。LLMs最显著的进展之一是，在广泛且多样化的数据集上训练了单一模型，这些数据跨越多个领域，这种范式被称为“一体化”。这种方法使LLMs具备了超强的泛化能力，有助于理解各种数据分布。借助这些能力，单一的LLM在各种领域展现出了出色的多功能性，这种范式被称为“多功能一体化”。然而，将这个想法应用于图领域仍然面临着巨大的挑战，跨领域预训练经常导致负迁移。这个问题在少样本学习场景中尤为重要，因为训练数据的匮乏需要引入外部知识源。为了应对这个挑战，我们提出了一种新颖的方法。

    arXiv:2402.09834v1 Announce Type: new  Abstract: Large Language Models (LLMs) have revolutionized the fields of computer vision (CV) and natural language processing (NLP). One of the most notable advancements of LLMs is that a single model is trained on vast and diverse datasets spanning multiple domains -- a paradigm we term `All in One'. This methodology empowers LLMs with super generalization capabilities, facilitating an encompassing comprehension of varied data distributions. Leveraging these capabilities, a single LLM demonstrates remarkable versatility across a variety of domains -- a paradigm we term `One for All'. However, applying this idea to the graph field remains a formidable challenge, with cross-domain pretraining often resulting in negative transfer. This issue is particularly important in few-shot learning scenarios, where the paucity of training data necessitates the incorporation of external knowledge sources. In response to this challenge, we propose a novel approa
    
[^96]: 利用GAN进行欺诈检测：使用合成交易数据进行模型训练

    Utilizing GANs for Fraud Detection: Model Training with Synthetic Transaction Data

    [https://arxiv.org/abs/2402.09830](https://arxiv.org/abs/2402.09830)

    本论文研究了利用生成对抗网络（GAN）进行欺诈检测的应用，比较了其与传统方法的优势。通过构建对抗性验证图的集合，有效防止了由机器人或自动系统引起的欺诈，并确保交易中的用户是真实的。

    

    异常检测是各个研究领域中的一个重要挑战，旨在识别偏离正常数据分布的实例。本文探讨了在欺诈检测中应用生成对抗网络（GAN）并将其与传统方法进行了比较的优势。GAN是一种人工神经网络（ANN）的类型，在建模复杂数据分布方面表现出了希望，使其成为异常检测的有效工具。本文系统地描述了GAN及其衍生模型的原则，并强调了它们在不同数据集上的欺诈检测应用。通过构建对抗性验证图的集合，我们将有效防止由机器人或自动系统引起的欺诈，并确保交易中的用户是真实的。

    arXiv:2402.09830v1 Announce Type: cross  Abstract: Anomaly detection is a critical challenge across various research domains, aiming to identify instances that deviate from normal data distributions. This paper explores the application of Generative Adversarial Networks (GANs) in fraud detection, comparing their advantages with traditional methods. GANs, a type of Artificial Neural Network (ANN), have shown promise in modeling complex data distributions, making them effective tools for anomaly detection. The paper systematically describes the principles of GANs and their derivative models, emphasizing their application in fraud detection across different datasets. And by building a collection of adversarial verification graphs, we will effectively prevent fraud caused by bots or automated systems and ensure that the users in the transaction are real. The objective of the experiment is to design and implement a fake face verification code and fraud detection system based on Generative A
    
[^97]: 音频恢复的扩散模型

    Diffusion Models for Audio Restoration

    [https://arxiv.org/abs/2402.09821](https://arxiv.org/abs/2402.09821)

    本文介绍了基于扩散模型的音频恢复算法，重点关注语音增强和音乐恢复任务。

    

    随着音频播放设备和快速数据传输的发展，对高音质的需求在娱乐和通信领域不断增长。然而，由于录制过程中的失真和干扰，或者由于不完善的传输管道，音频质量面临许多挑战。为了解决这个问题，音频恢复方法旨在从损坏的输入数据中恢复出清晰的音频信号。本文介绍了基于扩散模型的音频恢复算法，重点关注语音增强和音乐恢复任务。传统方法通常基于手工规则和统计启发法，从而建立了我们对音频信号的认识。近几十年来，越来越多的人转向利用深度神经网络（DNNs）的建模能力的数据驱动方法。深度生成模型中的扩散模型成为一种新兴方法。

    arXiv:2402.09821v1 Announce Type: cross  Abstract: With the development of audio playback devices and fast data transmission, the demand for high sound quality is rising, for both entertainment and communications. In this quest for better sound quality, challenges emerge from distortions and interferences originating at the recording side or caused by an imperfect transmission pipeline. To address this problem, audio restoration methods aim to recover clean sound signals from the corrupted input data. We present here audio restoration algorithms based on diffusion models, with a focus on speech enhancement and music restoration tasks. Traditional approaches, often grounded in handcrafted rules and statistical heuristics, have shaped our understanding of audio signals. In the past decades, there has been a notable shift towards data-driven methods that exploit the modeling capabilities of deep neural networks (DNNs). Deep generative models, and among them diffusion models, have emerged 
    
[^98]: 用深度学习增强金融行业的网络安全韧性，实现高级威胁检测

    Enhancing Cybersecurity Resilience in Finance with Deep Learning for Advanced Threat Detection

    [https://arxiv.org/abs/2402.09820](https://arxiv.org/abs/2402.09820)

    这项研究提出使用深度学习来增强金融行业的网络安全韧性，并实现高级威胁检测。目前的网络威胁检测方法往往基于规则和传统的机器学习方法，无法适用大规模数据应用，并且无法有效检测未知威胁。

    

    在互联网时代，人们的生活越来越依赖于今天的网络技术。然而，网络技术是一把双刃剑，给人们带来便利的同时也带来了许多安全挑战。保持网络安全和保护用户的合法利益是网络建设的核心。威胁检测是一个完整有效的防御系统的重要组成部分。在网络信息安全领域，网络攻击和网络防护的技术更新日益迅猛。如何有效地检测未知威胁是网络防护的关注焦点之一。目前，网络威胁检测通常基于规则和传统的机器学习方法，这些方法创建人工规则或提取常见的时空特征，不能应用于大规模数据应用，并且未知威胁的出现导致了系统的检测准确性降低。

    arXiv:2402.09820v1 Announce Type: cross  Abstract: In the age of the Internet, people's lives are increasingly dependent on today's network technology. However, network technology is a double-edged sword, bringing convenience to people but also posing many security challenges. Maintaining network security and protecting the legitimate interests of users is at the heart of network construction. Threat detection is an important part of a complete and effective defense system. In the field of network information security, the technical update of network attack and network protection is spiraling. How to effectively detect unknown threats is one of the concerns of network protection. Currently, network threat detection is usually based on rules and traditional machine learning methods, which create artificial rules or extract common spatiotemporal features, which cannot be applied to large-scale data applications, and the emergence of unknown threats causes the detection accuracy of the or
    
[^99]: 解决非凸强凹最小最大问题的两种信赖域类算法

    Two trust region type algorithms for solving nonconvex-strongly concave minimax problems

    [https://arxiv.org/abs/2402.09807](https://arxiv.org/abs/2402.09807)

    本文提出了两种信赖域类算法，用于解决非凸强凹最小最大问题，并可以在迭代次数为$\mathcal{O}(\epsilon^{-1.5})$内找到二阶稳定点。

    

    在本文中，我们提出了解决非凸强凹最小最大问题的最小最大信赖域（MINIMAX-TR）算法和具有收缩和扩张的最小最大信赖域算法（MINIMAX-TRACE）。这两种算法可以在$\mathcal{O}(\epsilon^{-1.5})$次迭代内找到$(\epsilon, \sqrt{\epsilon})$-二阶稳定点(SSP)，这与已知最好的迭代复杂度相匹配。

    arXiv:2402.09807v1 Announce Type: cross  Abstract: In this paper, we propose a Minimax Trust Region (MINIMAX-TR) algorithm and a Minimax Trust Region Algorithm with Contractions and Expansions(MINIMAX-TRACE) algorithm for solving nonconvex-strongly concave minimax problems. Both algorithms can find an $(\epsilon, \sqrt{\epsilon})$-second order stationary point(SSP) within $\mathcal{O}(\epsilon^{-1.5})$ iterations, which matches the best well known iteration complexity.
    
[^100]: 准则崩溃和损失分布控制

    Criterion collapse and loss distribution control

    [https://arxiv.org/abs/2402.09802](https://arxiv.org/abs/2402.09802)

    该论文研究了"准则崩溃"的概念，即优化一个度量指标意味着另一个度量指标的最优性。研究结果发现，对于损失的伯努利分布，CVaR和DRO的结果远超出现有研究，同时发现了一些特定条件下，单调准则如倾斜ERM无法避免崩溃，而非单调的替代方案可以。

    

    在这项工作中，我们考虑了"准则崩溃"的概念，即优化一个度量指标意味着另一个度量指标的最优性，特别关注各种学习准则下崩溃成误差概率最小化器的条件，从DRO和OCE风险（CVaR、倾斜ERM）到文献中探索的最新上升-下降算法的非单调准则（洪水、SoftAD）。我们展示了在伯努利分布损失的背景下，CVaR和DRO的现有结果远远超越了崩溃的范围，然后扩大了我们的范围，包括代理损失，展示了像倾斜ERM这样的单调准则无法避免崩溃的条件，而非单调的替代方案可以。

    arXiv:2402.09802v1 Announce Type: cross  Abstract: In this work, we consider the notion of "criterion collapse," in which optimization of one metric implies optimality in another, with a particular focus on conditions for collapse into error probability minimizers under a wide variety of learning criteria, ranging from DRO and OCE risks (CVaR, tilted ERM) to non-monotonic criteria underlying recent ascent-descent algorithms explored in the literature (Flooding, SoftAD). We show how collapse in the context of losses with a Bernoulli distribution goes far beyond existing results for CVaR and DRO, then expand our scope to include surrogate losses, showing conditions where monotonic criteria such as tilted ERM cannot avoid collapse, whereas non-monotonic alternatives can.
    
[^101]: 闭式滤波器在非线性系统中的应用

    Closed-form Filtering for Non-linear Systems

    [https://arxiv.org/abs/2402.09796](https://arxiv.org/abs/2402.09796)

    提出了一种基于高斯PSD模型的新型滤波器，可以在转换和观测都是高斯PSD模型时以闭式形式高效地进行滤波，并且提出的估计器具有强大的理论保证，适应转换概率的正则性。

    

    顺序贝叶斯滤波旨在估计隐藏马尔可夫模型的当前状态分布，给定过去的观测值。对于大多数应用领域来说，这个问题是难以解决的，除了像表格设置或具有高斯噪声的线性动力系统这样的明显情况。在这项工作中，我们提出了一种基于高斯PSD模型的新型滤波器，它在密度近似和计算效率方面具有多个优势。我们展示了当转换和观测都是高斯PSD模型时，滤波可以以闭式形式高效地进行。当转换和观测被高斯PSD模型近似时，我们证明了我们提出的估计器具有强大的理论保证，估计误差取决于近似的质量，并且适应转换概率的正则性。特别是，我们确定了我们的方法在某些情况下的适用范围，其中我们可以以闭式形式高效地进行滤波。

    arXiv:2402.09796v1 Announce Type: cross  Abstract: Sequential Bayesian Filtering aims to estimate the current state distribution of a Hidden Markov Model, given the past observations. The problem is well-known to be intractable for most application domains, except in notable cases such as the tabular setting or for linear dynamical systems with gaussian noise. In this work, we propose a new class of filters based on Gaussian PSD Models, which offer several advantages in terms of density approximation and computational efficiency. We show that filtering can be efficiently performed in closed form when transitions and observations are Gaussian PSD Models. When the transition and observations are approximated by Gaussian PSD Models, we show that our proposed estimator enjoys strong theoretical guarantees, with estimation error that depends on the quality of the approximation and is adaptive to the regularity of the transition probabilities. In particular, we identify regimes in which our 
    
[^102]: 检查生成对抗网络判别器中的病态偏见：以StyleGAN3模型为例的案例研究

    Examining Pathological Bias in a Generative Adversarial Network Discriminator: A Case Study on a StyleGAN3 Model

    [https://arxiv.org/abs/2402.09786](https://arxiv.org/abs/2402.09786)

    这项研究发现了StyleGAN3模型中判别器的病态偏见，它在图像和面部质量上的得分分层影响了不同性别、种族和其他类别的图像。

    

    生成对抗网络可以生成逼真的人脸，往往难以被人类区分出来。我们发现预训练的StyleGAN3模型中的判别器在图像和面部质量上系统地对得分进行分层，并且这不成比例地影响了不同性别、种族和其他类别的图像。我们检查了判别器在色彩和亮度方面对感知的种族和性别的偏见，然后检查了社会心理学中关于刻板印象研究中常见的偏见。

    arXiv:2402.09786v1 Announce Type: cross  Abstract: Generative adversarial networks generate photorealistic faces that are often indistinguishable by humans from real faces. We find that the discriminator in the pre-trained StyleGAN3 model, a popular GAN network, systematically stratifies scores by both image- and face-level qualities and that this disproportionately affects images across gender, race, and other categories. We examine the discriminator's bias for color and luminance across axes perceived race and gender; we then examine axes common in research on stereotyping in social psychology.
    
[^103]: MC-DBN：基于深度信念网络的模态补全模型

    MC-DBN: A Deep Belief Network-Based Model for Modality Completion

    [https://arxiv.org/abs/2402.09782](https://arxiv.org/abs/2402.09782)

    MC-DBN是一种基于深度信念网络的模态补全模型，利用完整数据的隐式特征来弥补附加不完整数据的差距，提高预测准确性。

    

    最近多模态人工智能（AI）的进展已经彻底改变了股市预测和心率监测领域。利用多样的数据源可以大大提高预测准确性。然而，额外的数据可能不总是与原始数据集相吻合。插值方法通常用于处理模态数据中的缺失值，但在稀疏信息情况下可能存在一些限制。为解决这一挑战，我们提出了一种模态补全的深度信念网络模型（MC-DBN）。该方法利用完整数据的隐式特征来弥补自身与附加不完整数据之间的差距。它确保增强的多模态数据与现实世界的动态特性密切相符，以提高模型的有效性。我们在两个来自股市预测和心率监测的数据集上对MC-DBN模型进行了评估。

    arXiv:2402.09782v1 Announce Type: cross  Abstract: Recent advancements in multi-modal artificial intelligence (AI) have revolutionized the fields of stock market forecasting and heart rate monitoring. Utilizing diverse data sources can substantially improve prediction accuracy. Nonetheless, additional data may not always align with the original dataset. Interpolation methods are commonly utilized for handling missing values in modal data, though they may exhibit limitations in the context of sparse information. Addressing this challenge, we propose a Modality Completion Deep Belief Network-Based Model (MC-DBN). This approach utilizes implicit features of complete data to compensate for gaps between itself and additional incomplete data. It ensures that the enhanced multi-modal data closely aligns with the dynamic nature of the real world to enhance the effectiveness of the model. We conduct evaluations of the MC-DBN model in two datasets from the stock market forecasting and heart rate
    
[^104]: TinyCL:一种用于自主系统持续学习的高效硬件架构

    TinyCL: An Efficient Hardware Architecture for Continual Learning on Autonomous Systems

    [https://arxiv.org/abs/2402.09780](https://arxiv.org/abs/2402.09780)

    TinyCL是一种用于自主系统持续学习的高效硬件架构，在CL中支持前向和反向传播，并通过滑动窗口的连续学习策略来减少内存访问。

    

    持续学习（CL）范式包括不断演化深度神经网络（DNN）模型的参数，以逐步学习执行新任务，而不降低先前任务的性能，即避免所谓的灾难性遗忘。然而，在基于CL的自主系统中，DNN参数更新对资源要求极高。现有的DNN加速器不能直接用于CL，因为它们只支持前向传播的执行。只有少数先前的架构执行反向传播和权重更新，但它们缺乏对CL的控制和管理。为此，我们设计了一个硬件架构TinyCL，用于在资源受限的自主系统上进行持续学习。它包括一个执行前向和反向传播的处理单元，以及一个管理基于内存的CL工作负载的控制单元。为了最小化内存访问，我们使用了滑动窗口的连续学习策略。

    arXiv:2402.09780v1 Announce Type: new  Abstract: The Continuous Learning (CL) paradigm consists of continuously evolving the parameters of the Deep Neural Network (DNN) model to progressively learn to perform new tasks without reducing the performance on previous tasks, i.e., avoiding the so-called catastrophic forgetting. However, the DNN parameter update in CL-based autonomous systems is extremely resource-hungry. The existing DNN accelerators cannot be directly employed in CL because they only support the execution of the forward propagation. Only a few prior architectures execute the backpropagation and weight update, but they lack the control and management for CL. Towards this, we design a hardware architecture, TinyCL, to perform CL on resource-constrained autonomous systems. It consists of a processing unit that executes both forward and backward propagation, and a control unit that manages memory-based CL workload. To minimize the memory accesses, the sliding window of the con
    
[^105]: 从变动性到稳定性：推荐系统基准化实践的进展

    From Variability to Stability: Advancing RecSys Benchmarking Practices

    [https://arxiv.org/abs/2402.09766](https://arxiv.org/abs/2402.09766)

    本论文提出了一种新的基准测试方法，通过使用多样化的开放数据集，并在多个度量指标上评估多种协同过滤算法，来研究数据集特征对算法性能的影响。这一方法填补了推荐系统算法比较中的不足之处，推进了评估实践。

    

    在快速发展的推荐系统领域中，新的算法经常通过对一组有限的任意选择的数据集进行评估来声称自己具有最先进的性能。然而，由于数据集特征对算法性能有重大影响，这种方法可能无法全面反映它们的有效性。为了解决这个问题，本文引入了一种新的基准测试方法，以促进公平和稳健的推荐系统算法比较，从而推进评估实践。通过利用包括本文介绍的两个数据集在内的30个开放数据集，并在9个度量指标上评估11种协同过滤算法，我们对数据集特征对算法性能的影响进行了重要的研究。我们进一步研究了将多个数据集的结果聚合成一个统一排名的可行性。通过严格的实验分析，我们发现......

    arXiv:2402.09766v1 Announce Type: cross  Abstract: In the rapidly evolving domain of Recommender Systems (RecSys), new algorithms frequently claim state-of-the-art performance based on evaluations over a limited set of arbitrarily selected datasets. However, this approach may fail to holistically reflect their effectiveness due to the significant impact of dataset characteristics on algorithm performance. Addressing this deficiency, this paper introduces a novel benchmarking methodology to facilitate a fair and robust comparison of RecSys algorithms, thereby advancing evaluation practices. By utilizing a diverse set of $30$ open datasets, including two introduced in this work, and evaluating $11$ collaborative filtering algorithms across $9$ metrics, we critically examine the influence of dataset characteristics on algorithm performance. We further investigate the feasibility of aggregating outcomes from multiple datasets into a unified ranking. Through rigorous experimental analysis, 
    
[^106]: 一种利用惯性传感器进行步态用户人口统计估计的框架

    A Framework For Gait-Based User Demography Estimation Using Inertial Sensors

    [https://arxiv.org/abs/2402.09761](https://arxiv.org/abs/2402.09761)

    这项研究提出了一种利用深度学习和层次相关传播（LRP）的框架，用于识别人类步态模式并估计用户的人口统计信息，如年龄和性别。

    

    人类步态已被证明为各种应用提供了关键的动作线索。识别人类步态的模式在安全、虚拟现实游戏、医学康复和疾病识别等各个应用领域得到了广泛采用。此外，可穿戴惯性传感器不仅广泛用于记录步态，还用于预测用户的人口统计信息。深度学习等机器学习技术与惯性传感器信号相结合，在识别人类步态模式和估计用户人口统计方面取得了有希望的结果。然而，这种深度学习模型的黑盒特性阻碍了研究人员揭示模型预测背后的原因。因此，我们提出利用深度学习和层次相关传播（LRP）来识别在识别用户的人口统计信息（如年龄和性别）方面起着重要作用的重要变量。

    arXiv:2402.09761v1 Announce Type: cross  Abstract: Human gait has been shown to provide crucial motion cues for various applications. Recognizing patterns in human gait has been widely adopted in various application areas such as security, virtual reality gaming, medical rehabilitation, and ailment identification. Furthermore, wearable inertial sensors have been widely used for not only recording gait but also to predict users' demography. Machine Learning techniques such as deep learning, combined with inertial sensor signals, have shown promising results in recognizing patterns in human gait and estimate users' demography. However, the black-box nature of such deep learning models hinders the researchers from uncovering the reasons behind the model's predictions. Therefore, we propose leveraging deep learning and Layer-Wise Relevance Propagation (LRP) to identify the important variables that play a vital role in identifying the users' demography such as age and gender. To assess the 
    
[^107]: Robust SVD变得简单：一种用于大规模数据分析的快速可靠算法

    Robust SVD Made Easy: A fast and reliable algorithm for large-scale data analysis

    [https://arxiv.org/abs/2402.09754](https://arxiv.org/abs/2402.09754)

    本研究提出了一种名为球形单位正则化SVD的高效算法，用于鲁棒的SVD逼近，该算法不受异常值干扰，计算可伸缩，并能提供准确的奇异向量逼近。相比竞争算法，该算法仅使用标准降秩SVD算法两次应用于适当缩放的数据，具有显著的计算速度优势。

    

    奇异值分解（SVD）是机器学习和统计数据分析中的重要工具。然而，它对数据矩阵中的异常值非常敏感。现有的鲁棒SVD算法往往在保证鲁棒性方面牺牲了速度，或者在只有少数异常值存在时失效。本研究介绍了一种高度不受异常值干扰，计算可伸缩且提供准确奇异向量逼近的高效算法，称为球形单位正则化SVD。该算法通过仅使用标准降秩SVD算法的两个应用于适当缩放的数据，实现了显著的计算速度优势，明显优于竞争算法的计算时间。为了评估逼近奇异向量及其子空间的抗数据污染能力，我们引入了矩阵值输入的新的失效点概念，包括逐行，c

    arXiv:2402.09754v1 Announce Type: new  Abstract: The singular value decomposition (SVD) is a crucial tool in machine learning and statistical data analysis. However, it is highly susceptible to outliers in the data matrix. Existing robust SVD algorithms often sacrifice speed for robustness or fail in the presence of only a few outliers. This study introduces an efficient algorithm, called Spherically Normalized SVD, for robust SVD approximation that is highly insensitive to outliers, computationally scalable, and provides accurate approximations of singular vectors. The proposed algorithm achieves remarkable speed by utilizing only two applications of a standard reduced-rank SVD algorithm to appropriately scaled data, significantly outperforming competing algorithms in computation times. To assess the robustness of the approximated singular vectors and their subspaces against data contamination, we introduce new notions of breakdown points for matrix-valued input, including row-wise, c
    
[^108]: 大规模语言模型的模型压缩和高效推理：一项综述

    Model Compression and Efficient Inference for Large Language Models: A Survey

    [https://arxiv.org/abs/2402.09748](https://arxiv.org/abs/2402.09748)

    这项综述研究了大规模语言模型的压缩和高效推理方法，包括量化、修剪、蒸馏、紧凑架构设计和动态网络等方面。大模型的突出特点是压缩后需要微调或重新训练，并且相关的成本很高。

    

    基于Transformer的大规模语言模型取得了巨大的成功。然而，在推理过程中所产生的显著的内存和计算成本使得在资源受限设备上部署大模型变得具有挑战性。本文从算法的角度探讨了大规模语言模型的压缩和高效推理方法。在分类方面，与较小的模型类似，用于大规模语言模型的压缩和加速算法仍可以分为量化、修剪、蒸馏、紧凑架构设计和动态网络。然而，与较小的模型相比，大规模语言模型有两个突出的特点：（1）大多数压缩算法在压缩后需要微调甚至重新训练模型。大模型最显著的方面是与模型微调或训练相关的非常高的成本。因此，许多针对大规模模型的算法都需要考虑这一点。

    arXiv:2402.09748v1 Announce Type: cross  Abstract: Transformer based large language models have achieved tremendous success. However, the significant memory and computational costs incurred during the inference process make it challenging to deploy large models on resource-constrained devices. In this paper, we investigate compression and efficient inference methods for large language models from an algorithmic perspective. Regarding taxonomy, similar to smaller models, compression and acceleration algorithms for large language models can still be categorized into quantization, pruning, distillation, compact architecture design, dynamic networks. However, Large language models have two prominent characteristics compared to smaller models: (1) Most of compression algorithms require finetuning or even retraining the model after compression. The most notable aspect of large models is the very high cost associated with model finetuning or training. Therefore, many algorithms for large mode
    
[^109]: 少即是多：有限资源条件下的集成学习在视网膜疾病识别中的应用

    Less is more: Ensemble Learning for Retinal Disease Recognition Under Limited Resources

    [https://arxiv.org/abs/2402.09747](https://arxiv.org/abs/2402.09747)

    本研究通过使用集成学习方法，在有限资源条件下，提高了视网膜疾病识别的性能，克服了视网膜OCT图像获取和标签过程的挑战。

    

    视网膜光学相干断层扫描（OCT）图像对于后眼段健康至关重要，因此，推进自动化图像分析方法对于为临床医生和研究人员提供量化数据、促进知情决策至关重要。基于深度学习（DL）的方法在执行这些分析任务方面已经取得了广泛的关注，并与耗时繁重的手动分析相比表现出了优秀的性能。然而，视网膜OCT图像的获取往往存在来自隐私问题和资源密集型的标签过程的挑战，这与DL模型需要大量数据才能取得优秀性能的普遍观念相矛盾。此外，可用计算资源的限制限制了高性能医疗人工智能的进展，尤其是在少数资源环境中。

    arXiv:2402.09747v1 Announce Type: cross  Abstract: Retinal optical coherence tomography (OCT) images provide crucial insights into the health of the posterior ocular segment. Therefore, the advancement of automated image analysis methods is imperative to equip clinicians and researchers with quantitative data, thereby facilitating informed decision-making. The application of deep learning (DL)-based approaches has gained extensive traction for executing these analysis tasks, demonstrating remarkable performance compared to labor-intensive manual analyses. However, the acquisition of Retinal OCT images often presents challenges stemming from privacy concerns and the resource-intensive labeling procedures, which contradicts the prevailing notion that DL models necessitate substantial data volumes for achieving superior performance. Moreover, limitations in available computational resources constrain the progress of high-performance medical artificial intelligence, particularly in less de
    
[^110]: 选择高质量数据用于训练语言模型的QuRating方法

    QuRating: Selecting High-Quality Data for Training Language Models

    [https://arxiv.org/abs/2402.09739](https://arxiv.org/abs/2402.09739)

    QuRating是一种选择高质量数据用于训练语言模型的方法，它能够捕捉人类直观感知的文本的抽象特征。在实验中发现，平衡质量和多样性是很重要的。

    

    选择高质量的预训练数据对于创建能力强的语言模型很重要，但现有方法依赖简单的启发式方法。我们介绍了一种名为QuRating的方法，用于选择能够捕捉人类直观感知的文本的抽象特征的预训练文本数据。在本文中，我们研究了四个特征 - 写作风格、所需专业知识、事实和琐事以及教育价值。我们发现，语言模型能够辨别这些特征，并观察到它们在进行文本的配对判断方面比直接评估文本质量更好。我们训练了一个QuRater模型，从配对判断中学习标量评分，并使用它为260B的训练语料库中的每个标准进行质量评级注释。在实验中，我们根据不同的质量评级选择了30B个令牌，并在所选数据上训练了13亿参数的语言模型。我们发现在质量和多样性之间保持平衡是很重要的。

    arXiv:2402.09739v1 Announce Type: new  Abstract: Selecting high-quality pre-training data is important for creating capable language models, but existing methods rely on simple heuristics. We introduce QuRating, a method for selecting pre-training data that captures the abstract qualities of texts which humans intuitively perceive. In this paper, we investigate four qualities - writing style, required expertise, facts & trivia, and educational value. We find that LLMs are able to discern these qualities and observe that they are better at making pairwise judgments of texts than at rating the quality of a text directly. We train a QuRater model to learn scalar ratings from pairwise judgments, and use it to annotate a 260B training corpus with quality ratings for each of the four criteria. In our experiments, we select 30B tokens according to the different quality ratings and train 1.3B-parameter language models on the selected data. We find that it is important to balance quality and di
    
[^111]: DFORM: 用于评估学习模型动态的可微分矢量场对齐方法

    DFORM: Diffeomorphic vector field alignment for assessing dynamics across learned models

    [https://arxiv.org/abs/2402.09735](https://arxiv.org/abs/2402.09735)

    本文提出了DFORM框架，用于评估学习模型的动态特性。DFORM通过学习非线性坐标变换，在学习模型之间提供连续的、最大一对一映射，从而比较它们的动态特性。这扩展了平滑轨道和拓扑的概念，并解决了模型动态对比中的困难。

    

    近年来，动力系统模型（例如循环神经网络）作为科学研究中生成假设的工具变得越来越流行。评估这些网络的动态特性对于理解它们的学习生成机制至关重要。然而，由于模型固有的非线性和坐标系统的差异，跨模型对比学习动态是具有挑战性的。本文提出了DFORM（用于比较学习模型动态的可微分矢量场对齐框架）方法。DFORM学习了一个非线性坐标变换，为学习模型的轨迹提供了连续的、最大一对一映射，从而近似地确定了模型之间的可微同胚关系。DFORM-transformed矢量场的不匹配定义了两个模型之间的轨道相似性，从而扩展了平滑轨道和拓扑概念。

    arXiv:2402.09735v1 Announce Type: new  Abstract: Dynamical system models such as Recurrent Neural Networks (RNNs) have become increasingly popular as hypothesis-generating tools in scientific research. Evaluating the dynamics in such networks is key to understanding their learned generative mechanisms. However, comparison of learned dynamics across models is challenging due to their inherent nonlinearity and because a priori there is no enforced equivalence of their coordinate systems. Here, we propose the DFORM (Diffeomorphic vector field alignment for comparing dynamics across learned models) framework. DFORM learns a nonlinear coordinate transformation which provides a continuous, maximally one-to-one mapping between the trajectories of learned models, thus approximating a diffeomorphism between them. The mismatch between DFORM-transformed vector fields defines the orbital similarity between two models, thus providing a generalization of the concepts of smooth orbital and topologica
    
[^112]: DOF: 使用前向传播加速高阶微分算子

    DOF: Accelerating High-order Differential Operators with Forward Propagation

    [https://arxiv.org/abs/2402.09730](https://arxiv.org/abs/2402.09730)

    DOF是一种高效的计算框架，用于加速高阶微分算子的计算，通过前向传播的方式，不丢失精度，在效率和内存消耗上有着明显的改进。

    

    解决偏微分方程(PDE)的高效方法对于分析复杂的物理系统至关重要。最近利用深度学习来解决PDE的进展显示出重要的潜力。然而，机器学习方法，如物理知识驱动的神经网络（PINN），在处理神经网络参数化函数的高阶导数时面临挑战。受到前向Laplacian的启发，一种加速Laplacian计算的最新方法，我们提出了一种高效的计算框架，Differential Operator with Forward-propagation（DOF），用于计算一般的二阶微分算子而不丢失精度。我们提供了对比现有方法优势的严格证明，证明了我们的方法在效率上提高了两倍，并且在任何架构上减少了内存消耗。实证结果表明，我们的方法超过了传统的自动微分（AutoDiff）方法。

    arXiv:2402.09730v1 Announce Type: new  Abstract: Solving partial differential equations (PDEs) efficiently is essential for analyzing complex physical systems. Recent advancements in leveraging deep learning for solving PDE have shown significant promise. However, machine learning methods, such as Physics-Informed Neural Networks (PINN), face challenges in handling high-order derivatives of neural network-parameterized functions. Inspired by Forward Laplacian, a recent method of accelerating Laplacian computation, we propose an efficient computational framework, Differential Operator with Forward-propagation (DOF), for calculating general second-order differential operators without losing any precision. We provide rigorous proof of the advantages of our method over existing methods, demonstrating two times improvement in efficiency and reduced memory consumption on any architectures. Empirical results illustrate that our method surpasses traditional automatic differentiation (AutoDiff)
    
[^113]: 有限预算下的迅速学习最佳臂识别

    Best Arm Identification for Prompt Learning under a Limited Budget

    [https://arxiv.org/abs/2402.09723](https://arxiv.org/abs/2402.09723)

    这项工作提出了一种在提示学习中考虑有限预算约束的方法，通过建立提示学习和多臂赌博机中固定预算最佳臂识别之间的联系，提出了一个通用框架TRIPLE，通过利用聚类和嵌入思想实现了两个增强方法。

    

    大型语言模型（LLMs）的显著指令跟随能力引发了对自动学习合适提示的兴趣。然而，虽然提出了许多有效的方法，但在学习过程中产生的成本（例如访问LLM和评估响应）尚未得到考虑。为克服这个限制，本工作在提示学习中明确引入了有限预算约束。为了开发有原则的解决方案，本研究在提示学习和多臂赌博机的固定预算最佳臂识别（BAI-FB）之间建立了一种新的联系。基于这种联系，提出了一个通用框架TRIPLE（用于提示学习的最佳臂识别），以系统地利用BAI-FB在提示学习中的力量。提示学习的独特特点进一步通过利用聚类和嵌入思想提出了TRIPLE的两个基于嵌入的增强方法。

    arXiv:2402.09723v1 Announce Type: cross  Abstract: The remarkable instruction-following capability of large language models (LLMs) has sparked a growing interest in automatically learning suitable prompts. However, while many effective methods have been proposed, the cost incurred during the learning process (e.g., accessing LLM and evaluating the responses) has not been considered. To overcome this limitation, this work explicitly incorporates a finite budget constraint into prompt learning. Towards developing principled solutions, a novel connection is established between prompt learning and fixed-budget best arm identification (BAI-FB) in multi-armed bandits (MAB). Based on this connection, a general framework TRIPLE (besT aRm Identification for Prompt LEarning) is proposed to harness the power of BAI-FB in prompt learning systematically. Unique characteristics of prompt learning further lead to two embedding-based enhancements of TRIPLE by exploiting the ideas of clustering and fun
    
[^114]: 说服一位学习代理

    Persuading a Learning Agent

    [https://arxiv.org/abs/2402.09721](https://arxiv.org/abs/2402.09721)

    在一个重复的贝叶斯说服问题中，即使没有承诺能力，委托人可以通过使用上下文无遗憾学习算法来实现与经典无学习模型中具有承诺的委托人的最优效用无限接近的效果；在代理人使用上下文无交换遗憾学习算法的情况下，委托人无法获得比具有承诺的无学习模型中的最优效用更高的效用。

    

    我们研究了一个重复的贝叶斯说服问题（更一般地，任何具有完全信息的广义委托-代理问题），其中委托人没有承诺能力，代理人使用算法来学习如何对委托人的信号做出响应。我们将这个问题简化为一个一次性的广义委托-代理问题，代理人近似地最佳响应。通过这个简化，我们可以证明：如果代理人使用上下文无遗憾学习算法，则委托人可以保证其效用与经典无学习模型中具有承诺的委托人的最优效用之间可以无限接近；如果代理人使用上下文无交换遗憾学习算法，则委托人无法获得比具有承诺的无学习模型中的最优效用更高的效用。委托人在学习模型与非学习模型中可以获得的效用之间的差距是有界的。

    arXiv:2402.09721v1 Announce Type: cross  Abstract: We study a repeated Bayesian persuasion problem (and more generally, any generalized principal-agent problem with complete information) where the principal does not have commitment power and the agent uses algorithms to learn to respond to the principal's signals. We reduce this problem to a one-shot generalized principal-agent problem with an approximately-best-responding agent. This reduction allows us to show that: if the agent uses contextual no-regret learning algorithms, then the principal can guarantee a utility that is arbitrarily close to the principal's optimal utility in the classic non-learning model with commitment; if the agent uses contextual no-swap-regret learning algorithms, then the principal cannot obtain any utility significantly more than the optimal utility in the non-learning model with commitment. The difference between the principal's obtainable utility in the learning model and the non-learning model is bound
    
[^115]: DPBalance：高效和公平的隐私预算调度机制用于作为一项服务的联邦学习

    DPBalance: Efficient and Fair Privacy Budget Scheduling for Federated Learning as a Service

    [https://arxiv.org/abs/2402.09715](https://arxiv.org/abs/2402.09715)

    本文提出了DPBalance，一种新颖的隐私预算调度机制，用于联邦学习作为一项服务（FLaaS）。该机制在效率和公平性之间进行了优化，通过综合考虑数据分析师级别的主导份额和FL特定的性能指标，实现了隐私预算的精确调度。

    

    联邦学习（FL）作为一种流行的分布式机器学习方案，实现了在不聚合原始数据的情况下进行协作模型训练。云服务提供商进一步采用联邦学习作为一项服务（FLaaS），允许数据分析师在具有差分隐私保护的数据上执行他们的FL训练流程。由于差分隐私的固有特性，对数据块强制执行的隐私级别可以视为需要进行精心调度以满足不同训练流程的隐私预算。现有的隐私预算调度研究分别优先考虑效率或公平性。在本文中，我们提出了DPBalance，一种新颖的隐私预算调度机制，同时优化了效率和公平性。我们首先开发了一个综合的效用函数，将数据分析师级别的主导份额和FL特定的性能指标结合起来。然后，我们设计了一个顺序分配机制。

    arXiv:2402.09715v1 Announce Type: cross  Abstract: Federated learning (FL) has emerged as a prevalent distributed machine learning scheme that enables collaborative model training without aggregating raw data. Cloud service providers further embrace Federated Learning as a Service (FLaaS), allowing data analysts to execute their FL training pipelines over differentially-protected data. Due to the intrinsic properties of differential privacy, the enforced privacy level on data blocks can be viewed as a privacy budget that requires careful scheduling to cater to diverse training pipelines. Existing privacy budget scheduling studies prioritize either efficiency or fairness individually. In this paper, we propose DPBalance, a novel privacy budget scheduling mechanism that jointly optimizes both efficiency and fairness. We first develop a comprehensive utility function incorporating data analyst-level dominant shares and FL-specific performance metrics. A sequential allocation mechanism is 
    
[^116]: 节点复制改善冷启动链路预测

    Node Duplication Improves Cold-start Link Prediction

    [https://arxiv.org/abs/2402.09711](https://arxiv.org/abs/2402.09711)

    本文研究了在链路预测中改进GNN在低度节点上的性能，提出了一种名为NodeDup的增强技术，通过复制低度节点并创建链接来提高性能。

    

    图神经网络（GNN）在图机器学习中非常突出，并在链路预测（LP）任务中展现了最先进的性能。然而，最近的研究表明，尽管整体上表现出色，GNN在低度节点上的表现却较差。在推荐系统等LP的实际应用中，改善低度节点的性能至关重要，因为这等同于解决冷启动问题，提高用户在少数观察的相互作用中的体验。本文研究了改进GNN在低度节点上的LP性能，同时保持其在高度节点上的性能，并提出了一种简单但非常有效的增强技术，称为NodeDup。具体而言，NodeDup在标准的监督LP训练方案中，在低度节点上复制节点并在节点和其副本之间创建链接。通过利用“多视图”视角，该方法可以显著提高LP的性能。

    arXiv:2402.09711v1 Announce Type: new  Abstract: Graph Neural Networks (GNNs) are prominent in graph machine learning and have shown state-of-the-art performance in Link Prediction (LP) tasks. Nonetheless, recent studies show that GNNs struggle to produce good results on low-degree nodes despite their overall strong performance. In practical applications of LP, like recommendation systems, improving performance on low-degree nodes is critical, as it amounts to tackling the cold-start problem of improving the experiences of users with few observed interactions. In this paper, we investigate improving GNNs' LP performance on low-degree nodes while preserving their performance on high-degree nodes and propose a simple yet surprisingly effective augmentation technique called NodeDup. Specifically, NodeDup duplicates low-degree nodes and creates links between nodes and their own duplicates before following the standard supervised LP training scheme. By leveraging a ''multi-view'' perspectiv
    
[^117]: 在开放无线接入网络中保护机器学习驱动的应用的数据隐私

    Preserving Data Privacy for ML-driven Applications in Open Radio Access Networks

    [https://arxiv.org/abs/2402.09710](https://arxiv.org/abs/2402.09710)

    本文研究了在5G开放无线接入网络（O-RAN）中共享数据库场景下的数据隐私问题，并提出了一种基于洗牌的可学习加密技术来保护机器学习模型的数据隐私。

    

    深度学习提供了一种改进频谱访问技术的有希望的解决方案，通过利用数据驱动的方法来管理和共享有限的频谱资源，用于新兴应用。对于其中几种应用，敏感的无线数据（如频谱图）存储在共享数据库或多方利益相关者云环境中，因此容易造成隐私泄漏。本文旨在通过研究5G开放无线接入网络（O-RAN）中共享数据库场景的典型案例来解决此类隐私问题，在这些场景中，我们在近实时（near-RT）无线接入网络智能控制器中有一个共享数据库。我们着重讨论了如何保护用于频谱共享和干扰缓解应用的机器学习（ML）模型所使用的数据，同时不影响模型和网络的性能。其中的基本想法是利用基于洗牌的可学习加密技术来加密数据。

    arXiv:2402.09710v1 Announce Type: cross  Abstract: Deep learning offers a promising solution to improve spectrum access techniques by utilizing data-driven approaches to manage and share limited spectrum resources for emerging applications. For several of these applications, the sensitive wireless data (such as spectrograms) are stored in a shared database or multistakeholder cloud environment and are therefore prone to privacy leaks. This paper aims to address such privacy concerns by examining the representative case study of shared database scenarios in 5G Open Radio Access Network (O-RAN) networks where we have a shared database within the near-real-time (near-RT) RAN intelligent controller. We focus on securing the data that can be used by machine learning (ML) models for spectrum sharing and interference mitigation applications without compromising the model and network performances. The underlying idea is to leverage a (i) Shuffling-based learnable encryption technique to encryp
    
[^118]: 无需稀疏模型的稀疏且准确的解释

    Sparse and Faithful Explanations Without Sparse Models

    [https://arxiv.org/abs/2402.09702](https://arxiv.org/abs/2402.09702)

    引入了稀疏解释值(SEV)，用于衡量机器学习模型的决策稀疏性。即使模型不是稀疏的，许多机器学习模型在SEV的衡量下仍具有低决策稀疏性。

    

    即使模型不满足全局的稀疏性，决策仍然可以用少量的特征准确地描述。例如，对于某人而言，尽管没有信用历史，但申请大笔贷款可能会被拒绝，这就忽视了与其信用价值相关的任何证据。在本论文中，我们引入了稀疏解释值（SEV），这是一种衡量机器学习模型稀疏性的新方法。在以上贷款拒绝的例子中，SEV为1，因为只需要一个因素来解释为什么贷款被拒绝。SEV是对决策稀疏性的衡量，而不是对整体模型稀疏性的衡量，并且我们能够证明许多机器学习模型——即使它们不是稀疏的——实际上在SEV的衡量下具有低决策稀疏性。SEV使用超立方体上的移动进行定义，使得SEV能够在各种模型类别上一致地定义，其中移动限制反映了模型的性质。

    arXiv:2402.09702v1 Announce Type: new  Abstract: Even if a model is not globally sparse, it is possible for decisions made from that model to be accurately and faithfully described by a small number of features. For instance, an application for a large loan might be denied to someone because they have no credit history, which overwhelms any evidence towards their creditworthiness. In this work, we introduce the Sparse Explanation Value (SEV), a new way of measuring sparsity in machine learning models. In the loan denial example above, the SEV is 1 because only one factor is needed to explain why the loan was denied. SEV is a measure of decision sparsity rather than overall model sparsity, and we are able to show that many machine learning models -- even if they are not sparse -- actually have low decision sparsity, as measured by SEV. SEV is defined using movements over a hypercube, allowing SEV to be defined consistently over various model classes, with movement restrictions reflectin
    
[^119]: 合并不同过滤器中的证据

    Combining Evidence Across Filtrations

    [https://arxiv.org/abs/2402.09698](https://arxiv.org/abs/2402.09698)

    这篇论文研究了合并使用不同过滤器计算的e进程的方法，探讨了其在顺序推理中的应用。

    

    在任何时刻有效的顺序推理中，已知任何可接受的推理方法必须基于测试鞅和它们的组合广义化，称为e进程，它们是非负进程，其在任何任意停时的期望上界不超过一。e进程量化了针对复合零假设的一系列结果的累积证据。本文研究了使用不同信息集（即过滤器）计算的e进程的合并方法，针对一个零假设。尽管在相同过滤器上构建的e进程可以轻松地合并（例如，通过平均），但在不同过滤器上构建的e进程不能那么容易地合并，因为它们在较粗的过滤器中的有效性不能转换为在更细的过滤器中的有效性。我们讨论了文献中三个具体例子：可交换性测试，独立性测试等。

    arXiv:2402.09698v1 Announce Type: cross  Abstract: In anytime-valid sequential inference, it is known that any admissible inference procedure must be based on test martingales and their composite generalization, called e-processes, which are nonnegative processes whose expectation at any arbitrary stopping time is upper-bounded by one. An e-process quantifies the accumulated evidence against a composite null hypothesis over a sequence of outcomes. This paper studies methods for combining e-processes that are computed using different information sets, i.e., filtrations, for a null hypothesis. Even though e-processes constructed on the same filtration can be combined effortlessly (e.g., by averaging), e-processes constructed on different filtrations cannot be combined as easily because their validity in a coarser filtration does not translate to validity in a finer filtration. We discuss three concrete examples of such e-processes in the literature: exchangeability tests, independence te
    
[^120]: 对离线强化学习中的奖励污染攻击的研究

    Reward Poisoning Attack Against Offline Reinforcement Learning

    [https://arxiv.org/abs/2402.09695](https://arxiv.org/abs/2402.09695)

    这项研究针对深度神经网络函数逼近的一般离线强化学习中的奖励污染攻击问题，提出了一种名为“策略对比攻击”的攻击策略。通过使低性能策略看起来像是高性能的，同时使高性能策略看起来像是低性能的，我们证明了这种攻击有效性。

    

    我们研究了针对深度神经网络函数逼近的一般离线强化学习中的奖励污染攻击问题。我们考虑了一个黑盒威胁模型，攻击者对学习算法完全不了解，并且其预算受到限制，限制了每个数据点的污染量以及总扰动。我们提出了一种名为“策略对比攻击”的攻击策略。其高层思想是使一些低性能策略看起来像是高性能的，同时使高性能策略看起来像是低性能的。据我们所知，我们首次提出了一种适用于一般离线强化学习场景的黑盒奖励污染攻击。我们提供了关于攻击设计的理论洞察，并在不同类型的学习数据集上经验证明我们的攻击对当前最先进的离线强化学习算法是有效的。

    arXiv:2402.09695v1 Announce Type: cross  Abstract: We study the problem of reward poisoning attacks against general offline reinforcement learning with deep neural networks for function approximation. We consider a black-box threat model where the attacker is completely oblivious to the learning algorithm and its budget is limited by constraining both the amount of corruption at each data point, and the total perturbation. We propose an attack strategy called `policy contrast attack'. The high-level idea is to make some low-performing policies appear as high-performing while making high-performing policies appear as low-performing. To the best of our knowledge, we propose the first black-box reward poisoning attack in the general offline RL setting. We provide theoretical insights on the attack design and empirically show that our attack is efficient against current state-of-the-art offline RL algorithms in different kinds of learning datasets.
    
[^121]: 鲁棒学习增强字典

    Robust Learning-Augmented Dictionaries

    [https://arxiv.org/abs/2402.09687](https://arxiv.org/abs/2402.09687)

    我们提出了一个学习增强的数据结构，通过预测访问频率增强跳表，实现了最佳一致性和鲁棒性。实验证明，与其他数据结构相比，RobustSL在合成数据集和真实数据集上表现出色。

    

    我们提出了第一个具有最佳一致性和鲁棒性的学习增强数据结构来实现字典。我们的数据结构名为RobustSL，它是一个通过对数据序列中元素的访问频率进行预测而增强的跳表。通过恰当的预测，RobustSL可以实现最佳一致性（实现静态最优性）。同时，它能够保持每个操作的对数运行时间，确保最佳的鲁棒性，即使预测是以对抗性方式生成的。因此，RobustSL具有林、罗和伍德洛夫（ICML 2022）以及曹等人（arXiv 2023）最近提出的学习增强数据结构的所有优势，同时提供了在之前工作中缺失的鲁棒性保证。数值实验表明，RobustSL在使用合成数据集和真实数据集时优于替代数据结构。

    arXiv:2402.09687v1 Announce Type: cross  Abstract: We present the first learning-augmented data structure for implementing dictionaries with optimal consistency and robustness. Our data structure, named RobustSL, is a skip list augmented by predictions of access frequencies of elements in a data sequence. With proper predictions, RobustSL has optimal consistency (achieves static optimality). At the same time, it maintains a logarithmic running time for each operation, ensuring optimal robustness, even if predictions are generated adversarially. Therefore, RobustSL has all the advantages of the recent learning-augmented data structures of Lin, Luo, and Woodruff (ICML 2022) and Cao et al. (arXiv 2023), while providing robustness guarantees that are absent in the previous work. Numerical experiments show that RobustSL outperforms alternative data structures using both synthetic and real datasets.
    
[^122]: HyperMagNet:一种基于磁度拉普拉斯的超图神经网络

    HyperMagNet: A Magnetic Laplacian based Hypergraph Neural Network

    [https://arxiv.org/abs/2402.09676](https://arxiv.org/abs/2402.09676)

    HyperMagNet是一种基于磁度拉普拉斯的超图神经网络，通过将超图表示为非可逆的马尔可夫链并构建磁度拉普拉斯矩阵作为输入，它在节点分类任务中表现出优越性。

    

    在数据科学领域，超图是对展示多种关系的数据的自然模型，而图只能捕捉到两两之间的关系。然而，许多现有的超图神经网络通过对称矩阵表示将超图有效地简化为无向图，可能会丢失重要信息。我们提出了一种替代超图神经网络的方法，其中将超图表示为非可逆的马尔可夫链。我们使用该马尔可夫链构建了一个复数埃尔米特拉普拉斯矩阵 - 磁度拉普拉斯矩阵，该矩阵作为我们提出的超图神经网络的输入。我们研究了HyperMagNet在节点分类任务中的效果，并证明其在基于图简化的超图神经网络上的优越性。

    arXiv:2402.09676v1 Announce Type: new  Abstract: In data science, hypergraphs are natural models for data exhibiting multi-way relations, whereas graphs only capture pairwise. Nonetheless, many proposed hypergraph neural networks effectively reduce hypergraphs to undirected graphs via symmetrized matrix representations, potentially losing important information. We propose an alternative approach to hypergraph neural networks in which the hypergraph is represented as a non-reversible Markov chain. We use this Markov chain to construct a complex Hermitian Laplacian matrix - the magnetic Laplacian - which serves as the input to our proposed hypergraph neural network. We study HyperMagNet for the task of node classification, and demonstrate its effectiveness over graph-reduction based hypergraph neural networks.
    
[^123]: PAL：对大型语言模型的代理引导黑盒攻击

    PAL: Proxy-Guided Black-Box Attack on Large Language Models

    [https://arxiv.org/abs/2402.09674](https://arxiv.org/abs/2402.09674)

    PAL是第一个黑盒查询攻击大型语言模型的优化算法，通过代理模型引导优化过程，并使用复杂的损失函数，取得了较高的攻击成功率。

    

    大型语言模型（LLMs）近几个月来越来越受欢迎，但在被操纵时它们展示出的危险能力令人担忧。尽管安全微调等技术旨在最小化有害使用，但最近的研究表明，LLMs仍然容易受到引发有毒回应的攻击。在这项工作中，我们引入了对LLMs的代理引导攻击（PAL），这是第一个基于优化的对LLMs的黑盒仅查询攻击。具体而言，它依赖于一个替代模型来引导优化过程，并采用了针对真实世界LLM API设计的复杂损失函数。我们的攻击在GPT-3.5-Turbo上达到84%的攻击成功率（ASR），在Llama-2-7B上达到48%，而目前最先进的方法仅为4%。我们还提出了GCG++，这是对GCG攻击的改进，在白盒Llama-2-7B上达到了94%的ASR，以及基于查询的攻击的强有力但简单的基准方法——LLMs上的随机搜索攻击（RAL）。

    arXiv:2402.09674v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have surged in popularity in recent months, but they have demonstrated concerning capabilities to generate harmful content when manipulated. While techniques like safety fine-tuning aim to minimize harmful use, recent works have shown that LLMs remain vulnerable to attacks that elicit toxic responses. In this work, we introduce the Proxy-Guided Attack on LLMs (PAL), the first optimization-based attack on LLMs in a black-box query-only setting. In particular, it relies on a surrogate model to guide the optimization and a sophisticated loss designed for real-world LLM APIs. Our attack achieves 84% attack success rate (ASR) on GPT-3.5-Turbo and 48% on Llama-2-7B, compared to 4% for the current state of the art. We also propose GCG++, an improvement to the GCG attack that reaches 94% ASR on white-box Llama-2-7B, and the Random-Search Attack on LLMs (RAL), a strong but simple baseline for query-based attacks. We
    
[^124]: 利用语言和视觉AI系统中的Alpha透明性的探索

    Exploiting Alpha Transparency In Language And Vision-Based AI Systems

    [https://arxiv.org/abs/2402.09671](https://arxiv.org/abs/2402.09671)

    这项研究揭示了利用PNG图像文件格式中的alpha透明层欺骗AI视觉系统的新漏洞，对现有的和实际应用的视觉系统提出了挑战。

    

    这项研究揭示了一种新颖的利用PNG图像文件格式的漏洞，具体是它们的alpha透明层，并且展示了这个漏洞对多个AI视觉系统的欺骗潜力。我们的方法利用这个alpha透明层作为一个对人类观察者不可见但完全可操作的秘密通道来欺骗AI图像处理器。受漏洞测试的范围包括苹果、微软、谷歌、Salesforce、Nvidia和Facebook等代表性视觉系统，突显了攻击的潜在广度。这个漏洞对现有的和实际应用的视觉系统提出了安全协议的挑战，从医学成像到自动驾驶技术。我们的实验表明，受到影响的系统，无论是依赖卷积神经网络还是最新的多模态语言模型，都不能通过简单的补丁或更新快速地缓解这些漏洞。相反，它们需要重新训练和架构变化，表明这些漏洞是持久的。

    arXiv:2402.09671v1 Announce Type: cross  Abstract: This investigation reveals a novel exploit derived from PNG image file formats, specifically their alpha transparency layer, and its potential to fool multiple AI vision systems. Our method uses this alpha layer as a clandestine channel invisible to human observers but fully actionable by AI image processors. The scope tested for the vulnerability spans representative vision systems from Apple, Microsoft, Google, Salesforce, Nvidia, and Facebook, highlighting the attack's potential breadth. This vulnerability challenges the security protocols of existing and fielded vision systems, from medical imaging to autonomous driving technologies. Our experiments demonstrate that the affected systems, which rely on convolutional neural networks or the latest multimodal language models, cannot quickly mitigate these vulnerabilities through simple patches or updates. Instead, they require retraining and architectural changes, indicating a persiste
    
[^125]: 如何训练数据高效的LLM模型

    How to Train Data-Efficient LLMs

    [https://arxiv.org/abs/2402.09668](https://arxiv.org/abs/2402.09668)

    本文研究了如何训练数据高效的LLM模型，提出了Ask-LLM和Density两种优秀的数据选择方法。

    

    大型语言模型（LLM）的训练十分昂贵。本文研究了用于预训练LLM的数据高效方法，即旨在优化模型质量和训练资源/数据消耗的帕累托前沿的技术。我们试图理解基于（i）昂贵的数据质量估计和（ii）基于特征空间的覆盖率和多样性测量的数据选择程序所带来的权衡。我们的第一种技术“Ask-LLM”利用调节指令的LLM的零样本推理能力来直接评估训练样例的质量。为了达到覆盖率，我们提出了密度采样，它根据数据分布选择多样的样本。在我们对19种采样器进行了数百个评估任务和预训练运行的对比研究中，我们发现Ask-LLM和Density是各自类别中最好的方法。

    arXiv:2402.09668v1 Announce Type: cross  Abstract: The training of large language models (LLMs) is expensive. In this paper, we study data-efficient approaches for pre-training LLMs, i.e., techniques that aim to optimize the Pareto frontier of model quality and training resource/data consumption. We seek to understand the tradeoffs associated with data selection routines based on (i) expensive-to-compute data-quality estimates, and (ii) maximization of coverage and diversity-based measures in the feature space. Our first technique, Ask-LLM, leverages the zero-shot reasoning capabilities of instruction-tuned LLMs to directly assess the quality of a training example. To target coverage, we propose Density sampling, which models the data distribution to select a diverse sample. In our comparison of 19 samplers, involving hundreds of evaluation tasks and pre-training runs, we find that Ask-LLM and Density are the best methods in their respective categories. Coverage sampling can recover th
    
[^126]: 用户建模与用户画像：综述

    User Modeling and User Profiling: A Comprehensive Survey

    [https://arxiv.org/abs/2402.09660](https://arxiv.org/abs/2402.09660)

    这篇综述论文介绍了用户建模与用户画像研究的现状、发展和未来方向。该研究主要关注在人工智能应用中构建准确的用户表示，包括利用大量数据进行建模以及采用深度学习和图数据技术等先进方法。

    

    人工智能（AI）融入日常生活，特别是通过信息检索和推荐系统，已经促使先进的用户建模和用户画像技术，以提供个性化体验。这些技术旨在基于与这些系统的互动中生成的大量数据构建准确的用户表示。本文对用户建模和用户画像研究的现状、发展和未来方向进行了全面综述。我们提供了一个历史概述，追溯了从早期的刻板模型到最新的深度学习技术，并提出了一个新的分类体系，涵盖了这一研究领域中的所有活动主题，包括最近的趋势。我们的综述突出了向更复杂的用户画像方法的范式转变，强调了隐式数据收集、多行为建模以及图数据的整合。

    arXiv:2402.09660v1 Announce Type: new  Abstract: The integration of artificial intelligence (AI) into daily life, particularly through information retrieval and recommender systems, has necessitated advanced user modeling and profiling techniques to deliver personalized experiences. These techniques aim to construct accurate user representations based on the rich amounts of data generated through interactions with these systems. This paper presents a comprehensive survey of the current state, evolution, and future directions of user modeling and profiling research. We provide a historical overview, tracing the development from early stereotype models to the latest deep learning techniques, and propose a novel taxonomy that encompasses all active topics in this research area, including recent trends. Our survey highlights the paradigm shifts towards more sophisticated user profiling methods, emphasizing implicit data collection, multi-behavior modeling, and the integration of graph data
    
[^127]: 数字与模拟传输在无线网络上的联邦学习中的比较

    Digital versus Analog Transmissions for Federated Learning over Wireless Networks

    [https://arxiv.org/abs/2402.09657](https://arxiv.org/abs/2402.09657)

    本文比较了数字和模拟传输在无线联邦学习中的效果，发现它们的本质区别在于通信和计算是否同时设计，数字方案分离了通信设计和具体任务，而模拟通信可以同时处理大规模设备的传输。

    

    本文定量比较了无线联邦学习中的两种有效通信方案，即数字传输和模拟传输，突出了它们的本质区别以及各自的应用场景。我们首先分析了数字和模拟传输方法，并在实际约束条件下建立了统一公平的比较方案。我们建立了一种通用收敛性分析，用于评估无线网络中联邦学习的性能。这些分析结果揭示了这两种范例的根本区别在于通信和计算是否同时设计。数字方案将通信设计与具体的联邦学习任务分离，使得支持大量设备同时上行传输成为困难。相比之下，模拟通信允许同时处理大规模设备的上行传输。

    arXiv:2402.09657v1 Announce Type: cross  Abstract: In this paper, we quantitatively compare these two effective communication schemes, i.e., digital and analog ones, for wireless federated learning (FL) over resource-constrained networks, highlighting their essential differences as well as their respective application scenarios. We first examine both digital and analog transmission methods, together with a unified and fair comparison scheme under practical constraints. A universal convergence analysis under various imperfections is established for FL performance evaluation in wireless networks. These analytical results reveal that the fundamental difference between the two paradigms lies in whether communication and computation are jointly designed or not. The digital schemes decouple the communication design from specific FL tasks, making it difficult to support simultaneous uplink transmission of massive devices with limited bandwidth. In contrast, the analog communication allows ove
    
[^128]: Atlassian的CI构建失败预测的从业者挑战和感知研究

    Practitioners' Challenges and Perceptions of CI Build Failure Predictions at Atlassian

    [https://arxiv.org/abs/2402.09651](https://arxiv.org/abs/2402.09651)

    Atlassian的研究调查了CI构建失败对软件开发过程和团队的影响，并研究了将CI构建预测工具集成到Bitbucket环境中所涉及的挑战和期望。

    

    持续集成（CI）构建失败可能会对软件开发过程和团队产生重大影响，如延迟发布新功能和降低开发人员的生产力。本研究报告了一项实证研究，调查了Atlassian在产品开发过程中的CI构建失败情况。我们的定量分析发现，代码库维度是影响CI构建失败的关键因素。此外，我们的定性调查发现，Atlassian开发人员认为CI构建失败是实践中的挑战性问题。此外，我们发现CI构建预测不仅可以提供对CI构建失败的积极见解，还可以促进团队决策。我们的研究为将CI构建预测工具集成到Bitbucket环境中所涉及的挑战和期望提供了有价值的见解，从而增强了CI流程。

    arXiv:2402.09651v1 Announce Type: cross  Abstract: Continuous Integration (CI) build failures could significantly impact the software development process and teams, such as delaying the release of new features and reducing developers' productivity. In this work, we report on an empirical study that investigates CI build failures throughout product development at Atlassian. Our quantitative analysis found that the repository dimension is the key factor influencing CI build failures. In addition, our qualitative survey revealed that Atlassian developers perceive CI build failures as challenging issues in practice. Furthermore, we found that the CI build prediction can not only provide proactive insight into CI build failures but also facilitate the team's decision-making. Our study sheds light on the challenges and expectations involved in integrating CI build prediction tools into the Bitbucket environment, providing valuable insights for enhancing CI processes.
    
[^129]: 利用足球广播视频中的估计姿势来预测犯规

    Foul prediction with estimated poses from soccer broadcast video

    [https://arxiv.org/abs/2402.09650](https://arxiv.org/abs/2402.09650)

    通过整合视频数据、边界框位置、图像细节和姿势信息，我们提出了一种创新的深度学习方法来预测足球犯规。实验结果表明，我们的模型在各方面表现出色。

    

    近年来，计算机视觉在运动员跟踪和姿势估计方面取得了显著进展。然而，在运动中使用姿势估计进行行为预测的研究较少，特别是对于足球犯规的预测具有挑战性，因为每个球员的图像尺寸较小，并且使用例如球和姿势信息的困难。在我们的研究中，我们引入了一种创新的深度学习方法来预测足球犯规。该方法通过整理一个新颖的足球犯规数据集，将视频数据、边界框位置、图像细节和姿势信息相结合。我们的模型利用卷积神经网络（CNN）和循环神经网络（RNN）的结合方法有效地融合这四种模态的信息。实验结果表明，我们的完整模型优于剥离模型，以及所有的RNN模块、边界框位置和图像、估计的姿势。

    arXiv:2402.09650v1 Announce Type: cross  Abstract: Recent advances in computer vision have made significant progress in tracking and pose estimation of sports players. However, there have been fewer studies on behavior prediction with pose estimation in sports, in particular, the prediction of soccer fouls is challenging because of the smaller image size of each player and of difficulty in the usage of e.g., the ball and pose information. In our research, we introduce an innovative deep learning approach for anticipating soccer fouls. This method integrates video data, bounding box positions, image details, and pose information by curating a novel soccer foul dataset. Our model utilizes a combination of convolutional and recurrent neural networks (CNNs and RNNs) to effectively merge information from these four modalities. The experimental results show that our full model outperformed the ablated models, and all of the RNN modules, bounding box position and image, and estimated pose wer
    
[^130]: 多模态优化方法：一项调查

    Multi-Fidelity Methods for Optimization: A Survey

    [https://arxiv.org/abs/2402.09638](https://arxiv.org/abs/2402.09638)

    多模态优化(MFO)是一种以高模态准确性和计算效率平衡的成本效益策略，通过多模态替代模型、忠诚度管理策略和优化技术来解决复杂计算挑战的方法。MFO在机器学习、工程设计优化和科学发现等多个关键领域有广泛应用的潜力。

    

    真实世界中的黑盒优化往往涉及耗时或昂贵的实验和模拟。多模态优化(MFO)通过一种分层的忠诚度方法，以高模态准确性和计算效率平衡的成本效益策略而脱颖而出。本调查以一个基于预训练语言模型的新颖文本挖掘框架为基础，对 MFO 进行了系统的探索。我们深入研究了 MFO 的基本原理和方法，重点关注三个核心组成部分——多模态代理模型、忠诚度管理策略和优化技术。此外，本调查还突出了 MFO 在机器学习、工程设计优化和科学发现等多个关键领域的不同应用，展示了 MFO 在解决复杂计算挑战方面的适应性和效果。此外，我们还展望了几种应用 MFO 的未来研究方向。

    arXiv:2402.09638v1 Announce Type: new  Abstract: Real-world black-box optimization often involves time-consuming or costly experiments and simulations. Multi-fidelity optimization (MFO) stands out as a cost-effective strategy that balances high-fidelity accuracy with computational efficiency through a hierarchical fidelity approach. This survey presents a systematic exploration of MFO, underpinned by a novel text mining framework based on a pre-trained language model. We delve deep into the foundational principles and methodologies of MFO, focusing on three core components -- multi-fidelity surrogate models, fidelity management strategies, and optimization techniques. Additionally, this survey highlights the diverse applications of MFO across several key domains, including machine learning, engineering design optimization, and scientific discovery, showcasing the adaptability and effectiveness of MFO in tackling complex computational challenges. Furthermore, we also envision several em
    
[^131]: MiMiC：表示空间中最小修改的对抗事实

    MiMiC: Minimally Modified Counterfactuals in the Representation Space

    [https://arxiv.org/abs/2402.09631](https://arxiv.org/abs/2402.09631)

    提出了一种新颖的对抗事实生成方法，利用闭式解决方案在表示空间中生成富有表达力的对抗事实，以减轻语言模型中的不良行为，该方法在地球移动问题方面提供理论上的保证，并对表示空间的几何组织进行改进。

    

    arXiv:2402.09631v1 公告类型：交叉学科 简介：语言模型经常表现出不良行为，如性别偏见或有毒语言。通过对表示空间进行干预，可以有效减轻这些问题，但两种常见的干预技术，即线性擦除和定向向量，并不能提供高度可控和表达丰富度。因此，我们提出了一种新颖的干预方法，旨在在表示空间中生成富有表达力的对抗事实，使源类别（例如“有毒”）的表示与目标类别（例如“非有毒”）的表示相似。这种方法利用高斯假设下的闭式解决方案，在地球移动问题方面提供了理论上的保证，并对表示空间的几何组织提供了进一步的改进。

    arXiv:2402.09631v1 Announce Type: cross  Abstract: Language models often exhibit undesirable behaviors, such as gender bias or toxic language. Interventions in the representation space were shown effective in mitigating such issues by altering the LM behavior. We first show that two prominent intervention techniques, Linear Erasure and Steering Vectors, do not enable a high degree of control and are limited in expressivity.   We then propose a novel intervention methodology for generating expressive counterfactuals in the representation space, aiming to make representations of a source class (e.g., ``toxic'') resemble those of a target class (e.g., ``non-toxic''). This approach, generalizing previous linear intervention techniques, utilizes a closed-form solution for the Earth Mover's problem under Gaussian assumptions and provides theoretical guarantees on the representation space's geometric organization. We further build on this technique and derive a nonlinear intervention that ena
    
[^132]: 通过强化学习实现无监督联邦学习的智能信息交换

    Smart Information Exchange for Unsupervised Federated Learning via Reinforcement Learning

    [https://arxiv.org/abs/2402.09629](https://arxiv.org/abs/2402.09629)

    本研究通过强化学习方法，提出了一种解决无监督联邦学习中数据交换问题的智能信息交换方法，该方法通过创建一个最优图形来选择数据传输的链接，以提高收敛速度和提高异常设备鲁棒性。

    

    分布式机器学习范 paradigm，如联邦学习（FL）的主要挑战之一是分布式设备中存在非独立同分布的本地数据集。设备间通信（D2D）已被证明是处理这一问题且对于有异常设备具有鲁棒性的有效工具。然而，在无监督情况下，由于缺少标签，如何进行数据交换并不明显。在本文中，我们提出了一种利用强化学习创建数据传输的最优图形的方法。目标是形成能够在无监督 FL 环境中考虑环境约束、提高收敛速度的链接。数值分析表明，所提出的方法在收敛速度和异常设备鲁棒性方面优于不同可用 FL 方案和基准数据集。

    arXiv:2402.09629v1 Announce Type: new  Abstract: One of the main challenges of decentralized machine learning paradigms such as Federated Learning (FL) is the presence of local non-i.i.d. datasets. Device-to-device transfers (D2D) between distributed devices has been shown to be an effective tool for dealing with this problem and robust to stragglers. In an unsupervised case, however, it is not obvious how data exchanges should take place due to the absence of labels. In this paper, we propose an approach to create an optimal graph for data transfer using Reinforcement Learning. The goal is to form links that will provide the most benefit considering the environment's constraints and improve convergence speed in an unsupervised FL environment. Numerical analysis shows the advantages in terms of convergence speed and straggler resilience of the proposed method to different available FL schemes and benchmark datasets.
    
[^133]: 多元轨迹的符合性自适应预测方法

    Conformalized Adaptive Forecasting of Heterogeneous Trajectories

    [https://arxiv.org/abs/2402.09623](https://arxiv.org/abs/2402.09623)

    本研究提出了一种新的符合性方法，通过结合在线符合性预测技术和解决回归中异方差性的方法，生成了同时预测边界，并能够可靠地覆盖新随机轨迹的整个路径。这种方法不仅有精确的有限样本保证，而且往往比之前的方法具有更丰富的预测结果。

    

    本文提出了一种新的符合性方法，用于生成同时预测边界，以具有足够高的概率覆盖新随机轨迹的整个路径。鉴于在运动规划应用中需要可靠的不确定性估计，其中不同物体的行为可能更或更少可预测，我们将来自单个和多个时间序列的在线符合性预测技术，以及解决回归中的异方差性的方法进行了融合。该解决方案既有原则性，提供了精确的有限样本保证，又有效，通常比先前的方法具有更丰富的预测结果。

    arXiv:2402.09623v1 Announce Type: cross  Abstract: This paper presents a new conformal method for generating simultaneous forecasting bands guaranteed to cover the entire path of a new random trajectory with sufficiently high probability. Prompted by the need for dependable uncertainty estimates in motion planning applications where the behavior of diverse objects may be more or less unpredictable, we blend different techniques from online conformal prediction of single and multiple time series, as well as ideas for addressing heteroscedasticity in regression. This solution is both principled, providing precise finite-sample guarantees, and effective, often leading to more informative predictions than prior methods.
    
[^134]: API Pack：一个用于API调用生成的大规模多语言数据集

    API Pack: A Massive Multilingual Dataset for API Call Generation

    [https://arxiv.org/abs/2402.09615](https://arxiv.org/abs/2402.09615)

    这个论文介绍了一个名为API Pack的大规模多语言数据集，旨在提高大型语言模型的API调用生成能力，通过实验证明了其在生成未见过的API调用方面的高准确率，并实现了跨语言的API调用生成

    

    我们介绍了API Pack，一个包含超过一百万个指令-API调用对的多语言数据集，旨在提高大型语言模型的API调用生成能力。通过实验，我们证明了API Pack在提升模型在这一特定任务上的效果的同时，保持其在一般编码方面的整体熟练程度。仅在20,000个Python实例上对CodeLlama-13B进行微调，其生成未见过的API调用的准确率比GPT-3.5和GPT-4分别高出10%和5%。扩展到100k个例子可以提高对训练期间未见过的新API的泛化能力。此外，实现了跨语言的API调用生成，而无需大量语言特定的数据。数据集、经过微调的模型和整体代码库可在https://github.com/anonymous_url上公开获取。

    arXiv:2402.09615v1 Announce Type: cross  Abstract: We introduce API Pack, a multilingual dataset featuring over one million instruction-API call pairs aimed at advancing large language models' API call generation capabilities. Through experiments, we demonstrate API Pack's efficacy in enhancing models for this specialized task while maintaining their overall proficiency at general coding. Fine-tuning CodeLlama-13B on just 20,000 Python instances yields over 10% and 5% higher accuracy than GPT-3.5 and GPT-4 respectively in generating unseen API calls. Scaling to 100k examples improves generalization to new APIs not seen during training. In addition, cross-lingual API call generation is achieved without needing extensive data per language. The dataset, fine-tuned models, and overall code base are publicly available at https://github.com/anonymous_url.
    
[^135]: 实现规模化隐私感知手语翻译

    Towards Privacy-Aware Sign Language Translation at Scale

    [https://arxiv.org/abs/2402.09611](https://arxiv.org/abs/2402.09611)

    本研究提出了一种两阶段框架，用于实现规模化隐私感知手语翻译。我们利用自监督视频预训练和有监督微调的方法，在数据稀缺和隐私风险的情况下实现了最先进的手语翻译性能。

    

    手语翻译的一个主要障碍是数据稀缺。目前在网络上可用的大部分手语数据由于缺乏对齐的字幕而无法用于训练监督模型。此外，使用大规模网络抓取的数据集来扩展手语翻译存在隐私风险，因为其中包含生物特征信息，负责任地开发手语翻译技术应该考虑这一点。在这项工作中，我们提出了一种针对规模化隐私感知手语翻译的两阶段框架，解决了这两个问题。我们引入了SSVP-SLT，它利用匿名和未注释的视频进行自监督视频预训练，然后利用经过筛选的平行数据集进行有监督的手语翻译微调。 SSVP-SLT在How2Sign数据集上实现了最新的微调和零次gloss-free手语翻译性能，比最强的基线模型提高了3个BLEU-4。通过受控实验，我们证明了我们的方法在多个语言和手语词汇上都具有较好的泛化能力。

    arXiv:2402.09611v1 Announce Type: new  Abstract: A major impediment to the advancement of sign language translation (SLT) is data scarcity. Much of the sign language data currently available on the web cannot be used for training supervised models due to the lack of aligned captions. Furthermore, scaling SLT using large-scale web-scraped datasets bears privacy risks due to the presence of biometric information, which the responsible development of SLT technologies should account for. In this work, we propose a two-stage framework for privacy-aware SLT at scale that addresses both of these issues. We introduce SSVP-SLT, which leverages self-supervised video pretraining on anonymized and unannotated videos, followed by supervised SLT finetuning on a curated parallel dataset. SSVP-SLT achieves state-of-the-art finetuned and zero-shot gloss-free SLT performance on the How2Sign dataset, outperforming the strongest respective baselines by over 3 BLEU-4. Based on controlled experiments, we fu
    
[^136]: 使用平方神经网络族的精确、快速和表达性泊松点过程

    Exact, Fast and Expressive Poisson Point Processes via Squared Neural Families

    [https://arxiv.org/abs/2402.09608](https://arxiv.org/abs/2402.09608)

    该论文介绍了使用平方神经网络族的精确、快速和表达性泊松点过程。通过利用两层神经网络的平方范数来参数化强度函数，可以获得更灵活和高效的方法。该方法在计算积分强度函数时具有封闭形式和二次时间复杂度，并且相比于传统方法更节约内存和时间。通过解决凸优化问题，可以获得对强度函数最终层的参数化重参数化的最大似然估计和最大后验估计。

    

    我们通过将强度函数的参数化为两层神经网络的平方范数引入了平方神经泊松点过程（SNEPPPs）。当隐藏层被固定且第二层只有一个神经元时，我们的方法类似于之前使用平方高斯过程或核方法，但允许隐藏层学习能够提供额外的灵活性。在许多感兴趣的情况下，积分强度函数可以得到封闭形式，并且可以以二次时间相对于隐藏神经元的数量进行计算。我们列举了比以前讨论过的更多这样的情况。我们的方法比简单实现平方或指数核方法或高斯过程更节约内存和时间。最大似然和最大后验估计可以通过解决（严格）凸优化问题来获得强度函数最终层的参数化重参数化。

    arXiv:2402.09608v1 Announce Type: new  Abstract: We introduce squared neural Poisson point processes (SNEPPPs) by parameterising the intensity function by the squared norm of a two layer neural network. When the hidden layer is fixed and the second layer has a single neuron, our approach resembles previous uses of squared Gaussian process or kernel methods, but allowing the hidden layer to be learnt allows for additional flexibility. In many cases of interest, the integrated intensity function admits a closed form and can be computed in quadratic time in the number of hidden neurons. We enumerate a far more extensive number of such cases than has previously been discussed. Our approach is more memory and time efficient than naive implementations of squared or exponentiated kernel methods or Gaussian processes. Maximum likelihood and maximum a posteriori estimates in a reparameterisation of the final layer of the intensity function can be obtained by solving a (strongly) convex optimisa
    
[^137]: 可扩展图自监督学习

    Scalable Graph Self-Supervised Learning

    [https://arxiv.org/abs/2402.09603](https://arxiv.org/abs/2402.09603)

    该论文提出了一种通过体积最大化项减少图自监督学习预训练损失函数计算成本的方法。实验证明，采用节点或维度采样可以降低损失计算的成本。

    

    在图的正则化自监督学习方法中，计算复杂度随节点数和嵌入维度的增加而增加。为了减轻非对比图自监督学习的可扩展性问题，我们提出了一种新方法，通过体积最大化项减少预训练损失函数的协方差矩阵计算成本。我们的工作重点是通过图节点或维度采样减少损失计算的成本。我们从理论上解释了为什么维度采样会导致准确的损失计算，并用数学推导支持了这种新方法。我们在节点级图预测任务上进行了实验，因为现实世界图的规模很大，所以在这方面进行自监督预训练是困难的。我们的实验表明，通过节点或维度采样可以减少损失计算的成本。

    arXiv:2402.09603v1 Announce Type: cross  Abstract: In regularization Self-Supervised Learning (SSL) methods for graphs, computational complexity increases with the number of nodes in graphs and embedding dimensions. To mitigate the scalability of non-contrastive graph SSL, we propose a novel approach to reduce the cost of computing the covariance matrix for the pre-training loss function with volume-maximization terms. Our work focuses on reducing the cost associated with the loss computation via graph node or dimension sampling. We provide theoretical insight into why dimension sampling would result in accurate loss computations and support it with mathematical derivation of the novel approach. We develop our experimental setup on the node-level graph prediction tasks, where SSL pre-training has shown to be difficult due to the large size of real world graphs. Our experiments demonstrate that the cost associated with the loss computation can be reduced via node or dimension sampling w
    
[^138]: 低秩图对比学习用于节点分类

    Low-Rank Graph Contrastive Learning for Node Classification

    [https://arxiv.org/abs/2402.09600](https://arxiv.org/abs/2402.09600)

    本研究提出了一种新颖且鲁棒的低秩图对比学习（LR-GCL）算法，应用于转导节点分类任务。该算法通过低秩正规化的对比学习训练一个编码器，并使用生成的特征进行线性转导分类。

    

    图神经网络（GNNs）广泛应用于学习节点表示，并在节点分类等各种任务中表现出色。然而，最近的研究表明，在现实世界的图数据中不可避免地存在噪声，这会严重降低GNNs的性能。在本文中，我们提出了一种新颖且鲁棒的GNN编码器，即低秩图对比学习（LR-GCL）。我们的方法通过两个步骤进行转导节点分类。首先，通过低秩正常对比学习训练一个名为LR-GCL的低秩GCL编码器。然后，使用LR-GCL生成的特征，使用线性转导分类算法对图中的未标记节点进行分类。我们的LR-GCL受到图数据和其标签的低频性质的启示，并在理论上受到我们关于转导学习的尖锐泛化界限的推动。

    arXiv:2402.09600v1 Announce Type: new  Abstract: Graph Neural Networks (GNNs) have been widely used to learn node representations and with outstanding performance on various tasks such as node classification. However, noise, which inevitably exists in real-world graph data, would considerably degrade the performance of GNNs revealed by recent studies. In this work, we propose a novel and robust GNN encoder, Low-Rank Graph Contrastive Learning (LR-GCL). Our method performs transductive node classification in two steps. First, a low-rank GCL encoder named LR-GCL is trained by prototypical contrastive learning with low-rank regularization. Next, using the features produced by LR-GCL, a linear transductive classification algorithm is used to classify the unlabeled nodes in the graph. Our LR-GCL is inspired by the low frequency property of the graph data and its labels, and it is also theoretically motivated by our sharp generalization bound for transductive learning. To the best of our kno
    
[^139]: 基于MCMC的学习

    MCMC-driven learning

    [https://arxiv.org/abs/2402.09598](https://arxiv.org/abs/2402.09598)

    这篇论文旨在统一解决MCMC和机器学习交叉领域的各种问题，包括黑盒变分推断、自适应MCMC、正规流构建和传输辅助MCMC、替代似然MCMC、大数据的MCMC核心集构建等，并提出一个通用的框架。

    

    这篇论文旨在作为《马尔科夫链蒙特卡罗手册》的一章出现。该章的目标是在马尔科夫链蒙特卡罗（MCMC）和机器学习之间的交叉点上统一各种问题，其中包括黑盒变分推断、自适应MCMC、正规流构建和传输辅助MCMC、替代似然MCMC、用于大数据的MCMC核心集构建、马尔科夫链梯度下降、马尔科夫得分攀爬等。通过这样做，可以将为每个问题开发的理论和方法进行翻译和推广。

    arXiv:2402.09598v1 Announce Type: cross  Abstract: This paper is intended to appear as a chapter for the Handbook of Markov Chain Monte Carlo. The goal of this chapter is to unify various problems at the intersection of Markov chain Monte Carlo (MCMC) and machine learning$\unicode{x2014}$which includes black-box variational inference, adaptive MCMC, normalizing flow construction and transport-assisted MCMC, surrogate-likelihood MCMC, coreset construction for MCMC with big data, Markov chain gradient descent, Markovian score climbing, and more$\unicode{x2014}$within one common framework. By doing so, the theory and methods developed for each may be translated and generalized.
    
[^140]: 基于标准血液检测结果和吸烟状况的可解释机器学习方法用于肺癌的肺病学水平检测

    Pulmonologists-Level lung cancer detection based on standard blood test results and smoking status using an explainable machine learning approach

    [https://arxiv.org/abs/2402.09596](https://arxiv.org/abs/2402.09596)

    本研究开发了一个基于动态集成选择（DES）的机器学习模型，通过利用标准血液样本分析和吸烟史数据进行肺癌的肺病学水平检测，在丹麦南部地区的大量患有风险的人群中进行了验证。模型在肺病专家提供的诊断预测方面取得了良好的效果。（摘要总结）

    

    肺癌（LC）仍然是癌症相关死亡的主要原因，主要是由于晚期诊断。因此，有效的早期检测策略非常重要。近年来，机器学习（ML）在医疗领域展示了相当大的潜力，可以帮助检测各种疾病。在这个回顾性开发和验证研究中，我们开发了一个基于动态集成选择（DES）的ML模型用于LC检测。该模型利用了来自丹麦患有风险的大量人口的标准血液样本分析和吸烟史数据。该研究包括2009年至2018年在丹麦南部地区疑似LC的所有患者。我们通过DES模型验证和比较了五位肺病专家提供的诊断预测。在38944名患者中，9940名患者有完整数据，其中2505名（25%）患有LC。DES模型的roc曲线下面积达到0.5。

    arXiv:2402.09596v1 Announce Type: new  Abstract: Lung cancer (LC) remains the primary cause of cancer-related mortality, largely due to late-stage diagnoses. Effective strategies for early detection are therefore of paramount importance. In recent years, machine learning (ML) has demonstrated considerable potential in healthcare by facilitating the detection of various diseases. In this retrospective development and validation study, we developed an ML model based on dynamic ensemble selection (DES) for LC detection. The model leverages standard blood sample analysis and smoking history data from a large population at risk in Denmark. The study includes all patients examined on suspicion of LC in the Region of Southern Denmark from 2009 to 2018. We validated and compared the predictions by the DES model with diagnoses provided by five pulmonologists. Among the 38,944 patients, 9,940 had complete data of which 2,505 (25\%) had LC. The DES model achieved an area under the roc curve of 0.
    
[^141]: 重构随机几何图的几何形状

    Reconstructing the Geometry of Random Geometric Graphs

    [https://arxiv.org/abs/2402.09591](https://arxiv.org/abs/2402.09591)

    该论文通过在底层空间中采样的图来有效地重构随机几何图的几何形状。该方法基于流形假设，即底层空间是低维流形，并且连接概率是嵌入在$\mathbb{R}^N$中的流形中点之间欧几里德距离的严格递减函数。

    

    随机几何图是在度量空间上定义的随机图模型。该模型首先从度量空间中采样点，然后以依赖于它们之间距离的概率独立地连接每对采样点。在本工作中，我们展示了如何在流形假设下有效地从采样的图中重构底层空间的几何形状，即假设底层空间是低维流形，并且连接概率是嵌入在$\mathbb{R}^N$中的流形中点之间欧几里德距离的严格递减函数。我们的工作补充了大量关于流形学习的工作，其目标是从在流形中采样的点及其（近似的）距离中恢复出流形。

    arXiv:2402.09591v1 Announce Type: new  Abstract: Random geometric graphs are random graph models defined on metric spaces. Such a model is defined by first sampling points from a metric space and then connecting each pair of sampled points with probability that depends on their distance, independently among pairs. In this work, we show how to efficiently reconstruct the geometry of the underlying space from the sampled graph under the manifold assumption, i.e., assuming that the underlying space is a low dimensional manifold and that the connection probability is a strictly decreasing function of the Euclidean distance between the points in a given embedding of the manifold in $\mathbb{R}^N$. Our work complements a large body of work on manifold learning, where the goal is to recover a manifold from sampled points sampled in the manifold along with their (approximate) distances.
    
[^142]: MLTCP:用于DNN训练的拥塞控制技术

    MLTCP: Congestion Control for DNN Training

    [https://arxiv.org/abs/2402.09589](https://arxiv.org/abs/2402.09589)

    MLTCP是一种用于加速共享GPU集群中的DNN训练作业的拥塞控制技术，通过在每个训练迭代发送的字节数进行缩放，使不同作业的流能够高效利用网络极大地加快训练作业的完成时间。

    

    我们提出了MLTCP，一种技术来增强当前的拥塞控制算法，以加速在共享GPU集群中进行的DNN训练作业。MLTCP使竞争网络带宽的作业的通信阶段相互交错，从而高效利用网络。MLTCP的核心是一个基于关键概念洞察的非常简单的原则：DNN训练流应该根据每个训练迭代发送的字节数来缩放其拥塞窗口大小。我们展示了将这个原则整合到当前的拥塞控制协议中是直接的：通过在Reno、CUBIC或DCQCN中添加30-60行代码，MLTCP可以在几个训练迭代内将不同作业的流稳定地转化为交错状态，无论竞争流的数量或每个流的开始时间如何。我们对流行的DNN训练作业进行的实验表明，启用MLTCP可以加快平均和99th pe的结束时间

    arXiv:2402.09589v1 Announce Type: cross  Abstract: We present MLTCP, a technique to augment today's congestion control algorithms to accelerate DNN training jobs in shared GPU clusters. MLTCP enables the communication phases of jobs that compete for network bandwidth to interleave with each other, thereby utilizing the network efficiently. At the heart of MLTCP lies a very simple principle based on a key conceptual insight: DNN training flows should scale their congestion window size based on the number of bytes sent at each training iteration. We show that integrating this principle into today's congestion control protocols is straightforward: by adding 30-60 lines of code to Reno, CUBIC, or DCQCN, MLTCP stabilizes flows of different jobs into an interleaved state within a few training iterations, regardless of the number of competing flows or the start time of each flow. Our experiments with popular DNN training jobs demonstrate that enabling MLTCP accelerates the average and 99th pe
    
[^143]: WERank: 针对自监督学习中的等级退化预防的权重正规化方法

    WERank: Towards Rank Degradation Prevention for Self-Supervised Learning Using Weight Regularization

    [https://arxiv.org/abs/2402.09586](https://arxiv.org/abs/2402.09586)

    本文提出了一个新的网络权重正则化方法WERank，用于防止自监督学习中的维度坍塌问题。通过实验证明了该方法的有效性，并在图像自监督学习中进行了验证。

    

    自监督学习中常见的问题是维度坍塌（也称为等级退化），其中学到的表示被映射到表示空间的低维子空间。最新的防止该问题的方法包括使用对比损失、正则化技术或架构技巧。本文提出了一种新的网络权重正则化方法WERank，用于预防网络不同层次的维度坍塌。我们通过实证和数学论证证明了该正则化方法在防止维度坍塌方面的有效性。我们还验证了在图像自监督学习中WERank的影响，由于缺乏适当的数据增强，维度坍塌更加明显。

    arXiv:2402.09586v1 Announce Type: new  Abstract: A common phenomena confining the representation quality in Self-Supervised Learning (SSL) is dimensional collapse (also known as rank degeneration), where the learned representations are mapped to a low dimensional subspace of the representation space. The State-of-the-Art SSL methods have shown to suffer from dimensional collapse and fall behind maintaining full rank. Recent approaches to prevent this problem have proposed using contrastive losses, regularization techniques, or architectural tricks. We propose WERank, a new regularizer on the weight parameters of the network to prevent rank degeneration at different layers of the network. We provide empirical evidence and mathematical justification to demonstrate the effectiveness of the proposed regularization method in preventing dimensional collapse. We verify the impact of WERank on graph SSL where dimensional collapse is more pronounced due to the lack of proper data augmentation. 
    
[^144]: 基于机器学习的无线定位中的复杂度降低：最小描述特征

    Complexity Reduction in Machine Learning-Based Wireless Positioning: Minimum Description Features

    [https://arxiv.org/abs/2402.09580](https://arxiv.org/abs/2402.09580)

    本文设计了一种定位神经网络（P-NN），通过最小描述特征降低了基于深度学习的无线定位中的复杂度，并开发了一种新的方法来自适应地选择特征空间的大小。

    

    最近的一系列研究一直致力于基于深度学习的无线定位（WP）。尽管这些WP算法在不同信道条件下表现出了高精度和鲁棒性，但它们也存在一个主要缺点：它们需要处理高维特征，这对于移动应用来说可能是禁止的。在本工作中，我们设计了一个定位神经网络（P-NN），通过精心设计的最小描述特征，大大降低了基于深度学习的WP的复杂度。我们的特征选择基于最大功率测量及其时间位置，以传达进行WP所需的信息。我们还开发了一种新的方法来自适应地选择特征空间的大小，该方法通过在信号二进制选择上使用信息论度量，优化了期望有用信息量和分类能力之间的平衡。

    arXiv:2402.09580v1 Announce Type: new  Abstract: A recent line of research has been investigating deep learning approaches to wireless positioning (WP). Although these WP algorithms have demonstrated high accuracy and robust performance against diverse channel conditions, they also have a major drawback: they require processing high-dimensional features, which can be prohibitive for mobile applications. In this work, we design a positioning neural network (P-NN) that substantially reduces the complexity of deep learning-based WP through carefully crafted minimum description features. Our feature selection is based on maximum power measurements and their temporal locations to convey information needed to conduct WP. We also develop a novel methodology for adaptively selecting the size of feature space, which optimizes over balancing the expected amount of useful information and classification capability, quantified using information-theoretic measures on the signal bin selection. Numeri
    
[^145]: 蝴蝶引起的变化：利用群体储备转换器进行远见预测

    Changes by Butterflies: Farsighted Forecasting with Group Reservoir Transformer

    [https://arxiv.org/abs/2402.09573](https://arxiv.org/abs/2402.09573)

    我们提出了一种群体储备转换器的架构，通过解决历史序列的挑战和初始条件的敏感性，实现更准确、更稳健地预测长期事件。在多元时间序列中，我们的模型相比最先进的DNN模型表现出更高的准确率，最高可减少89.43%的误差。

    

    在混沌中，两个初始条件之间的微小差异会随着时间的推移呈指数级放大，导致遥远的结果，也被称为蝴蝶效应。因此，远期充满了不确定性，难以预测。我们引入了群体储备转换器来通过克服混沌中的两个挑战（1）大量的历史序列和（2）对初始条件的敏感性来更准确、更稳健地预测长期事件。将一个储备装置连接到转换器上以高效地处理任意长度的历史数据，并通过扩展一组储备装置来减少由于初始化变化而产生的不确定性。我们的架构在多元时间序列中始终优于最先进的DNN模型，包括NLinear、Pyformer、Informer、Autoformer和基准Transformer，其误差减少高达-89.43％，适用于ETTh、ETTm和空气质量等各个领域。

    arXiv:2402.09573v1 Announce Type: cross  Abstract: In Chaos, a minor divergence between two initial conditions exhibits exponential amplification over time, leading to far-away outcomes, known as the butterfly effect. Thus, the distant future is full of uncertainty and hard to forecast. We introduce Group Reservoir Transformer to predict long-term events more accurately and robustly by overcoming two challenges in Chaos: (1) the extensive historical sequences and (2) the sensitivity to initial conditions. A reservoir is attached to a Transformer to efficiently handle arbitrarily long historical lengths, with an extension of a group of reservoirs to reduce the uncertainty due to the initialization variations. Our architecture consistently outperforms state-of-the-art DNN models in multivariate time series, including NLinear, Pyformer, Informer, Autoformer, and the baseline Transformer, with an error reduction of up to -89.43\% in various fields such as ETTh, ETTm, and air quality, demon
    
[^146]: Neyman-Pearson分类中的无分布率

    Distribution-Free Rates in Neyman-Pearson Classification

    [https://arxiv.org/abs/2402.09560](https://arxiv.org/abs/2402.09560)

    该论文提供了一个关于Neyman-Pearson分类中无分布率的完整特征，通过简单的几何条件，即三点分离条件，刻画了硬分类器和简单分类器之间的二分条件。

    

    我们考虑Neyman-Pearson分类问题，该问题模拟了不平衡分类设置，在这种设置中，最小化与分布$\mu_1$相关的错误，同时保证与另一个分布$\mu_0$相关的错误较低。给定一个固定的VC分类器类$\mathcal{H}$，我们提供了可能的无分布率的完整特征，即所有配对$(\mu_0, \mu_1)$的极小化率。这些速率涉及到了硬分类器和简单分类器之间的二分条件，它们是根据一个简单的几何条件，即三点分离条件来刻画的，与VC维度略有关联。

    arXiv:2402.09560v1 Announce Type: new  Abstract: We consider the problem of Neyman-Pearson classification which models unbalanced classification settings where error w.r.t. a distribution $\mu_1$ is to be minimized subject to low error w.r.t. a different distribution $\mu_0$. Given a fixed VC class $\mathcal{H}$ of classifiers to be minimized over, we provide a full characterization of possible distribution-free rates, i.e., minimax rates over the space of all pairs $(\mu_0, \mu_1)$. The rates involve a dichotomy between hard and easy classes $\mathcal{H}$ as characterized by a simple geometric condition, a three-points-separation condition, loosely related to VC dimension.
    
[^147]: 提高时间序列表示学习的双向生成预训练模型

    Bidirectional Generative Pre-training for Improving Time Series Representation Learning

    [https://arxiv.org/abs/2402.09558](https://arxiv.org/abs/2402.09558)

    这项论文提出了一种名为BiTimelyGPT的模型，通过双向的预训练任务在时间序列数据上学习表示，展示了优越的性能，可用于神经功能预测、疾病诊断和生理病征识别。

    

    学习时间序列表示以用于判别任务一直是一项长期的挑战。当前的预训练方法要么是单向的下一个标记预测，要么是随机屏蔽标记预测。我们提出了一种新颖的架构，称为双向及时生成预训练Transformer（BiTimelyGPT），它通过交替的Transformer层在时间序列数据上进行了下一个标记和上一个标记的预测。这种预训练任务保留了时间序列的原始分布和数据形状。此外，全秩前向和后向注意力矩阵具有更具表现力的表示能力。 使用生物信号数据，BiTimelyGPT在预测神经功能、疾病诊断和生理病征方面表现出了优越性能。通过可视化注意力热图，我们观察到预训练的BiTimelyGPT能够从时间序列中识别出具有判别性的片段。

    arXiv:2402.09558v1 Announce Type: new  Abstract: Learning time-series representations for discriminative tasks has been a long-standing challenge. Current pre-training methods are limited in either unidirectional next-token prediction or randomly masked token prediction. We propose a novel architecture called Bidirectional Timely Generative Pre-trained Transformer (BiTimelyGPT), which pre-trains on time-series data by both next-token and previous-token predictions in alternating transformer layers. This pre-training task preserves original distribution and data shapes of the time-series. Additionally, the full-rank forward and backward attention matrices exhibit more expressive representation capabilities. Using biosignal data, BiTimelyGPT demonstrates superior performance in predicting neurological functionality, disease diagnosis, and physiological signs. By visualizing the attention heatmap, we observe that the pre-trained BiTimelyGPT can identify discriminative segments from time-s
    
[^148]: 提高离线策略学习的数据集聚类

    Dataset Clustering for Improved Offline Policy Learning

    [https://arxiv.org/abs/2402.09550](https://arxiv.org/abs/2402.09550)

    本文研究了一种名为多行为的数据集特征，提出了一种行为感知的深层聚类方法，将多行为数据集划分为若干单一行为子集，从而提高了离线策略学习的性能。

    

    离线策略学习旨在从先前收集的数据集中发现决策制定策略，而无需与环境进行额外的在线交互。由于训练数据集是固定的，其质量成为影响学习策略性能的关键因素。本文研究了一种我们称之为多行为的数据集特征，指示数据集是使用展现不同行为的多个策略收集的。相反，一个单一行为的数据集将仅使用一个策略收集。我们观察到，从单一行为数据集学习的策略通常比从多行为数据集学习的策略表现更好，尽管单一行为数据集拥有更少的示例和较低的多样性。因此，我们提出了一种行为感知的深层聚类方法，将多行为数据集划分为若干单一行为子集，从而有助于下游策略学习。

    arXiv:2402.09550v1 Announce Type: new  Abstract: Offline policy learning aims to discover decision-making policies from previously-collected datasets without additional online interactions with the environment. As the training dataset is fixed, its quality becomes a crucial determining factor in the performance of the learned policy. This paper studies a dataset characteristic that we refer to as multi-behavior, indicating that the dataset is collected using multiple policies that exhibit distinct behaviors. In contrast, a uni-behavior dataset would be collected solely using one policy. We observed that policies learned from a uni-behavior dataset typically outperform those learned from multi-behavior datasets, despite the uni-behavior dataset having fewer examples and less diversity. Therefore, we propose a behavior-aware deep clustering approach that partitions multi-behavior datasets into several uni-behavior subsets, thereby benefiting downstream policy learning. Our approach is fl
    
[^149]: 逐层近端回放：一种在线连续学习的近端点方法

    Layerwise Proximal Replay: A Proximal Point Method for Online Continual Learning

    [https://arxiv.org/abs/2402.09542](https://arxiv.org/abs/2402.09542)

    这项工作针对在线连续学习中经验回放造成的优化不稳定问题进行了改进，提出了一种逐层近端回放（LPR）方法，通过优化几何的修改来平衡新数据和回放数据的学习，从而改善了回放式在线连续学习方法的准确性。

    

    在在线连续学习中，神经网络逐步从非独立同分布的数据流中学习。几乎所有的在线连续学习方法都使用经验回放来同时防止灾难性遗忘和过度拟合先前的数据。我们的工作展示了这种方法的一个局限性：使用经验回放训练的网络往往具有不稳定的优化轨迹，影响其整体准确度。令人惊讶的是，即使回放缓冲区存储了所有先前的训练样本，这些不稳定性仍然存在，这表明这个问题与灾难性遗忘是无关的。我们通过对优化几何的简单修改来最小化这些不稳定性。我们的解决方案，逐层近端回放（LPR），在只允许逐渐改变过去数据的隐藏激活的同时，平衡了从新数据和回放数据中的学习。我们证明了LPR在基于回放的在线连续学习方法中持续改进。

    arXiv:2402.09542v1 Announce Type: new  Abstract: In online continual learning, a neural network incrementally learns from a non-i.i.d. data stream. Nearly all online continual learning methods employ experience replay to simultaneously prevent catastrophic forgetting and underfitting on past data. Our work demonstrates a limitation of this approach: networks trained with experience replay tend to have unstable optimization trajectories, impeding their overall accuracy. Surprisingly, these instabilities persist even when the replay buffer stores all previous training examples, suggesting that this issue is orthogonal to catastrophic forgetting. We minimize these instabilities through a simple modification of the optimization geometry. Our solution, Layerwise Proximal Replay (LPR), balances learning from new and replay data while only allowing for gradual changes in the hidden activation of past data. We demonstrate that LPR consistently improves replay-based online continual learning me
    
[^150]: 为什么具有较大ε的差分隐私可以防御实际成员推理攻击？

    Why Does Differential Privacy with Large Epsilon Defend Against Practical Membership Inference Attacks?

    [https://arxiv.org/abs/2402.09540](https://arxiv.org/abs/2402.09540)

    本论文研究了为什么具有较大ε的差分隐私可以防御实际成员推理攻击，因为实际攻击者可能缺乏准确的私有数据知识，并且在实际应用中，数据集可能相对容易被防御。

    

    对于较小的隐私参数ε，ε-差分隐私（DP）提供了一个强大的最坏情况保证，即没有成员推理攻击（MIA）能够成功确定一个人的数据是否被用于训练机器学习模型。DP的保证是最坏情况下的，因为：a）即使攻击者已经知道数据集中除一个人的记录之外的所有记录；b）它在所有数据集上均匀适用。在实际应用中，这样的最坏情况保证可能过于严格：实际攻击者可能缺乏（几乎所有）私有数据的精确知识，并且我们的数据集可能在某种意义上比最坏情况的数据集更容易被防御。这些考虑推动了具有大的隐私参数（例如ε≥7）的DP模型的工业部署，并且经验上观察到具有大ε的DP可以成功防御最先进的MIA。现有的DP模型研究一般集中于小ε，因此尚不清楚为什么具有较大ε的DP可以防御实际成员推理攻击。

    arXiv:2402.09540v1 Announce Type: cross  Abstract: For small privacy parameter $\epsilon$, $\epsilon$-differential privacy (DP) provides a strong worst-case guarantee that no membership inference attack (MIA) can succeed at determining whether a person's data was used to train a machine learning model. The guarantee of DP is worst-case because: a) it holds even if the attacker already knows the records of all but one person in the data set; and b) it holds uniformly over all data sets. In practical applications, such a worst-case guarantee may be overkill: practical attackers may lack exact knowledge of (nearly all of) the private data, and our data set might be easier to defend, in some sense, than the worst-case data set. Such considerations have motivated the industrial deployment of DP models with large privacy parameter (e.g. $\epsilon \geq 7$), and it has been observed empirically that DP with large $\epsilon$ can successfully defend against state-of-the-art MIAs. Existing DP the
    
[^151]: 流形密度函数：用于验证流形学习的内在方法

    The Manifold Density Function: An Intrinsic Method for the Validation of Manifold Learning

    [https://arxiv.org/abs/2402.09529](https://arxiv.org/abs/2402.09529)

    我们提出了一种流形密度函数的内在方法，用于验证流形学习技术，能够适应各种黎曼流形，并证明了其收敛性和鲁棒性。

    

    我们引入了流形密度函数，这是一种用于验证流形学习技术的内在方法。我们的方法是基于Ripley的K函数，并在无监督设置中将流形学习算法的输出与潜在流形的结构进行匹配。我们的流形密度函数适用于广泛的黎曼流形类别。特别地，我们使用高斯-博内定理将流形密度函数推广到了一般的二维流形，并证明了对于超曲面，可以使用第一个拉普拉斯特征值来近似流形密度函数。我们证明了理想的收敛性和鲁棒性属性。

    arXiv:2402.09529v1 Announce Type: new  Abstract: We introduce the manifold density function, which is an intrinsic method to validate manifold learning techniques. Our approach adapts and extends Ripley's $K$-function, and categorizes in an unsupervised setting the extent to which an output of a manifold learning algorithm captures the structure of a latent manifold. Our manifold density function generalizes to broad classes of Riemannian manifolds. In particular, we extend the manifold density function to general two-manifolds using the Gauss-Bonnet theorem, and demonstrate that the manifold density function for hypersurfaces is well approximated using the first Laplacian eigenvalue. We prove desirable convergence and robustness properties.
    
[^152]: Higgs鉴别的引导量子压缩

    Guided Quantum Compression for Higgs Identification

    [https://arxiv.org/abs/2402.09524](https://arxiv.org/abs/2402.09524)

    Higgs鉴别的引导量子压缩模型将预处理和量子分类算法统一为可训练模型，解决了量子机器学习中使用自动编码器导致分类性能降低的问题，能够有效鉴别LHC中的希格斯玻色子。

    

    arXiv：2402.09524v1 公告类型：交叉摘要：量子机器学习提供了一种基本新颖且有前景的数据分析方法。然而，许多数据集对当前可用的量子计算机来说过于复杂。因此，传统上，量子机器学习应用通过使用降维算法（如自动编码器）在通过量子模型之前对数据进行预处理。我们展示了使用经典自动编码器作为独立的预处理步骤可以显著降低量子机器学习算法的分类性能。为了改善这个问题，我们设计了一种将预处理和量子分类算法统一到单个可训练模型中的架构：引导量子压缩模型。通过使用该模型在LHC的质子-质子碰撞中鉴别希格斯玻色子的实用性得到了证明，而传统方法则无效。

    arXiv:2402.09524v1 Announce Type: cross  Abstract: Quantum machine learning provides a fundamentally novel and promising approach to analyzing data. However, many data sets are too complex for currently available quantum computers. Consequently, quantum machine learning applications conventionally resort to dimensionality reduction algorithms, e.g., auto-encoders, before passing data through the quantum models. We show that using a classical auto-encoder as an independent preprocessing step can significantly decrease the classification performance of a quantum machine learning algorithm. To ameliorate this issue, we design an architecture that unifies the preprocessing and quantum classification algorithms into a single trainable model: the guided quantum compression model. The utility of this model is demonstrated by using it to identify the Higgs boson in proton-proton collisions at the LHC, where the conventional approach proves ineffective. Conversely, the guided quantum compressio
    
[^153]: 安全代码生成的指令调优

    Instruction Tuning for Secure Code Generation

    [https://arxiv.org/abs/2402.09497](https://arxiv.org/abs/2402.09497)

    现代语言模型在编程中得到广泛应用，指令调优是一个增强其实用性的关键过程。然而，现有的方案忽视了生成代码的安全性。本文提出了SafeCoder，通过安全微调和标准指令调优相结合，来优化安全性和实用性。

    

    现代语言模型(LMs)在日常和专业环境中得到了广泛的认可，尤其在编程中。指令调优是一种关键的过程，通过训练LMs遵循用户指令和人类偏好，从而大大增强了LMs的实用性。然而，现有的指令调优方案忽视了一个关键方面：生成代码的安全性。因此，即使是最先进的指令调优的LMs也经常产生不安全的代码，带来了重大的安全风险。在这项工作中，我们引入了SafeCoder来填补这个差距。SafeCoder使用一个多样化和高质量的数据集进行安全为中心的微调，我们使用自动化流水线收集了这个数据集。我们将安全微调与标准的指令调优相结合，以便同时优化安全性和实用性。尽管简单，但我们展示了SafeCoder的有效性。

    arXiv:2402.09497v1 Announce Type: cross  Abstract: Modern language models (LMs) have gained widespread acceptance in everyday and professional contexts, particularly in programming. An essential procedure enabling this adoption is instruction tuning, which substantially enhances LMs' practical utility by training them to follow user instructions and human preferences. However, existing instruction tuning schemes overlook a crucial aspect: the security of generated code. As a result, even the state-of-the-art instruction-tuned LMs frequently produce unsafe code, posing significant security risks. In this work, we introduce SafeCoder to address this gap. SafeCoder performs security-centric fine-tuning using a diverse and high-quality dataset that we collected using an automated pipeline. We integrate the security fine-tuning with standard instruction tuning, to facilitate a joint optimization of both security and utility. Despite its simplicity, we show that SafeCoder is effective across
    
[^154]: 关于基于网络特征在欺诈检测中潜力的研究

    On the Potential of Network-Based Features for Fraud Detection

    [https://arxiv.org/abs/2402.09495](https://arxiv.org/abs/2402.09495)

    本文研究了基于网络特征在欺诈检测中的潜力，通过使用个性化的PageRank算法来捕捉欺诈的社会动态。实验结果表明，集成PPR可以提高模型的预测能力并提供独特有价值的信息。

    

    在线交易欺诈给企业和消费者带来了重大挑战，面临着重大的经济损失。传统的基于规则的系统难以跟上欺诈战术的演变，导致高误报率和漏报率。机器学习技术通过利用历史数据识别欺诈模式提供了一个有希望的解决方案。本文探讨使用个性化的PageRank（PPR）算法通过分析金融账户之间的关系来捕捉欺诈的社会动态。主要目标是比较传统特征与添加PPR在欺诈检测模型中的性能。结果表明，集成PPR可以提高模型的预测能力，超过基准模型。此外，PPR特征提供了独特而有价值的信息，通过其高特征重要性得分得以证明。特征稳定性分析证实了一致的结果。

    arXiv:2402.09495v1 Announce Type: cross  Abstract: Online transaction fraud presents substantial challenges to businesses and consumers, risking significant financial losses. Conventional rule-based systems struggle to keep pace with evolving fraud tactics, leading to high false positive rates and missed detections. Machine learning techniques offer a promising solution by leveraging historical data to identify fraudulent patterns. This article explores using the personalised PageRank (PPR) algorithm to capture the social dynamics of fraud by analysing relationships between financial accounts. The primary objective is to compare the performance of traditional features with the addition of PPR in fraud detection models. Results indicate that integrating PPR enhances the model's predictive power, surpassing the baseline model. Additionally, the PPR feature provides unique and valuable information, evidenced by its high feature importance score. Feature stability analysis confirms consist
    
[^155]: PMGDA: 基于偏好的多梯度下降算法

    PMGDA: A Preference-based Multiple Gradient Descent Algorithm

    [https://arxiv.org/abs/2402.09492](https://arxiv.org/abs/2402.09492)

    PMGDA是一个基于偏好的多梯度下降算法，可以 efficiently 在多目标机器学习应用中找到与决策者偏好完全匹配的帕累托最优解。

    

    针对多目标机器学习应用中的问题，如多任务学习和多目标强化学习，寻找与决策者给定偏好完全匹配的帕累托最优解是非常重要的。然而，这些问题通常规模较大，虽然有可用的梯度信息，但现有的算法无法很好地处理。为了解决这个关键问题，本文提出了一种新颖的“预测-校正”框架，用于找到决策者所需的精确帕累托最优解。在该框架中，引入了一个约束函数来在搜索过程中使解与用户特定偏好对齐，这个约束函数可以与多个目标函数同时优化。实验结果表明，我们提出的方法可以有效地找到标准基准测试、多任务和多目标强化学习问题的精确帕累托最优解。

    arXiv:2402.09492v1 Announce Type: new  Abstract: It is desirable in many multi-objective machine learning applications, such as multi-task learning and multi-objective reinforcement learning, to find a Pareto optimal solution that can exactly match a given preference of decision-makers. These problems are often large-scale with available gradient information but cannot be handled very well by the existing algorithms. To tackle this critical issue, this paper proposes a novel predict-and-correct framework for locating the exact Pareto optimal solutions required by a decision maker. In the proposed framework, a constraint function is introduced in the search progress to align the solution with a user-specific preference, which can be optimized simultaneously with multiple objective functions. Experimental results show that our proposed method can efficiently find exact Pareto optimal solutions for standard benchmarks, multi-task, and multi-objective reinforcement learning problems with m
    
[^156]: 基于物联网和机器学习的智能农业温室控制系统

    Intelligent Agricultural Greenhouse Control System Based on Internet of Things and Machine Learning

    [https://arxiv.org/abs/2402.09488](https://arxiv.org/abs/2402.09488)

    这项研究提出了一种基于物联网和机器学习的智能农业温室控制系统，通过监测和调控温室内环境条件，提高作物生长效率和产量，减少资源浪费。

    

    本研究试图将物联网和机器学习相结合，构建一个先进的农业温室控制系统。通过对温室内固有环境参数的细致监测和机器学习算法的整合，能够适当调控温室内的条件。预期的结果是增加作物生长效率和产量，同时减少资源浪费。在全球人口持续增长和气候变化不断加剧的背景下，农业面临前所未有的挑战。传统农业范式已经被证明无法满足食品安全和生产效率的要求。在这种背景下，温室农业成为一种可行的解决方案，为作物种植提供了一个受控的环境来增加产量，改善品质。

    arXiv:2402.09488v1 Announce Type: cross  Abstract: This study endeavors to conceptualize and execute a sophisticated agricultural greenhouse control system grounded in the amalgamation of the Internet of Things (IoT) and machine learning. Through meticulous monitoring of intrinsic environmental parameters within the greenhouse and the integration of machine learning algorithms, the conditions within the greenhouse are aptly modulated. The envisaged outcome is an enhancement in crop growth efficiency and yield, accompanied by a reduction in resource wastage. In the backdrop of escalating global population figures and the escalating exigencies of climate change, agriculture confronts unprecedented challenges. Conventional agricultural paradigms have proven inadequate in addressing the imperatives of food safety and production efficiency. Against this backdrop, greenhouse agriculture emerges as a viable solution, proffering a controlled milieu for crop cultivation to augment yields, refin
    
[^157]: UMOEA/D：基于分解的均匀 Pareto 目标的多目标进化算法

    UMOEA/D: A Multiobjective Evolutionary Algorithm for Uniform Pareto Objectives based on Decomposition

    [https://arxiv.org/abs/2402.09486](https://arxiv.org/abs/2402.09486)

    该论文提出了一种多目标进化算法UMOEA/D来构建均匀分布的Pareto目标，以解决先前多目标优化方法中有限多样性的问题。

    

    多目标优化在许多应用中很常见，其中构建 Pareto 前沿（PF）以显示各种偏好下的最优解。以往的方法通常利用 Pareto 目标集（PF 上的粒子）来表示整个 PF。然而，很少有研究探讨 PF 上 Pareto 目标的经验分布，这隐含地限制了先前方法中多样性和代表性 Pareto 目标的生成。为了弥补这个差距，我们在本文中提出构建 PF 上“均匀分布”的 Pareto 目标，以减轻先前多目标优化方法中的有限多样性问题。我们是第一个正式定义多目标优化问题中“均匀性”的研究者。我们使用神经网络优化 Pareto 前沿上的最大最小距离，得到渐近和非渐近均匀 Pareto 目标。我们提出的方法得到了验证。

    arXiv:2402.09486v1 Announce Type: new  Abstract: Multiobjective optimization (MOO) is prevalent in numerous applications, in which a Pareto front (PF) is constructed to display optima under various preferences. Previous methods commonly utilize the set of Pareto objectives (particles on the PF) to represent the entire PF. However, the empirical distribution of the Pareto objectives on the PF is rarely studied, which implicitly impedes the generation of diverse and representative Pareto objectives in previous methods. To bridge the gap, we suggest in this paper constructing \emph{uniformly distributed} Pareto objectives on the PF, so as to alleviate the limited diversity found in previous MOO approaches. We are the first to formally define the concept of ``uniformity" for an MOO problem. We optimize the maximal minimal distances on the Pareto front using a neural network, resulting in both asymptotically and non-asymptotically uniform Pareto objectives. Our proposed method is validated 
    
[^158]: 具有公共数据的Oracle-Efficient差分隐私学习

    Oracle-Efficient Differentially Private Learning with Public Data

    [https://arxiv.org/abs/2402.09483](https://arxiv.org/abs/2402.09483)

    这项研究提出了一种具有公共数据的计算高效算法，可以在满足差分隐私条件的情况下学习私有数据，以提高学习算法性能。

    

    由于在隐私约束下许多函数类的可学习性的统计下限，最近出现了利用公共数据提高私有学习算法性能的兴趣。在这种模型中，算法必须始终保证相对于私有样本的差分隐私，并在私有数据分布与公共数据分布足够接近时确保学习保证。先前的研究表明，当有足够的公共非标记数据时，可以使私有学习在统计上可以处理，但得到的算法都是计算效率低下的。在这项工作中，我们提出了第一种可计算高效的算法，可以在函数类可非私有学习时明确利用公共数据进行私有学习，其中我们对计算效率的概念是相对于优化调用次数的。

    arXiv:2402.09483v1 Announce Type: cross  Abstract: Due to statistical lower bounds on the learnability of many function classes under privacy constraints, there has been recent interest in leveraging public data to improve the performance of private learning algorithms. In this model, algorithms must always guarantee differential privacy with respect to the private samples while also ensuring learning guarantees when the private data distribution is sufficiently close to that of the public data. Previous work has demonstrated that when sufficient public, unlabelled data is available, private learning can be made statistically tractable, but the resulting algorithms have all been computationally inefficient. In this work, we present the first computationally efficient, algorithms to provably leverage public data to learn privately whenever a function class is learnable non-privately, where our notion of computational efficiency is with respect to the number of calls to an optimization o
    
[^159]: 数据重构攻击与防御：一个系统评估

    Data Reconstruction Attacks and Defenses: A Systematic Evaluation

    [https://arxiv.org/abs/2402.09478](https://arxiv.org/abs/2402.09478)

    本研究提出了一种在联合学习环境中的强力重构攻击，可以重构中间特征，并且对大部分先前的方法表现更好。实证研究表明，在防御机制中，梯度修剪是对抗最先进攻击最有效的策略。

    

    重构攻击和防御对于理解机器学习中的数据泄漏问题至关重要。然而，先前的工作主要集中在梯度反转攻击的经验观察上，缺乏理论基础，并且无法区分防御方法的有用性与攻击方法的计算限制。在这项工作中，我们提出了一种在联合学习环境中的强力重构攻击。该攻击可以重构中间特征，并与大部分先前的方法相比表现更好。在这种更强的攻击下，我们从理论和实证两方面全面调查了最常见的防御方法的效果。我们的研究结果表明，在各种防御机制中，如梯度剪辑、dropout、添加噪音、局部聚合等等，梯度修剪是对抗最先进攻击最有效的策略。

    arXiv:2402.09478v1 Announce Type: cross  Abstract: Reconstruction attacks and defenses are essential in understanding the data leakage problem in machine learning. However, prior work has centered around empirical observations of gradient inversion attacks, lacks theoretical groundings, and was unable to disentangle the usefulness of defending methods versus the computational limitation of attacking methods. In this work, we propose a strong reconstruction attack in the setting of federated learning. The attack reconstructs intermediate features and nicely integrates with and outperforms most of the previous methods. On this stronger attack, we thoroughly investigate both theoretically and empirically the effect of the most common defense methods. Our findings suggest that among various defense mechanisms, such as gradient clipping, dropout, additive noise, local aggregation, etc., gradient pruning emerges as the most effective strategy to defend against state-of-the-art attacks.
    
[^160]: PANORAMIA: 无需重新训练的机器学习模型隐私审计

    PANORAMIA: Privacy Auditing of Machine Learning Models without Retraining

    [https://arxiv.org/abs/2402.09477](https://arxiv.org/abs/2402.09477)

    PANORAMIA是一种无需重新训练的机器学习模型隐私审计方案，通过使用生成的“非成员”数据进行成员推断攻击，可以量化大规模ML模型的隐私泄露，而无需控制训练过程或重新训练模型，只需要访问训练数据的子集。

    

    我们引入了一种隐私审计方案，该方案依赖于使用生成的“非成员”数据进行成员推断攻击来对ML模型进行隐私审计。这个方案被称为PANORAMIA，它可以量化大规模ML模型的隐私泄露，而无需控制训练过程或重新训练模型，只需要访问训练数据的子集。为了证明其适用性，我们在多个ML领域进行了审计，包括图像和表格数据分类以及大规模语言模型。

    arXiv:2402.09477v1 Announce Type: cross  Abstract: We introduce a privacy auditing scheme for ML models that relies on membership inference attacks using generated data as "non-members". This scheme, which we call PANORAMIA, quantifies the privacy leakage for large-scale ML models without control of the training process or model re-training and only requires access to a subset of the training data. To demonstrate its applicability, we evaluate our auditing scheme across multiple ML domains, ranging from image and tabular data classification to large-scale language models.
    
[^161]: 解读心率信号：一种基于视觉变压器技术的可解释性房颤检测方法

    Deciphering Heartbeat Signatures: A Vision Transformer Approach to Explainable Atrial Fibrillation Detection from ECG Signals

    [https://arxiv.org/abs/2402.09474](https://arxiv.org/abs/2402.09474)

    本研究使用视觉变压器方法解读心率信号，提高心脏疾病检测模型的解释性和可靠性。

    

    基于可穿戴单导联心电图（ECG）设备的远程患者监测在结合人工智能（AI）方法进行自动心脏疾病检测方面具有巨大潜力。先前的研究已经应用基于深度学习的AI方法进行心脏疾病检测，但由于目前AI算法的黑盒特性，这些模型尚未被广泛接受作为临床诊断的可靠辅助工具。尤其需要确定ECG信号中贡献于准确诊断的关键特征，从而提高模型的可解释性。本研究开发了一种基于视觉变压器的方法，通过单导联ECG数据识别房颤，并提出了一种残差网络（ResNet）方法以作对比。

    arXiv:2402.09474v1 Announce Type: cross  Abstract: Remote patient monitoring based on wearable single-lead electrocardiogram (ECG) devices has significant potential for enabling the early detection of heart disease, especially in combination with artificial intelligence (AI) approaches for automated heart disease detection. There have been prior studies applying AI approaches based on deep learning for heart disease detection. However, these models are yet to be widely accepted as a reliable aid for clinical diagnostics, in part due to the current black-box perception surrounding many AI algorithms. In particular, there is a need to identify the key features of the ECG signal that contribute toward making an accurate diagnosis, thereby enhancing the interpretability of the model. In the present study, we develop a vision transformer approach to identify atrial fibrillation based on single-lead ECG data. A residual network (ResNet) approach is also developed for comparison with the visi
    
[^162]: 列生成的一对多反事实解释

    One-for-many Counterfactual Explanations by Column Generation

    [https://arxiv.org/abs/2402.09473](https://arxiv.org/abs/2402.09473)

    本文提出了一个列生成框架，用于解决一对多反事实解释的问题。框架通过限制每个解释中可集体改变的特征数量，旨在尽可能少地使用解释来解释所有实例。相比于现有的混合整数规划方法，该框架在可扩展性、计算性能和解决方案质量方面具有优势。

    

    在本文中，我们考虑了一个问题，即如何生成一组针对一组实例的反事实解释，采用一对多分配规则，其中一个解释被分配给一个实例子组。我们首次解决了在考虑稀疏性的情况下最小化解释所需数量的问题，通过限制每个解释中允许集体改变的特征数量。我们开发了一个新颖的列生成框架，用于高效搜索解释。我们的框架可以应用于任何黑盒分类器，如神经网络。与文献中的简单混合整数规划公式的简单适应相比，列生成框架在可扩展性、计算性能和解决方案的质量方面占优势。

    arXiv:2402.09473v1 Announce Type: new  Abstract: In this paper, we consider the problem of generating a set of counterfactual explanations for a group of instances, with the one-for-many allocation rule, where one explanation is allocated to a subgroup of the instances. For the first time, we solve the problem of minimizing the number of explanations needed to explain all the instances, while considering sparsity by limiting the number of features allowed to be changed collectively in each explanation. A novel column generation framework is developed to efficiently search for the explanations. Our framework can be applied to any black-box classifier, like neural networks. Compared with a simple adaptation of a mixed-integer programming formulation from the literature, the column generation framework dominates in terms of scalability, computational performance and quality of the solutions.
    
[^163]: 用于随机参数化的机器学习

    Machine Learning for Stochastic Parametrisation

    [https://arxiv.org/abs/2402.09471](https://arxiv.org/abs/2402.09471)

    本文介绍了使用机器学习技术进行随机参数化的方法，旨在改善天气和气候预测模型的准确性和速度。

    

    用于天气和气候预测的大气模型通常以确定性方式构建。换句话说，在给定已解析尺度变量的特定状态下，估计并使用子网格尺度过程的最可能强迫项来预测大尺度流的演化。然而，大气中缺乏尺度分离意味着这种方法在预测中是一个很大的误差来源。近年来，出现了一种替代模式：使用随机技术来表征小尺度过程的不确定性。这些技术现在广泛应用于天气、次季节、季节和气候时间尺度上。与此同时，近年来也取得了使用机器学习（ML）替代参数化方案的重大进展。这有潜力加快并改进我们的数值模型。然而，目前的重点主要是确定性的。

    arXiv:2402.09471v1 Announce Type: new  Abstract: Atmospheric models used for weather and climate prediction are traditionally formulated in a deterministic manner. In other words, given a particular state of the resolved scale variables, the most likely forcing from the sub-grid scale processes is estimated and used to predict the evolution of the large-scale flow. However, the lack of scale-separation in the atmosphere means that this approach is a large source of error in forecasts. Over recent years, an alternative paradigm has developed: the use of stochastic techniques to characterise uncertainty in small-scale processes. These techniques are now widely used across weather, sub-seasonal, seasonal, and climate timescales. In parallel, recent years have also seen significant progress in replacing parametrisation schemes using machine learning (ML). This has the potential to both speed up and improve our numerical models. However, the focus to date has largely been on deterministic a
    
[^164]: 滚动扩散模型

    Rolling Diffusion Models

    [https://arxiv.org/abs/2402.09470](https://arxiv.org/abs/2402.09470)

    本文介绍了一种滚动扩散模型，用于处理时间数据，通过滑动窗口去噪并根据帧在序列中的时间先后分配不同的噪声量，更好地捕捉到复杂的时间动态。通过实验证明，在视频预测和混沌流体动力学预测任务中，该模型优于传统扩散方法。

    

    最近，扩散模型越来越多地应用于时间数据，如视频、流体力学模拟或气候数据。这些方法通常将后续帧在扩散过程中的噪声量视为相等。本文探讨了滚动扩散：一种使用滑动窗口去噪的新方法。它确保扩散过程逐渐通过时间进行破坏，通过将更多的噪声分配给序列中出现较晚的帧，反映出随着生成过程的展开，对未来的不确定性越来越大。通过实证研究，我们表明当时间动态复杂时，滚动扩散优于标准扩散。特别是在使用Kinetics-600视频数据集进行视频预测任务和混沌流体动力学预测实验中证明了这一结果。

    arXiv:2402.09470v1 Announce Type: new  Abstract: Diffusion models have recently been increasingly applied to temporal data such as video, fluid mechanics simulations, or climate data. These methods generally treat subsequent frames equally regarding the amount of noise in the diffusion process. This paper explores Rolling Diffusion: a new approach that uses a sliding window denoising process. It ensures that the diffusion process progressively corrupts through time by assigning more noise to frames that appear later in a sequence, reflecting greater uncertainty about the future as the generation process unfolds. Empirically, we show that when the temporal dynamics are complex, Rolling Diffusion is superior to standard diffusion. In particular, this result is demonstrated in a video prediction task using the Kinetics-600 video dataset and in a chaotic fluid dynamics forecasting experiment.
    
[^165]: 神经网络中的傅立叶电路：解锁大规模语言模型在数学推理和模运算中的潜力

    Fourier Circuits in Neural Networks: Unlocking the Potential of Large Language Models in Mathematical Reasoning and Modular Arithmetic

    [https://arxiv.org/abs/2402.09469](https://arxiv.org/abs/2402.09469)

    本研究探索了神经网络和Transformer在数学推理和模运算中的潜力。我们分析了单隐藏层神经网络和单层Transformer在解决复杂代数学习任务中的特征。阐明了边缘最大化原则对单隐藏层神经网络的影响。

    

    在机器学习不断发展的背景下，理解神经网络和Transformer所利用的内部表示是一个关键挑战。本研究在近期的研究基础上，对网络采用特定计算策略背后的原因进行了探索。我们的研究聚焦于涉及k个输入的复杂代数学习任务，即模运算的加法。我们对单隐藏层神经网络和单层Transformer在解决这一任务中学到的特征进行了深入的分析。我们理论框架的一个关键是阐明边缘最大化原则对单隐藏层神经网络采用的特征的影响。其中，p表示模数，Dp表示k个输入的模运算数据集，m表示网络输出。

    arXiv:2402.09469v1 Announce Type: new  Abstract: In the evolving landscape of machine learning, a pivotal challenge lies in deciphering the internal representations harnessed by neural networks and Transformers. Building on recent progress toward comprehending how networks execute distinct target functions, our study embarks on an exploration of the underlying reasons behind networks adopting specific computational strategies. We direct our focus to the complex algebraic learning task of modular addition involving $k$ inputs. Our research presents a thorough analytical characterization of the features learned by stylized one-hidden layer neural networks and one-layer Transformers in addressing this task.   A cornerstone of our theoretical framework is the elucidation of how the principle of margin maximization shapes the features adopted by one-hidden layer neural networks. Let $p$ denote the modulus, $D_p$ denote the dataset of modular arithmetic with $k$ inputs and $m$ denote the net
    
[^166]: 最优阈值线性赌博机

    Optimal Thresholding Linear Bandit

    [https://arxiv.org/abs/2402.09467](https://arxiv.org/abs/2402.09467)

    本论文研究了具有固定置信度的随机线性赌博机的ε-阈值赌博机问题，并提出了一种在渐近意义上是最优的算法。

    

    我们研究了一种新颖的纯探索问题：具有固定置信度的随机线性赌博机的ε-阈值赌博机问题(TBP)。我们证明了样本复杂性的下界，并将设计用于线性情况下的最佳臂识别算法扩展到了TBP，该算法在渐近意义上是最优的。

    arXiv:2402.09467v1 Announce Type: cross  Abstract: We study a novel pure exploration problem: the $\epsilon$-Thresholding Bandit Problem (TBP) with fixed confidence in stochastic linear bandits. We prove a lower bound for the sample complexity and extend an algorithm designed for Best Arm Identification in the linear case to TBP that is asymptotically optimal.
    
[^167]: 不同算法（可能）揭示出不同的模式：一个脑龄预测案例研究

    Different Algorithms (Might) Uncover Different Patterns: A Brain-Age Prediction Case Study

    [https://arxiv.org/abs/2402.09464](https://arxiv.org/abs/2402.09464)

    本文研究了在脑电图研究中不同算法是否能一致揭示出脑龄预测的假设，发现虽然大多数模型揭示了相似的发现，但也存在差异。

    

    机器学习是一个快速发展的领域，具有广泛的应用，包括生物信号分析，在这个领域中，新颖的算法通常能改善现有技术水平。然而，对算法多样性的鲁棒性，即不同算法是否能一致揭示出相似的发现，很少得到探究。本文研究了在脑电图研究中从已有研究中验证的脑龄预测假设是否适用于不同算法。首先，我们调研了文献，并确定了各种已知对脑龄预测有信息量的特征。我们采用了不同的特征提取技术、处理步骤和模型，并利用了SHAP值的解释能力将我们的发现与现有研究相对齐。我们的模型中很少有几个在我们使用的特定数据集上达到了最先进的性能。此外，分析表明，大多数模型虽然揭示了相似的发现，但也存在差异。

    arXiv:2402.09464v1 Announce Type: cross  Abstract: Machine learning is a rapidly evolving field with a wide range of applications, including biological signal analysis, where novel algorithms often improve the state-of-the-art. However, robustness to algorithmic variability - measured by different algorithms, consistently uncovering similar findings - is seldom explored. In this paper we investigate whether established hypotheses in brain-age prediction from EEG research validate across algorithms. First, we surveyed literature and identified various features known to be informative for brain-age prediction. We employed diverse feature extraction techniques, processing steps, and models, and utilized the interpretative power of SHapley Additive exPlanations (SHAP) values to align our findings with the existing research in the field. Few of our models achieved state-of-the-art performance on the specific data-set we utilized. Moreover, analysis demonstrated that while most models do unc
    
[^168]: WaveNet架构在具有可学习扩张和数据增强的RF信号分离上的新方法

    A Novel Approach to WaveNet Architecture for RF Signal Separation with Learnable Dilation and Data Augmentation

    [https://arxiv.org/abs/2402.09461](https://arxiv.org/abs/2402.09461)

    本文提出了一种新的WaveNet架构的适应方法，引入了可学习的扩张参数，显著提高了在密集RF频谱中的信号分离。该方法通过改进模型架构和创新的数据增强策略，成功提高了模型对复杂信号源的识别能力，实现了显著的性能改进，并在竞赛中取得了第一名的成绩。

    

    本文提出了一种新的WaveNet架构的适应方法，介绍了可学习的扩张参数，显著增强了在密集RF频谱中的信号分离。我们通过改进模型架构和创新的数据增强策略，显著提高了模型对复杂信号源的识别能力。本文详细介绍了我们的综合方法，包括改进的模型架构、数据准备技术和关键的训练策略，这些对我们的成功至关重要。我们的方法的有效性通过显著的改进得到了证明：在OFDM-QPSK与EMI信号1的BER为$10^{-3}$情况下，SINR增加了58.82％，超过了传统基准。值得注意的是，我们的模型在挑战中获得了第一名\cite{datadrivenrf2024}，证明了其卓越的性能并确立了一个新的标杆。

    arXiv:2402.09461v1 Announce Type: cross  Abstract: In this paper, we address the intricate issue of RF signal separation by presenting a novel adaptation of the WaveNet architecture that introduces learnable dilation parameters, significantly enhancing signal separation in dense RF spectrums. Our focused architectural refinements and innovative data augmentation strategies have markedly improved the model's ability to discern complex signal sources. This paper details our comprehensive methodology, including the refined model architecture, data preparation techniques, and the strategic training strategy that have been pivotal to our success. The efficacy of our approach is evidenced by the substantial improvements recorded: a 58.82\% increase in SINR at a BER of $10^{-3}$ for OFDM-QPSK with EMI Signal 1, surpassing traditional benchmarks. Notably, our model achieved first place in the challenge \cite{datadrivenrf2024}, demonstrating its superior performance and establishing a new stand
    
[^169]: 无监督学习的端到端无延迟生成固定滤波主动噪声控制

    Unsupervised learning based end-to-end delayless generative fixed-filter active noise control

    [https://arxiv.org/abs/2402.09460](https://arxiv.org/abs/2402.09460)

    本文提出了一种无监督学习的端到端无延迟生成固定滤波主动噪声控制方法，通过将协处理器和实时控制器集成到一个可微分的ANC系统中，不仅省略了标注过程，而且在噪声降低性能方面表现更好。

    

    我们通过先前的生成固定滤波主动噪声控制（GFANC）框架的高效协同，实现了无延迟噪声控制。然而，协处理器中的一维卷积神经网络（1D CNN）需要使用标注的噪声数据集进行初始训练。标注噪声数据可能会消耗大量资源，并且可能引入一些偏差。本文提出了一种无监督的GFANC方法，以简化1D CNN训练过程并增强其实用性。在训练过程中，将协处理器和实时控制器集成到一个端到端可微分的ANC系统中。这使得我们可以将累积的平方误差信号作为训练1D CNN的损失函数。通过这种无监督学习范式，无监督GFANC方法不仅省略了标注过程，而且在噪声降低性能方面表现更好。

    arXiv:2402.09460v1 Announce Type: cross  Abstract: Delayless noise control is achieved by our earlier generative fixed-filter active noise control (GFANC) framework through efficient coordination between the co-processor and real-time controller. However, the one-dimensional convolutional neural network (1D CNN) in the co-processor requires initial training using labelled noise datasets. Labelling noise data can be resource-intensive and may introduce some biases. In this paper, we propose an unsupervised-GFANC approach to simplify the 1D CNN training process and enhance its practicality. During training, the co-processor and real-time controller are integrated into an end-to-end differentiable ANC system. This enables us to use the accumulated squared error signal as the loss for training the 1D CNN. With this unsupervised learning paradigm, the unsupervised-GFANC method not only omits the labelling process but also exhibits better noise reduction performance compared to the supervise
    
[^170]: 面向人体部位定向跟踪和3D运动可视化的定制IMU无线可穿戴系统

    Custom IMU-Based Wearable System for Robust 2.4 GHz Wireless Human Body Parts Orientation Tracking and 3D Movement Visualization on an Avatar

    [https://arxiv.org/abs/2402.09459](https://arxiv.org/abs/2402.09459)

    这项研究的目标是通过构建可负担的定制IMU无线可穿戴系统，在人体运动分析中实现对身体部位定向跟踪和3D运动可视化。

    

    最近的研究确认了使用惯性测量单元（IMU）的系统对于人体运动分析的适用性。然而，高端的商业化IMU解决方案价格昂贵且复杂，无法普及在广大潜在用户中的使用。市场上出现了一些功能较少的低端商业化解决方案，试图填补这一空白，但仍存在一些需要克服的限制。与此同时，在医疗和运动应用领域，越来越多的科学论文使用的是非商业化的、自制的IMU系统。尽管这些解决方案可以促进这项技术的普及化使用，但它们的功能更为有限，并且如何设计和构建它们的描述在文献中仍然稀缺。本文的目的是：（1）证明构建一种可负担的定制解决方案是可行的，旨在同时追踪多个人体部位的定向和运动。

    arXiv:2402.09459v1 Announce Type: cross  Abstract: Recent studies confirm the applicability of Inertial Measurement Unit (IMU)-based systems for human motion analysis. Notwithstanding, high-end IMU-based commercial solutions are yet too expensive and complex to democratize their use among a wide range of potential users. Less featured entry-level commercial solutions are being introduced in the market, trying to fill this gap, but still present some limitations that need to be overcome. At the same time, there is a growing number of scientific papers using not commercial, but custom do-it-yourself IMU-based systems in medical and sports applications. Even though these solutions can help to popularize the use of this technology, they have more limited features and the description on how to design and build them from scratch is yet too scarce in the literature. The aim of this work is two-fold: (1) Proving the feasibility of building an affordable custom solution aimed at simultaneous mu
    
[^171]: 未知博弈中乐观的汤普森抽样方法用于无遗憾学习

    Optimistic Thompson Sampling for No-Regret Learning in Unknown Games

    [https://arxiv.org/abs/2402.09456](https://arxiv.org/abs/2402.09456)

    该论文提出了一种在未知博弈中进行无遗憾学习的乐观的汤普森抽样方法，通过利用对手的行动和奖励结构信息，显著减少了实验预算，成功地缓解了多机构问题。此外，研究还引入了乐观-无遗憾框架，将现有算法与提出的方法相结合。

    

    许多涉及多个决策者的真实世界问题可以建模为一个具有部分观测的未知博弈。为了解决部分信息和多机构的挑战，我们开发了汤普森抽样类型的算法，利用对手的行动和奖励结构的信息。我们的方法在实际应用中，如交通路由和雷达感知中，显著减少了实验预算，与基准算法相比，减少了十倍以上。我们证明，在对奖励结构有一定假设的情况下，遗憾界限仅对总行动空间大小呈对数依赖，有效缓解了多机构问题。此外，本研究引入了乐观-无遗憾框架，该框架将我们提出的方法和领域内现有的算法相结合，是一项新的贡献。

    arXiv:2402.09456v1 Announce Type: cross  Abstract: Many real-world problems involving multiple decision-makers can be modeled as an unknown game characterized by partial observations. Addressing the challenges posed by partial information and the curse of multi-agency, we developed Thompson sampling-type algorithms, leveraging information about opponent's action and reward structures. Our approach significantly reduces experimental budgets, achieving a more than tenfold reduction compared to baseline algorithms in practical applications like traffic routing and radar sensing. We demonstrate that, under certain assumptions about the reward structure, the regret bound exhibits merely a logarithmic dependence on the total action space size, effectively mitigating the curse of multi-agency. Additionally, this research introduces the Optimism-then-NoRegret framework, a novel contribution that integrates both our proposed methodologies and existing algorithms in the field.
    
[^172]: 使用Wasserstein生成对抗网络提高EEG信号分类准确性

    Improving EEG Signal Classification Accuracy Using Wasserstein Generative Adversarial Networks

    [https://arxiv.org/abs/2402.09453](https://arxiv.org/abs/2402.09453)

    该论文提出了一种通过使用Wasserstein生成对抗网络(WGAN)来提高EEG信号分类准确性的实际解决方案。 WGAN在BCI2000数据集上进行训练，并通过改进的平均准确率和测量得分证明了生成的EEG信号的质量。

    

    静息生态（EEG）在记录脑部活动中起着重要作用，并且对脑机接口（BCI）技术的发展至关重要。然而，EEG信号的有限可用性和高度变异性给创建可靠的BCI带来了巨大的挑战。为了解决这个问题，我们提出了一个实际的解决方案，利用深度学习和Wasserstein生成对抗网络（WGAN）的最新发展。WGAN是在BCI2000数据集上进行训练的，该数据集包括来自45个个体的约1500个EEG记录和64个通道。通过三个分类器评估生成的EEG信号，得到了改进的平均准确率。使用Frechet Inception Distance（FID）测量的生成信号质量为1.345（睁眼）和11.565（闭眼）。即使没有频谱或空间损失项，我们的WGAN模型仍能模拟出频谱和空间特性。

    arXiv:2402.09453v1 Announce Type: cross  Abstract: Electroencephalography (EEG) plays a vital role in recording brain activities and is integral to the development of brain-computer interface (BCI) technologies. However, the limited availability and high variability of EEG signals present substantial challenges in creating reliable BCIs. To address this issue, we propose a practical solution drawing on the latest developments in deep learning and Wasserstein Generative Adversarial Network (WGAN). The WGAN was trained on the BCI2000 dataset, consisting of around 1500 EEG recordings and 64 channels from 45 individuals. The generated EEG signals were evaluated via three classifiers yielding improved average accuracies. The quality of generated signals measured using Frechet Inception Distance (FID) yielded scores of 1.345 and 11.565 for eyes-open and closed respectively. Even without a spectral or spatial loss term, our WGAN model was able to emulate the spectral and spatial properties of
    
[^173]: 基于WiFi的家庭医疗中真实世界患者活动监测的数据分布动态

    Data Distribution Dynamics in Real-World WiFi-Based Patient Activity Monitoring for Home Healthcare

    [https://arxiv.org/abs/2402.09452](https://arxiv.org/abs/2402.09452)

    本文研究了在家庭医疗场景中使用WiFi信号进行日常活动监测的应用，通过在不同环境中部署系统和分析数据变化，指导了稳健、上下文感知的WiFi感知系统的开发，提高了老年护理的生活质量。

    

    本文研究了在家庭医疗场景中使用WiFi信号进行日常活动监测的应用。尽管基于WiFi的活动识别在实验室环境中很有前景，但在真实世界环境中面临环境、主体和系统配置变量等挑战，影响准确性和适应性。研究包括在不同环境中部署系统和分析数据变化。研究旨在指导真实环境下稳健、上下文感知的WiFi感知系统的开发，用于老年护理。研究结果表明WiFi感知的活动检测正在发生变化，弥合了学术研究与实际应用之间的差距，通过技术提高生活质量。

    arXiv:2402.09452v1 Announce Type: cross  Abstract: This paper examines the application of WiFi signals for real-world monitoring of daily activities in home healthcare scenarios. While the state-of-the-art of WiFi-based activity recognition is promising in lab environments, challenges arise in real-world settings due to environmental, subject, and system configuration variables, affecting accuracy and adaptability. The research involved deploying systems in various settings and analyzing data shifts. It aims to guide realistic development of robust, context-aware WiFi sensing systems for elderly care. The findings suggest a shift in WiFi-based activity sensing, bridging the gap between academic research and practical applications, enhancing life quality through technology.
    
[^174]: 引导遮蔽表示学习以捕捉心电图的时空关系

    Guiding Masked Representation Learning to Capture Spatio-Temporal Relationship of Electrocardiogram

    [https://arxiv.org/abs/2402.09450](https://arxiv.org/abs/2402.09450)

    本研究提出了一种叫做ST-MEM的模型，通过重构遮蔽的心电图数据来学习时空特征，该模型在心律失常分类任务中优于其他自监督学习方法。

    

    心电图（ECG）广泛用作监测心脏起源的电信号的诊断工具。近年来，机器学习的研究努力集中在使用ECG信号进行各种疾病筛查的应用上。然而，适应疾病筛查应用是具有挑战性的，因为标记的ECG数据有限。通过自监督学习（SSL）实现通用表示是克服标记数据稀缺性的常用方法；然而，在ECG数据上纯粹应用SSL，而不考虑ECG信号固有的时空关系，可能会产生次优的结果。本文介绍了ST-MEM（时空遮蔽心电图建模），该模型通过重构遮蔽的12导联ECG数据来学习时空特征。在各种实验设置中，ST-MEM在心律失常分类任务中的性能优于其他SSL基线方法。

    arXiv:2402.09450v1 Announce Type: cross  Abstract: Electrocardiograms (ECG) are widely employed as a diagnostic tool for monitoring electrical signals originating from a heart. Recent machine learning research efforts have focused on the application of screening various diseases using ECG signals. However, adapting to the application of screening disease is challenging in that labeled ECG data are limited. Achieving general representation through self-supervised learning (SSL) is a well-known approach to overcome the scarcity of labeled data; however, a naive application of SSL to ECG data, without considering the spatial-temporal relationships inherent in ECG signals, may yield suboptimal results. In this paper, we introduce ST-MEM (Spatio-Temporal Masked Electrocardiogram Modeling), designed to learn spatio-temporal features by reconstructing masked 12-lead ECG data. ST-MEM outperforms other SSL baseline methods in various experimental settings for arrhythmia classification tasks. Mo
    
[^175]: 普通EEG与三极EEG在高性能到颤抓握BCI系统中的比较研究

    A Comparative Study of Conventional and Tripolar EEG for High-Performance Reach-to-Grasp BCI Systems

    [https://arxiv.org/abs/2402.09448](https://arxiv.org/abs/2402.09448)

    比较传统EEG与三极EEG在高性能到颤抓握BCI系统中的有效性，包括信噪比、空间分辨率、ERPs和小波时频分析。

    

    本研究旨在比较传统EEG与三极EEG在提升运动障碍个体的BCI应用方面的有效性。重点是解读和解码各种抓握动作，如力握和精确握持。目标是确定哪种EEG技术在处理和翻译与抓握相关的脑电信号方面更为有效。研究涉及对十名健康参与者进行实验，参与者进行了两种不同的握持运动：力握和精确握持，无运动条件作为基线。我们的研究在解码抓握动作方面对EEG和三极EEG进行了全面比较。该比较涵盖了几个关键参数，包括信噪比（SNR）、通过功能连接的空间分辨率、ERPs和小波时频分析。此外，我们的研究还涉及从...

    arXiv:2402.09448v1 Announce Type: cross  Abstract: This study aims to enhance BCI applications for individuals with motor impairments by comparing the effectiveness of tripolar EEG (tEEG) with conventional EEG. The focus is on interpreting and decoding various grasping movements, such as power grasp and precision grasp. The goal is to determine which EEG technology is more effective in processing and translating grasp related neural signals. The approach involved experimenting on ten healthy participants who performed two distinct grasp movements: power grasp and precision grasp, with a no movement condition serving as the baseline. Our research presents a thorough comparison between EEG and tEEG in decoding grasping movements. This comparison spans several key parameters, including signal to noise ratio (SNR), spatial resolution via functional connectivity, ERPs, and wavelet time frequency analysis. Additionally, our study involved extracting and analyzing statistical features from th
    
[^176]: 非侵入性脑电图信号的小波分析区分复杂和自然的抓握类型

    Wavelet Analysis of Noninvasive EEG Signals Discriminates Complex and Natural Grasp Types

    [https://arxiv.org/abs/2402.09447](https://arxiv.org/abs/2402.09447)

    该研究使用小波分析技术对非侵入性脑电图信号进行解码，成功区分复杂和自然的抓握类型，并且证明了小波特征在基于脑电图的抓握区分中的有效性。

    

    该研究旨在通过对脑电图（EEG）信号进行解码，为灵巧的神经假肢开发和脑机接口（BCI）应用来区分手部抓握，特别是针对运动障碍患者。具体而言，它专注于使用一种新的基于脑电图的BCI平台和小波信号处理，区分两种复杂的自然力量和精确抓握类型以及一种中立条件作为无运动条件。小波分析涉及从小波能量系数生成时间频率和拓扑图。然后，通过使用机器学习技术和新型小波特征，我们实现了高平均准确率：多类别为85.16%，无运动 vs 力量为95.37%，无运动 vs 精确为95.40%，力量 vs 精确为88.07%，证明了这些特征在基于脑电图的抓握区分中的有效性。与先前的研究不同，我们研究的关键部分是排列特征重要性的部分。

    arXiv:2402.09447v1 Announce Type: cross  Abstract: This research aims to decode hand grasps from Electroencephalograms (EEGs) for dexterous neuroprosthetic development and Brain-Computer Interface (BCI) applications, especially for patients with motor disorders. Particularly, it focuses on distinguishing two complex natural power and precision grasps in addition to a neutral condition as a no-movement condition using a new EEG-based BCI platform and wavelet signal processing. Wavelet analysis involved generating time-frequency and topographic maps from wavelet power coefficients. Then, by using machine learning techniques with novel wavelet features, we achieved high average accuracies: 85.16% for multiclass, 95.37% for No-Movement vs Power, 95.40% for No-Movement vs Precision, and 88.07% for Power vs Precision, demonstrating the effectiveness of these features in EEG-based grasp differentiation. In contrast to previous studies, a critical part of our study was permutation feature impo
    
[^177]: iMove: 探索用于健身活动识别的生物阻抗传感技术

    iMove: Exploring Bio-impedance Sensing for Fitness Activity Recognition

    [https://arxiv.org/abs/2402.09445](https://arxiv.org/abs/2402.09445)

    通过传感器融合和对比学习，研究证明生物阻抗传感技术可以改进基于IMU的健身追踪，提高分类模型的精度。

    

    自动和精确的健身活动识别对于促进健康生活方式和个性化预防性医疗具有益处。虽然IMU目前是主要的健身追踪模式，但通过iMove，我们展示了生物阻抗可以通过传感器融合和对比学习来改善基于IMU的健身追踪。为了评估我们的方法，我们进行了一项实验，包括十个受试者在五天内进行的六种上身健身活动，以收集来自两只手腕的生物阻抗和左手腕IMU的同步数据。对比学习框架利用两种模态来训练更好的仅基于IMU的分类模型，其中生物阻抗只在训练阶段需要，通过这种方式，输入单个IMU的平均宏F1分数提高了3.22％，达到84.71％，而IMU基线模型为81.49％。我们还展示了生物阻抗如何能够...

    arXiv:2402.09445v1 Announce Type: cross  Abstract: Automatic and precise fitness activity recognition can be beneficial in aspects from promoting a healthy lifestyle to personalized preventative healthcare. While IMUs are currently the prominent fitness tracking modality, through iMove, we show bio-impedence can help improve IMU-based fitness tracking through sensor fusion and contrastive learning.To evaluate our methods, we conducted an experiment including six upper body fitness activities performed by ten subjects over five days to collect synchronized data from bio-impedance across two wrists and IMU on the left wrist.The contrastive learning framework uses the two modalities to train a better IMU-only classification model, where bio-impedance is only required at the training phase, by which the average Macro F1 score with the input of a single IMU was improved by 3.22 \% reaching 84.71 \% compared to the 81.49 \% of the IMU baseline model. We have also shown how bio-impedance can 
    
[^178]: 预测疲劳的 EEG 算法综述

    Review of algorithms for predicting fatigue using EEG

    [https://arxiv.org/abs/2402.09443](https://arxiv.org/abs/2402.09443)

    该研究综述了使用 EEG 信号进行疲劳预测的机器学习算法，并评估了不同算法在基于 EEG 数据预测个体疲劳水平方面的效果。

    

    疲劳检测对于提高交通、医疗和工业等各个领域的安全性、生产力和福祉至关重要。本科学论文对使用机器学习算法检测生理疲劳的方法进行了全面的调查，使用脑电图（EEG）信号。本研究的主要目标是评估不同算法在基于 EEG 数据预测个体疲劳水平方面的功效。

    arXiv:2402.09443v1 Announce Type: cross  Abstract: Fatigue detection is of paramount importance in enhancing safety, productivity, and well-being across diverse domains, including transportation, healthcare, and industry. This scientific paper presents a comprehensive investigation into the application of machine learning algorithms for the detection of physiological fatigue using Electroencephalogram (EEG) signals. The primary objective of this study was to assess the efficacy of various algorithms in predicting an individual's level of fatigue based on EEG data.
    
[^179]: 基于深度学习的智能反射表面辅助集成感知与通信系统的信道估计

    Deep-Learning Channel Estimation for IRS-Assisted Integrated Sensing and Communication System

    [https://arxiv.org/abs/2402.09441](https://arxiv.org/abs/2402.09441)

    本文针对智能反射表面辅助集成感知与通信系统中的信道估计问题，提出了一个基于深度学习的三阶段方法。该方法通过解耦问题，分别估计直接感知和通信信道、反射通信信道和反射感知信道，以应对智能反射表面的信号处理能力不足和感知与通信信号之间的互相干扰。

    

    本文首次关注智能反射表面辅助集成感知与通信系统中的信道估计问题。由于被动智能反射表面缺乏信号处理能力，并且在集成感知与通信系统中存在感知和通信信号之间的互相干扰，这个问题具有挑战性。我们提出了一个三阶段的方法来解决这个估计问题，其中包括在第一阶段估计直接感知和通信信道，在第二阶段估计反射通信信道，在第三阶段估计反射感知信道。所提出的三阶段方法基于一个深度学习框架，其中包括两种不同的卷积神经网络结构来进行信道估计。

    arXiv:2402.09441v1 Announce Type: cross  Abstract: Integrated sensing and communication (ISAC), and intelligent reflecting surface (IRS) are envisioned as revolutionary technologies to enhance spectral and energy efficiencies for next wireless system generations. For the first time, this paper focuses on the channel estimation problem in an IRS-assisted ISAC system. This problem is challenging due to the lack of signal processing capacity in passive IRS, as well as the presence of mutual interference between sensing and communication (SAC) signals in ISAC systems. A three-stage approach is proposed to decouple the estimation problem into sub-ones, including the estimation of the direct SAC channels in the first stage, reflected communication channel in the second stage, and reflected sensing channel in the third stage. The proposed three-stage approach is based on a deep-learning framework, which involves two different convolutional neural network (CNN) architectures to estimate the ch
    
[^180]: 基于极限学习机的智能反射面辅助多用户ISAC系统的信道估计

    Extreme Learning Machine-based Channel Estimation in IRS-Assisted Multi-User ISAC System

    [https://arxiv.org/abs/2402.09440](https://arxiv.org/abs/2402.09440)

    本文提出了一种基于极限学习机的智能反射面辅助多用户ISAC系统的信道估计方法，该方法通过将估计问题分解成子问题来解决了感知和通信信号干扰以及被动式IRS缺乏信号处理能力的挑战。该方法可以在保持低成本需求的情况下实现对SAC信道和下行通信信道的准确估计。

    

    最近，智能反射面（IRS）辅助的多用户集成感知和通信（ISAC）系统已经被研究以提供高频谱和能量有效性传输。本文首次提出了一个实用的信道估计方法，用于IRS辅助的多用户ISAC系统。在这样的系统中，估计问题具有挑战性，因为感知和通信（SAC）信号相互干扰，被动式的IRS缺乏信号处理能力。本文提出了一个两阶段的方法，将整体估计问题逐步转化为子问题，依次包括直接和反射信道的估计。基于此方案，ISAC基站（BS）估计与目标和上行用户相关的所有SAC信道，而每个下行用户单独估计下行通信信道。考虑到ISAC基站和下行用户的低成本需求，本文的方法具有实用性和可行性。

    arXiv:2402.09440v1 Announce Type: cross  Abstract: Multi-user integrated sensing and communication (ISAC) assisted by intelligent reflecting surface (IRS) has been recently investigated to provide a high spectral and energy efficiency transmission. This paper proposes a practical channel estimation approach for the first time to an IRS-assisted multiuser ISAC system. The estimation problem in such a system is challenging since the sensing and communication (SAC) signals interfere with each other, and the passive IRS lacks signal processing ability. A two-stage approach is proposed to transfer the overall estimation problem into sub-ones, successively including the direct and reflected channels estimation. Based on this scheme, the ISAC base station (BS) estimates all the SAC channels associated with the target and uplink users, while each downlink user estimates the downlink communication channels individually. Considering a low-cost demand of the ISAC BS and downlink users, the propos
    
[^181]: 基于深度学习的智能反射式面辅助ISAC系统的信道估计

    Deep-Learning-Based Channel Estimation for IRS-Assisted ISAC System

    [https://arxiv.org/abs/2402.09439](https://arxiv.org/abs/2402.09439)

    本文提出了一种基于深度学习的框架，在IRS辅助的ISAC系统中解决了信道估计问题。通过设计两种不同的神经网络架构，该方法在不同的信道环境下实现了优越性能。

    

    综合感知和通信（ISAC）以及智能反射式面（IRS）被视为未来无线网络的有希望的技术。本文研究了IRS辅助的ISAC系统中的信道估计问题。提出了一种基于深度学习的框架来估计该系统中的感知和通信（S&C）信道。考虑到S&C信道的不同传播环境，设计了两种深度神经网络（DNN）架构来实现此框架。第一个DNN被设计在ISAC基站上用于估计感知信道，而第二个DNN架构则被分配给每个下行用户设备用于估计其通信信道。此外，精心设计了用于训练DNN的输入-输出对。仿真结果表明，与各种信噪比条件下的基准方案相比，所提出的估计方法具有优势。

    arXiv:2402.09439v1 Announce Type: cross  Abstract: Integrated sensing and communication (ISAC) and intelligent reflecting surface (IRS) are viewed as promising technologies for future generations of wireless networks. This paper investigates the channel estimation problem in an IRS-assisted ISAC system. A deep-learning framework is proposed to estimate the sensing and communication (S&C) channels in such a system. Considering different propagation environments of the S&C channels, two deep neural network (DNN) architectures are designed to realize this framework. The first DNN is devised at the ISAC base station for estimating the sensing channel, while the second DNN architecture is assigned to each downlink user equipment to estimate its communication channel. Moreover, the input-output pairs to train the DNNs are carefully designed. Simulation results show the superiority of the proposed estimation approach compared to the benchmark scheme under various signal-to-noise ratio conditi
    
[^182]: 基于EEG的无主题深度架构用于运动想象分类

    Subject-Independent Deep Architecture for EEG-based Motor Imagery Classification

    [https://arxiv.org/abs/2402.09438](https://arxiv.org/abs/2402.09438)

    本研究提出了一种基于无主题深度架构的方法，用于EEG信号的运动想象分类。该方法通过无监督和半监督的方式进行训练，能够在有限的标记样本情况下独立地对不同受试者进行分类。具体而言，通过无监督学习获得潜在特征，然后使用监督学习进行分类。

    

    基于脑电图(EEG)的运动想象(MI)分类是非侵入性脑-计算机接口(BCI)系统中广泛使用的技术。由于受到不同受试者之间的异质性和标记数据不足的影响，设计一个能够在有限标记样本情况下独立于受试者进行MI分类的分类器是可取的。为了克服这些限制，我们提出了一种新颖的基于无主题半监督深度架构(SSDA)的方法。所提出的SSDA包含两部分：无监督部分和监督部分。训练集包含来自多个受试者的有标记和无标记数据样本。首先，无监督部分称为列式时空自编码器(CST-AE)，通过最大化原始数据和重构数据之间的相似度来提取所有训练样本的潜在特征。采用尺度缩放方法来降低维度。

    arXiv:2402.09438v1 Announce Type: cross  Abstract: Motor imagery (MI) classification based on electroencephalogram (EEG) is a widely-used technique in non-invasive brain-computer interface (BCI) systems. Since EEG recordings suffer from heterogeneity across subjects and labeled data insufficiency, designing a classifier that performs the MI independently from the subject with limited labeled samples would be desirable. To overcome these limitations, we propose a novel subject-independent semi-supervised deep architecture (SSDA). The proposed SSDA consists of two parts: an unsupervised and a supervised element. The training set contains both labeled and unlabeled data samples from multiple subjects. First, the unsupervised part, known as the columnar spatiotemporal auto-encoder (CST-AE), extracts latent features from all the training samples by maximizing the similarity between the original and reconstructed data. A dimensional scaling approach is employed to reduce the dimensionality o
    
[^183]: 解开不完美之谜：一种融合小波的多层异构网络用于人体活动识别中的不完美穿戴式传感器数据

    Disentangling Imperfect: A Wavelet-Infused Multilevel Heterogeneous Network for Human Activity Recognition in Flawed Wearable Sensor Data

    [https://arxiv.org/abs/2402.09434](https://arxiv.org/abs/2402.09434)

    该论文提出了一种融合小波的多层异构网络（MHNN）用于处理不完美的穿戴式传感器数据。研究团队通过多层离散小波分解提取了多分辨率特征，实现了对不同频率信号的区分，以抑制噪音。

    

    可穿戴设备的流行和普及为基于传感器的人体活动识别提供了利用基于深度学习的算法的新机会。尽管取得了令人印象深刻的进展，但仍然存在两个主要挑战。第一，由于传感器的放置和其他问题以及数据传输故障，传感器数据通常不完整或嘈杂，需要填充缺失值，这也会引入噪音。第二，人体活动具有多尺度特征。因此，不同的人群甚至同一个人在不同情况下可能表现不同。为了解决这些挑战，我们提出了一种名为MHNN的多层异构神经网络，用于传感器数据分析。我们利用多层离散小波分解从传感器数据中提取多分辨率特征。这样可以区分具有不同频率的信号，从而抑制噪音。

    arXiv:2402.09434v1 Announce Type: cross  Abstract: The popularity and diffusion of wearable devices provides new opportunities for sensor-based human activity recognition that leverages deep learning-based algorithms. Although impressive advances have been made, two major challenges remain. First, sensor data is often incomplete or noisy due to sensor placement and other issues as well as data transmission failure, calling for imputation of missing values, which also introduces noise. Second, human activity has multi-scale characteristics. Thus, different groups of people and even the same person may behave differently under different circumstances. To address these challenges, we propose a multilevel heterogeneous neural network, called MHNN, for sensor data analysis. We utilize multilevel discrete wavelet decomposition to extract multi-resolution features from sensor data. This enables distinguishing signals with different frequencies, thereby suppressing noise. As the components res
    
[^184]: 基于电气行为关联挖掘的家庭短期能耗预测研究

    Electrical Behavior Association Mining for Household ShortTerm Energy Consumption Forecasting

    [https://arxiv.org/abs/2402.09433](https://arxiv.org/abs/2402.09433)

    本文提出了一种基于电气行为关联挖掘的家庭短期能耗预测方法，通过概率关联模型和卷积神经网络门控循环单元的结合，实现了显著的准确性提升。

    

    准确的家庭短期能耗预测(STECF)对家庭能源管理至关重要，但由于个别住户的高度随机行为，技术上具有挑战性。为了提高日前程度的STECF准确性，本文提出了一种新的STECF方法，利用电气行为中的关联挖掘。首先，提出了一种概率化的关联量化和发现方法，用于建模行为之间的关联，并生成关联群集。然后，采用卷积神经网络门控循环单元(CNN-GRU)进行预测，以探索时间相关性并提高准确性。测试结果表明，该方法在STECF方面得到了显著的提升。

    arXiv:2402.09433v1 Announce Type: cross  Abstract: Accurate household short-term energy consumption forecasting (STECF) is crucial for home energy management, but it is technically challenging, due to highly random behaviors of individual residential users. To improve the accuracy of STECF on a day-ahead scale, this paper proposes an novel STECF methodology that leverages association mining in electrical behaviors. First, a probabilistic association quantifying and discovering method is proposed to model the pairwise behaviors association and generate associated clusters. Then, a convolutional neural network-gated recurrent unit (CNN-GRU) based forecasting is provided to explore the temporal correlation and enhance accuracy. The testing results demonstrate that this methodology yields a significant enhancement in the STECF.
    
[^185]: 利用可持续深层径向函数在智能城市中的交通智能增强分析

    An Enhanced Analysis of Traffic Intelligence in Smart Cities Using Sustainable Deep Radial Function

    [https://arxiv.org/abs/2402.09432](https://arxiv.org/abs/2402.09432)

    本论文通过利用深度径向基函数网络，提出了一种新的策略来增强智能城市交通智能。深度RBF网络能够从交通数据中提取有价值的见解，并实现更精确的预测和决策。

    

    智能城市通过运用先进技术优化城市基础设施的各个方面，如交通系统，改变了城市居民的生活方式。有效的交通管理是智能城市的关键组成部分，因为它直接影响了居民和游客的生活质量。本文利用深度径向基函数（RBF）网络描述了一种增强智能城市交通智能的新策略。传统的交通分析方法经常依赖于简单的模型，不能捕捉到城市交通系统的复杂模式和动力学。深度学习技术，如深度RBF网络，有潜力从交通数据中提取有价值的见解，实现更精确的预测和决策。在本文中，我们提出了一种基于RBF的方法来增强智能城市的交通智能。深度RBF网络结合了适应性和普适性。

    arXiv:2402.09432v1 Announce Type: new  Abstract: Smart cities have revolutionized urban living by incorporating sophisticated technologies to optimize various aspects of urban infrastructure, such as transportation systems. Effective traffic management is a crucial component of smart cities, as it has a direct impact on the quality of life of residents and tourists. Utilizing deep radial basis function (RBF) networks, this paper describes a novel strategy for enhancing traffic intelligence in smart cities. Traditional methods of traffic analysis frequently rely on simplistic models that are incapable of capturing the intricate patterns and dynamics of urban traffic systems. Deep learning techniques, such as deep RBF networks, have the potential to extract valuable insights from traffic data and enable more precise predictions and decisions. In this paper, we propose an RBF based method for enhancing smart city traffic intelligence. Deep RBF networks combine the adaptability and general
    
[^186]: DoorINet: 一种用于门贴式物联网应用的深度学习惯性框架

    DoorINet: A Deep-Learning Inertial Framework for Door-Mounted IoT Applications

    [https://arxiv.org/abs/2402.09427](https://arxiv.org/abs/2402.09427)

    DoorINet是一种用于门贴式物联网应用的深度学习惯性框架，无需使用磁力计即可计算航向角度。

    

    许多物联网应用使用低成本的微型电动机械惯性传感器，其中一个常见的任务是方向估计。为了应对这种任务，应用姿态和航向参考系统算法。利用陀螺仪读数，通过加速度计读数更新姿态角度，利用磁力计测量更新航向角度。在室内环境中，磁力计受到干扰，会降低其性能。这主要影响到估计航向角度的应用，比如找到衣柜或冰箱门的航向角度。为了解决这种情况，我们提出了DoorINet，一种用于门贴式低成本惯性传感器的端到端深度学习框架，无需使用磁力计即可计算航向角度。为了评估我们的方法，我们记录了一个包含391分钟加速度计和陀螺仪测量的独特数据集。

    arXiv:2402.09427v1 Announce Type: cross  Abstract: Many Internet of Things applications utilize low-cost, micro, electro-mechanical inertial sensors. A common task is orientation estimation. To tackle such a task, attitude and heading reference system algorithms are applied. Relying on the gyroscope readings, the accelerometer readings are used to update the attitude angles, and magnetometer measurements are utilized to update the heading angle. In indoor environments, magnetometers suffer from interference that degrades their performance. This mainly influences applications focused on estimating the heading angle like finding the heading angle of a closet or fridge door. To circumvent such situations, we propose DoorINet, an end-to-end deep-learning framework to calculate the heading angle from door-mounted, low-cost inertial sensors without using magnetometers. To evaluate our approach, we record a unique dataset containing 391 minutes of accelerometer and gyroscope measurements and 
    
[^187]: 图Koopman自编码器用于预测隐蔽通信抵抗无人机监视

    Graph Koopman Autoencoder for Predictive Covert Communication Against UAV Surveillance

    [https://arxiv.org/abs/2402.09426](https://arxiv.org/abs/2402.09426)

    本论文提出了一种结合了图神经网络（GNN）和Koopman理论的新框架，用于在无人机监视的情况下实现地面的低概率检测（LPD）通信

    

    低概率检测（LPD）通信旨在模糊射频（RF）信号的存在，不仅仅是隐藏通信内容。然而，使用无人机（UAVs）引入了一个挑战，因为无人机可以通过在特定感兴趣区域盘旋来检测地面的射频信号。随着现代监视中无人机的不断使用，有一个迫切需要全面了解它们未知的非线性动态轨迹，从而有效实施LPD通信。不幸的是，这些关键信息通常无法直接获取，给LPD通信带来了重大障碍。为了解决这个问题，我们考虑了一个案例研究，即在多个从事监视的UAV存在时，实现地面LPD通信。我们提出了一种结合了图神经网络（GNN）和Koopman理论的新框架来预测轨迹

    arXiv:2402.09426v1 Announce Type: cross  Abstract: Low Probability of Detection (LPD) communication aims to obscure the very presence of radio frequency (RF) signals, going beyond just hiding the content of the communication. However, the use of Unmanned Aerial Vehicles (UAVs) introduces a challenge, as UAVs can detect RF signals from the ground by hovering over specific areas of interest. With the growing utilization of UAVs in modern surveillance, there is a crucial need for a thorough understanding of their unknown nonlinear dynamic trajectories to effectively implement LPD communication. Unfortunately, this critical information is often not readily available, posing a significant hurdle in LPD communication. To address this issue, we consider a case-study for enabling terrestrial LPD communication in the presence of multiple UAVs that are engaged in surveillance. We introduce a novel framework that combines graph neural networks (GNN) with Koopman theory to predict the trajectories
    
[^188]: 使用近似脉冲卷积变换器进行癫痫发作检测与预测

    Epilepsy Seizure Detection and Prediction using an Approximate Spiking Convolutional Transformer

    [https://arxiv.org/abs/2402.09424](https://arxiv.org/abs/2402.09424)

    本文介绍了一种名为Spiking Conformer的神经形态脉冲卷积变换器，用于从头皮长期脑电图（EEG）记录中检测和预测癫痫发作片段。通过利用基于脉冲的加法操作和近似脉冲神经元层，该模型显著降低了计算成本，同时保持准确性。

    

    癫痫是一种常见的神经系统疾病。及时预测癫痫发作并进行干预治疗可以显著减少患者的意外伤害，保护患者的生命和健康。本文提出了一种名为Spiking Conformer的神经形态脉冲卷积变换器，用于从头皮长期脑电图（EEG）记录中检测和预测癫痫发作片段。我们报告了使用Spiking Conformer模型对波士顿儿童医院-MIT (CHB-MIT) EEG数据集进行评估的结果。通过利用基于脉冲的加法操作，Spiking Conformer模型在与非脉冲模型相比显著降低了分类计算成本。此外，我们引入了一种近似脉冲神经元层，进一步降低了近38%的触发脉冲神经元更新，同时不损失准确性。使用原始EEG数据作为输入，所提出的Spiking Conformer模型实现了平均的敏感性...

    arXiv:2402.09424v1 Announce Type: cross  Abstract: Epilepsy is a common disease of the nervous system. Timely prediction of seizures and intervention treatment can significantly reduce the accidental injury of patients and protect the life and health of patients. This paper presents a neuromorphic Spiking Convolutional Transformer, named Spiking Conformer, to detect and predict epileptic seizure segments from scalped long-term electroencephalogram (EEG) recordings. We report evaluation results from the Spiking Conformer model using the Boston Children's Hospital-MIT (CHB-MIT) EEG dataset. By leveraging spike-based addition operations, the Spiking Conformer significantly reduces the classification computational cost compared to the non-spiking model. Additionally, we introduce an approximate spiking neuron layer to further reduce spike-triggered neuron updates by nearly 38% without sacrificing accuracy. Using raw EEG data as input, the proposed Spiking Conformer achieved an average sens
    
[^189]: 基于脑电图的抑郁症产生鉴别器

    EEG Based Generative Depression Discriminator

    [https://arxiv.org/abs/2402.09421](https://arxiv.org/abs/2402.09421)

    本文通过构建一个生成检测网络，利用脑电图信号学习与抑郁症相关的脑活动，并根据脑活动重新生成目标电极信号，从而实现了对不同类别脑电信号的分类判断。

    

    抑郁症是一种非常常见但严重的情绪障碍。在本文中，我们根据三个生理定律构建了一个生成检测网络（GDN）。我们的目标是希望神经网络能够学习与脑电图信号相关的脑活动，并根据脑活动重新生成目标电极信号。我们训练了两个生成器，第一个生成器学习抑郁症脑活动的特征，第二个生成器学习对照组脑活动的特征。在测试中，将一个脑电信号片段分别输入两个生成器，如果脑电信号与脑活动的关系符合某一类别的特征，那么由相应类别生成器生成的信号更接近原始信号。因此，可以确定与某一段脑电信号对应的类别。

    arXiv:2402.09421v1 Announce Type: cross  Abstract: Depression is a very common but serious mood disorder.In this paper, We built a generative detection network(GDN) in accordance with three physiological laws. Our aim is that we expect the neural network to learn the relevant brain activity based on the EEG signal and, at the same time, to regenerate the target electrode signal based on the brain activity. We trained two generators, the first one learns the characteristics of depressed brain activity, and the second one learns the characteristics of control group's brain activity. In the test, a segment of EEG signal was put into the two generators separately, if the relationship between the EEG signal and brain activity conforms to the characteristics of a certain category, then the signal generated by the generator of the corresponding category is more consistent with the original signal. Thus it is possible to determine the category corresponding to a certain segment of EEG signal. 
    
[^190]: 从对数频率轴上的高斯函数中导出的多维 Gabor-类滤波器

    Multidimensional Gabor-Like Filters Derived from Gaussian Functions on Logarithmic Frequency Axes

    [https://arxiv.org/abs/2402.09419](https://arxiv.org/abs/2402.09419)

    本文提出了一种新的类小波函数，通过对数频率轴上的高斯函数进行傅里叶逆变换，得到类似于Gabor滤波器的多维滤波器，它可以表示不同大小的定向短时信号振荡，并包含固有的低通滤波器。

    

    本文提出了一种新的类小波函数，该函数可方便地创建滤波器组，主要通过在频率域上对数频率轴上的高斯函数进行傅里叶逆变换来实现。得到的滤波器类似于 Gabor 滤波器，表示不同大小的定向短时信号振荡。该类小波函数可以看作是一种广义的对数-Gabor 滤波器，具有多维特性，始终使用对数频率轴上的高斯函数，并在频率域原点处固有地包含低通滤波器。

    arXiv:2402.09419v1 Announce Type: cross  Abstract: A novel wavelet-like function is presented that makes it convenient to create filter banks given mainly two parameters that influence the focus area and the filter count. This is accomplished by computing the inverse Fourier transform of Gaussian functions on logarithmic frequency axes in the frequency domain. The resulting filters are similar to Gabor filters and represent oriented brief signal oscillations of different sizes. The wavelet-like function can be thought of as a generalized Log-Gabor filter that is multidimensional, always uses Gaussian functions on logarithmic frequency axes, and innately includes low-pass filters from Gaussian functions located at the frequency domain origin.
    
[^191]: 蛋白质表示学习的深度流形转换

    Deep Manifold Transformation for Protein Representation Learning

    [https://arxiv.org/abs/2402.09416](https://arxiv.org/abs/2402.09416)

    提出了一种深度流形转换方法，用于优化蛋白质表示学习，通过应用流形学习策略和新的损失函数来提高学到的嵌入的质量和适应性。

    

    蛋白质表示学习在生物学的各个任务中都非常重要，如药物设计和蛋白质结构或功能预测，其中主要受益于蛋白质语言模型和图神经网络。这些模型可以通过掩蔽和与任务相关的损失函数捕捉到蛋白质序列和结构的内在模式。然而，学到的蛋白质表示通常不是很优化，导致性能下降，原因包括数据有限、难以适应新任务等。为了解决这个问题，我们提出了一种新的深度流形转换方法，用于通用蛋白质表示学习（DMTPRL）。它采用流形学习策略来改进学到的嵌入的质量和适应性。具体而言，我们在训练过程中应用了一种基于图间节点相似性的新型流形学习损失。

    arXiv:2402.09416v1 Announce Type: cross  Abstract: Protein representation learning is critical in various tasks in biology, such as drug design and protein structure or function prediction, which has primarily benefited from protein language models and graph neural networks. These models can capture intrinsic patterns from protein sequences and structures through masking and task-related losses. However, the learned protein representations are usually not well optimized, leading to performance degradation due to limited data, difficulty adapting to new tasks, etc. To address this, we propose a new \underline{d}eep \underline{m}anifold \underline{t}ransformation approach for universal \underline{p}rotein \underline{r}epresentation \underline{l}earning (DMTPRL). It employs manifold learning strategies to improve the quality and adaptability of the learned embeddings. Specifically, we apply a novel manifold learning loss during training based on the graph inter-node similarity. Our propos
    
[^192]: 通过信息论奖励建模来减轻奖励作弊问题

    Mitigating Reward Hacking via Information-Theoretic Reward Modeling

    [https://arxiv.org/abs/2402.09345](https://arxiv.org/abs/2402.09345)

    本文提出了一种名为InfoRM的奖励建模框架，通过引入变分信息瓶颈目标和模型复杂度调节机制，解决了奖励作弊问题，并利用集成聚类偏差得分（ICDS）来检测奖励过度优化。

    

    尽管强化学习从人类反馈（RLHF）中的成功在与人类价值观的语言模型的对齐方面，奖励作弊问题，也被称为奖励过度优化，仍然是一个关键挑战，主要源于奖励建模的局限性，即奖励模型的泛化能力和偏好数据集的不一致性。在这项工作中，我们从信息论的视角来解决这个问题，并提出了一种可推广和鲁棒的奖励建模框架，称为InfoRM，通过引入变分信息瓶颈目标来过滤出不相关的信息，并开发一种模型复杂度调节机制。值得注意的是，我们进一步发现了过度优化与潜变量空间的异常值之间的相关性，将InfoRM作为检测奖励过度优化的一种有前途的工具。受到这一发现的启发，我们提出了集成聚类偏差得分（ICDS），用于量化过优化问题。

    arXiv:2402.09345v1 Announce Type: cross Abstract: Despite the success of reinforcement learning from human feedback (RLHF) in aligning language models with human values, reward hacking, also termed reward overoptimization, remains a critical challenge, which primarily stems from limitations in reward modeling, i.e., generalizability of the reward model and inconsistency in the preference dataset. In this work, we tackle this problem from an information theoretic-perspective, and propose a generalizable and robust framework for reward modeling, namely InfoRM, by introducing a variational information bottleneck objective to filter out irrelevant information and developing a mechanism for model complexity modulation. Notably, we further identify a correlation between overoptimization and outliers in the latent space, establishing InfoRM as a promising tool for detecting reward overoptimization. Inspired by this finding, we propose the Integrated Cluster Deviation Score (ICDS), which quant
    
[^193]: EcoVal:一种高效的机器学习数据估值框架

    EcoVal: An Efficient Data Valuation Framework for Machine Learning

    [https://arxiv.org/abs/2402.09288](https://arxiv.org/abs/2402.09288)

    EcoVal是一种高效的机器学习数据估值框架，通过估计每个数据的内在和外在价值，实现了快速实用地估算机器学习模型数据的价值。

    

    在机器学习工作流中量化数据的价值可以在机器学习倡议中做出更具战略意义的决策中起到关键作用。现有的基于Shapley值的机器学习数据估值框架在计算方面非常昂贵，因为需要大量重复训练模型才能获得Shapley值。在本文中，我们介绍了一种高效的数据估值框架EcoVal，以快速实用的方式估算机器学习模型数据的价值。我们不直接处理独立的数据样本，而是确定类似的数据点簇的价值。这个价值进一步在所有成员簇点之间传播。我们展示了可以通过估计每个数据的内在和外在价值来确定整体数据价值。这是通过将模型的性能建模为“生产函数”来实现的，这是一个非常重要的概念。

    arXiv:2402.09288v1 Announce Type: new Abstract: Quantifying the value of data within a machine learning workflow can play a pivotal role in making more strategic decisions in machine learning initiatives. The existing Shapley value based frameworks for data valuation in machine learning are computationally expensive as they require considerable amount of repeated training of the model to obtain the Shapley value. In this paper, we introduce an efficient data valuation framework EcoVal, to estimate the value of data for machine learning models in a fast and practical manner. Instead of directly working with individual data sample, we determine the value of a cluster of similar data points. This value is further propagated amongst all the member cluster points. We show that the overall data value can be determined by estimating the intrinsic and extrinsic value of each data. This is enabled by formulating the performance of a model as a \textit{production function}, a concept which is po
    
[^194]: 使用图卷积神经网络的汽车碰撞安全结构动力学的多层次代理学习

    Multi-Hierarchical Surrogate Learning for Structural Dynamics of Automotive Crashworthiness Using Graph Convolutional Neural Networks

    [https://arxiv.org/abs/2402.09234](https://arxiv.org/abs/2402.09234)

    该论文提出了使用图卷积神经网络的多层次代理学习框架，用于汽车碰撞安全结构动力学研究。该框架能够通过创建一系列适应不同计算环境和准确度要求的代理模型，从而提高碰撞仿真的效率和精确度。

    

    碰撞仿真在提高车辆安全性、设计优化和伤害风险估计方面发挥着重要作用。然而，使用最先进的高保真模型进行这类问题的数值解需要大量的计算工作。传统的数据驱动代理建模方法通过创建低维嵌入来演化动力学，以规避这种计算工作。大多数方法直接在从数值离散化获取的高分辨率数据上操作，这既昂贵又复杂，无法在大范围的空间距离上映射信息流动。此外，使用固定分辨率的方法阻止了代理模型对计算能力环境、不同的可视化分辨率和不同的精确度要求进行自适应。因此，我们提出了一个多层次框架，用于结构化地创建一系列用于卡丁车碰撞安全性的代理模型。

    arXiv:2402.09234v1 Announce Type: new Abstract: Crash simulations play an essential role in improving vehicle safety, design optimization, and injury risk estimation. Unfortunately, numerical solutions of such problems using state-of-the-art high-fidelity models require significant computational effort. Conventional data-driven surrogate modeling approaches create low-dimensional embeddings for evolving the dynamics in order to circumvent this computational effort. Most approaches directly operate on high-resolution data obtained from numerical discretization, which is both costly and complicated for mapping the flow of information over large spatial distances. Furthermore, working with a fixed resolution prevents the adaptation of surrogate models to environments with variable computing capacities, different visualization resolutions, and different accuracy requirements. We thus propose a multi-hierarchical framework for structurally creating a series of surrogate models for a kart fr
    
[^195]: 快速采用，隐藏风险：大型语言模型定制的双重影响

    Rapid Adoption, Hidden Risks: The Dual Impact of Large Language Model Customization

    [https://arxiv.org/abs/2402.09179](https://arxiv.org/abs/2402.09179)

    本文介绍了针对不可信定制语言模型的指令后门攻击，通过在定制语言模型中设计带有后门指令的提示，实现攻击者预期的结果。攻击包括三个级别，不需要对后端语言模型进行任何修改。

    

    自然语言生成模型的定制化需求不断增加，导致了像GPT这样的解决方案的开发。这些解决方案通过自然语言提示来促进定制的语言模型的创建，无需编码。然而，第三方定制语言模型的可信度仍然是一个重要的问题。在本文中，我们提出了针对与不可信定制语言模型（例如GPT）集成的应用的首个指令后门攻击。具体来说，这些攻击通过设计带有后门指令的提示，将后门嵌入到定制语言模型的版本中，当输入包含预定义的触发器时，输出攻击者期望的结果。我们的攻击包括三个级别：单词级别、语法级别和语义级别，采用不同类型的触发器，并具有逐步隐蔽性。我们强调，我们的攻击不需要对后端语言模型进行微调或任何修改，严格遵循GPT的开发。

    arXiv:2402.09179v1 Announce Type: cross Abstract: The increasing demand for customized Large Language Models (LLMs) has led to the development of solutions like GPTs. These solutions facilitate tailored LLM creation via natural language prompts without coding. However, the trustworthiness of third-party custom versions of LLMs remains an essential concern. In this paper, we propose the first instruction backdoor attacks against applications integrated with untrusted customized LLMs (e.g., GPTs). Specifically, these attacks embed the backdoor into the custom version of LLMs by designing prompts with backdoor instructions, outputting the attacker's desired result when inputs contain the pre-defined triggers. Our attack includes 3 levels of attacks: word-level, syntax-level, and semantic-level, which adopt different types of triggers with progressive stealthiness. We stress that our attacks do not require fine-tuning or any modification to the backend LLMs, adhering strictly to GPTs devel
    
[^196]: 探索大型语言模型的对抗能力

    Exploring the Adversarial Capabilities of Large Language Models

    [https://arxiv.org/abs/2402.09132](https://arxiv.org/abs/2402.09132)

    本研究探索了大型语言模型的对抗能力，并发现其能够成功地制造对抗性示例以愚弄安全措施，特别是在仇恨言论检测方面具有重大影响。

    

    大型语言模型（LLMs）的普及引发了广泛和普遍的兴趣，因为它们具有强大的语言生成能力，为行业和研究提供了巨大的潜力。尽管以前的研究探讨了LLMs的安全性和隐私问题，但这些模型能否表现出对抗行为的程度仍然尚未完全探索。为了填补这一空白，我们研究常见的公开可用LLMs是否具有能力扰乱文本样本以愚弄安全措施，即所谓的对抗示例或攻击。更具体地说，我们调查LLMs是否本质上能够从良性样本中制造对抗性示例以愚弄现有的安全防线。我们的实验重点关注仇恨言论检测，发现LLMs成功地找到了对抗性扰动，有效地破坏了对仇恨言论检测系统的防御。我们的发现对（半）自动化安全评估和防御具有重要影响。

    arXiv:2402.09132v1 Announce Type: new Abstract: The proliferation of large language models (LLMs) has sparked widespread and general interest due to their strong language generation capabilities, offering great potential for both industry and research. While previous research delved into the security and privacy issues of LLMs, the extent to which these models can exhibit adversarial behavior remains largely unexplored. Addressing this gap, we investigate whether common publicly available LLMs have inherent capabilities to perturb text samples to fool safety measures, so-called adversarial examples resp.~attacks. More specifically, we investigate whether LLMs are inherently able to craft adversarial examples out of benign samples to fool existing safe rails. Our experiments, which focus on hate speech detection, reveal that LLMs succeed in finding adversarial perturbations, effectively undermining hate speech detection systems. Our findings carry significant implications for (semi-)aut
    
[^197]: 面向对抗性破坏的健壮模型驱动强化学习

    Towards Robust Model-Based Reinforcement Learning Against Adversarial Corruption

    [https://arxiv.org/abs/2402.08991](https://arxiv.org/abs/2402.08991)

    本研究通过引入对抗性健壮的乐观MLE（CR-OMLE）算法，解决了模型驱动强化学习中对抗性破坏的挑战，实现了对转移模型的健壮估计。

    

    本研究解决了模型驱动强化学习中对抗性破坏的挑战，其中转移动力学可以被对手破坏。现有研究主要集中在模型无关强化学习的情景下，通常采用健壮的最小二乘回归来进行值函数估计。然而，这些技术不能直接应用于模型驱动的强化学习。在本文中，我们专注于模型驱动的强化学习，并采用最大似然估计（MLE）方法来学习转移模型。我们的工作涵盖了在线和离线两种情况。在在线情况下，我们引入了一种名为对抗性健壮的乐观MLE（CR-OMLE）的算法，它利用基于总变差（TV）的信息比率作为MLE的不确定权重。我们证明了CR-OMLE的遗憾度为$ \tilde {\mathcal {O}}（\sqrt {T} + C）$，其中$ C $表示经过$ T $个回合后的累计破坏水平。

    arXiv:2402.08991v1 Announce Type: cross Abstract: This study tackles the challenges of adversarial corruption in model-based reinforcement learning (RL), where the transition dynamics can be corrupted by an adversary. Existing studies on corruption-robust RL mostly focus on the setting of model-free RL, where robust least-square regression is often employed for value function estimation. However, these techniques cannot be directly applied to model-based RL. In this paper, we focus on model-based RL and take the maximum likelihood estimation (MLE) approach to learn transition model. Our work encompasses both online and offline settings. In the online setting, we introduce an algorithm called corruption-robust optimistic MLE (CR-OMLE), which leverages total-variation (TV)-based information ratios as uncertainty weights for MLE. We prove that CR-OMLE achieves a regret of $\tilde{\mathcal{O}}(\sqrt{T} + C)$, where $C$ denotes the cumulative corruption level after $T$ episodes. We also pro
    
[^198]: 《对于数值逼近遍历SDE的分布的Wasserstein距离估计》修正

    Correction to "Wasserstein distance estimates for the distributions of numerical approximations to ergodic stochastic differential equations"

    [https://arxiv.org/abs/2402.08711](https://arxiv.org/abs/2402.08711)

    修正了《对于数值逼近遍历SDE的分布的Wasserstein距离估计》中的错误局部误差估计，提出了一种方法来分析数值离散遍历SDE的Wasserstein-2距离的非渐近保证，并解决了实践中维度依赖性的问题。

    

    本文对San-Serna和Zygalakis的《对于数值逼近遍历SDE的分布的Wasserstein距离估计》中的非渐近保证数值离散分析方法进行了修正。他们分析了UBU积分器，该积分器是二阶强型的，并且每个步骤只需要一次梯度评估，从而得到了理想的非渐近保证，特别是在Wasserstein-2距离中到达离目标分布 $\epsilon > 0$ 的距离仅需 $\mathcal{O}(d^{1/4}\epsilon^{-1/2})$ 步。然而，Sanz-Serna和Zygalakis (2021)中的局部误差估计存在错误，在实践中需要更强的假设才能实现这些复杂度估计。本文解决了理论与实践中观察到的许多应用场景中的维度依赖性。

    arXiv:2402.08711v1 Announce Type: cross Abstract: A method for analyzing non-asymptotic guarantees of numerical discretizations of ergodic SDEs in Wasserstein-2 distance is presented by Sanz-Serna and Zygalakis in ``Wasserstein distance estimates for the distributions of numerical approximations to ergodic stochastic differential equations". They analyze the UBU integrator which is strong order two and only requires one gradient evaluation per step, resulting in desirable non-asymptotic guarantees, in particular $\mathcal{O}(d^{1/4}\epsilon^{-1/2})$ steps to reach a distance of $\epsilon > 0$ in Wasserstein-2 distance away from the target distribution. However, there is a mistake in the local error estimates in Sanz-Serna and Zygalakis (2021), in particular, a stronger assumption is needed to achieve these complexity estimates. This note reconciles the theory with the dimension dependence observed in practice in many applications of interest.
    
[^199]: 图神经网络的同态计数：关于基础的一切

    Homomorphism Counts for Graph Neural Networks: All About That Basis

    [https://arxiv.org/abs/2402.08595](https://arxiv.org/abs/2402.08595)

    本研究展示了基于图神经网络的同态计数对于增强其表达能力的重要性，并提出了一种更细致的方法来融合目标模式的同态计数。这种方法比现有方法更具表达力且没有额外的计算复杂度开销。

    

    图神经网络是用于学习图上不变函数的架构。大量研究已经探讨了图神经网络的性质，并确定了一些限制，特别是与其表达能力相关的限制。它们无法计数图中的某些模式（例如循环）是这些限制的核心，因为许多需要学习的函数依赖于计数这些模式的能力。两种突出的范例旨在通过丰富图特征的子图或同态模式计数来解决这个限制。在这项工作中，我们展示了这两种方法在某种意义上都是次优的，并主张采用一种更细致的方法，将目标模式的“基础”中的同态计数纳入考虑。与现有方法相比，这产生了更加表达力的架构，而不会带来任何额外的计算复杂度开销。我们证明了一系列理论结论。

    Graph neural networks are architectures for learning invariant functions over graphs. A large body of work has investigated the properties of graph neural networks and identified several limitations, particularly pertaining to their expressive power. Their inability to count certain patterns (e.g., cycles) in a graph lies at the heart of such limitations, since many functions to be learned rely on the ability of counting such patterns. Two prominent paradigms aim to address this limitation by enriching the graph features with subgraph or homomorphism pattern counts. In this work, we show that both of these approaches are sub-optimal in a certain sense and argue for a more fine-grained approach, which incorporates the homomorphism counts of all structures in the "basis" of the target pattern. This yields strictly more expressive architectures without incurring any additional overhead in terms of computational complexity compared to existing approaches. We prove a series of theoretical r
    
[^200]: 通过去噪扩散恢复模型解决拉普拉斯算子的正向和反向问题

    Denoising Diffusion Restoration Tackles Forward and Inverse Problems for the Laplace Operator

    [https://arxiv.org/abs/2402.08563](https://arxiv.org/abs/2402.08563)

    本论文提出了一种新的方法，通过使用去噪扩散恢复模型（DDRM）解决了拉普拉斯算子的反向和正向问题，对于泊松方程的解和参数恢复有着显著的改善。

    

    扩散模型已成为一类有前景的生成模型，将噪声输入映射为逼真的图像。最近，它们被应用于生成偏微分方程（PDE）的解。然而，它们在拉普拉斯算子的反向问题上仍然存在困难，例如泊松方程，因为幅度较大的特征值会放大测量噪声。本文提出了一种新的方法，通过使用去噪扩散恢复模型（DDRM）来解决PDE的反向和正向问题。DDRM被用于线性反问题，通过利用线性算子的奇异值分解（SVD）来恢复原始干净信号。同样地，我们提出了一种方法来通过利用拉普拉斯算子的特征值和特征函数来恢复泊松方程的解和参数。我们的结果表明，使用去噪扩散恢复显著改善了估计结果。

    Diffusion models have emerged as a promising class of generative models that map noisy inputs to realistic images. More recently, they have been employed to generate solutions to partial differential equations (PDEs). However, they still struggle with inverse problems in the Laplacian operator, for instance, the Poisson equation, because the eigenvalues that are large in magnitude amplify the measurement noise. This paper presents a novel approach for the inverse and forward solution of PDEs through the use of denoising diffusion restoration models (DDRM). DDRMs were used in linear inverse problems to restore original clean signals by exploiting the singular value decomposition (SVD) of the linear operator. Equivalently, we present an approach to restore the solution and the parameters in the Poisson equation by exploiting the eigenvalues and the eigenfunctions of the Laplacian operator. Our results show that using denoising diffusion restoration significantly improves the estimation o
    
[^201]: 基于10万小时数据的10亿参数文本到语音模型的经验教训

    BASE TTS: Lessons from building a billion-parameter Text-to-Speech model on 100K hours of data

    [https://arxiv.org/abs/2402.08093](https://arxiv.org/abs/2402.08093)

    基于10万小时数据的10亿参数文本到语音模型BASE TTS在语音自然度上达到了最新技术水平，并且能够展现自然的韵律。

    

    我们介绍了一个名为BASE TTS的文本到语音（TTS）模型，其中BASE代表大规模自适应可流式TTS和新出现的能力。BASE TTS是迄今为止最大的TTS模型，训练于10万小时的公共领域语音数据，实现了语音自然度的最新技术水平。它采用了一个10亿参数的自回归Transformer，将原始文本转换为离散代码（"speechcodes"），然后通过基于卷积的解码器将这些speechcodes以增量、可流式的方式转换为波形。此外，我们的speechcodes采用了一种新颖的语音标记化技术，具有说话者ID解耦和字节对编码的压缩特性。与大量数据训练的大语言模型广泛报道的"新出现的能力"类似，我们展示了使用10K+小时和500M+参数构建的BASE TTS变体在文本复杂句子上开始展现自然的韵律。我们设计了...

    We introduce a text-to-speech (TTS) model called BASE TTS, which stands for $\textbf{B}$ig $\textbf{A}$daptive $\textbf{S}$treamable TTS with $\textbf{E}$mergent abilities. BASE TTS is the largest TTS model to-date, trained on 100K hours of public domain speech data, achieving a new state-of-the-art in speech naturalness. It deploys a 1-billion-parameter autoregressive Transformer that converts raw texts into discrete codes ("speechcodes") followed by a convolution-based decoder which converts these speechcodes into waveforms in an incremental, streamable manner. Further, our speechcodes are built using a novel speech tokenization technique that features speaker ID disentanglement and compression with byte-pair encoding. Echoing the widely-reported "emergent abilities" of large language models when trained on increasing volume of data, we show that BASE TTS variants built with 10K+ hours and 500M+ parameters begin to demonstrate natural prosody on textually complex sentences. We design
    
[^202]: 学习神经收缩动力学：扩展线性化和全局保证

    Learning Neural Contracting Dynamics: Extended Linearization and Global Guarantees

    [https://arxiv.org/abs/2402.08090](https://arxiv.org/abs/2402.08090)

    本论文提出了扩展线性化收缩动力学（ELCD），是第一个具有全局收缩性保证的神经网络动力系统，通过参数化非线性向量场的扩展线性化实现。通过在数据空间和潜在空间之间训练微分同胚，并在潜在空间中强制收缩性，ELCD能在面对不确定性时保持全局稳定性和鲁棒性。

    

    在学习的动态系统中，全局稳定性和鲁棒性保证对于确保系统在面对不确定性时的良好行为至关重要。我们提出了扩展线性化收缩动力学（ELCD），该系统是第一个具有任意度量下全局收缩性保证的基于神经网络的动力系统。ELCD的关键特性是非线性向量场扩展线性化的参数化。在其最基本形式下，ELCD保证全局指数稳定、平衡收缩以及在某些度量下全局收缩。为了实现在数据空间中相对于更一般度量的收缩，我们训练数据空间和潜在空间之间的微分同胚，并在潜在空间中强制收缩性，从而确保数据空间的全局收缩性。我们在2D、4D和8D的LASA数据集上展示了ELCD的性能。

    Global stability and robustness guarantees in learned dynamical systems are essential to ensure well-behavedness of the systems in the face of uncertainty. We present Extended Linearized Contracting Dynamics (ELCD), the first neural network-based dynamical system with global contractivity guarantees in arbitrary metrics. The key feature of ELCD is a parametrization of the extended linearization of the nonlinear vector field. In its most basic form, ELCD is guaranteed to be (i) globally exponentially stable, (ii) equilibrium contracting, and (iii) globally contracting with respect to some metric. To allow for contraction with respect to more general metrics in the data space, we train diffeomorphisms between the data space and a latent space and enforce contractivity in the latent space, which ensures global contractivity in the data space. We demonstrate the performance of ELCD on the $2$D, $4$D, and $8$D LASA datasets.
    
[^203]: 使用语言反馈模型来改进政策

    Policy Improvement using Language Feedback Models

    [https://arxiv.org/abs/2402.07876](https://arxiv.org/abs/2402.07876)

    本文介绍了一种使用语言反馈模型（LFMs）改进政策的方法，通过识别期望的行为并进行模仿学习，我们在任务完成率、泛化性能和人类可解释性方面取得了显著改进。

    

    我们引入了语言反馈模型（LFMs），用于在指令遵循中识别期望的行为-有助于实现指令中指定任务的行动-以进行模仿学习。为了训练LFMs，我们从大型语言模型（LLMs）获取对视觉轨迹进行语言描述的反馈。首先，通过使用LFMs识别期望模仿的行为，我们在三种不同的语言基础环境（Touchdown，ScienceWorld和ALFWorld）上，在任务完成率上改善了强行为克隆的基线方法。其次，与LLMs直接预测行动相比，使用LFMs在LLM输出标记的数量相同的情况下表现更好。第三，LFMs适应未见环境，通过一轮适应使任务完成率提高了3.5-12.0％。最后，可以修改LFM以提供人类可解释的反馈，无需性能损失，从而允许人类验证模仿学习的期望行为。

    We introduce Language Feedback Models (LFMs) that identify desirable behaviour - actions that help achieve tasks specified in the instruction - for imitation learning in instruction following. To train LFMs, we obtain feedback from Large Language Models (LLMs) on visual trajectories verbalized to language descriptions. First, by using LFMs to identify desirable behaviour to imitate, we improve in task-completion rate over strong behavioural cloning baselines on three distinct language grounding environments (Touchdown, ScienceWorld, and ALFWorld). Second, LFMs outperform using LLMs as experts to directly predict actions, when controlling for the number of LLM output tokens. Third, LFMs generalize to unseen environments, improving task-completion rate by 3.5-12.0% through one round of adaptation. Finally, LFM can be modified to provide human-interpretable feedback without performance loss, allowing human verification of desirable behaviour for imitation learning.
    
[^204]: 在时间领域中进行泛化：库普曼算子的应用

    Generalizing across Temporal Domains with Koopman Operators

    [https://arxiv.org/abs/2402.07834](https://arxiv.org/abs/2402.07834)

    本研究在时间领域泛化问题中提出了库普曼算子的应用，通过对齐条件分布来减小泛化界限。通过使用库普曼算子，我们可以有效地处理时变分布，从而解决时间领域泛化问题。

    

    在领域泛化的领域中，构建一种能够在没有目标数据的情况下适用于目标领域的预测模型仍然具有挑战性。当考虑到领域之间的演化动态时，这个问题变得更加复杂。虽然已经提出了各种方法来解决这个问题，但对基础泛化理论的全面理解仍然缺乏。在本研究中，我们提出了新的理论结果，通过对齐条件分布来减小泛化界限。我们的分析为通过应用库普曼神经算子来解决时间领域泛化问题提供了关键动机，从而产生了时间库普曼网络（TKNets）。通过使用库普曼算子，我们有效地使用库普曼理论的原则来处理时间领域泛化中遇到的时变分布，其中测量函数被用来建立线性过渡关系。

    In the field of domain generalization, the task of constructing a predictive model capable of generalizing to a target domain without access to target data remains challenging. This problem becomes further complicated when considering evolving dynamics between domains. While various approaches have been proposed to address this issue, a comprehensive understanding of the underlying generalization theory is still lacking. In this study, we contribute novel theoretic results that aligning conditional distribution leads to the reduction of generalization bounds. Our analysis serves as a key motivation for solving the Temporal Domain Generalization (TDG) problem through the application of Koopman Neural Operators, resulting in Temporal Koopman Networks (TKNets). By employing Koopman Operators, we effectively address the time-evolving distributions encountered in TDG using the principles of Koopman theory, where measurement functions are sought to establish linear transition relations betwe
    
[^205]: 基于上下文学习的通用链接预测器

    Universal link predictor by In-context Learning

    [https://arxiv.org/abs/2402.07738](https://arxiv.org/abs/2402.07738)

    这项工作介绍了一种基于上下文学习的通用链接预测器(UniLP)，它将启发式方法的广泛适用性和参数模型的模式学习能力相结合，实现了自主学习目标图中的链接模式并具有跨不同图的泛化能力。

    

    链接预测是图机器学习中的一项关键任务，其目标是推断图中缺失或未来的链接。传统方法利用基于广泛观察到的连接模式的启发式方法，具有广泛的适用性和泛化性，无需进行模型训练。尽管这些方法很有用，但它们受制于人为推导的启发式方法，缺乏数据驱动方法的适应性。相反，参数链接预测器擅长于从数据中自动学习连接模式并取得最先进的效果，但在不同图之间直接转移上存在问题。相反，它需要进行大量的训练和超参数优化来适应目标图。在这项工作中，我们介绍了通用链接预测器（UniLP），这是一种新颖的模型，将启发式方法的广泛适用性与参数模型的模式学习能力相结合。UniLP设计为自主学习目标图中的链接模式，并具有跨不同图的泛化能力。

    Link prediction is a crucial task in graph machine learning, where the goal is to infer missing or future links within a graph. Traditional approaches leverage heuristic methods based on widely observed connectivity patterns, offering broad applicability and generalizability without the need for model training. Despite their utility, these methods are limited by their reliance on human-derived heuristics and lack the adaptability of data-driven approaches. Conversely, parametric link predictors excel in automatically learning the connectivity patterns from data and achieving state-of-the-art but fail short to directly transfer across different graphs. Instead, it requires the cost of extensive training and hyperparameter optimization to adapt to the target graph. In this work, we introduce the Universal Link Predictor (UniLP), a novel model that combines the generalizability of heuristic approaches with the pattern learning capabilities of parametric models. UniLP is designed to autono
    
[^206]: MAGNETO：边缘人体活动识别的边缘AI--隐私和个性化

    MAGNETO: Edge AI for Human Activity Recognition -- Privacy and Personalization

    [https://arxiv.org/abs/2402.07180](https://arxiv.org/abs/2402.07180)

    本文提出了一种名为MAGNETO的边缘AI平台，通过从云端推向边缘进行增量人体活动学习，避免了云端与边缘设备之间的数据传输，实现了数据隐私保护、低延迟处理和高度个性化。

    

    人体活动识别（HAR）是一个成熟的领域，现代机器学习（ML）技术显著推动了其发展。尽管公司成功地将HAR整合到消费品中，但它们通常依赖于预定义的活动集，这限制了用户级（边缘设备）的个性化。尽管在增量学习方面取得了进展，能够使用新数据更新模型，但这通常发生在云端，需要定期在云端和边缘设备之间进行数据传输，从而引发数据隐私问题。在本文中，我们提出了一种名为MAGNETO的边缘AI平台，将HAR任务从云端推向边缘。MAGNETO允许在边缘设备上直接进行增量人体活动学习，而无需与云端进行任何数据交换。这可以提供强大的隐私保证、低处理延迟和高度的个性化。特别地，我们在Android设备上演示了MAGNETO，从数据采集到结果可视化，验证了整个流程。

    Human activity recognition (HAR) is a well-established field, significantly advanced by modern machine learning (ML) techniques. While companies have successfully integrated HAR into consumer products, they typically rely on a predefined activity set, which limits personalizations at the user level (edge devices). Despite advancements in Incremental Learning for updating models with new data, this often occurs on the Cloud, necessitating regular data transfers between cloud and edge devices, thus leading to data privacy issues. In this paper, we propose MAGNETO, an Edge AI platform that pushes HAR tasks from the Cloud to the Edge. MAGNETO allows incremental human activity learning directly on the Edge devices, without any data exchange with the Cloud. This enables strong privacy guarantees, low processing latency, and a high degree of personalization for users. In particular, we demonstrate MAGNETO in an Android device, validating the whole pipeline from data collection to result visua
    
[^207]: 自然语言强化学习

    Natural Language Reinforcement Learning

    [https://arxiv.org/abs/2402.07157](https://arxiv.org/abs/2402.07157)

    本研究将自然语言表示和强化学习原则相结合，提出了自然语言强化学习（NLRL）框架，解决了强化学习在样本效率低、解释性不足和缺乏监督信号等方面的限制问题，通过实验验证了其有效性和可解释性。

    

    强化学习（RL）在学习决策任务的策略方面展现出了令人瞩目的能力。然而，RL常常面临样本效率低、解释性不足和缺乏稀疏监督信号等问题的限制。为了解决这些问题，我们从人类学习过程中汲取灵感，引入了自然语言强化学习（NLRL），创新性地将RL原则与自然语言表示结合起来。具体而言，NLRL在自然语言空间中重新定义了任务目标、策略、价值函数、Bellman方程和策略迭代等RL概念。我们还展示了如何利用最新的大型语言模型（LLM）如GPT-4来实现NLRL。对表格MDPs的初步实验表明了NLRL框架的有效性、高效性和可解释性。

    Reinforcement Learning (RL) has shown remarkable abilities in learning policies for decision-making tasks. However, RL is often hindered by issues such as low sample efficiency, lack of interpretability, and sparse supervision signals. To tackle these limitations, we take inspiration from the human learning process and introduce Natural Language Reinforcement Learning (NLRL), which innovatively combines RL principles with natural language representation. Specifically, NLRL redefines RL concepts like task objectives, policy, value function, Bellman equation, and policy iteration in natural language space. We present how NLRL can be practically implemented with the latest advancements in large language models (LLMs) like GPT-4. Initial experiments over tabular MDPs demonstrate the effectiveness, efficiency, and also interpretability of the NLRL framework.
    
[^208]: 对于临床恶化预测的变分时间序列模型中预测变异性的解释

    Explain Variance of Prediction in Variational Time Series Models for Clinical Deterioration Prediction

    [https://arxiv.org/abs/2402.06808](https://arxiv.org/abs/2402.06808)

    本文提出了使用delta方法确定性地近似预测的变异性的方法，并采用SHAP方法来归因于变异的贡献。该方法适用于临床恶化预测中的变分时间序列模型，可以在提高预测精度的同时提供解释性。

    

    在医疗领域中，由于许多模型无关方法的应用，深度学习应用所作出的预测分数的可解释性得到了改善。然而，对于住院病人的每日或每小时恶化风险预测，不仅预测的风险概率分数很重要，风险分数的变异性也对辅助临床决策起着关键作用。在本文中，我们建议使用delta方法以确定性地近似预测的变异性，从而可以采用SHAP方法来归因于变异的贡献。通过对变分模型中的条件隐藏空间进行采样来估计预测的变异性，并基于变异性博弈的Shapley值将其传播到输入的临床变量上。该方法适用于变分循环神经网络和变分转换器等变分时间序列模型。我们进一步认为，变分时间序列模型非常适合在预测精度和解释性之间取得平衡。

    In healthcare, thanks to many model agnostic methods, explainability of the prediction scores made by deep learning applications has improved. However, we note that for daily or hourly risk of deterioration prediction of in-hospital patients, not only the predicted risk probability score matters, but also the variance of the risk scores play key roles in aiding clinical decision making. In this paper, we propose to use delta's method to approximate variance of prediction deterministically, such that the SHAP method can be adopted to attribute contribution of variance. The prediction variance is estimated by sampling the conditional hidden space in variational models and is propagated to input clinical variables based on Shapley values of the variance game. This approach works with variational time series models such as variational recurrent neural networks and variational transformers. We further argue that variational time series models are perfect fits for achieving a balance between
    
[^209]: 顺序流匹配用于生成建模

    Sequential Flow Matching for Generative Modeling

    [https://arxiv.org/abs/2402.06461](https://arxiv.org/abs/2402.06461)

    本文提出了一种称为SeqRF的新方法，用于通过直线化概率流来减小全局截断误差，并以此加速取样和提高综合质量。

    

    直接引导连续时间生成模型（例如扩散模型或基于流的模型）的概率流是通过数值解算器快速取样的关键。现有方法通过直接生成噪声和数据分布之间的联合分布的概率路径来学习线性路径。ODE模型的仿真速度慢的一个重要原因是ODE轨迹的高曲率导致的ODE求解器的全局截断误差，这会在低NFE范围内放大数值解算器的截断误差。为了解决这个挑战，我们提出了一种称为SeqRF的新方法，它是一种学习技术，用于直线化概率流以减小全局截断误差，从而加速取样并提高综合质量。通过理论和实证研究，我们首先观察到了SeqRF的直线化特性。

    Straightening the probability flow of the continuous-time generative models, such as diffusion models or flow-based models, is the key to fast sampling through the numerical solvers, existing methods learn a linear path by directly generating the probability path the joint distribution between the noise and data distribution. One key reason for the slow sampling speed of the ODE-based solvers that simulate these generative models is the global truncation error of the ODE solver, caused by the high curvature of the ODE trajectory, which explodes the truncation error of the numerical solvers in the low-NFE regime. To address this challenge, We propose a novel method called SeqRF, a learning technique that straightens the probability flow to reduce the global truncation error and hence enable acceleration of sampling and improve the synthesis quality. In both theoretical and empirical studies, we first observe the straightening property of our SeqRF. Through empirical evaluations via SeqR
    
[^210]: 物理信息神经网络的多尺度建模：从复杂系统的大尺度动力学到小尺度预测

    Multiscale Modelling with Physics-informed Neural Network: from Large-scale Dynamics to Small-scale Predictions in Complex Systems

    [https://arxiv.org/abs/2402.05067](https://arxiv.org/abs/2402.05067)

    本文提出了利用物理信息神经网络进行多尺度建模的方法，通过解耦大尺度和小尺度动力学，并在正交基函数空间中近似小尺度系统。实验结果表明该方法在处理液体动力学问题以及更复杂的情况下具有较高的有效性和适用性。

    

    多尺度现象在各个科学领域中普遍存在，对于准确有效地预测复杂系统中的多尺度动力学提出了普遍的挑战。本文提出了一种通过解耦方法对多尺度动力学进行表征的新的求解模式。通过独立地建模大尺度动力学，并将小尺度动力学视为从属系统，我们开发了一种谱PINN方法，在正交基函数空间中接近小尺度系统。通过大量的数值实验，包括一维Kuramot-Sivashinsky (KS)方程、二维和三维Navier-Stokes (NS)方程，我们展示了该方法的有效性，展示了它在液体动力学问题中的多样性。此外，我们还深入研究了该方法在更复杂问题中的应用，包括非均匀网格、复杂几何形状、带噪声的大尺度数据和高维小尺度动力学。

    Multiscale phenomena manifest across various scientific domains, presenting a ubiquitous challenge in accurately and effectively predicting multiscale dynamics in complex systems. In this paper, a novel solving mode is proposed for characterizing multiscale dynamics through a decoupling method. By modelling large-scale dynamics independently and treating small-scale dynamics as a slaved system, a Spectral PINN is developed to approach the small-scale system in an orthogonal basis functional space. The effectiveness of the method is demonstrated through extensive numerical experiments, including one-dimensional Kuramot-Sivashinsky (KS) equation, two- and three-dimensional Navier-Stokes (NS) equations, showcasing its versatility in addressing problems of fluid dynamics. Furthermore, we also delve into the application of the proposed approach to more complex problems, including non-uniform meshes, complex geometries, large-scale data with noise, and high-dimensional small-scale dynamics. 
    
[^211]: L4Q: 通过基于LoRA的量化训练在大型语言模型上提供参数高效的量化训练

    L4Q: Parameter Efficient Quantization-Aware Training on Large Language Models via LoRA-wise LSQ

    [https://arxiv.org/abs/2402.04902](https://arxiv.org/abs/2402.04902)

    L4Q是一种参数高效的量化感知训练算法，通过基于LoRA的学习的量化步长，解决了大型语言模型中量化训练的挑战。

    

    后训练量化(PTQ)和量化感知训练(QAT)方法正在流行起来，以缓解大型语言模型(LLMs)所带来的高内存和计算成本。在资源受限的情况下，尽管后者具有更高的准确性潜力，但由于其减少的训练开销，通常首选后训练量化。同时，介绍了参数高效微调方法，如低秩适应（LoRA），并最近的工作已经探索了量化感知参数高效微调技术。然而，这些方法可能缺乏通用性，因为它们依赖于预量化模型的配置。由非线性量化或混合精度权重引起的效果可能会受到影响，并且重新训练特定量化参数可能会影响最优性能。为了应对这些挑战，我们提出了L4Q，一种参数高效的量化感知训练算法。L4Q利用了基于LoRA的学习的量化步长。

    Post-training quantization (PTQ) and quantization-aware training (QAT) methods are gaining popularity in mitigating the high memory and computational costs associated with Large Language Models (LLMs). In resource-constrained scenarios, PTQ, with its reduced training overhead, is often preferred over QAT, despite the latter's potential for higher accuracy. Meanwhile, parameter-efficient fine-tuning (PEFT) methods like low-rank adaptation (LoRA) have been introduced, and recent efforts have explored quantization-aware PEFT techniques. However, these approaches may lack generality due to their reliance on the pre-quantized model's configuration. Their effectiveness may be compromised by non-linearly quantized or mixed-precision weights, and the retraining of specific quantization parameters might impede optimal performance. To address these challenges, we propose L4Q, an algorithm for parameter-efficient quantization-aware training. L4Q leverages LoRA-wise learned quantization step size 
    
[^212]: 视觉语言模型的开放词汇校准

    Open-Vocabulary Calibration for Vision-Language Models

    [https://arxiv.org/abs/2402.04655](https://arxiv.org/abs/2402.04655)

    本文研究了视觉语言模型中的开放词汇校准问题，在提示学习的背景下发现现有的校准方法不足以解决该问题。为此，提出了一种称为 Distance-Aware Ca 的简单而有效的方法来解决问题。

    

    视觉语言模型 (VLM) 已经成为强大的工具，在处理图像识别、文本驱动的视觉内容生成、视觉聊天机器人等各种开放词汇任务上展现出了强大的能力。近年来，人们在提高 VLM 下游性能的适应方法上投入了大量的努力和资源，尤其是在参数高效的微调方法（如提示学习）上。然而，一个被大大忽视的关键问题是在微调的 VLM 中的置信度校准问题，在实际部署这样的模型时会大大降低可靠性。本文通过系统地研究提示学习背景下的置信度校准问题，发现现有的校准方法不能解决这个问题，尤其是在开放词汇的设置中。为了解决这个问题，我们提出了一种简单而有效的方法，称为 "Distance-Aware Ca"

    Vision-language models (VLMs) have emerged as formidable tools, showing their strong capability in handling various open-vocabulary tasks in image recognition, text-driven visual content generation, and visual chatbots, to name a few. In recent years, considerable efforts and resources have been devoted to adaptation methods for improving downstream performance of VLMs, particularly on parameter-efficient fine-tuning methods like prompt learning. However, a crucial aspect that has been largely overlooked is the confidence calibration problem in fine-tuned VLMs, which could greatly reduce reliability when deploying such models in the real world. This paper bridges the gap by systematically investigating the confidence calibration problem in the context of prompt learning and reveals that existing calibration methods are insufficient to address the problem, especially in the open-vocabulary setting. To solve the problem, we present a simple and effective approach called Distance-Aware Ca
    
[^213]: 关于现代Hopfield模型计算限制的一个细粒度复杂性分析

    On Computational Limits of Modern Hopfield Models: A Fine-Grained Complexity Analysis

    [https://arxiv.org/abs/2402.04520](https://arxiv.org/abs/2402.04520)

    通过细粒度复杂性分析，我们研究了现代Hopfield模型的记忆检索计算限制，发现了一种基于模式范数的相变行为，并且建立了有效变体的上界条件。使用低秩逼近的方法，我们提供了有效构造的示例，同时证明了计算时间下界、记忆检索误差界和指数记忆容量。

    

    我们从细粒度复杂性分析的角度研究了现代Hopfield模型的记忆检索动力学的计算限制。我们的主要贡献是基于模式的范数对所有可能的现代Hopfield模型的效率进行相变行为的刻画。具体来说，我们建立了对输入查询模式和记忆模式的范数的上界标准。仅在这个标准之下，假设满足Strong Exponential Time Hypothesis (SETH)，存在子二次（高效）变体的现代Hopfield模型。为了展示我们的理论，当有效标准成立时，我们提供了现代Hopfield模型使用低秩逼近的有效构造的正式示例。这包括一个计算时间的下界导出，与$\Max\{$存储的记忆模式数量，输入查询序列的长度$\}$线性缩放。此外，我们证明了记忆检索误差界和指数记忆容量。

    We investigate the computational limits of the memory retrieval dynamics of modern Hopfield models from the fine-grained complexity analysis. Our key contribution is the characterization of a phase transition behavior in the efficiency of all possible modern Hopfield models based on the norm of patterns. Specifically, we establish an upper bound criterion for the norm of input query patterns and memory patterns. Only below this criterion, sub-quadratic (efficient) variants of the modern Hopfield model exist, assuming the Strong Exponential Time Hypothesis (SETH). To showcase our theory, we provide a formal example of efficient constructions of modern Hopfield models using low-rank approximation when the efficient criterion holds. This includes a derivation of a lower bound on the computational time, scaling linearly with $\Max\{$# of stored memory patterns, length of input query sequence$\}$. In addition, we prove its memory retrieval error bound and exponential memory capacity.
    
[^214]: 推荐系统中的交叉双边公平性

    Intersectional Two-sided Fairness in Recommendation

    [https://arxiv.org/abs/2402.02816](https://arxiv.org/abs/2402.02816)

    本文针对推荐系统中的交叉双边公平性问题，提出了一种名为交叉双边公平推荐（ITFR）的新方法，通过利用锐度感知损失感知劣势群体，使用协作损失平衡开发不同交叉群体的一致区分能力，并利用预测得分归一化来公平对待不同交叉群体中的正例。实验证明该方法在提高交叉双边公平性方面取得了显著效果。

    

    近年来，推荐系统的公平性引起了越来越多的关注。根据涉及的利益相关者，推荐系统的公平性可分为用户公平性、物品公平性和同时考虑用户和物品公平性的双边公平性。然而，我们认为即使推荐系统是双边公平的，交叉双边不公平仍然可能存在，这在本文使用真实世界数据的实证研究中得到了观察和展示，并且以前尚未得到很好的研究。为了缓解这个问题，我们提出了一种新方法，称为交叉双边公平推荐（ITFR）。我们的方法利用一个锐度感知损失来感知劣势群体，然后使用协作损失平衡来开发不同交叉群体的一致区分能力。此外，我们利用预测得分归一化来调整正面预测得分，以公平地对待不同交叉群体中的正例。广泛的实验结果表明，我们的方法在提高交叉双边公平性方面取得了显著的效果。

    Fairness of recommender systems (RS) has attracted increasing attention recently. Based on the involved stakeholders, the fairness of RS can be divided into user fairness, item fairness, and two-sided fairness which considers both user and item fairness simultaneously. However, we argue that the intersectional two-sided unfairness may still exist even if the RS is two-sided fair, which is observed and shown by empirical studies on real-world data in this paper, and has not been well-studied previously. To mitigate this problem, we propose a novel approach called Intersectional Two-sided Fairness Recommendation (ITFR). Our method utilizes a sharpness-aware loss to perceive disadvantaged groups, and then uses collaborative loss balance to develop consistent distinguishing abilities for different intersectional groups. Additionally, predicted score normalization is leveraged to align positive predicted scores to fairly treat positives in different intersectional groups. Extensive experime
    
[^215]: 具有过多风险的鲁棒多任务学习

    Robust Multi-Task Learning with Excess Risks

    [https://arxiv.org/abs/2402.02009](https://arxiv.org/abs/2402.02009)

    提出了一种具有过多风险的多任务学习（ExcessMTL）方法，根据任务到收敛的距离来更新任务权重，以克服存在标签噪声时现有方法的限制。

    

    多任务学习（MTL）通过优化所有任务损失的凸组合来考虑为多个任务学习一个联合模型。为了解决优化问题，现有方法使用自适应权重更新方案，根据各自的损失动态调整任务权重，以优先考虑困难任务。然而，在存在标签噪声的情况下，这些算法会面临巨大挑战，因为过多的权重往往被分配给具有相对较大贝叶斯最优误差的噪声任务，从而掩盖其他任务并导致整体性能下降。为了克服这个限制，我们提出了具有过多风险的多任务学习（ExcessMTL），这是一种基于过多风险的任务平衡方法，通过任务到收敛的距离来更新任务权重。直观来说，ExcessMTL将更高的权重分配给较差训练的距离收敛较远的任务。为了估计过多风险，我们开发了一种高效而准确的方法。

    Multi-task learning (MTL) considers learning a joint model for multiple tasks by optimizing a convex combination of all task losses. To solve the optimization problem, existing methods use an adaptive weight updating scheme, where task weights are dynamically adjusted based on their respective losses to prioritize difficult tasks. However, these algorithms face a great challenge whenever label noise is present, in which case excessive weights tend to be assigned to noisy tasks that have relatively large Bayes optimal errors, thereby overshadowing other tasks and causing performance to drop across the board. To overcome this limitation, we propose Multi-Task Learning with Excess Risks (ExcessMTL), an excess risk-based task balancing method that updates the task weights by their distances to convergence instead. Intuitively, ExcessMTL assigns higher weights to worse-trained tasks that are further from convergence. To estimate the excess risks, we develop an efficient and accurate method 
    
[^216]: 对抗性量子机器学习：信息论的泛化分析

    Adversarial Quantum Machine Learning: An Information-Theoretic Generalization Analysis

    [https://arxiv.org/abs/2402.00176](https://arxiv.org/abs/2402.00176)

    本文研究了对抗性训练的量子分类器的泛化特性，并提出了新颖的信息论上界。

    

    类似于经典分类器，量子分类器也容易受到扰动其输入的对抗性攻击。一种有希望的对策是采用一个攻击感知或对抗性的损失函数来训练量子分类器。本文研究了针对有界范数白盒攻击进行对抗性训练的量子分类器的泛化特性。具体来说，量子对手通过将输入状态ρ(x)转化为与原始状态ρ(x)在p-Schatten距离上ε接近的状态λ来最大化分类器的损失。在量子嵌入ρ(x)的适当假设下，我们对对抗性训练的量子分类器在p = 1和p = ∞时的泛化误差导出了新颖的信息论上界。导出的上界包含两个项：第一个是经典数据和量子嵌入之间的2-Rényi相互信息的指数函数，

    In a manner analogous to their classical counterparts, quantum classifiers are vulnerable to adversarial attacks that perturb their inputs. A promising countermeasure is to train the quantum classifier by adopting an attack-aware, or adversarial, loss function. This paper studies the generalization properties of quantum classifiers that are adversarially trained against bounded-norm white-box attacks. Specifically, a quantum adversary maximizes the classifier's loss by transforming an input state $\rho(x)$ into a state $\lambda$ that is $\epsilon$-close to the original state $\rho(x)$ in $p$-Schatten distance. Under suitable assumptions on the quantum embedding $\rho(x)$, we derive novel information-theoretic upper bounds on the generalization error of adversarially trained quantum classifiers for $p = 1$ and $p = \infty$. The derived upper bounds consist of two terms: the first is an exponential function of the 2-R\'enyi mutual information between classical data and quantum embedding,
    
[^217]: SWEA:通过主题词嵌入修改改变大型语言模型中的事实知识

    SWEA: Changing Factual Knowledge in Large Language Models via Subject Word Embedding Altering

    [https://arxiv.org/abs/2401.17809](https://arxiv.org/abs/2401.17809)

    提出了一种主题词嵌入修改框架（SWEA），通过在推理阶段修改主题的表示来编辑知识，保护模型的原始权重，避免不可逆的损害和额外的推理开销。

    

    模型编辑近来引起了广泛关注。目前的模型编辑方法主要涉及修改模型参数或向现有模型添加附加模块。然而，前者会对LLM造成不可逆的影响，而后者会产生额外的推理开销，并且模糊的向量匹配并不总是可靠的。为了解决这些问题，我们提出了一种可扩展的主题词嵌入修改（SWEA）框架，它在推理阶段修改主题的表示，并实现编辑知识的目标。SWEA在模型外部使用精确的关键匹配，并进行可靠的主题词嵌入修改，从而保护模型的原始权重而不增加推理开销。然后，我们提出优化抑制融合方法，首先优化编辑目标的嵌入向量，然后抑制知识嵌入维度（KED）以获得最终融合的嵌入。我们因此提出了SWEAOS元方法。

    Model editing has recently gained widespread attention. Current model editing methods primarily involve modifying model parameters or adding additional modules to the existing model. However, the former causes irreversible damage to LLMs, while the latter incurs additional inference overhead and fuzzy vector matching is not always reliable. To address these issues, we propose an expandable Subject Word Embedding Altering (SWEA) framework, which modifies the representation of subjects and achieve the goal of editing knowledge during the inference stage. SWEA uses precise key matching outside the model and performs reliable subject word embedding altering, thus protecting the original weights of the model without increasing inference overhead. We then propose optimizing then suppressing fusion method, which first optimizes the embedding vector for the editing target and then suppresses the Knowledge Embedding Dimension (KED) to obtain the final fused embedding. We thus propose SWEAOS met
    
[^218]: OntoMedRec: 用于药物推荐的基于本体编码器的逻辑预训练模型无关性方法

    OntoMedRec: Logically-Pretrained Model-Agnostic Ontology Encoders for Medication Recommendation

    [https://arxiv.org/abs/2401.15814](https://arxiv.org/abs/2401.15814)

    OntoMedRec是一种基于本体编码器的逻辑预训练、模型无关的医学推荐方法，通过解决医学本体数据稀缺问题，提高了各种模型在EHR数据集和少量药物的入院情况下的性能。

    

    大多数现有的药物推荐模型通过基于电子健康记录（EHR）学习医学概念的表示，并使用学习到的表示进行推荐。然而，大多数药物在数据集中出现的时间有限，导致了其表示的学习不足。医学本体是医学术语的分层分类系统，相似的术语在某个层次上属于同一类别。本文提出了OntoMedRec，一种逻辑预训练和模型无关的医学本体编码器，用于解决医学本体数据稀缺问题。我们在基准数据集上进行了全面的实验，评估了OntoMedRec的有效性，结果表明OntoMedRec的集成改善了各种模型在整个EHR数据集和仅有少量药物的入院情况下的性能。

    arXiv:2401.15814v2 Announce Type: replace  Abstract: Most existing medication recommendation models learn representations for medical concepts based on electronic health records (EHRs) and make recommendations with learnt representations. However, most medications appear in the dataset for limited times, resulting in insufficient learning of their representations. Medical ontologies are the hierarchical classification systems for medical terms where similar terms are in the same class on a certain level. In this paper, we propose OntoMedRec, the logically-pretrained and model-agnostic medical Ontology Encoders for Medication Recommendation that addresses data sparsity problem with medical ontologies. We conduct comprehensive experiments on benchmark datasets to evaluate the effectiveness of OntoMedRec, and the result shows the integration of OntoMedRec improves the performance of various models in both the entire EHR datasets and the admissions with few-shot medications. We provide the
    
[^219]: 在自组织映射中使用拓扑投影进行最小程度监督学习

    Minimally Supervised Learning using Topological Projections in Self-Organizing Maps

    [https://arxiv.org/abs/2401.06923](https://arxiv.org/abs/2401.06923)

    这篇论文介绍了一种基于自组织映射的拓扑投影半监督学习方法，可以有效利用大量无标签数据集中的信息，显著降低进行参数预测所需的标记数据点数量。

    

    参数预测对于许多应用至关重要，有助于深入解释和决策。然而，在许多现实生活领域，如电力系统、医学和工程领域，为某些数据集获取地面真实标签可能非常昂贵，因为它们可能需要广泛和昂贵的实验室测试。在本研究中，我们引入了一种基于自组织映射（SOM）的拓扑投影的半监督学习方法，可以显著减少所需的标记数据点数量以进行参数预测，有效地利用大型无标签数据集中包含的信息。我们提出的方法首先在无标签数据上训练SOM，然后将少量可用的标记数据点分配给关键的最佳匹配单元（BMU）。对于新遇到的数据点，利用最接近的n个标记数据点的平均值进行估计。

    arXiv:2401.06923v2 Announce Type: replace  Abstract: Parameter prediction is essential for many applications, facilitating insightful interpretation and decision-making. However, in many real life domains, such as power systems, medicine, and engineering, it can be very expensive to acquire ground truth labels for certain datasets as they may require extensive and expensive laboratory testing. In this work, we introduce a semi-supervised learning approach based on topological projections in self-organizing maps (SOMs), which significantly reduces the required number of labeled data points to perform parameter prediction, effectively exploiting information contained in large unlabeled datasets. Our proposed method first trains SOMs on unlabeled data and then a minimal number of available labeled data points are assigned to key best matching units (BMU). The values estimated for newly-encountered data points are computed utilizing the average of the $n$ closest labeled data points in the
    
[^220]: 通过数据增强和动态抽样方法增强神经定理证明能力

    Enhancing Neural Theorem Proving through Data Augmentation and Dynamic Sampling Method

    [https://arxiv.org/abs/2312.14188](https://arxiv.org/abs/2312.14188)

    本论文提出了一种名为DS-Prover的动态抽样方法，用于增强神经定理证明的能力。该方法通过动态确定应用于扩展当前目标的策略数量，并调整探索和开发之间的平衡，从而使证明搜索过程更加高效。此外，作者还通过增加训练数据集，将简化和重写策略与多个前提进行分解。

    

    定理证明是数学中的一项基本任务。随着大型语言模型（LLMs）和交互式定理证明器（ITPs）如Lean的出现，人们对将LLMs和ITPs集成以自动化定理证明的兴趣日益增长。在这种方法中，LLM生成证明步骤（策略），而ITP检查这些策略在当前目标上的适用性。这两个系统共同完成证明过程。在本文中，我们介绍了DS-Prover，一种用于定理证明的全新动态抽样方法。该方法通过动态确定要应用于扩展当前目标的策略数量，考虑到剩余时间与总分配时间之间的比较，从而使证明搜索过程更加高效，随着时间的推移调整探索和开发之间的平衡。我们还通过将简化和重写策略与多个前提进行分解来增加训练数据集。

    arXiv:2312.14188v2 Announce Type: replace  Abstract: Theorem proving is a fundamental task in mathematics. With the advent of large language models (LLMs) and interactive theorem provers (ITPs) like Lean, there has been growing interest in integrating LLMs and ITPs to automate theorem proving. In this approach, the LLM generates proof steps (tactics), and the ITP checks the applicability of the tactics at the current goal. The two systems work together to complete the proof. In this paper, we introduce DS-Prover, a novel dynamic sampling method for theorem proving. This method dynamically determines the number of tactics to apply to expand the current goal, taking into account the remaining time compared to the total allocated time for proving a theorem. This makes the proof search process more efficient by adjusting the balance between exploration and exploitation as time passes. We also augment the training dataset by decomposing simplification and rewrite tactics with multiple premi
    
[^221]: 保护您的分数：具有差分隐私保障的接触追踪

    Protect Your Score: Contact Tracing With Differential Privacy Guarantees

    [https://arxiv.org/abs/2312.11581](https://arxiv.org/abs/2312.11581)

    这篇论文提出了具有差分隐私保障的接触追踪算法，以解决隐私问题限制接触追踪的部署。该算法在多种情景下展现了卓越性能，并通过在发布每个风险分数时保护个体健康状况的隐私。

    

    2020年和2021年的流行病对经济和社会产生了巨大的影响，研究表明，接触追踪算法可以在早期遏制病毒方面起到关键作用。尽管在更有效的接触追踪算法方面已经取得了重大进展，但我们认为目前的隐私问题阻碍了其部署。接触追踪算法的本质在于传递一个风险分数的通信。然而，恰恰是将这个分数传递给用户，对手可以利用这个分数来评估个体的私人健康状况。我们确定了一个现实的攻击场景，并针对这种攻击提出了具有差分隐私保障的接触追踪算法。该算法在两个最常用的基于代理的COVID19模拟器上进行了测试，并在各种情景下展现了卓越性能，特别是在逼真的测试场景中，同时发布每个风险分数时。

    arXiv:2312.11581v2 Announce Type: replace-cross  Abstract: The pandemic in 2020 and 2021 had enormous economic and societal consequences, and studies show that contact tracing algorithms can be key in the early containment of the virus. While large strides have been made towards more effective contact tracing algorithms, we argue that privacy concerns currently hold deployment back. The essence of a contact tracing algorithm constitutes the communication of a risk score. Yet, it is precisely the communication and release of this score to a user that an adversary can leverage to gauge the private health status of an individual. We pinpoint a realistic attack scenario and propose a contact tracing algorithm with differential privacy guarantees against this attack. The algorithm is tested on the two most widely used agent-based COVID19 simulators and demonstrates superior performance in a wide range of settings. Especially for realistic test scenarios and while releasing each risk score w
    
[^222]: 学习自发现：关于积极抑制人工神经网络单意义神经元的研究

    Learning from Emergence: A Study on Proactively Inhibiting the Monosemantic Neurons of Artificial Neural Networks

    [https://arxiv.org/abs/2312.11560](https://arxiv.org/abs/2312.11560)

    本文研究了积极抑制人工神经网络中的单意义神经元，这对于提高性能具有重要意义，并提出了一种基于自发现的方法来实现抑制。

    

    最近，随着大型语言模型的成功，自发现受到了研究界的广泛关注。与现有文献不同，我们提出了一个关键因素的假设，即在规模扩大的过程中高度促进性能的因素：减少只能与特定特征形成一对一关系的单意义神经元。单意义神经元往往更稀疏，并对大型模型的性能产生负面影响。受到这一观点的启发，我们提出了一种直观的思路来识别和抑制单意义神经元。然而，实现这一目标是一个非平凡的任务，因为没有统一的定量评估指标，简单地禁止单意义神经元并不能促进神经网络的多意思性。因此，本文提出了从自发现中学习的方法，并展开了关于积极抑制单意义神经元的研究。具体来说，我们首先提出了一种新的方法

    arXiv:2312.11560v2 Announce Type: replace-cross  Abstract: Recently, emergence has received widespread attention from the research community along with the success of large language models. Different from the literature, we hypothesize a key factor that highly promotes the performance during the increase of scale: the reduction of monosemantic neurons that can only form one-to-one correlations with specific features. Monosemantic neurons tend to be sparser and have negative impacts on the performance in large models. Inspired by this insight, we propose an intuitive idea to identify monosemantic neurons and inhibit them. However, achieving this goal is a non-trivial task as there is no unified quantitative evaluation metric and simply banning monosemantic neurons does not promote polysemanticity in neural networks. Therefore, we propose to learn from emergence and present a study on proactively inhibiting the monosemantic neurons in this paper. More specifically, we first propose a new
    
[^223]: GINN-LP：一种用于发现多元Laurent多项式方程的可解释性神经网络

    GINN-LP: A Growing Interpretable Neural Network for Discovering Multivariate Laurent Polynomial Equations

    [https://arxiv.org/abs/2312.10913](https://arxiv.org/abs/2312.10913)

    GINN-LP是一种可解释的神经网络，用于发现多元Laurent多项式方程的形式和系数。它采用了一种名为“幂项逼近块”的新型可解释性神经网络块，并通过神经网络增长策略和稀疏正则化来优化方程的表示。

    

    传统机器学习通常被视为一个黑盒优化问题，不会产生将输入和输出连接起来的可解释性函数。然而，发现这种可解释性函数的能力是可取的。在这项工作中，我们提出了GINN-LP，一种可解释的神经网络，用于发现数据集的基础方程的形式和系数，当假设方程的形式是多元Laurent多项式时。这是通过一种新的可解释性神经网络块，名为“幂项逼近块”，由对数和指数激活函数组成来实现的。GINN-LP是端到端可微分的，可以使用反向传播进行训练。我们提出了一种神经网络增长策略，能够找到代表数据的Laurent多项式中的合适项数，同时还提出了稀疏正则化方法来优化方程的稀疏性。

    arXiv:2312.10913v2 Announce Type: replace-cross  Abstract: Traditional machine learning is generally treated as a black-box optimization problem and does not typically produce interpretable functions that connect inputs and outputs. However, the ability to discover such interpretable functions is desirable. In this work, we propose GINN-LP, an interpretable neural network to discover the form and coefficients of the underlying equation of a dataset, when the equation is assumed to take the form of a multivariate Laurent Polynomial. This is facilitated by a new type of interpretable neural network block, named the "power-term approximator block", consisting of logarithmic and exponential activation functions. GINN-LP is end-to-end differentiable, making it possible to use backpropagation for training. We propose a neural network growth strategy that will enable finding the suitable number of terms in the Laurent polynomial that represents the data, along with sparsity regularization to 
    
[^224]: 针对增强学习代理的个性化路径补救方法

    Personalized Path Recourse for Reinforcement Learning Agents

    [https://arxiv.org/abs/2312.08724](https://arxiv.org/abs/2312.08724)

    该论文介绍了一种针对增强学习代理的个性化路径补救方法，该方法通过编辑动作路径来实现期望目标，同时保持与代理的原始路径相似度高，并且个性化适应代理的行为模式。这种方法适用于纠正或改进动作或数据序列以实现预定目标。

    

    这篇论文介绍了一种名为个性化路径补救的新方法，用于为增强学习代理生成补救路径。其目标是通过编辑给定的动作路径以达到期望的目标（例如，与代理的原始路径相比取得更好的结果），同时确保与代理的原始路径高度相似并个性化适应代理。个性化是指新路径在从策略函数中观察到的代理行为模式方面的定制程度。我们训练一个个性化的补救代理来生成这样的个性化路径，这些路径是使用考虑目标、相似性和个性化的奖励函数获得的。该方法适用于增强学习和监督学习设置，以纠正或改进动作序列或数据序列以达到预定的目标。该方法在不同的设置中进行了评估。实验证明

    arXiv:2312.08724v2 Announce Type: replace-cross  Abstract: This paper introduces Personalized Path Recourse, a novel method that generates recourse paths for a reinforcement learning agent. The goal is to edit a given path of actions to achieve desired goals (e.g., better outcomes compared to the agent's original path) while ensuring a high similarity to the agent's original paths and being personalized to the agent. Personalization refers to the extent to which the new path is tailored to the agent's observed behavior patterns from their policy function. We train a personalized recourse agent to generate such personalized paths, which are obtained using reward functions that consider the goal, similarity, and personalization. The proposed method is applicable to both reinforcement learning and supervised learning settings for correcting or improving sequences of actions or sequences of data to achieve a pre-determined goal. The method is evaluated in various settings. Experiments show
    
[^225]: 预测顶点失败的连通性预测器

    Connectivity Oracles for Predictable Vertex Failures

    [https://arxiv.org/abs/2312.08489](https://arxiv.org/abs/2312.08489)

    论文研究了在预测算法范式下设计支持顶点失败的连通性预测器的问题，并提出了一种数据结构，能够以预处理时间和查询时间的多项式关系来处理失败顶点集合。

    

    设计支持顶点失败的连通性预测器是针对无向图的基本数据结构问题之一。已有的研究在查询时间方面已经有了很好的理解：以前的作品[Duan-Pettie STOC'10; Long-Saranurak FOCS'22]实现了与失败顶点数量成线性关系的查询时间，并且在需要多项式时间的预处理和多项式时间的更新的条件下是有条件最优的。我们在预测算法的范式下重新审视了这个问题：我们问，如果可以预测到失败顶点集合，查询时间是否可以提高。更具体地说，我们设计了一个数据结构，给定一个图G=(V,E)和一个预测会失败的顶点集合\widehat{D} \subseteq V（其中d=|\widehat{D}|），将其预处理时间为$\tilde{O}(d|E|)$，然后可以接收一个更新，该更新以对称差分形式给出。

    arXiv:2312.08489v2 Announce Type: replace-cross  Abstract: The problem of designing connectivity oracles supporting vertex failures is one of the basic data structures problems for undirected graphs. It is already well understood: previous works [Duan--Pettie STOC'10; Long--Saranurak FOCS'22] achieve query time linear in the number of failed vertices, and it is conditionally optimal as long as we require preprocessing time polynomial in the size of the graph and update time polynomial in the number of failed vertices.   We revisit this problem in the paradigm of algorithms with predictions: we ask if the query time can be improved if the set of failed vertices can be predicted beforehand up to a small number of errors. More specifically, we design a data structure that, given a graph $G=(V,E)$ and a set of vertices predicted to fail $\widehat{D} \subseteq V$ of size $d=|\widehat{D}|$, preprocesses it in time $\tilde{O}(d|E|)$ and then can receive an update given as the symmetric differ
    
[^226]: 可推广的Transformer预训练用于超长时间序列预测

    Extrapolatable Transformer Pre-training for Ultra Long Time-Series Forecasting

    [https://arxiv.org/abs/2312.00817](https://arxiv.org/abs/2312.00817)

    提出了一种名为TimelyGPT的可推广的Transformer预训练模型，该模型通过可推广的位置嵌入和循环注意力以及时间卷积模块有效地捕捉超长时间序列数据中的全局和局部时间依赖关系。

    

    大规模预训练模型（PTMs），如BERT和GPT，最近在自然语言处理和计算机视觉领域取得了巨大成功。然而，PTMs在时间序列数据上的发展滞后。这凸显了现有基于transformer的架构的局限性，特别是它们处理大规模数据和捕捉长期时间依赖性的可扩展性。本研究提出了即时生成预训练Transformer（TimelyGPT）。TimelyGPT采用可推广位置（xPos）嵌入将趋势和周期模式编码到时间序列表示中。它还集成了循环注意力和时间卷积模块，以有效地捕捉全局和局部的时间依赖关系。我们的实验表明，TimelyGPT在建模连续监测的生物信号和经常出现在纵向电磁波领域中不规则采样的时间序列数据方面表现出色。

    arXiv:2312.00817v2 Announce Type: replace-cross  Abstract: Large-scale pre-trained models (PTMs) such as BERT and GPT have recently achieved great success in Natural Language Processing and Computer Vision domains. However, the development of PTMs on time-series data is lagging behind. This underscores the limitations of the existing transformer-based architectures, particularly their scalability to handle large-scale data and ability to capture long-term temporal dependencies. In this study, we present Timely Generative Pre-trained Transformer (TimelyGPT). TimelyGPT employs an extrapolatable position (xPos) embedding to encode trend and periodic patterns into time-series representations. It also integrates recurrent attention and temporal convolution modules to effectively capture global-local temporal dependencies. Our experiments show that TimelyGPT excels in modeling continuously monitored biosignals and irregularly-sampled time series data commonly observed in longitudinal electro
    
[^227]: 从视觉基础模型中进行知识迁移用于高效训练小型任务特定模型

    Knowledge Transfer from Vision Foundation Models for Efficient Training of Small Task-specific Models

    [https://arxiv.org/abs/2311.18237](https://arxiv.org/abs/2311.18237)

    本文提出了一个简单的任务导向的知识迁移方法，用于高效训练小型任务特定模型。实验结果表明，该方法在多个目标任务上表现出了更好的性能，并且还展示了高达9倍的性能提升。

    

    在许多下游任务中，基于大规模数据集预训练的视觉基础模型在有限标记的目标数据上展现出了令人印象深刻的性能。然而，由于推理计算成本高，这些模型无法应用于许多实际应用。为了解决这个问题，我们提出了一个简单的任务导向的知识迁移方法，以高效解决如何利用大规模视觉基础模型的知识来训练小型任务特定模型的问题。我们在五个目标任务上的实验结果表明，该方法在超过Task-Agnostic VFM蒸馏、Web-Scale CLIP预训练、监督式ImageNet预训练和自监督DINO预训练29.8%、22.1%、13.7%和11.6%的方面表现出更好的性能。此外，所提出的方法还展现出了高达9倍的性能提升。

    arXiv:2311.18237v2 Announce Type: replace-cross  Abstract: Vision Foundation Models (VFMs) pretrained on massive datasets exhibit impressive performance on various downstream tasks, especially with limited labeled target data. However, due to their high inference compute cost, these models cannot be deployed for many real-world applications. Motivated by this, we ask the following important question, "How can we leverage the knowledge from a large VFM to train a small task-specific model for a new target task with limited labeled training data?", and propose a simple task-oriented knowledge transfer approach as a highly effective solution to this problem. Our experimental results on five target tasks show that the proposed approach outperforms task-agnostic VFM distillation, web-scale CLIP pretraining, supervised ImageNet pretraining, and self-supervised DINO pretraining by up to 11.6%, 22.1%, 13.7%, and 29.8%, respectively. Furthermore, the proposed approach also demonstrates up to 9x
    
[^228]: AutArch：一种用于考古目录中物体检测和自动化记录的人工智能辅助工作流程

    AutArch: An AI-assisted workflow for object detection and automated recording in archaeological catalogues

    [https://arxiv.org/abs/2311.17978](https://arxiv.org/abs/2311.17978)

    这篇论文介绍了AutArch，一种用于考古目录中物体检测和自动化记录的人工智能辅助工作流程，并提出了一种新的数据收集方法，通过自动化从遗留资源中提取数据，解决了现有记录质量和标准不一致的挑战。

    

    这篇论文的背景是利用人工智能和大数据从异构的已发表资源中创建大规模统一的考古数据集，比如遗物目录。论文关注的是一致考古数据组合的挑战。由于现有记录在质量和记录标准上存在差异，我们无法简单地合并现有记录。因此，必须从已发表的考古插图中重新创建记录。只有通过自动化的帮助，这才是可行的途径。本文的贡献是一个新的工作流程，用于从考古遗物目录中收集数据，这些目录作为遗留资源存在，比如大型未排序的PDF文件中的考古绘图和照片；该工作流程依赖于支持图像处理、物体检测以及验证和调整自动获取数据的交互手段的自定义软件（AutArch）。我们集成了人工智能技术。

    arXiv:2311.17978v2 Announce Type: replace-cross  Abstract: The context of this paper is the creation of large uniform archaeological datasets from heterogeneous published resources, such as find catalogues - with the help of AI and Big Data. The paper is concerned with the challenge of consistent assemblages of archaeological data. We cannot simply combine existing records, as they differ in terms of quality and recording standards. Thus, records have to be recreated from published archaeological illustrations. This is only a viable path with the help of automation. The contribution of this paper is a new workflow for collecting data from archaeological find catalogues available as legacy resources, such as archaeological drawings and photographs in large unsorted PDF files; the workflow relies on custom software (AutArch) supporting image processing, object detection, and interactive means of validating and adjusting automatically retrieved data. We integrate artificial intelligence (
    
[^229]: ASI:评估深度学习模型的准确性-稳定性指数

    ASI: Accuracy-Stability Index for Evaluating Deep Learning Models

    [https://arxiv.org/abs/2311.15332](https://arxiv.org/abs/2311.15332)

    该论文引入了准确性-稳定性指数（ASI），它是一种综合考虑准确度和稳定性的定量评估深度学习模型的指标。实验结果展示了ASI的应用，提供了一个用于可视化ASI、平均准确度和变异系数的3D曲面模型。这项研究解决了深度学习模型定量基准评估指标的重要问题，并提供了一种准确评估深度学习模型准确性和稳定性的新方法。

    

    在深度学习研究中，模型的不断引入使得有效和高效的评估变得至关重要。现有方法通常强调准确度指标，忽视了稳定性。为解决这个问题，本文引入了准确性-稳定性指数（ASI），它是一种综合考虑准确度和稳定性的定量评估深度学习模型的指标。实验结果展示了ASI的应用，同时提供了一个用于可视化ASI、平均准确度和变异系数的3D曲面模型。本文解决了深度学习模型定量基准评估指标的重要问题，并提供了一种准确评估深度学习模型准确性和稳定性的新方法。文章最后还对潜在弱点进行了讨论，并概述了未来的研究方向。

    arXiv:2311.15332v2 Announce Type: replace-cross  Abstract: In the context of deep learning research, where model introductions continually occur, the need for effective and efficient evaluation remains paramount. Existing methods often emphasize accuracy metrics, overlooking stability. To address this, the paper introduces the Accuracy-Stability Index (ASI), a quantitative measure incorporating both accuracy and stability for assessing deep learning models. Experimental results demonstrate the application of ASI, and a 3D surface model is presented for visualizing ASI, mean accuracy, and coefficient of variation. This paper addresses the important issue of quantitative benchmarking metrics for deep learning models, providing a new approach for accurately evaluating accuracy and stability of deep learning models. The paper concludes with discussions on potential weaknesses and outlines future research directions.
    
[^230]: 交叉验证和突变验证在模型选择中的实证比较

    Empirical Comparison between Cross-Validation and Mutation-Validation in Model Selection

    [https://arxiv.org/abs/2311.14079](https://arxiv.org/abs/2311.14079)

    本研究通过对比基准和实际数据集，实证比较了突变验证（MV）和交叉验证（CV）在模型选择中的表现。结果发现，MV和CV在选择模型的泛化性能方面基本等效，但MV在选择简单模型和计算成本方面具有优势。

    

    突变验证（MV）是一种近期提出的模型选择方法，因其独特特性和潜在益处而受到广泛关注，与广泛使用的交叉验证（CV）方法相比。在本研究中，我们对MV和k折交叉验证（CV）进行了基准和实际数据集的实证比较。通过使用贝叶斯测试，我们比较了产生三个后验概率的泛化估计：实际等效性、CV优势和MV优势。我们还评估了所选模型的能力差异和计算效率。我们发现，在各种机器学习算法和大多数基准数据集中，MV和CV都选择具有实际等效泛化性能的模型。MV在选择较简单模型和较低计算成本方面具有优势。然而，在某些情况下，MV选择过于简单的模型导致欠拟合。

    arXiv:2311.14079v2 Announce Type: replace  Abstract: Mutation validation (MV) is a recently proposed approach for model selection, garnering significant interest due to its unique characteristics and potential benefits compared to the widely used cross-validation (CV) method. In this study, we empirically compared MV and $k$-fold CV using benchmark and real-world datasets. By employing Bayesian tests, we compared generalization estimates yielding three posterior probabilities: practical equivalence, CV superiority, and MV superiority. We also evaluated the differences in the capacity of the selected models and computational efficiency. We found that both MV and CV select models with practically equivalent generalization performance across various machine learning algorithms and the majority of benchmark datasets. MV exhibited advantages in terms of selecting simpler models and lower computational costs. However, in some cases MV selected overly simplistic models leading to underfitting
    
[^231]: 分析Hugging Face上机器学习模型的演化和维护

    Analyzing the Evolution and Maintenance of ML Models on Hugging Face

    [https://arxiv.org/abs/2311.13380](https://arxiv.org/abs/2311.13380)

    本文通过仓库挖掘和文本分析的方式，对Hugging Face上的机器学习模型的演化和维护进行了研究。研究发现了Hugging Face的整体增长和受欢迎程度，揭示了ML领域、框架使用、作者分组等方面的趋势，同时也探讨了开发者社区中普遍存在的主题和见解以及模型的维护状态和演化情况。

    

    Hugging Face（HF）已成为机器学习（ML）模型开发和分享的重要平台。本研究通过使用HF Hub API收集的数据，对超过380,000个模型进行仓库挖掘，旨在探索HF上托管的模型的社区参与、演化和维护等方面，这些方面在现有文献中尚未全面探讨。我们首先审查了HF的整体增长和受欢迎程度，揭示了ML领域、框架使用、作者分组以及标签和数据集的演化趋势。通过对模型卡片描述的文本分析，我们还试图确定开发者社区中普遍存在的主题和见解。我们的研究进一步涵盖了模型维护方面，在这方面我们评估了ML模型的维护状态，将提交消息分类为不同的类别（校正性、完善性和适应性），分析了模型的演化情况等。

    arXiv:2311.13380v2 Announce Type: cross  Abstract: Hugging Face (HF) has established itself as a crucial platform for the development and sharing of machine learning (ML) models. This repository mining study, which delves into more than 380,000 models using data gathered via the HF Hub API, aims to explore the community engagement, evolution, and maintenance around models hosted on HF, aspects that have yet to be comprehensively explored in the literature. We first examine the overall growth and popularity of HF, uncovering trends in ML domains, framework usage, authors grouping and the evolution of tags and datasets used. Through text analysis of model card descriptions, we also seek to identify prevalent themes and insights within the developer community. Our investigation further extends to the maintenance aspects of models, where we evaluate the maintenance status of ML models, classify commit messages into various categories (corrective, perfective, and adaptive), analyze the evol
    
[^232]: 模型市场的调节: AI中介平台的平台治理难题

    Moderating Model Marketplaces: Platform Governance Puzzles for AI Intermediaries

    [https://arxiv.org/abs/2311.12573](https://arxiv.org/abs/2311.12573)

    本论文研究了模型市场的调节问题，分析了AI中介平台面临的平台治理挑战，并总结了业界的相关实践，包括许可、访问和使用限制、自动内容调节以及公开政策制定。

    

    arXiv: 2311.12573v2 公告类型: replace-cross 摘要: AI开发社区越来越多地利用托管中介平台，如Hugging Face，为用户上传的模型和训练数据提供便捷访问。这些模型市场降低了成千上万用户的技术部署门槛，但也可能被用于许多潜在有害和非法的方式。在本文中，我们解释了AI系统如何既能“包含”内容又能是开放式工具，从而成为迄今为止最棘手的平台治理挑战之一。我们提供了几个案例研究来分析模型市场如何管理模型，这些案例跨越了三个具有代表性的平台，即Hugging Face、GitHub和Civitai。基于这些分析，我们总结了业界正在制定的重要（但仍然有限）应对调节需求的做法：许可、访问和使用限制、自动内容调节以及公开政策制定。

    arXiv:2311.12573v2 Announce Type: replace-cross  Abstract: The AI development community is increasingly making use of hosting intermediaries such as Hugging Face provide easy access to user-uploaded models and training data. These model marketplaces lower technical deployment barriers for hundreds of thousands of users, yet can be used in numerous potentially harmful and illegal ways. In this article, we explain ways in which AI systems, which can both `contain' content and be open-ended tools, present one of the trickiest platform governance challenges seen to date. We provide case studies of several incidents across three illustrative platforms -- Hugging Face, GitHub and Civitai -- to examine how model marketplaces moderate models. Building on this analysis, we outline important (and yet nevertheless limited) practices that industry has been developing to respond to moderation demands: licensing, access and use restrictions, automated content moderation, and open policy development.
    
[^233]: 基于超平面优化的神经网络中的随机线性投影损失

    Random Linear Projections Loss for Hyperplane-Based Optimization in Neural Networks

    [https://arxiv.org/abs/2311.12356](https://arxiv.org/abs/2311.12356)

    本研究引入了一种名为随机线性投影（RLP）损失的新方法，通过利用数据中的几何关系来提高神经网络训练效率。实证评估表明，使用RLP损失训练的神经网络优于传统损失函数训练的网络，在更少的数据样本下实现更好的性能，并且对于添加噪声表现更强鲁棒性。

    

    提出了一种名为随机线性投影（RLP）损失的新方法，通过利用数据中的几何关系来提高训练效率。与传统的旨在最小化逐点误差的损失函数不同，RLP损失通过最小化连接固定大小的特征-预测对和特征-标签对的超平面集之间的距离来操作。我们通过在基准数据集和合成示例上进行的实证评估表明，使用RLP损失训练的神经网络优于使用传统损失函数训练的网络，可以在更少的数据样本下实现更好的性能，并且对于添加噪声表现更强鲁棒性。我们还提供了支持我们实证结果的理论分析。

    arXiv:2311.12356v2 Announce Type: replace  Abstract: Advancing loss function design is pivotal for optimizing neural network training and performance. This work introduces Random Linear Projections (RLP) loss, a novel approach that enhances training efficiency by leveraging geometric relationships within the data. Distinct from traditional loss functions that target minimizing pointwise errors, RLP loss operates by minimizing the distance between sets of hyperplanes connecting fixed-size subsets of feature-prediction pairs and feature-label pairs. Our empirical evaluations, conducted across benchmark datasets and synthetic examples, demonstrate that neural networks trained with RLP loss outperform those trained with traditional loss functions, achieving improved performance with fewer data samples, and exhibiting greater robustness to additive noise. We provide theoretical analysis supporting our empirical findings.
    
[^234]: 用于纠正眼动阅读数据中的垂直漂移的双输入流转换器

    Dual input stream transformer for vertical drift correction in eye-tracking reading data

    [https://arxiv.org/abs/2311.06095](https://arxiv.org/abs/2311.06095)

    这篇论文介绍了一种名为Dual Input Stream Transformer（DIST）的转换器，用于解决眼动阅读数据中由于垂直漂移而产生的注视点分配问题。通过与经典方法进行比较，我们展示了DIST在不同数据集上的高准确性，并通过将多个DIST模型实例组合成一个集成模型进一步提高了准确率。这项研究对于解决阅读研究中手动分配文本行的瓶颈具有重要意义。

    

    我们引入了一种新颖的双输入流转换器（DIST），用于解决眼动数据中将注视点分配到实际所关注的文本行的挑战性问题。这一后处理步骤对于阅读数据的分析至关重要，因为存在垂直漂移的噪声。我们在九个不同的数据集上评估DIST与十一种经典方法进行比较。我们证明了将多个DIST模型实例组合成一个集成模型可以在所有数据集上实现高准确性。将DIST集成模型与最佳的经典方法进一步组合，平均准确率达到98.17％。我们的方法在解决阅读研究中手动分配文本行的瓶颈方面迈出了重要一步。通过广泛的分析和消融研究，我们确定了促成DIST成功的关键因素，包括t

    arXiv:2311.06095v2 Announce Type: replace-cross  Abstract: We introduce a novel Dual Input Stream Transformer (DIST) for the challenging problem of assigning fixation points from eye-tracking data collected during passage reading to the line of text that the reader was actually focused on. This post-processing step is crucial for analysis of the reading data due to the presence of noise in the form of vertical drift. We evaluate DIST against eleven classical approaches on a comprehensive suite of nine diverse datasets. We demonstrate that combining multiple instances of the DIST model in an ensemble achieves high accuracy across all datasets. Further combining the DIST ensemble with the best classical approach yields an average accuracy of 98.17 %. Our approach presents a significant step towards addressing the bottleneck of manual line assignment in reading research. Through extensive analysis and ablation studies, we identify key factors that contribute to DIST's success, including t
    
[^235]: 提升流式时间序列分割的等级

    Raising the ClaSS of Streaming Time Series Segmentation

    [https://arxiv.org/abs/2310.20431](https://arxiv.org/abs/2310.20431)

    ClaSS是一种新颖、高效且高精度的流式时间序列分割算法，通过自监督时间序列分类评估同质性，并应用统计测试检测显著的变化点。

    

    今天，普遍存在的传感器发射高频数值测量流，反映了人类、动物、工业、商业和自然过程的特性。这些过程的变化，例如由外部事件或内部状态变化引起的，会表现为记录信号中的变化。流式时间序列分割（STSS）的任务是将流分割为对应于所观察的过程或实体状态的连续可变大小的分段。分割操作本身必须能够应对输入信号的频率。我们引入了ClaSS，一种新颖、高效且高精度的STSS算法。ClaSS使用自监督时间序列分类评估潜在分割的同质性，并应用统计测试来检测显著的变化点（CPs）。在我们的实验证评中使用了两个大型基准和六个真实世界的数据档案。

    arXiv:2310.20431v2 Announce Type: replace-cross  Abstract: Ubiquitous sensors today emit high frequency streams of numerical measurements that reflect properties of human, animal, industrial, commercial, and natural processes. Shifts in such processes, e.g. caused by external events or internal state changes, manifest as changes in the recorded signals. The task of streaming time series segmentation (STSS) is to partition the stream into consecutive variable-sized segments that correspond to states of the observed processes or entities. The partition operation itself must in performance be able to cope with the input frequency of the signals. We introduce ClaSS, a novel, efficient, and highly accurate algorithm for STSS. ClaSS assesses the homogeneity of potential partitions using self-supervised time series classification and applies statistical tests to detect significant change points (CPs). In our experimental evaluation using two large benchmarks and six real-world data archives, 
    
[^236]: 扩散模型中的可重复性和一致性的出现

    The Emergence of Reproducibility and Consistency in Diffusion Models

    [https://arxiv.org/abs/2310.05264](https://arxiv.org/abs/2310.05264)

    该论文研究了扩散模型中的一致模型可重复性现象，实验证实了无论模型框架、模型架构或训练过程如何，不同的扩散模型都能够一致地达到相同的数据分布和评分函数。此外，研究发现扩散模型在学习过程中受训练数据规模的影响，表现出两种不同的训练模式：记忆化模式和泛化模式。

    

    在这项工作中，我们研究了扩散模型中的一个有趣且普遍存在的现象，我们称之为“一致的模型可重复性”：在给定相同的起始噪声输入和确定性采样器的情况下，不同的扩散模型通常产生非常相似的输出。我们通过全面的实验证实了这一现象，表明不同的扩散模型无论扩散模型框架、模型架构或训练过程如何，在数据分布和评分函数上都能够一致地达到相同的结果。更令人惊讶的是，我们进一步的调查表明，扩散模型在学习受训数据规模影响下的不同分布。这一点得到了两种不同训练模式下模型可重复性的体现：（i）“记忆化模式”，其中扩散模型过度拟合于训练数据分布，和（ii）“泛化模式”，其中模型学习到了基础数据分布。

    arXiv:2310.05264v2 Announce Type: replace  Abstract: In this work, we investigate an intriguing and prevalent phenomenon of diffusion models which we term as "consistent model reproducibility": given the same starting noise input and a deterministic sampler, different diffusion models often yield remarkably similar outputs. We confirm this phenomenon through comprehensive experiments, implying that different diffusion models consistently reach the same data distribution and scoring function regardless of diffusion model frameworks, model architectures, or training procedures. More strikingly, our further investigation implies that diffusion models are learning distinct distributions affected by the training data size. This is supported by the fact that the model reproducibility manifests in two distinct training regimes: (i) "memorization regime", where the diffusion model overfits to the training data distribution, and (ii) "generalization regime", where the model learns the underlyin
    
[^237]: 通过策略合并实现舰队学习

    Fleet Learning via Policy Merging

    [https://arxiv.org/abs/2310.01362](https://arxiv.org/abs/2310.01362)

    本文研究了通过策略合并解决机器人群体学习中的数据存储和传输问题，并提出了一种基于循环神经网络的分布式学习方法。该方法能够在Meta-World环境中将50个任务的策略行为整合，并在大多数训练任务上表现良好。

    

    机器人群体通过与环境互动产生的大量异构流数据存储或传输上的困难，机器人团队需要通过在不同环境中的异构经验来共同获得多样化的技能。本文研究了从这种分布式异构数据集中进行策略合并作为潜在解决方案的问题。为了在舰队环境中高效合并策略，我们提出了FLEET-MERGE，一种基于循环神经网络参数化控制策略的分布式学习实例，考虑了参数化控制策略中的排列不变性。我们表明，FLEET-MERGE在Meta-World环境中对50个任务进行训练的策略行为进行了整合，并且几乎在所有训练任务上表现良好。

    arXiv:2310.01362v2 Announce Type: replace-cross  Abstract: Fleets of robots ingest massive amounts of heterogeneous streaming data silos generated by interacting with their environments, far more than what can be stored or transmitted with ease. At the same time, teams of robots should co-acquire diverse skills through their heterogeneous experiences in varied settings. How can we enable such fleet-level learning without having to transmit or centralize fleet-scale data? In this paper, we investigate policy merging (PoMe) from such distributed heterogeneous datasets as a potential solution. To efficiently merge policies in the fleet setting, we propose FLEET-MERGE, an instantiation of distributed learning that accounts for the permutation invariance that arises when parameterizing the control policies with recurrent neural networks. We show that FLEET-MERGE consolidates the behavior of policies trained on 50 tasks in the Meta-World environment, with good performance on nearly all train
    
[^238]: 通过生成轨迹模型增强层次环境设计

    Enhancing the Hierarchical Environment Design via Generative Trajectory Modeling

    [https://arxiv.org/abs/2310.00301](https://arxiv.org/abs/2310.00301)

    本文通过引入层次MDP框架，提出了一种在资源约束下增强环境设计的方法，通过上层教师智能体生成适当的训练环境，以促进学生智能体的学习能力发展。

    

    无监督环境设计（UED）是一种自动生成训练环境课程的范例，使在这些环境中训练的智能体能够发展通用能力，即实现良好的零-shot转移性能。然而，现有的UED方法主要关注对开放式智能体训练的环境进行随机生成，这在资源有限的情况下，例如对生成环境数量的限制方面是不实际的。本文引入了一个层次MDP框架，用于在资源约束下进行环境设计。它由一个上层强化学习教师智能体和一个下层学生智能体的合作组成。强化学习教师可以利用先前发现的环境结构，通过观察学生智能体的策略表示在学生能力的前沿生成适当的训练环境。

    arXiv:2310.00301v2 Announce Type: replace-cross  Abstract: Unsupervised Environment Design (UED) is a paradigm for automatically generating a curriculum of training environments, enabling agents trained in these environments to develop general capabilities, i.e., achieving good zero-shot transfer performance. However, existing UED approaches focus primarily on the random generation of environments for open-ended agent training. This is impractical in scenarios with limited resources, such as the constraints on the number of generated environments. In this paper, we introduce a hierarchical MDP framework for environment design under resource constraints. It consists of an upper-level RL teacher agent that generates suitable training environments for a lower-level student agent. The RL teacher can leverage previously discovered environment structures and generate environments at the frontier of the student's capabilities by observing the student policy's representation. Moreover, to redu
    
[^239]: 异常不敏感的卡尔曼滤波：理论和应用

    Outlier-Insensitive Kalman Filtering: Theory and Applications

    [https://arxiv.org/abs/2309.09505](https://arxiv.org/abs/2309.09505)

    本文提出了一种异常不敏感的卡尔曼滤波算法，通过对标准更新步骤的短小迭代过程，减轻了离群值对滤波性能的有害影响。通过将每个潜在的离群值建模为具有未知方差的正态过程，并应用在线估计算法，该方法在滤波场景中表现出竞争性能且对离群值具有鲁棒性。

    

    动态系统从噪声观测中进行状态估计是许多应用中的基本任务。通常使用线性卡尔曼滤波器（KF）来解决这个问题，但是由于其凸二次目标函数的敏感性，当观测中存在离群值时，其性能可能会显著降低。为了缓解这种情况，可以应用异常检测算法。本文提出了一种无参数算法，可以在只对KF的标准更新步骤进行短小的迭代过程时，减轻离群值的有害影响。为此，我们将每个潜在的离群值建模为具有未知方差的正态过程，并通过期望最大化或交替最大化算法进行在线估计。仿真和实地实验评估证明了我们方法的竞争性能，展示了其在滤波场景中对离群值的鲁棒性。

    arXiv:2309.09505v2 Announce Type: replace-cross  Abstract: State estimation of dynamical systems from noisy observations is a fundamental task in many applications. It is commonly addressed using the linear Kalman filter (KF), whose performance can significantly degrade in the presence of outliers in the observations, due to the sensitivity of its convex quadratic objective function. To mitigate such behavior, outlier detection algorithms can be applied. In this work, we propose a parameter-free algorithm which mitigates the harmful effect of outliers while requiring only a short iterative process of the standard update step of the KF. To that end, we model each potential outlier as a normal process with unknown variance and apply online estimation through either expectation maximization or alternating maximization algorithms. Simulations and field experiment evaluations demonstrate competitive performance of our method, showcasing its robustness to outliers in filtering scenarios comp
    
[^240]: 线性马尔可夫决策过程的速率最优策略优化

    Rate-Optimal Policy Optimization for Linear Markov Decision Processes

    [https://arxiv.org/abs/2308.14642](https://arxiv.org/abs/2308.14642)

    本文中，我们研究了在线周期性线性马尔可夫决策过程中的遗憾最小化问题，并提出了一种与周期数K成比率最优的遗憾收敛率O(√K)。这是首个针对带有乐观反馈的随机设置使用基于策略优化的方法并建立与K最优收敛速率的研究，也是首个针对具有全信息反馈的对抗设置并建立与K最优速率的研究，目前尚未找到具有最优速率保证的算法。

    

    我们研究在线周期性线性马尔可夫决策过程中最小化遗憾，并且得到了与$K$（表示周期数）成比率最优的$\widetilde{O}(\sqrt{K})$的遗憾。我们的工作是首次在带有乐观反馈的随机设置中使用基于策略优化的方法建立了与$K$最优（相对于$K$）的收敛速率，也是首次建立在完全信息反馈的敌对设置中与$K$最优的速率，这种情况下目前没有已知具有最优速率保证的算法。

    arXiv:2308.14642v2 Announce Type: replace  Abstract: We study regret minimization in online episodic linear Markov Decision Processes, and obtain rate-optimal $\widetilde O (\sqrt K)$ regret where $K$ denotes the number of episodes. Our work is the first to establish the optimal (w.r.t.~$K$) rate of convergence in the stochastic setting with bandit feedback using a policy optimization based approach, and the first to establish the optimal (w.r.t.~$K$) rate in the adversarial setup with full information feedback, for which no algorithm with an optimal rate guarantee is currently known.
    
[^241]: 隐式图神经扩散网络：收敛性、泛化性和过度平滑问题

    Implicit Graph Neural Diffusion Networks: Convergence, Generalization, and Over-Smoothing

    [https://arxiv.org/abs/2308.03306](https://arxiv.org/abs/2308.03306)

    这篇论文介绍了隐式图神经扩散网络的设计框架，并解决了其收敛性、泛化性和过度平滑问题。这个框架允许学习度量和图扩散强度，同时提出了一个新的模型来避免过度平滑问题。

    

    最近，隐式图神经网络在解决图学习问题方面取得了显著的成功。然而，设计不良的隐式图神经网络层可能对学习图度量具有有限的适应性，经验过度平滑问题，或者表现出次优收敛和泛化性能，可能阻碍它们的实际性能。为了解决这些问题，我们引入了一个基于参数化图拉普拉斯算子的几何框架，用于设计隐式图扩散层。我们的框架允许从数据中学习顶点和边缘空间的度量，以及图扩散强度。我们展示了隐式图神经网络层可以被看作是一个迭代解析能量最小化问题的定点方程，并给出了在训练过程中可能遭受过度平滑问题的条件。我们进一步提出了一个新的隐式图神经网络模型来避免过度平滑问题。

    arXiv:2308.03306v2 Announce Type: replace  Abstract: Implicit Graph Neural Networks (GNNs) have achieved significant success in addressing graph learning problems recently. However, poorly designed implicit GNN layers may have limited adaptability to learn graph metrics, experience over-smoothing issues, or exhibit suboptimal convergence and generalization properties, potentially hindering their practical performance. To tackle these issues, we introduce a geometric framework for designing implicit graph diffusion layers based on a parameterized graph Laplacian operator. Our framework allows learning the metrics of vertex and edge spaces, as well as the graph diffusion strength from data. We show how implicit GNN layers can be viewed as the fixed-point equation of a Dirichlet energy minimization problem and give conditions under which it may suffer from over-smoothing during training (OST) and inference (OSI). We further propose a new implicit GNN model to avoid OST and OSI. We establi
    
[^242]: 傅里叶混合窗口注意力：加速长序列时间序列预测的Informer方法

    Fourier-Mixed Window Attention: Accelerating Informer for Long Sequence Time-Series Forecasting

    [https://arxiv.org/abs/2307.00493](https://arxiv.org/abs/2307.00493)

    本文提出了一种名为FWin的快速本地全局窗口注意力方法，用于加速长序列时间序列预测的Informer方法。通过实验证明，该方法可以提高预测准确性并加速推断速度，同时在非线性回归模型中表现出与Softmax全注意力相媲美甚至更优的效果。

    

    我们研究了一种快速的本地全局窗口注意力方法，用于加速Informer在长序列时间序列预测中的应用。虽然窗口注意力是局部的和具有相当大的计算节约，但它缺乏捕获全局令牌信息的能力，这通过后续的傅里叶变换块进行补偿。我们的方法名为FWin，不依赖于Informer的ProbSparse注意力中的查询稀疏性假设和经验性近似。通过对单变量和多变量数据集的实验，我们证明了FWin transformers可以提高Informer的整体预测准确性，同时将其推断速度加速40%至50%。我们还在非线性回归模型中展示了学习到的FWin类型注意力在时间序列数据上通过从Informer模型的全注意力层中提取的关键向量来逼近甚至胜过基于Softmax全注意力的方法。

    arXiv:2307.00493v2 Announce Type: replace-cross  Abstract: We study a fast local-global window-based attention method to accelerate Informer for long sequence time-series forecasting. While window attention is local and a considerable computational saving, it lacks the ability to capture global token information which is compensated by a subsequent Fourier transform block. Our method, named FWin, does not rely on query sparsity hypothesis and an empirical approximation underlying the ProbSparse attention of Informer. Through experiments on univariate and multivariate datasets, we show that FWin transformers improve the overall prediction accuracies of Informer while accelerating its inference speeds by 40 to 50 %. We also show in a nonlinear regression model that a learned FWin type attention approaches or even outperforms softmax full attention based on key vectors extracted from an Informer model's full attention layer acting on time series data.
    
[^243]: Hyp-OW: 利用超几何距离的层次结构学习增强开放世界目标检测

    Hyp-OW: Exploiting Hierarchical Structure Learning with Hyperbolic Distance Enhances Open World Object Detection

    [https://arxiv.org/abs/2306.14291](https://arxiv.org/abs/2306.14291)

    Hyp-OW是一种利用超几何距离的层次结构学习增强开放世界目标检测的方法，通过超类正则化器学习和建模已知项目的层次表示，通过基于相似度距离的重新标记模块有效地检测未知对象。

    

    开放世界目标检测(OWOD)是一项具有挑战性且现实的任务，超越了标准目标检测任务的范围。它需要在检测已知和未知对象的同时，整合学习到的知识用于未来的任务。然而，“未知性”在不同上下文中有很大的变化。例如，在自动驾驶场景中，树通常被认为是背景的一部分，但在家庭环境中可能具有重要性。我们认为这种上下文信息应该已经嵌入到已知类别中。换句话说，已知和未知项之间应该存在语义或潜在的结构关系等待发现。受到这一观察的启发，我们提出了Hyp-OW，一种通过超类正则化器来学习和建模已知项目的层次表示的方法。利用这种表示，我们可以通过基于相似度距离的重新标记模块有效地检测未知对象。大量实验证明了我们方法的有效性。

    Open World Object Detection (OWOD) is a challenging and realistic task that extends beyond the scope of standard Object Detection task. It involves detecting both known and unknown objects while integrating learned knowledge for future tasks. However, the level of "unknownness" varies significantly depending on the context. For example, a tree is typically considered part of the background in a self-driving scene, but it may be significant in a household context. We argue that this contextual information should already be embedded within the known classes. In other words, there should be a semantic or latent structure relationship between the known and unknown items to be discovered. Motivated by this observation, we propose Hyp-OW, a method that learns and models hierarchical representation of known items through a SuperClass Regularizer. Leveraging this representation allows us to effectively detect unknown objects using a similarity distance-based relabeling module. Extensive experi
    
[^244]: DistriBlock: 通过利用输出分布的特征识别对抗性音频样本

    DistriBlock: Identifying adversarial audio samples by leveraging characteristics of the output distribution

    [https://arxiv.org/abs/2305.17000](https://arxiv.org/abs/2305.17000)

    DistriBlock提出了一种能够识别对抗性音频样本的有效检测策略，通过利用输出分布的特征，包括中位数、最大值和最小值、熵以及与后续时间步骤的分布之间的散度，应用二元分类器进行预测。这项研究证明了DistriBlock在识别对抗性音频样本方面的有效性。

    

    对抗性攻击可能误导自动语音识别（ASR）系统，使其预测任意目标文本，从而构成明显的安全威胁。为了防止这种攻击，我们提出了DistriBlock，一种适用于任何ASR系统的高效检测策略，该系统在每个时间步骤上预测输出标记的概率分布。我们对该分布的一组特征进行测量：输出概率的中位数、最大值和最小值，分布的熵，以及与后续时间步骤的分布之间的Kullback-Leibler和Jensen-Shannon散度。然后，通过利用对良性和对抗性数据观察到的特征，我们应用二元分类器，包括简单的基于阈值的分类、这种分类器的集合以及神经网络。通过对不同最先进的ASR系统和语言数据集进行广泛分析，我们证明了DistriBlock在识别对抗性音频样本方面的有效性。

    arXiv:2305.17000v2 Announce Type: replace-cross  Abstract: Adversarial attacks can mislead automatic speech recognition (ASR) systems into predicting an arbitrary target text, thus posing a clear security threat. To prevent such attacks, we propose DistriBlock, an efficient detection strategy applicable to any ASR system that predicts a probability distribution over output tokens in each time step. We measure a set of characteristics of this distribution: the median, maximum, and minimum over the output probabilities, the entropy of the distribution, as well as the Kullback-Leibler and the Jensen-Shannon divergence with respect to the distributions of the subsequent time step. Then, by leveraging the characteristics observed for both benign and adversarial data, we apply binary classifiers, including simple threshold-based classification, ensembles of such classifiers, and neural networks. Through extensive analysis across different state-of-the-art ASR systems and language data sets, 
    
[^245]: 边缘深度学习应用中的层次推断在线算法

    Online Algorithms for Hierarchical Inference in Deep Learning applications at the Edge

    [https://arxiv.org/abs/2304.00891](https://arxiv.org/abs/2304.00891)

    这项研究提出了一种在线算法，用于解决边缘深度学习应用中资源受限的边缘设备的层次推断问题，以在保证推断准确性的同时实现低延迟、带宽节省和能量效率的好处。

    

    我们考虑一种资源受限的边缘设备（如物联网传感器或微控制单元），其中嵌入有一个小型机器学习模型（S-ML）用于通用分类应用，以及一个托管大型机器学习模型（L-ML）的边缘服务器（ES）。由于S-ML的推断准确性低于L-ML，在所有数据样本都被发送到ES进行推断会导致较高的推断准确性，但这违背了在ED上嵌入S-ML的目的，剥夺了进行本地推断的低延迟、带宽节省和能量效率的好处。为了充分发挥ED推断和ES推断的优势，我们探索了层次推断（HI）的思想，即只有在S-ML推断正确时才接受，否则将数据样本发送到L-ML进行推断。然而，理想的HI实现是不可行的，因为正确性的判断。。。

    arXiv:2304.00891v2 Announce Type: replace  Abstract: We consider a resource-constrained Edge Device (ED), such as an IoT sensor or a microcontroller unit, embedded with a small-size ML model (S-ML) for a generic classification application and an Edge Server (ES) that hosts a large-size ML model (L-ML). Since the inference accuracy of S-ML is lower than that of the L-ML, offloading all the data samples to the ES results in high inference accuracy, but it defeats the purpose of embedding S-ML on the ED and deprives the benefits of reduced latency, bandwidth savings, and energy efficiency of doing local inference. In order to get the best out of both worlds, i.e., the benefits of doing inference on the ED and the benefits of doing inference on ES, we explore the idea of Hierarchical Inference (HI), wherein S-ML inference is only accepted when it is correct, otherwise the data sample is offloaded for L-ML inference. However, the ideal implementation of HI is infeasible as the correctness o
    
[^246]: 零阶优化遇到人工反馈：通过排名预测实现可证明学习

    Zeroth-Order Optimization Meets Human Feedback: Provable Learning via Ranking Oracles

    [https://arxiv.org/abs/2303.03751](https://arxiv.org/abs/2303.03751)

    零阶优化算法ZO-RankSGD解决了一个新兴的优化挑战，即只能通过排名预测来评估黑盒目标函数。该算法利用一种新颖的随机估计器来确定下降方向，并保证收敛到一个稳定点。此外，该算法还可用于增强学习中的策略优化问题，特别是当只有对于回报排名的排名预测时。

    

    在这项研究中，我们探讨了一种新兴的优化挑战，其中涉及到一个只能通过排名预测来评估的黑盒目标函数-这种情况在实际场景中经常遇到，特别是当函数由人类评判员评估时。这种挑战受到了强化学习与人工反馈（RLHF）的启发，这是一种最近用来提高大型语言模型（LLMs）性能的方法。我们引入了一种创新的零阶优化算法ZO-RankSGD来解决这个优化问题，并提供了理论保证。我们的算法利用一种新颖的基于排名的随机估计器来确定下降方向，并保证收敛到一个稳定点。此外，ZO-RankSGD可以直接应用于增强学习中的策略优化问题，特别是当只有对于回报排名的排名预测时。

    arXiv:2303.03751v2 Announce Type: replace-cross  Abstract: In this study, we delve into an emerging optimization challenge involving a black-box objective function that can only be gauged via a ranking oracle-a situation frequently encountered in real-world scenarios, especially when the function is evaluated by human judges. Such challenge is inspired from Reinforcement Learning with Human Feedback (RLHF), an approach recently employed to enhance the performance of Large Language Models (LLMs) using human guidance. We introduce ZO-RankSGD, an innovative zeroth-order optimization algorithm designed to tackle this optimization problem, accompanied by theoretical assurances. Our algorithm utilizes a novel rank-based random estimator to determine the descent direction and guarantees convergence to a stationary point. Moreover, ZO-RankSGD is readily applicable to policy optimization problems in Reinforcement Learning (RL), particularly when only ranking oracles for the episode reward are a
    
[^247]: 基于cGAN的增强人体活动识别的高维IMU传感器数据生成

    cGAN-Based High Dimensional IMU Sensor Data Generation for Enhanced Human Activity Recognition in Therapeutic Activities

    [https://arxiv.org/abs/2302.07998](https://arxiv.org/abs/2302.07998)

    本论文开发了一种基于cGAN的TheraGAN网络，用于生成与康复活动相关的高维IMU传感器数据。通过引入简单活动，简化了生成过程。该方法能够帮助解决传统活动识别分类器中训练数据不足的问题。

    

    人体活动识别是康复、健康监测和人机交互等应用的核心技术。可穿戴设备，尤其是IMU传感器，以相对较低的成本提供了丰富的人体运动特征，可用于活动识别。开发鲁棒的活动识别分类器一直是研究人员关注的一个主要问题。主要问题之一是通常存在训练数据不足的问题，这使得开发深度分类器变得困难，有时甚至不可能。在本文中，开发了一种新颖的GAN网络TheraGAN，用于生成与康复活动相关的IMU信号。生成的信号包括来自6个通道的IMU数据，即角速度和线性加速度。此外，引入简单活动简化了不同长度活动的生成过程。为了评估生成的信号，进行了几个定性实验和定量实验。

    arXiv:2302.07998v2 Announce Type: replace-cross  Abstract: Human activity recognition is a core technology for applications such as rehabilitation, health monitoring, and human-computer interactions. Wearable devices, especially IMU sensors, provide rich features of human movements at a reasonable cost, which can be leveraged in activity recognition. Developing a robust classifier for activity recognition has always been of interest to researchers. One major problem is that there is usually a deficit of training data, which makes developing deep classifiers difficult and sometimes impossible. In this work, a novel GAN network called TheraGAN was developed to generate IMU signals associated with rehabilitation activities. The generated signal comprises data from a 6-channel IMU, i.e., angular velocities and linear accelerations. Also, introducing simple activities simplified the generation process for activities of varying lengths. To evaluate the generated signals, several qualitative 
    
[^248]: 使用给定的子任务分解学习复杂的团队合作任务

    Learning Complex Teamwork Tasks Using a Given Sub-task Decomposition

    [https://arxiv.org/abs/2302.04944](https://arxiv.org/abs/2302.04944)

    通过使用专家提供的任务分解为更简单的多智能体子任务，并将其转移到目标任务中进行集体调整，我们的方法可以有效地学习复杂的多智能体任务，并在解决复杂目标任务所需的时间步数上实现了显著的减少。

    

    通过多智能体强化学习训练团队完成复杂任务可能面临诸如在大型联合策略空间中搜索策略和因互相适应而导致的非稳定性等挑战。为了促进对复杂多智能体任务的高效学习，我们提出了一种方法，该方法使用专家提供的任务分解为更简单的多智能体子任务。在每个子任务中，对整个团队的子集进行训练以获取特定于子任务的策略。然后将子团队合并并迁移到目标任务中，在那里他们的策略被集体调整以解决更复杂的目标任务。我们通过实验证明，这种方法可以显著减少解决复杂目标任务所需的时间步数，相对于从头开始训练。然而，我们还发现并研究了基于子任务分解的天真实现方法的两个问题。

    arXiv:2302.04944v2 Announce Type: replace-cross  Abstract: Training a team to complete a complex task via multi-agent reinforcement learning can be difficult due to challenges such as policy search in a large joint policy space, and non-stationarity caused by mutually adapting agents. To facilitate efficient learning of complex multi-agent tasks, we propose an approach which uses an expert-provided decomposition of a task into simpler multi-agent sub-tasks. In each sub-task, a subset of the entire team is trained to acquire sub-task-specific policies. The sub-teams are then merged and transferred to the target task, where their policies are collectively fine-tuned to solve the more complex target task. We show empirically that such approaches can greatly reduce the number of timesteps required to solve a complex target task relative to training from-scratch. However, we also identify and investigate two problems with naive implementations of approaches based on sub-task decomposition, 
    
[^249]: 关于风险敏感指数成本马尔可夫决策过程中修改的策略迭代的收敛性的研究

    On the Convergence of Modified Policy Iteration in Risk Sensitive Exponential Cost Markov Decision Processes

    [https://arxiv.org/abs/2302.03811](https://arxiv.org/abs/2302.03811)

    这项研究证明了在有限状态和动作空间的情况下，修改的策略迭代算法（MPI）在风险敏感问题中的收敛性，并提供了与已有结果不同的证明方法。

    

    修改的策略迭代（MPI）是一种将策略迭代和值迭代相结合的动态规划算法。MPI的收敛性在折扣和平均成本MDP的背景下已经得到了广泛研究。本文研究了指数成本风险敏感MDP的形式，该形式对模型参数具有一定的鲁棒性。虽然针对风险敏感MDP已经对策略迭代和值迭代进行了深入研究，但MPI却未被探索。我们首次证明了在有限状态和动作空间的情况下，MPI也对风险敏感问题收敛。由于指数成本形式涉及乘法贝尔曼方程，我们的主要贡献是一种与折扣和风险中立平均成本问题以及风险敏感值和策略迭代方法不同的收敛证明。我们总结了我们的一个

    arXiv:2302.03811v2 Announce Type: replace-cross  Abstract: Modified policy iteration (MPI) is a dynamic programming algorithm that combines elements of policy iteration and value iteration. The convergence of MPI has been well studied in the context of discounted and average-cost MDPs. In this work, we consider the exponential cost risk-sensitive MDP formulation, which is known to provide some robustness to model parameters. Although policy iteration and value iteration have been well studied in the context of risk sensitive MDPs, MPI is unexplored. We provide the first proof that MPI also converges for the risk-sensitive problem in the case of finite state and action spaces. Since the exponential cost formulation deals with the multiplicative Bellman equation, our main contribution is a convergence proof which is quite different than existing results for discounted and risk-neutral average-cost problems as well as risk sensitive value and policy iteration approaches. We conclude our a
    
[^250]: 一种潜变量空间相关感知自编码器用于偏斜数据中的异常检测

    A Latent Space Correlation-Aware Autoencoder for Anomaly Detection in Skewed Data

    [https://arxiv.org/abs/2301.00462](https://arxiv.org/abs/2301.00462)

    这项工作提出了一种潜变量空间相关感知自编码器，用于解决传感器数据偏斜且非高斯性的异常检测问题。

    

    无监督学习的潜变量空间中的异常检测在高维空间中区分异常数据和正常数据变得困难，因此变得越来越重要。过去已经探索了在潜变量空间中检测异常的密度估计和基于距离的方法。这些方法证明在潜变量空间中保留输入数据的有价值属性有助于更好地重构测试数据。此外，现实世界的传感器数据是偏斜且非高斯性的，使得基于平均的估计器对于偏斜数据不可靠。再者，基于重构误差的异常检测方法依赖于欧氏距离，不考虑特征空间中的有用相关信息，并且在该数据偏离训练分布时无法准确地重构数据。在这项工作中，我们解决了基于重构误差的自编码器的局限性，并提出了一种基于核的自编码器来解决偏斜数据中的异常检测问题。

    arXiv:2301.00462v3 Announce Type: replace  Abstract: Unsupervised learning-based anomaly detection in latent space has gained importance since discriminating anomalies from normal data becomes difficult in high-dimensional space. Both density estimation and distance-based methods to detect anomalies in latent space have been explored in the past. These methods prove that retaining valuable properties of input data in latent space helps in the better reconstruction of test data. Moreover, real-world sensor data is skewed and non-Gaussian in nature, making mean-based estimators unreliable for skewed data. Again, anomaly detection methods based on reconstruction error rely on Euclidean distance, which does not consider useful correlation information in the feature space and also fails to accurately reconstruct the data when it deviates from the training distribution. In this work, we address the limitations of reconstruction error-based autoencoders and propose a kernelized autoencoder th
    
[^251]: SimCS：领域增量在线继续分割的模拟

    SimCS: Simulation for Domain Incremental Online Continual Segmentation

    [https://arxiv.org/abs/2211.16234](https://arxiv.org/abs/2211.16234)

    本论文提出了一个新的方法SimCS，用于解决在线领域增量继续分割的问题。与现有方法相比，SimCS在资源有限的情况下，对不同领域的密集标记图像进行连续训练，无需任务边界信息，展现了更好的性能。

    

    连续学习是迈向终身智能的一步，其中模型可以持续从最近收集的数据中学习而不会遗忘先前的知识。现有的连续学习方法主要关注于具有清晰任务边界和无限计算预算的分类设置。本研究探讨了在线领域增量继续分割（ODICS）的问题，其中模型在不同领域的密集标记图像批次上进行连续训练，计算资源有限，并且没有关于任务边界的信息。ODICS在许多实际应用中出现。在自动驾驶中，这可能对应于在时间上对一系列城市中的分割模型进行训练的现实场景。我们分析了几种现有的连续学习方法，并展示了它们在这种设置中的性能较差，尽管在类别增量分割中表现良好。我们提出了一种新的方法SimCS，它采用增量领域学习和在线连续学习，并在多个数据集上进行了实验证明其有效性。

    arXiv:2211.16234v2 Announce Type: replace-cross  Abstract: Continual Learning is a step towards lifelong intelligence where models continuously learn from recently collected data without forgetting previous knowledge. Existing continual learning approaches mostly focus on image classification in the class-incremental setup with clear task boundaries and unlimited computational budget. This work explores the problem of Online Domain-Incremental Continual Segmentation (ODICS), where the model is continually trained over batches of densely labeled images from different domains, with limited computation and no information about the task boundaries. ODICS arises in many practical applications. In autonomous driving, this may correspond to the realistic scenario of training a segmentation model over time on a sequence of cities. We analyze several existing continual learning methods and show that they perform poorly in this setting despite working well in class-incremental segmentation. We p
    
[^252]: 当少即是多：关于半监督软件缺陷预测器"共同训练"的价值

    When Less is More: On the Value of "Co-training" for Semi-Supervised Software Defect Predictors

    [https://arxiv.org/abs/2211.05920](https://arxiv.org/abs/2211.05920)

    该论文研究了半监督软件缺陷预测器中的"共同训练"方法的价值，并发现这种方法可以利用少量标记数据取得与使用全部数据相媲美的预测结果。

    

    在标记模块为缺陷或非缺陷的任务中，标记数据是一项昂贵的任务。因此，可用于训练的标记数据往往受到限制。半监督分类器使用较少的标记数据来训练模型。然而，有许多半监督方法，包括自标签、共同训练、最大间隔和基于图的方法等。在软件工程领域，只有少数几种方法被用于测试（例如预测缺陷），而且这些方法只在少数几个项目上进行了测试。本文将55种半监督学习器应用于714个项目上，发现半监督的"共同训练方法"比其他方法表现更好。具体而言，在标记了仅2.5%的数据后，使用共同训练方法的预测结果与使用100%的数据进行预测的结果相媲美。然而，需要谨慎使用共同训练方法，因为共同训练方法的选择具体取决于...

    arXiv:2211.05920v2 Announce Type: replace-cross  Abstract: Labeling a module defective or non-defective is an expensive task. Hence, there are often limits on how much-labeled data is available for training. Semi-supervised classifiers use far fewer labels for training models. However, there are numerous semi-supervised methods, including self-labeling, co-training, maximal-margin, and graph-based methods, to name a few. Only a handful of these methods have been tested in SE for (e.g.) predicting defects and even there, those methods have been tested on just a handful of projects.   This paper applies a wide range of 55 semi-supervised learners to over 714 projects. We find that semi-supervised "co-training methods" work significantly better than other approaches. Specifically, after labeling, just   2.5% of data, then make predictions that are competitive to those using 100% of the data.   That said, co-training needs to be used cautiously since the specific choice of co-training meth
    
[^253]: FedMT: 混合类型标签的联邦学习

    FedMT: Federated Learning with Mixed-type Labels

    [https://arxiv.org/abs/2210.02042](https://arxiv.org/abs/2210.02042)

    本文提出了一种概念新颖的联邦学习设置，即具有混合类型标签的联邦学习，在其中不同的中心可以使用不同的标签准则。为了有效地训练具有混合类型标签的模型，作者提出了一种理论指导和模型无关的方法。

    

    在联邦学习（FL）中，分类器（例如深度网络）在多个中心的数据集上进行训练，而无需在这些中心之间交换数据，从而提高了样本效率。在传统的FL设置中，通常在所有参与训练的中心中采用相同的标签准则。这个限制极大地限制了FL的适用性。例如，在疾病诊断中使用的标准很可能在临床中心之间存在差异，这与传统FL的设置不匹配。在本文中，我们考虑了一个重要但尚未充分探索的FL设置，即具有混合类型标签的FL，其中各个中心可以使用不同的标签准则，从而导致中心间标签空间的差异，并对为传统设置设计的现有FL方法提出了挑战。为了有效而高效地训练具有混合类型标签的模型，我们提出了一种基于理论指导和模型无关的方法

    arXiv:2210.02042v3 Announce Type: replace-cross Abstract: In federated learning (FL), classifiers (e.g., deep networks) are trained on datasets from multiple centers without exchanging data across them, and thus improves sample efficiency. In the classical setting of FL, the same labeling criterion is usually employed across all centers being involved in training. This constraint greatly limits the applicability of FL. For example, standards used for disease diagnosis are more likely to be different across clinical centers, which mismatches the classical FL setting. In this paper, we consider an important yet under-explored setting of FL, namely FL with mixed-type labels where different labeling criteria can be employed by various centers, leading to inter-center label space differences and challenging existing FL methods designed for the classical setting. To effectively and efficiently train models with mixed-type labels, we propose a theory-guided and model-agnostic approach that ca
    
[^254]: PixTrack：使用NeRF模板和特征度量对物体的6DoF姿态进行精确跟踪

    PixTrack: Precise 6DoF Object Pose Tracking using NeRF Templates and Feature-metric Alignment

    [https://arxiv.org/abs/2209.03910](https://arxiv.org/abs/2209.03910)

    PixTrack是一种基于视觉的物体姿态跟踪框架，使用NeRF模板和特征度量对齐方法，能够精确跟踪物体的6DoF姿态，而且无需数据注释或轨迹平滑。方法具有高度精确、鲁棒且无抖动的特点，同时计算效率高，可用于多目标跟踪。

    

    我们提出了PixTrack，一种基于视觉的物体姿态跟踪框架，使用新颖的视图合成和深度特征度量对齐。我们遵循基于SfM的重新定位范式，使用神经辐射场来规范地表示被跟踪的物体。我们的评估表明，我们的方法在单目RGB图像和RGB-D图像中产生高度精确、鲁棒且无抖动的物体6DoF姿态估计，无需任何数据注释或轨迹平滑。我们的方法也具有计算效率高的特点，通过简单的CPU多进程可以实现多目标跟踪而无需改变我们的算法。我们的代码可在以下链接找到：https://github.com/GiantAI/pixtrack

    arXiv:2209.03910v2 Announce Type: replace-cross  Abstract: We present PixTrack, a vision based object pose tracking framework using novel view synthesis and deep feature-metric alignment. We follow an SfM-based relocalization paradigm where we use a Neural Radiance Field to canonically represent the tracked object. Our evaluations demonstrate that our method produces highly accurate, robust, and jitter-free 6DoF pose estimates of objects in both monocular RGB images and RGB-D images without the need of any data annotation or trajectory smoothing. Our method is also computationally efficient making it easy to have multi-object tracking with no alteration to our algorithm through simple CPU multiprocessing. Our code is available at: https://github.com/GiantAI/pixtrack
    
[^255]: 差分隐私图学习的敏感性有界个性化PageRank算法

    Differentially Private Graph Learning via Sensitivity-Bounded Personalized PageRank

    [https://arxiv.org/abs/2207.06944](https://arxiv.org/abs/2207.06944)

    本论文提出了一种敏感性有界的个性化PageRank算法，能够保护用户隐私。该算法在保持准确性的同时，实现了差分隐私图学习的几种工具。

    

    个性化PageRank(PPR)是一种基本工具，用于无监督学习图表示，如节点排序、标注和图嵌入。然而，随着数据隐私成为最近最重要的关注点之一，现有的PPR算法并未设计用于保护用户隐私。PPR对输入图的边非常敏感：仅差一个边的差异可能会导致PPR向量发生巨大改变，从而可能泄漏用户私密数据。在这篇论文中，我们提出了一种算法，该算法输出近似PPR，并对输入边具有可证明的敏感性边界。此外，我们证明了当输入图具有大度数时，我们的算法达到与非私密算法相似的准确性。我们敏感性有界PPR直接意味着图学习的几种私密算法，如差分隐私(DP)PPR排序、DP节点分类和DP节点嵌入。为了补充我们的理论分析，我们还通过实验证明了算法的实际性能。

    Personalized PageRank (PPR) is a fundamental tool in unsupervised learning of graph representations such as node ranking, labeling, and graph embedding. However, while data privacy is one of the most important recent concerns, existing PPR algorithms are not designed to protect user privacy. PPR is highly sensitive to the input graph edges: the difference of only one edge may cause a big change in the PPR vector, potentially leaking private user data.   In this work, we propose an algorithm which outputs an approximate PPR and has provably bounded sensitivity to input edges. In addition, we prove that our algorithm achieves similar accuracy to non-private algorithms when the input graph has large degrees. Our sensitivity-bounded PPR directly implies private algorithms for several tools of graph learning, such as, differentially private (DP) PPR ranking, DP node classification, and DP node embedding. To complement our theoretical analysis, we also empirically verify the practical perfor
    
[^256]: 对神经网络的任意数据污染攻击

    Indiscriminate Data Poisoning Attacks on Neural Networks

    [https://arxiv.org/abs/2204.09092](https://arxiv.org/abs/2204.09092)

    本研究关注对神经网络的任意数据污染攻击，利用二阶信息进行优化设计出了有效的攻击方法，并通过大量实验证明了对深度神经网络的影响。

    

    数据污染攻击是指恶意对手通过将“污染”的数据注入到训练过程中来影响模型的攻击，近年来引起了广泛的关注。本研究对现有的污染攻击进行了详细研究，并将其与解决顺序斯塔克伯格博弈的新老算法联系起来。通过为攻击者选择适当的损失函数，并利用利用二阶信息的算法进行优化，我们设计了对神经网络有效的污染攻击。我们提出了高效的实现方法，利用现代自动微分软件包同时、协调地生成数万个污染点，与现有的逐个生成污染点的方法相比。此外，我们还进行了大量实验证明了数据污染攻击对深度神经网络的影响。

    arXiv:2204.09092v2 Announce Type: replace  Abstract: Data poisoning attacks, in which a malicious adversary aims to influence a model by injecting "poisoned" data into the training process, have attracted significant recent attention. In this work, we take a closer look at existing poisoning attacks and connect them with old and new algorithms for solving sequential Stackelberg games. By choosing an appropriate loss function for the attacker and optimizing with algorithms that exploit second-order information, we design poisoning attacks that are effective on neural networks. We present efficient implementations that exploit modern auto-differentiation packages and allow simultaneous and coordinated generation of tens of thousands of poisoned points, in contrast to existing methods that generate poisoned points one by one. We further perform extensive experiments that empirically explore the effect of data poisoning attacks on deep neural networks.
    
[^257]: 基于排序的快速可解释聚类算法

    Fast and explainable clustering based on sorting

    [https://arxiv.org/abs/2202.01456](https://arxiv.org/abs/2202.01456)

    CLASSIX是一种快速可解释的聚类算法，它通过排序后的数据的贪婪聚合和群组合并来进行聚类。该算法具有与最先进的聚类算法相媲美的性能，并且具有线性空间复杂性和近线性时间复杂性。

    

    我们引入了一种快速可解释的聚类方法，称为CLASSIX。它由两个阶段组成，即将排序后的数据聚合成附近数据点组成的群组的贪婪聚合阶段，然后将群组合并成聚类。该算法由两个标量参数控制，一个是聚合的距离参数，另一个是控制最小聚类大小的参数。我们进行了广泛的实验，对合成和真实数据集的聚类性能进行了全面评估，涵盖了各种聚类形状和低到高的特征维度。实验结果表明，CLASSIX可以与最先进的聚类算法竞争。该算法具有线性空间复杂性，在广泛的问题范围内实现了接近线性的时间复杂性。其固有的简单性使得可以生成对计算的聚类的直观解释。

    arXiv:2202.01456v2 Announce Type: replace  Abstract: We introduce a fast and explainable clustering method called CLASSIX. It consists of two phases, namely a greedy aggregation phase of the sorted data into groups of nearby data points, followed by the merging of groups into clusters. The algorithm is controlled by two scalar parameters, namely a distance parameter for the aggregation and another parameter controlling the minimal cluster size. Extensive experiments are conducted to give a comprehensive evaluation of the clustering performance on synthetic and real-world datasets, with various cluster shapes and low to high feature dimensionality. Our experiments demonstrate that CLASSIX competes with state-of-the-art clustering algorithms. The algorithm has linear space complexity and achieves near linear time complexity on a wide range of problems. Its inherent simplicity allows for the generation of intuitive explanations of the computed clusters.
    
[^258]: ED2: 连续控制的环境动力学分解世界模型

    ED2: Environment Dynamics Decomposition World Models for Continuous Control

    [https://arxiv.org/abs/2112.02817](https://arxiv.org/abs/2112.02817)

    提出了一种环境动力学分解世界模型构建框架ED2，能够通过发现子动力学并进行分解预测，更准确地构建世界模型。

    

    Model-based reinforcement learning (MBRL)在实践中相对于model-free RL实现了显著的样本效率，但其性能常常受限于模型预测误差的存在。为了减少模型误差，标准的MBRL方法训练一个精心设计的网络来拟合整个环境动力学，但这浪费了可以分别建模的多个子动力学的丰富信息，从而能更准确地构建世界模型。本文提出了环境动力学分解（ED2）的创新世界模型构建框架，其以一种分解的方式对环境进行建模。ED2包含两个关键组成部分：子动力学发现（SD2）和动力学分解预测（D2P）。SD2能够自动发现环境中的子动力学，然后D2P根据这些子动力学构建分解的世界模型。ED2可以与现有方法轻松结合使用。

    arXiv:2112.02817v2 Announce Type: replace-cross  Abstract: Model-based reinforcement learning (MBRL) achieves significant sample efficiency in practice in comparison to model-free RL, but its performance is often limited by the existence of model prediction error. To reduce the model error, standard MBRL approaches train a single well-designed network to fit the entire environment dynamics, but this wastes rich information on multiple sub-dynamics which can be modeled separately, allowing us to construct the world model more accurately. In this paper, we propose the Environment Dynamics Decomposition (ED2), a novel world model construction framework that models the environment in a decomposing manner. ED2 contains two key components: sub-dynamics discovery (SD2) and dynamics decomposition prediction (D2P). SD2 discovers the sub-dynamics in an environment automatically and then D2P constructs the decomposed world model following the sub-dynamics. ED2 can be easily combined with existing
    
[^259]: 结构通过架构：无需正则化的结构化表示

    Structure by Architecture: Structured Representations without Regularization

    [https://arxiv.org/abs/2006.07796](https://arxiv.org/abs/2006.07796)

    我们提出了一种自我监督的结构化表示学习方法，使用无需正则化的自动编码器架构。通过依赖潜变量的独立性进行采样，我们避免了重构质量和生成性能之间的权衡。我们的模型能够学习出一种有序的结构化表示，改善了生成、解缠和外推等多个下游任务的性能。

    

    我们研究了自我监督结构化表示学习的问题，使用自动编码器进行下游任务，如生成模型。与大多数方法依赖于匹配任意的、相对非结构化的先验分布进行采样的情况不同，我们提出了一种仅仅依赖于潜变量的独立性的采样技术，从而避免了在VAE中通常观察到的重构质量和生成性能之间的权衡。我们设计了一种新颖的自动编码器架构，能够学习出一种无需过度正则化的结构化表示。我们的结构解码器学习了一个层次的潜变量，从而无需额外的正则化或监督来对信息进行排序。我们演示了这些模型如何学习出改善各种下游任务的表示，包括生成、解缠和外推，使用了几个具有挑战性的任务。

    arXiv:2006.07796v4 Announce Type: replace  Abstract: We study the problem of self-supervised structured representation learning using autoencoders for downstream tasks such as generative modeling. Unlike most methods which rely on matching an arbitrary, relatively unstructured, prior distribution for sampling, we propose a sampling technique that relies solely on the independence of latent variables, thereby avoiding the trade-off between reconstruction quality and generative performance typically observed in VAEs. We design a novel autoencoder architecture capable of learning a structured representation without the need for aggressive regularization. Our structural decoders learn a hierarchy of latent variables, thereby ordering the information without any additional regularization or supervision. We demonstrate how these models learn a representation that improves results in a variety of downstream tasks including generation, disentanglement, and extrapolation using several challengi
    
[^260]: 通过HawkEye Loss在支持向量回归中提高效率和鲁棒性

    Enhancing Efficiency and Robustness in Support Vector Regression with HawkEye Loss. (arXiv:2401.16785v1 [cs.LG])

    [http://arxiv.org/abs/2401.16785](http://arxiv.org/abs/2401.16785)

    通过引入名为HawkEye损失函数的新的对称损失函数，本文解决了支持向量回归在处理离群值和噪声时遇到的挑战，并提供了增强的泛化性能和鲁棒性。

    

    支持向量回归（SVR）由于其在各个领域的广泛应用而受到了显著的关注，在面对离群值和噪声时，SVR遇到了挑战，主要是由于使用了ε-insensitive损失函数。为了解决这个限制，具有有界损失函数的SVR已成为一种吸引人的替代方案，提供了增强的泛化性能和鲁棒性。值得注意的是，最近的研究关注于设计具有平滑特性的有界损失函数，促进了梯度优化算法的采用。然而，需要强调的是，这些有界和平滑的损失函数不具有一个不敏感的区域。在本文中，我们通过引入一种名为HawkEye损失函数的新的对称损失函数来解决上述约束。值得注意的是，HawkEye损失函数作为SVR中的第一个损失函数突出显示出来。

    Support vector regression (SVR) has garnered significant popularity over the past two decades owing to its wide range of applications across various fields. Despite its versatility, SVR encounters challenges when confronted with outliers and noise, primarily due to the use of the $\varepsilon$-insensitive loss function. To address this limitation, SVR with bounded loss functions has emerged as an appealing alternative, offering enhanced generalization performance and robustness. Notably, recent developments focus on designing bounded loss functions with smooth characteristics, facilitating the adoption of gradient-based optimization algorithms. However, it's crucial to highlight that these bounded and smooth loss functions do not possess an insensitive zone. In this paper, we address the aforementioned constraints by introducing a novel symmetric loss function named the HawkEye loss function. It is worth noting that the HawkEye loss function stands out as the first loss function in SVR
    
[^261]: 大型语言模型的自我解释是否可靠?

    Are self-explanations from Large Language Models faithful?. (arXiv:2401.07927v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2401.07927](http://arxiv.org/abs/2401.07927)

    大型语言模型的自我解释是否可靠是一个重要的AI安全考虑因素，我们提出使用自洽性检测作为评估其可靠性和解释能力的方法。

    

    经过训练的大型语言模型在许多任务上表现出色，甚至能够提供其行为的解释。由于这些模型对公众是直接可访问的，因此存在这样的风险，即令人信服但错误的解释可能导致对大型语言模型的无支撑的自信。因此，解释能力和可靠性是AI安全的重要考虑因素。评估自我解释的可靠性和可解释性是一项具有挑战性的任务，因为这些模型对于人类来说过于复杂，无法注释什么是正确的解释。为了解决这个问题，我们提出使用自洽性检测作为可靠性的衡量指标。例如，如果一个大型语言模型说某组词对于做出预测很重要，那么在没有这些词的情况下，它应该无法做出相同的预测。虽然自洽性检测是一种常见的可靠性方法，但之前尚未应用于大型语言模型的自我解释中。我们将自洽性检测应用于...

    Instruction-tuned large language models (LLMs) excel at many tasks, and will even provide explanations for their behavior. Since these models are directly accessible to the public, there is a risk that convincing and wrong explanations can lead to unsupported confidence in LLMs. Therefore, interpretability-faithfulness of self-explanations is an important consideration for AI Safety. Assessing the interpretability-faithfulness of these explanations, termed self-explanations, is challenging as the models are too complex for humans to annotate what is a correct explanation. To address this, we propose employing self-consistency checks as a measure of faithfulness. For example, if an LLM says a set of words is important for making a prediction, then it should not be able to make the same prediction without these words. While self-consistency checks are a common approach to faithfulness, they have not previously been applied to LLM's self-explanations. We apply self-consistency checks to t
    
[^262]: 大语言模型的零样本位置去偏方法

    Zero-Shot Position Debiasing for Large Language Models. (arXiv:2401.01218v1 [cs.CL])

    [http://arxiv.org/abs/2401.01218](http://arxiv.org/abs/2401.01218)

    本文提出了一种零样本位置去偏方法（ZOE）来降低大语言模型（LLMs）的位置偏差问题，该方法利用预训练的LLMs的无监督响应进行去偏。实验证实ZOE在多个数据集和任务中均表现出优异的性能。

    

    微调已被证明是改善大语言模型（LLMs）领域性能的有效方法。然而，LLMs可能适应数据集偏见和预测的捷径，导致生成性能差。实验结果显示，LLMs容易表现出位置偏差，即利用位于开头或末尾或输入中特定位置线索的信息。现有的减轻位置偏差的工作需要外部偏差知识或带注释的非偏倚样本，在实际中不太实用。在这项工作中，我们提出了一种零样本位置去偏（ZOE）框架对LLMs进行位置去偏。ZOE利用预训练的LLMs的无监督响应进行去偏，因此不需要任何外部知识或数据集。为了提高无监督响应的质量，我们提出了一种主从对齐（MSA）模块来修剪这些响应。对八个数据集和五个任务的实验表明，ZOE始终优于其他方法。

    Fine-tuning has been demonstrated to be an effective method to improve the domain performance of large language models (LLMs). However, LLMs might fit the dataset bias and shortcuts for prediction, leading to poor generation performance. Experimental result shows that LLMs are prone to exhibit position bias, i.e., leveraging information positioned at the beginning or end, or specific positional cues within the input. Existing works on mitigating position bias require external bias knowledge or annotated non-biased samples, which is unpractical in reality. In this work, we propose a zero-shot position debiasing (ZOE) framework to mitigate position bias for LLMs. ZOE leverages unsupervised responses from pre-trained LLMs for debiasing, thus without any external knowledge or datasets. To improve the quality of unsupervised responses, we propose a master-slave alignment (MSA) module to prune these responses. Experiments on eight datasets and five tasks show that ZOE consistently outperform
    
[^263]: 学会说母语：以母语风格激发大型语言模型的能力

    Speak Like a Native: Prompting Large Language Models in a Native Style. (arXiv:2311.13538v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2311.13538](http://arxiv.org/abs/2311.13538)

    本文提出了一种名为AlignedCoT的新颖有效方法，通过将上下文示例与大型语言模型（LLMs）的母语风格对齐，提高了LLMs的推理能力和性能。

    

    大型语言模型（LLMs）与上下文学习（ICL）已成为许多自然语言处理任务的现代工具选择。然而，上下文示例的文本风格如何影响LLMs的性能仍然不足。本文提出了一种名为AlignedCoT的新颖有效的方法，通过将上下文示例与LLMs的母语风格对齐来提高LLMs的推理能力。 "母语"是指LLMs的固有特征，可以通过零-shot场景探测。 AlignedCoT广泛适用于ICL方法，可以轻松与最先进的技术结合，进一步提高LLMs的性能。我们在数学问答、常识推理和文本理解等多个基准测试上进行了广泛而全面的实验。实证结果表明，我们的AlignedCoT相比精心手工制作的演示文稿显著提高了性能。

    In-context learning (ICL) with large language models (LLMs) has become the modern tools of choice for many natural language processing tasks. However, how the text style of in-context examples influences the performance of LLMs still remains under-explored. This paper presents a novel and effective approach, named \textbf{AlignedCoT}, to improve the reasoning capability of LLMs by aligning the in-context examples with the native style of LLMs.''Native'' refers to the inherent characteristic of LLMs which can be probed by zero-shot scenarios.AlignedCoT is widely applicable to ICL methods, making it easy to combine with state-of-the-art techniques to further improve the LLMs' performance. We conduct extensive and comprehensive experiments on several benchmarks on mathematical question-answering, common-sense reasoning, and text understanding. The empirical results demonstrate that our AlignedCoT significantly improves performance over the carefully handcrafted demonstrations. Specificall
    
[^264]: 更好的公平性胜于遗憾：针对公平GNN的对抗性缺失数据填充

    Better Fair than Sorry: Adversarial Missing Data Imputation for Fair GNNs. (arXiv:2311.01591v1 [cs.LG])

    [http://arxiv.org/abs/2311.01591](http://arxiv.org/abs/2311.01591)

    该论文提出了一种针对公平GNN的对抗性缺失数据填充模型，以解决现有公平GNN的假设问题。实验证明此模型的有效性。

    

    本文解决了在缺失保护属性的情况下学习公平图神经网络（GNNs）的问题。在许多相关任务中，决策可能会对特定社区产生不成比例的影响，而GNNs已经在这些任务中取得了最先进的结果。然而，现有的公平GNNs工作要么假设保护属性是完全被观察到的，要么假设缺失数据的填充是公平的。实际上，填充中的偏差会传播到模型的结果中，导致它们过高地估计了其预测的公平性。我们通过提出Better Fair than Sorry（BFtS），为公平GNNs使用的保护属性的公平缺失数据填充模型来解决这个挑战。BFtS背后的关键设计原则是填充应该近似于公平GNN的最困难情况，即在最优化公平性最困难的情况下。我们使用一个三方对抗方案来实现这个想法，在这个方案中，两个对手共同对抗公平GNN。通过使用合成和实际数据集的实验证明了BFtS的有效性。

    This paper addresses the problem of learning fair Graph Neural Networks (GNNs) under missing protected attributes. GNNs have achieved state-of-the-art results in many relevant tasks where decisions might disproportionately impact specific communities. However, existing work on fair GNNs assumes that either protected attributes are fully-observed or that the missing data imputation is fair. In practice, biases in the imputation will be propagated to the model outcomes, leading them to overestimate the fairness of their predictions. We address this challenge by proposing Better Fair than Sorry (BFtS), a fair missing data imputation model for protected attributes used by fair GNNs. The key design principle behind BFtS is that imputations should approximate the worst-case scenario for the fair GNN -- i.e. when optimizing fairness is the hardest. We implement this idea using a 3-player adversarial scheme where two adversaries collaborate against the fair GNN. Experiments using synthetic and
    
[^265]: 贝叶斯多状态Bennett接受比率方法

    Bayesian Multistate Bennett Acceptance Ratio Methods. (arXiv:2310.20699v2 [physics.chem-ph] UPDATED)

    [http://arxiv.org/abs/2310.20699](http://arxiv.org/abs/2310.20699)

    贝叶斯多状态Bennett接受比率方法（BayesMBAR）是多状态Bennett接受比率（MBAR）方法的贝叶斯推广。通过整合采样配置和先验分布，BayesMBAR计算了自由能的后验分布，并提供更准确的不确定性估计。

    

    多状态Bennett接受比率（MBAR）方法是计算热力学状态下自由能的一种常用方法。在这项工作中，我们引入了BayesMBAR，即MBAR方法的贝叶斯推广。通过将来自热力学状态的采样配置与先验分布进行整合，BayesMBAR计算了自由能的后验分布。利用后验分布，我们推导出自由能的估计值并计算它们的相关不确定性。值得注意的是，当使用均匀先验分布时，BayesMBAR可以恢复MBAR的结果，但提供更准确的不确定性估计。此外，当有关自由能的先验知识可用时，BayesMBAR可以通过使用非均匀先验分布将这些信息纳入估计过程中。作为示例，我们展示了通过引入关于自由能曲面平滑性的先验知识，BayesMBAR比MBAR方法提供更准确的估计结果。

    The multistate Bennett acceptance ratio (MBAR) method is a prevalent approach for computing free energies of thermodynamic states. In this work, we introduce BayesMBAR, a Bayesian generalization of the MBAR method. By integrating configurations sampled from thermodynamic states with a prior distribution, BayesMBAR computes a posterior distribution of free energies. Using the posterior distribution, we derive free energy estimations and compute their associated uncertainties. Notably, when a uniform prior distribution is used, BayesMBAR recovers the MBAR's result but provides more accurate uncertainty estimates. Additionally, when prior knowledge about free energies is available, BayesMBAR can incorporate this information into the estimation procedure by using non-uniform prior distributions. As an example, we show that, by incorporating the prior knowledge about the smoothness of free energy surfaces, BayesMBAR provides more accurate estimates than the MBAR method. Given MBAR's widespr
    
[^266]: 打破、模仿、修复：通过生成人类攻击提高鲁棒性

    Break it, Imitate it, Fix it: Robustness by Generating Human-Like Attacks. (arXiv:2310.16955v1 [cs.LG])

    [http://arxiv.org/abs/2310.16955](http://arxiv.org/abs/2310.16955)

    本研究提出了一个对抗训练框架，使用有限的人类对手示例生成更有用的大规模对抗示例，有效提高了自然语言处理系统对于人类对手的鲁棒性。

    

    现实世界中的自然语言处理系统需要对抗人类对手具有鲁棒性。收集人类对手的示例进行训练是一种有效但昂贵的解决方案。另一方面，训练针对小扰动（如词替换）的合成攻击实际上并不能提高对抗人类对手的鲁棒性。本文提出了一个对抗训练框架，使用有限的人类对手示例来生成更有用的大规模对抗示例。我们通过ANLI和仇恨言论检测基准数据集进行实验，这两个数据集是通过迭代的对抗人类和模型的过程收集得到的。与仅在观察到的人类攻击上进行训练相比，也在我们的合成对抗示例上进行训练可以提高模型对未来回合的鲁棒性。在ANLI上，我们看到了对当前攻击集的准确率提升（44.1% -> 50.1%），以及对两个未见过的人类生成攻击回合的准确率提升（32.5% -> 43%）。

    Real-world natural language processing systems need to be robust to human adversaries. Collecting examples of human adversaries for training is an effective but expensive solution. On the other hand, training on synthetic attacks with small perturbations - such as word-substitution - does not actually improve robustness to human adversaries. In this paper, we propose an adversarial training framework that uses limited human adversarial examples to generate more useful adversarial examples at scale. We demonstrate the advantages of this system on the ANLI and hate speech detection benchmark datasets - both collected via an iterative, adversarial human-and-model-in-the-loop procedure. Compared to training only on observed human attacks, also training on our synthetic adversarial examples improves model robustness to future rounds. In ANLI, we see accuracy gains on the current set of attacks (44.1%$\,\to\,$50.1%) and on two future unseen rounds of human generated attacks (32.5%$\,\to\,$43
    
[^267]: 绝对策略优化

    Absolute Policy Optimization. (arXiv:2310.13230v1 [cs.LG])

    [http://arxiv.org/abs/2310.13230](http://arxiv.org/abs/2310.13230)

    这篇论文提出了绝对策略优化（APO）的方法，通过优化一个新颖的目标函数，在保证性能下界的同时，实现了连续控制任务和Atari游戏中的令人瞩目的结果。

    

    近年来，基于信任域的在线策略强化学习在解决复杂控制任务和游戏场景方面取得了令人瞩目的结果。然而，这一类别中现有的最先进算法主要强调对预期性能的改进，缺乏对最坏情况下性能结果的控制能力。为了解决这个限制，我们引入了一个新颖的目标函数；通过优化该函数，可以确保近乎总体性能样本的下界（绝对性能）呈现单调改进。考虑到这一具有突破性的理论进展，我们通过一系列的近似对这个理论基础算法进行了改进，得到了一种实用的解决方案称为绝对策略优化（APO）。我们的实验证明了我们的方法在具有挑战性的连续控制基准任务上的有效性，并将其适用性扩展到掌握Atari游戏。我们的发现表明，APO在提高性能的同时也显著改善了最坏情况下的性能结果。

    In recent years, trust region on-policy reinforcement learning has achieved impressive results in addressing complex control tasks and gaming scenarios. However, contemporary state-of-the-art algorithms within this category primarily emphasize improvement in expected performance, lacking the ability to control over the worst-case performance outcomes. To address this limitation, we introduce a novel objective function; by optimizing which, it will lead to guaranteed monotonic improvement in the lower bound of near-total performance samples (absolute performance). Considering this groundbreaking theoretical advancement, we then refine this theoretically grounded algorithm through a series of approximations, resulting in a practical solution called Absolute Policy Optimization (APO). Our experiments demonstrate the effectiveness of our approach across challenging continuous control benchmark tasks and extend its applicability to mastering Atari games. Our findings reveal that APO signifi
    
[^268]: 基于因果相似性的分层贝叶斯模型

    Causal Similarity-Based Hierarchical Bayesian Models. (arXiv:2310.12595v1 [cs.LG])

    [http://arxiv.org/abs/2310.12595](http://arxiv.org/abs/2310.12595)

    本文提出了一种基于因果相似性的分层贝叶斯模型，通过学习如何从具有相似因果机制的训练任务中汇集数据来提高机器学习算法对新任务的泛化能力。

    

    机器学习的关键挑战是对新数据的泛化能力。本研究探讨了对由相关任务组成的数据集进行泛化的问题，这些任务可能在因果机制上存在差异。例如，复杂疾病的观察性医学数据在不同患者间具有疾病因果机制的异质性，这给需要对训练数据集之外的新患者进行泛化的机器学习算法带来了挑战。常用的处理异质性数据集的方法包括为整个数据集学习一个全局模型，为每个任务的数据学习本地模型，或者利用分层、元学习和多任务学习方法从汇集的多个任务的数据中学习泛化。本文提出了基于因果相似性的分层贝叶斯模型，通过学习如何从具有相似因果机制的训练任务中汇集数据来提高对新任务的泛化能力。我们应用这种通用建模方法

    The key challenge underlying machine learning is generalisation to new data. This work studies generalisation for datasets consisting of related tasks that may differ in causal mechanisms. For example, observational medical data for complex diseases suffers from heterogeneity in causal mechanisms of disease across patients, creating challenges for machine learning algorithms that need to generalise to new patients outside of the training dataset. Common approaches for learning supervised models with heterogeneous datasets include learning a global model for the entire dataset, learning local models for each tasks' data, or utilising hierarchical, meta-learning and multi-task learning approaches to learn how to generalise from data pooled across multiple tasks. In this paper we propose causal similarity-based hierarchical Bayesian models to improve generalisation to new tasks by learning how to pool data from training tasks with similar causal mechanisms. We apply this general modelling
    
[^269]: NeuroCUT：一种用于鲁棒图分区的神经方法

    NeuroCUT: A Neural Approach for Robust Graph Partitioning. (arXiv:2310.11787v1 [cs.LG])

    [http://arxiv.org/abs/2310.11787](http://arxiv.org/abs/2310.11787)

    NeuroCUT是一种神经方法，用于解决鲁棒的图分区问题。它通过两个关键创新，即对图拓扑和分区计数具有归纳性，以及利用强化学习基础，能够从数据中学习启发式方法。

    

    图分区旨在将图分割为k个不相交的子集，同时优化特定的分区目标。由于其组合性质，大部分与图分区相关的问题都呈现出NP难度。因此，传统的近似算法依赖于启发式方法，有时带有近似保证，有时则没有。不幸的是，传统方法针对特定的分区目标进行优化，不适用于其他已知的文献中的分区目标。为了克服这个限制，并直接从数据中学习启发式方法，神经方法应运而生，并展示出令人期待的结果。在本研究中，我们通过一个新颖的框架NeuroCut扩展了这一领域的工作。NeuroCut在现有方法上引入了两个关键创新。首先，它对图拓扑和分区计数具有归纳性，这些信息在查询时提供。其次，通过利用强化学习基础

    Graph partitioning aims to divide a graph into $k$ disjoint subsets while optimizing a specific partitioning objective. The majority of formulations related to graph partitioning exhibit NP-hardness due to their combinatorial nature. As a result, conventional approximation algorithms rely on heuristic methods, sometimes with approximation guarantees and sometimes without. Unfortunately, traditional approaches are tailored for specific partitioning objectives and do not generalize well across other known partitioning objectives from the literature. To overcome this limitation, and learn heuristics from the data directly, neural approaches have emerged, demonstrating promising outcomes. In this study, we extend this line of work through a novel framework, NeuroCut. NeuroCut introduces two key innovations over prevailing methodologies. First, it is inductive to both graph topology and the partition count, which is provided at query time. Second, by leveraging a reinforcement learning base
    
[^270]: ByteStack-ID: 基于灰度图像的网络入侵检测的集成堆叠模型，利用负载字节频率

    ByteStack-ID: Integrated Stacked Model Leveraging Payload Byte Frequency for Grayscale Image-based Network Intrusion Detection. (arXiv:2310.09298v1 [cs.CR])

    [http://arxiv.org/abs/2310.09298](http://arxiv.org/abs/2310.09298)

    ByteStack-ID是一种基于灰度图像和负载字节频率的集成堆叠模型，用于数据包级入侵检测。它能迅速准确地识别网络流量中的各种攻击类型，并与传统方法有所不同。

    

    在不断发展的网络安全领域中，迅速准确地识别网络流量中的各种攻击类型至关重要。本文介绍了"ByteStack-ID"，一种专为数据包级入侵检测而设计的创新方法。ByteStack-ID核心是利用从负载数据的频率分布生成的灰度图像，这是一种突破性的技术，极大地提高了模型识别复杂数据模式的能力。值得注意的是，我们的方法完全基于数据包级信息，与传统的基于流量数据的网络入侵检测系统（NIDS）有所不同。在基本堆叠方法的基础上，ByteStack-ID与传统的堆叠方法不同。它将附加的元学习器层无缝集成到连接的基础学习器中，创建了一个高度优化的统一模型。

    In the ever-evolving realm of network security, the swift and accurate identification of diverse attack classes within network traffic is of paramount importance. This paper introduces "ByteStack-ID," a pioneering approach tailored for packet-level intrusion detection. At its core, ByteStack-ID leverages grayscale images generated from the frequency distributions of payload data, a groundbreaking technique that greatly enhances the model's ability to discern intricate data patterns. Notably, our approach is exclusively grounded in packet-level information, a departure from conventional Network Intrusion Detection Systems (NIDS) that predominantly rely on flow-based data. While building upon the fundamental concept of stacking methodology, ByteStack-ID diverges from traditional stacking approaches. It seamlessly integrates additional meta learner layers into the concatenated base learners, creating a highly optimized, unified model. Empirical results unequivocally confirm the outstandin
    
[^271]: Fourier神经操作符的初始化偏差：重新审视混沌边缘

    Initialization Bias of Fourier Neural Operator: Revisiting the Edge of Chaos. (arXiv:2310.06379v1 [cs.LG])

    [http://arxiv.org/abs/2310.06379](http://arxiv.org/abs/2310.06379)

    本文研究了Fourier神经操作符(FNO)的初始化偏差，提出了一种FNO版本的He初始化方案，通过模式截断和密集连接网络相似的特点，解决了训练不稳定的负初始化偏差问题。

    

    本文研究了Fourier神经操作符(FNO)的初始化偏差。建立了一个针对FNO的平均场理论，从“混沌边缘”的视角分析了随机FNO的行为。我们揭示了前向和反向传播行为表现出与FNO独特的特征，这是由模式截断引起的，同时也展示了与密集连接网络相似的特点。基于这一观察，我们还提出了一种FNO版本的He初始化方案，以减轻导致训练不稳定的负初始化偏差。实验结果显示了我们初始化方案的有效性，使得32层FNO的训练稳定，无需额外技术或显著性能下降。

    This paper investigates the initialization bias of the Fourier neural operator (FNO). A mean-field theory for FNO is established, analyzing the behavior of the random FNO from an ``edge of chaos'' perspective. We uncover that the forward and backward propagation behaviors exhibit characteristics unique to FNO, induced by mode truncation, while also showcasing similarities to those of densely connected networks. Building upon this observation, we also propose a FNO version of the He initialization scheme to mitigate the negative initialization bias leading to training instability. Experimental results demonstrate the effectiveness of our initialization scheme, enabling stable training of a 32-layer FNO without the need for additional techniques or significant performance degradation.
    
[^272]: 无标记的域外数据改善了泛化能力

    Unlabeled Out-Of-Domain Data Improves Generalization. (arXiv:2310.00027v1 [stat.ML])

    [http://arxiv.org/abs/2310.00027](http://arxiv.org/abs/2310.00027)

    这个论文提出了一种新的框架，可以将无标记的域外数据纳入半监督分类问题，从而改善泛化能力。该框架结合了分布鲁棒优化与自监督训练，并利用了高效的多项式时间算法。在理论上，该框架在高斯混合分类问题中得到了验证。

    

    我们提出了一种将无标记数据纳入半监督分类问题的新框架，其中考虑了最小化鲁棒性损失函数或非鲁棒性损失函数的情景。值得注意的是，我们允许无标记样本在总变差意义上略微偏离域内分布。我们的框架的核心思想是将分布鲁棒优化（DRO）与自监督训练相结合。因此，我们还利用了训练阶段的高效多项式时间算法。从理论上讲，我们将我们的框架应用于在$\mathbb{R}^d$中的两个高斯混合分类问题，除了来自真实分布的$m$个独立标记样本之外，还给出了一组$n$个（通常$n\gg m$）域外和无标记样本。已知仅使用标记数据，泛化误差可以通过$\propto\left(d/m\right)$进行界定。

    We propose a novel framework for incorporating unlabeled data into semi-supervised classification problems, where scenarios involving the minimization of either i) adversarially robust or ii) non-robust loss functions have been considered. Notably, we allow the unlabeled samples to deviate slightly (in total variation sense) from the in-domain distribution. The core idea behind our framework is to combine Distributionally Robust Optimization (DRO) with self-supervised training. As a result, we also leverage efficient polynomial-time algorithms for the training stage. From a theoretical standpoint, we apply our framework on the classification problem of a mixture of two Gaussians in $\mathbb{R}^d$, where in addition to the $m$ independent and labeled samples from the true distribution, a set of $n$ (usually with $n\gg m$) out of domain and unlabeled samples are gievn as well. Using only the labeled data, it is known that the generalization error can be bounded by $\propto\left(d/m\right
    
[^273]: 通过Sobolev训练的二维Copula逼近变换：2-Cats网络

    Differential 2D Copula Approximating Transforms via Sobolev Training: 2-Cats Networks. (arXiv:2309.16391v1 [cs.LG])

    [http://arxiv.org/abs/2309.16391](http://arxiv.org/abs/2309.16391)

    本文介绍了一种通过Sobolev训练的2-Cats网络，它能够非参数地逼近任何二维Copula，并且在估计输出方面优于现有技术。

    

    Copula是一种强大的统计工具，用于捕捉数据维度之间的依赖关系。在应用Copula时，我们可以通过首先估计独立的边际分布（一个简单任务），然后估计连接边际的单个Copula函数C（一个困难任务）来估计多元分布函数。对于二维数据，Copula是一个形如C：(u，v)∈\mathbf{I}^2\rightarrow \mathbf{I}的二次增函数，其中\mathbf{I}=[0，1]。在本文中，我们展示了神经网络（NNs）如何能够非参数地逼近任何二维Copula。我们的方法被称为2-Cats，受到物理启发的神经网络和Sobolev训练文献的启发。我们不仅证明了我们能够比现有技术更好地估计2D Copula的输出，而且我们的方法是非参数的，并且符合Copula C的数学性质。

    Copulas are a powerful statistical tool that captures dependencies across data dimensions. When applying Copulas, we can estimate multivariate distribution functions by initially estimating independent marginals, an easy task, and then a single copulating function, $C$, to connect the marginals, a hard task. For two-dimensional data, a copula is a two-increasing function of the form $C: (u,v)\in \mathbf{I}^2 \rightarrow \mathbf{I}$, where $\mathbf{I} = [0, 1]$. In this paper, we show how Neural Networks (NNs) can approximate any two-dimensional copula non-parametrically. Our approach, denoted as 2-Cats, is inspired by the Physics-Informed Neural Networks and Sobolev Training literature. Not only do we show that we can estimate the output of a 2d Copula better than the state-of-the-art, our approach is non-parametric and respects the mathematical properties of a Copula $C$.
    
[^274]: 对于神经网络的大批量训练泛化性能的LARS再审视

    Revisiting LARS for Large Batch Training Generalization of Neural Networks. (arXiv:2309.14053v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2309.14053](http://arxiv.org/abs/2309.14053)

    本文通过对大批量训练技术的研究，提出了一种新的算法TVLARS，该算法利用可配置的函数替代了热身阶段，以实现对于神经网络的稳健训练。实验证明，在大多数情况下，TVLARS比LARS和LAMB都有更好的性能表现，特别是在自监督学习方面。

    

    本文通过在不同场景下使用逐层自适应缩放比(LARS)来探索大批量训练技术，揭示了一些见解。具有热身阶段的LARS算法由于冗余的比例缩放导致在早期陷入尖锐的极小化器。此外，后期固定的陡峭下降限制了深度神经网络有效地遍历早期尖锐的极小化器。基于这些发现，我们提出了一种新的算法Time Varying LARS (TVLARS)，它用可配置的类似sigmoid函数替代了热身阶段，以实现在初始阶段的稳健训练。TVLARS在早期促进了梯度探索，超越了尖锐的优化器，并逐渐过渡到LARS以实现后期的稳健性。广泛的实验表明，在大多数情况下，TVLARS始终优于LARS和LAMB，分类场景中的改进达到2\%。值得注意的是，在所有自监督学习的案例中，TVLARS都胜过了LARS和LAMB，并且性能提升了

    This paper explores Large Batch Training techniques using layer-wise adaptive scaling ratio (LARS) across diverse settings, uncovering insights. LARS algorithms with warm-up tend to be trapped in sharp minimizers early on due to redundant ratio scaling. Additionally, a fixed steep decline in the latter phase restricts deep neural networks from effectively navigating early-phase sharp minimizers. Building on these findings, we propose Time Varying LARS (TVLARS), a novel algorithm that replaces warm-up with a configurable sigmoid-like function for robust training in the initial phase. TVLARS promotes gradient exploration early on, surpassing sharp optimizers and gradually transitioning to LARS for robustness in later phases. Extensive experiments demonstrate that TVLARS consistently outperforms LARS and LAMB in most cases, with up to 2\% improvement in classification scenarios. Notably, in all self-supervised learning cases, TVLARS dominates LARS and LAMB with performance improvements of
    
[^275]: MEDL-U：基于证据深度学习的不确定性感知的3D自动标注

    MEDL-U: Uncertainty-aware 3D Automatic Annotation based on Evidential Deep Learning. (arXiv:2309.09599v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2309.09599](http://arxiv.org/abs/2309.09599)

    本文提出了一种基于证据深度学习的方法，用于解决3D对象检测中伪标签的模糊性问题，并生成准确的伪标签和量化伪标签的不确定性。

    

    深度学习在基于3D对象检测方面取得了进展，这需要大规模数据集的支持。然而，这一要求引入了手动标注的挑战，这通常是繁重且耗时的。为了解决这个问题，文献中出现了几种弱监督的3D对象检测框架，可以自动生成未标记数据的伪标签。然而，这些生成的伪标签包含噪声，不如人工标注的准确。在本文中，我们提出了第一种解决伪标签中固有模糊性的方法，引入了一种基于证据深度学习（EDL）的不确定性估计框架。具体而言，我们提出了MEDL-U，它是基于MTrans的EDL框架，不仅生成伪标签，还量化了相关的不确定性。然而，将EDL应用于3D对象检测面临三个主要挑战：(1)相对较低的伪标签准确性。

    Advancements in deep learning-based 3D object detection necessitate the availability of large-scale datasets. However, this requirement introduces the challenge of manual annotation, which is often both burdensome and time-consuming. To tackle this issue, the literature has seen the emergence of several weakly supervised frameworks for 3D object detection which can automatically generate pseudo labels for unlabeled data. Nevertheless, these generated pseudo labels contain noise and are not as accurate as those labeled by humans. In this paper, we present the first approach that addresses the inherent ambiguities present in pseudo labels by introducing an Evidential Deep Learning (EDL) based uncertainty estimation framework. Specifically, we propose MEDL-U, an EDL framework based on MTrans, which not only generates pseudo labels but also quantifies the associated uncertainties. However, applying EDL to 3D object detection presents three primary challenges: (1) relatively lower pseudolab
    
[^276]: 在2022/23年监测乌克兰马里乌波尔市的城市变化

    Monitoring Urban Changes in Mariupol/Ukraine in 2022/23. (arXiv:2309.08607v1 [cs.CY])

    [http://arxiv.org/abs/2309.08607](http://arxiv.org/abs/2309.08607)

    本文研究证明使用历史数据进行迁移学习是解决城市变化监测问题的可行方案，通过使用合成孔径雷达和光学多光谱观测数据，成功监测了乌克兰马里乌波尔市在俄乌冲突开始阶段的相关城市变化。

    

    不断监测城市变化的能力具有巨大的社会经济利益。之前的研究已经展示了使用深度神经网络（DNNs）和迁移学习在这一领域的方法。然而，它们未能展示在训练或迁移领域之外的时间尺度。本研究在现有研究的基础上，证明了使用历史数据进行迁移学习是可行的解决方案，仍然可以对以后的年份进行城市变化监测。我们考虑了一个对公共和免费高分辨率图像访问有限的情况来指导迁移。为了提供高时空分辨率，我们的监测方法的核心数据包括来自Sentinel 1（合成孔径雷达）和Sentinel 2（光学多光谱）的多模态合成孔径雷达和光学多光谱观测。我们选择了实际应用我们的方法来监测乌克兰马里乌波尔市与俄乌冲突开始时的相关城市变化。

    The ability to constantly monitor urban changes is of large socio-economic interest. Previous works have already shown approaches in this field with the use of Deep Neural Networks (DNNs) and transfer learning. However, they fell short in demonstrating temporal scale outside of either the training or transfer domain.  This work builds on existing research and proves that transfer learning with the use of historic data is a feasible solution, which still allows the urban change monitoring of later years. We considered a case with limited access to public and free Very High Resolution (VHR) imagery to guide the transfer. To provide a high temporal resolution, the core data of our monitoring method comprised multi-modal Synthetic Aperture Radar (SAR) and optical multispectral observations from Sentinel 1 and Sentinel 2, respectively.  We chose a practical application of our methods for monitoring urban-related changes in the city of Mariupol in Ukraine during the beginning of the Russo-Uk
    
[^277]: ConR: 用于深度不平衡回归的对比正则化器

    ConR: Contrastive Regularizer for Deep Imbalanced Regression. (arXiv:2309.06651v1 [cs.LG])

    [http://arxiv.org/abs/2309.06651](http://arxiv.org/abs/2309.06651)

    ConR是一种对比正则化器，通过建模全局和局部标签相似性，防止少数样本的特征被折叠到其多数邻居中，有效地处理深度不平衡回归问题。

    

    不平衡分布在现实世界的数据中很常见。它们对深度神经网络提出了约束，以表示少数类别标签并避免对多数类别的偏见。大量的不平衡方法处理了分类标签空间，但在连续标签空间的回归问题上未能有效应用。相反，连续标签之间的局部和全局关联为在特征空间中有效建模关系提供了有价值的见解。在这项工作中，我们提出了ConR，一种对比正则化器，它在特征空间中建模全局和局部标签相似性，防止少数样本的特征被折叠到它们的多数邻居中。通过将预测的相似性作为特征相似性的指示器，ConR区分了标签空间和特征空间之间的不一致，并对这些不一致施加惩罚。ConR通过两个主要策略关注标签空间的连续性。

    Imbalanced distributions are ubiquitous in real-world data. They create constraints on Deep Neural Networks to represent the minority labels and avoid bias towards majority labels. The extensive body of imbalanced approaches address categorical label spaces but fail to effectively extend to regression problems where the label space is continuous. Conversely, local and global correlations among continuous labels provide valuable insights towards effectively modelling relationships in feature space. In this work, we propose ConR, a contrastive regularizer that models global and local label similarities in feature space and prevents the features of minority samples from being collapsed into their majority neighbours. Serving the similarities of the predictions as an indicator of feature similarities, ConR discerns the dissagreements between the label space and feature space and imposes a penalty on these disagreements. ConR minds the continuous nature of label space with two main strategi
    
[^278]: 通过生成对抗神经算子实现宽带地面运动合成: 开发与验证

    Broadband Ground Motion Synthesis via Generative Adversarial Neural Operators: Development and Validation. (arXiv:2309.03447v1 [physics.geo-ph])

    [http://arxiv.org/abs/2309.03447](http://arxiv.org/abs/2309.03447)

    本论文提出了一种使用生成对抗神经算子的数据驱动地面运动合成模型，可以根据不同参数生成三分量加速度时间历史。通过使用神经算子架构，模型训练不受数据采样频率影响。研究结果表明，该模型在验证和应用实例中表现出色，并可用于生成日本地震动数据。

    

    我们提出了一种使用生成对抗神经算子（GANO）的数据驱动地面运动合成模型，该模型结合了机器学习和开放获取的强震动数据集，可以根据矩震级（M）、断裂距离（R_{rup}）、顶部30m处的时间平均剪切波速度（V_{S30}）和构造环境或断层类型生成三分量加速度时间历史。我们使用神经算子，这是一种分辨率无关的架构，可以保证模型训练与数据采样频率无关。首先，我们提出了条件地面运动合成算法（以下简称cGM-GANO）并讨论其与先前工作相比的优势。接下来，我们使用南加州地震中心（SCEC）宽带平台（BBP）产生的模拟地震动验证了cGM-GANO框架。最后，我们在日本的KiK-net数据集上训练了cGM-GANO，表明该框架可以重新生成地震动数据。

    We present a data-driven model for ground-motion synthesis using a Generative Adversarial Neural Operator (GANO) that combines recent advancements in machine learning and open access strong motion data sets to generate three-component acceleration time histories conditioned on moment magnitude ($M$), rupture distance ($R_{rup}$), time-average shear-wave velocity at the top $30m$ ($V_{S30}$), and tectonic environment or style of faulting. We use Neural Operators, a resolution invariant architecture that guarantees that the model training is independent of the data sampling frequency. We first present the conditional ground-motion synthesis algorithm (referred to heretofore as cGM-GANO) and discuss its advantages compared to previous work. Next, we verify the cGM-GANO framework using simulated ground motions generated with the Southern California Earthquake Center (SCEC) Broadband Platform (BBP). We lastly train cGM-GANO on a KiK-net dataset from Japan, showing that the framework can rec
    
[^279]: 交互式和集中式差分隐私在Bandit问题中的应用

    Interactive and Concentrated Differential Privacy for Bandits. (arXiv:2309.00557v1 [stat.ML])

    [http://arxiv.org/abs/2309.00557](http://arxiv.org/abs/2309.00557)

    本文研究了在交互学习和推荐系统中隐私保护的Bandit问题，并引入了集中差分隐私的概念。通过提供关于有限臂和线性Bandit问题遗憾的下界，我们揭示了不同隐私预算下的难度区域，并发现集中差分隐私可以比全局差分隐私更有效地保护隐私，我们提出了两种相应的算法。

    

    Bandit问题在交互式学习方案和现代推荐系统中起着至关重要的作用。然而，这些系统通常依赖于敏感的用户数据，因此隐私是一个重要问题。本文通过交互式差分隐私的视角研究了基于可信集中式决策者的Bandit问题的隐私性。虽然已经对纯ε-全局差分隐私的Bandit问题进行了广泛研究，但我们在理解零集中差分隐私(zCDP)的Bandit问题方面做出了贡献。针对有限臂和线性Bandit问题，我们提供了关于遗憾的最小最大和问题相关下界，从而量化了这些情况下ρ-全局zCDP的代价。这些下界揭示了基于隐私预算ρ的两个困难区域，并表明ρ-全局zCDP比纯ε-全局差分隐私产生的遗憾更小。我们提出了两种有限臂和线性Bandit问题的ρ-全局zCDP算法，即AdaC-UCB和AdaC-GOPE。这两个算法都使用了高斯机制的共同策略。

    Bandits play a crucial role in interactive learning schemes and modern recommender systems. However, these systems often rely on sensitive user data, making privacy a critical concern. This paper investigates privacy in bandits with a trusted centralized decision-maker through the lens of interactive Differential Privacy (DP). While bandits under pure $\epsilon$-global DP have been well-studied, we contribute to the understanding of bandits under zero Concentrated DP (zCDP). We provide minimax and problem-dependent lower bounds on regret for finite-armed and linear bandits, which quantify the cost of $\rho$-global zCDP in these settings. These lower bounds reveal two hardness regimes based on the privacy budget $\rho$ and suggest that $\rho$-global zCDP incurs less regret than pure $\epsilon$-global DP. We propose two $\rho$-global zCDP bandit algorithms, AdaC-UCB and AdaC-GOPE, for finite-armed and linear bandits respectively. Both algorithms use a common recipe of Gaussian mechanism 
    
[^280]: 逃离样本陷阱：使用配对距离估计器快速准确地估计认识不确定性

    Escaping the Sample Trap: Fast and Accurate Epistemic Uncertainty Estimation with Pairwise-Distance Estimators. (arXiv:2308.13498v1 [cs.LG])

    [http://arxiv.org/abs/2308.13498](http://arxiv.org/abs/2308.13498)

    本文介绍了使用配对距离估计器对集成模型进行认识不确定性估计的新方法，相比于常用的深度学习方法，该方法能够更快速、更准确地在更大的空间和更高维度上估计认识不确定性。

    

    本文介绍了一种使用配对距离估计器（PaiDEs）对集成模型进行认识不确定性估计的新方法。这些估计器利用模型组件之间的配对距离来建立熵的边界，并将这些边界作为基于信息准则的估计值。与最近基于样本的蒙特卡洛估计器用于认识不确定性估计的深度学习方法不同，PaiDEs能够在更大的空间（最多100倍）上以更快的速度（最多100倍）估计认识不确定性，并在更高维度上具有更准确的性能。为了验证我们的方法，我们进行了一系列用于评估认识不确定性估计的实验：一维正弦数据，摆动物体（Pendulum-v0），跳跃机器人（Hopper-v2），蚂蚁机器人（Ant-v2）和人形机器人（Humanoid-v2）。对于每个实验设置，我们应用了主动学习框架来展示PaiDEs在认识不确定性估计中的优势。

    This work introduces a novel approach for epistemic uncertainty estimation for ensemble models using pairwise-distance estimators (PaiDEs). These estimators utilize the pairwise-distance between model components to establish bounds on entropy and uses said bounds as estimates for information-based criterion. Unlike recent deep learning methods for epistemic uncertainty estimation, which rely on sample-based Monte Carlo estimators, PaiDEs are able to estimate epistemic uncertainty up to 100$\times$ faster, over a larger space (up to 100$\times$) and perform more accurately in higher dimensions. To validate our approach, we conducted a series of experiments commonly used to evaluate epistemic uncertainty estimation: 1D sinusoidal data, Pendulum-v0, Hopper-v2, Ant-v2 and Humanoid-v2. For each experimental setting, an Active Learning framework was applied to demonstrate the advantages of PaiDEs for epistemic uncertainty estimation.
    
[^281]: 规范化就是你所需要的：理解极端标签偏移下的层归一化联邦学习

    Normalization Is All You Need: Understanding Layer-Normalized Federated Learning under Extreme Label Shift. (arXiv:2308.09565v1 [cs.LG])

    [http://arxiv.org/abs/2308.09565](http://arxiv.org/abs/2308.09565)

    本论文揭示了层归一化和联邦学习中的标签偏移问题之间的深刻联系，通过在联邦学习中应用特征归一化，使得对严重倾斜的数据集进行加速全局训练，从而在极端标签偏移下获得显著改进。

    

    层归一化（LN）是一个广泛采用的深度学习技术，特别在基础模型的时代。最近，已经证明LN在非独立同分布数据上的联邦学习（FL）中非常有效。然而，它为什么以及如何起作用仍然是个谜。在这项工作中，我们揭示了层归一化和联邦学习中的标签偏移问题之间的深刻联系。为了更好地理解FL中的层归一化，我们确定了规范化方法在FL中的关键贡献机制，称之为特征归一化（FN），它在分类器头之前将归一化应用于潜在特征表示。虽然LN和FN不会提高表达能力，但它们控制特征崩溃和局部过拟合，使得对严重倾斜的数据集进行加速全局训练。经验证明，规范化在极端标签偏移下可以引起标准基准的显著改进。此外，我们还进行了大量的割除研究。

    Layer normalization (LN) is a widely adopted deep learning technique especially in the era of foundation models. Recently, LN has been shown to be surprisingly effective in federated learning (FL) with non-i.i.d. data. However, exactly why and how it works remains mysterious. In this work, we reveal the profound connection between layer normalization and the label shift problem in federated learning. To understand layer normalization better in FL, we identify the key contributing mechanism of normalization methods in FL, called feature normalization (FN), which applies normalization to the latent feature representation before the classifier head. Although LN and FN do not improve expressive power, they control feature collapse and local overfitting to heavily skewed datasets, and thus accelerates global training. Empirically, we show that normalization leads to drastic improvements on standard benchmarks under extreme label shift. Moreover, we conduct extensive ablation studies to unde
    
[^282]: MDB：互动查询数据集和模型

    MDB: Interactively Querying Datasets and Models. (arXiv:2308.06686v1 [cs.DB])

    [http://arxiv.org/abs/2308.06686](http://arxiv.org/abs/2308.06686)

    MDB是一个调试框架，用于互动查询数据集和模型。它通过集成函数式编程与关系代数，能够快速迭代和优化查询，发现和描述错误和模型行为。实验证明，MDB比其他工具能够实现更快的查询速度加快和查询长度缩短。

    

    随着模型的训练和部署，开发者需要能够系统地调试在机器学习流程中出现的错误。我们提出了MDB，一个用于互动查询数据集和模型的调试框架。MDB通过将函数式编程与关系代数结合起来，构建了一个对数据集和模型预测的数据库进行表达性查询的工具。查询可重用且易于修改，使得调试人员能够快速迭代和优化查询，以发现和描述错误和模型行为。我们在目标检测、偏差发现、图像分类和数据填充任务中评估了MDB在自动驾驶视频、大型语言模型和医疗记录上的性能。我们的实验证明，MDB比其他基准测试工具能够实现最高10倍的查询速度加快和40%的查询长度缩短。在用户研究中，我们发现开发者能够成功构建复杂查询来描述机器学习模型的错误。

    As models are trained and deployed, developers need to be able to systematically debug errors that emerge in the machine learning pipeline. We present MDB, a debugging framework for interactively querying datasets and models. MDB integrates functional programming with relational algebra to build expressive queries over a database of datasets and model predictions. Queries are reusable and easily modified, enabling debuggers to rapidly iterate and refine queries to discover and characterize errors and model behaviors. We evaluate MDB on object detection, bias discovery, image classification, and data imputation tasks across self-driving videos, large language models, and medical records. Our experiments show that MDB enables up to 10x faster and 40\% shorter queries than other baselines. In a user study, we find developers can successfully construct complex queries that describe errors of machine learning models.
    
[^283]: A/B测试和具有非稳态鲁棒性的线性赌博机最佳臂识别问题

    A/B Testing and Best-arm Identification for Linear Bandits with Robustness to Non-stationarity. (arXiv:2307.15154v1 [cs.LG])

    [http://arxiv.org/abs/2307.15154](http://arxiv.org/abs/2307.15154)

    本文研究了在非稳态环境中的线性赌博机的最佳臂识别问题，提出了一种具有鲁棒性的算法来解决。该算法通过在每个时间步从一个G-最优设计中随机选择臂来实现最佳臂的鲁棒识别。

    

    本文研究了在可能存在非稳态环境下的线性赌博机中的固定预算最佳臂识别问题。给定有限臂集合X，固定预算T以及不可预测的参数序列θ，算法的目标是以尽可能高的概率正确识别最佳臂x*。之前的工作已经在稳态设置下进行了研究，并且证明了错误概率随着预算的增加而指数下降。但在许多现实世界的A/B/n多变量测试场景中，环境是非稳态的，而一个期望稳态的算法很容易失败。为了具有鲁棒的识别能力，众所周知，如果在每个时间步从X的一个G-最优设计中以随机和非自适应的方式选择臂，那么可以实现最佳臂的鲁棒识别。

    We investigate the fixed-budget best-arm identification (BAI) problem for linear bandits in a potentially non-stationary environment. Given a finite arm set $\mathcal{X}\subset\mathbb{R}^d$, a fixed budget $T$, and an unpredictable sequence of parameters $\left\lbrace\theta_t\right\rbrace_{t=1}^{T}$, an algorithm will aim to correctly identify the best arm $x^* := \arg\max_{x\in\mathcal{X}}x^\top\sum_{t=1}^{T}\theta_t$ with probability as high as possible. Prior work has addressed the stationary setting where $\theta_t = \theta_1$ for all $t$ and demonstrated that the error probability decreases as $\exp(-T /\rho^*)$ for a problem-dependent constant $\rho^*$. But in many real-world $A/B/n$ multivariate testing scenarios that motivate our work, the environment is non-stationary and an algorithm expecting a stationary setting can easily fail. For robust identification, it is well-known that if arms are chosen randomly and non-adaptively from a G-optimal design over $\mathcal{X}$ at each 
    
[^284]: 隐私放大通过重要性采样

    Privacy Amplification via Importance Sampling. (arXiv:2307.10187v1 [cs.CR])

    [http://arxiv.org/abs/2307.10187](http://arxiv.org/abs/2307.10187)

    通过重要性采样进行隐私放大，可以同时增强隐私保护和提高效用。我们提供了一个一般的结果来量化选择概率权重对隐私放大的影响，并展示了异质采样概率可以在保持子采样大小不变的情况下获得更好的隐私和效用。

    

    我们研究了通过重要性采样对数据集进行子采样作为差分隐私机制的预处理步骤来增强隐私保护的性质。这扩展了已有的通过子采样进行隐私放大的结果到重要性采样，其中每个数据点的权重为其被选择概率的倒数。每个点的选择概率的权重对隐私的影响并不明显。一方面，较低的选择概率会导致更强的隐私放大。另一方面，权重越高，在点被选择时，点对机制输出的影响就越强。我们提供了一个一般的结果来量化这两个影响之间的权衡。我们展示了异质采样概率可以同时比均匀子采样具有更强的隐私和更好的效用，并保持子采样大小不变。特别地，我们制定和解决了隐私优化采样的问题，即寻找...

    We examine the privacy-enhancing properties of subsampling a data set via importance sampling as a pre-processing step for differentially private mechanisms. This extends the established privacy amplification by subsampling result to importance sampling where each data point is weighted by the reciprocal of its selection probability. The implications for privacy of weighting each point are not obvious. On the one hand, a lower selection probability leads to a stronger privacy amplification. On the other hand, the higher the weight, the stronger the influence of the point on the output of the mechanism in the event that the point does get selected. We provide a general result that quantifies the trade-off between these two effects. We show that heterogeneous sampling probabilities can lead to both stronger privacy and better utility than uniform subsampling while retaining the subsample size. In particular, we formulate and solve the problem of privacy-optimal sampling, that is, finding
    
[^285]: 可编程合成表格数据生成

    Programmable Synthetic Tabular Data Generation. (arXiv:2307.03577v1 [cs.LG])

    [http://arxiv.org/abs/2307.03577](http://arxiv.org/abs/2307.03577)

    这项工作介绍了ProgSyn，第一个可编程的合成表格数据生成算法，它允许对生成的数据进行全面的自定义，并且通过预训练和微调生成模型来确保高质量的数据和遵守自定义规范。

    

    由于隐私、数据质量和数据共享的限制，大量的表格数据仍然被低效利用。尽管训练一个能够生成类似原始分布的合成数据的生成模型可以解决其中一些问题，但大多数应用程序还需要额外的生成数据约束。现有的合成数据方法存在局限性，因为它们通常只处理特定的约束条件，例如差分隐私（DP）或增加公平性，并且缺乏一个可访问的接口来声明一般规范。在这项工作中，我们介绍了ProgSyn，这是第一个可编程的合成表格数据生成算法，它允许对生成的数据进行全面的自定义。为了确保高质量的数据并遵守自定义规范，ProgSyn在原始数据集上进行预训练生成模型，并在提供的规范上自动推导出可微分损失进行微调。这些规范可以使用统计和。

    Large amounts of tabular data remain underutilized due to privacy, data quality, and data sharing limitations. While training a generative model producing synthetic data resembling the original distribution addresses some of these issues, most applications require additional constraints from the generated data. Existing synthetic data approaches are limited as they typically only handle specific constraints, e.g., differential privacy (DP) or increased fairness, and lack an accessible interface for declaring general specifications. In this work, we introduce ProgSyn, the first programmable synthetic tabular data generation algorithm that allows for comprehensive customization over the generated data. To ensure high data quality while adhering to custom specifications, ProgSyn pre-trains a generative model on the original dataset and fine-tunes it on a differentiable loss automatically derived from the provided specifications. These can be programmatically declared using statistical and
    
[^286]: 学习受限动力学的稳定神经微分方程

    Stabilized Neural Differential Equations for Learning Constrained Dynamics. (arXiv:2306.09739v1 [cs.LG])

    [http://arxiv.org/abs/2306.09739](http://arxiv.org/abs/2306.09739)

    本文提出了一种稳定神经微分方程（SNDEs）的方法，可以强制使用任意流形约束。该方法通过添加稳定项使约束流形成为渐进稳定的，并且在实验中表现优于现有方法。

    

    最近出现了许多成功的从数据学习动态系统的方法。然而，确保推断出的动态系统保留已知约束条件（例如守恒定律或对允许的系统状态的限制）仍然具有挑战性。我们提出了稳定神经微分方程（SNDEs）的方法，这是一种用于神经微分方程强制使用任意流形约束的方法。我们的方法基于一个稳定项，当添加到原始动态系统中时，可以将约束流形成为渐进稳定的。由于其简单性，我们的方法与所有常见的神经常微分方程（NODE）模型兼容并广泛适用。在广泛的经验评估中，我们证明SNDE在扩展可纳入NODE训练的约束类型方面胜过现有方法。

    Many successful methods to learn dynamical systems from data have recently been introduced. However, assuring that the inferred dynamics preserve known constraints, such as conservation laws or restrictions on the allowed system states, remains challenging. We propose stabilized neural differential equations (SNDEs), a method to enforce arbitrary manifold constraints for neural differential equations. Our approach is based on a stabilization term that, when added to the original dynamics, renders the constraint manifold provably asymptotically stable. Due to its simplicity, our method is compatible with all common neural ordinary differential equation (NODE) models and broadly applicable. In extensive empirical evaluations, we demonstrate that SNDEs outperform existing methods while extending the scope of which types of constraints can be incorporated into NODE training.
    
[^287]: 多尺度流用于鲁棒和最优宇宙学分析

    Multiscale Flow for Robust and Optimal Cosmological Analysis. (arXiv:2306.04689v1 [astro-ph.CO])

    [http://arxiv.org/abs/2306.04689](http://arxiv.org/abs/2306.04689)

    用多尺度流进行二维宇宙学数据的生成和建模，可识别不同尺度的信息并显著胜过现有方法。

    

    本文提出了多尺度流(Convolutional Normalizing Flow)用于生成二维宇宙学数据，并对之进行建模和分析。该模型通过小波基础分解宇宙学场，然后将不同级别的小波分量分别建模。通过逐项求和得出原始宇宙学场的对数可能性，从而分离不同尺度的信息并识别其中未知的尺度相关系统性。

    We propose Multiscale Flow, a generative Normalizing Flow that creates samples and models the field-level likelihood of two-dimensional cosmological data such as weak lensing. Multiscale Flow uses hierarchical decomposition of cosmological fields via a wavelet basis, and then models different wavelet components separately as Normalizing Flows. The log-likelihood of the original cosmological field can be recovered by summing over the log-likelihood of each wavelet term. This decomposition allows us to separate the information from different scales and identify distribution shifts in the data such as unknown scale-dependent systematics. The resulting likelihood analysis can not only identify these types of systematics, but can also be made optimal, in the sense that the Multiscale Flow can learn the full likelihood at the field without any dimensionality reduction. We apply Multiscale Flow to weak lensing mock datasets for cosmological inference, and show that it significantly outperform
    
[^288]: 分布式SGD算法的稳定性与泛化分析改进

    Improved Stability and Generalization Analysis of the Decentralized SGD Algorithm. (arXiv:2306.02939v1 [cs.LG])

    [http://arxiv.org/abs/2306.02939](http://arxiv.org/abs/2306.02939)

    本文提出了新的算法稳定性理论来改进分布式SGD算法的泛化性能分析，推翻了现有技术对通信图负面影响的观点，并展示了D-SGD在凸设置中与经典SGD算法泛化界相同。

    

    本文基于算法稳定性，提出了分布式随机梯度下降(D-SGD)算法的新的泛化误差分析方法。得到的结果大大改进了现有技术，并推翻了它们关于通信图对泛化的负面影响的观点。例如，在凸设置中，无论图的选择如何，D-SGD具有与经典SGD算法相同的泛化界。我们发现这种反直觉的结果来自于考虑本地参数的平均值，这会隐藏一个与分布式场景不兼容的最终全局平均化步骤。考虑到这一观察结果，我们倡导分析本地参数的上确界，并展示了在这种情况下，图确实对泛化产生影响。与之前的结果不同，我们的分析即使对于非连接图也能产生非平凡边界。

    This paper presents a new generalization error analysis for the Decentralized Stochastic Gradient Descent (D-SGD) algorithm based on algorithmic stability. The obtained results largely improve upon state-of-the-art results, and even invalidate their claims that the communication graph has a detrimental effect on generalization. For instance, we show that in convex settings, D-SGD has the same generalization bounds as the classical SGD algorithm, no matter the choice of graph. We exhibit that this counter-intuitive result comes from considering the average of local parameters, which hides a final global averaging step incompatible with the decentralized scenario. In light of this observation, we advocate to analyze the supremum over local parameters and show that in this case, the graph does have an impact on the generalization. Unlike prior results, our analysis yields non-vacuous bounds even for non-connected graphs.
    
[^289]: 通过分离学习解决混淆节点问题

    Clarify Confused Nodes Through Separated Learning. (arXiv:2306.02285v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2306.02285](http://arxiv.org/abs/2306.02285)

    本文提出了使用邻域混淆度量来分离学习解决图神经网络中混淆节点的问题。这种方法可以更可靠地区分异质节点和同质节点，并改善性能。

    

    图神经网络（GNN）在图导向任务中取得了显著的进展。然而，现实世界的图中不可避免地包含一定比例的异质节点，这挑战了经典GNN的同质性假设，并阻碍了其性能。现有研究大多数仍设计了具有异质节点和同质节点间共享权重的通用模型。尽管这些努力中包含了高阶信息和多通道架构，但往往效果不佳。少数研究尝试训练不同节点组的分离学习，但受到了不合适的分离度量和低效率的影响。本文首先提出了一种新的度量指标，称为邻域混淆（NC），以便更可靠地分离节点。我们观察到具有不同NC值的节点组在组内准确度和可视化嵌入上存在一定差异。这为基于邻域混淆的图卷积网络（NC-GCN）铺平了道路。

    Graph neural networks (GNNs) have achieved remarkable advances in graph-oriented tasks. However, real-world graphs invariably contain a certain proportion of heterophilous nodes, challenging the homophily assumption of classical GNNs and hindering their performance. Most existing studies continue to design generic models with shared weights between heterophilous and homophilous nodes. Despite the incorporation of high-order messages or multi-channel architectures, these efforts often fall short. A minority of studies attempt to train different node groups separately but suffer from inappropriate separation metrics and low efficiency. In this paper, we first propose a new metric, termed Neighborhood Confusion (NC), to facilitate a more reliable separation of nodes. We observe that node groups with different levels of NC values exhibit certain differences in intra-group accuracy and visualized embeddings. These pave the way for Neighborhood Confusion-guided Graph Convolutional Network (N
    
[^290]: 通过安全层实现垂直联邦学习的高效安全聚合

    vFedSec: Efficient Secure Aggregation for Vertical Federated Learning via Secure Layer. (arXiv:2305.16794v1 [cs.CR])

    [http://arxiv.org/abs/2305.16794](http://arxiv.org/abs/2305.16794)

    vFedSec提出了一个用于垂直联邦学习的新型Secure Layer，旨在使用最先进的安全模块，实现安全高效的联合训练。实验结果表明，该方法在保护数据隐私效果显著，不会影响训练性能。

    

    隐私保护联邦学习主要关注横向划分的数据集，而在许多有趣的问题中，个体数据点分散在垂直的客户端/组织中。这种情况下的联邦学习需要参与者之间交换中间输出和梯度，若不考虑隐私和安全问题，可能会导致隐私泄露的风险。本文提出了vFedSec，通过创新性的安全层设计和最先进的安全模块，在保护数据隐私的同时，实现了垂直联邦学习的安全和高效。理论上证明了我们的方法不影响训练绩效，同时有效保护私人数据。实验结果也表明了我们的设计的应用性和保护能力。

    Most work in privacy-preserving federated learning (FL) has been focusing on horizontally partitioned datasets where clients share the same sets of features and can train complete models independently. However, in many interesting problems, individual data points are scattered across different clients/organizations in a vertical setting. Solutions for this type of FL require the exchange of intermediate outputs and gradients between participants, posing a potential risk of privacy leakage when privacy and security concerns are not considered. In this work, we present vFedSec - a novel design with an innovative Secure Layer for training vertical FL securely and efficiently using state-of-the-art security modules in secure aggregation. We theoretically demonstrate that our method does not impact the training performance while protecting private data effectively. Empirically results also show its applicability with extensive experiments that our design can achieve the protection with negl
    
[^291]: 通过事后对数归一化和温度缩放改善深度神经网络的选择分类性能

    Improving selective classification performance of deep neural networks through post-hoc logit normalization and temperature scaling. (arXiv:2305.15508v1 [cs.LG])

    [http://arxiv.org/abs/2305.15508](http://arxiv.org/abs/2305.15508)

    本文提出了一种$p$-NormSoftmax的事后置信度估计器来提高深度神经网络的选择分类性能。

    

    本文解决深度神经网络的选择分类问题，其中模型可以避免潜在错误通过放弃低置信度的预测。我们针对的是优化固定分类器的置信度估计器，旨在增强其误分类检测性能，即通过将更高的置信度值分配给正确的预测来区分正确和不正确的预测。我们提出了一个简单有效的事后置信度估计器$p$-NormSoftmax，通过对数进行$p$-范数归一化和温度缩放得到。

    This paper addresses the problem of selective classification for deep neural networks, where a model is allowed to abstain from low-confidence predictions to avoid potential errors. Specifically, we tackle the problem of optimizing the confidence estimator of a fixed classifier, aiming to enhance its misclassification detection performance, i.e., its ability to discriminate between correct and incorrect predictions by assigning higher confidence values to the correct ones. Previous work has found that different classifiers exhibit varying levels of misclassification detection performance, particularly when using the maximum softmax probability (MSP) as a measure of confidence. However, we argue that these findings are mainly due to a sub-optimal confidence estimator being used for each model. To overcome this issue, we propose a simple and efficient post-hoc confidence estimator, named $p$-NormSoftmax, which consists of transforming the logits through $p$-norm normalization and tempera
    
[^292]: 学习结构化成分：迈向模块化且可解释的多元时间序列预测

    Learning Structured Components: Towards Modular and Interpretable Multivariate Time Series Forecasting. (arXiv:2305.13036v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2305.13036](http://arxiv.org/abs/2305.13036)

    本文提出了一个名为SCNN的模块化和解释性的预测框架，旨在单独对空间-时间模式的每个成分进行建模。SCNN使用预定义的MTS生成过程，将MTS数据分解为结构化和异构成分，然后分别推断这些成分的演化，能够实现比现有先进模型更高的性能。

    

    多元时间序列（MTS）预测是许多现实世界应用中一个极为重要和基本的问题。 MTS预测中的核心问题是如何有效地建模复杂的空间 - 时间模式。 在本文中，我们开发了一个模块化且可解释的预测框架，该框架旨在单独对空间 - 时间模式的每个成分进行建模。 我们将此框架命名为SCNN，缩写为结构化基于成分的神经网络。

    Multivariate time-series (MTS) forecasting is a paramount and fundamental problem in many real-world applications. The core issue in MTS forecasting is how to effectively model complex spatial-temporal patterns. In this paper, we develop a modular and interpretable forecasting framework, which seeks to individually model each component of the spatial-temporal patterns. We name this framework SCNN, short for Structured Component-based Neural Network. SCNN works with a pre-defined generative process of MTS, which arithmetically characterizes the latent structure of the spatial-temporal patterns. In line with its reverse process, SCNN decouples MTS data into structured and heterogeneous components and then respectively extrapolates the evolution of these components, the dynamics of which is more traceable and predictable than the original MTS. Extensive experiments are conducted to demonstrate that SCNN can achieve superior performance over state-of-the-art models on three real-world data
    
[^293]: 通过贝叶斯主动学习实现自校正贝叶斯优化

    Self-Correcting Bayesian Optimization through Bayesian Active Learning. (arXiv:2304.11005v1 [cs.LG])

    [http://arxiv.org/abs/2304.11005](http://arxiv.org/abs/2304.11005)

    该论文提出了SAL和SCoreBO两种方法，用于提高高斯过程模型的超参数选择和贝叶斯优化的表现。

    

    高斯过程已成为贝叶斯优化和主动学习中的首选模型。然而，高斯过程的完全发挥需要巧妙选择超参数，而在文献中很少有关于找到正确超参数的努力。我们演示了选择好的超参数对于高斯过程的影响，并提出了两个明确优先考虑此目标的收购函数。统计距离主动学习（SAL）考虑后验样本的平均不一致性，由统计距离测量。结果显示，在许多测试函数上，它胜过了贝叶斯主动学习的最新结果。然后，我们引入了自校正贝叶斯优化（SCoreBO），它将SAL扩展到同时执行贝叶斯优化和主动超参数学习。相比传统BO，SCoreBO以改进的速度学习模型超参数，同时在最新的贝叶斯优化搜索中取得更好的表现。

    Gaussian processes are cemented as the model of choice in Bayesian optimization and active learning. Yet, they are severely dependent on cleverly chosen hyperparameters to reach their full potential, and little effort is devoted to finding the right hyperparameters in the literature. We demonstrate the impact of selecting good hyperparameters for GPs and present two acquisition functions that explicitly prioritize this goal. Statistical distance-based Active Learning (SAL) considers the average disagreement among samples from the posterior, as measured by a statistical distance. It is shown to outperform the state-of-the-art in Bayesian active learning on a number of test functions. We then introduce Self-Correcting Bayesian Optimization (SCoreBO), which extends SAL to perform Bayesian optimization and active hyperparameter learning simultaneously. SCoreBO learns the model hyperparameters at improved rates compared to vanilla BO, while outperforming the latest Bayesian optimization met
    
[^294]: 使用物理知识约束的神经网络进行微震源成像

    Microseismic source imaging using physics-informed neural networks with hard constraints. (arXiv:2304.04315v1 [physics.geo-ph])

    [http://arxiv.org/abs/2304.04315](http://arxiv.org/abs/2304.04315)

    本论文提出一种使用物理知识约束的神经网络（PINNs）进行直接微震成像的方法，能够生成聚焦的源图像，即使只有极少的记录。数值实验表明，该方法可以产生可靠且精确的结果。

    

    微震源成像在被动地震监测中起着重要作用，但由于稀疏的测量数据容易出现混叠问题，导致其常常失败。因此，我们提出了一种基于物理知识约束的神经网络（PINN）的直接微震成像框架，它可以生成聚焦的源图像，即使只有极少的记录。我们使用PINNs表示多频波场，然后应用逆傅里叶变换来提取源图像。特别地，我们通过硬约束来修改频域波场的表示形式，从而本质上满足边界条件（表层上的测量数据），避免了在PINNs中平衡数据和PDE损失的困难。此外，我们提出了关于深度的因果性损失实现，以提高PINNs的收敛性。在Overthrust模型上的数值实验表明，该方法可以产生可靠且精确的结果。

    Microseismic source imaging plays a significant role in passive seismic monitoring. However, such a process is prone to failure due to the aliasing problem when dealing with sparse measured data. Thus, we propose a direct microseismic imaging framework based on physics-informed neural networks (PINNs), which can generate focused source images, even with very sparse recordings. We use the PINNs to represent a multi-frequency wavefield and then apply the inverse Fourier transform to extract the source image. Specially, we modify the representation of the frequency-domain wavefield to inherently satisfy the boundary conditions (the measured data on the surface) by means of the hard constraint, which helps to avoid the difficulty in balancing the data and PDE losses in PINNs. Furthermore, we propose the causality loss implementation with respect to depth to enhance the convergence of PINNs. The numerical experiments on the Overthrust model show that the method can admit reliable and accura
    
[^295]: 使用Rockpool将脉冲神经网络应用程序训练部署到混合信号神经形态芯片Dynap-SE2上

    Training and Deploying Spiking NN Applications to the Mixed-Signal Neuromorphic Chip Dynap-SE2 with Rockpool. (arXiv:2303.12167v1 [cs.ET])

    [http://arxiv.org/abs/2303.12167](http://arxiv.org/abs/2303.12167)

    本文介绍了一种通过优化网络参数和注入对抗性参数噪声，将SNN应用程序离线训练和部署到Dynap-SE2混合信号神经形态处理器的新方法。优化后的网络表现出很强的鲁棒性，对于硬件约束的真实世界应用程序有很大的潜力。

    

    混合信号神经形态处理器利用脉冲神经网络（SNN）内的稀疏异步计算提供极低功耗的边缘推理负载。然而，由于模拟硬件参数的受限可控性以及由于制造非理想性所导致的模拟电路的无意参数和动态变化，将稳健的应用程序部署到这些设备是复杂的。本文展示了一种用于将SNN应用程序离线训练和部署到混合信号神经形态处理器Dynap-SE2的新型方法。该方法利用一种无监督的重量量化方法来优化网络的参数，并结合在训练过程中注入对抗性参数噪声。优化的网络表现出很强的鲁棒性，可以抵御量化和设备不匹配的影响，使该方法成为具有硬件约束的真实世界应用程序的有前景的候选方法。这项工作扩展了开源设计工具Rockpool。

    Mixed-signal neuromorphic processors provide extremely low-power operation for edge inference workloads, taking advantage of sparse asynchronous computation within Spiking Neural Networks (SNNs). However, deploying robust applications to these devices is complicated by limited controllability over analog hardware parameters, unintended parameter and dynamics variations of analog circuits due to fabrication non-idealities. Here we demonstrate a novel methodology for offline training and deployment of spiking neural networks (SNNs) to the mixed-signal neuromorphic processor Dynap-SE2. The methodology utilizes an unsupervised weight quantization method to optimize the network's parameters, coupled with adversarial parameter noise injection during training. The optimized network is shown to be robust to the effects of quantization and device mismatch, making the method a promising candidate for real-world applications with hardware constraints. This work extends Rockpool, an open-source de
    
[^296]: 逆可解性和安全性及其在联邦学习中的应用

    Inverse Solvability and Security with Applications to Federated Learning. (arXiv:2211.14115v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2211.14115](http://arxiv.org/abs/2211.14115)

    介绍了逆可解性和安全性的概念，以及其在联邦学习中的应用。论文提供了模型示例，展示了如何通过增加用户数量来增加可解性和安全性。

    

    我们介绍了逆可解性和安全性的概念，适用于一般线性前向模型，并展示了如何将其应用于联邦学习中使用的模型。我们提供了这样的模型的示例，其逆可解性和安全性在本文中得到定义。我们还展示了如何利用参与给定迭代的大量用户来增加可解性和安全性。最后，我们讨论了所提出概念的可能扩展，包括非线性情况。

    We introduce the concepts of inverse solvability and security for a generic linear forward model and demonstrate how they can be applied to models used in federated learning. We provide examples of such models which differ in the resulting inverse solvability and security as defined in this paper. We also show how the large number of users participating in a given iteration of federated learning can be leveraged to increase both solvability and security. Finally, we discuss possible extensions of the presented concepts including the nonlinear case.
    

