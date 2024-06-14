# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Predictive Performance Comparison of Decision Policies Under Confounding](https://arxiv.org/abs/2404.00848) | 提出了一种方法，通过现代识别方法比较决策政策的预测性能，关键在于可以安全忽略不确定性区域。 |
| [^2] | [A Geometric Explanation of the Likelihood OOD Detection Paradox](https://arxiv.org/abs/2403.18910) | 高似然区域将不会被生成如果它们包含最小概率质量，基于此观察提出了一种通过本地固有维度估计进行离群检测的方法 |
| [^3] | [Provably Robust Score-Based Diffusion Posterior Sampling for Plug-and-Play Image Reconstruction](https://arxiv.org/abs/2403.17042) | 开发了一个算法框架，用于将基于得分的扩散模型作为通用非线性逆的表达数据先验。 |
| [^4] | [Self-Improvement for Neural Combinatorial Optimization: Sample without Replacement, but Improvement](https://arxiv.org/abs/2403.15180) | 通过结合循环式随机束搜索和来自可证策略改进的更新策略，本研究在神经组合优化中引入了一种新的训练方法，从而在最小采样次数中实现逐步改进的解决方案。 |
| [^5] | [Larimar: Large Language Models with Episodic Memory Control](https://arxiv.org/abs/2403.11901) | Larimar提出了一种大脑启发的架构，通过分布式情节记忆增强LLMs，实现了动态、一次性的知识更新，无需昂贵的重新训练或微调，且在速度和灵活性上表现出色。 |
| [^6] | [Sound Event Detection and Localization with Distance Estimation](https://arxiv.org/abs/2403.11827) | 本文将声音事件检测和定位任务扩展为具有距离估计的3D SELD，探讨了两种集成距离估计的方法，并在Ambisonic和双耳版本的声音场景下进行了实验。 |
| [^7] | [Using Uncertainty Quantification to Characterize and Improve Out-of-Domain Learning for PDEs](https://arxiv.org/abs/2403.10642) | 通过集成多个神经算子来提高对区域外学习的不确定性估计，从而解决现有方法在OOD测试输入上的失败 |
| [^8] | [LLM-Assisted Light: Leveraging Large Language Model Capabilities for Human-Mimetic Traffic Signal Control in Complex Urban Environments](https://arxiv.org/abs/2403.08337) | 本研究提出了将大型语言模型(LLMs)整合到交通信号控制(TSC)系统中的创新方法，以解决传统TSC系统在适应不熟悉场景方面的限制，并提出了一个混合框架，使得LLMs与一系列感知和决策工具相结合，从而提升TSC系统对城市交通复杂性和变异性的管理能力。 |
| [^9] | [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/abs/2403.02310) | 引入了一种有效的LLM推理调度程序Sarathi-Serve，通过分块预装填技术平衡了GPU计算饱和和单个标记处理的挑战，实现了高吞吐量和低延迟。 |
| [^10] | [Shifted Interpolation for Differential Privacy](https://arxiv.org/abs/2403.00278) | 本文在统一框架下建立了“通过迭代实现隐私放大”现象，提高了先前分析的水平，并由此获得了其他差分隐私概念更紧密的隐私核算。 |
| [^11] | [Batch and match: black-box variational inference with a score-based divergence](https://arxiv.org/abs/2402.14758) | BaM是一种基于分数的离散的BBVI替代方法，针对高方差梯度估计慢收敛问题，能够在高斯变分族中通过封闭形式的近端更新进行优化，在目标分布为高斯时，批处理大小趋于无穷时变分参数更新将指数快速收敛到目标均值和协方差，BaM在多种生成模型推断中表现出良好性能 |
| [^12] | [Theoretical Analysis of Submodular Information Measures for Targeted Data Subset Selection](https://arxiv.org/abs/2402.13454) | 通过推导与相关性和覆盖性相关的基于相似度的界限，为子模互信息在目标数据子集选择中的表现提供了理论保证 |
| [^13] | [CounterCurate: Enhancing Physical and Semantic Visio-Linguistic Compositional Reasoning via Counterfactual Examples](https://arxiv.org/abs/2402.13254) | 本研究提出CounterCurate框架，通过对比例子和生成式微调，全面提升视觉-语言组合推理能力，解决了物理推理和语义对照微调方面的关键问题，实现了显著性能改进。 |
| [^14] | [Random Projection Layers for Multidimensional Time Sires Forecasting](https://arxiv.org/abs/2402.10487) | 提出了一种全MLP时间序列预测架构RPMixer，通过将随机投影层集成到模型中，增加了块输出之间的多样性，提高了整体性能 |
| [^15] | [Classification Diffusion Models](https://arxiv.org/abs/2402.10095) | 提出了一种分类扩散模型（CDMs），该模型采用了去噪扩散模型（DDM）的形式，并利用一个分类器来预测加在干净信号上的噪声量，取得了在图像、视频和音频生成方面的最先进结果。 |
| [^16] | [Predictive Linear Online Tracking for Unknown Targets](https://arxiv.org/abs/2402.10036) | 本文提出了一种名为预测性线性在线追踪（PLOT）的算法，用于在线追踪未知目标。该算法使用具有指数遗忘的递归最小二乘法来学习目标的时变动态模型，并在递推视线控制的框架下使用所学模型进行最优策略。与先前的工作不同，我们的理论结果适用于非平稳目标。 |
| [^17] | [Get More with LESS: Synthesizing Recurrence with KV Cache Compression for Efficient LLM Inference](https://arxiv.org/abs/2402.09398) | 这项研究提出了一个称为LESS的方法，通过集成一个固定尺寸的缓存和基于驱逐的缓存方法，可以在大型语言模型中减小内存占用的问题，同时保持全部标记的可查询能力，并在多种任务上显示出良好的性能。 |
| [^18] | [Chinese MentalBERT: Domain-Adaptive Pre-training on Social Media for Chinese Mental Health Text Analysis](https://arxiv.org/abs/2402.09151) | 本文介绍了一种领域自适应预训练模型Chinese MentalBERT，该模型针对中国社交媒体上心理健康文本分析进行了优化，在预训练过程中加入心理学词典，提高了模型的适用性。 |
| [^19] | [SLEB: Streamlining LLMs through Redundancy Verification and Elimination of Transformer Blocks](https://arxiv.org/abs/2402.09025) | SLEB是一种通过消除冗余的Transformer块来优化LLM流程的新方法，它成功加速了LLM的推理过程。 |
| [^20] | [Understanding Practical Membership Privacy of Deep Learning](https://arxiv.org/abs/2402.06674) | 该论文利用最先进的成员推理攻击方法系统地测试了细调大型图像分类模型的实际隐私漏洞，并发现数据集中每个类别的示例数量以及训练结束时的大梯度与成员推理攻击的漏洞之间存在关联。 |
| [^21] | [The Loss Landscape of Shallow ReLU-like Neural Networks: Stationary Points, Saddle Escaping, and Network Embedding](https://arxiv.org/abs/2402.05626) | 本文研究了使用ReLU-like激活函数以经验平方损失训练的单隐藏层神经网络的损失景观，提出了稳定点条件和逃逸神经元的定义，并将鞍点逃逸与逃逸神经元的参数变化联系起来。 |
| [^22] | [Conformal Monte Carlo Meta-learners for Predictive Inference of Individual Treatment Effects](https://arxiv.org/abs/2402.04906) | 本研究提出了一种新方法，即一致性蒙特卡洛元学习模型，用于预测个体治疗效果。通过利用一致性预测系统、蒙特卡洛采样和CATE元学习模型，该方法生成可用于个性化决策的预测分布。实验结果显示，该方法在保持较小区间宽度的情况下具有强大的实验覆盖范围，可以提供真实个体治疗效果的估计。 |
| [^23] | [Informed Reinforcement Learning for Situation-Aware Traffic Rule Exceptions](https://arxiv.org/abs/2402.04168) | 本论文引入了基于信息的增强学习方法，将结构化的规则手册作为知识源，利用情境感知的奖励设计评估学习轨迹，以实现对需要控制交通规则例外的情景的学习。 |
| [^24] | [Subsampling is not Magic: Why Large Batch Sizes Work for Differentially Private Stochastic Optimisation](https://arxiv.org/abs/2402.03990) | 通过研究差分隐私随机梯度下降（DP-SGD）中的总梯度方差，我们发现大批次大小有助于减小則采樣引起的方差，从而提高优化效果。 |
| [^25] | [LLMs Can't Plan, But Can Help Planning in LLM-Modulo Frameworks](https://arxiv.org/abs/2402.01817) | LLMs无法独自进行规划或自我验证，但在规划/推理任务中可以作为通用近似知识源发挥更大作用。 |
| [^26] | [Multi-group Learning for Hierarchical Groups](https://arxiv.org/abs/2402.00258) | 本研究将多组学习扩展到具有层次结构的情况，设计了一个近乎最优的样本复杂度的算法，输出可解释且确定性的决策树预测器，并在真实数据集上取得了有吸引力的广义化特性。 |
| [^27] | [Reliability and Interpretability in Science and Deep Learning](https://arxiv.org/abs/2401.07359) | 这篇论文强调了科学与深度学习中模型假设的重要性，并提供了对模型假设认识论复杂性的分析，同时结合标准错误分析与深度神经网络模型的特点，来评估模型可靠性。 |
| [^28] | [Emergence of In-Context Reinforcement Learning from Noise Distillation](https://arxiv.org/abs/2312.12275) | 该论文介绍了一种从噪声中生成上下文强化学习的方法，通过构建噪声注入课程来获取学习历史，可以实现在学习数据集中超过最优策略的性能表现。 |
| [^29] | [Low-Cost High-Power Membership Inference Attacks](https://arxiv.org/abs/2312.03262) | 提出了一种新颖、高效且强大的成员推断攻击（RMIA），具有更准确的建模和更高的测试能力，适用于隐私风险评估。 |
| [^30] | [LLM Paternity Test: Generated Text Detection with LLM Genetic Inheritance](https://arxiv.org/abs/2305.12519) | LLM-Pat提出了一种基于模型的生成文本检测方法，通过重建并比较候选文本与其对应的“兄弟”文本的相似性，从而判断候选文本是否由机器生成。 |
| [^31] | [KGLiDS: A Platform for Semantic Abstraction, Linking, and Automation of Data Science](https://arxiv.org/abs/2303.02204) | 提出了一个可扩展平台KGLiDS，利用机器学习和知识图技术来抽象和捕获数据科学工具及其联系的语义，从而支持数据发现和管道自动化。 |
| [^32] | [Design Your Own Universe: A Physics-Informed Agnostic Method for Enhancing Graph Neural Networks.](http://arxiv.org/abs/2401.14580) | 本文提出了一种物理信息引导的无偏方法来增强图神经网络，通过引入附加节点和使用正负权重重连连接来丰富图结构，以解决过度平滑和过度压缩的问题。 |
| [^33] | [Code Simulation Challenges for Large Language Models.](http://arxiv.org/abs/2401.09074) | 大型语言模型在模拟计算机代码和算法执行方面遇到挑战，性能随着代码长度的增加而迅速下降。在处理短程序或标准过程时，它们能以低错误率按顺序执行指令，但对于复杂的程序，特别是包含关键路径和冗余指令的程序，模拟效果较差。我们提出了一种逐行模拟代码执行的方法来解决这个问题。 |
| [^34] | [Leveraging Public Representations for Private Transfer Learning.](http://arxiv.org/abs/2312.15551) | 该论文探讨了如何利用公共数据来改进私有学习的问题。研究发现，通过学习公共数据中的共享表示，可以在两种迁移学习场景中实现最优的学习效果。在单任务迁移场景中，算法在给定子空间范围内搜索线性模型，并实现了最优超额风险。在多任务个性化场景中，足够的公共数据可以消除私有协调需求，并通过纯局部学习达到相同的效用。 |
| [^35] | [Label Propagation for Graph Label Noise.](http://arxiv.org/abs/2310.16560) | 本文研究了图中的标签噪声问题，提出了一种基于标签传播的算法来处理任意异质性的图标签噪声，以纠正噪声标签并为未标记的节点分配标签。 |
| [^36] | [Audio Editing with Non-Rigid Text Prompts.](http://arxiv.org/abs/2310.12858) | 本文研究了使用非刚性文本编辑进行音频编辑的方法，并展示了其在保持输入音频一致性方面的优势。 |
| [^37] | [Penetrative AI: Making LLMs Comprehend the Physical World.](http://arxiv.org/abs/2310.09605) | 本文探讨了渗透式人工智能的概念，旨在使LLMs能够通过物联网传感器与执行器与物理世界进行交互和推理。初步研究结果表明，LLMs具有独特的能力，能够应用内嵌的世界知识解释物联网传感器数据并进行物理领域的推理。 |
| [^38] | [Feature Learning and Generalization in Deep Networks with Orthogonal Weights.](http://arxiv.org/abs/2310.07765) | 我们通过使用正交矩阵集合初始化权重并使用tanh激活函数，解决了全连接深度神经网络在初始化中具有与深度无关的线性波动问题。此外，我们发现神经切向核（NTK）及其后代的所有相关函数在逆宽度的主导阶段在深度约为20的位置饱和，而不是不断增长。 |
| [^39] | [A simple connection from loss flatness to compressed representations in neural networks.](http://arxiv.org/abs/2310.01770) | 该论文研究了深度神经网络中损失平坦性和神经表示压缩之间的关系，通过简单的数学关系，证明了损失平坦性与神经表示的压缩相关。 |
| [^40] | [PPG to ECG Signal Translation for Continuous Atrial Fibrillation Detection via Attention-based Deep State-Space Modeling.](http://arxiv.org/abs/2309.15375) | 通过基于注意力的深度状态空间建模，我们提出了一种不受个体限制的方法，将PPG信号转换为ECG，用于连续性心房颤动检测。 |
| [^41] | [Defending Against Alignment-Breaking Attacks via Robustly Aligned LLM.](http://arxiv.org/abs/2309.14348) | 本文提出了一种稳健对齐的LLM（RA-LLM），用于防御可能发生的对齐破坏攻击。RA-LLM可以直接在现有的对齐LLM上构建，并通过稳健的对齐检查函数来确保其有效性。 |
| [^42] | [Quantum-Noise-driven Generative Diffusion Models.](http://arxiv.org/abs/2308.12013) | 该论文提出了三种量子噪声驱动的生成扩散模型，利用了量子特性以克服传统模型的主要计算困难，并建议将量子噪声视为可利用的特性而非问题。 |
| [^43] | [Can Transformers Learn Optimal Filtering for Unknown Systems?.](http://arxiv.org/abs/2308.08536) | 本文研究了使用transformers进行最优输出估计问题，通过训练一个transformer来在未知系统上进行预测，并命名为元输出预测器（MOP）。我们观察到，尽管MOP没有访问模型的权限，但在大多数线性动态系统中，它的性能与基于卡尔曼滤波器的最优输出估计器相当，在具有非独立同分布噪声和时变动态的挑战性场景中也表现优秀。 |
| [^44] | [On (Normalised) Discounted Cumulative Gain as an Offline Evaluation Metric for Top-$n$ Recommendation.](http://arxiv.org/abs/2307.15053) | 本文批判性审视了(Normalised) Discounted Cumulative Gain作为Top-n推荐离线评估指标的方法，并研究了何时可以期望这些指标逼近在线实验的金标准结果。 |
| [^45] | [Suppressing unknown disturbances to dynamical systems using machine learning.](http://arxiv.org/abs/2307.03690) | 本文提出了一种使用机器学习的无模型方法，可以仅通过系统在已知强迫函数影响下的观测，识别和抑制未知系统的未知干扰。这项方法对训练函数有非常温和的限制，能够稳健地识别和抑制大类别的未知干扰。 |
| [^46] | [Informed POMDP: Leveraging Additional Information in Model-Based RL.](http://arxiv.org/abs/2306.11488) | 本文提出了Informed POMDP，这是一种新的学习范式，通过学习环境模型来利用训练时可用的额外信息，该模型可以提高Dreamer算法策略的收敛速度。 |
| [^47] | [Graph Laplacian Learning with Exponential Family Noise.](http://arxiv.org/abs/2306.08201) | 该论文将图信号处理框架推广到指数族噪声分布，提出了一种交替算法，用于从噪声信号中估计图拉普拉斯和未观测的平滑表示。实验结果表明，该算法可以在噪声模型不匹配情况下优于其他拉普拉斯估计方法。 |
| [^48] | [Adaptive Sparsity Level during Training for Efficient Time Series Forecasting with Transformers.](http://arxiv.org/abs/2305.18382) | 本文提出了“具有自适应稀疏度级别的修剪”(PALS), 通过稀疏训练和训练期间方法中的“扩张”机制，在Transformer模型中实现高效的时间序列预测。 |
| [^49] | [Contrastive Domain Generalization via Logit Attribution Matching.](http://arxiv.org/abs/2305.07888) | 本论文提出了一种名为对比领域泛化（CDG）的新方法，通过强烈对比的数据对所展示的语义不变性进行利用。同时，提出了一种正则化技术——Logit Attribution Matching (LAM)，以实现CDG。实验结果表明，LAM仅使用少量配对数据就能胜过最先进的DG方法，且有助于模型更好地关注对领域泛化至关重要的语义特征。 |
| [^50] | [LASER: Neuro-Symbolic Learning of Semantic Video Representations.](http://arxiv.org/abs/2304.07647) | LASER提出了一种神经符号学习方法来学习语义视频表示，通过逻辑规范捕捉视频数据中的时空属性，能够对齐原始视频和规范，有效地训练低级感知模型以提取符合所需高级规范的视频表示。 |
| [^51] | [A Unified Characterization of Private Learnability via Graph Theory.](http://arxiv.org/abs/2304.03996) | 本文提供了一个统一的框架，使用图论的语言刻画了差分隐私的两种情形下，纯粹和近似的学习性。我们通过定义矛盾图$G$来捕捉 $\mathcal{H}$ 的组合结构，发现分数团数和团数是描述差分隐私学习性的重要因素，并提出了几种算法对其进行估计。 |
| [^52] | [Manifold Learning by Mixture Models of VAEs for Inverse Problems.](http://arxiv.org/abs/2303.15244) | 本文提出了一种用混合VAE模型学习流形的方法，并将其用于解决逆问题，结果表现出良好的性能，可用于模糊和电阻抗层析成像。 |
| [^53] | [Unsupervised domain adaptation by learning using privileged information.](http://arxiv.org/abs/2303.09350) | 本文提出利用特权信息进行领域适应（DALUPI）算法，以在学习中放宽假设条件并提高样本效率，通过减少错误来促进医学图像分析等应用的发展。 |
| [^54] | [The Certification Paradox: Certifications Admit Better Attacks.](http://arxiv.org/abs/2302.04379) | 本文指出了一个"认证悖论"，认证虽然可以展示模型的稳健性，但额外揭示了有关认证模型的信息也成为新的攻击面，导致更好的攻击效果。 |
| [^55] | [Autobidders with Budget and ROI Constraints: Efficiency, Regret, and Pacing Dynamics.](http://arxiv.org/abs/2301.13306) | 本文提出了一个基于梯度的学习算法，可以在多种拍卖方式下满足预算和ROI约束，并达到个体后悔逐渐减小；结果表明，当各自竞争时，期望资金流动至少达到最优分配的期望流动的一半。 |
| [^56] | [Actor-Critic or Critic-Actor? A Tale of Two Time Scales.](http://arxiv.org/abs/2210.04470) | 这篇论文提出了一种评论演员算法，它在快速和慢速时间尺度上计算价值函数和策略，该算法与演员评论算法在准确性和计算成本方面表现相当。 |
| [^57] | [Regret Lower Bounds for Learning Linear Quadratic Gaussian Systems.](http://arxiv.org/abs/2201.01680) | 本论文证明了在学习未知线性高斯系统与二次代价时，存在遗憾下限，并且这个下限的比例尺度级别为 $\sqrt{T}$。通过对控制理论参数的准确捕捉，我们证明难以控制的系统也难以学习控制。同样地，对于一类部分观察到的系统，我们的结果表明了具有较差可观测结构的系统也难以学习控制。 |
| [^58] | [Hierarchical Correlation Clustering and Tree Preserving Embedding.](http://arxiv.org/abs/2002.07756) | 本文提出了一种分层相关聚类方法，可应用于正负配对不相似度，并研究了使用此方法进行无监督表征学习的方法。 |

# 详细

[^1]: 决策政策在混杂情况下的预测性能比较

    Predictive Performance Comparison of Decision Policies Under Confounding

    [https://arxiv.org/abs/2404.00848](https://arxiv.org/abs/2404.00848)

    提出了一种方法，通过现代识别方法比较决策政策的预测性能，关键在于可以安全忽略不确定性区域。

    

    预测模型通常被引入决策任务中，其基本理念是它们可以提升决策政策的性能。然而，与通常存在于未明确规定和依赖不可观测因素的现有决策政策相比较预测性能是具有挑战性的。这些不确定性来源通常在实践中被通过对数据生成机制进行强假设来处理。在这项研究中，我们提出了一种方法，来比较决策政策的预测性能，根据因果推断和离线评估文献中的各种现代识别方法进行评估（例如，工具变量，边际敏感性模型，近端变量）。我们的方法的关键是我们可以安全地忽略政策比较中的不确定性区域。我们开发了一种有限样本估计遗憾区间的实用方法。

    arXiv:2404.00848v1 Announce Type: new  Abstract: Predictive models are often introduced to decision-making tasks under the rationale that they improve performance over an existing decision-making policy. However, it is challenging to compare predictive performance against an existing decision-making policy that is generally under-specified and dependent on unobservable factors. These sources of uncertainty are often addressed in practice by making strong assumptions about the data-generating mechanism. In this work, we propose a method to compare the predictive performance of decision policies under a variety of modern identification approaches from the causal inference and off-policy evaluation literatures (e.g., instrumental variable, marginal sensitivity model, proximal variable). Key to our method is the insight that there are regions of uncertainty that we can safely ignore in the policy comparison. We develop a practical approach for finite-sample estimation of regret intervals u
    
[^2]: 对离群数据检测悖论的似然几何解释

    A Geometric Explanation of the Likelihood OOD Detection Paradox

    [https://arxiv.org/abs/2403.18910](https://arxiv.org/abs/2403.18910)

    高似然区域将不会被生成如果它们包含最小概率质量，基于此观察提出了一种通过本地固有维度估计进行离群检测的方法

    

    基于似然的深度生成模型(DGMs)通常表现出令人困惑的行为：当在相对复杂的数据集上训练时，它们会给来自更简单来源的离群数据赋予更高的似然值。更使人感到神秘的是，尽管具有更高的似然值，但这些DGMs从未生成过离群样本。这个双管齐下的悖论尚未得到最终解释，使得基于似然的离群检测不可靠。我们的主要观察是，如果高似然区域中包含了最小概率质量，那么这些区域将不会被生成。我们演示了在围绕低维流形数据的地方可能出现大密度但低概率质量的看似矛盾情况。我们还展示了通过本地固有维度(LID)估计可以识别这种场景，并提出了一种通过预训练的DGM获得的似然和LID估计相配对的离群检测方法。

    arXiv:2403.18910v1 Announce Type: cross  Abstract: Likelihood-based deep generative models (DGMs) commonly exhibit a puzzling behaviour: when trained on a relatively complex dataset, they assign higher likelihood values to out-of-distribution (OOD) data from simpler sources. Adding to the mystery, OOD samples are never generated by these DGMs despite having higher likelihoods. This two-pronged paradox has yet to be conclusively explained, making likelihood-based OOD detection unreliable. Our primary observation is that high-likelihood regions will not be generated if they contain minimal probability mass. We demonstrate how this seeming contradiction of large densities yet low probability mass can occur around data confined to low-dimensional manifolds. We also show that this scenario can be identified through local intrinsic dimension (LID) estimation, and propose a method for OOD detection which pairs the likelihoods and LID estimates obtained from a pre-trained DGM. Our method can b
    
[^3]: 可证实鲁棒的基于得分的扩散后验采样用于即插即用图像重建

    Provably Robust Score-Based Diffusion Posterior Sampling for Plug-and-Play Image Reconstruction

    [https://arxiv.org/abs/2403.17042](https://arxiv.org/abs/2403.17042)

    开发了一个算法框架，用于将基于得分的扩散模型作为通用非线性逆的表达数据先验。

    

    在科学和工程中的许多任务中，目标是从已知描述某种感知或成像模式的已知前向模型收集的少量测量中推断未知图像。由于资源限制，这个任务通常非常不适合，这就需要采纳表达丰富的先验信息来规范解空间。由于其令人印象深刻的经验成功，基于分数的扩散模型已经成为图像重建中一个具有吸引力的表达先验的候选者。为了一次性容纳多样的任务，开发将图像先验分布的无条件评分函数与灵活的前向模型选择相结合的高效、一致和鲁棒算法非常重要。这项工作开发了一个算法框架，用于将基于得分的扩散模型作为通用非线性逆的表达数据先验。

    arXiv:2403.17042v1 Announce Type: cross  Abstract: In a great number of tasks in science and engineering, the goal is to infer an unknown image from a small number of measurements collected from a known forward model describing certain sensing or imaging modality. Due to resource constraints, this task is often extremely ill-posed, which necessitates the adoption of expressive prior information to regularize the solution space. Score-based diffusion models, due to its impressive empirical success, have emerged as an appealing candidate of an expressive prior in image reconstruction. In order to accommodate diverse tasks at once, it is of great interest to develop efficient, consistent and robust algorithms that incorporate {\em unconditional} score functions of an image prior distribution in conjunction with flexible choices of forward models.   This work develops an algorithmic framework for employing score-based diffusion models as an expressive data prior in general nonlinear invers
    
[^4]: 自我改进用于神经组合优化问题：无需替换进行采样，但改进

    Self-Improvement for Neural Combinatorial Optimization: Sample without Replacement, but Improvement

    [https://arxiv.org/abs/2403.15180](https://arxiv.org/abs/2403.15180)

    通过结合循环式随机束搜索和来自可证策略改进的更新策略，本研究在神经组合优化中引入了一种新的训练方法，从而在最小采样次数中实现逐步改进的解决方案。

    

    目前，对于端到端的构造性神经组合优化方法通常是使用行为克隆来训练策略，从专家解决方案中或使用策略梯度从强化学习中进行训练。虽然行为克隆方法很直接，但需要昂贵的专家解决方案，而策略梯度方法往往计算要求很高，难以进行精细调整。在这项工作中，我们桥接了这两种方法，并通过在每个纪元中使用当前模型对随机实例进行多个解决方案的采样，然后选择最佳解作为专家轨迹进行监督模仿学习，从而简化了训练过程。为了在最小采样次数中实现逐步改进的解决方案，我们引入了一种将循环式随机束搜索与一种推导自可证策略改进的更新策略相结合的方法。该策略通过利用几乎没有通信开销的样本序列的优势，在轮之间调整策略，以精细化策略。

    arXiv:2403.15180v1 Announce Type: new  Abstract: Current methods for end-to-end constructive neural combinatorial optimization usually train a policy using behavior cloning from expert solutions or policy gradient methods from reinforcement learning. While behavior cloning is straightforward, it requires expensive expert solutions, and policy gradient methods are often computationally demanding and complex to fine-tune. In this work, we bridge the two and simplify the training process by sampling multiple solutions for random instances using the current model in each epoch and then selecting the best solution as an expert trajectory for supervised imitation learning. To achieve progressively improving solutions with minimal sampling, we introduce a method that combines round-wise Stochastic Beam Search with an update strategy derived from a provable policy improvement. This strategy refines the policy between rounds by utilizing the advantage of the sampled sequences with almost no com
    
[^5]: Larimar: 具有情节记忆控制的大型语言模型

    Larimar: Large Language Models with Episodic Memory Control

    [https://arxiv.org/abs/2403.11901](https://arxiv.org/abs/2403.11901)

    Larimar提出了一种大脑启发的架构，通过分布式情节记忆增强LLMs，实现了动态、一次性的知识更新，无需昂贵的重新训练或微调，且在速度和灵活性上表现出色。

    

    本文提出了Larimar - 一种新颖的、受大脑启发的架构，用于增强大型语言模型(LLMs)的分布式情节记忆。 Larimar的记忆允许动态、一次性更新知识，无需进行计算昂贵的重新训练或微调。在多个事实编辑基准测试上的实验结果表明，Larimar在速度方面表现优异 - 根据基础LLM的不同，速度提升为4-10倍，并且由于提出的架构简单、不依赖于LLM，因此具有良好的灵活性和通用性。我们进一步提供了选择性事实遗忘和输入上下文长度概括机制，并展示了它们的有效性。

    arXiv:2403.11901v1 Announce Type: cross  Abstract: Efficient and accurate updating of knowledge stored in Large Language Models (LLMs) is one of the most pressing research challenges today. This paper presents Larimar - a novel, brain-inspired architecture for enhancing LLMs with a distributed episodic memory. Larimar's memory allows for dynamic, one-shot updates of knowledge without the need for computationally expensive re-training or fine-tuning. Experimental results on multiple fact editing benchmarks demonstrate that Larimar attains accuracy comparable to most competitive baselines, even in the challenging sequential editing setup, but also excels in speed - yielding speed-ups of 4-10x depending on the base LLM - as well as flexibility due to the proposed architecture being simple, LLM-agnostic, and hence general. We further provide mechanisms for selective fact forgetting and input context length generalization with Larimar and show their effectiveness.
    
[^6]: 具有距离估计的声音事件检测和定位

    Sound Event Detection and Localization with Distance Estimation

    [https://arxiv.org/abs/2403.11827](https://arxiv.org/abs/2403.11827)

    本文将声音事件检测和定位任务扩展为具有距离估计的3D SELD，探讨了两种集成距离估计的方法，并在Ambisonic和双耳版本的声音场景下进行了实验。

    

    声音事件检测和定位(SELD)是识别声音事件及其对应到达方向(DOA)的综合任务。尽管这一任务在近年来已得到广泛研究并具有许多应用，但它未能提供有关声源位置的完整信息。本文通过将任务扩展为具有距离估计的声音事件检测、定位(3D SELD)来克服这一问题。我们研究了两种集成距离估计在SELD核心中的方法 - 一种是多任务方法，通过单独模型输出来处理问题，另一种是通过将多ACCDOA方法扩展以包括距离信息而获得的单任务方法。我们对Ambisonic和双耳版本的STARSS23：Sony-TAU Realistic Spatial Soundscapes 2023开展了研究。此外，我们的研究涉及与距离估计部分相关的损失函数的实验。

    arXiv:2403.11827v1 Announce Type: cross  Abstract: Sound Event Detection and Localization (SELD) is a combined task of identifying sound events and their corresponding direction-of-arrival (DOA). While this task has numerous applications and has been extensively researched in recent years, it fails to provide full information about the sound source position. In this paper, we overcome this problem by extending the task to Sound Event Detection, Localization with Distance Estimation (3D SELD). We study two ways of integrating distance estimation within the SELD core - a multi-task approach, in which the problem is tackled by a separate model output, and a single-task approach obtained by extending the multi-ACCDOA method to include distance information. We investigate both methods for the Ambisonic and binaural versions of STARSS23: Sony-TAU Realistic Spatial Soundscapes 2023. Moreover, our study involves experiments on the loss function related to the distance estimation part. Our resu
    
[^7]: 使用不确定性量化来表征和改进偏微分方程的区域外学习

    Using Uncertainty Quantification to Characterize and Improve Out-of-Domain Learning for PDEs

    [https://arxiv.org/abs/2403.10642](https://arxiv.org/abs/2403.10642)

    通过集成多个神经算子来提高对区域外学习的不确定性估计，从而解决现有方法在OOD测试输入上的失败

    

    存在于科学机器学习（SciML）领域中的现有工作表明，通过数据驱动学习解算符可以为经典数值偏微分方程（PDE）求解器提供一个快速的近似替代方案。在其中，神经算子（NOs）已经被认为尤为具有前景。我们观察到，对于区域外（OOD）测试输入，几种NOs的不确定性量化（UQ）方法甚至在模型对于域内任务的解近似良好时也会失败。为了解决这个限制，我们展示了集成几个NOs可以识别高误差区域，并提供良好与预测误差相关的不确定性估计。基于此，我们提出了一种经济有效的替代方案，DiverseNO，通过鼓励其最后前向传播层中的多个头部进行多样化预测来模拟集成的属性。然后，我们介绍了一种使用t

    arXiv:2403.10642v1 Announce Type: new  Abstract: Existing work in scientific machine learning (SciML) has shown that data-driven learning of solution operators can provide a fast approximate alternative to classical numerical partial differential equation (PDE) solvers. Of these, Neural Operators (NOs) have emerged as particularly promising. We observe that several uncertainty quantification (UQ) methods for NOs fail for test inputs that are even moderately out-of-domain (OOD), even when the model approximates the solution well for in-domain tasks. To address this limitation, we show that ensembling several NOs can identify high-error regions and provide good uncertainty estimates that are well-correlated with prediction errors. Based on this, we propose a cost-effective alternative, DiverseNO, that mimics the properties of the ensemble by encouraging diverse predictions from its multiple heads in the last feed-forward layer. We then introduce Operator-ProbConserv, a method that uses t
    
[^8]: LLM辅助下的交通信号控制：利用大型语言模型在复杂城市环境中实现人类仿生交通信号控制

    LLM-Assisted Light: Leveraging Large Language Model Capabilities for Human-Mimetic Traffic Signal Control in Complex Urban Environments

    [https://arxiv.org/abs/2403.08337](https://arxiv.org/abs/2403.08337)

    本研究提出了将大型语言模型(LLMs)整合到交通信号控制(TSC)系统中的创新方法，以解决传统TSC系统在适应不熟悉场景方面的限制，并提出了一个混合框架，使得LLMs与一系列感知和决策工具相结合，从而提升TSC系统对城市交通复杂性和变异性的管理能力。

    

    大都市地区的交通拥堵是一个具有深远经济、环境和社会影响的巨大挑战。因此，有效的拥堵管理至关重要，交通信号控制(TSC)系统在这方面起着至关重要的作用。为了回应传统TSC系统在管理城市交通流动的复杂性和变异性方面经常表现出的不足，本研究引入了一种创新方法，将大型语言模型(LLMs)整合到TSC中，利用其先进的推理和决策能力。具体来说，提出了一个混合框架，将LLMs与一套感知和决策工具相结合，有助于探讨静态和动态交通信息。

    arXiv:2403.08337v1 Announce Type: cross  Abstract: Traffic congestion in metropolitan areas presents a formidable challenge with far-reaching economic, environmental, and societal ramifications. Therefore, effective congestion management is imperative, with traffic signal control (TSC) systems being pivotal in this endeavor. Conventional TSC systems, designed upon rule-based algorithms or reinforcement learning (RL), frequently exhibit deficiencies in managing the complexities and variabilities of urban traffic flows, constrained by their limited capacity for adaptation to unfamiliar scenarios. In response to these limitations, this work introduces an innovative approach that integrates Large Language Models (LLMs) into TSC, harnessing their advanced reasoning and decision-making faculties. Specifically, a hybrid framework that augments LLMs with a suite of perception and decision-making tools is proposed, facilitating the interrogation of both the static and dynamic traffic informatio
    
[^9]: 在LLM推理中平衡吞吐量和延迟权衡的研究：Sarathi-Serve方法

    Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve

    [https://arxiv.org/abs/2403.02310](https://arxiv.org/abs/2403.02310)

    引入了一种有效的LLM推理调度程序Sarathi-Serve，通过分块预装填技术平衡了GPU计算饱和和单个标记处理的挑战，实现了高吞吐量和低延迟。

    

    每个LLM服务请求经历两个阶段。首先是prefill阶段，处理整个输入提示以生成一个输出标记；第二个是decode阶段，逐个生成其余的输出标记。Prefill迭代具有较高的延迟，但由于输入提示的并行处理，可以使GPU计算饱和。相比之下，decode迭代具有较低的延迟，但也仅使用较低的计算资源，因为每个请求只处理一个标记。这使得对解码来说批处理非常有效，因此对整体吞吐量也很有效。然而，批量处理多个请求会导致prefill和decode迭代交错进行，这使得在实现高吞吐量和低延迟之间的平衡变得具有挑战性。我们引入了一个高效的LLM推理调度程序Sarathi-Serve，灵感来自我们最初为优化Sarathi的吞吐量提出的技术。Sarathi-Serve利用了从Sarathi中引入的分块prefill技术。

    arXiv:2403.02310v1 Announce Type: new  Abstract: Each LLM serving request goes through two phases. The first is prefill which processes the entire input prompt to produce one output token and the second is decode which generates the rest of output tokens, one-at-a-time. Prefill iterations have high latency but saturate GPU compute due to parallel processing of the input prompt. In contrast, decode iterations have low latency but also low compute utilization because a decode iteration processes only a single token per request. This makes batching highly effective for decodes and consequently for overall throughput. However, batching multiple requests leads to an interleaving of prefill and decode iterations which makes it challenging to achieve both high throughput and low latency.   We introduce an efficient LLM inference scheduler Sarathi-Serve inspired by the techniques we originally proposed for optimizing throughput in Sarathi. Sarathi-Serve leverages chunked-prefills from Sarathi 
    
[^10]: 差分隐私的平移插值

    Shifted Interpolation for Differential Privacy

    [https://arxiv.org/abs/2403.00278](https://arxiv.org/abs/2403.00278)

    本文在统一框架下建立了“通过迭代实现隐私放大”现象，提高了先前分析的水平，并由此获得了其他差分隐私概念更紧密的隐私核算。

    

    喧嚣的梯度下降及其变种是差分隐私机器学习中主导的算法。量化它们的隐私泄漏是一个基本问题，然而即使在凸损失的基础设置中，紧致的表征仍然是开放的。本文通过在$f$-差分隐私的统一框架下建立（和改进）“通过迭代实现隐私放大”现象，提高了先前分析的水平--这种方法紧紧捕捉了隐私损失的所有方面，并立即获得了其他差分隐私概念（如$(\varepsilon,\delta)$-DP和Renyi DP）更紧密的隐私核算。我们的关键技术见解是构建了揭示了流行的平移散度论证的平移插值过程，使得超越基于散度的差分隐私放宽的泛化成为可能。值得注意的是，这导致了在强凸基础设置中的第一个精确隐私分析。

    arXiv:2403.00278v1 Announce Type: new  Abstract: Noisy gradient descent and its variants are the predominant algorithms for differentially private machine learning. It is a fundamental question to quantify their privacy leakage, yet tight characterizations remain open even in the foundational setting of convex losses. This paper improves over previous analyses by establishing (and refining) the "privacy amplification by iteration" phenomenon in the unifying framework of $f$-differential privacy--which tightly captures all aspects of the privacy loss and immediately implies tighter privacy accounting in other notions of differential privacy, e.g., $(\varepsilon,\delta)$-DP and Renyi DP. Our key technical insight is the construction of shifted interpolated processes that unravel the popular shifted-divergences argument, enabling generalizations beyond divergence-based relaxations of DP. Notably, this leads to the first exact privacy analysis in the foundational setting of strongly convex
    
[^11]: 批处理和匹配：基于分数的离散的黑匣子变分推断

    Batch and match: black-box variational inference with a score-based divergence

    [https://arxiv.org/abs/2402.14758](https://arxiv.org/abs/2402.14758)

    BaM是一种基于分数的离散的BBVI替代方法，针对高方差梯度估计慢收敛问题，能够在高斯变分族中通过封闭形式的近端更新进行优化，在目标分布为高斯时，批处理大小趋于无穷时变分参数更新将指数快速收敛到目标均值和协方差，BaM在多种生成模型推断中表现出良好性能

    

    大多数主要的黑匣子变分推断（BBVI）实现都是基于优化随机证据下界（ELBO）。但是，这种BBVI方法通常由于其梯度估计的高方差而收敛缓慢。在本文中，我们提出了批处理和匹配（BaM），这是一种基于分数的离散的BBVI替代方法。值得注意的是，这种基于分数的离散可以通过对具有全协方差矩阵的高斯变分族使用封闭形式的近端更新进行优化。我们分析了当目标分布为高斯分布时BaM的收敛性，并证明在批量大小趋于无穷时变分参数更新会指数收敛到目标均值和协方差。我们还评估了BaM在源自层次和深度生成模型后验推断的高斯和非高斯目标分布上的性能。在这些实验中，我们发现BaM在...

    arXiv:2402.14758v1 Announce Type: cross  Abstract: Most leading implementations of black-box variational inference (BBVI) are based on optimizing a stochastic evidence lower bound (ELBO). But such approaches to BBVI often converge slowly due to the high variance of their gradient estimates. In this work, we propose batch and match (BaM), an alternative approach to BBVI based on a score-based divergence. Notably, this score-based divergence can be optimized by a closed-form proximal update for Gaussian variational families with full covariance matrices. We analyze the convergence of BaM when the target distribution is Gaussian, and we prove that in the limit of infinite batch size the variational parameter updates converge exponentially quickly to the target mean and covariance. We also evaluate the performance of BaM on Gaussian and non-Gaussian target distributions that arise from posterior inference in hierarchical and deep generative models. In these experiments, we find that BaM ty
    
[^12]: 面向目标数据子集选择的子模信息量的理论分析

    Theoretical Analysis of Submodular Information Measures for Targeted Data Subset Selection

    [https://arxiv.org/abs/2402.13454](https://arxiv.org/abs/2402.13454)

    通过推导与相关性和覆盖性相关的基于相似度的界限，为子模互信息在目标数据子集选择中的表现提供了理论保证

    

    随着在机器学习任务中使用的数据量增加，定位特定数据子集的能力变得越来越重要。为了帮助实现这一能力，最近提出的子模互信息（SMI）已经在文献中有效应用于执行使用示例查询集进行定位子集选择的多个任务。然而，所有这些工作都没有在理论上保证SMI对于子集相关性和目标数据的覆盖性的敏感性。我们首次通过推导与相关性和目标数据覆盖相关的基于相似度的界限，提供了此类保证。通过这些界限，我们展示了在多个应用中已经表现成功的SMI函数在理论上确保实现良好的查询相关性和查询覆盖。

    arXiv:2402.13454v1 Announce Type: new  Abstract: With increasing volume of data being used across machine learning tasks, the capability to target specific subsets of data becomes more important. To aid in this capability, the recently proposed Submodular Mutual Information (SMI) has been effectively applied across numerous tasks in literature to perform targeted subset selection with the aid of a exemplar query set. However, all such works are deficient in providing theoretical guarantees for SMI in terms of its sensitivity to a subset's relevance and coverage of the targeted data. For the first time, we provide such guarantees by deriving similarity-based bounds on quantities related to relevance and coverage of the targeted data. With these bounds, we show that the SMI functions, which have empirically shown success in multiple applications, are theoretically sound in achieving good query relevance and query coverage.
    
[^13]: CounterCurate: 通过对照例子增强物理和语义视觉-语言组合推理能力

    CounterCurate: Enhancing Physical and Semantic Visio-Linguistic Compositional Reasoning via Counterfactual Examples

    [https://arxiv.org/abs/2402.13254](https://arxiv.org/abs/2402.13254)

    本研究提出CounterCurate框架，通过对比例子和生成式微调，全面提升视觉-语言组合推理能力，解决了物理推理和语义对照微调方面的关键问题，实现了显著性能改进。

    

    我们提出CounterCurate，一个框架，全面提升对比和生成式多模态模型的视觉-语言组合推理能力。特别地，我们确定了两个尚未充分探讨的关键问题：忽视了基于物理的推理（计数和位置理解），以及利用高性能文本和图像生成模型进行语义反事实微调的潜力。我们的工作开创了一个解决这些空白的方法。我们首先突出了多模态模型（如CLIP和LLaVA）在基于物理的组合推理中几乎无法胜任的表现。然后，我们应用简单的数据增强，使用基于图像的生成模型GLIGEN生成微调数据，使得性能显著提高：在我们新的策划的Flickr30k-Positions基准测试中，CLIP和LLaVA的性能分别提高了+33%和+37%。此外，我们利用了高性能文本和图像生成模型的能力。

    arXiv:2402.13254v1 Announce Type: cross  Abstract: We propose CounterCurate, a framework to comprehensively improve the visio-linguistic compositional reasoning capability for both contrastive and generative multimodal models. In particular, we identify two under-explored critical problems: the neglect of the physically grounded reasoning (counting and position understanding) and the potential of using highly capable text and image generation models for semantic counterfactual fine-tuning. Our work pioneers an approach that addresses these gaps. We first spotlight the near-chance performance of multimodal models like CLIP and LLaVA in physically grounded compositional reasoning. We then apply simple data augmentation using a grounded image generation model, GLIGEN, to generate finetuning data, resulting in significant performance improvements: +33% and +37% for CLIP and LLaVA, respectively, on our newly curated Flickr30k-Positions benchmark. Moreover, we exploit the capabilities of hig
    
[^14]: 针对多维时间序列预测的随机投影层

    Random Projection Layers for Multidimensional Time Sires Forecasting

    [https://arxiv.org/abs/2402.10487](https://arxiv.org/abs/2402.10487)

    提出了一种全MLP时间序列预测架构RPMixer，通过将随机投影层集成到模型中，增加了块输出之间的多样性，提高了整体性能

    

    多层感知器（MLP）混合模型已被证明对时间序列预测问题有效。然而，当将此类模型应用于高维时间序列（例如空间-时间数据集中的时间序列）时，由于过拟合问题，其性能可能会下降。本文提出了一种全MLP时间序列预测架构，称为RPMixer。我们的方法利用了深度神经网络的集成式行为，其中网络中的每个单独块的作用类似于集成模型中的基本学习器，特别是在引入身份映射残差连接时。通过将随机投影层集成到我们的模型中，我们增加了块输出之间的多样性，从而提高了RPMixer的整体性能。对大规模空间-时间预测基准数据集进行的大量实验表明，我们提出的方法胜过了

    arXiv:2402.10487v1 Announce Type: cross  Abstract: All-Multi-Layer Perceptron (all-MLP) mixer models have been shown to be effective for time series forecasting problems. However, when such a model is applied to high-dimensional time series (e.g., the time series in a spatial-temporal dataset), its performance is likely to degrade due to overfitting issues. In this paper, we propose an all-MLP time series forecasting architecture, referred to as RPMixer. Our method leverages the ensemble-like behavior of deep neural networks, where each individual block within the network acts like a base learner in an ensemble model, especially when identity mapping residual connections are incorporated. By integrating random projection layers into our model, we increase the diversity among the blocks' outputs, thereby enhancing the overall performance of RPMixer. Extensive experiments conducted on large-scale spatial-temporal forecasting benchmark datasets demonstrate that our proposed method outperf
    
[^15]: 分类扩散模型

    Classification Diffusion Models

    [https://arxiv.org/abs/2402.10095](https://arxiv.org/abs/2402.10095)

    提出了一种分类扩散模型（CDMs），该模型采用了去噪扩散模型（DDM）的形式，并利用一个分类器来预测加在干净信号上的噪声量，取得了在图像、视频和音频生成方面的最先进结果。

    

    arXiv：2402.10095v1 公告类型：新的 摘要：一种学习数据分布的突出方法家族依赖于密度比估计（DRE），其中模型被训练来$\textit{分类}$数据样本和来自某个参考分布的样本。这些技术在简单的低维环境中取得了成功，但在复杂的高维数据（如图像）中无法取得良好的结果。学习分布的另一种方法家族是去噪扩散模型（DDM），其中模型被训练来$\textit{去噪}$数据样本。这些方法在图像、视频和音频生成方面取得了最先进的结果。在这项工作中，我们提出了$\textit{分类扩散模型}$（CDMs），这是一种生成技术，它采用了DDM的去噪基本形式，同时利用一个分类器来预测加在干净信号上的噪声量，类似于DRE方法。我们的方法基于这样一个观察，即MSE最优化的d

    arXiv:2402.10095v1 Announce Type: new  Abstract: A prominent family of methods for learning data distributions relies on density ratio estimation (DRE), where a model is trained to $\textit{classify}$ between data samples and samples from some reference distribution. These techniques are successful in simple low-dimensional settings but fail to achieve good results on complex high-dimensional data, like images. A different family of methods for learning distributions is that of denoising diffusion models (DDMs), in which a model is trained to $\textit{denoise}$ data samples. These approaches achieve state-of-the-art results in image, video, and audio generation. In this work, we present $\textit{Classification Diffusion Models}$ (CDMs), a generative technique that adopts the denoising-based formalism of DDMs while making use of a classifier that predicts the amount of noise added to a clean signal, similarly to DRE methods. Our approach is based on the observation that an MSE-optimal d
    
[^16]: 预测性线性在线追踪未知目标

    Predictive Linear Online Tracking for Unknown Targets

    [https://arxiv.org/abs/2402.10036](https://arxiv.org/abs/2402.10036)

    本文提出了一种名为预测性线性在线追踪（PLOT）的算法，用于在线追踪未知目标。该算法使用具有指数遗忘的递归最小二乘法来学习目标的时变动态模型，并在递推视线控制的框架下使用所学模型进行最优策略。与先前的工作不同，我们的理论结果适用于非平稳目标。

    

    本文研究了在线线性控制系统中的追踪问题，目标是跟随一个移动的目标。与经典的追踪控制不同，目标是未知的、非平稳的，并且它的状态逐步揭示，因此适合在线非随机控制的框架。我们考虑了二次成本的情况，并提出了一种新算法，称为预测性线性在线追踪（PLOT）。该算法使用具有指数遗忘的递归最小二乘法来学习目标的时变动态模型。所学模型在递推视线控制的框架下用于优化策略。我们证明了PLOT的动态遗憾与$\mathcal{O}(\sqrt{TV_T})$成比例，其中$V_T$是目标动力学的总变化量，$T$是时间长度。与先前的工作不同，我们的理论结果适用于非平稳目标。我们在一个真实的四旋翼机上实现了PLOT，并提供了开源代码。

    arXiv:2402.10036v1 Announce Type: cross  Abstract: In this paper, we study the problem of online tracking in linear control systems, where the objective is to follow a moving target. Unlike classical tracking control, the target is unknown, non-stationary, and its state is revealed sequentially, thus, fitting the framework of online non-stochastic control. We consider the case of quadratic costs and propose a new algorithm, called predictive linear online tracking (PLOT). The algorithm uses recursive least squares with exponential forgetting to learn a time-varying dynamic model of the target. The learned model is used in the optimal policy under the framework of receding horizon control. We show the dynamic regret of PLOT scales with $\mathcal{O}(\sqrt{TV_T})$, where $V_T$ is the total variation of the target dynamics and $T$ is the time horizon. Unlike prior work, our theoretical results hold for non-stationary targets. We implement PLOT on a real quadrotor and provide open-source so
    
[^17]: 使用KV缓存压缩合成循环以提高LLM推断的效率

    Get More with LESS: Synthesizing Recurrence with KV Cache Compression for Efficient LLM Inference

    [https://arxiv.org/abs/2402.09398](https://arxiv.org/abs/2402.09398)

    这项研究提出了一个称为LESS的方法，通过集成一个固定尺寸的缓存和基于驱逐的缓存方法，可以在大型语言模型中减小内存占用的问题，同时保持全部标记的可查询能力，并在多种任务上显示出良好的性能。

    

    许多计算因素限制了大型语言模型的广泛部署。本文关注于由键值(KV)缓存引起的内存瓶颈，这是一种计算快捷方式，在解码过程中需要存储先前的KV对。现有的KV缓存方法通过修剪或驱逐相对不重要的KV对的大片区域，显著减少缓存的内存占用，但在需要重新收集大多数前一个标记的任务中，它们的成功有限。为了缓解这个问题，我们提出了LESS，它将一个（几乎免费的）固定尺寸的缓存与基于驱逐的缓存方法简单地集成在一起，以便所有的标记可以在后续的解码步骤中查询。它能够在时间上保留信息，在多种任务上展现出合理性，我们展示了LESS可以帮助减小缓存所有内容的性能差距，有时甚至可以与其相匹配，同时具有高效性。

    arXiv:2402.09398v1 Announce Type: cross Abstract: Many computational factors limit broader deployment of large language models. In this paper, we focus on a memory bottleneck imposed by the key-value (KV) cache, a computational shortcut that requires storing previous KV pairs during decoding. While existing KV cache methods approach this problem by pruning or evicting large swaths of relatively less important KV pairs to dramatically reduce the memory footprint of the cache, they can have limited success in tasks that require recollecting a majority of previous tokens. To alleviate this issue, we propose LESS, a simple integration of a (nearly free) constant sized cache with eviction-based cache methods, such that all tokens can be queried at later decoding steps. Its ability to retain information throughout time shows merit on a variety of tasks where we demonstrate LESS can help reduce the performance gap from caching everything, sometimes even matching it, all while being efficient.
    
[^18]: Chinese MentalBERT: 在社交媒体上针对中国心理健康文本分析的领域自适应预训练

    Chinese MentalBERT: Domain-Adaptive Pre-training on Social Media for Chinese Mental Health Text Analysis

    [https://arxiv.org/abs/2402.09151](https://arxiv.org/abs/2402.09151)

    本文介绍了一种领域自适应预训练模型Chinese MentalBERT，该模型针对中国社交媒体上心理健康文本分析进行了优化，在预训练过程中加入心理学词典，提高了模型的适用性。

    

    受到社交媒体的影响，心理问题在当前环境中普遍存在，并且社交媒体成为个人分享感受的重要出口。这导致每天产生大量数据，其中负面情绪有潜力引发危机。因此需要开发出能够高效分析这些数据的模型。虽然预训练语言模型广泛显示出效果，但针对心理学等特定领域的预训练模型存在明显缺失。为解决这一问题，我们从中国社交媒体平台收集了大量数据，并丰富了公开可用数据集，创建了一个包含336万条文本条目的综合数据库。为提高模型在心理文本分析上的适用性，我们将心理学词典融入预训练的掩码机制。在现有的中文语言模型基础上进行构建。

    arXiv:2402.09151v1 Announce Type: new Abstract: In the current environment, psychological issues are prevalent and widespread, with social media serving as a key outlet for individuals to share their feelings. This results in the generation of vast quantities of data daily, where negative emotions have the potential to precipitate crisis situations. There is a recognized need for models capable of efficient analysis. While pre-trained language models have demonstrated their effectiveness broadly, there's a noticeable gap in pre-trained models tailored for specialized domains like psychology. To address this, we have collected a huge dataset from Chinese social media platforms and enriched it with publicly available datasets to create a comprehensive database encompassing 3.36 million text entries. To enhance the model's applicability to psychological text analysis, we integrated psychological lexicons into the pre-training masking mechanism. Building on an existing Chinese language mod
    
[^19]: SLEB: 通过冗余验证和消除Transformer块优化LLM的流程

    SLEB: Streamlining LLMs through Redundancy Verification and Elimination of Transformer Blocks

    [https://arxiv.org/abs/2402.09025](https://arxiv.org/abs/2402.09025)

    SLEB是一种通过消除冗余的Transformer块来优化LLM流程的新方法，它成功加速了LLM的推理过程。

    

    大型语言模型（LLM）在各种自然语言处理任务中证明了其高效性。然而，它们庞大的参数数量给实际部署带来了重大挑战。精简，一种旨在减小LLM大小和复杂度的技术，通过从网络中删除冗余组件提供了潜在解决方案。尽管精简有希望，但现有方法往往难以实现显著的端到端LLM推理加速。本文中，我们引入了SLEB，一种通过消除冗余的Transformer块来优化LLM流程的新方法。我们选择Transformer块作为精简的基本单位，因为LLM在相邻块的输出之间具有块级别的冗余和高相似性。这个选择使我们能够有效地增强LLM的处理速度。我们的实验证明，SLEB成功加速了LLM的推理过程。

    arXiv:2402.09025v1 Announce Type: new Abstract: Large language models (LLMs) have proven to be highly effective across various natural language processing tasks. However, their large number of parameters poses significant challenges for practical deployment. Pruning, a technique aimed at reducing the size and complexity of LLMs, offers a potential solution by removing redundant components from the network. Despite the promise of pruning, existing methods often struggle to achieve substantial end-to-end LLM inference speedup. In this paper, we introduce SLEB, a novel approach designed to streamline LLMs by eliminating redundant transformer blocks. We choose the transformer block as the fundamental unit for pruning, because LLMs exhibit block-level redundancy with high similarity between the outputs of neighboring blocks. This choice allows us to effectively enhance the processing speed of LLMs. Our experimental results demonstrate that SLEB successfully accelerates LLM inference without
    
[^20]: 理解深度学习的实际成员隐私

    Understanding Practical Membership Privacy of Deep Learning

    [https://arxiv.org/abs/2402.06674](https://arxiv.org/abs/2402.06674)

    该论文利用最先进的成员推理攻击方法系统地测试了细调大型图像分类模型的实际隐私漏洞，并发现数据集中每个类别的示例数量以及训练结束时的大梯度与成员推理攻击的漏洞之间存在关联。

    

    我们应用最先进的成员推理攻击（MIA）来系统地测试细调大型图像分类模型的实际隐私漏洞。我们的重点是理解使数据集和样本容易受到成员推理攻击的特性。在数据集特性方面，我们发现数据中每个类别的示例数量与成员推理攻击的漏洞之间存在强烈的幂律依赖关系，这是以攻击的真阳性率（在低假阳性率下测量）来衡量的。对于个别样本而言，在训练结束时产生的大梯度与成员推理攻击的漏洞之间存在很强的相关性。

    We apply a state-of-the-art membership inference attack (MIA) to systematically test the practical privacy vulnerability of fine-tuning large image classification models.We focus on understanding the properties of data sets and samples that make them vulnerable to membership inference. In terms of data set properties, we find a strong power law dependence between the number of examples per class in the data and the MIA vulnerability, as measured by true positive rate of the attack at a low false positive rate. For an individual sample, large gradients at the end of training are strongly correlated with MIA vulnerability.
    
[^21]: 浅层ReLU-like神经网络的损失景观：稳定点、鞍点逃逸和网络嵌入

    The Loss Landscape of Shallow ReLU-like Neural Networks: Stationary Points, Saddle Escaping, and Network Embedding

    [https://arxiv.org/abs/2402.05626](https://arxiv.org/abs/2402.05626)

    本文研究了使用ReLU-like激活函数以经验平方损失训练的单隐藏层神经网络的损失景观，提出了稳定点条件和逃逸神经元的定义，并将鞍点逃逸与逃逸神经元的参数变化联系起来。

    

    本文研究了使用ReLU-like激活函数以经验平方损失训练的单隐藏层神经网络的损失景观。由于激活函数是不可微的，目前还不清楚如何完全描述稳定点。我们提出了适用于非可微和可微情况的稳定点条件。此外，我们还展示了如果一个稳定点不包含“逃逸神经元”（通过一阶条件定义），那么它必定是一个局部最小值。此外，在标量输出情况下，逃逸神经元的存在保证了稳定点不是局部最小值。我们的结果进一步描述了从无穷小（消失）初始化开始的浅层ReLU-like网络的鞍点到鞍点的训练过程，直接将鞍点逃逸与逃逸神经元的参数变化联系起来。此外，我们还完全讨论了网络嵌入的方式。

    In this paper, we investigate the loss landscape of one-hidden-layer neural networks with ReLU-like activation functions trained with the empirical squared loss. As the activation function is non-differentiable, it is so far unclear how to completely characterize the stationary points. We propose the conditions for stationarity that apply to both non-differentiable and differentiable cases. Additionally, we show that, if a stationary point does not contain "escape neurons", which are defined with first-order conditions, then it must be a local minimum. Moreover, for the scalar-output case, the presence of an escape neuron guarantees that the stationary point is not a local minimum. Our results refine the description of the saddle-to-saddle training process starting from infinitesimally small (vanishing) initialization for shallow ReLU-like networks, linking saddle escaping directly with the parameter changes of escape neurons. Moreover, we are also able to fully discuss how network emb
    
[^22]: 预测个体治疗效果的一致性蒙特卡洛元学习模型

    Conformal Monte Carlo Meta-learners for Predictive Inference of Individual Treatment Effects

    [https://arxiv.org/abs/2402.04906](https://arxiv.org/abs/2402.04906)

    本研究提出了一种新方法，即一致性蒙特卡洛元学习模型，用于预测个体治疗效果。通过利用一致性预测系统、蒙特卡洛采样和CATE元学习模型，该方法生成可用于个性化决策的预测分布。实验结果显示，该方法在保持较小区间宽度的情况下具有强大的实验覆盖范围，可以提供真实个体治疗效果的估计。

    

    认识干预效果，即治疗效果，对于决策至关重要。用条件平均治疗效果 (CATE) 估计等方法通常只提供治疗效果的点估计，而常常需要额外的不确定性量化。因此，我们提出了一个新方法，即一致性蒙特卡洛 (CMC) 元学习模型，利用一致性预测系统、蒙特卡洛采样和 CATE 元学习模型，来产生可用于个性化决策的预测分布。此外，我们展示了结果噪声分布的特定假设如何严重影响这些不确定性预测。尽管如此，CMC框架展示了强大的实验覆盖范围，同时保持较小的区间宽度，以提供真实个体治疗效果的估计。

    Knowledge of the effect of interventions, called the treatment effect, is paramount for decision-making. Approaches to estimating this treatment effect, e.g. by using Conditional Average Treatment Effect (CATE) estimators, often only provide a point estimate of this treatment effect, while additional uncertainty quantification is frequently desired instead. Therefore, we present a novel method, the Conformal Monte Carlo (CMC) meta-learners, leveraging conformal predictive systems, Monte Carlo sampling, and CATE meta-learners, to instead produce a predictive distribution usable in individualized decision-making. Furthermore, we show how specific assumptions on the noise distribution of the outcome heavily affect these uncertainty predictions. Nonetheless, the CMC framework shows strong experimental coverage while retaining small interval widths to provide estimates of the true individual treatment effect.
    
[^23]: 基于信息的增强学习用于情境感知交通规则例外

    Informed Reinforcement Learning for Situation-Aware Traffic Rule Exceptions

    [https://arxiv.org/abs/2402.04168](https://arxiv.org/abs/2402.04168)

    本论文引入了基于信息的增强学习方法，将结构化的规则手册作为知识源，利用情境感知的奖励设计评估学习轨迹，以实现对需要控制交通规则例外的情景的学习。

    

    增强学习是一个非常活跃的研究领域，具有很多有前景的进展。然而，在自动驾驶领域，通常只研究非常简单的场景。常见的方法使用非可解释的控制命令作为动作空间，以及缺乏结构的奖励设计。在本文中，我们引入了基于信息的增强学习，将结构化的规则手册作为知识源集成进来。我们学习轨迹，并使用情境感知的奖励设计来评估它们，从而产生动态奖励，使代理能够学习需要控制交通规则例外的情况。我们的方法适用于任意增强学习模型。我们成功地展示了最近的基于模型的代理在复杂场景中的高完成率。

    Reinforcement Learning is a highly active research field with promising advancements. In the field of autonomous driving, however, often very simple scenarios are being examined. Common approaches use non-interpretable control commands as the action space and unstructured reward designs which lack structure. In this work, we introduce Informed Reinforcement Learning, where a structured rulebook is integrated as a knowledge source. We learn trajectories and asses them with a situation-aware reward design, leading to a dynamic reward which allows the agent to learn situations which require controlled traffic rule exceptions. Our method is applicable to arbitrary RL models. We successfully demonstrate high completion rates of complex scenarios with recent model-based agents.
    
[^24]: 則采樣并不是魔法: 大批量大小為什麼適用於差分隱私隨機優化

    Subsampling is not Magic: Why Large Batch Sizes Work for Differentially Private Stochastic Optimisation

    [https://arxiv.org/abs/2402.03990](https://arxiv.org/abs/2402.03990)

    通过研究差分隐私随机梯度下降（DP-SGD）中的总梯度方差，我们发现大批次大小有助于减小則采樣引起的方差，从而提高优化效果。

    

    我們研究了批次大小對差分隱私隨機梯度下降（DP-SGD）中總梯度方差的影響，尋求對大批次大小有用性的理論解釋。由於DP-SGD是現代差分隱私深度學習的基礎，其性質已被廣泛研究，最近的工作在實踐中發現大批次大小有益。然而，對於這種好處的理論解釋目前最多只能說是啟發式的。我們首先觀察到，在DP-SGD中，總梯度方差可以分解為由則采樣和噪聲引起的方差。然後，我們證明在無限次迭代的極限情況下，有效的噪聲引起的方差對批次大小是不變的。剩下的則采樣引起的方差隨著批次大小的增大而減小，因此大批次大小減小了有效的總梯度方差。我們在數值上確認這種漸進的情況在實際環境中是相關的，當批次大小不小的時候會起作用，並且發現

    We study the effect of the batch size to the total gradient variance in differentially private stochastic gradient descent (DP-SGD), seeking a theoretical explanation for the usefulness of large batch sizes. As DP-SGD is the basis of modern DP deep learning, its properties have been widely studied, and recent works have empirically found large batch sizes to be beneficial. However, theoretical explanations of this benefit are currently heuristic at best. We first observe that the total gradient variance in DP-SGD can be decomposed into subsampling-induced and noise-induced variances. We then prove that in the limit of an infinite number of iterations, the effective noise-induced variance is invariant to the batch size. The remaining subsampling-induced variance decreases with larger batch sizes, so large batches reduce the effective total gradient variance. We confirm numerically that the asymptotic regime is relevant in practical settings when the batch size is not small, and find tha
    
[^25]: LLMs无法规划，但可以在LLM-Modulo框架中帮助规划

    LLMs Can't Plan, But Can Help Planning in LLM-Modulo Frameworks

    [https://arxiv.org/abs/2402.01817](https://arxiv.org/abs/2402.01817)

    LLMs无法独自进行规划或自我验证，但在规划/推理任务中可以作为通用近似知识源发挥更大作用。

    

    关于大型语言模型（LLMs）在规划和推理任务中的角色存在很大的困惑。一方面有人过于乐观地声称只需正确提示或自我验证策略，LLMs就能完成这些任务。另一方面，也有人过于悲观地认为LLMs在规划/推理任务中仅能作为问题规范的简单翻译器，并将问题交给外部符号求解器。在这篇立场文章中，我们认为这两种极端观点都是错误的。我们认为自回归LLMs本身不能进行规划或自我验证（毕竟这是一种推理形式），并对文献中的误解原因进行了一些阐述。我们还将辩称LLMs应该被视为具有更有意义的角色的通用近似知识源，能在规划/推理任务中发挥更大的作用。

    There is considerable confusion about the role of Large Language Models (LLMs) in planning and reasoning tasks. On one side are over-optimistic claims that LLMs can indeed do these tasks with just the right prompting or self-verification strategies. On the other side are perhaps over-pessimistic claims that all that LLMs are good for in planning/reasoning tasks are as mere translators of the problem specification from one syntactic format to another, and ship the problem off to external symbolic solvers. In this position paper, we take the view that both these extremes are misguided. We argue that auto-regressive LLMs cannot, by themselves, do planning or self-verification (which is after all a form of reasoning), and shed some light on the reasons for misunderstandings in the literature. We will also argue that LLMs should be viewed as universal approximate knowledge sources that have much more meaningful roles to play in planning/reasoning tasks beyond simple front-end/back-end forma
    
[^26]: 多组学习的层次组模型

    Multi-group Learning for Hierarchical Groups

    [https://arxiv.org/abs/2402.00258](https://arxiv.org/abs/2402.00258)

    本研究将多组学习扩展到具有层次结构的情况，设计了一个近乎最优的样本复杂度的算法，输出可解释且确定性的决策树预测器，并在真实数据集上取得了有吸引力的广义化特性。

    

    多组学习模型将学习场景规范化为单一预测器在多个可能重叠的兴趣子组上必须广义化。我们将多组学习的研究扩展到了具有层次结构的自然情况。我们设计了一个算法，用于输出可解释且确定性的决策树预测器，具有近乎最优的样本复杂度。然后，我们对该算法进行经验评估，并发现它在具有层次组结构的真实数据集上具有有吸引力的广义化特性。

    The multi-group learning model formalizes the learning scenario in which a single predictor must generalize well on multiple, possibly overlapping subgroups of interest. We extend the study of multi-group learning to the natural case where the groups are hierarchically structured. We design an algorithm for this setting that outputs an interpretable and deterministic decision tree predictor with near-optimal sample complexity. We then conduct an empirical evaluation of our algorithm and find that it achieves attractive generalization properties on real datasets with hierarchical group structure.
    
[^27]: 科学与深度学习中的可靠性和解释性

    Reliability and Interpretability in Science and Deep Learning

    [https://arxiv.org/abs/2401.07359](https://arxiv.org/abs/2401.07359)

    这篇论文强调了科学与深度学习中模型假设的重要性，并提供了对模型假设认识论复杂性的分析，同时结合标准错误分析与深度神经网络模型的特点，来评估模型可靠性。

    

    近年来，机器学习（ML）方法的可靠性问题日益重要，并且与此相关的不确定性分析已经激发了大量的研究。然而，大部分研究都仅将标准错误分析应用于深度神经网络（DNN）模型，这在很大程度上与标准科学建模有所不同。因此，有必要将标准错误分析与对DNN模型与标准科学建模的可能差异以及这些差异在可靠性评估中可能产生的影响的更深层次的认识论分析相结合。本文提供了几个贡献。首先，强调了模型假设（在ML和传统科学中均存在）在无理论科学的错觉下的普遍作用。其次，从（认识论的）复杂性角度分析了模型假设，同时还展示了模型假设在可靠性评估中的作用。

    In recent years, the question of the reliability of Machine Learning (ML) methods has acquired significant importance, and the analysis of the associated uncertainties has motivated a growing amount of research. However, most of these studies have applied standard error analysis to ML models, and in particular Deep Neural Network (DNN) models, which represent a rather significant departure from standard scientific modelling. It is therefore necessary to integrate the standard error analysis with a deeper epistemological analysis of the possible differences between DNN models and standard scientific modelling and the possible implications of these differences in the assessment of reliability. This article offers several contributions. First, it emphasises the ubiquitous role of model assumptions (both in ML and traditional Science) against the illusion of theory-free science. Secondly, model assumptions are analysed from the point of view of their (epistemic) complexity, which is shown 
    
[^28]: 从噪声蒸馏中出现的上下文强化学习

    Emergence of In-Context Reinforcement Learning from Noise Distillation

    [https://arxiv.org/abs/2312.12275](https://arxiv.org/abs/2312.12275)

    该论文介绍了一种从噪声中生成上下文强化学习的方法，通过构建噪声注入课程来获取学习历史，可以实现在学习数据集中超过最优策略的性能表现。

    

    最近在强化学习领域中，我们进行了大量关于变形金刚能够适应各种环境和任务的能力的研究。目前的上下文强化学习方法受到数据要求的限制，需要由强化学习代理生成或通过最优策略标记的数据。为了解决这个普遍存在的问题，我们提出了AD$^\varepsilon$，一种新的数据获取方法，可以通过噪声诱导的课程来实现上下文强化学习。我们展示了构建一个帮助获取学习历史的合成噪声注入课程是可行的。此外，我们通过实验证明，即使无需使用最优策略生成，上下文强化学习仍然能够以2倍的边界优于学习数据集中的最优策略。

    Recently, extensive studies in Reinforcement Learning have been carried out on the ability of transformers to adapt in-context to various environments and tasks. Current in-context RL methods are limited by their strict requirements for data, which needs to be generated by RL agents or labeled with actions from an optimal policy. In order to address this prevalent problem, we propose AD$^\varepsilon$, a new data acquisition approach that enables in-context Reinforcement Learning from noise-induced curriculum. We show that it is viable to construct a synthetic noise injection curriculum which helps to obtain learning histories. Moreover, we experimentally demonstrate that it is possible to alleviate the need for generation using optimal policies, with in-context RL still able to outperform the best suboptimal policy in a learning dataset by a 2x margin.
    
[^29]: 低成本高功率成员推断攻击

    Low-Cost High-Power Membership Inference Attacks

    [https://arxiv.org/abs/2312.03262](https://arxiv.org/abs/2312.03262)

    提出了一种新颖、高效且强大的成员推断攻击（RMIA），具有更准确的建模和更高的测试能力，适用于隐私风险评估。

    

    成员推断攻击（MIA）旨在检测特定数据点是否在训练机器学习模型时使用。最近一些强大的攻击具有较高的计算成本，并在不同条件下表现不一致，使它们对于实际的隐私风险评估不可靠。我们设计了一种新颖、高效且强大的成员推断攻击（RMIA），能够准确区分模型的总体数据和训练数据，同时计算开销最小。我们通过在似然比检验中更准确地建模零假设设置，并有效地利用来自总体的参考模型和参考数据样本，实现了这一目标。我们的算法在真正率（true-positive rate）方面表现出比先前方法更高的测试能力，整个TPR-FPR曲线都具备这种优势，即使在极低的误报率下（低至0）也是如此。在计算约束条件下，只有有限数量的情况下，

    arXiv:2312.03262v2 Announce Type: replace-cross  Abstract: Membership inference attacks (MIA) aim to detect if a particular data point was used in training a machine learning model. Recent strong attacks have high computational costs and inconsistent performance under varying conditions, rendering them unreliable for practical privacy risk assessment. We design a novel, efficient, and robust membership inference attack (RMIA) which accurately differentiates between population data and training data of a model, with minimal computational overhead. We achieve this by a more accurate modeling of the null hypothesis setting in our likelihood ratio tests, and effectively leveraging both reference models and reference data samples from the population. Our algorithm exhibits superior test power (true-positive rate) compared to prior methods, throughout the TPR-FPR curve including at extremely low false-positive rates (as low as 0). Under computation constraints, where only a limited number of
    
[^30]: LLM亲子鉴定：LLM遗传继承中的生成文本检测

    LLM Paternity Test: Generated Text Detection with LLM Genetic Inheritance

    [https://arxiv.org/abs/2305.12519](https://arxiv.org/abs/2305.12519)

    LLM-Pat提出了一种基于模型的生成文本检测方法，通过重建并比较候选文本与其对应的“兄弟”文本的相似性，从而判断候选文本是否由机器生成。

    

    大语言模型（LLMs）可以生成携带各种滥用风险的文本，包括抄袭、在电子商务平台上发布虚假评论，或者制作引人注目的虚假推文。因此，检测文本是否由机器生成变得越来越重要。虽然现有的检测方法表现出色，但由于严重依赖训练数据，它们往往缺乏泛化能力。为缓解这一问题，我们提出了一种与模型相关的生成文本检测方法，即LLM亲子鉴定（LLM-Pat）。具体而言，给定任何候选文本（"子类"），LLM-Pat使用一个中间LLM（"父类"）重建与给定文本对应的"兄弟"文本，然后衡量候选文本与其"兄弟"文本之间的相似性。高相似性表明候选文本是由机器生成，类似于基因特征。我们已构建了数据集...

    arXiv:2305.12519v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) can generate texts that carry the risk of various misuses, including plagiarism, planting fake reviews on e-commerce platforms, or creating inflammatory false tweets. Detecting whether a text is machine-generated has thus become increasingly important. While existing detection methods exhibit superior performance, they often lack generalizability due to their heavy dependence on training data. To alleviate this problem, we propose a model-related generated text detection method, the LLM Paternity Test (LLM-Pat). Specifically, given any candidate text (\textit{child}), LLM-Pat employs an intermediary LLM (\textit{parent}) to reconstruct a \textit{sibling} text corresponding to the given text and then measures the similarity between candidate texts and their sibling texts. High similarity indicates that the candidate text is machine-generated, akin to genetic traits. We have constructed datasets encom
    
[^31]: KGLiDS：用于数据科学的语义抽象、链接和自动化平台

    KGLiDS: A Platform for Semantic Abstraction, Linking, and Automation of Data Science

    [https://arxiv.org/abs/2303.02204](https://arxiv.org/abs/2303.02204)

    提出了一个可扩展平台KGLiDS，利用机器学习和知识图技术来抽象和捕获数据科学工具及其联系的语义，从而支持数据发现和管道自动化。

    

    最近几年，我们见证了学术界和工业界对应用数据科学技术来分析大量数据的日益浓厚兴趣。在这个过程中，我们创造了大量的工具（数据集、管道脚本等）。然而，尚未有系统性的尝试来全面收集和利用这些工具中隐含的所有知识和经验。相反，数据科学家从同事那里恢复信息和专业知识，或通过反复试验学习。因此，本文提出了一种可扩展的平台，KGLiDS，利用机器学习和知识图技术来抽象和捕获数据科学工具及其联系的语义。基于这些信息，KGLiDS能够支持各种下游应用，如数据发现和管道自动化。我们的全面评估涵盖了数据发现、数据清洗、转换和AutoM等用例。

    arXiv:2303.02204v3 Announce Type: replace  Abstract: In recent years, we have witnessed the growing interest from academia and industry in applying data science technologies to analyze large amounts of data. In this process, a myriad of artifacts (datasets, pipeline scripts, etc.) are created. However, there has been no systematic attempt to holistically collect and exploit all the knowledge and experiences that are implicitly contained in those artifacts. Instead, data scientists recover information and expertise from colleagues or learn via trial and error. Hence, this paper presents a scalable platform, KGLiDS, that employs machine learning and knowledge graph technologies to abstract and capture the semantics of data science artifacts and their connections. Based on this information, KGLiDS enables various downstream applications, such as data discovery and pipeline automation. Our comprehensive evaluation covers use cases in data discovery, data cleaning, transformation, and AutoM
    
[^32]: 设计你自己的宇宙：一种物理信息引导的无偏方法来增强图神经网络

    Design Your Own Universe: A Physics-Informed Agnostic Method for Enhancing Graph Neural Networks. (arXiv:2401.14580v1 [cs.LG])

    [http://arxiv.org/abs/2401.14580](http://arxiv.org/abs/2401.14580)

    本文提出了一种物理信息引导的无偏方法来增强图神经网络，通过引入附加节点和使用正负权重重连连接来丰富图结构，以解决过度平滑和过度压缩的问题。

    

    物理信息引导的图神经网络通过缓解常见的GNN挑战（如过度平滑化、过度压缩和异质适应）在学习图结构数据方面取得了显著的性能。尽管取得了这些进展，仍然在开发一种简单而有效的范式来适当地整合处理所有这些挑战的先前方法。在本文中，我们将GNN的传播与物理粒子系统进行类比，提出了一种模型无关的增强框架。该框架通过引入附加节点和使用正负权重重连连接来丰富图结构，受节点标记信息的指导。我们理论上验证了通过我们的方法增强的GNN可以有效地避免过度平滑问题，并对过度压缩具有鲁棒性。此外，我们对重连图进行了谱分析，证明了相应的GNN可以...

    Physics-informed Graph Neural Networks have achieved remarkable performance in learning through graph-structured data by mitigating common GNN challenges such as over-smoothing, over-squashing, and heterophily adaption. Despite these advancements, the development of a simple yet effective paradigm that appropriately integrates previous methods for handling all these challenges is still underway. In this paper, we draw an analogy between the propagation of GNNs and particle systems in physics, proposing a model-agnostic enhancement framework. This framework enriches the graph structure by introducing additional nodes and rewiring connections with both positive and negative weights, guided by node labeling information. We theoretically verify that GNNs enhanced through our approach can effectively circumvent the over-smoothing issue and exhibit robustness against over-squashing. Moreover, we conduct a spectral analysis on the rewired graph to demonstrate that the corresponding GNNs can f
    
[^33]: 大型语言模型中的代码模拟挑战

    Code Simulation Challenges for Large Language Models. (arXiv:2401.09074v1 [cs.LG])

    [http://arxiv.org/abs/2401.09074](http://arxiv.org/abs/2401.09074)

    大型语言模型在模拟计算机代码和算法执行方面遇到挑战，性能随着代码长度的增加而迅速下降。在处理短程序或标准过程时，它们能以低错误率按顺序执行指令，但对于复杂的程序，特别是包含关键路径和冗余指令的程序，模拟效果较差。我们提出了一种逐行模拟代码执行的方法来解决这个问题。

    

    我们调查了大型语言模型（LLMs）在模拟计算机代码和算法执行方面的能力。我们首先研究了直线程序，并展示了当前LLMs在处理这样简单的程序时表现出的性能较差——性能随着代码长度的增加而迅速下降。接着，我们研究了LLMs在模拟包含关键路径和冗余指令的程序方面的能力。我们还通过排序算法和嵌套循环超越了直线程序的模拟，并展示了程序的计算复杂性直接影响LLMs模拟其执行的能力。我们观察到LLMs只有在处理短程序或标准过程时才能以低错误率按顺序执行指令。LLMs的代码模拟与它们的模式识别和记忆能力存在矛盾：在记忆对任务有害的情况下，我们提出了一种新的提示方法，逐行模拟代码的执行。

    We investigate the extent to which Large Language Models (LLMs) can simulate the execution of computer code and algorithms. We begin by looking straight line programs, and show that current LLMs demonstrate poor performance even with such simple programs -- performance rapidly degrades with the length of code. We then investigate the ability of LLMs to simulate programs that contain critical paths and redundant instructions. We also go beyond straight line program simulation with sorting algorithms and nested loops, and we show the computational complexity of a routine directly affects the ability of an LLM to simulate its execution. We observe that LLMs execute instructions sequentially and with a low error margin only for short programs or standard procedures. LLMs' code simulation is in tension with their pattern recognition and memorisation capabilities: on tasks where memorisation is detrimental, we propose a novel prompting method to simulate code execution line by line. Empirica
    
[^34]: 利用公共表示来进行私有迁移学习

    Leveraging Public Representations for Private Transfer Learning. (arXiv:2312.15551v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2312.15551](http://arxiv.org/abs/2312.15551)

    该论文探讨了如何利用公共数据来改进私有学习的问题。研究发现，通过学习公共数据中的共享表示，可以在两种迁移学习场景中实现最优的学习效果。在单任务迁移场景中，算法在给定子空间范围内搜索线性模型，并实现了最优超额风险。在多任务个性化场景中，足够的公共数据可以消除私有协调需求，并通过纯局部学习达到相同的效用。

    

    受到将公共数据纳入差分隐私学习的最新实证成功的启发，我们在理论上研究了从公共数据中学到的共享表示如何改进私有学习。我们探讨了线性回归的两种常见迁移学习场景，两者都假设公共任务和私有任务（回归向量）在高维空间中共享一个低秩子空间。在第一种单任务迁移场景中，目标是学习一个在所有用户之间共享的单一模型，每个用户对应数据集中的一行。我们提供了匹配的上下界，证明了我们的算法在给定子空间估计范围内搜索线性模型的算法类中实现了最优超额风险。在多任务模型个性化的第二种情景中，我们表明在有足够的公共数据情况下，用户可以避免私有协调，因为在给定子空间内纯粹的局部学习可以达到相同的效用。

    Motivated by the recent empirical success of incorporating public data into differentially private learning, we theoretically investigate how a shared representation learned from public data can improve private learning. We explore two common scenarios of transfer learning for linear regression, both of which assume the public and private tasks (regression vectors) share a low-rank subspace in a high-dimensional space. In the first single-task transfer scenario, the goal is to learn a single model shared across all users, each corresponding to a row in a dataset. We provide matching upper and lower bounds showing that our algorithm achieves the optimal excess risk within a natural class of algorithms that search for the linear model within the given subspace estimate. In the second scenario of multitask model personalization, we show that with sufficient public data, users can avoid private coordination, as purely local learning within the given subspace achieves the same utility. Take
    
[^35]: 图标签传播算法应对图标签噪声问题

    Label Propagation for Graph Label Noise. (arXiv:2310.16560v1 [cs.LG])

    [http://arxiv.org/abs/2310.16560](http://arxiv.org/abs/2310.16560)

    本文研究了图中的标签噪声问题，提出了一种基于标签传播的算法来处理任意异质性的图标签噪声，以纠正噪声标签并为未标记的节点分配标签。

    

    标签噪声是大型数据集中常见的挑战，它会显著降低深度神经网络的泛化能力。大部分现有研究都集中在计算机视觉中的噪声标签，然而，图模型将节点特征和图拓扑结构作为输入，通过消息传递机制更容易受到标签噪声的影响。近期，只有少数几篇文章提出了解决图中标签噪声的方法。其中一个主要限制是它们假设图是同构的，并且标签是平滑分布的。然而，现实世界中的图可能包含不同程度的异质性甚至是异质性的主导，导致当前方法的不足。本文研究任意异质性条件下的图标签噪声问题，旨在纠正噪声标签并为之前未标记的节点分配标签。我们首先进行了两个实证分析，探讨图同质性对图标签噪声的影响。接着，我们提出了一种基于标签传播的算法来处理任意异质性的图标签噪声。

    Label noise is a common challenge in large datasets, as it can significantly degrade the generalization ability of deep neural networks. Most existing studies focus on noisy labels in computer vision; however, graph models encompass both node features and graph topology as input, and become more susceptible to label noise through message-passing mechanisms. Recently, only a few works have been proposed to tackle the label noise on graphs. One major limitation is that they assume the graph is homophilous and the labels are smoothly distributed. Nevertheless, real-world graphs may contain varying degrees of heterophily or even be heterophily-dominated, leading to the inadequacy of current methods. In this paper, we study graph label noise in the context of arbitrary heterophily, with the aim of rectifying noisy labels and assigning labels to previously unlabeled nodes. We begin by conducting two empirical analyses to explore the impact of graph homophily on graph label noise. Following o
    
[^36]: 非刚性文本提示的音频编辑

    Audio Editing with Non-Rigid Text Prompts. (arXiv:2310.12858v1 [cs.SD])

    [http://arxiv.org/abs/2310.12858](http://arxiv.org/abs/2310.12858)

    本文研究了使用非刚性文本编辑进行音频编辑的方法，并展示了其在保持输入音频一致性方面的优势。

    

    本文探讨了使用非刚性文本编辑进行音频编辑。我们展示了所提出的编辑流程能够创建与输入音频保持一致的音频编辑结果。我们探索了能够进行添加、风格转换和修复的文本提示。我们定量和定性地证明了这些编辑能够优于最近发布的文本提示音频生成模型Audio-LDM的结果。对结果的定性检查表明，我们的方法给出了更加保持输入音频原始起始和结束的编辑结果。

    In this paper, we explore audio-editing with non-rigid text edits. We show that the proposed editing pipeline is able to create audio edits that remain faithful to the input audio. We explore text prompts that perform addition, style transfer, and in-painting. We quantitatively and qualitatively show that the edits are able to obtain results which outperform Audio-LDM, a recently released text-prompted audio generation model. Qualitative inspection of the results points out that the edits given by our approach remain more faithful to the input audio in terms of keeping the original onsets and offsets of the audio events.
    
[^37]: 渗透式人工智能：使LLMs理解物理世界

    Penetrative AI: Making LLMs Comprehend the Physical World. (arXiv:2310.09605v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2310.09605](http://arxiv.org/abs/2310.09605)

    本文探讨了渗透式人工智能的概念，旨在使LLMs能够通过物联网传感器与执行器与物理世界进行交互和推理。初步研究结果表明，LLMs具有独特的能力，能够应用内嵌的世界知识解释物联网传感器数据并进行物理领域的推理。

    

    最近大型语言模型（LLMs）的发展展示了它们在各种任务中的显著能力。然而，关于LLMs的性质以及它们在涉及真实物理世界信息的任务中整合常识人类知识的潜力仍存在疑问。本文通过探索LLMs如何通过物联网传感器和执行器与物理世界进行交互和推理来探讨这些问题，这一概念称为“渗透式人工智能”。论文在LLMs能够透过处理感知信号的两个层面上探索了这种扩展。我们的初步研究结果表明，LLMs（ChatGPT是我们研究中的代表性例子）在应用内嵌的世界知识解释物联网传感器数据并对物理领域的任务进行推理方面具有相当独特的能力。这不仅为LLMs开辟了新的应用领域。

    Recent developments in Large Language Models (LLMs) have demonstrated their remarkable capabilities across a range of tasks. Questions, however, persist about the nature of LLMs and their potential to integrate common-sense human knowledge when performing tasks involving information about the real physical world. This paper delves into these questions by exploring how LLMs can be extended to interact with and reason about the physical world through IoT sensors and actuators, a concept that we term "Penetrative AI". The paper explores such an extension at two levels of LLMs' ability to penetrate into the physical world via the processing of sensory signals. Our preliminary findings indicate that LLMs, with ChatGPT being the representative example in our exploration, have considerable and unique proficiency in employing the embedded world knowledge for interpreting IoT sensor data and reasoning over them about tasks in the physical realm. Not only this opens up new applications for LLMs 
    
[^38]: 具有正交权重的深度网络中的特征学习与泛化

    Feature Learning and Generalization in Deep Networks with Orthogonal Weights. (arXiv:2310.07765v1 [cs.LG])

    [http://arxiv.org/abs/2310.07765](http://arxiv.org/abs/2310.07765)

    我们通过使用正交矩阵集合初始化权重并使用tanh激活函数，解决了全连接深度神经网络在初始化中具有与深度无关的线性波动问题。此外，我们发现神经切向核（NTK）及其后代的所有相关函数在逆宽度的主导阶段在深度约为20的位置饱和，而不是不断增长。

    

    通过使用从正交矩阵集合初始化的权重和tanh激活函数，我们展示了全连接深度神经网络在初始化时具有与宽度无关的前激活波动，这是通过计算证明的。此外，我们通过数值实验证明，在初始化时，涉及神经切向核（NTK）及其后代的所有相关函数在逆宽度的主导阶段饱和在深度约为20的位置，而不是像高斯初始化的情况那样不断增长。

    Fully-connected deep neural networks with weights initialized from independent Gaussian distributions can be tuned to criticality, which prevents the exponential growth or decay of signals propagating through the network. However, such networks still exhibit fluctuations that grow linearly with the depth of the network, which may impair the training of networks with width comparable to depth. We show analytically that rectangular networks with tanh activations and weights initialized from the ensemble of orthogonal matrices have corresponding preactivation fluctuations which are independent of depth, to leading order in inverse width. Moreover, we demonstrate numerically that, at initialization, all correlators involving the neural tangent kernel (NTK) and its descendants at leading order in inverse width -- which govern the evolution of observables during training -- saturate at a depth of $\sim 20$, rather than growing without bound as in the case of Gaussian initializations. We spec
    
[^39]: 损失平坦性与神经网络中压缩表示的简单联系

    A simple connection from loss flatness to compressed representations in neural networks. (arXiv:2310.01770v1 [cs.LG])

    [http://arxiv.org/abs/2310.01770](http://arxiv.org/abs/2310.01770)

    该论文研究了深度神经网络中损失平坦性和神经表示压缩之间的关系，通过简单的数学关系，证明了损失平坦性与神经表示的压缩相关。

    

    对深度神经网络的泛化能力进行研究的方法有很多种，包括至少两种不同的方法：一种基于参数空间中损失景观的形状，另一种基于特征空间中表示流形的结构（即单位活动的空间）。这两种方法相关但很少同时进行研究和明确关联。在这里，我们提出了一种简单的分析方法来建立这种联系。我们展示了在深度神经网络学习的最后阶段，神经表示流形的体积压缩与正在进行的参数优化所探索的最小值周围的损失平坦性相关。我们证明了这可以由一个相对简单的数学关系来预测：损失平坦性意味着神经表示的压缩。我们的结果与\citet{ma_linear_2021}的先前研究密切相关，该研究展示了平坦性（即小特征值）与表示流形的体积压缩之间的关系。

    Deep neural networks' generalization capacity has been studied in a variety of ways, including at least two distinct categories of approach: one based on the shape of the loss landscape in parameter space, and the other based on the structure of the representation manifold in feature space (that is, in the space of unit activities). These two approaches are related, but they are rarely studied together and explicitly connected. Here, we present a simple analysis that makes such a connection. We show that, in the last phase of learning of deep neural networks, compression of the volume of the manifold of neural representations correlates with the flatness of the loss around the minima explored by ongoing parameter optimization. We show that this is predicted by a relatively simple mathematical relationship: loss flatness implies compression of neural representations. Our results build closely on prior work of \citet{ma_linear_2021}, which shows how flatness (i.e., small eigenvalues of t
    
[^40]: 通过基于注意力的深度状态空间建模，将PPG信号转换为ECG，用于连续性心房颤动检测

    PPG to ECG Signal Translation for Continuous Atrial Fibrillation Detection via Attention-based Deep State-Space Modeling. (arXiv:2309.15375v1 [cs.LG])

    [http://arxiv.org/abs/2309.15375](http://arxiv.org/abs/2309.15375)

    通过基于注意力的深度状态空间建模，我们提出了一种不受个体限制的方法，将PPG信号转换为ECG，用于连续性心房颤动检测。

    

    电信号图（ECG或EKG）是一种测量心脏电活动的医学测试。ECG常用于诊断和监测各种心脏疾病，包括心律失常、心肌梗塞和心力衰竭。然而，传统的ECG需要临床测量，限制了其在医疗机构的应用。相比之下，单导联ECG已经在佩戴式设备上应用广泛。另一种ECG的替代方法是光浊度脉搏检测（PPG），它采用非侵入性、低成本的光学方法来测量心脏生理学，使其成为捕捉日常生活中重要心脏信号的合适选择。虽然ECG和PPG之间具有强烈的相关性，但后者并没有提供明显的临床诊断价值。在这里，我们提出了一种不受个体限制的基于注意力的深度状态空间模型，用于将PPG信号转换为ECG，从而实现连续性心房颤动检测。

    An electrocardiogram (ECG or EKG) is a medical test that measures the heart's electrical activity. ECGs are often used to diagnose and monitor a wide range of heart conditions, including arrhythmias, heart attacks, and heart failure. On the one hand, the conventional ECG requires clinical measurement, which restricts its deployment to medical facilities. On the other hand, single-lead ECG has become popular on wearable devices using administered procedures. An alternative to ECG is Photoplethysmography (PPG), which uses non-invasive, low-cost optical methods to measure cardiac physiology, making it a suitable option for capturing vital heart signs in daily life. As a result, it has become increasingly popular in health monitoring and is used in various clinical and commercial wearable devices. While ECG and PPG correlate strongly, the latter does not offer significant clinical diagnostic value. Here, we propose a subject-independent attention-based deep state-space model to translate P
    
[^41]: 通过稳健对齐的LLM抵御对齐破坏攻击

    Defending Against Alignment-Breaking Attacks via Robustly Aligned LLM. (arXiv:2309.14348v1 [cs.CL])

    [http://arxiv.org/abs/2309.14348](http://arxiv.org/abs/2309.14348)

    本文提出了一种稳健对齐的LLM（RA-LLM），用于防御可能发生的对齐破坏攻击。RA-LLM可以直接在现有的对齐LLM上构建，并通过稳健的对齐检查函数来确保其有效性。

    

    最近，大型语言模型（LLMs）取得了显著的进展，并在各个领域得到广泛应用。不幸的是，人们越来越担心LLMs可能被滥用来生成有害或恶意内容。尽管有一系列的研究专注于对齐LLMs与人类价值观，并防止它们生成不适当的内容，但这些对齐通常是脆弱的，并且可以通过对抗优化或手工构建的越狱提示来绕过。在这项工作中，我们介绍了一种稳健对齐的LLM（RA-LLM），以防范潜在的对齐破坏攻击。RA-LLM可以直接构建在现有的对齐LLM上，通过具有稳健对齐检查功能的方法，而无需对原始LLM进行任何昂贵的重新训练或微调。此外，我们还通过理论分析验证了RA-LLM在防御对齐破坏攻击方面的有效性。通过现实世界的实验，

    Recently, Large Language Models (LLMs) have made significant advancements and are now widely used across various domains. Unfortunately, there has been a rising concern that LLMs can be misused to generate harmful or malicious content. Though a line of research has focused on aligning LLMs with human values and preventing them from producing inappropriate content, such alignments are usually vulnerable and can be bypassed by alignment-breaking attacks via adversarially optimized or handcrafted jailbreaking prompts. In this work, we introduce a Robustly Aligned LLM (RA-LLM) to defend against potential alignment-breaking attacks. RA-LLM can be directly constructed upon an existing aligned LLM with a robust alignment checking function, without requiring any expensive retraining or fine-tuning process of the original LLM. Furthermore, we also provide a theoretical analysis for RA-LLM to verify its effectiveness in defending against alignment-breaking attacks. Through real-world experiments
    
[^42]: 量子噪声驱动的生成扩散模型

    Quantum-Noise-driven Generative Diffusion Models. (arXiv:2308.12013v1 [quant-ph])

    [http://arxiv.org/abs/2308.12013](http://arxiv.org/abs/2308.12013)

    该论文提出了三种量子噪声驱动的生成扩散模型，利用了量子特性以克服传统模型的主要计算困难，并建议将量子噪声视为可利用的特性而非问题。

    

    通过机器学习技术实现的生成模型是从有限的训练样本中推断出复杂和未知数据分布并产生新的合成数据的强大工具。扩散模型是一种新兴的框架，最近在创建合成文本和高质量图像方面已经超越了生成对抗性网络的性能。在这里，我们提出并讨论了扩散模型的量子推generalization，即三种可能在实际量子系统上进行实验的量子噪声驱动的生成扩散模型。我们的想法是利用独特的量子特性，特别是目前可用的有噪声量子处理器不可避免地受到的相干性、纠缠性和噪声之间的非平凡相互作用，以克服传统扩散模型在推断过程中的主要计算负担。因此，我们建议将量子噪声不作为需要检测和解决的问题，而是作为一种可利用的特性，使得扩散模型能够更好地工作。

    Generative models realized with machine learning techniques are powerful tools to infer complex and unknown data distributions from a finite number of training samples in order to produce new synthetic data. Diffusion models are an emerging framework that have recently overcome the performance of the generative adversarial networks in creating synthetic text and high-quality images. Here, we propose and discuss the quantum generalization of diffusion models, i.e., three quantum-noise-driven generative diffusion models that could be experimentally tested on real quantum systems. The idea is to harness unique quantum features, in particular the non-trivial interplay among coherence, entanglement and noise that the currently available noisy quantum processors do unavoidably suffer from, in order to overcome the main computational burdens of classical diffusion models during inference. Hence, we suggest to exploit quantum noise not as an issue to be detected and solved but instead as a ver
    
[^43]: Transformers能否学习用于未知系统的最优滤波？

    Can Transformers Learn Optimal Filtering for Unknown Systems?. (arXiv:2308.08536v1 [eess.SY])

    [http://arxiv.org/abs/2308.08536](http://arxiv.org/abs/2308.08536)

    本文研究了使用transformers进行最优输出估计问题，通过训练一个transformer来在未知系统上进行预测，并命名为元输出预测器（MOP）。我们观察到，尽管MOP没有访问模型的权限，但在大多数线性动态系统中，它的性能与基于卡尔曼滤波器的最优输出估计器相当，在具有非独立同分布噪声和时变动态的挑战性场景中也表现优秀。

    

    Transformers在自然语言处理中取得了显著的成功，然而它们在动态系统中的潜力仍然大部分未被探索。本文研究了使用transformers进行最优输出估计问题，它使用过去的所有输出来生成预测。我们使用来自先验分布的各种系统来训练transformer，然后在先前未见过的相同分布的系统上评估其性能。结果表明，获得的transformer就像一个预测算法，它可以在上下文中学习并快速适应和预测不同的系统，因此我们称之为元输出预测器（MOP）。尽管MOP没有访问模型的权限，但在大多数线性动态系统中，它的性能与基于卡尔曼滤波器的最优输出估计器相当。通过大量的数值实验，我们观察到MOP在具有非独立同分布噪声和时变动态的挑战性场景中也表现优秀。

    Transformers have demonstrated remarkable success in natural language processing; however, their potential remains mostly unexplored for problems arising in dynamical systems. In this work, we investigate the optimal output estimation problem using transformers, which generate output predictions using all the past ones. We train the transformer using various systems drawn from a prior distribution and then evaluate its performance on previously unseen systems from the same distribution. As a result, the obtained transformer acts like a prediction algorithm that learns in-context and quickly adapts to and predicts well for different systems - thus we call it meta-output-predictor (MOP). MOP matches the performance of the optimal output estimator, based on Kalman filter, for most linear dynamical systems even though it does not have access to a model. We observe via extensive numerical experiments that MOP also performs well in challenging scenarios with non-i.i.d. noise, time-varying dy
    
[^44]: 关于(Normalised) Discounted Cumulative Gain作为Top-n推荐的离线评估指标的论文翻译

    On (Normalised) Discounted Cumulative Gain as an Offline Evaluation Metric for Top-$n$ Recommendation. (arXiv:2307.15053v1 [cs.IR])

    [http://arxiv.org/abs/2307.15053](http://arxiv.org/abs/2307.15053)

    本文批判性审视了(Normalised) Discounted Cumulative Gain作为Top-n推荐离线评估指标的方法，并研究了何时可以期望这些指标逼近在线实验的金标准结果。

    

    推荐方法通常通过两种方式进行评估：(1) 通过(模拟)在线实验，通常被视为金标准，或者(2) 通过一些离线评估程序，目标是近似在线实验的结果。文献中采用了几种离线评估指标，受信息检索领域中常见的排名指标的启发。(Normalised) Discounted Cumulative Gain (nDCG)是其中一种广泛采用的度量标准，在很多年里，更高的(n)DCG值被用来展示新方法在Top-n推荐中的最新进展。我们的工作对这种方法进行了批判性的审视，并研究了我们何时可以期望这些指标逼近在线实验的金标准结果。我们从第一原理上正式提出了DCG被认为是在线奖励的无偏估计的假设，并给出了这个指标的推导。

    Approaches to recommendation are typically evaluated in one of two ways: (1) via a (simulated) online experiment, often seen as the gold standard, or (2) via some offline evaluation procedure, where the goal is to approximate the outcome of an online experiment. Several offline evaluation metrics have been adopted in the literature, inspired by ranking metrics prevalent in the field of Information Retrieval. (Normalised) Discounted Cumulative Gain (nDCG) is one such metric that has seen widespread adoption in empirical studies, and higher (n)DCG values have been used to present new methods as the state-of-the-art in top-$n$ recommendation for many years.  Our work takes a critical look at this approach, and investigates when we can expect such metrics to approximate the gold standard outcome of an online experiment. We formally present the assumptions that are necessary to consider DCG an unbiased estimator of online reward and provide a derivation for this metric from first principles
    
[^45]: 使用机器学习抑制动力系统中的未知干扰

    Suppressing unknown disturbances to dynamical systems using machine learning. (arXiv:2307.03690v1 [eess.SY])

    [http://arxiv.org/abs/2307.03690](http://arxiv.org/abs/2307.03690)

    本文提出了一种使用机器学习的无模型方法，可以仅通过系统在已知强迫函数影响下的观测，识别和抑制未知系统的未知干扰。这项方法对训练函数有非常温和的限制，能够稳健地识别和抑制大类别的未知干扰。

    

    识别和抑制动力系统中的未知干扰是一个在许多不同领域中应用的问题。在本文中，我们提出了一种无模型的方法，仅基于系统在已知强迫函数影响下的先前观测来识别和抑制未知系统的未知干扰。我们发现，在对训练函数有非常温和的限制下，我们的方法能够稳健地识别和抑制大类别的未知干扰。我们通过一个示例说明了我们的方案，其中识别和抑制了 Lorenz 系统的混沌干扰。

    Identifying and suppressing unknown disturbances to dynamical systems is a problem with applications in many different fields. In this Letter, we present a model-free method to identify and suppress an unknown disturbance to an unknown system based only on previous observations of the system under the influence of a known forcing function. We find that, under very mild restrictions on the training function, our method is able to robustly identify and suppress a large class of unknown disturbances. We illustrate our scheme with an example where a chaotic disturbance to the Lorenz system is identified and suppressed.
    
[^46]: 透视额外信息的Informed POMDP: 模型驱动强化学习中的利用

    Informed POMDP: Leveraging Additional Information in Model-Based RL. (arXiv:2306.11488v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2306.11488](http://arxiv.org/abs/2306.11488)

    本文提出了Informed POMDP，这是一种新的学习范式，通过学习环境模型来利用训练时可用的额外信息，该模型可以提高Dreamer算法策略的收敛速度。

    

    本文通过考虑训练时可用的额外信息，推广了POMDP中的交互学习问题。首先，我们引入了Informed POMDP，这是一种新的学习范式，清晰区分了训练信息和执行观察。接下来，我们提出了一个目标，用于从历史记录中学习出充分统计信息，以实现最优控制并利用这些信息。然后，我们展示了这个Informed目标由学习一个环境模型组成，从其中我们可以采样隐式轨迹。最后，我们证明在几个环境中，使用这个Informed环境模型可以大大提高Dreamer算法策略的收敛速度。这些结果以及提议的简单适应性，倡导在使用模型驱动强化学习学习POMDP时，要系统地考虑可用的额外信息。

    In this work, we generalize the problem of learning through interaction in a POMDP by accounting for eventual additional information available at training time. First, we introduce the informed POMDP, a new learning paradigm offering a clear distinction between the training information and the execution observation. Next, we propose an objective for learning a sufficient statistic from the history for the optimal control that leverages this information. We then show that this informed objective consists of learning an environment model from which we can sample latent trajectories. Finally, we show for the Dreamer algorithm that the convergence speed of the policies is sometimes greatly improved on several environments by using this informed environment model. Those results and the simplicity of the proposed adaptation advocate for a systematic consideration of eventual additional information when learning in a POMDP using model-based RL.
    
[^47]: 指数族噪声下的图拉普拉斯学习

    Graph Laplacian Learning with Exponential Family Noise. (arXiv:2306.08201v1 [cs.LG])

    [http://arxiv.org/abs/2306.08201](http://arxiv.org/abs/2306.08201)

    该论文将图信号处理框架推广到指数族噪声分布，提出了一种交替算法，用于从噪声信号中估计图拉普拉斯和未观测的平滑表示。实验结果表明，该算法可以在噪声模型不匹配情况下优于其他拉普拉斯估计方法。

    

    应用图机器学习方法时，面临的常见挑战是系统的底层图往往是未知的。虽然为连续的图信号提出了不同的图推断方法，但对于其他类型的数据（例如离散计数），推断其底层图结构的方法尚未得到充分探索。本文将图信号处理（GSP）框架推广到指数族噪声分布，以建模各种数据类型的图学习中。我们提出了一种交替算法，从噪声信号中估计图拉普拉斯和未观测的平滑表示。我们在合成和真实数据中展示了我们的新算法在噪声模型不匹配情况下优于竞争拉普拉斯估计方法的表现。

    A common challenge in applying graph machine learning methods is that the underlying graph of a system is often unknown. Although different graph inference methods have been proposed for continuous graph signals, inferring the graph structure underlying other types of data, such as discrete counts, is under-explored. In this paper, we generalize a graph signal processing (GSP) framework for learning a graph from smooth graph signals to the exponential family noise distribution to model various data types. We propose an alternating algorithm that estimates the graph Laplacian as well as the unobserved smooth representation from the noisy signals. We demonstrate in synthetic and real-world data that our new algorithm outperforms competing Laplacian estimation methods under noise model mismatch.
    
[^48]: 适应性稀疏度训练过程中的变化，用于利用Transformer进行高效时间序列预测

    Adaptive Sparsity Level during Training for Efficient Time Series Forecasting with Transformers. (arXiv:2305.18382v1 [cs.LG])

    [http://arxiv.org/abs/2305.18382](http://arxiv.org/abs/2305.18382)

    本文提出了“具有自适应稀疏度级别的修剪”(PALS), 通过稀疏训练和训练期间方法中的“扩张”机制，在Transformer模型中实现高效的时间序列预测。

    

    高效的时间序列预测对于深度神经网络应用变得至关重要。通过稀疏连接和减小模型尺寸，可以实现DNN的高效性。然而，在训练过程中自动确定稀疏度仍然是一个具有挑战性的任务，因为不同数据集中的损失稀疏度权衡是异构的。本文提出了“具有自适应稀疏度级别的修剪”(PALS)，来自动寻求损失和稀疏性之间的最佳平衡，无需预定义稀疏水平。PALS从稀疏训练和训练期间方法中吸取灵感。它在训练稀疏神经网络中引入了新颖的“扩张”机制，允许模型动态收缩、扩张或保持稳定，以找到适当的稀疏度。本文专注于在以Transformer著称的模型中实现效率, 该模型以其出色的时间序列预测表现而闻名。

    Efficient time series forecasting has become critical for real-world applications, particularly with deep neural networks (DNNs). Efficiency in DNNs can be achieved through sparse connectivity and reducing the model size. However, finding the sparsity level automatically during training remains a challenging task due to the heterogeneity in the loss-sparsity tradeoffs across the datasets. In this paper, we propose \enquote{\textbf{P}runing with \textbf{A}daptive \textbf{S}parsity \textbf{L}evel} (\textbf{PALS}), to automatically seek an optimal balance between loss and sparsity, all without the need for a predefined sparsity level. PALS draws inspiration from both sparse training and during-training methods. It introduces the novel "expand" mechanism in training sparse neural networks, allowing the model to dynamically shrink, expand, or remain stable to find a proper sparsity level. In this paper, we focus on achieving efficiency in transformers known for their excellent time series f
    
[^49]: 通过Logit Attribution匹配实现对比度领域泛化

    Contrastive Domain Generalization via Logit Attribution Matching. (arXiv:2305.07888v1 [cs.LG])

    [http://arxiv.org/abs/2305.07888](http://arxiv.org/abs/2305.07888)

    本论文提出了一种名为对比领域泛化（CDG）的新方法，通过强烈对比的数据对所展示的语义不变性进行利用。同时，提出了一种正则化技术——Logit Attribution Matching (LAM)，以实现CDG。实验结果表明，LAM仅使用少量配对数据就能胜过最先进的DG方法，且有助于模型更好地关注对领域泛化至关重要的语义特征。

    

    领域泛化是机器学习中一个重要的开放性问题。深度模型容易受到甚至微小程度的领域偏移的影响，在实际应用中严重损害其可靠性。为了缓解这个问题，大多数现有的方法在多个训练领域上加强各种不变量限制。然而，这种方法通常不能为新的测试领域提供很好的性能保证。在本文中，我们研究了一种不同的方法，名为对比领域泛化（CDG），它利用强烈对比的数据对所展示的语义不变性，而不是多个域。我们提出了一个因果领域泛化理论，展示了CDG的潜在能力；同时，我们还提出了一种正则化技术——Logit Attribution Matching (LAM)，以实现CDG。我们在实证上展示了，LAM仅使用少量配对数据就能胜过最先进的DG方法，而且LAM有助于模型更好地关注对领域泛化至关重要的语义特征。

    Domain Generalization (DG) is an important open problem in machine learning. Deep models are susceptible to domain shifts of even minute degrees, which severely compromises their reliability in real applications. To alleviate the issue, most existing methods enforce various invariant constraints across multiple training domains. However,such an approach provides little performance guarantee for novel test domains in general. In this paper, we investigate a different approach named Contrastive Domain Generalization (CDG), which exploits semantic invariance exhibited by strongly contrastive data pairs in lieu of multiple domains. We present a causal DG theory that shows the potential capability of CDG; together with a regularization technique, Logit Attribution Matching (LAM), for realizing CDG. We empirically show that LAM outperforms state-of-the-art DG methods with only a small portion of paired data and that LAM helps models better focus on semantic features which are crucial to DG.
    
[^50]: LASER：神经符号学习语义视频表示

    LASER: Neuro-Symbolic Learning of Semantic Video Representations. (arXiv:2304.07647v1 [cs.CV])

    [http://arxiv.org/abs/2304.07647](http://arxiv.org/abs/2304.07647)

    LASER提出了一种神经符号学习方法来学习语义视频表示，通过逻辑规范捕捉视频数据中的时空属性，能够对齐原始视频和规范，有效地训练低级感知模型以提取符合所需高级规范的视频表示。

    

    现代涉及视频的AI应用（如视频-文本对齐、视频搜索和视频字幕）受益于对视频语义的细致理解。现有的视频理解方法要么需要大量注释，要么基于不可解释的通用嵌入，可能会忽略重要细节。我们提出了LASER，这是一种神经符号方法，通过利用能够捕捉视频数据中丰富的时空属性的逻辑规范来学习语义视频表示。特别地，我们通过原始视频与规范之间的对齐来公式化问题。对齐过程有效地训练了低层感知模型，以提取符合所需高层规范的细粒度视频表示。我们的流程可以端到端地训练，并可纳入从规范导出的对比和语义损失函数。我们在两个具有丰富空间和时间信息的数据集上评估了我们的方法。

    Modern AI applications involving video, such as video-text alignment, video search, and video captioning, benefit from a fine-grained understanding of video semantics. Existing approaches for video understanding are either data-hungry and need low-level annotation, or are based on general embeddings that are uninterpretable and can miss important details. We propose LASER, a neuro-symbolic approach that learns semantic video representations by leveraging logic specifications that can capture rich spatial and temporal properties in video data. In particular, we formulate the problem in terms of alignment between raw videos and specifications. The alignment process efficiently trains low-level perception models to extract a fine-grained video representation that conforms to the desired high-level specification. Our pipeline can be trained end-to-end and can incorporate contrastive and semantic loss functions derived from specifications. We evaluate our method on two datasets with rich sp
    
[^51]: 用图论统一刻画差分隐私可学习性

    A Unified Characterization of Private Learnability via Graph Theory. (arXiv:2304.03996v1 [cs.LG])

    [http://arxiv.org/abs/2304.03996](http://arxiv.org/abs/2304.03996)

    本文提供了一个统一的框架，使用图论的语言刻画了差分隐私的两种情形下，纯粹和近似的学习性。我们通过定义矛盾图$G$来捕捉 $\mathcal{H}$ 的组合结构，发现分数团数和团数是描述差分隐私学习性的重要因素，并提出了几种算法对其进行估计。

    

    我们提供了一个统一的框架来刻画纯粹的和近似的差分隐私学习性。该框架使用了图论的语言:对于一个概念类 $\mathcal{H}$,我们定义了 $\mathcal{H}$ 的矛盾图 $G$。它的顶点是可实现的数据集，如果两个数据集 $S$，$S'$ 相互矛盾(即，在 $S$ 和 $S'$ 中有一个点 $x$ 具有不同的标记)，则它们之间有一条边连接。我们的主要发现是，$G$ 的组合结构与在差分隐私下学习 $\mathcal{H}$ 密切相关。在纯粹的差分隐私下学习 $\mathcal{H}$ 的捕获为 $G$ 的分数团数。在近似差分隐私下学习 $\mathcal{H}$ 的捕获为 $G$ 的团数。因此，我们确定了描述差分隐私可学习性的图论维度：团维和分数团维。同时，我们揭示了矛盾图的一些性质，这些性质可能是独立感兴趣的。我们还提出了几种算法来估计 $G$ 的这些度量，通过这些算法，我们实现了几种概念类的实验研究。

    We provide a unified framework for characterizing pure and approximate differentially private (DP) learnabiliity. The framework uses the language of graph theory: for a concept class $\mathcal{H}$, we define the contradiction graph $G$ of $\mathcal{H}$. It vertices are realizable datasets, and two datasets $S,S'$ are connected by an edge if they contradict each other (i.e., there is a point $x$ that is labeled differently in $S$ and $S'$). Our main finding is that the combinatorial structure of $G$ is deeply related to learning $\mathcal{H}$ under DP. Learning $\mathcal{H}$ under pure DP is captured by the fractional clique number of $G$. Learning $\mathcal{H}$ under approximate DP is captured by the clique number of $G$. Consequently, we identify graph-theoretic dimensions that characterize DP learnability: the clique dimension and fractional clique dimension. Along the way, we reveal properties of the contradiction graph which may be of independent interest. We also suggest several o
    
[^52]: 用混合VAE模型学习流形来解决逆问题

    Manifold Learning by Mixture Models of VAEs for Inverse Problems. (arXiv:2303.15244v1 [cs.LG])

    [http://arxiv.org/abs/2303.15244](http://arxiv.org/abs/2303.15244)

    本文提出了一种用混合VAE模型学习流形的方法，并将其用于解决逆问题，结果表现出良好的性能，可用于模糊和电阻抗层析成像。

    

    在实践中，使用生成模型表示高维数据的流形已被证明具有计算效率。然而，这要求数据流形具有全局参数化。为了表示任意拓扑的流形，我们提出了学习变分自编码器的混合模型。这里，每个编码器-解码器对表示流形的一个图表。我们提出了一种损失函数来最大化似然估计模型权重，并选择一个架构，为我们提供图表及其逆的解析表达式。一旦学习了流形，我们将其用于通过将数据拟合项限制在学习的流形上来解决逆问题。为了解决所产生的最小化问题，我们在学习的流形上提出了一种黎曼梯度下降算法。我们展示了我们的方法在低维玩具例子以及模糊和电阻抗层析成像方面的性能。

    Representing a manifold of very high-dimensional data with generative models has been shown to be computationally efficient in practice. However, this requires that the data manifold admits a global parameterization. In order to represent manifolds of arbitrary topology, we propose to learn a mixture model of variational autoencoders. Here, every encoder-decoder pair represents one chart of a manifold. We propose a loss function for maximum likelihood estimation of the model weights and choose an architecture that provides us the analytical expression of the charts and of their inverses. Once the manifold is learned, we use it for solving inverse problems by minimizing a data fidelity term restricted to the learned manifold. To solve the arising minimization problem we propose a Riemannian gradient descent algorithm on the learned manifold. We demonstrate the performance of our method for low-dimensional toy examples as well as for deblurring and electrical impedance tomography on cert
    
[^53]: 利用特权信息进行无监督领域自适应

    Unsupervised domain adaptation by learning using privileged information. (arXiv:2303.09350v1 [cs.LG])

    [http://arxiv.org/abs/2303.09350](http://arxiv.org/abs/2303.09350)

    本文提出利用特权信息进行领域适应（DALUPI）算法，以在学习中放宽假设条件并提高样本效率，通过减少错误来促进医学图像分析等应用的发展。

    

    成功的无监督领域自适应（UDA）只在强假设条件下得以实现，如协变量移位和输入领域之间的重叠。后者在高维应用中经常被违反，比如图像分类，在面对这种挑战时，图像分类仍然是算法开发的灵感和基准。本文表明，获取源域和目标域样本的有关信息能够帮助放宽这些假设，并在学习中提高样本效率，代价是收集更丰富的变量集。我们称之为利用特权信息进行领域适应（DALUPI）。为此，我们提出了一个简单的两阶段学习算法，并提出了一个针对多标签图像分类的实用端到端算法，受到我们分析的启发。通过一系列实验，包括医学图像分析的应用，我们证明了在学习过程中加入特权信息可以减少错误。

    Successful unsupervised domain adaptation (UDA) is guaranteed only under strong assumptions such as covariate shift and overlap between input domains. The latter is often violated in high-dimensional applications such as image classification which, despite this challenge, continues to serve as inspiration and benchmark for algorithm development. In this work, we show that access to side information about examples from the source and target domains can help relax these assumptions and increase sample efficiency in learning, at the cost of collecting a richer variable set. We call this domain adaptation by learning using privileged information (DALUPI). Tailored for this task, we propose a simple two-stage learning algorithm inspired by our analysis and a practical end-to-end algorithm for multi-label image classification. In a suite of experiments, including an application to medical image analysis, we demonstrate that incorporating privileged information in learning can reduce errors i
    
[^54]: 认证悖论: 认证会揭示更好的攻击

    The Certification Paradox: Certifications Admit Better Attacks. (arXiv:2302.04379v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.04379](http://arxiv.org/abs/2302.04379)

    本文指出了一个"认证悖论"，认证虽然可以展示模型的稳健性，但额外揭示了有关认证模型的信息也成为新的攻击面，导致更好的攻击效果。

    

    在保证有一个有界区域内不存在对抗样本的情况下，认证机制在展示神经网络的稳健性方面扮演着重要角色。本文提出了一个问题: 认证是否会有任何意想不到的后果，通过揭示有关认证模型的额外信息？我们以肯定的答案回答了这个问题，证明了认证不仅测量模型的稳健性，而且展现了新的攻击面。我们提出了"认证感知攻击"，在针对经过认证的模型进行攻击时，这种攻击会比以前的任何方法更频繁地产生更小的对抗性扰动。我们的攻击实现了最多34%的扰动规范中位数的减小(比较目标和攻击实例)，同时需要的计算时间比PGD等方法少了90%。我们的攻击实现了如此显着的扰动大小和计算成本的降低，突显了以认证作为对抗攻击防御的一种悖论。具体来说，认证不仅揭示了稳健模型的属性，而且还可以用来发起更有效的攻击。

    In guaranteeing that no adversarial examples exist within a bounded region, certification mechanisms play an important role in demonstrating the robustness of neural networks. In this work we ask: Could certifications have any unintended consequences, through exposing additional information about certified models? We answer this question in the affirmative, demonstrating that certifications not only measure model robustness but also present a new attack surface. We propose \emph{Certification Aware Attacks}, that produce smaller adversarial perturbations more than twice as frequently as any prior approach, when launched against certified models. Our attacks achieve an up to $34\%$ reduction in the median perturbation norm (comparing target and attack instances), while requiring $90 \%$ less computational time than approaches like PGD. That our attacks achieve such significant reductions in perturbation size and computational cost highlights an apparent paradox in deploying certificatio
    
[^55]: 带有预算和ROI约束的自动出价算法：效率、后悔和节奏动态

    Autobidders with Budget and ROI Constraints: Efficiency, Regret, and Pacing Dynamics. (arXiv:2301.13306v2 [cs.GT] UPDATED)

    [http://arxiv.org/abs/2301.13306](http://arxiv.org/abs/2301.13306)

    本文提出了一个基于梯度的学习算法，可以在多种拍卖方式下满足预算和ROI约束，并达到个体后悔逐渐减小；结果表明，当各自竞争时，期望资金流动至少达到最优分配的期望流动的一半。

    

    我们研究了自动出价算法在在线广告平台上进行博弈的情况。每个自动出价算法被赋予任务，在多轮重复拍卖中，最大化其广告主的总价值，同时受到预算和/或投资回报率约束。我们提出了一种基于梯度的学习算法，它可以保证满足所有约束条件，并达到逐渐减小的个体后悔。我们的算法仅使用自助反馈，并可与第一或第二价格拍卖以及任何“中间”拍卖格式一起使用。我们的主要结果是，当这些自动出价算法相互竞争时，所有轮次的期望资金流动 welfare 都至少达到了任何分配所实现的期望最优流动 welfare 的一半。这在出价动态是否收敛到均衡以及广告主估值之间的相关结构如何不同的情况下均成立。

    We study a game between autobidding algorithms that compete in an online advertising platform. Each autobidder is tasked with maximizing its advertiser's total value over multiple rounds of a repeated auction, subject to budget and/or return-on-investment constraints. We propose a gradient-based learning algorithm that is guaranteed to satisfy all constraints and achieves vanishing individual regret. Our algorithm uses only bandit feedback and can be used with the first- or second-price auction, as well as with any "intermediate" auction format. Our main result is that when these autobidders play against each other, the resulting expected liquid welfare over all rounds is at least half of the expected optimal liquid welfare achieved by any allocation. This holds whether or not the bidding dynamics converges to an equilibrium and regardless of the correlation structure between advertiser valuations.
    
[^56]: 演员评论或评论演员？两个时间尺度的故事。

    Actor-Critic or Critic-Actor? A Tale of Two Time Scales. (arXiv:2210.04470v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.04470](http://arxiv.org/abs/2210.04470)

    这篇论文提出了一种评论演员算法，它在快速和慢速时间尺度上计算价值函数和策略，该算法与演员评论算法在准确性和计算成本方面表现相当。

    

    我们重新审视基于表格的演员评论算法的标准公式，将其视为两个时间尺度的随机逼近，其中价值函数在快速时间尺度上计算，策略在慢速时间尺度上计算。这模拟了策略迭代。我们首先观察到，时间尺度的反转实际上会模拟值迭代，并且是一种合法的算法。我们提供了收敛性证明，并通过带有线性和非线性函数逼近器的函数逼近测试两种方法，并观察到我们提出的评论演员算法在准确性和计算成本方面与演员评论算法相当。

    We revisit the standard formulation of tabular actor-critic algorithm as a two time-scale stochastic approximation with value function computed on a faster time-scale and policy computed on a slower time-scale. This emulates policy iteration. We begin by observing that reversal of the time scales will in fact emulate value iteration and is a legitimate algorithm. We provide a proof of convergence and compare the two empirically with and without function approximation (with both linear and nonlinear function approximators) and observe that our proposed critic-actor algorithm performs on par with actor-critic in terms of both accuracy and computational effort.
    
[^57]: 学习线性二次高斯系统的遗憾下限

    Regret Lower Bounds for Learning Linear Quadratic Gaussian Systems. (arXiv:2201.01680v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2201.01680](http://arxiv.org/abs/2201.01680)

    本论文证明了在学习未知线性高斯系统与二次代价时，存在遗憾下限，并且这个下限的比例尺度级别为 $\sqrt{T}$。通过对控制理论参数的准确捕捉，我们证明难以控制的系统也难以学习控制。同样地，对于一类部分观察到的系统，我们的结果表明了具有较差可观测结构的系统也难以学习控制。

    

    我们为自适应控制未知的线性高斯系统与二次代价建立遗憾下限。我们结合了实验设计、估计理论和某些信息矩阵的扰动界限的思想，得到了关于时间跨度$T$的遗憾下限，其比例尺度级别为 $\sqrt{T}$。我们的下限准确地捕捉了控制理论参数的作用，并且我们能够表明难以控制的系统也难以学习控制；当具体化为状态反馈系统时，我们恢复了早期工作的维度依赖关系，但改善了随系统理论常数（如系统成本和格拉米恩矩阵）的比例尺度。此外，我们将结果扩展到一类部分观察到的系统，并证明具有较差可观测结构的系统也难以学习控制。

    TWe establish regret lower bounds for adaptively controlling an unknown linear Gaussian system with quadratic costs. We combine ideas from experiment design, estimation theory and a perturbation bound of certain information matrices to derive regret lower bounds exhibiting scaling on the order of magnitude $\sqrt{T}$ in the time horizon $T$. Our bounds accurately capture the role of control-theoretic parameters and we are able to show that systems that are hard to control are also hard to learn to control; when instantiated to state feedback systems we recover the dimensional dependency of earlier work but with improved scaling with system-theoretic constants such as system costs and Gramians. Furthermore, we extend our results to a class of partially observed systems and demonstrate that systems with poor observability structure also are hard to learn to control.
    
[^58]: 分层相关聚类和维持树结构嵌入

    Hierarchical Correlation Clustering and Tree Preserving Embedding. (arXiv:2002.07756v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2002.07756](http://arxiv.org/abs/2002.07756)

    本文提出了一种分层相关聚类方法，可应用于正负配对不相似度，并研究了使用此方法进行无监督表征学习的方法。

    

    我们提出了一种分层相关聚类方法，扩展了著名的相关聚类方法，可以产生适用于正负配对不相似度的分层聚类。接下来，我们研究了使用这种分层相关聚类的无监督表征学习。为此，我们首先研究将相应的分层嵌入用于维持树结构嵌入和特征提取。然后，我们研究了最小最大距离度量扩展到相关聚类的方法，作为另一种表征学习范式。最后，我们在多个数据集上展示了我们方法的性能。

    We propose a hierarchical correlation clustering method that extends the well-known correlation clustering to produce hierarchical clusters applicable to both positive and negative pairwise dissimilarities. Then, in the following, we study unsupervised representation learning with such hierarchical correlation clustering. For this purpose, we first investigate embedding the respective hierarchy to be used for tree-preserving embedding and feature extraction. Thereafter, we study the extension of minimax distance measures to correlation clustering, as another representation learning paradigm. Finally, we demonstrate the performance of our methods on several datasets.
    

