# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Steered Diffusion: A Generalized Framework for Plug-and-Play Conditional Image Synthesis.](http://arxiv.org/abs/2310.00224) | Steered Diffusion是一个通用的框架，利用近期基于扩散的生成模型的细粒度生成控制能力，实现了零样本条件图像生成的高质量合成。 |
| [^2] | [Beyond Random Noise: Insights on Anonymization Strategies from a Latent Bandit Study.](http://arxiv.org/abs/2310.00221) | 本文通过使用隐性强盗设置和不同的聚合策略，评估了隐私和推荐器性能之间的权衡，为定制隐私技术的需求提供了洞察。研究结果表明，对个体用户的数据记录添加拉普拉斯机制的噪声是不合适的选择，它在任何噪声水平下都会产生最大的遗憾。 |
| [^3] | [Pairwise Proximal Policy Optimization: Harnessing Relative Feedback for LLM Alignment.](http://arxiv.org/abs/2310.00212) | 该论文提出了一种新的强化学习框架，使用相对反馈来调整大型语言模型（LLMs）的行为，解决了现有方法在优化比较损失训练的奖励时存在的限制。同时，还提出了一种新的基于轨迹的策略梯度算法（PPPO），用于更有效地进行算法设计和函数逼近。 |
| [^4] | [A Prefrontal Cortex-inspired Architecture for Planning in Large Language Models.](http://arxiv.org/abs/2310.00194) | 这个论文提出了一个受前额叶皮层启发的大型语言模型规划架构，利用多个基于LLM的模块实现规划的自主协调，从而在处理需要多步推理或目标导向规划的任务时取得了较好的效果。 |
| [^5] | [On the Equivalence of Graph Convolution and Mixup.](http://arxiv.org/abs/2310.00183) | 这项研究发现，在两个温和的条件下，图卷积可以被视为Mixup的一种特殊形式，它在训练和测试阶段都被应用。 |
| [^6] | [Motif: Intrinsic Motivation from Artificial Intelligence Feedback.](http://arxiv.org/abs/2310.00166) | 本文提出了一种名为Motif的方法，通过与大型语言模型（LLM）交互来获得先验知识，并将其用于代理程序的强化学习训练。实验证明，Motif的内在奖励相比直接最大化得分的算法在挑战性游戏中获得了更高的游戏得分，并在之前没有取得进展的任务上取得了显著的进展。 |
| [^7] | [Detection-Oriented Image-Text Pretraining for Open-Vocabulary Detection.](http://arxiv.org/abs/2310.00161) | 这项研究提出了一种面向检测的图像-文本预训练方法，旨在弥合图像级预训练和开放词汇目标检测之间的差距。通过检测器架构和对比损失，该方法能够从噪声图像-文本对中学习到新出现的物体-语义线索，并提出了一种平移窗口学习方法来改进主干网络的表示。在LVIS开放词汇检测基准上，该方法取得了显著优于其他方法的40.4的掩码AP$_r$结果。 |
| [^8] | [Self-Specialization: Uncovering Latent Expertise within Large Language Models.](http://arxiv.org/abs/2310.00160) | 该论文研究了大型语言模型的自我特化，通过使用专业领域的数据和少量标记种子进行自我对齐，提高了在目标领域的零样本和少样本性能。 |
| [^9] | [Feedback-guided Data Synthesis for Imbalanced Classification.](http://arxiv.org/abs/2310.00158) | 本论文介绍了一种反馈引导数据合成的方法，通过从分类器到生成模型的反馈来驱动采样，将静态数据集增强为包含有用的合成样本，以提高分类器的性能。 |
| [^10] | [Learning Generalizable Tool-use Skills through Trajectory Generation.](http://arxiv.org/abs/2310.00156) | 通过轨迹生成，我们提出了一种学习通用工具使用技能的方法，可以适应不同形状的工具，从而使自主系统能够处理复杂的可变形物体操作任务。 |
| [^11] | [Primal-Dual Continual Learning: Stability and Plasticity through Lagrange Multipliers.](http://arxiv.org/abs/2310.00154) | 本文提出了原始-对偶持续学习方法，通过利用拉格朗日对偶解决受限学习问题，实现了稳定性和可塑性。作者通过分析任务层面和样本层面的约束，在基于记忆的方法中分配资源，取得了较好的效果。 |
| [^12] | [3D Reconstruction in Noisy Agricultural Environments: A Bayesian Optimization Perspective for View Planning.](http://arxiv.org/abs/2310.00145) | 本论文提出了一种在噪声环境中进行3D重建的新方法，通过观测规划，合理选择相机位置并考虑噪声对重建性能的影响，提高了3D重建结果的质量和效率。 |
| [^13] | [Probabilistic Sampling-Enhanced Temporal-Spatial GCN: A Scalable Framework for Transaction Anomaly Detection in Ethereum Networks.](http://arxiv.org/abs/2310.00144) | 该研究提出了一种基于概率采样增强的时空图卷积网络框架，用于以太坊网络中的交易异常检测。通过将图卷积网络与时态随机游走相结合，利用时间序列的复杂性提供更精细的交易异常检测机制。实验结果表明，与传统的图卷积网络相比，该框架在检测异常和交易突发方面有显著的性能提升。这项研究强调了以太坊交易数据中时间线索的潜力，并展示了使用该框架进行交易异常检测的可行性。 |
| [^14] | [GASS: Generalizing Audio Source Separation with Large-scale Data.](http://arxiv.org/abs/2310.00140) | 本文研究了一种使用大规模数据进行音频源分离的通用方法（GASS），在有限分布范围内表现出良好的效果，并展示了其在声音事件和语音分离方面的泛化能力。然而，在分离超出分布的电影和音乐内容方面仍存在挑战。 |
| [^15] | [ABScribe: Rapid Exploration of Multiple Writing Variations in Human-AI Co-Writing Tasks using Large Language Models.](http://arxiv.org/abs/2310.00117) | ABScribe是一种界面，支持在人工智能与人类共同写作任务中快速探索多种写作变化。用户可以使用大型语言模型提示快速生成多个变体，这些变体以可重用的按钮形式呈现，并且可以通过上下文工具栏进行快速的就地比较。 |
| [^16] | [Certified Robustness via Dynamic Margin Maximization and Improved Lipschitz Regularization.](http://arxiv.org/abs/2310.00116) | 本文提出了一种基于动态边界最大化和改进的Lipschitz正则化的认证鲁棒性训练算法，通过增加输出空间中的边界和正则化模型的Lipschitz常数来提高深度分类器对抗性扰动的鲁棒性。 |
| [^17] | [HyperMask: Adaptive Hypernetwork-based Masks for Continual Learning.](http://arxiv.org/abs/2310.00113) | HyperMask是一种用于持续学习的方法，它使用基于超网络的掩码来训练一个单一网络，以克服人工神经网络在多任务上的灾难性遗忘问题。 |
| [^18] | [FashionFlow: Leveraging Diffusion Models for Dynamic Fashion Video Synthesis from Static Imagery.](http://arxiv.org/abs/2310.00106) | 本研究提出了一种名为FashionFlow的图像到视频生成器，利用扩散模型从静态图像生成短视频。我们通过开发并连接与扩散模型相关的组件来实现这一目标，其中包括使用伪3D卷积层高效生成视频，并利用VAE和CLIP编码器捕捉关键特征。研究结果展示了成功合成时尚视频的能力，能够展示服装的合身度和外观，为在线时尚行业的购物体验提供改进和增强的潜力。 |
| [^19] | [Multilingual Natural Language ProcessingModel for Radiology Reports -- The Summary is all you need!.](http://arxiv.org/abs/2310.00100) | 本研究通过在多语言文本到文本变换器模型上微调，开发了一个能够自动在多语言中总结放射学报告的模型。该模型有助于提高未来深度学习模型的研究和发展，且能够应用于不同族裔背景的患者数据。 |
| [^20] | [Voice2Action: Language Models as Agent for Efficient Real-Time Interaction in Virtual Reality.](http://arxiv.org/abs/2310.00092) | 本研究提出了Voice2Action，一种使用语言模型作为代理人在虚拟现实中进行高效实时交互的框架。通过对定制语音信号和文本命令进行分层分析，并将执行任务分成交互子集，Voice2Action能够比其他方法更高效和准确地执行。 |
| [^21] | [SocREval: Large Language Models with the Socratic Method for Reference-Free Reasoning Evaluation.](http://arxiv.org/abs/2310.00074) | 本论文提出了一种称为SocREval的方法，利用GPT-4和苏格拉底方法进行无参考推理评估，以解决当前复杂推理模型评估中遇到的挑战。 |
| [^22] | [Emotional Listener Portrait: Realistic Listener Motion Simulation in Conversation.](http://arxiv.org/abs/2310.00068) | 本论文提出了一种情感听众肖像（ELP）模型，采用了显式离散设计，能根据对话中不同情绪生成自然多样又可控的响应，解决了面部表情生成中的非确定性问题。 |
| [^23] | [AI ensemble for signal detection of higher order gravitational wave modes of quasi-circular, spinning, non-precessing binary black hole mergers.](http://arxiv.org/abs/2310.00052) | 本研究提出了使用AI集成同时处理双重LIGO探测器和Virgo探测器数据的模型，成功训练出能够探测秩序更高的引力波模式的AI分类器，并使用迁移学习估计了潜在二进制黑洞的总质量。通过实验验证了该集成在处理大量信号时的性能。 |
| [^24] | [LoRA ensembles for large language model fine-tuning.](http://arxiv.org/abs/2310.00035) | 本文提出了一种使用低秩适配器（LoRA）的集成方法，用于解决大型语言模型微调中存在的不确定性量化问题，并提供了一个参数高效的微调技术。这种方法可以构建大规模的LoRA适配器集成，并具有与基础预训练模型相近的计算资源需求。 |
| [^25] | [PB-LLM: Partially Binarized Large Language Models.](http://arxiv.org/abs/2310.00034) | 本文提出的PB-LLM是一种部分二值化的大型语言模型压缩方法，可以在保持语言推理能力的同时实现极低比特量化，并通过后训练量化和量化感知训练等方法恢复量化LLMM的容量。 |
| [^26] | [Adversarial Driving Behavior Generation Incorporating Human Risk Cognition for Autonomous Vehicle Evaluation.](http://arxiv.org/abs/2310.00029) | 本论文开发了一种新型框架，结合人类风险认知来生成对手驾驶行为，用于评估自动驾驶车辆的有效性和弱点。 |
| [^27] | [De-SaTE: Denoising Self-attention Transformer Encoders for Li-ion Battery Health Prognostics.](http://arxiv.org/abs/2310.00023) | 本研究提出了De-SaTE方法，通过利用多个去噪模块以及自注意力变换编码器，准确预测锂离子电池的剩余寿命（RUL），为预防性维护和预测性分析提供关键指标估计。 |
| [^28] | [Adaptive Communications in Collaborative Perception with Domain Alignment for Autonomous Driving.](http://arxiv.org/abs/2310.00013) | 这篇论文提出了一个通信的协同感知框架ACC-DA，通过动态调整通信图和自适应数据重构机制来增强自动驾驶中的感知能力。 |
| [^29] | [Operator-free Equilibrium on the Sphere.](http://arxiv.org/abs/2310.00012) | 本论文通过引入一个新的准则，建立连续的、可导的核函数，简化了在球面上等分点集的计算，实现了在无需涉及运算符的情况下对潜在点系统进行探索，并得到了与蒙特卡洛方法相比更高效的近似目标的方法。 |
| [^30] | [Artificial Empathy Classification: A Survey of Deep Learning Techniques, Datasets, and Evaluation Scales.](http://arxiv.org/abs/2310.00010) | 这篇论文综述了人工共情的分类研究，介绍了深度学习技术、数据集和评估标准的最新进展，指出训练人工共情的标准流程包括情绪识别、分析和响应动作。其中深度学习技术在虚拟代理和机器人中的应用有较高影响力。 |
| [^31] | [LLM-grounded Video Diffusion Models.](http://arxiv.org/abs/2309.17444) | 使用LLM-grounded Video Diffusion (LVD)模型，通过先生成动态场景布局，再通过这些布局指导视频生成的扩散模型，解决了当前模型在复杂的时空提示和不正确的运动生成方面的困难。 |
| [^32] | [Learning Decentralized Flocking Controllers with Spatio-Temporal Graph Neural Network.](http://arxiv.org/abs/2309.17437) | 本论文提出了一种名为STGNN的时空图神经网络，该网络在去中心化群集控制中通过结合空间和时间扩展来更好地模拟集中式控制策略，从而提高了预测的效果和准确性。 |
| [^33] | [Data Filtering Networks.](http://arxiv.org/abs/2309.17425) | 本文研究了学习数据过滤网络用于筛选大型未策划数据集的问题，并构建了新的数据过滤网络，从而产生最先进的图像-文本数据集。 |
| [^34] | [PlaceNav: Topological Navigation through Place Recognition.](http://arxiv.org/abs/2309.17260) | PlaceNav是一种通过地点识别进行拓扑导航的方法，将机器人无关部分分为导航特定和通用的计算机视觉组件，通过使用非机器人来源的大规模数据集增加训练数据的可用性，同时通过地点识别来提高导航性能。新模型的性能提高了76%。 |
| [^35] | [Can LLMs Effectively Leverage Structural Information for Graph Learning: When and Why.](http://arxiv.org/abs/2309.16595) | 本文研究了大型语言模型（LLM）在图数据中的应用，发现LLM可以从结构信息中受益，尤其是在文本节点特征缺乏的情况下，而LLM的性能与数据泄露没有显著相关。 |
| [^36] | [Cooperation Dynamics in Multi-Agent Systems: Exploring Game-Theoretic Scenarios with Mean-Field Equilibria.](http://arxiv.org/abs/2309.16263) | 本文研究在多智能体系统中激发合作的策略和方法，通过分析现有的合作策略和引入鼓励团队回报的修改，解决了在分布式系统中存在的现实困境。同时，利用均值场博弈理论，建立了无限大智能体集合中的平衡解和奖励结构。 |
| [^37] | [Lyra: Orchestrating Dual Correction in Automated Theorem Proving.](http://arxiv.org/abs/2309.15806) | Lyra是一种新的框架，通过引入工具修正和猜想修正两种机制，增强了大规模语言模型在形式化定理证明领域的有效性，减轻了幻觉，并提高了证明的准确性。 |
| [^38] | [Class Incremental Learning via Likelihood Ratio Based Task Prediction.](http://arxiv.org/abs/2309.15048) | 该论文提出了一种基于似然比的任务预测的类增量学习方法，利用离群检测器进行任务标识预测，解决了无任务标识符的测试样本的任务预测问题。 |
| [^39] | [Are Human-generated Demonstrations Necessary for In-context Learning?.](http://arxiv.org/abs/2309.14681) | 本文研究了上下文学习中人工生成的演示是否有必要，并提出了一种新的自反思提示策略（SEC），通过这种策略，大型语言模型（LLMs）可以自行生成演示和最终输出，避免了手动生成过程的复杂性。 |
| [^40] | [Joint Audio and Speech Understanding.](http://arxiv.org/abs/2309.14405) | LTU-AS是一个具有普适音频感知和高级推理能力的机器学习模型，可以同时识别和联合理解口语文本、语音声音学和非语音音频事件。 |
| [^41] | [Physics of Language Models: Part 3.2, Knowledge Manipulation.](http://arxiv.org/abs/2309.14402) | 本文研究了语言模型在推理过程中操控知识的能力，发现预训练模型在知识检索方面表现出色，但在简单的分类、比较和逆向搜索任务中表现不佳。作者还提供了一个合成数据集进行实验，验证了这些内在的弱点：语言模型无法高效地操控知识。 |
| [^42] | [LinGCN: Structural Linearized Graph Convolutional Network for Homomorphically Encrypted Inference.](http://arxiv.org/abs/2309.14331) | LinGCN是一个旨在减少乘法深度和优化HE基于GCN推断性能的框架，通过结构化线性化算法和参数化的离散指示函数的联合训练，实现细粒度的节点级非线性位置选择。 |
| [^43] | [Skill Check: Some Considerations on the Evaluation of Gamemastering Models for Role-playing Games.](http://arxiv.org/abs/2309.13702) | 本文讨论了从交互式叙事和自然语言处理的角度对角色扮演游戏中游戏主持进行建模的挑战，并提出了三个测试类别来评估对话系统。 |
| [^44] | [A Model-Agnostic Graph Neural Network for Integrating Local and Global Information.](http://arxiv.org/abs/2309.13459) | MaGNet是一种模型无关的图神经网络框架，能够顺序地整合不同顺序的信息，并通过识别有影响力的紧凑图结构提供有意义且可解释的结果。 |
| [^45] | [State-space Models with Layer-wise Nonlinearity are Universal Approximators with Exponential Decaying Memory.](http://arxiv.org/abs/2309.13414) | 本论文证明了堆叠具有逐层非线性激活的状态空间模型足以逼近任何连续的序列到序列关系，并且发现其加强了模型学习复杂序列模式的能力。然而，状态空间模型并不能根本解决指数衰减记忆的问题。 |
| [^46] | [Exploring the Impact of Training Data Distribution and Subword Tokenization on Gender Bias in Machine Translation.](http://arxiv.org/abs/2309.12491) | 这项研究探索了训练数据分布和子词标记对机器翻译中性别偏见的影响。研究发现，模型训练语料库中性别形式的不平衡是导致性别偏见的主要因素，而子词拆分的影响较小。同时，研究还发现，通过分析子词拆分可以很好地估计训练数据中性别形式的不平衡。最后，通过仅微调标记嵌入层可以减少女性和男性之间性别预测准确性的差距。 |
| [^47] | [SAVME: Efficient Safety Validation for Autonomous Systems Using Meta-Learning.](http://arxiv.org/abs/2309.12474) | 本论文提出了一种使用元学习进行高效安全验证的方法，通过集成贝叶斯方法和多臂赌博机框架，加速验证过程。该方法学习在测试中容易引发故障的场景参数分布和能够进行快速准确模拟的保真度设置分布，并通过评估保真度设置分布是否有助于对新场景的场景参数分布进行更快的学习来进一步提高效率。 |
| [^48] | [LMSYS-Chat-1M: A Large-Scale Real-World LLM Conversation Dataset.](http://arxiv.org/abs/2309.11998) | LMSYS-Chat-1M是一个包含一百万个实际对话的大规模数据集，通过其多样性和用例展示了其在理解和推进LLM能力方面的价值。 |
| [^49] | [Design of Chain-of-Thought in Math Problem Solving.](http://arxiv.org/abs/2309.11054) | 本论文研究了数学问题解决中思路链的设计方法，对比了自然语言思路链和程序思路链的效果，并发现程序思路链通常在数学问题解决中更加有效，特别是自我描述程序具有更大多样性且性能更高。此外，研究还发现Python是程序思路链的较好选择。实验结果为未来思路链设计提供了宝贵指导。 |
| [^50] | ["With Great Power Comes Great Responsibility!": Student and Instructor Perspectives on the influence of LLMs on Undergraduate Engineering Education.](http://arxiv.org/abs/2309.10694) | 本文通过调查和访谈，填补了关于LLMs在本科工程教育中的使用和观点的研究空白，为学生和教师对LLMs的采用提供了洞见和建议。 |
| [^51] | [Multi-Agent Deep Reinforcement Learning for Cooperative and Competitive Autonomous Vehicles using AutoDRIVE Ecosystem.](http://arxiv.org/abs/2309.10007) | 本研究提出了一个模块化和并行化的多智能体深度强化学习框架，在AutoDRIVE生态系统中培养合作与竞争行为。我们利用该生态系统开发了准确物理和逼真图形的数字孪生体，并使用它来训练和部署多智能体强化学习策略，实现了在自主车辆中的合作和竞争行为。 |
| [^52] | [Mechanic Maker 2.0: Reinforcement Learning for Evaluating Generated Rules.](http://arxiv.org/abs/2309.09476) | 本论文研究了将强化学习应用于游戏规则生成的人类游戏评估，并通过实验结果表明，强化学习生成的规则与传统基线方法有所不同，可能更适合人类使用。 |
| [^53] | [Landscape-Sketch-Step: An AI/ML-Based Metaheuristic for Surrogate Optimization Problems.](http://arxiv.org/abs/2309.07936) | Landscape-Sketch-Step是一种基于AI/ML的元启发式方法，结合了机器学习、随机优化和强化学习技术，用于解决成本函数评估昂贵、不可访问或禁止的代理优化问题。 |
| [^54] | [MMICL: Empowering Vision-language Model with Multi-Modal In-Context Learning.](http://arxiv.org/abs/2309.07915) | MMICL提出了一种用于视觉-语言模型的架构和训练数据设计，以解决VLM在理解复杂多模态提示方面的困难。 |
| [^55] | [Self-Refined Large Language Model as Automated Reward Function Designer for Deep Reinforcement Learning in Robotics.](http://arxiv.org/abs/2309.06687) | 提出一种自我改进机制的大型语言模型（LLM）框架用于自动化奖励函数设计，在深度强化学习中展现了潜在的应用价值。 |
| [^56] | [Zero-Shot Recommendations with Pre-Trained Large Language Models for Multimodal Nudging.](http://arxiv.org/abs/2309.01026) | 该论文提出了一种利用生成型AI领域的新技术进行零样本推荐的方法，通过将多模态输入转化为文本描述，并利用预训练的语言模型计算语义嵌入，实现了对非平稳内容的推荐。在合成的多模态暗示环境中进行实验证明了该方法的有效性。 |
| [^57] | [On the Implicit Bias of Adam.](http://arxiv.org/abs/2309.00079) | 本文证明了RMSProp和Adam存在隐式规范化作用，其取决于超参数和训练阶段，并讨论了这些证明事实对泛化的影响。 |
| [^58] | [BioCoder: A Benchmark for Bioinformatics Code Generation with Contextual Pragmatic Knowledge.](http://arxiv.org/abs/2308.16458) | BioCoder是一个用于评估预训练模型在生成生物信息学代码方面的基准，涵盖了函数代码生成中的包依赖关系、类声明和全局变量，并通过模糊测试框架进行评估。 |
| [^59] | [Escaping the Sample Trap: Fast and Accurate Epistemic Uncertainty Estimation with Pairwise-Distance Estimators.](http://arxiv.org/abs/2308.13498) | 本文介绍了使用配对距离估计器对集成模型进行认识不确定性估计的新方法，相比于常用的深度学习方法，该方法能够更快速、更准确地在更大的空间和更高维度上估计认识不确定性。 |
| [^60] | [Prompt-Based Length Controlled Generation with Reinforcement Learning.](http://arxiv.org/abs/2308.12030) | 提出了一种基于提示的长度控制方法，利用强化学习和奖励模型来实现大型语言模型（LLM）的长度受控生成。该方法可以有效减少推理成本并满足不同需求。 |
| [^61] | [Dynamic Open Vocabulary Enhanced Safe-landing with Intelligence (DOVESEI).](http://arxiv.org/abs/2308.11471) | 本文提出了一种动态开放词汇增强的智能安全着陆系统，通过利用开放词汇图像分割的能力实现无人机的视觉伺服，适应不同场景且无需大量数据积累进行模型改进，可以处理100米高度的操作。 |
| [^62] | [Evaluating the Instruction-Following Robustness of Large Language Models to Prompt Injection.](http://arxiv.org/abs/2308.10819) | 该论文提出了一个用于评估大型语言模型对注入的对抗性指令的鲁棒性的基准，旨在量化模型受到注入指令影响的程度，并评估其区分原始用户指令和注入指令的能力。 |
| [^63] | [Towards Probabilistic Causal Discovery, Inference & Explanations for Autonomous Drones in Mine Surveying Tasks.](http://arxiv.org/abs/2308.10047) | 本文针对无人机在盐矿勘测任务中面临的因果挑战，提出了一个概率因果框架，包括因果规划、在线适应和事后解释，以解决混淆变量、非平稳性和建模困难等问题。 |
| [^64] | [Ada-QPacknet -- adaptive pruning with bit width reduction as an efficient continual learning method without forgetting.](http://arxiv.org/abs/2308.07939) | Ada-QPacknet是一种自适应剪枝与位宽缩减的高效继续学习方法，通过剪枝和量化技术生成任务子网络，在动态和复杂环境中实现了与浮点数子网络相似的准确性。 |
| [^65] | [Ground Manipulator Primitive Tasks to Executable Actions using Large Language Models.](http://arxiv.org/abs/2308.06810) | 本文提出了一种利用大型语言模型将操纵器的原始任务转换为机器人的低层动作的方法，通过设计类似程序函数的提示，实现了对位置/力的设定点的生成，从而实现了混合控制 |
| [^66] | [Adv-Inpainting: Generating Natural and Transferable Adversarial Patch via Attention-guided Feature Fusion.](http://arxiv.org/abs/2308.05320) | 本文提出了一种称为Adv-Inpainting的创新攻击框架，通过注意力引导的特征融合生成自然且可迁移的对抗性贴纸，相比于传统的对抗性贴纸方法，该方法在生成图案和边界方面更加自然，并具有更强的迁移性能。 |
| [^67] | [FLIPS: Federated Learning using Intelligent Participant Selection.](http://arxiv.org/abs/2308.03901) | 本文介绍了FLIPS，这是一个用于管理联邦学习中数据和参与者异质性的中间件系统。FLIPS通过标签分布聚类和智能参与者选择，并使用可信执行环境来确保隐私保护。实证评估表明，FLIPS相比随机方法有更好的性能。 |
| [^68] | [Relation-Oriented: Toward Knowledge-Aligned Causal AI.](http://arxiv.org/abs/2307.16387) | 本研究从创新的关系导向视角出发，探讨了当前的建模范式中的观察模型与实际理解的不对齐问题，并提出了关系定义的表示学习方法作为实现关系导向建模的实践方法。 |
| [^69] | [A Real-World WebAgent with Planning, Long Context Understanding, and Program Synthesis.](http://arxiv.org/abs/2307.12856) | 这篇论文介绍了一种名为WebAgent的LLM驱动代理，通过自我经验学习，在真实网站上完成任务。该方法通过规划、总结和生成代码来提高在真实网站上的成功率。 |
| [^70] | [A decision making framework for recommended maintenance of road segments.](http://arxiv.org/abs/2307.10085) | 这项研究提出了一个决策框架，通过整合多种人工智能决策技术和历史数据，为道路管理部门提供科学决策工具和证据，以解决道路维护的问题。 |
| [^71] | [Adversarial Likelihood Estimation with One-way Flows.](http://arxiv.org/abs/2307.09882) | 本文提出了一种通过单向流进行对抗性似然估计的方法，并使用重要性采样解决了Wasserstein GAN中分区函数有偏估计的问题。同时，通过最大化生成器的熵，提高了模式覆盖效果。这种方法通过计算生成样本的密度来实现对分区函数的无偏估计和生成器熵的计算。 |
| [^72] | [Memorization Through the Lens of Curvature of Loss Function Around Samples.](http://arxiv.org/abs/2307.05831) | 本研究通过对损失函数曲率进行分析，研究了神经网络在不同样本上的泛化与记忆化特性。我们发现高曲率的样本通常是具有标签错误或冲突的长尾样本，并在CIFAR100数据集上发现了一种新的失败模型。通过对部分样本进行随机标签错误，我们展示了曲率排序可以有效识别出这些样本。 |
| [^73] | [Bridge the Performance Gap in Peak-hour Series Forecasting: The Seq2Peak Framework.](http://arxiv.org/abs/2307.01597) | 本文提出了Seq2Peak框架，针对高峰小时序列预测任务，该框架通过解决高度非平稳性和性能评估问题，成功缩小了在常规时间序列预测模型中观察到的性能差距。 |
| [^74] | [RefSAM: Efficiently Adapting Segmenting Anything Model for Referring Video Object Segmentation.](http://arxiv.org/abs/2307.00997) | 本文介绍了RefSAM模型，该模型通过在线方式从不同时间戳的多视图信息中加入SAM的潜力，探索其在指代视频对象分割（RVOS）中的应用。通过使用跨模态MLP和分层稠密注意模块，我们改进了SAM模型，实现了对不同形态的精确理解，并取得了令人印象深刻的性能表现。 |
| [^75] | [DoReMi: Grounding Language Model by Detecting and Recovering from Plan-Execution Misalignment.](http://arxiv.org/abs/2307.00329) | DoReMi是一种新颖的语言模型基础架构，通过检测和修复计划与执行之间的不一致性来实现语言模型的基础。该架构利用视觉问答模型检查约束条件以发现不一致，并调用语言模型进行重新规划以实现恢复。 |
| [^76] | [Unsupervised Polychromatic Neural Representation for CT Metal Artifact Reduction.](http://arxiv.org/abs/2306.15203) | 本文提出了一种新颖的多色彩神经表示法（Polyner），用于解决CT成像中存在金属伪影的挑战性问题。Polyner通过建模非线性反问题，准确模拟CT采集过程，并利用无监督训练的神经网络架构恢复原始物体信息。实验证明Polyner在金属伪影减少方面的有效性。 |
| [^77] | [Pointwise-in-Time Explanation for Linear Temporal Logic Rules.](http://arxiv.org/abs/2306.13956) | 本文提出了一个可以评估给定路径规划中特定时间点上的单个线性时间逻辑(LTL)约束的相关性和状态的框架，可以用于在离散时间、离散空间中执行有限计划的代理任务中，为用户提供时间点解释和规则参数状态的洞察力。 |
| [^78] | [SPRINT: Scalable Policy Pre-Training via Language Instruction Relabeling.](http://arxiv.org/abs/2306.11886) | SPRINT 提出了一种离线策略预训练方法，通过指令重标记及离线强化学习实现可扩展的预训练任务，大大减少了预训练所需的人力，同时使机器人能够获取更丰富的技能库，相较于之前的预训练方法，能够更快地学习新的长时间跨度任务。 |
| [^79] | [Textbooks Are All You Need.](http://arxiv.org/abs/2306.11644) | phi-1是一个新的大型代码语言模型，通过精心训练和优化，尽管规模相对较小，但在准确率和新的性质方面表现出了令人惊讶的结果。 |
| [^80] | [Hyperbolic Active Learning for Semantic Segmentation under Domain Shift.](http://arxiv.org/abs/2306.11180) | 这项研究首次在Poincaré双曲球模型中运用超bolic活跃学习方法，利用区域内像素嵌入的半径变化作为新的数据获取策略，以提升域转移下语义分割的性能。 |
| [^81] | [Revealing the Illusion of Joint Multimodal Understanding in VideoQA Models.](http://arxiv.org/abs/2306.08889) | 通过设计轻量级探针QUAG和替代方法QUAG-attention，发现视频问答模型在多模态理解方面存在幻象，即使在多模态损伤下仍能保持高性能，且用更少的计算量实现相似的性能。 |
| [^82] | [Mol-Instructions: A Large-Scale Biomolecular Instruction Dataset for Large Language Models.](http://arxiv.org/abs/2306.08018) | Mol-Instructions是一个专门为生物分子领域设计的综合指令数据集，可以显著提高大语言模型在生物领域中的适应能力和认知敏锐度。 |
| [^83] | [Push: Concurrent Probabilistic Programming for Bayesian Deep Learning.](http://arxiv.org/abs/2306.06528) | Push是一个并发概率编程库，用于贝叶斯深度学习（BDL），可以在多GPU硬件上执行BDL推理算法。该库通过将神经网络表示为粒子，并允许粒子之间的异步通信和各种参数更新，简化了BDL实验和扩展粒子操作的过程。 |
| [^84] | [Gode -- Integrating Biochemical Knowledge Graph into Pre-training Molecule Graph Neural Network.](http://arxiv.org/abs/2306.01631) | 本研究提出了一种新的方法，在分子结构和生物医学知识图谱中集成多个领域信息，通过自我监督策略预先训练更广泛和更强大的表示，并在化学属性预测任务上展示出出色的性能。 |
| [^85] | [Masked Autoencoders with Multi-Window Local-Global Attention Are Better Audio Learners.](http://arxiv.org/abs/2306.00561) | 多窗口本地-全局注意力的掩码自编码器（MW-MAE）在音频学习任务中表现出更好的性能和通用表示能力，并具有更好的可扩展性。 |
| [^86] | [Stable Anisotropic Regularization.](http://arxiv.org/abs/2305.19358) | 本文提出了一种新颖的正则化方法I-STAR，可以增加模型的稳定性，提高性能，并改善自然语言处理中的组合表示问题。 |
| [^87] | [How to Query Human Feedback Efficiently in RL?.](http://arxiv.org/abs/2305.18505) | 该论文提出了一种针对强化学习中人类反馈查询的有效采样方法，以在最少的人类反馈下学习最佳策略，并可应用于具有线性参数化和未知过渡的偏好模型，并引入了基于行动比较反馈的RLHF。 |
| [^88] | [C-MCTS: Safe Planning with Monte Carlo Tree Search.](http://arxiv.org/abs/2305.16209) | C-MCTS 提出了一种解决有约束的决策问题的方法，通过训练安全评判器进行成本估计，并在部署期间通过剪枝不安全轨迹来限制探索，实现了更高的奖励和更高效的规划步骤。 |
| [^89] | [Passive learning of active causal strategies in agents and language models.](http://arxiv.org/abs/2305.16183) | 通过被动学习，在智能体和语言模型中可以学习到一般化的主动因果策略，用于确定和使用因果关系结构。通过模仿专家数据进行训练的智能体能够在测试时推断和使用从未出现的因果链接，并将实验策略推广到从未观察到的新变量集。 |
| [^90] | [Self-contradictory Hallucinations of Large Language Models: Evaluation, Detection and Mitigation.](http://arxiv.org/abs/2305.15852) | 本文对大型语言模型的自相矛盾幻觉进行了评估、检测和缓解，探究了这一幻觉形式的普遍存在性。通过设计框架有效触发自相矛盾，发现不同语言模型中这种现象都频繁出现。ChatGPT和GPT-4能够准确识别自相矛盾，而Vicuna-13B则有些困难。 |
| [^91] | [Size Generalizability of Graph Neural Networks on Biological Data: Insights and Practices from the Spectral Perspective.](http://arxiv.org/abs/2305.15611) | 本文通过谱角度的方法，研究了GNNs的尺寸可泛化性问题，并在真实生物数据集上进行了实验，发现GNNs在度分布和谱分布偏移时均表现敏感，在同一数据集的大图上的性能仍然下降，揭示了 GNNs的尺寸可泛化性问题。 |
| [^92] | [STAR: Improving Low-Resource Information Extraction by Structure-to-Text Data Generation with Large Language Models.](http://arxiv.org/abs/2305.15090) | STAR是一种利用大型语言模型合成数据实例的数据生成方法，用于改进低资源信息抽取，为实际应用提供了需要最少人工标注的解决方案。 |
| [^93] | [Training Diffusion Models with Reinforcement Learning.](http://arxiv.org/abs/2305.13301) | 本文研究了利用强化学习方法直接优化扩散模型以实现下游对象的问题，并提出一种称之为去噪扩散策略优化（DDPO）的有效策略梯度算法，能够适应难以通过提示表达的图像压缩等目标，以及通过人类反馈得出的美学质量等目标。 |
| [^94] | [GraphCare: Enhancing Healthcare Predictions with Personalized Knowledge Graphs.](http://arxiv.org/abs/2305.12788) | 本论文提出了一种名为GraphCare的框架，通过使用个性化知识图谱来改进基于电子健康记录的医疗预测，并通过在两个公共数据集上的实验证明了其有效性。 |
| [^95] | [Multimodal Web Navigation with Instruction-Finetuned Foundation Models.](http://arxiv.org/abs/2305.11854) | 本文研究使用视觉语言基础模型进行数据驱动离线训练的 Web 代理，提出了一个指令跟随多模态代理WebGUM，将微调指令微调语言模型和视觉转换器，能够有效提高代理的基于视觉感知、HTML 理解和多步推理的能力。 |
| [^96] | [CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing.](http://arxiv.org/abs/2305.11738) | 本文提出了一个名为CRITIC的框架，使得大型语言模型可以通过与工具的交互校正自己的错误，从而避免生成出现不一致和问题行为的结果。 |
| [^97] | [Structural Pruning for Diffusion Models.](http://arxiv.org/abs/2305.10924) | 本文提出了一种名为Diff-Pruning的高效压缩方法，通过一个Taylor展开过程来识别重要权重，从而从预先存在的模型中学习轻量级扩散模型，性能稳定，并在训练效率上显著提高。 |
| [^98] | [Semantically Aligned Task Decomposition in Multi-Agent Reinforcement Learning.](http://arxiv.org/abs/2305.10865) | 该论文提出了一种多智能体强化学习中的新方法SAMA，通过提前训练的语言模型和任务分解来解决ASG方法存在的样本效率问题和生成非实际任务奖励的子目标的问题。 |
| [^99] | [ConvXAI: Delivering Heterogeneous AI Explanations via Conversations to Support Human-AI Scientific Writing.](http://arxiv.org/abs/2305.09770) | ConvXAI是一个基于对话的XAI系统，它集成了多种XAI类型，并将实际用户需求嵌入设计中，以提高实用性。 |
| [^100] | [Assessing Working Memory Capacity of ChatGPT.](http://arxiv.org/abs/2305.03731) | 本文评估了最先进语言模型ChatGPT的工作记忆容量，结果显示其在N-back任务的行为表现与人类参与者相似，这为设计具有人类级认知能力的人工智能系统提供了关键洞察。 |
| [^101] | [High-dimensional Bayesian Optimization via Semi-supervised Learning with Optimized Unlabeled Data Sampling.](http://arxiv.org/abs/2305.02614) | 本文提出基于半监督学习的高维贝叶斯优化方法，利用特定的未标记数据采样、参数化采样分布的优化及动态选择无标记数据等策略，解决了高维贝叶斯优化难以处理的问题。 |
| [^102] | [Differentially Private In-Context Learning.](http://arxiv.org/abs/2305.01639) | 本文提出了DP-ICL，实现了在隐私保证下对新任务的适应性。经过四个基准测试，发现其性能与非私有ICL相当。 |
| [^103] | [Lightweight, Pre-trained Transformers for Remote Sensing Timeseries.](http://arxiv.org/abs/2304.14065) | 设计针对远程传感器数据的自监督学习模型和训练技术，可以得到表现更好且更小的模型。预训练的遥感时间序列Transformer（Presto）在几个遥感基准测试中实现了最先进的结果。 |
| [^104] | [Sample-efficient Model-based Reinforcement Learning for Quantum Control.](http://arxiv.org/abs/2304.09718) | 本论文提出了一种基于模型的强化学习方法，通过受到神经常微分方程进展的启发，这个方法采用自动微分的ODE表达由可学习的汉密尔顿安排参数化的模型来近似环境，在门控制和汉密尔顿参数的学习中通过系统交互解决问题。该方法在样本复杂度方面比标准基于模型自由的强化学习方法具有一个数量级的优势，适用于噪声时变门优化。 |
| [^105] | [Precise localization of corneal reflections in eye images using deep learning trained on synthetic data.](http://arxiv.org/abs/2304.05673) | 该论文提出了一种使用深度学习在眼部图像中准确定位角膜反射的方法，无需对真实眼部图像进行注释，仅使用模拟数据进行训练，该方法表现出色且提供了一种可行的解决方案。 |
| [^106] | [A review of ensemble learning and data augmentation models for class imbalanced problems: combination, implementation and evaluation.](http://arxiv.org/abs/2304.02858) | 本文研究了集成学习和数据增强方法的应用，针对类别不平衡问题，通过计算评估，找到了最有效的组合。 |
| [^107] | [Subject-driven Text-to-Image Generation via Apprenticeship Learning.](http://arxiv.org/abs/2304.00186) | 该论文提出了一种基于徒弟学习的面向主题的文本到图像生成器SuTI，能够通过将大量基于主题的专家模型的数据输入徒弟模型，学习并推断出新主题的最佳专家模型，从而生成高品质的自定义图像，且速度比传统方法更快。 |
| [^108] | [Analyzing the Contextual Shortcomings of Artificial General Intelligence.](http://arxiv.org/abs/2304.00002) | 论文讨论了人工智能(AI)专家对于人工通用智能(AGI)的决策所需技能并不了解的问题。 虽然当前的机器可以模拟特定的人类属性，但 AGI 是在这样的前提下开发出来的：这可以很容易地扩展到一般智能水平。这会分散当前研究的注意力，远离相关问题。 |
| [^109] | [Efficient Deep Learning of Robust, Adaptive Policies using Tube MPC-Guided Data Augmentation.](http://arxiv.org/abs/2303.15688) | 本论文提出了一种高效的深度学习算法，可以学习具有鲁棒性和自适应能力的策略，通过引导数据增强，使用修改后的IL过程，并在学习适应性位置和姿态控制策略方面进行应用。 |
| [^110] | [Policy Optimization for Personalized Interventions in Behavioral Health.](http://arxiv.org/abs/2303.12206) | 研究如何通过数字平台传递的行为健康介入最大化健康结果和治疗成本，提出了一个名为DecompPI的新算法，从离线数据进行预测任务，减轻了在线实验的需要，并在理论上证明了该算法的可扩展性和渐近收敛性。 |
| [^111] | [IFAN: An Explainability-Focused Interaction Framework for Humans and NLP Models.](http://arxiv.org/abs/2303.03124) | IFAN是一个面向人类和NLP模型的可解释性交互框架，通过用户的实时反馈和适配器层的对齐，有效地减轻了偏见的仇恨言论分类器。 |
| [^112] | [EvoPrompting: Language Models for Code-Level Neural Architecture Search.](http://arxiv.org/abs/2302.14838) | EvoPrompting利用语言模型作为自适应变异和交叉操作符来进行神经架构搜索，在MNIST-1D数据集和CLRS算法推理基准上都取得了比人类设计的架构更好的性能表现。 |
| [^113] | [Construction numbers: How to build a graph?.](http://arxiv.org/abs/2302.13186) | 论文研究了计算偏序的线性扩展数量问题，并研究了由包含关系确定的图形的顶点和边的偏序，找到了路径、环、星形图、双星形图和完全图的构造序列数量，并提出了公式，同时研究了结构和应用。 |
| [^114] | [HUST bearing: a practical dataset for ball bearing fault diagnosis.](http://arxiv.org/abs/2302.12533) | HUST轴承是一个实用的球轴承故障诊断数据集，其中包含90个带有6种故障类型（内部裂纹、外部裂纹、球体裂纹和它们的2种组合）的5种不同类型轴承的振动数据。研究者使用经典机器学习分类方法以及先进的非监督迁移学习算法对该数据集进行了评估，实验结果表明在分类任务上准确率可达到100%，在非监督迁移学习任务上准确率为60-80%。 |
| [^115] | [Long Horizon Temperature Scaling.](http://arxiv.org/abs/2302.03686) | 提出了一种长时间尺度温度缩放（LHTS）方法，用于从温度缩放的联合分布中采样。LHTS可以优化样本的长时间尺度似然，并且在图像扩散模型和字符/语言自回归模型上展示了优势。 |
| [^116] | [Domain-Agnostic Molecular Generation with Self-feedback.](http://arxiv.org/abs/2301.11259) | MolGen是一个专注于分子生成的预训练语言模型，使用了领域无关的分子前缀调整和自我反馈的范式，实现了化学有效性、多样性、新颖性和复杂性的突破，在分子生成领域表现出了出色的性能。 |
| [^117] | [Short-length SSVEP data extension by a novel generative adversarial networks based framework.](http://arxiv.org/abs/2301.05599) | 本文提出了一种基于GAN的端到端信号转化网络TEGAN，可以将短SSVEP信号转换成长的人工SSVEP信号，并显著提高BCI系统的效率和准确性。 |
| [^118] | [Silent Killer: A Stealthy, Clean-Label, Black-Box Backdoor Attack.](http://arxiv.org/abs/2301.02615) | 默默杀手是一种隐蔽的、无标签的、黑盒子后门攻击，它使用了隐蔽的毒物和触发器，在无标签攻击中使用通用对抗扰动作为触发器，通过渐变对齐来提高成功率，并在MNIST、CIFAR10和ImageNet数据集上取得了最新的成果。 |
| [^119] | [Learning a Generic Value-Selection Heuristic Inside a Constraint Programming Solver.](http://arxiv.org/abs/2301.01913) | 本论文提出了一种通用学习过程，用于在约束规划求解器内获取一个值选择启发式方法，以解决当前通用值选择启发式方法较为稀缺的问题。 |
| [^120] | [Improving Multi-task Learning via Seeking Task-based Flat Regions.](http://arxiv.org/abs/2211.13723) | 通过寻找基于任务的平坦区域，可以改进多任务学习并提高模型性能，但需要正确使用正则化技术以避免次优解。 |
| [^121] | [L-MAE: Masked Autoencoders are Semantic Segmentation Datasets Augmenter.](http://arxiv.org/abs/2211.11242) | 本文提出了一种新的标签遮罩自编码器（L-MAE）方法，用于生成完整的语义分割标签。该方法首次将遮罩自编码器应用于下游任务，并采用了标签和图像的融合策略。这种方法能够解决大型模型和专业领域数据集中存在的数据标注不准确问题。 |
| [^122] | [Holistic Evaluation of Language Models.](http://arxiv.org/abs/2211.09110) | 我们提出了语言模型的整体评估（HELM），通过对潜在场景和度量进行分类并采用多度量方法，提高语言模型的透明度和可信度。 |
| [^123] | [Risk of Bias in Chest Radiography Deep Learning Foundation Models.](http://arxiv.org/abs/2209.02965) | 该研究分析了一种最近发布的胸部X光基础模型中的偏倚风险，并发现在生物性别和种族之间存在亚组性能差距。 |
| [^124] | [The emergence of division of labor through decentralized social sanctioning.](http://arxiv.org/abs/2208.05568) | 本研究通过引入社会规范模型，展示了分散社会制裁的出现模式能够解决以自利为导向的终身学习个体中的分工问题。 |
| [^125] | [Approximation Guarantees for the Non-Dominated Sorting Genetic Algorithm II (NSGA-II).](http://arxiv.org/abs/2203.02693) | 最近的研究发现，当种群规模足够大时，NSGA-II可以高效地计算完整的帕累托前沿。但当种群规模较小时，NSGA-II对帕累托前沿的逼近效果较差，特别是存在较大的间隙。这篇论文通过数学证明，揭示了这一问题的原因，并提出了两种改进方法。 |
| [^126] | [DEBOSH: Deep Bayesian Shape Optimization.](http://arxiv.org/abs/2109.13337) | 本论文提出了一种基于不确定性的方法，针对形状优化，在利用图神经网络预测工业设计性能时，解决了形状偏离训练集时预测不可靠的问题，并通过有效的贝叶斯优化提高了结果形状的质量。 |
| [^127] | [FUTURE-AI: Guiding Principles and Consensus Recommendations for Trustworthy Artificial Intelligence in Medical Imaging.](http://arxiv.org/abs/2109.09658) | 本论文介绍了一系列从经验、共识和最佳实践中提炼出的指导原则，旨在引领医学影像中值得信赖的人工智能的发展，提高信任、安全性和应用水平。 |
| [^128] | [Dynamics of specialization in neural modules under resource constraints.](http://arxiv.org/abs/2106.02626) | 本研究使用人工神经网络模拟实验，发现结构模块化并不一定能够确保功能专业化，在特定环境和资源限制下，才能够出现专业化现象。 |
| [^129] | [iCORPP: Interleaved Commonsense Reasoning and Probabilistic Planning on Robots.](http://arxiv.org/abs/2004.08672) | iCORPP提出了一种新型算法，能够在机器人的决策过程中同时推理世界状态、动态和构建任务导向的控制器。 |

# 详细

[^1]: Steered Diffusion: 一种广义的插件式条件图像合成框架

    Steered Diffusion: A Generalized Framework for Plug-and-Play Conditional Image Synthesis. (arXiv:2310.00224v1 [cs.CV])

    [http://arxiv.org/abs/2310.00224](http://arxiv.org/abs/2310.00224)

    Steered Diffusion是一个通用的框架，利用近期基于扩散的生成模型的细粒度生成控制能力，实现了零样本条件图像生成的高质量合成。

    

    条件生成模型通常需要大量的注释训练集才能实现高质量的合成。因此，设计能够执行插件式合成的模型引起了很大的兴趣，即使用预定义或预训练的模型来指导生成过程（例如使用语言），而该模型并没有明确训练在生成任务上。然而，这种指导通常只对合成高级语义有用，而不是编辑图像到图像转换任务中的细粒度细节。为了解决这个问题，并借助最近基于扩散的生成模型提供的强大细粒度生成控制能力，我们引入了Steered Diffusion，这是一个通用的框架，用于使用为无条件生成而训练的扩散模型进行逼真的零样本条件图像生成。其核心思想是通过设计使用预训练的逆模型损失来在推理时指导扩散模型的图像生成。

    Conditional generative models typically demand large annotated training sets to achieve high-quality synthesis. As a result, there has been significant interest in designing models that perform plug-and-play generation, i.e., to use a predefined or pretrained model, which is not explicitly trained on the generative task, to guide the generative process (e.g., using language). However, such guidance is typically useful only towards synthesizing high-level semantics rather than editing fine-grained details as in image-to-image translation tasks. To this end, and capitalizing on the powerful fine-grained generative control offered by the recent diffusion-based generative models, we introduce Steered Diffusion, a generalized framework for photorealistic zero-shot conditional image generation using a diffusion model trained for unconditional generation. The key idea is to steer the image generation of the diffusion model at inference time via designing a loss using a pre-trained inverse mod
    
[^2]: 超越随机噪声：通过隐性强盗研究洞察匿名化策略

    Beyond Random Noise: Insights on Anonymization Strategies from a Latent Bandit Study. (arXiv:2310.00221v1 [cs.LG])

    [http://arxiv.org/abs/2310.00221](http://arxiv.org/abs/2310.00221)

    本文通过使用隐性强盗设置和不同的聚合策略，评估了隐私和推荐器性能之间的权衡，为定制隐私技术的需求提供了洞察。研究结果表明，对个体用户的数据记录添加拉普拉斯机制的噪声是不合适的选择，它在任何噪声水平下都会产生最大的遗憾。

    

    本文研究了在用户共享知识进行推荐任务的学习场景中，隐私问题。我们的研究为隐私保护机器学习领域的研究增加了贡献，并强调了需要针对特定攻击模式而非依赖一刀切解决方案的定制隐私技术。我们使用了隐性强盗设置来评估隐私和推荐器性能之间的权衡，通过采用各种聚合策略，如平均、最近邻和聚类结合噪声注入。更具体地说，我们模拟了一个利用对手收集的公开可获得的辅助信息进行链接攻击的情景。我们在三个开放的真实数据集上的结果表明，对个体用户的数据记录添加拉普拉斯机制的噪声是一个糟糕的选择。它相对于去匿名化概率和ADS度量来说，在任何噪声水平下都提供了最大的遗憾。

    This paper investigates the issue of privacy in a learning scenario where users share knowledge for a recommendation task. Our study contributes to the growing body of research on privacy-preserving machine learning and underscores the need for tailored privacy techniques that address specific attack patterns rather than relying on one-size-fits-all solutions. We use the latent bandit setting to evaluate the trade-off between privacy and recommender performance by employing various aggregation strategies, such as averaging, nearest neighbor, and clustering combined with noise injection. More specifically, we simulate a linkage attack scenario leveraging publicly available auxiliary information acquired by the adversary. Our results on three open real-world datasets reveal that adding noise using the Laplace mechanism to an individual user's data record is a poor choice. It provides the highest regret for any noise level, relative to de-anonymization probability and the ADS metric. Inst
    
[^3]: 两两邻近策略优化: 利用相对反馈进行LLM对齐

    Pairwise Proximal Policy Optimization: Harnessing Relative Feedback for LLM Alignment. (arXiv:2310.00212v1 [cs.LG])

    [http://arxiv.org/abs/2310.00212](http://arxiv.org/abs/2310.00212)

    该论文提出了一种新的强化学习框架，使用相对反馈来调整大型语言模型（LLMs）的行为，解决了现有方法在优化比较损失训练的奖励时存在的限制。同时，还提出了一种新的基于轨迹的策略梯度算法（PPPO），用于更有效地进行算法设计和函数逼近。

    

    大型语言模型（LLMs）通过在大型语料库上预先训练来获取广泛的世界知识。然而，由于接触到低质量数据，LLMs可能表现出与人类价值不一致的有害行为。引导LLMs朝着有益行为方向发展的主导方法涉及使用人类反馈的强化学习（RLHF），其中Proximal Policy Optimization（PPO）是默认的RL优化器。尽管其有效性，但PPO在优化基于比较损失训练的奖励时存在局限性。主要问题是，由于需要校准奖励尺度，PPO对于包含相同偏好信息的等价奖励函数不具备不变性。此外，与基于轨迹的优化相比，PPO对于基于令牌的更新的需求引入了函数逼近和算法设计方面的复杂性。本文提出了一种新的框架，基于相对反馈的强化学习，以及一种新颖的基于轨迹的策略梯度算法，Pairwise Proximal Policy Optimization（PPPO），用于解决上述问题。

    Large Language Models (LLMs) can acquire extensive world knowledge through pre-training on large corpora. However, due to exposure to low-quality data, LLMs may exhibit harmful behavior without aligning with human values. The dominant approach for steering LLMs towards beneficial behavior involves Reinforcement Learning with Human Feedback (RLHF), with Proximal Policy Optimization (PPO) serving as the default RL optimizer. Despite its effectiveness, PPO has limitations when optimizing rewards trained from comparison-based loss. Primarily, PPO is not invariant to equivalent reward functions containing identical preference information due to the need to calibrate the reward scale. Additionally, PPO's necessity for token-wise updates introduces complexity in both function approximation and algorithm design compared to trajectory-wise optimization. This paper proposes a new framework, reinforcement learning with relative feedback, and a novel trajectory-wise policy gradient algorithm, Pair
    
[^4]: 受前额叶皮层启发的大型语言模型规划架构

    A Prefrontal Cortex-inspired Architecture for Planning in Large Language Models. (arXiv:2310.00194v1 [cs.AI])

    [http://arxiv.org/abs/2310.00194](http://arxiv.org/abs/2310.00194)

    这个论文提出了一个受前额叶皮层启发的大型语言模型规划架构，利用多个基于LLM的模块实现规划的自主协调，从而在处理需要多步推理或目标导向规划的任务时取得了较好的效果。

    

    大型语言模型（LLM）在许多任务上展现出惊人的性能，但它们经常在需要多步推理或目标导向规划的任务中遇到困难。为了解决这个问题，我们从人脑中获取灵感，即通过前额叶皮层（PFC）中专门模块的重复交互来完成规划。这些模块执行冲突监测、状态预测、状态评估、任务分解和任务协调等功能。我们发现LLM有时能够单独执行这些功能，但在服务于一个目标时往往难以自主协调它们。因此，我们提出了一个带有多个基于LLM（GPT-4）模块的黑盒架构。该架构通过专门的PFC启发模块的交互将一个更大的问题分解为多个对LLM的简短自动调用，从而改善规划能力。我们在两个具有挑战性的规划任务上评估了组合架构。

    Large language models (LLMs) demonstrate impressive performance on a wide variety of tasks, but they often struggle with tasks that require multi-step reasoning or goal-directed planning. To address this, we take inspiration from the human brain, in which planning is accomplished via the recurrent interaction of specialized modules in the prefrontal cortex (PFC). These modules perform functions such as conflict monitoring, state prediction, state evaluation, task decomposition, and task coordination. We find that LLMs are sometimes capable of carrying out these functions in isolation, but struggle to autonomously coordinate them in the service of a goal. Therefore, we propose a black box architecture with multiple LLM-based (GPT-4) modules. The architecture improves planning through the interaction of specialized PFC-inspired modules that break down a larger problem into multiple brief automated calls to the LLM. We evaluate the combined architecture on two challenging planning tasks -
    
[^5]: 图卷积和Mixup之间的等价性研究

    On the Equivalence of Graph Convolution and Mixup. (arXiv:2310.00183v1 [cs.LG])

    [http://arxiv.org/abs/2310.00183](http://arxiv.org/abs/2310.00183)

    这项研究发现，在两个温和的条件下，图卷积可以被视为Mixup的一种特殊形式，它在训练和测试阶段都被应用。

    

    本文研究了图卷积和Mixup技术之间的关系。图卷积在图神经网络中是通过聚合邻居样本的特征来学习特定节点或样本的代表性特征。而Mixup是一种数据增强技术，通过对多个样本的特征和独热标签进行平均来生成新的示例。这两种技术之间的一个共同之处是它们利用了来自多个样本的信息来得出特征表示。本研究旨在探索这两种方法之间是否存在联系。我们的调查发现，在两个温和的条件下，图卷积可以被视为Mixup的一种特殊形式，它在训练和测试阶段都被应用。这两个条件是：1）\textit{同质改标} - 将目标节点的标签分配给其所有邻居，以及2）\textit{测试时Mixup} - 在测试时对特征进行Mixup。我们确定了这两个条件的数学表达，并通过实验验证了这个等价关系的有效性。

    This paper investigates the relationship between graph convolution and Mixup techniques. Graph convolution in a graph neural network involves aggregating features from neighboring samples to learn representative features for a specific node or sample. On the other hand, Mixup is a data augmentation technique that generates new examples by averaging features and one-hot labels from multiple samples. One commonality between these techniques is their utilization of information from multiple samples to derive feature representation. This study aims to explore whether a connection exists between these two approaches. Our investigation reveals that, under two mild conditions, graph convolution can be viewed as a specialized form of Mixup that is applied during both the training and testing phases. The two conditions are: 1) \textit{Homophily Relabel} - assigning the target node's label to all its neighbors, and 2) \textit{Test-Time Mixup} - Mixup the feature during the test time. We establis
    
[^6]: Motif: 来自人工智能反馈的内在动机

    Motif: Intrinsic Motivation from Artificial Intelligence Feedback. (arXiv:2310.00166v1 [cs.AI])

    [http://arxiv.org/abs/2310.00166](http://arxiv.org/abs/2310.00166)

    本文提出了一种名为Motif的方法，通过与大型语言模型（LLM）交互来获得先验知识，并将其用于代理程序的强化学习训练。实验证明，Motif的内在奖励相比直接最大化得分的算法在挑战性游戏中获得了更高的游戏得分，并在之前没有取得进展的任务上取得了显著的进展。

    

    在没有先验知识的情况下，探索丰富的环境并评估自己的行动是非常具有挑战性的。在本文中，我们提出了Motif，一种用大型语言模型（LLM）将先验知识与代理程序接口的通用方法。Motif的基本思想是将LLMs用于决策，而无需与环境进行交互：它通过从LLM中产生对配对标题的偏好来构建内在奖励，然后使用该奖励对代理程序进行强化学习训练。我们在具有挑战性、开放性和程序生成的NetHack游戏上评估了Motif的性能和行为。令人惊讶的是，仅通过学习最大化其内在奖励，Motif的游戏得分比直接训练以最大化得分的算法更高。当将Motif的内在奖励与环境奖励相结合时，我们的方法明显优于现有方法，并在以前从未取得进展的任务上取得进展。

    Exploring rich environments and evaluating one's actions without prior knowledge is immensely challenging. In this paper, we propose Motif, a general method to interface such prior knowledge from a Large Language Model (LLM) with an agent. Motif is based on the idea of grounding LLMs for decision-making without requiring them to interact with the environment: it elicits preferences from an LLM over pairs of captions to construct an intrinsic reward, which is then used to train agents with reinforcement learning. We evaluate Motif's performance and behavior on the challenging, open-ended and procedurally-generated NetHack game. Surprisingly, by only learning to maximize its intrinsic reward, Motif achieves a higher game score than an algorithm directly trained to maximize the score itself. When combining Motif's intrinsic reward with the environment reward, our method significantly outperforms existing approaches and makes progress on tasks where no advancements have ever been made with
    
[^7]: 面向检测的图像-文本预训练方法用于开放词汇检测

    Detection-Oriented Image-Text Pretraining for Open-Vocabulary Detection. (arXiv:2310.00161v1 [cs.CV])

    [http://arxiv.org/abs/2310.00161](http://arxiv.org/abs/2310.00161)

    这项研究提出了一种面向检测的图像-文本预训练方法，旨在弥合图像级预训练和开放词汇目标检测之间的差距。通过检测器架构和对比损失，该方法能够从噪声图像-文本对中学习到新出现的物体-语义线索，并提出了一种平移窗口学习方法来改进主干网络的表示。在LVIS开放词汇检测基准上，该方法取得了显著优于其他方法的40.4的掩码AP$_r$结果。

    

    我们提出了一种基于面向检测的图像-文本预训练的新的开放词汇检测方法，以填补图像级预训练和开放词汇目标检测之间的差距。在预训练阶段，我们用检测器架构替代常用的分类架构，通过使检测器头部能够从噪声图像-文本对中学习，更好地满足检测的区域级识别需求。我们的方法只使用标准的对比损失而不使用伪标签，是对对比学习方法的简单而有效的扩展，可以学习到新出现的物体-语义线索。此外，我们提出了一种基于窗口注意力的平移窗口学习方法，使主干网络的表示更加鲁棒、平移不变，并且不受窗口模式的偏差影响。在流行的LVIS开放词汇检测基准上，我们的方法使用常见的ViT-L主干网络取得了40.4的掩码AP$_r$新的最优结果，明显优于其他方法。

    We present a new open-vocabulary detection approach based on detection-oriented image-text pretraining to bridge the gap between image-level pretraining and open-vocabulary object detection. At the pretraining phase, we replace the commonly used classification architecture with the detector architecture, which better serves the region-level recognition needs of detection by enabling the detector heads to learn from noisy image-text pairs. Using only standard contrastive loss and no pseudo-labeling, our approach is a simple yet effective extension of the contrastive learning method to learn emergent object-semantic cues. In addition, we propose a shifted-window learning approach upon window attention to make the backbone representation more robust, translation-invariant, and less biased by the window pattern. On the popular LVIS open-vocabulary detection benchmark, our approach sets a new state of the art of 40.4 mask AP$_r$ using the common ViT-L backbone, significantly outperforming t
    
[^8]: 自我特化：揭示大型语言模型中的潜在专业知识

    Self-Specialization: Uncovering Latent Expertise within Large Language Models. (arXiv:2310.00160v1 [cs.CL])

    [http://arxiv.org/abs/2310.00160](http://arxiv.org/abs/2310.00160)

    该论文研究了大型语言模型的自我特化，通过使用专业领域的数据和少量标记种子进行自我对齐，提高了在目标领域的零样本和少样本性能。

    

    最近的研究表明，自我调整的有效性，即通过使用少量人类编写的种子数据自动生成教学数据，使大型语言模型自动对齐以遵循一般指示。在这项工作中，我们不再关注一般对齐，而是专注于专家领域特化的自我对齐（例如，生物医学），发现它对于提高目标领域的零样本和少样本性能非常有效。首先，我们介绍了现有对齐模型在专业领域内的基准结果，揭示了“通用”指示跟随训练对下游专家领域性能的边际效应。为了解决这个问题，我们探索了自我特化，利用领域特定的未标记数据和少量标记种子进行自我对齐过程。当通过检索来减少产生幻觉并提高对齐的并发性后，自我特化提供了一种解决方案。

    Recent works have demonstrated the effectiveness of self-alignment in which a large language model is, by itself, aligned to follow general instructions through the automatic generation of instructional data using a handful of human-written seeds. Instead of general alignment, in this work, we focus on self-alignment for expert domain specialization (e.g., biomedicine), discovering it to be very effective for improving zero-shot and few-shot performance in target domains of interest. As a preliminary, we first present the benchmark results of existing aligned models within a specialized domain, which reveals the marginal effect that "generic" instruction-following training has on downstream expert domains' performance. To remedy this, we explore self-specialization that leverages domain-specific unlabelled data and a few labeled seeds for the self-alignment process. When augmented with retrieval to reduce hallucination and enhance concurrency of the alignment, self-specialization offer
    
[^9]: 不平衡分类中的反馈引导数据合成

    Feedback-guided Data Synthesis for Imbalanced Classification. (arXiv:2310.00158v1 [cs.CV])

    [http://arxiv.org/abs/2310.00158](http://arxiv.org/abs/2310.00158)

    本论文介绍了一种反馈引导数据合成的方法，通过从分类器到生成模型的反馈来驱动采样，将静态数据集增强为包含有用的合成样本，以提高分类器的性能。

    

    当前机器学习中的现状是使用来自长尾分布的真实图像的静态数据集进行训练。最近生成模型的进展使研究人员开始用合成数据增强这些静态数据集，并在分类任务上报告了适度的性能改进。我们假设这些性能提升受到从分类器到生成模型的反馈不足的限制，这将促进生成样本的有用性以提高分类器的性能。在这项工作中，我们介绍了一种用有用的合成样本增强静态数据集的框架，该框架利用从分类器到生成模型的一次性反馈来驱动采样。为了使该框架有效，我们发现样本必须接近手头任务的真实数据支持，并且具有足够的多样性。我们在一个长尾数据集（ImageNe...上验证了三个反馈标准。

    Current status quo in machine learning is to use static datasets of real images for training, which often come from long-tailed distributions. With the recent advances in generative models, researchers have started augmenting these static datasets with synthetic data, reporting moderate performance improvements on classification tasks. We hypothesize that these performance gains are limited by the lack of feedback from the classifier to the generative model, which would promote the usefulness of the generated samples to improve the classifier's performance. In this work, we introduce a framework for augmenting static datasets with useful synthetic samples, which leverages one-shot feedback from the classifier to drive the sampling of the generative model. In order for the framework to be effective, we find that the samples must be close to the support of the real data of the task at hand, and be sufficiently diverse. We validate three feedback criteria on a long-tailed dataset (ImageNe
    
[^10]: 通过轨迹生成学习具有通用性的工具使用技能

    Learning Generalizable Tool-use Skills through Trajectory Generation. (arXiv:2310.00156v1 [cs.RO])

    [http://arxiv.org/abs/2310.00156](http://arxiv.org/abs/2310.00156)

    通过轨迹生成，我们提出了一种学习通用工具使用技能的方法，可以适应不同形状的工具，从而使自主系统能够处理复杂的可变形物体操作任务。

    

    高效利用工具的自主系统可以帮助人们完成许多常见任务，如烹饪和清洁。然而，当前的系统在适应新工具方面远远不及人类的智能水平。基于可及性的先前工作通常对环境做出了很强的假设，并且无法扩展到更复杂、接触丰富的任务。 在这项工作中，我们解决了这个挑战，并探索了代理如何学习使用以前未见过的工具来操纵可变形物体。 我们提出了将工具使用轨迹作为一系列点云的生成模型，可以推广到不同的工具形状。对于任何新的工具，我们首先生成一个工具使用轨迹，然后优化工具姿势序列以与生成的轨迹对齐。我们为四种不同的具有挑战性的可变形物体操纵任务训练了一个单一模型。我们的模型仅使用每个任务的单个工具的示范数据进行训练，并且能够...

    Autonomous systems that efficiently utilize tools can assist humans in completing many common tasks such as cooking and cleaning. However, current systems fall short of matching human-level of intelligence in terms of adapting to novel tools. Prior works based on affordance often make strong assumptions about the environments and cannot scale to more complex, contact-rich tasks. In this work, we tackle this challenge and explore how agents can learn to use previously unseen tools to manipulate deformable objects. We propose to learn a generative model of the tool-use trajectories as a sequence of point clouds, which generalizes to different tool shapes. Given any novel tool, we first generate a tool-use trajectory and then optimize the sequence of tool poses to align with the generated trajectory. We train a single model for four different challenging deformable object manipulation tasks. Our model is trained with demonstration data from just a single tool for each task and is able to 
    
[^11]: 原始-对偶持续学习：通过拉格朗日乘子实现稳定性和可塑性

    Primal-Dual Continual Learning: Stability and Plasticity through Lagrange Multipliers. (arXiv:2310.00154v1 [cs.LG])

    [http://arxiv.org/abs/2310.00154](http://arxiv.org/abs/2310.00154)

    本文提出了原始-对偶持续学习方法，通过利用拉格朗日对偶解决受限学习问题，实现了稳定性和可塑性。作者通过分析任务层面和样本层面的约束，在基于记忆的方法中分配资源，取得了较好的效果。

    

    持续学习固有地是一个受限学习问题。目标是在“无遗忘”要求下学习一个预测器。尽管之前有几项研究将其形式化为这样一个问题，但它们没有明确解决这个受限问题。在这项工作中，我们展示了直接解决这个受限优化问题是可行且有益的。为此，我们利用了最近在限制性学习中的拉格朗日对偶的结果。我们聚焦于基于记忆的方法，其中可以将先前任务中的一小部分样本存储在回放缓冲区中。在这个设置中，我们分析了持续学习问题的两个版本：一个在任务层面上有约束的粗糙方法和一个在样本层面上有约束的精细方法。我们展示了对偶变量指示了最优值对于约束扰动的敏感性。然后，我们利用这个结果在粗糙方法中对缓冲区进行了划分，将更多资源分配给更难的任务。

    Continual learning is inherently a constrained learning problem. The goal is to learn a predictor under a \emph{no-forgetting} requirement. Although several prior studies formulate it as such, they do not solve the constrained problem explicitly. In this work, we show that it is both possible and beneficial to undertake the constrained optimization problem directly. To do this, we leverage recent results in constrained learning through Lagrangian duality. We focus on memory-based methods, where a small subset of samples from previous tasks can be stored in a replay buffer. In this setting, we analyze two versions of the continual learning problem: a coarse approach with constraints at the task level and a fine approach with constraints at the sample level. We show that dual variables indicate the sensitivity of the optimal value with respect to constraint perturbations. We then leverage this result to partition the buffer in the coarse approach, allocating more resources to harder task
    
[^12]: 在嘈杂的农业环境中的3D重建：基于贝叶斯优化视角的观测规划

    3D Reconstruction in Noisy Agricultural Environments: A Bayesian Optimization Perspective for View Planning. (arXiv:2310.00145v1 [cs.RO])

    [http://arxiv.org/abs/2310.00145](http://arxiv.org/abs/2310.00145)

    本论文提出了一种在噪声环境中进行3D重建的新方法，通过观测规划，合理选择相机位置并考虑噪声对重建性能的影响，提高了3D重建结果的质量和效率。

    

    3D重建是机器人技术中一项基础任务，因其在农业、水下和城市环境等实际场景中产生了重大影响而受到关注。其中一种重要的方法是通过合理安放相机来最大化视觉信息，提高3D重建结果，称为观测规划。通过将几何标准应用于选择较少但更有信息量的图像，可以避免需要大量的任意图像，从而显著提高3D重建性能。然而，在各种真实场景中考虑到存在的噪声可能是具有挑战性的，特别是当没有提供有关噪声的先验信息时。为此，本研究提出了一种新的几何函数，考虑到现有的噪声，仅依靠相对较少的噪声实现来计算，而不需要其封闭形式。

    3D reconstruction is a fundamental task in robotics that gained attention due to its major impact in a wide variety of practical settings, including agriculture, underwater, and urban environments. An important approach for this task, known as view planning, is to judiciously place a number of cameras in positions that maximize the visual information improving the resulting 3D reconstruction. Circumventing the need for a large number of arbitrary images, geometric criteria can be applied to select fewer yet more informative images to markedly improve the 3D reconstruction performance. Nonetheless, incorporating the noise of the environment that exists in various real-world scenarios into these criteria may be challenging, particularly when prior information about the noise is not provided. To that end, this work advocates a novel geometric function that accounts for the existing noise, relying solely on a relatively small number of noise realizations without requiring its closed-form e
    
[^13]: 基于概率采样增强的时空图卷积网络：一种可扩展的以太坊网络交易异常检测框架

    Probabilistic Sampling-Enhanced Temporal-Spatial GCN: A Scalable Framework for Transaction Anomaly Detection in Ethereum Networks. (arXiv:2310.00144v1 [cs.LG])

    [http://arxiv.org/abs/2310.00144](http://arxiv.org/abs/2310.00144)

    该研究提出了一种基于概率采样增强的时空图卷积网络框架，用于以太坊网络中的交易异常检测。通过将图卷积网络与时态随机游走相结合，利用时间序列的复杂性提供更精细的交易异常检测机制。实验结果表明，与传统的图卷积网络相比，该框架在检测异常和交易突发方面有显著的性能提升。这项研究强调了以太坊交易数据中时间线索的潜力，并展示了使用该框架进行交易异常检测的可行性。

    

    以太坊网络的快速演进需要先进的技术来确保其对潜在威胁的鲁棒性并保持透明度。虽然图神经网络（GNN）在此类平台的异常检测方面取得了先导性成果，但捕捉空间和时间事务模式的复杂性仍然是一个挑战。本研究提出了一种将图卷积网络（GCNs）与使用概率采样增强的时态随机游走（TRW）相结合的方法，以弥合这一差距。与传统的GCNs不同，我们的方法利用TRW的优势来识别以太坊交易中复杂的时间序列，从而提供更细致入微的交易异常检测机制。初步评估表明，我们的TRW-GCN框架在检测异常和交易突发的性能指标上显著提高了传统GCNs的表现。这项研究不仅强调了以太坊交易数据中时间线索的潜力，同时揭示了使用概率采样增强的时空GCNs进行交易异常检测的可行性。

    The rapid evolution of the Ethereum network necessitates sophisticated techniques to ensure its robustness against potential threats and to maintain transparency. While Graph Neural Networks (GNNs) have pioneered anomaly detection in such platforms, capturing the intricacies of both spatial and temporal transactional patterns has remained a challenge. This study presents a fusion of Graph Convolutional Networks (GCNs) with Temporal Random Walks (TRW) enhanced by probabilistic sampling to bridge this gap. Our approach, unlike traditional GCNs, leverages the strengths of TRW to discern complex temporal sequences in Ethereum transactions, thereby providing a more nuanced transaction anomaly detection mechanism. Preliminary evaluations demonstrate that our TRW-GCN framework substantially advances the performance metrics over conventional GCNs in detecting anomalies and transaction bursts. This research not only underscores the potential of temporal cues in Ethereum transactional data but a
    
[^14]: GASS：使用大规模数据进行音频源分离的泛化方法

    GASS: Generalizing Audio Source Separation with Large-scale Data. (arXiv:2310.00140v1 [cs.SD])

    [http://arxiv.org/abs/2310.00140](http://arxiv.org/abs/2310.00140)

    本文研究了一种使用大规模数据进行音频源分离的通用方法（GASS），在有限分布范围内表现出良好的效果，并展示了其在声音事件和语音分离方面的泛化能力。然而，在分离超出分布的电影和音乐内容方面仍存在挑战。

    

    通用源分离的目标是分离任意混合音频中的音频源，消除仅操作于特定领域（如语音或音乐）的约束。然而，通用源分离的潜力受限于大多数现有作品主要关注具有主要声音事件的混合以及小型训练数据集对于监督学习的潜力限制。本文研究了使用大规模数据集以有监督的方式训练的单一通用音频源分离（GASS）模型，该模型能够分离语音、音乐和声音事件。我们对GASS模型进行了多样化任务的评估。我们的强有力分布结果显示了GASS模型的可行性，而在声音事件和语音分离方面的竞争性超出分布性能则显示了其泛化能力。然而，GASS模型在分离超出分布的电影和音乐内容方面具有挑战性。我们还对每个数据集对GASS模型进行了微调，并始终表现优于其他模型。

    Universal source separation targets at separating the audio sources of an arbitrary mix, removing the constraint to operate on a specific domain like speech or music. Yet, the potential of universal source separation is limited because most existing works focus on mixes with predominantly sound events, and small training datasets also limit its potential for supervised learning. Here, we study a single general audio source separation (GASS) model trained to separate speech, music, and sound events in a supervised fashion with a large-scale dataset. We assess GASS models on a diverse set of tasks. Our strong in-distribution results show the feasibility of GASS models, and the competitive out-of-distribution performance in sound event and speech separation shows its generalization abilities. Yet, it is challenging for GASS models to generalize for separating out-of-distribution cinematic and music content. We also fine-tune GASS models on each dataset and consistently outperform the ones
    
[^15]: ABScribe: 使用大型语言模型在人工智能与人类共同写作任务中快速探索多种写作变化

    ABScribe: Rapid Exploration of Multiple Writing Variations in Human-AI Co-Writing Tasks using Large Language Models. (arXiv:2310.00117v1 [cs.HC])

    [http://arxiv.org/abs/2310.00117](http://arxiv.org/abs/2310.00117)

    ABScribe是一种界面，支持在人工智能与人类共同写作任务中快速探索多种写作变化。用户可以使用大型语言模型提示快速生成多个变体，这些变体以可重用的按钮形式呈现，并且可以通过上下文工具栏进行快速的就地比较。

    

    通过重新书写文本来探索替代想法是写作过程的关键。最先进的大型语言模型（LLM）可以简化写作变化生成的过程。然而，当前的界面存在同时考虑多种变化的挑战：在不覆盖文本的情况下创建新的版本可能很困难，而按顺序粘贴它们可能会使文档变得杂乱，增加工作量，并打断作者的流程。为了解决这个问题，我们提出了ABScribe，一种支持在人工智能与人类共同写作任务中快速且结构化地探索写作变化的界面。通过ABScribe，用户可以使用LLM提示快速产生多个变体，这些变体会自动转换成可重用的按钮形式。变体在文本段落中被存储在相邻位置，通过在上下文工具栏上的鼠标悬停交互进行快速的就地比较。我们对12名撰写人员进行的用户研究表明，ABScribe能显著减轻任务负荷（d = 1.20, p < 0.001），提高用户的认知程度。

    Exploring alternative ideas by rewriting text is integral to the writing process. State-of-the-art large language models (LLMs) can simplify writing variation generation. However, current interfaces pose challenges for simultaneous consideration of multiple variations: creating new versions without overwriting text can be difficult, and pasting them sequentially can clutter documents, increasing workload and disrupting writers' flow. To tackle this, we present ABScribe, an interface that supports rapid, yet visually structured, exploration of writing variations in human-AI co-writing tasks. With ABScribe, users can swiftly produce multiple variations using LLM prompts, which are auto-converted into reusable buttons. Variations are stored adjacently within text segments for rapid in-place comparisons using mouse-over interactions on a context toolbar. Our user study with 12 writers shows that ABScribe significantly reduces task workload (d = 1.20, p < 0.001), enhances user perceptions o
    
[^16]: 动态边界最大化和改进的Lipschitz正则化的认证鲁棒性

    Certified Robustness via Dynamic Margin Maximization and Improved Lipschitz Regularization. (arXiv:2310.00116v1 [cs.LG])

    [http://arxiv.org/abs/2310.00116](http://arxiv.org/abs/2310.00116)

    本文提出了一种基于动态边界最大化和改进的Lipschitz正则化的认证鲁棒性训练算法，通过增加输出空间中的边界和正则化模型的Lipschitz常数来提高深度分类器对抗性扰动的鲁棒性。

    

    为了提高深度分类器对抗性扰动的鲁棒性，已经提出了许多方法，例如设计具有更好鲁棒性性质的新架构（例如，Lipschitz-capped网络）或修改训练过程本身（例如，最小-最大优化，约束学习或正则化）。然而，这些方法对于增加输入（特征）空间中的边界可能并不有效。因此，越来越多的人开始对开发能够直接操纵输入空间中的决策边界的训练过程感兴趣。在本文中，我们在该类别的最新发展基础上，开发了一种鲁棒训练算法，其目标是在输出（logit）空间中增加边界，并沿着脆弱方向正则化模型的Lipschitz常数。我们证明这两个目标可以直接促进输入空间中更大的边界。为此，我们开发了一种可扩展的方法来计算...

    To improve the robustness of deep classifiers against adversarial perturbations, many approaches have been proposed, such as designing new architectures with better robustness properties (e.g., Lipschitz-capped networks), or modifying the training process itself (e.g., min-max optimization, constrained learning, or regularization). These approaches, however, might not be effective at increasing the margin in the input (feature) space. As a result, there has been an increasing interest in developing training procedures that can directly manipulate the decision boundary in the input space. In this paper, we build upon recent developments in this category by developing a robust training algorithm whose objective is to increase the margin in the output (logit) space while regularizing the Lipschitz constant of the model along vulnerable directions. We show that these two objectives can directly promote larger margins in the input space. To this end, we develop a scalable method for calcula
    
[^17]: HyperMask: 自适应的基于超网络的掩码用于持续学习

    HyperMask: Adaptive Hypernetwork-based Masks for Continual Learning. (arXiv:2310.00113v1 [cs.LG])

    [http://arxiv.org/abs/2310.00113](http://arxiv.org/abs/2310.00113)

    HyperMask是一种用于持续学习的方法，它使用基于超网络的掩码来训练一个单一网络，以克服人工神经网络在多任务上的灾难性遗忘问题。

    

    当人工神经网络在多个任务上顺序训练时，往往会出现灾难性遗忘的问题。为了克服这个问题，已经存在许多持续学习策略，其中最有效的之一是基于超网络的方法。超网络根据任务的特征生成目标模型的权重。然而，该模型的主要限制是超网络对于每个任务可以产生完全不同的网络结构，因此每个任务都是单独解决的。模型在学习后续任务时不使用之前任务所关联的网络信息，并实际上产生了新的网络架构。为了解决这个问题，我们使用了彩票票证假设，该假设认为存在稀疏的子网络（即中奖票），可以保持完整网络的性能。在本文中，我们提出了一种名为HyperMask的方法，该方法为所有任务训练一个单一网络。超网络产生半二进制掩码，以获取目标子网络。

    Artificial neural networks suffer from catastrophic forgetting when they are sequentially trained on multiple tasks. To overcome this problem, there exist many continual learning strategies. One of the most effective is the hypernetwork-based approach. The hypernetwork generates the weights of a target model based on the task's identity. The model's main limitation is that hypernetwork can produce completely different nests for each task. Consequently, each task is solved separately. The model does not use information from the network dedicated to previous tasks and practically produces new architectures when it learns the subsequent tasks. To solve such a problem, we use the lottery ticket hypothesis, which postulates the existence of sparse subnetworks, named winning tickets, that preserve the performance of a full network.  In the paper, we propose a method called HyperMask, which trains a single network for all tasks. Hypernetwork produces semi-binary masks to obtain target subnetw
    
[^18]: FashionFlow: 利用扩散模型从静态图像生成动态时尚视频

    FashionFlow: Leveraging Diffusion Models for Dynamic Fashion Video Synthesis from Static Imagery. (arXiv:2310.00106v1 [cs.CV])

    [http://arxiv.org/abs/2310.00106](http://arxiv.org/abs/2310.00106)

    本研究提出了一种名为FashionFlow的图像到视频生成器，利用扩散模型从静态图像生成短视频。我们通过开发并连接与扩散模型相关的组件来实现这一目标，其中包括使用伪3D卷积层高效生成视频，并利用VAE和CLIP编码器捕捉关键特征。研究结果展示了成功合成时尚视频的能力，能够展示服装的合身度和外观，为在线时尚行业的购物体验提供改进和增强的潜力。

    

    我们的研究介绍了一种新的图像到视频生成器，称为FashionFlow。通过利用扩散模型，我们能够从静态图像创建短视频。我们的方法涉及开发并连接与扩散模型相关的组件，这使得我们的工作与众不同。这些组件包括使用伪3D卷积层高效生成视频。VAE和CLIP编码器从静态图像中捕捉到重要特征，以影响扩散模型。我们的研究展示了成功合成具有不同角度的模特一边摆姿势，展示服装的合身度和外观的时尚视频。我们的发现对于改进和提升在线时尚行业的购物体验有很大的潜力。

    Our study introduces a new image-to-video generator called FashionFlow. By utilising a diffusion model, we are able to create short videos from still images. Our approach involves developing and connecting relevant components with the diffusion model, which sets our work apart. The components include the use of pseudo-3D convolutional layers to generate videos efficiently. VAE and CLIP encoders capture vital characteristics from still images to influence the diffusion model. Our research demonstrates a successful synthesis of fashion videos featuring models posing from various angles, showcasing the fit and appearance of the garment. Our findings hold great promise for improving and enhancing the shopping experience for the online fashion industry.
    
[^19]: 放射学报告的多语言自然语言处理模型--摘要是你需要的一切！

    Multilingual Natural Language ProcessingModel for Radiology Reports -- The Summary is all you need!. (arXiv:2310.00100v1 [cs.CL])

    [http://arxiv.org/abs/2310.00100](http://arxiv.org/abs/2310.00100)

    本研究通过在多语言文本到文本变换器模型上微调，开发了一个能够自动在多语言中总结放射学报告的模型。该模型有助于提高未来深度学习模型的研究和发展，且能够应用于不同族裔背景的患者数据。

    

    放射学报告的印象部分总结了重要的放射学发现，并在向医生传达这些发现时起到了关键作用。然而，对于放射科医生来说，准备这些摘要既耗时又容易出错。最近，已经开发了许多用于放射学报告摘要的模型。然而，目前还没有能够在多种语言中总结这些报告的模型。这样的模型可以极大地改进未来的研究和融合来自不同族裔背景的患者数据的深度学习模型的发展。本研究通过在公开可用的基于多语言文本到文本变换器的模型上微调，自动化地生成了不同语言的放射学印象，以总结英语、葡萄牙语和德语的放射学报告中的发现。在一项盲测中，两位有执业资格的放射科医生表示，对于至少70%的系统生成的摘要，其质量

    The impression section of a radiology report summarizes important radiology findings and plays a critical role in communicating these findings to physicians. However, the preparation of these summaries is time-consuming and error-prone for radiologists. Recently, numerous models for radiology report summarization have been developed. Nevertheless, there is currently no model that can summarize these reports in multiple languages. Such a model could greatly improve future research and the development of Deep Learning models that incorporate data from patients with different ethnic backgrounds. In this study, the generation of radiology impressions in different languages was automated by fine-tuning a model, publicly available, based on a multilingual text-to-text Transformer to summarize findings available in English, Portuguese, and German radiology reports. In a blind test, two board-certified radiologists indicated that for at least 70% of the system-generated summaries, the quality 
    
[^20]: Voice2Action: 语言模型作为虚拟现实中高效实时交互的代理人

    Voice2Action: Language Models as Agent for Efficient Real-Time Interaction in Virtual Reality. (arXiv:2310.00092v1 [cs.CL])

    [http://arxiv.org/abs/2310.00092](http://arxiv.org/abs/2310.00092)

    本研究提出了Voice2Action，一种使用语言模型作为代理人在虚拟现实中进行高效实时交互的框架。通过对定制语音信号和文本命令进行分层分析，并将执行任务分成交互子集，Voice2Action能够比其他方法更高效和准确地执行。

    

    大型语言模型（LLMs）被训练和调整以仅仅使用少量示例来遵循自然语言指令，并被提示为任务驱动的自主代理人，以适应不同的执行环境来源。然而，在虚拟现实（VR）中部署代理LLMs一直是具有挑战性的，其原因是在线交互的效率低下以及3D环境中复杂的操作类别。在这项工作中，我们提出了Voice2Action，一个通过动作和实体提取来分层分析定制语音信号和文本命令，并将执行任务实时分成规范的交互子集，并通过环境反馈来防止错误。在具有合成指令数据的城市工程VR环境中的实验结果表明，Voice2Action能够比没有优化的方法更高效和准确地执行。

    Large Language Models (LLMs) are trained and aligned to follow natural language instructions with only a handful of examples, and they are prompted as task-driven autonomous agents to adapt to various sources of execution environments. However, deploying agent LLMs in virtual reality (VR) has been challenging due to the lack of efficiency in online interactions and the complex manipulation categories in 3D environments. In this work, we propose Voice2Action, a framework that hierarchically analyzes customized voice signals and textual commands through action and entity extraction and divides the execution tasks into canonical interaction subsets in real-time with error prevention from environment feedback. Experiment results in an urban engineering VR environment with synthetic instruction data show that Voice2Action can perform more efficiently and accurately than approaches without optimizations.
    
[^21]: SocREval：使用苏格拉底方法进行无参考推理评估的大规模语言模型

    SocREval: Large Language Models with the Socratic Method for Reference-Free Reasoning Evaluation. (arXiv:2310.00074v1 [cs.CL])

    [http://arxiv.org/abs/2310.00074](http://arxiv.org/abs/2310.00074)

    本论文提出了一种称为SocREval的方法，利用GPT-4和苏格拉底方法进行无参考推理评估，以解决当前复杂推理模型评估中遇到的挑战。

    

    为了全面评估当前模型在复杂推理方面的能力，以可扩展的方式评估它们的逐步推理是至关重要的。现有的基于参考的评估指标依赖于人工注释的推理链来评估模型导出的推理链。然而，这样的“黄金标准”人工编写的推理链可能不是唯一的，并且其获取通常是劳动密集型的。现有的无参考推理指标消除了人工制作推理链的需求作为参考，但通常需要在具有人工推理链的数据集上进行微调，这复杂化了流程并引发了在不同数据集上泛化性的担忧。为了解决这些挑战，我们利用GPT-4自动评估推理链质量，消除了对人工制作参考的需求。利用苏格拉底方法，我们设计了定制化提示来增强无参考推理评估，这就是我们称之为SocREval（苏格拉底方法）的方法。

    To comprehensively assess the capacity of current models for complex reasoning, it is crucial to assess their step-by-step reasoning in a scalable manner. Established reference-based evaluation metrics rely on human-annotated reasoning chains to assess the model-derived chains. However, such ``gold-standard'' human-written reasoning chains may not be unique and their acquisition is often labor-intensive. Existing reference-free reasoning metrics eliminate the need for human-crafted reasoning chains as references, but they typically require fine-tuning on datasets with human-derived reasoning chains, which complicates the process and raises concerns regarding generalizability across diverse datasets. To address these challenges, we harness GPT-4 to automatically evaluate reasoning chain quality, obviating the need for human-crafted references. Leveraging the Socratic method, we devise tailored prompts to enhance reference-free reasoning evaluation, which we term SocREval (Socratic metho
    
[^22]: 情感听众肖像：真实的听众动作模拟对话

    Emotional Listener Portrait: Realistic Listener Motion Simulation in Conversation. (arXiv:2310.00068v1 [cs.GR])

    [http://arxiv.org/abs/2310.00068](http://arxiv.org/abs/2310.00068)

    本论文提出了一种情感听众肖像（ELP）模型，采用了显式离散设计，能根据对话中不同情绪生成自然多样又可控的响应，解决了面部表情生成中的非确定性问题。

    

    听者头部生成主要关注在根据讲话者传递的信息生成听者的非语言行为（例如微笑）。生成这样的响应时一个重要的挑战是对话中精细面部表情的非确定性特性，这取决于讲话者和听者的情绪和态度。为了解决这个问题，我们提出了情感听众肖像（ELP），它将每个细粒度面部动作视为若干离散动作编码词的组合，并显式地建模了不同情感下动作的概率分布。由于“显式”和“离散”的设计，我们的ELP模型不仅可以通过从学习的分布中采样自动生成对给定讲话者的自然多样的响应，还可以生成具有预先确定态度的可控响应。在几个定量度量指标下，我们的ELP表现出显著的结果。

    Listener head generation centers on generating non-verbal behaviors (e.g., smile) of a listener in reference to the information delivered by a speaker. A significant challenge when generating such responses is the non-deterministic nature of fine-grained facial expressions during a conversation, which varies depending on the emotions and attitudes of both the speaker and the listener. To tackle this problem, we propose the Emotional Listener Portrait (ELP), which treats each fine-grained facial motion as a composition of several discrete motion-codewords and explicitly models the probability distribution of the motions under different emotion in conversation. Benefiting from the ``explicit'' and ``discrete'' design, our ELP model can not only automatically generate natural and diverse responses toward a given speaker via sampling from the learned distribution but also generate controllable responses with a predetermined attitude. Under several quantitative metrics, our ELP exhibits sig
    
[^23]: AI集成用于探测秩序更高的引力波模式：准圆形，旋转，非进动的二进制黑洞合并。(arXiv:2310.00052v1 [astro-ph.IM])

    AI ensemble for signal detection of higher order gravitational wave modes of quasi-circular, spinning, non-precessing binary black hole mergers. (arXiv:2310.00052v1 [astro-ph.IM])

    [http://arxiv.org/abs/2310.00052](http://arxiv.org/abs/2310.00052)

    本研究提出了使用AI集成同时处理双重LIGO探测器和Virgo探测器数据的模型，成功训练出能够探测秩序更高的引力波模式的AI分类器，并使用迁移学习估计了潜在二进制黑洞的总质量。通过实验验证了该集成在处理大量信号时的性能。

    

    我们引入了时空图模型，同时处理来自双重先进的LIGO探测器和先进的Virgo探测器的数据。我们使用240万个描述准圆形，旋转，非进动二进制黑洞合并的\texttt {IMRPhenomXPHM}波形来训练这些AI分类器，其中组分质量$m_{\{1,2\}}\in[3M_\odot, 50 M_\odot]$，个体自旋$s^z_{\{1,2\}}\in[-0.9, 0.9]$; 并且包括$(\ell, |m|) = \{(2, 2), (2, 1), (3, 3), (3, 2), (4, 4)\}$模式以及$\ell = 3, |m| = 2$谐波中的模式混合效应。我们使用Summit超级计算机上的96个NVIDIA V100 GPU进行分布式训练，在22小时内训练这些AI分类器。然后我们使用迁移学习创建了AI预测器，用于估计所有AI分类器集合识别出的潜在二进制黑洞的总质量。我们使用了这个集合、3个AI分类器和2个预测器来处理一个为期一年的测试集，其中注入了30万个信号。

    We introduce spatiotemporal-graph models that concurrently process data from the twin advanced LIGO detectors and the advanced Virgo detector. We trained these AI classifiers with 2.4 million \texttt{IMRPhenomXPHM} waveforms that describe quasi-circular, spinning, non-precessing binary black hole mergers with component masses $m_{\{1,2\}}\in[3M_\odot, 50 M_\odot]$, and individual spins $s^z_{\{1,2\}}\in[-0.9, 0.9]$; and which include the $(\ell, |m|) = \{(2, 2), (2, 1), (3, 3), (3, 2), (4, 4)\}$ modes, and mode mixing effects in the $\ell = 3, |m| = 2$ harmonics. We trained these AI classifiers within 22 hours using distributed training over 96 NVIDIA V100 GPUs in the Summit supercomputer. We then used transfer learning to create AI predictors that estimate the total mass of potential binary black holes identified by all AI classifiers in the ensemble. We used this ensemble, 3 AI classifiers and 2 predictors, to process a year-long test set in which we injected 300,000 signals. This ye
    
[^24]: 大型语言模型微调中的LoRA集成

    LoRA ensembles for large language model fine-tuning. (arXiv:2310.00035v1 [cs.LG])

    [http://arxiv.org/abs/2310.00035](http://arxiv.org/abs/2310.00035)

    本文提出了一种使用低秩适配器（LoRA）的集成方法，用于解决大型语言模型微调中存在的不确定性量化问题，并提供了一个参数高效的微调技术。这种方法可以构建大规模的LoRA适配器集成，并具有与基础预训练模型相近的计算资源需求。

    

    细调的语言模型往往表现出较差的不确定性量化，表现为过于自信、校准不佳以及对测试数据或超出分布的样本的预测结果不可靠。为了缓解这个问题，本文提出了一种使用低秩适配器（LoRA）的集成方法，该方法是一种参数高效的微调技术。这些低秩适配器表示的参数数量非常小，比基础预训练模型小几个数量级。因此，可以构建大规模的LoRA适配器集成，几乎具有相同的计算资源需求。

    Finetuned LLMs often exhibit poor uncertainty quantification, manifesting as overconfidence, poor calibration, and unreliable prediction results on test data or out-of-distribution samples. One approach commonly used in vision for alleviating this issue is a deep ensemble, which constructs an ensemble by training the same model multiple times using different random initializations. However, there is a huge challenge to ensembling LLMs: the most effective LLMs are very, very large. Keeping a single LLM in memory is already challenging enough: keeping an ensemble of e.g. 5 LLMs in memory is impossible in many settings. To address these issues, we propose an ensemble approach using Low-Rank Adapters (LoRA), a parameter-efficient fine-tuning technique. Critically, these low-rank adapters represent a very small number of parameters, orders of magnitude less than the underlying pre-trained model. Thus, it is possible to construct large ensembles of LoRA adapters with almost the same computat
    
[^25]: PB-LLM: 部分二值化大型语言模型

    PB-LLM: Partially Binarized Large Language Models. (arXiv:2310.00034v1 [cs.LG])

    [http://arxiv.org/abs/2310.00034](http://arxiv.org/abs/2310.00034)

    本文提出的PB-LLM是一种部分二值化的大型语言模型压缩方法，可以在保持语言推理能力的同时实现极低比特量化，并通过后训练量化和量化感知训练等方法恢复量化LLMM的容量。

    

    本文探讨了网络二值化，一种压缩模型权重为单个比特的量化的激进形式，专门应用于大型语言模型（LLMs）的压缩。由于之前的二值化方法会导致LLMs崩溃，我们提出了一种新颖的方法，部分二值化LLM（PB-LLM），可以实现极低比特量化，并同时保持量化LLMs的语言推理能力。具体而言，我们的研究首先揭示了现有二值化算法的原生应用的无效性，并强调了显著权重在实现低位量化中的重要作用。因此，PB-LLM在二进制化过程中过滤了一小部分显著权重，将它们分配到高位存储中，即部分二值化。PB-LLM在后训练量化（PTQ）和量化感知训练（QAT）的角度分析后，扩展了恢复量化LLMM容量的能力。在PTQ下，结合了GPTQ的概念，我们重构了...

    This paper explores network binarization, a radical form of quantization, compressing model weights to a single bit, specifically for Large Language Models (LLMs) compression. Due to previous binarization methods collapsing LLMs, we propose a novel approach, Partially-Binarized LLM (PB-LLM), which can achieve extreme low-bit quantization while maintaining the linguistic reasoning capacity of quantized LLMs. Specifically, our exploration first uncovers the ineffectiveness of naive applications of existing binarization algorithms and highlights the imperative role of salient weights in achieving low-bit quantization. Thus, PB-LLM filters a small ratio of salient weights during binarization, allocating them to higher-bit storage, i.e., partially-binarization. PB-LLM is extended to recover the capacities of quantized LMMs, by analyzing from the perspective of post-training quantization (PTQ) and quantization-aware training (QAT). Under PTQ, combining the concepts from GPTQ, we reconstruct 
    
[^26]: 融入人类风险认知的对抗驾驶行为生成技术用于自动驾驶车辆评估

    Adversarial Driving Behavior Generation Incorporating Human Risk Cognition for Autonomous Vehicle Evaluation. (arXiv:2310.00029v1 [cs.AI])

    [http://arxiv.org/abs/2310.00029](http://arxiv.org/abs/2310.00029)

    本论文开发了一种新型框架，结合人类风险认知来生成对手驾驶行为，用于评估自动驾驶车辆的有效性和弱点。

    

    这篇论文关注于开发一种用于生成对手驾驶行为的新型框架，以暴露出自动驾驶车辆面对的有效和合理的风险事件。具体而言，采用强化学习与累积前景理论相结合的方法来学习对手行为，累积前景理论能够表示人类的风险认知。然后，提出了扩展版本的深度确定性策略梯度技术，用于训练对手策略，同时保证了训练的稳定性。在高保真的硬件在环（HiL）平台上进行了基于并线情景的对比案例研究，结果证明了对手的有效性，可以推断出被测试自动驾驶车辆的弱点。

    Autonomous vehicle (AV) evaluation has been the subject of increased interest in recent years both in industry and in academia. This paper focuses on the development of a novel framework for generating adversarial driving behavior of background vehicle interfering against the AV to expose effective and rational risky events. Specifically, the adversarial behavior is learned by a reinforcement learning (RL) approach incorporated with the cumulative prospect theory (CPT) which allows representation of human risk cognition. Then, the extended version of deep deterministic policy gradient (DDPG) technique is proposed for training the adversarial policy while ensuring training stability as the CPT action-value function is leveraged. A comparative case study regarding the cut-in scenario is conducted on a high fidelity Hardware-in-the-Loop (HiL) platform and the results demonstrate the adversarial effectiveness to infer the weakness of the tested AV.
    
[^27]: De-SaTE：用于锂离子电池健康预测的去噪自注意力变换编码器

    De-SaTE: Denoising Self-attention Transformer Encoders for Li-ion Battery Health Prognostics. (arXiv:2310.00023v1 [cs.LG])

    [http://arxiv.org/abs/2310.00023](http://arxiv.org/abs/2310.00023)

    本研究提出了De-SaTE方法，通过利用多个去噪模块以及自注意力变换编码器，准确预测锂离子电池的剩余寿命（RUL），为预防性维护和预测性分析提供关键指标估计。

    

    锂离子电池在各个行业中得到了广泛应用，从为便携式电子设备供电到推动电动汽车和支持能源存储系统。有效管理锂离子电池的一个核心挑战是准确预测其剩余寿命（RUL），这是预防性维护和预测性分析的关键指标。本研究提出了一种新的方法，利用多个去噪模块的能量，每个模块都经过训练来处理电池数据中常见的噪声类型。具体而言，我们使用去噪自动编码器和小波去噪器来生成编码/分解表示，然后将其通过专用自注意力变换编码器进行处理。在NASA和CALCE数据集上进行了大量实验后，我们能够表征多种噪声模式下的广泛健康指标估计。我们发现我们报告的误差

    Lithium Ion (Li-ion) batteries have gained widespread popularity across various industries, from powering portable electronic devices to propelling electric vehicles and supporting energy storage systems. A central challenge in managing Li-ion batteries effectively is accurately predicting their Remaining Useful Life (RUL), which is a critical measure for proactive maintenance and predictive analytics. This study presents a novel approach that harnesses the power of multiple denoising modules, each trained to address specific types of noise commonly encountered in battery data. Specifically we use a denoising auto-encoder and a wavelet denoiser to generate encoded/decomposed representations, which are subsequently processed through dedicated self-attention transformer encoders. After extensive experimentation on the NASA and CALCE datasets, we are able to characterize a broad spectrum of health indicator estimations under a set of diverse noise patterns. We find that our reported error
    
[^28]: 自适应驾驶中基于领域匹配的协同感知的通信

    Adaptive Communications in Collaborative Perception with Domain Alignment for Autonomous Driving. (arXiv:2310.00013v1 [cs.AI])

    [http://arxiv.org/abs/2310.00013](http://arxiv.org/abs/2310.00013)

    这篇论文提出了一个通信的协同感知框架ACC-DA，通过动态调整通信图和自适应数据重构机制来增强自动驾驶中的感知能力。

    

    通过通信允许车辆交换补充信息，多个连接的自动驾驶车辆之间的协同感知可以极大地增强感知能力。尽管之前的方法取得了进展，但由于通道变化和协同车辆之间的数据异构性，仍然存在挑战。为了解决这些问题，我们提出了ACC-DA，一个通道感知的协同感知框架，它可以动态调整通信图并最小化平均传输延迟，同时减轻数据异构性带来的副作用。我们的创新点包括三个方面。首先，我们设计了一种最小化传输延迟的方法，根据不同的通道信息状态构建通信图并最小化传输延迟。然后，我们提出了一种自适应数据重构机制，可以动态调整码率-畸变折衷以增强感知效率。此外，它最小化了时域丢失。

    Collaborative perception among multiple connected and autonomous vehicles can greatly enhance perceptive capabilities by allowing vehicles to exchange supplementary information via communications. Despite advances in previous approaches, challenges still remain due to channel variations and data heterogeneity among collaborative vehicles. To address these issues, we propose ACC-DA, a channel-aware collaborative perception framework to dynamically adjust the communication graph and minimize the average transmission delay while mitigating the side effects from the data heterogeneity. Our novelties lie in three aspects. We first design a transmission delay minimization method, which can construct the communication graph and minimize the transmission delay according to different channel information state. We then propose an adaptive data reconstruction mechanism, which can dynamically adjust the rate-distortion trade-off to enhance perception efficiency. Moreover, it minimizes the temporal
    
[^29]: 在球面上无运算符均衡

    Operator-free Equilibrium on the Sphere. (arXiv:2310.00012v1 [math.NA])

    [http://arxiv.org/abs/2310.00012](http://arxiv.org/abs/2310.00012)

    本论文通过引入一个新的准则，建立连续的、可导的核函数，简化了在球面上等分点集的计算，实现了在无需涉及运算符的情况下对潜在点系统进行探索，并得到了与蒙特卡洛方法相比更高效的近似目标的方法。

    

    我们提出了一种广义最小差异度，它源于Legendre的ODE和球谐函数理论，提供了一个新的球面上等分点集的准则。建立了一个连续的、可导的核函数，以简化广义最小差异度的计算。我们考虑了通过Pycke统计生成的确定性点来对球面上的Frank函数进行积分，并研究了不同核函数嵌入的点系统的差异。进行了定量实验并对结果进行了分析。我们推导的模型可以通过导数探索具有最小差异度的潜在点系统，而无需涉及伪微分算子和Beltrami算子。与蒙特卡洛方法生成的随机点相比，我们的方法只需要少量点即可在任意维度中近似目标。

    We propose a generalized minimum discrepancy, which derives from Legendre's ODE and spherical harmonic theoretics to provide a new criterion of equidistributed pointsets on the sphere. A continuous and derivative kernel in terms of elementary functions is established to simplify the computation of the generalized minimum discrepancy. We consider the deterministic point generated from Pycke's statistics to integrate a Franke function for the sphere and investigate the discrepancies of points systems embedding with different kernels. Quantitive experiments are conducted and the results are analyzed. Our deduced model can explore latent point systems, that have the minimum discrepancy without the involvement of pseudodifferential operators and Beltrami operators, by the use of derivatives. Compared to the random point generated from the Monte Carlo method, only a few points generated by our method are required to approximate the target in arbitrary dimensions.
    
[^30]: 人工共情分类：深度学习技术、数据集和评估标准的综述

    Artificial Empathy Classification: A Survey of Deep Learning Techniques, Datasets, and Evaluation Scales. (arXiv:2310.00010v1 [cs.RO])

    [http://arxiv.org/abs/2310.00010](http://arxiv.org/abs/2310.00010)

    这篇论文综述了人工共情的分类研究，介绍了深度学习技术、数据集和评估标准的最新进展，指出训练人工共情的标准流程包括情绪识别、分析和响应动作。其中深度学习技术在虚拟代理和机器人中的应用有较高影响力。

    

    近十年来，机器学习（ML）和辅助发展机器人学（ADR）领域的研究人员对人工共情（AE）作为可能的未来人机交互（HRI）范式产生了兴趣。人类从出生开始就学会共情，因此在机器人和智能机器中灌输这种感觉是具有挑战性的。然而，通过对大量数据和时间进行训练，在某种程度上可以使机器人模仿共情。人工共情的标准工作流程包括三个阶段：1）使用从视频或文本数据中提取的特征进行情绪识别（ER），2）分析感知的情绪或共情程度以选择最佳行动方案，3）执行响应动作。最近的研究显示，使用虚拟代理或机器人的AE常常涉及深度学习（DL）技术。

    From the last decade, researchers in the field of machine learning (ML) and assistive developmental robotics (ADR) have taken an interest in artificial empathy (AE) as a possible future paradigm for human-robot interaction (HRI). Humans learn empathy since birth, therefore, it is challenging to instill this sense in robots and intelligent machines. Nevertheless, by training over a vast amount of data and time, imitating empathy, to a certain extent, can be possible for robots. Training techniques for AE, along with findings from the field of empathetic AI research, are ever-evolving. The standard workflow for artificial empathy consists of three stages: 1) Emotion Recognition (ER) using the retrieved features from video or textual data, 2) analyzing the perceived emotion or degree of empathy to choose the best course of action, and 3) carrying out a response action. Recent studies that show AE being used with virtual agents or robots often include Deep Learning (DL) techniques. For ins
    
[^31]: LLM基于视频扩散模型

    LLM-grounded Video Diffusion Models. (arXiv:2309.17444v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2309.17444](http://arxiv.org/abs/2309.17444)

    使用LLM-grounded Video Diffusion (LVD)模型，通过先生成动态场景布局，再通过这些布局指导视频生成的扩散模型，解决了当前模型在复杂的时空提示和不正确的运动生成方面的困难。

    

    文字条件下的扩散模型已经成为神经视频生成的一个有希望的工具。然而，目前的模型仍然在复杂的时空提示方面存在困难，通常生成受限制或不正确的运动（例如，甚至缺乏从左向右移动的物体的提示能力）。为了解决这些限制，我们引入了LLM基于视频扩散（LVD）。LVD不直接从文本输入中生成视频，而是首先利用大型语言模型（LLM）根据文本输入生成动态场景布局，然后使用生成的布局来指导视频生成的扩散模型。我们展示了LLM能够从单纯的文本中理解复杂的时空动态，并生成与实际世界中通常观察到的提示和物体运动模式密切对齐的布局。然后，我们提出通过调整注意力图来指导视频扩散模型与这些布局进行交互。我们的方法无需训练。

    Text-conditioned diffusion models have emerged as a promising tool for neural video generation. However, current models still struggle with intricate spatiotemporal prompts and often generate restricted or incorrect motion (e.g., even lacking the ability to be prompted for objects moving from left to right). To address these limitations, we introduce LLM-grounded Video Diffusion (LVD). Instead of directly generating videos from the text inputs, LVD first leverages a large language model (LLM) to generate dynamic scene layouts based on the text inputs and subsequently uses the generated layouts to guide a diffusion model for video generation. We show that LLMs are able to understand complex spatiotemporal dynamics from text alone and generate layouts that align closely with both the prompts and the object motion patterns typically observed in the real world. We then propose to guide video diffusion models with these layouts by adjusting the attention maps. Our approach is training-free 
    
[^32]: 学习具有时空图神经网络的去中心化群集控制器

    Learning Decentralized Flocking Controllers with Spatio-Temporal Graph Neural Network. (arXiv:2309.17437v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2309.17437](http://arxiv.org/abs/2309.17437)

    本论文提出了一种名为STGNN的时空图神经网络，该网络在去中心化群集控制中通过结合空间和时间扩展来更好地模拟集中式控制策略，从而提高了预测的效果和准确性。

    

    最近的一系列研究探索了在群集机器人中使用图神经网络(GNN)进行去中心化控制。然而，观察到仅依靠相邻状态是不足以模仿集中式控制策略的。为了解决这个问题，之前的研究提出了将$L$-跳延迟状态纳入计算中。虽然这种方法很有前途，但它可能导致远离的群体成员之间缺乏一致性，小集群的形成，从而导致连贯的集群行为失败。相反，我们的方法利用了时空GNN，命名为STGNN，它包括了空间和时间的扩展。空间扩展收集来自远处领航者的延迟状态，而时间扩展则纳入来自相邻领航者的先前状态。通过从这两个扩展中收集更广泛、更全面的信息，可以实现更有效、更准确的预测。我们开发了一个...

    Recently a line of researches has delved the use of graph neural networks (GNNs) for decentralized control in swarm robotics. However, it has been observed that relying solely on the states of immediate neighbors is insufficient to imitate a centralized control policy. To address this limitation, prior studies proposed incorporating $L$-hop delayed states into the computation. While this approach shows promise, it can lead to a lack of consensus among distant flock members and the formation of small clusters, consequently resulting in the failure of cohesive flocking behaviors. Instead, our approach leverages spatiotemporal GNN, named STGNN that encompasses both spatial and temporal expansions. The spatial expansion collects delayed states from distant neighbors, while the temporal expansion incorporates previous states from immediate neighbors. The broader and more comprehensive information gathered from both expansions results in more effective and accurate predictions. We develop an
    
[^33]: 数据过滤网络

    Data Filtering Networks. (arXiv:2309.17425v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2309.17425](http://arxiv.org/abs/2309.17425)

    本文研究了学习数据过滤网络用于筛选大型未策划数据集的问题，并构建了新的数据过滤网络，从而产生最先进的图像-文本数据集。

    

    大型训练集已成为机器学习的基石，并为语言建模和多模态学习的最新进展奠定了基础。虽然对于预训练的数据采集仍然是一种常见的范式，但数据策划往往仍然是临时的。一种常见的方法是首先从网络上收集大量数据，然后通过各种启发式方法将此候选池筛选到实际的训练集中。在这项工作中，我们研究了学习数据过滤网络（DFN）用于筛选大型未策划数据集的问题。我们的主要发现是，用于筛选的网络的质量与其在下游任务上的表现是不同的：例如，一个在ImageNet上表现良好的模型可能会产生比一个在ImageNet上准确率较低但在一小部分高质量数据上进行训练的模型更差的训练集。基于我们的洞察力，我们构建了新的数据过滤网络，从而产生了最先进的图像-文本数据集。具体而言，我们表现最佳的数据集DFN-5B使我们能够进行训练。

    Large training sets have become a cornerstone of machine learning and are the foundation for recent advances in language modeling and multimodal learning. While data curation for pre-training is often still ad-hoc, one common paradigm is to first collect a massive pool of data from the Web and then filter this candidate pool down to an actual training set via various heuristics. In this work, we study the problem of learning a data filtering network (DFN) for this second step of filtering a large uncurated dataset. Our key finding is that the quality of a network for filtering is distinct from its performance on downstream tasks: for instance, a model that performs well on ImageNet can yield worse training sets than a model with low ImageNet accuracy that is trained on a small amount of high-quality data. Based on our insights, we construct new data filtering networks that induce state-of-the-art image-text datasets. Specifically, our best performing dataset DFN-5B enables us to train 
    
[^34]: PlaceNav: 通过地点识别进行拓扑导航

    PlaceNav: Topological Navigation through Place Recognition. (arXiv:2309.17260v1 [cs.RO])

    [http://arxiv.org/abs/2309.17260](http://arxiv.org/abs/2309.17260)

    PlaceNav是一种通过地点识别进行拓扑导航的方法，将机器人无关部分分为导航特定和通用的计算机视觉组件，通过使用非机器人来源的大规模数据集增加训练数据的可用性，同时通过地点识别来提高导航性能。新模型的性能提高了76%。

    

    最近的研究结果表明，将拓扑导航分为机器人无关和机器人特定的组件可以提高导航性能，通过使用不同类型机器人收集的数据来训练机器人无关部分。然而，导航方法仍受到适合训练数据的稀缺性和计算缩放性差的限制。在本文中，我们提出了一个名为PlaceNav的方法，将机器人无关部分分为导航特定和通用的计算机视觉组件。我们利用视觉地点识别来选择拓扑导航流程中的子目标。这使得子目标选择更高效，并能够利用非机器人来源的大规模数据集增加训练数据的可用性。地点识别使得贝叶斯滤波成为可能，进一步通过增加子目标的时间一致性来提高导航性能。我们的实验结果验证了这一设计，并且新模型的性能提高了76%。

    Recent results suggest that splitting topological navigation into robot-independent and robot-specific components improves navigation performance by enabling the robot-independent part to be trained with data collected by different robot types. However, the navigation methods are still limited by the scarcity of suitable training data and suffer from poor computational scaling. In this work, we present~\methodname, subdividing the robot-independent part into navigation-specific and generic computer vision components. We utilize visual place recognition for the subgoal selection of the topological navigation pipeline. This makes subgoal selection more efficient and enables leveraging large-scale datasets from non-robotics sources, increasing training data availability. Bayes filtering, enabled by place recognition, further improves navigation performance by increasing the temporal consistency of subgoals. Our experimental results verify the design and the new model obtains a 76% higher 
    
[^35]: LLM能否有效利用结构信息进行图学习：何时何地。

    Can LLMs Effectively Leverage Structural Information for Graph Learning: When and Why. (arXiv:2309.16595v1 [cs.LG])

    [http://arxiv.org/abs/2309.16595](http://arxiv.org/abs/2309.16595)

    本文研究了大型语言模型（LLM）在图数据中的应用，发现LLM可以从结构信息中受益，尤其是在文本节点特征缺乏的情况下，而LLM的性能与数据泄露没有显著相关。

    

    本文研究了大型语言模型（LLM）在结构化数据（特别是图数据）上的应用，这是LLM文献中尚未充分探索的重要数据形态。我们旨在了解在节点分类任务中，何时何地引入图数据中的结构信息可以提高LLM的预测性能。为了解决“何时”问题，我们研究了多种编码结构信息的提示方法，设置中文本节点特征丰富或稀缺。对于“为什么”问题，我们探讨了LLM性能的两个潜在因素：数据泄露和同质性。我们的研究结果表明：（i）LLM可以从结构信息中受益，尤其是在文本节点特征缺乏的情况下；（ii）没有实质性的证据表明LLM性能与数据泄露有显著相关；（iii）LLM在目标节点上的性能与正向相关。

    This paper studies Large Language Models (LLMs) for structured data--particularly graphs--a crucial data modality that remains underexplored in the LLM literature. We aim to understand when and why the incorporation of structural information inherent in graph data can improve the prediction performance of LLMs on node classification tasks. To address the ``when'' question, we examine a variety of prompting methods for encoding structural information, in settings where textual node features are either rich or scarce. For the ``why'' questions, we probe into two potential contributing factors to the LLM performance: data leakage and homophily. Our exploration of these questions reveals that (i) LLMs can benefit from structural information, especially when textual node features are scarce; (ii) there is no substantial evidence indicating that the performance of LLMs is significantly attributed to data leakage; and (iii) the performance of LLMs on a target node is strongly positively relat
    
[^36]: 多智能体系统中的合作动力学：探索具有均值场均衡的博弈理论情景

    Cooperation Dynamics in Multi-Agent Systems: Exploring Game-Theoretic Scenarios with Mean-Field Equilibria. (arXiv:2309.16263v1 [cs.GT])

    [http://arxiv.org/abs/2309.16263](http://arxiv.org/abs/2309.16263)

    本文研究在多智能体系统中激发合作的策略和方法，通过分析现有的合作策略和引入鼓励团队回报的修改，解决了在分布式系统中存在的现实困境。同时，利用均值场博弈理论，建立了无限大智能体集合中的平衡解和奖励结构。

    

    合作是多智能体系统（MAS）和多智能体强化学习（MARL）中的基本要素，通常要求智能体在个体收益和集体回报之间保持平衡。本文旨在研究在博弈理论情景中激发合作的策略，例如迭代囚徒困境，在这种情况下，智能体必须优化个体和团队的结果。分析了现有的合作策略对于促进重复博弈中团队导向行为的有效性。提出了一种修改，即鼓励团队回报也将导致更高的个体收益，解决了分布式系统中存在的现实困境。研究还扩展到智能体人口指数增长的情景（$N \longrightarrow +\infty$），在这种情况下，传统计算和平衡确定具有挑战性。利用均值场博弈理论，建立了无限大智能体集合中的平衡解和奖励结构。

    Cooperation is fundamental in Multi-Agent Systems (MAS) and Multi-Agent Reinforcement Learning (MARL), often requiring agents to balance individual gains with collective rewards. In this regard, this paper aims to investigate strategies to invoke cooperation in game-theoretic scenarios, namely the Iterated Prisoner's Dilemma, where agents must optimize both individual and group outcomes. Existing cooperative strategies are analyzed for their effectiveness in promoting group-oriented behavior in repeated games. Modifications are proposed where encouraging group rewards will also result in a higher individual gain, addressing real-world dilemmas seen in distributed systems. The study extends to scenarios with exponentially growing agent populations ($N \longrightarrow +\infty$), where traditional computation and equilibrium determination are challenging. Leveraging mean-field game theory, equilibrium solutions and reward structures are established for infinitely large agent sets in repea
    
[^37]: Lyra: 自动定理证明中的双重修正策略的编排

    Lyra: Orchestrating Dual Correction in Automated Theorem Proving. (arXiv:2309.15806v1 [cs.CL])

    [http://arxiv.org/abs/2309.15806](http://arxiv.org/abs/2309.15806)

    Lyra是一种新的框架，通过引入工具修正和猜想修正两种机制，增强了大规模语言模型在形式化定理证明领域的有效性，减轻了幻觉，并提高了证明的准确性。

    

    大规模语言模型（LLMs）为形式化定理证明领域提供了一个有趣的探索途径。然而，它们的全部潜力，尤其是关于幻觉的减轻和通过证明器错误消息的细化，仍然是一个尚未深入研究的领域。为了增强LLMs在该领域的有效性，我们引入了Lyra，一种采用两种不同修正机制的新框架：工具修正（TC）和猜想修正（CC）。为了在形式证明的后处理中实现工具修正，我们利用先前的知识来利用预定义的证明工具（如Sledgehammer）来指导替换不正确的工具。工具修正显著减轻了幻觉，从而提高了证明的整体准确性。此外，我们引入了猜想修正，一种错误反馈机制，旨在与证明器互动，通过证明器的错误消息进一步完善形式证明的猜想。

    Large Language Models (LLMs) present an intriguing avenue for exploration in the field of formal theorem proving. Nevertheless, their full potential, particularly concerning the mitigation of hallucinations and refinement through prover error messages, remains an area that has yet to be thoroughly investigated. To enhance the effectiveness of LLMs in the field, we introduce the Lyra, a new framework that employs two distinct correction mechanisms: Tool Correction (TC) and Conjecture Correction (CC). To implement Tool Correction in the post-processing of formal proofs, we leverage prior knowledge to utilize predefined prover tools (e.g., Sledgehammer) for guiding the replacement of incorrect tools. Tool Correction significantly contributes to mitigating hallucinations, thereby improving the overall accuracy of the proof. In addition, we introduce Conjecture Correction, an error feedback mechanism designed to interact with prover to refine formal proof conjectures with prover error messa
    
[^38]: 基于似然比的任务预测的类增量学习

    Class Incremental Learning via Likelihood Ratio Based Task Prediction. (arXiv:2309.15048v1 [cs.LG])

    [http://arxiv.org/abs/2309.15048](http://arxiv.org/abs/2309.15048)

    该论文提出了一种基于似然比的任务预测的类增量学习方法，利用离群检测器进行任务标识预测，解决了无任务标识符的测试样本的任务预测问题。

    

    类增量学习是一种具有挑战性的不断学习的设置，通过顺序学习一系列任务。每个任务由一组唯一的类组成。类增量学习的关键特点是，在测试时不提供每个测试样本的任务标识符（或任务ID）。为每个测试样本预测任务ID是一个具有挑战性的问题。一种新兴的理论上合理且有效的方法是根据任务增量学习的方法，在共享网络中为所有任务训练每个任务的任务特定模型，以处理遗忘。该方法中每个任务的模型是一个非常规分类器而不是传统分类器的离群检测器。离群检测器可以对任务内（分布内（IND））的类进行预测和识别离群数据。在推断期间，离群检测能力是每个测试样本的任务ID预测的关键。然而，本文认为使用传统的离群检测器进行任务ID预测是次优的。

    Class incremental learning (CIL) is a challenging setting of continual learning, which learns a series of tasks sequentially. Each task consists of a set of unique classes. The key feature of CIL is that no task identifier (or task-id) is provided at test time for each test sample. Predicting the task-id for each test sample is a challenging problem. An emerging theoretically justified and effective approach is to train a task-specific model for each task in a shared network for all tasks based on a task-incremental learning (TIL) method to deal with forgetting. The model for each task in this approach is an out-of-distribution (OOD) detector rather than a conventional classifier. The OOD detector can perform both within-task (in-distribution (IND)) class prediction and OOD detection. The OOD detection capability is the key for task-id prediction during inference for each test sample. However, this paper argues that using a traditional OOD detector for task-id prediction is sub-optimal
    
[^39]: 人工生成的演示是否对于上下文学习有必要？

    Are Human-generated Demonstrations Necessary for In-context Learning?. (arXiv:2309.14681v1 [cs.LG])

    [http://arxiv.org/abs/2309.14681](http://arxiv.org/abs/2309.14681)

    本文研究了上下文学习中人工生成的演示是否有必要，并提出了一种新的自反思提示策略（SEC），通过这种策略，大型语言模型（LLMs）可以自行生成演示和最终输出，避免了手动生成过程的复杂性。

    

    尽管大型语言模型（LLMs）具备良好的少样本能力，但在上下文学习（ICL）的标准范式中存在以下弊端：易受选定演示的影响，生成这些演示的复杂性。本文提出了对于ICL，人工生成的演示是否有必要的基本问题，并提出了自反思提示策略（SEC），这是一种不依赖人工演示的范例。SEC的关键点在于，不使用手工制作的示例作为ICL中的演示，而是要求LLMs首先自行创建演示，然后生成最终输出。SEC是一种灵活的框架，可适应原始ICL和“思维链”（CoT），并且更加便捷：因为可以节省示例和理由的手动生成过程。在算术推理、常识推理和多任务语言理解方面进行了大量实验。

    Despite the promising few-shot ability of large language models (LLMs), the standard paradigm of In-context Learning (ICL) suffers the disadvantages of susceptibility to selected demonstrations and the intricacy to generate these demonstrations. In this paper, we raise the fundamental question that whether human-generated demonstrations are necessary for ICL. To answer this question, we propose self-contemplation prompting strategy (SEC), a paradigm free from human-crafted demonstrations. The key point of SEC is that, instead of using hand-crafted examples as demonstrations in ICL, SEC asks LLMs to first create demonstrations on their own, based on which the final output is generated. SEC is a flexible framework and can be adapted to both the vanilla ICL and the chain-of-thought (CoT), but with greater ease: as the manual-generation process of both examples and rationale can be saved. Extensive experiments in arithmetic reasoning, commonsense reasoning, multi-task language understandin
    
[^40]: 联合音频和语音理解

    Joint Audio and Speech Understanding. (arXiv:2309.14405v1 [cs.SD])

    [http://arxiv.org/abs/2309.14405](http://arxiv.org/abs/2309.14405)

    LTU-AS是一个具有普适音频感知和高级推理能力的机器学习模型，可以同时识别和联合理解口语文本、语音声音学和非语音音频事件。

    

    人类周围充斥着包括语音和非语音声音在内的音频信号。对语音和非语音音频事件的识别和理解，以及对它们之间关系的深刻理解，构成了基本的认知能力。我们首次构建了一个名为LTU-AS的机器学习模型，它具有类似于人类的普遍音频感知和高级推理能力。具体而言，通过将Whisper作为感知模块和LLaMA作为推理模块进行集成，LTU-AS可以同时识别和联合理解口语文本、语音声音学以及非语音音频事件 - 几乎可以从音频信号中感知到的一切。

    Humans are surrounded by audio signals that include both speech and non-speech sounds. The recognition and understanding of speech and non-speech audio events, along with a profound comprehension of the relationship between them, constitute fundamental cognitive capabilities. For the first time, we build a machine learning model, called LTU-AS, that has a conceptually similar universal audio perception and advanced reasoning ability. Specifically, by integrating Whisper as a perception module and LLaMA as a reasoning module, LTU-AS can simultaneously recognize and jointly understand spoken text, speech paralinguistics, and non-speech audio events - almost everything perceivable from audio signals.
    
[^41]: 语言模型的物理学：第3.2部分，知识操控

    Physics of Language Models: Part 3.2, Knowledge Manipulation. (arXiv:2309.14402v1 [cs.CL])

    [http://arxiv.org/abs/2309.14402](http://arxiv.org/abs/2309.14402)

    本文研究了语言模型在推理过程中操控知识的能力，发现预训练模型在知识检索方面表现出色，但在简单的分类、比较和逆向搜索任务中表现不佳。作者还提供了一个合成数据集进行实验，验证了这些内在的弱点：语言模型无法高效地操控知识。

    

    语言模型可以存储大量事实知识，但它们在使用这些知识进行逻辑推理方面的能力仍然存在问题。本文探讨了语言模型在推理过程中操控其存储知识的能力。我们重点研究了四种操控类型：检索（例如，“A的属性X是什么”）、分类（例如，“A的属性X是奇数还是偶数”）、比较（例如，“在属性X中A是否大于B”）和逆向搜索（例如，“哪个人的属性X等于T”）。我们观察到，像GPT2/3/4这样的预训练语言模型在知识检索方面表现出色，但在简单的分类或比较任务中很难胜任，除非在训练和推理过程中采用了Chain of Thoughts（CoTs）。无论提示是什么，它们在逆向知识搜索中表现都很差。我们的主要贡献是一个为控制实验而设计的合成数据集，证实了这些内在的弱点：语言模型无法高效地操控知识。

    Language models can store vast amounts of factual knowledge, but their ability to use this knowledge for logical reasoning remains questionable. This paper explores a language model's ability to manipulate its stored knowledge during inference. We focus on four manipulation types: retrieval (e.g., "What is person A's attribute X"), classification (e.g., "Is A's attribute X even or odd?"), comparison (e.g., "Is A greater than B in attribute X?") and inverse search (e.g., "Which person's attribute X equals T?")  We observe that pre-trained language models like GPT2/3/4 excel in knowledge retrieval but struggle with simple classification or comparison tasks unless Chain of Thoughts (CoTs) are employed during both training and inference. They also perform poorly in inverse knowledge search, irrespective of the prompts. Our primary contribution is a synthetic dataset for a controlled experiment that confirms these inherent weaknesses: a language model cannot efficiently manipulate knowledge
    
[^42]: LinGCN: 结构化的线性化图卷积网络用于同态加密推断

    LinGCN: Structural Linearized Graph Convolutional Network for Homomorphically Encrypted Inference. (arXiv:2309.14331v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2309.14331](http://arxiv.org/abs/2309.14331)

    LinGCN是一个旨在减少乘法深度和优化HE基于GCN推断性能的框架，通过结构化线性化算法和参数化的离散指示函数的联合训练，实现细粒度的节点级非线性位置选择。

    

    图卷积网络（GCN）模型的规模增长已经在个人医疗和金融系统等多个应用领域取得了超越人类表现的革命性进展。然而，在云端部署GCN引发了对客户数据可能受到对抗性攻击的隐私问题。为了解决安全问题，采用同态加密（HE）的隐私保护机器学习（PPML）可以确保敏感客户数据的安全。然而，在实际应用中，这引入了相当大的计算开销。为了解决这些挑战，我们提出了LinGCN，这是一个旨在减少乘法深度并优化HE基于GCN推断性能的框架。LinGCN围绕三个关键要素展开：（1）可微的结构化线性化算法，搭配参数化的离散指示函数，通过与模型权重一起进行联合训练以满足优化目标。这种策略促进了细粒度的节点级非线性位置选择，从而实现了

    The growth of Graph Convolution Network (GCN) model sizes has revolutionized numerous applications, surpassing human performance in areas such as personal healthcare and financial systems. The deployment of GCNs in the cloud raises privacy concerns due to potential adversarial attacks on client data. To address security concerns, Privacy-Preserving Machine Learning (PPML) using Homomorphic Encryption (HE) secures sensitive client data. However, it introduces substantial computational overhead in practical applications. To tackle those challenges, we present LinGCN, a framework designed to reduce multiplication depth and optimize the performance of HE based GCN inference. LinGCN is structured around three key elements: (1) A differentiable structural linearization algorithm, complemented by a parameterized discrete indicator function, co-trained with model weights to meet the optimization goal. This strategy promotes fine-grained node-level non-linear location selection, resulting in a 
    
[^43]: 技能检测：评估角色扮演游戏中游戏主持模型的一些考虑

    Skill Check: Some Considerations on the Evaluation of Gamemastering Models for Role-playing Games. (arXiv:2309.13702v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2309.13702](http://arxiv.org/abs/2309.13702)

    本文讨论了从交互式叙事和自然语言处理的角度对角色扮演游戏中游戏主持进行建模的挑战，并提出了三个测试类别来评估对话系统。

    

    在角色扮演游戏中，游戏主持（GM）是负责游戏的玩家，必须设计玩家面临的挑战并讲述他们行动的结果。本文从交互式叙事和自然语言处理的角度讨论了对GM进行建模的挑战。在讨论这些挑战后，我们提出了三个测试类别来评估这些对话系统，并使用ChatGPT、Bard和OpenAssistant作为开箱即用的GM进行测试。

    In role-playing games a Game Master (GM) is the player in charge of the game, who must design the challenges the players face and narrate the outcomes of their actions. In this work we discuss some challenges to model GMs from an Interactive Storytelling and Natural Language Processing perspective. Following those challenges we propose three test categories to evaluate such dialogue systems, and we use them to test ChatGPT, Bard and OpenAssistant as out-of-the-box GMs.
    
[^44]: 模型无关的图神经网络用于整合局部和全局信息的研究

    A Model-Agnostic Graph Neural Network for Integrating Local and Global Information. (arXiv:2309.13459v1 [stat.ML])

    [http://arxiv.org/abs/2309.13459](http://arxiv.org/abs/2309.13459)

    MaGNet是一种模型无关的图神经网络框架，能够顺序地整合不同顺序的信息，并通过识别有影响力的紧凑图结构提供有意义且可解释的结果。

    

    图神经网络（GNNs）在各种以图为重点的任务中取得了令人满意的性能。尽管取得了成功，但现有的GNN存在两个重要限制：由于黑盒特性，结果缺乏可解释性；无法学习不同顺序的表示。为了解决这些问题，我们提出了一种新的模型无关的图神经网络（MaGNet）框架，能够顺序地整合不同顺序的信息，从高阶邻居中提取知识，并通过识别有影响力的紧凑图结构提供有意义且可解释的结果。特别地，MaGNet由两个组件组成：图拓扑下复杂关系的潜在表示的估计模型和识别有影响力的节点、边和重要节点特征的解释模型。从理论上，我们通过经验Rademacher复杂度建立了MaGNet的泛化误差界，并展示了其强大的能力。

    Graph Neural Networks (GNNs) have achieved promising performance in a variety of graph-focused tasks. Despite their success, existing GNNs suffer from two significant limitations: a lack of interpretability in results due to their black-box nature, and an inability to learn representations of varying orders. To tackle these issues, we propose a novel Model-agnostic Graph Neural Network (MaGNet) framework, which is able to sequentially integrate information of various orders, extract knowledge from high-order neighbors, and provide meaningful and interpretable results by identifying influential compact graph structures. In particular, MaGNet consists of two components: an estimation model for the latent representation of complex relationships under graph topology, and an interpretation model that identifies influential nodes, edges, and important node features. Theoretically, we establish the generalization error bound for MaGNet via empirical Rademacher complexity, and showcase its pow
    
[^45]: 具有逐层非线性的状态空间模型是带有指数衰减记忆的全能逼近器

    State-space Models with Layer-wise Nonlinearity are Universal Approximators with Exponential Decaying Memory. (arXiv:2309.13414v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2309.13414](http://arxiv.org/abs/2309.13414)

    本论文证明了堆叠具有逐层非线性激活的状态空间模型足以逼近任何连续的序列到序列关系，并且发现其加强了模型学习复杂序列模式的能力。然而，状态空间模型并不能根本解决指数衰减记忆的问题。

    

    由于其简单有效的网络结构，状态空间模型在序列建模中变得越来越受欢迎。然而，沿时间方向缺乏非线性激活限制了模型的容量。本文证明了堆叠具有逐层非线性激活的状态空间模型足以逼近任何连续的序列到序列关系。我们的研究结果表明，逐层非线性激活的添加提高了模型学习复杂序列模式的能力。与此同时，可以从理论和实证上看到，状态空间模型并不根本解决指数衰减记忆的问题。理论结果经过了数值验证。

    State-space models have gained popularity in sequence modelling due to their simple and efficient network structures. However, the absence of nonlinear activation along the temporal direction limits the model's capacity. In this paper, we prove that stacking state-space models with layer-wise nonlinear activation is sufficient to approximate any continuous sequence-to-sequence relationship. Our findings demonstrate that the addition of layer-wise nonlinear activation enhances the model's capacity to learn complex sequence patterns. Meanwhile, it can be seen both theoretically and empirically that the state-space models do not fundamentally resolve the exponential decaying memory issue. Theoretical results are justified by numerical verifications.
    
[^46]: 探索训练数据分布和子词标记对机器翻译中的性别偏见的影响

    Exploring the Impact of Training Data Distribution and Subword Tokenization on Gender Bias in Machine Translation. (arXiv:2309.12491v1 [cs.CL])

    [http://arxiv.org/abs/2309.12491](http://arxiv.org/abs/2309.12491)

    这项研究探索了训练数据分布和子词标记对机器翻译中性别偏见的影响。研究发现，模型训练语料库中性别形式的不平衡是导致性别偏见的主要因素，而子词拆分的影响较小。同时，研究还发现，通过分析子词拆分可以很好地估计训练数据中性别形式的不平衡。最后，通过仅微调标记嵌入层可以减少女性和男性之间性别预测准确性的差距。

    

    我们研究了标记化对机器翻译中性别偏见的影响，这是之前的研究中被大多数人忽视的一个方面。具体而言，我们关注的是训练数据中性别职业名称的频率、它们在子词标记器词汇表中的表示以及性别偏见之间的相互作用。我们观察到，女性和非刻板印象的性别职业名称的变形（例如，西班牙语中的"doctora"表示"女医生"）往往被拆分成多个子词标记。我们的结果表明，模型训练语料库中性别形式的不平衡是导致性别偏见的主要因素，其影响大于子词拆分。我们展示了分析子词拆分可以很好地估计训练数据中性别形式的不平衡，并且可以在语料库不公开的情况下使用。我们还证明，仅微调标记嵌入层可以减少女性和男性之间性别预测准确性的差距。

    We study the effect of tokenization on gender bias in machine translation, an aspect that has been largely overlooked in previous works. Specifically, we focus on the interactions between the frequency of gendered profession names in training data, their representation in the subword tokenizer's vocabulary, and gender bias. We observe that female and non-stereotypical gender inflections of profession names (e.g., Spanish "doctora" for "female doctor") tend to be split into multiple subword tokens. Our results indicate that the imbalance of gender forms in the model's training corpus is a major factor contributing to gender bias and has a greater impact than subword splitting. We show that analyzing subword splits provides good estimates of gender-form imbalance in the training data and can be used even when the corpus is not publicly available. We also demonstrate that fine-tuning just the token embedding layer can decrease the gap in gender prediction accuracy between female and male 
    
[^47]: SAVME: 使用元学习进行自动系统的高效安全验证

    SAVME: Efficient Safety Validation for Autonomous Systems Using Meta-Learning. (arXiv:2309.12474v1 [cs.RO])

    [http://arxiv.org/abs/2309.12474](http://arxiv.org/abs/2309.12474)

    本论文提出了一种使用元学习进行高效安全验证的方法，通过集成贝叶斯方法和多臂赌博机框架，加速验证过程。该方法学习在测试中容易引发故障的场景参数分布和能够进行快速准确模拟的保真度设置分布，并通过评估保真度设置分布是否有助于对新场景的场景参数分布进行更快的学习来进一步提高效率。

    

    在部署前，发现自动系统的潜在故障非常重要。虚构法常被用来评估此类系统的安全性，但运行准确模拟的成本很高。我们提出了一种贝叶斯方法，将元学习策略与多臂赌博机框架相结合，加速验证过程。我们的方法涉及学习在测试中容易引发故障的场景参数分布，以及能够进行快速准确模拟的保真度设置分布。在元学习的精神下，我们还评估学习到的保真度设置分布是否有助于对新场景的场景参数分布进行更快的学习。我们使用先进的3D驾驶模拟器展示了我们的方法，并整合了16种保真度设置。

    Discovering potential failures of an autonomous system is important prior to deployment. Falsification-based methods are often used to assess the safety of such systems, but the cost of running many accurate simulation can be high. The validation can be accelerated by identifying critical failure scenarios for the system under test and by reducing the simulation runtime. We propose a Bayesian approach that integrates meta-learning strategies with a multi-armed bandit framework. Our method involves learning distributions over scenario parameters that are prone to triggering failures in the system under test, as well as a distribution over fidelity settings that enable fast and accurate simulations. In the spirit of meta-learning, we also assess whether the learned fidelity settings distribution facilitates faster learning of the scenario parameter distributions for new scenarios. We showcase our methodology using a cutting-edge 3D driving simulator, incorporating 16 fidelity settings fo
    
[^48]: LMSYS-Chat-1M：一个大规模实际语言模型对话数据集

    LMSYS-Chat-1M: A Large-Scale Real-World LLM Conversation Dataset. (arXiv:2309.11998v1 [cs.CL])

    [http://arxiv.org/abs/2309.11998](http://arxiv.org/abs/2309.11998)

    LMSYS-Chat-1M是一个包含一百万个实际对话的大规模数据集，通过其多样性和用例展示了其在理解和推进LLM能力方面的价值。

    

    随着大规模语言模型（LLM）在各种应用中的广泛使用，研究人们如何在实际场景中与其交互变得越来越重要。在本文中，我们介绍了LMSYS-Chat-1M，这是一个包含一百万个与25个最先进的LLM进行的实际对话的大规模数据集。这个数据集是从我们的Vicuna演示和Chatbot Arena网站上的21万个独立IP地址中收集而来的。我们提供了数据集内容的概述，包括其策划过程、基本统计数据和主题分布，强调其多样性、独特性和规模。我们通过四个用例展示了它的多样性：开发与GPT-4表现相似的内容过滤模型、构建一个安全基准、训练与Vicuna表现相似的指令跟随模型、创建具有挑战性的基准问题。我们相信这个数据集将成为我们理解和推进LLM能力的宝贵资源。

    Studying how people interact with large language models (LLMs) in real-world scenarios is increasingly important due to their widespread use in various applications. In this paper, we introduce LMSYS-Chat-1M, a large-scale dataset containing one million real-world conversations with 25 state-of-the-art LLMs. This dataset is collected from 210K unique IP addresses in the wild on our Vicuna demo and Chatbot Arena website. We offer an overview of the dataset's content, including its curation process, basic statistics, and topic distribution, highlighting its diversity, originality, and scale. We demonstrate its versatility through four use cases: developing content moderation models that perform similarly to GPT-4, building a safety benchmark, training instruction-following models that perform similarly to Vicuna, and creating challenging benchmark questions. We believe that this dataset will serve as a valuable resource for understanding and advancing LLM capabilities. The dataset is pub
    
[^49]: 数学问题解决中的思路链设计

    Design of Chain-of-Thought in Math Problem Solving. (arXiv:2309.11054v1 [cs.CL])

    [http://arxiv.org/abs/2309.11054](http://arxiv.org/abs/2309.11054)

    本论文研究了数学问题解决中思路链的设计方法，对比了自然语言思路链和程序思路链的效果，并发现程序思路链通常在数学问题解决中更加有效，特别是自我描述程序具有更大多样性且性能更高。此外，研究还发现Python是程序思路链的较好选择。实验结果为未来思路链设计提供了宝贵指导。

    

    思路链在数学问题解决中扮演着至关重要的角色。我们对设计思路链的方法进行了全面的考察，比较了传统自然语言思路链和各种程序思路链，包括自我描述程序、注释描述程序和非描述程序。此外，我们还研究了编程语言对程序思路链的影响，比较了Python和Wolfram语言。通过对GSM8K、MATHQA和SVAMP进行广泛实验，我们发现程序思路链在数学问题解决中通常具有更好的效果。值得注意的是，具有30B参数的最佳组合明显超过了GPT-3.5-turbo。结果表明，自我描述程序提供了更大的多样性，因此通常可以实现更高的性能。我们还发现，Python是程序思路链的更好选择比Wolfram语言。实验结果为未来考虑因素提供了宝贵的指导。

    Chain-of-Thought (CoT) plays a crucial role in reasoning for math problem solving. We conduct a comprehensive examination of methods for designing CoT, comparing conventional natural language CoT with various program CoTs, including the self-describing program, the comment-describing program, and the non-describing program. Furthermore, we investigate the impact of programming language on program CoTs, comparing Python and Wolfram Language. Through extensive experiments on GSM8K, MATHQA, and SVAMP, we find that program CoTs often have superior effectiveness in math problem solving. Notably, the best performing combination with 30B parameters beats GPT-3.5-turbo by a significant margin. The results show that self-describing program offers greater diversity and thus can generally achieve higher performance. We also find that Python is a better choice of language than Wolfram for program CoTs. The experimental results provide a valuable guideline for future CoT designs that take into acco
    
[^50]: "伴随着伟大的力量而来的是伟大的责任！": 学生和教师对LLMs对本科工程教育影响的观点

    "With Great Power Comes Great Responsibility!": Student and Instructor Perspectives on the influence of LLMs on Undergraduate Engineering Education. (arXiv:2309.10694v2 [cs.HC] UPDATED)

    [http://arxiv.org/abs/2309.10694](http://arxiv.org/abs/2309.10694)

    本文通过调查和访谈，填补了关于LLMs在本科工程教育中的使用和观点的研究空白，为学生和教师对LLMs的采用提供了洞见和建议。

    

    大型语言模型（LLMs）的流行引发了学术界的讨论，学生们探索了基于LLMs的课程查询工具，教师们则探索了基于LLMs的教学和研究。尽管正在努力开发专为学生和教师定制的LLMs工具，但缺乏全面的用户研究来捕捉学生和教师对LLMs的观点。本文通过在印度的本科工程院校进行调查和访谈，来填补这一空白。本文使用了1306份学生调查回答、112份学生访谈和27份教师访谈，探讨了ChatGPT（一种流行的LLM）在学术上的使用情况、感知到的好处、威胁和挑战，并提出了增强学生和教师采用LLMs的建议。这些洞见进一步用于讨论LLMs的实际影响。

    The rise in popularity of Large Language Models (LLMs) has prompted discussions in academic circles, with students exploring LLM-based tools for coursework inquiries and instructors exploring them for teaching and research. Even though a lot of work is underway to create LLM-based tools tailored for students and instructors, there is a lack of comprehensive user studies that capture the perspectives of students and instructors regarding LLMs. This paper addresses this gap by conducting surveys and interviews within undergraduate engineering universities in India. Using 1306 survey responses among students, 112 student interviews, and 27 instructor interviews around the academic usage of ChatGPT (a popular LLM), this paper offers insights into the current usage patterns, perceived benefits, threats, and challenges, as well as recommendations for enhancing the adoption of LLMs among students and instructors. These insights are further utilized to discuss the practical implications of LLM
    
[^51]: 自主车辆间的多智能体深度强化学习在AutoDRIVE生态系统中的合作与竞争

    Multi-Agent Deep Reinforcement Learning for Cooperative and Competitive Autonomous Vehicles using AutoDRIVE Ecosystem. (arXiv:2309.10007v1 [cs.RO])

    [http://arxiv.org/abs/2309.10007](http://arxiv.org/abs/2309.10007)

    本研究提出了一个模块化和并行化的多智能体深度强化学习框架，在AutoDRIVE生态系统中培养合作与竞争行为。我们利用该生态系统开发了准确物理和逼真图形的数字孪生体，并使用它来训练和部署多智能体强化学习策略，实现了在自主车辆中的合作和竞争行为。

    

    本研究提出了一个模块化和可并行化的多智能体深度强化学习框架，用于在自主车辆中培养合作和竞争行为。我们引入了AutoDRIVE生态系统作为一个工具，开发出与真实的Nigel和F1TENTH两种比例自主车辆平台具有独特特性和能力的准确物理和逼真图形的数字孪生体，并利用这个生态系统来训练和部署多智能体强化学习策略。我们首先研究了一个交叉路口穿越问题，使用一组合作车辆（Nigel）在单个或多个智能体学习环境中共享有限状态信息，采用一种公共策略方法。然后我们研究了一个对抗性的头对头自主赛车问题，使用另一组车辆（F1TENTH）在多个智能体学习环境中采用个体策略方法。在任何一组实验中，都采用了分散学习架构。

    This work presents a modular and parallelizable multi-agent deep reinforcement learning framework for imbibing cooperative as well as competitive behaviors within autonomous vehicles. We introduce AutoDRIVE Ecosystem as an enabler to develop physically accurate and graphically realistic digital twins of Nigel and F1TENTH, two scaled autonomous vehicle platforms with unique qualities and capabilities, and leverage this ecosystem to train and deploy multi-agent reinforcement learning policies. We first investigate an intersection traversal problem using a set of cooperative vehicles (Nigel) that share limited state information with each other in single as well as multi-agent learning settings using a common policy approach. We then investigate an adversarial head-to-head autonomous racing problem using a different set of vehicles (F1TENTH) in a multi-agent learning setting using an individual policy approach. In either set of experiments, a decentralized learning architecture was adopted
    
[^52]: 机械化生成器2.0: 强化学习用于评估生成的游戏规则

    Mechanic Maker 2.0: Reinforcement Learning for Evaluating Generated Rules. (arXiv:2309.09476v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2309.09476](http://arxiv.org/abs/2309.09476)

    本论文研究了将强化学习应用于游戏规则生成的人类游戏评估，并通过实验结果表明，强化学习生成的规则与传统基线方法有所不同，可能更适合人类使用。

    

    自动游戏设计（AGD）是研究自动生成游戏规则的技术游戏研究的一个长期课题。 AGD方法通常依赖于对人类玩家游戏的近似，可以是客观函数或AI代理。尽管如此，大部分这些近似器是静态的，也就是说，它们不能反映人类玩家在游戏中的学习和提高能力。本文中，我们研究了将强化学习（RL）应用于生成规则的人类游戏评估中。我们在Unity中重新创建了经典的AGD环境Mechanic Maker作为一个全新的开源生成规则框架。我们的结果表明，RL与A*代理基线产生了不同的规则集，这些规则可能更适合人类使用。

    Automated game design (AGD), the study of automatically generating game rules, has a long history in technical games research. AGD approaches generally rely on approximations of human play, either objective functions or AI agents. Despite this, the majority of these approximators are static, meaning they do not reflect human player's ability to learn and improve in a game. In this paper, we investigate the application of Reinforcement Learning (RL) as an approximator for human play for rule generation. We recreate the classic AGD environment Mechanic Maker in Unity as a new, open-source rule generation framework. Our results demonstrate that RL produces distinct sets of rules from an A* agent baseline, which may be more usable by humans.
    
[^53]: Landscape-Sketch-Step: 一种基于AI/ML的元启发式方法解决代理优化问题

    Landscape-Sketch-Step: An AI/ML-Based Metaheuristic for Surrogate Optimization Problems. (arXiv:2309.07936v1 [cs.LG])

    [http://arxiv.org/abs/2309.07936](http://arxiv.org/abs/2309.07936)

    Landscape-Sketch-Step是一种基于AI/ML的元启发式方法，结合了机器学习、随机优化和强化学习技术，用于解决成本函数评估昂贵、不可访问或禁止的代理优化问题。

    

    本文介绍了一种新的全局优化启发式方法，用于在成本函数的评估非常昂贵、不可访问或甚至禁止的场景下进行优化。该方法称为Landscape-Sketch-Step（LSS），结合了机器学习、随机优化和强化学习技术，依赖于先前采样点的历史信息，以明智地选择应评估成本函数的参数值。与复制交换蒙特卡洛方法相比，该方法所需的成本函数评估次数与模拟退火方法相当，这在高通量计算或高性能计算任务等环境中尤为重要，因为评估要么计算成本高昂，要么需要很长时间才能完成。该方法与标准的代理优化技术也不同，因为它不构建代理模型。

    In this paper, we introduce a new heuristics for global optimization in scenarios where extensive evaluations of the cost function are expensive, inaccessible, or even prohibitive. The method, which we call Landscape-Sketch-and-Step (LSS), combines Machine Learning, Stochastic Optimization, and Reinforcement Learning techniques, relying on historical information from previously sampled points to make judicious choices of parameter values where the cost function should be evaluated at. Unlike optimization by Replica Exchange Monte Carlo methods, the number of evaluations of the cost function required in this approach is comparable to that used by Simulated Annealing, quality that is especially important in contexts like high-throughput computing or high-performance computing tasks, where evaluations are either computationally expensive or take a long time to be performed. The method also differs from standard Surrogate Optimization techniques, for it does not construct a surrogate model
    
[^54]: MMICL：多模态上下文学习增强视觉-语言模型

    MMICL: Empowering Vision-language Model with Multi-Modal In-Context Learning. (arXiv:2309.07915v1 [cs.CL])

    [http://arxiv.org/abs/2309.07915](http://arxiv.org/abs/2309.07915)

    MMICL提出了一种用于视觉-语言模型的架构和训练数据设计，以解决VLM在理解复杂多模态提示方面的困难。

    

    从深度学习的复苏开始，借助大型语言模型（LLM）的视觉-语言模型（VLM）变得非常流行。然而，尽管LLM可以利用丰富的背景知识和任务信息进行上下文学习，大多数VLM在理解复杂的多模态提示（包含多个图像）方面仍然面临困难。这个问题可以追溯到VLM的架构设计或预训练数据。具体来说，当前的VLM主要强调利用带有单个图像的多模态数据，而不是带有交错多个图像和文本的多模态提示。尽管一些新提出的VLM可以处理带有多个图像的用户提示，但预训练数据没有提供比从Web抓取时交错图像和文本更复杂的多模态提示。我们提出了MMICL，从模型和数据的角度来解决这个问题。我们引入了一个精心设计的架构，能够无缝地集成视觉和语言信息，并提供更丰富的多模态训练数据。

    Starting from the resurgence of deep learning, vision-language models (VLMs) benefiting from large language models (LLMs) have never been so popular. However, while LLMs can utilize extensive background knowledge and task information with in-context learning, most VLMs still struggle with understanding complex multi-modal prompts with multiple images. The issue can traced back to the architectural design of VLMs or pre-training data. Specifically, the current VLMs primarily emphasize utilizing multi-modal data with a single image some, rather than multi-modal prompts with interleaved multiple images and text. Even though some newly proposed VLMs could handle user prompts with multiple images, pre-training data does not provide more sophisticated multi-modal prompts than interleaved image and text crawled from the web. We propose MMICL to address the issue by considering both the model and data perspectives. We introduce a well-designed architecture capable of seamlessly integrating vis
    
[^55]: 用于机器人深度强化学习的自我改进型大型语言模型作为自动化奖励函数设计师

    Self-Refined Large Language Model as Automated Reward Function Designer for Deep Reinforcement Learning in Robotics. (arXiv:2309.06687v1 [cs.RO])

    [http://arxiv.org/abs/2309.06687](http://arxiv.org/abs/2309.06687)

    提出一种自我改进机制的大型语言模型（LLM）框架用于自动化奖励函数设计，在深度强化学习中展现了潜在的应用价值。

    

    虽然深度强化学习在众多机器人应用中取得了显著的成功，但设计高性能的奖励函数仍然是一项具有挑战性的任务，通常需要大量的人工输入。最近，广泛采用大型语言模型（LLM）来解决需要深入常识知识的任务，如推理和规划。意识到奖励函数设计与这种知识本质上是相关的，LLM在这个背景下提供了很大的潜力。受此启发，我们在这项工作中提出了一种新颖的LLM框架，具有自我改进机制，用于自动化奖励函数设计。该框架以自然语言输入为基础，由LLM制定一个初始的奖励函数。然后，评估奖励函数的性能，并将结果呈现给LLM以指导其自我改进的过程。通过多种连续机器人任务的实验验证了我们提出的框架的性能。

    Although Deep Reinforcement Learning (DRL) has achieved notable success in numerous robotic applications, designing a high-performing reward function remains a challenging task that often requires substantial manual input. Recently, Large Language Models (LLMs) have been extensively adopted to address tasks demanding in-depth common-sense knowledge, such as reasoning and planning. Recognizing that reward function design is also inherently linked to such knowledge, LLM offers a promising potential in this context. Motivated by this, we propose in this work a novel LLM framework with a self-refinement mechanism for automated reward function design. The framework commences with the LLM formulating an initial reward function based on natural language inputs. Then, the performance of the reward function is assessed, and the results are presented back to the LLM for guiding its self-refinement process. We examine the performance of our proposed framework through a variety of continuous robot
    
[^56]: 使用预训练的大型语言模型进行多模态暗示的零样本推荐

    Zero-Shot Recommendations with Pre-Trained Large Language Models for Multimodal Nudging. (arXiv:2309.01026v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2309.01026](http://arxiv.org/abs/2309.01026)

    该论文提出了一种利用生成型AI领域的新技术进行零样本推荐的方法，通过将多模态输入转化为文本描述，并利用预训练的语言模型计算语义嵌入，实现了对非平稳内容的推荐。在合成的多模态暗示环境中进行实验证明了该方法的有效性。

    

    我们提出了一种利用生成型人工智能领域最新进展的方法，用于零样本推荐多模态非平稳内容。我们建议将不同模态的输入渲染为文本描述，并利用预训练的LLM计算语义嵌入获取它们的数值表示。一旦获得所有内容项的统一表示，可以通过计算适当的相似度度量来进行推荐，而无需进行额外的学习。我们在一个合成的多模态暗示环境中演示了我们的方法，其中输入包括表格、文本和视觉数据。

    We present a method for zero-shot recommendation of multimodal non-stationary content that leverages recent advancements in the field of generative AI. We propose rendering inputs of different modalities as textual descriptions and to utilize pre-trained LLMs to obtain their numerical representations by computing semantic embeddings. Once unified representations of all content items are obtained, the recommendation can be performed by computing an appropriate similarity metric between them without any additional learning. We demonstrate our approach on a synthetic multimodal nudging environment, where the inputs consist of tabular, textual, and visual data.
    
[^57]: 关于Adam的隐式偏差

    On the Implicit Bias of Adam. (arXiv:2309.00079v1 [cs.LG])

    [http://arxiv.org/abs/2309.00079](http://arxiv.org/abs/2309.00079)

    本文证明了RMSProp和Adam存在隐式规范化作用，其取决于超参数和训练阶段，并讨论了这些证明事实对泛化的影响。

    

    在以前的文献中，后向误差分析被用来找到近似梯度下降轨迹的常微分方程（ODEs）。发现有限步长会隐式地规范化解决方案，因为出现在ODE中的项会惩罚损失梯度的二范数。我们证明了RMSProp和Adam中是否存在类似的隐式规范化取决于它们的超参数和训练阶段，但涉及的“范数”不同：对应的ODE项要么惩罚（扰动的）损失梯度的一范数，要么相反地阻止其减小（后一种情况是典型的）。我们还进行了数值实验，并讨论了这些证明事实如何影响泛化。

    In previous literature, backward error analysis was used to find ordinary differential equations (ODEs) approximating the gradient descent trajectory. It was found that finite step sizes implicitly regularize solutions because terms appearing in the ODEs penalize the two-norm of the loss gradients. We prove that the existence of similar implicit regularization in RMSProp and Adam depends on their hyperparameters and the training stage, but with a different "norm" involved: the corresponding ODE terms either penalize the (perturbed) one-norm of the loss gradients or, on the contrary, hinder its decrease (the latter case being typical). We also conduct numerical experiments and discuss how the proven facts can influence generalization.
    
[^58]: BioCoder: 一种带有上下文语用知识的生物信息学代码生成基准

    BioCoder: A Benchmark for Bioinformatics Code Generation with Contextual Pragmatic Knowledge. (arXiv:2308.16458v1 [cs.LG])

    [http://arxiv.org/abs/2308.16458](http://arxiv.org/abs/2308.16458)

    BioCoder是一个用于评估预训练模型在生成生物信息学代码方面的基准，涵盖了函数代码生成中的包依赖关系、类声明和全局变量，并通过模糊测试框架进行评估。

    

    预训练的语言模型（如ChatGPT）显著改进了代码生成。随着这些模型的扩大，需要输出来处理更复杂的任务的需求也越来越多。此外，在生物信息学中，生成功能程序由于领域知识量大、需要复杂的数据操作和复杂的功能依赖关系而面临额外的挑战。在这里，我们介绍了BioCoder，这是一个用于评估现有预训练模型在生成生物信息学代码方面的基准。与函数代码生成有关，BioCoder涵盖了可能的包依赖关系、类声明和全局变量。它包括来自GitHub的1026个Python和Java函数和1243个方法，以及来自Rosalind项目的253个示例。BioCoder还结合了一个用于评估的模糊测试框架，我们已经应用它来评估许多模型，包括InCoder、CodeGen、CodeGen2、SantaCoder、StarCoder、StarCoder+、InstructCodeT。

    Pre-trained language models like ChatGPT have significantly improved code generation. As these models scale up, there is an increasing need for the output to handle more intricate tasks. Moreover, in bioinformatics, generating functional programs poses additional notable challenges due to the amount of domain knowledge, the need for complicated data operations, and intricate functional dependencies between the operations. Here, we present BioCoder, a benchmark developed to evaluate existing pre-trained models in generating bioinformatics code. In relation to function-code generation, BioCoder covers potential package dependencies, class declarations, and global variables. It incorporates 1026 functions and 1243 methods in Python and Java from GitHub and 253 examples from the Rosalind Project. BioCoder incorporates a fuzz-testing framework for evaluation, and we have applied it to evaluate many models including InCoder, CodeGen, CodeGen2, SantaCoder, StarCoder, StarCoder+, InstructCodeT
    
[^59]: 逃离样本陷阱：使用配对距离估计器快速准确地估计认识不确定性

    Escaping the Sample Trap: Fast and Accurate Epistemic Uncertainty Estimation with Pairwise-Distance Estimators. (arXiv:2308.13498v1 [cs.LG])

    [http://arxiv.org/abs/2308.13498](http://arxiv.org/abs/2308.13498)

    本文介绍了使用配对距离估计器对集成模型进行认识不确定性估计的新方法，相比于常用的深度学习方法，该方法能够更快速、更准确地在更大的空间和更高维度上估计认识不确定性。

    

    本文介绍了一种使用配对距离估计器（PaiDEs）对集成模型进行认识不确定性估计的新方法。这些估计器利用模型组件之间的配对距离来建立熵的边界，并将这些边界作为基于信息准则的估计值。与最近基于样本的蒙特卡洛估计器用于认识不确定性估计的深度学习方法不同，PaiDEs能够在更大的空间（最多100倍）上以更快的速度（最多100倍）估计认识不确定性，并在更高维度上具有更准确的性能。为了验证我们的方法，我们进行了一系列用于评估认识不确定性估计的实验：一维正弦数据，摆动物体（Pendulum-v0），跳跃机器人（Hopper-v2），蚂蚁机器人（Ant-v2）和人形机器人（Humanoid-v2）。对于每个实验设置，我们应用了主动学习框架来展示PaiDEs在认识不确定性估计中的优势。

    This work introduces a novel approach for epistemic uncertainty estimation for ensemble models using pairwise-distance estimators (PaiDEs). These estimators utilize the pairwise-distance between model components to establish bounds on entropy and uses said bounds as estimates for information-based criterion. Unlike recent deep learning methods for epistemic uncertainty estimation, which rely on sample-based Monte Carlo estimators, PaiDEs are able to estimate epistemic uncertainty up to 100$\times$ faster, over a larger space (up to 100$\times$) and perform more accurately in higher dimensions. To validate our approach, we conducted a series of experiments commonly used to evaluate epistemic uncertainty estimation: 1D sinusoidal data, Pendulum-v0, Hopper-v2, Ant-v2 and Humanoid-v2. For each experimental setting, an Active Learning framework was applied to demonstrate the advantages of PaiDEs for epistemic uncertainty estimation.
    
[^60]: 基于提示的长度受控生成与强化学习

    Prompt-Based Length Controlled Generation with Reinforcement Learning. (arXiv:2308.12030v1 [cs.CL])

    [http://arxiv.org/abs/2308.12030](http://arxiv.org/abs/2308.12030)

    提出了一种基于提示的长度控制方法，利用强化学习和奖励模型来实现大型语言模型（LLM）的长度受控生成。该方法可以有效减少推理成本并满足不同需求。

    

    最近，大型语言模型（LLM）如ChatGPT和GPT-4因其惊人的改进和性能而受到广泛关注。长度受控生成成为LLM中的一个重要话题，它还使用户能够充分利用LLM的能力在更多实际场景中生成所需长度的合适答案或文章。此外，LLM中的自回归生成非常耗时，而控制生成长度的能力可以通过限制长度任意降低推理成本，从而满足不同需求。因此，我们旨在提出一种基于提示的长度控制方法来实现长度受控生成，这种方法也可以广泛应用于类似GPT的LLM中。具体而言，我们采用强化学习，使用可训练或基于规则的奖励模型提供奖励信号，进一步通过对预定义目标长度进行奖励来影响LLM的生成。实验证明...

    Recently, large language models (LLMs) like ChatGPT and GPT-4 have attracted great attention given their surprising improvement and performance. Length controlled generation of LLMs emerges as an important topic, which also enables users to fully leverage the capability of LLMs in more real-world scenarios like generating a proper answer or essay of a desired length. In addition, the autoregressive generation in LLMs is extremely time-consuming, while the ability of controlling this generated length can arbitrarily reduce the inference cost by limiting the length, and thus satisfy different needs. Therefore, we aim to propose a prompt-based length control method to achieve this length controlled generation, which can also be widely applied in GPT-style LLMs. In particular, we adopt reinforcement learning with the reward signal given by either trainable or rule-based reward model, which further affects the generation of LLMs via rewarding a pre-defined target length. Experiments show th
    
[^61]: 动态开放词汇增强的智能安全着陆（DOVESEI）

    Dynamic Open Vocabulary Enhanced Safe-landing with Intelligence (DOVESEI). (arXiv:2308.11471v1 [cs.RO])

    [http://arxiv.org/abs/2308.11471](http://arxiv.org/abs/2308.11471)

    本文提出了一种动态开放词汇增强的智能安全着陆系统，通过利用开放词汇图像分割的能力实现无人机的视觉伺服，适应不同场景且无需大量数据积累进行模型改进，可以处理100米高度的操作。

    

    本研究针对城市空中机器人的基础步骤之一，即安全着陆。我们关注安全着陆感知堆栈中最关键的方面之一，即分割。我们提出了一种简化的反应式无人机系统，利用开放词汇图像分割的能力实现视觉伺服。这种方法可以适应各种场景，并通过其开放词汇方法，最小化调整需求，绕过对内部模型进行大量数据积累以进行改进的必要性。考虑到当地当局的限制，我们的主要关注点是从100米高度起飞的操作。这个选择是有意的，因为许多之前的工作处理的高度仅限于30米，与小型立体相机的能力相吻合。因此，我们采用传统的三维路径规划方法来导航剩下的20米。利用单目相机和图像

    This work targets what we consider to be the foundational step for urban airborne robots, a safe landing. Our attention is directed toward what we deem the most crucial aspect of the safe landing perception stack: segmentation. We present a streamlined reactive UAV system that employs visual servoing by harnessing the capabilities of open vocabulary image segmentation. This approach can adapt to various scenarios with minimal adjustments, bypassing the necessity for extensive data accumulation for refining internal models, thanks to its open vocabulary methodology. Given the limitations imposed by local authorities, our primary focus centers on operations originating from altitudes of 100 meters. This choice is deliberate, as numerous preceding works have dealt with altitudes up to 30 meters, aligning with the capabilities of small stereo cameras. Consequently, we leave the remaining 20m to be navigated using conventional 3D path planning methods. Utilizing monocular cameras and image 
    
[^62]: 评估大型语言模型对提示注入的指令跟随鲁棒性的研究

    Evaluating the Instruction-Following Robustness of Large Language Models to Prompt Injection. (arXiv:2308.10819v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2308.10819](http://arxiv.org/abs/2308.10819)

    该论文提出了一个用于评估大型语言模型对注入的对抗性指令的鲁棒性的基准，旨在量化模型受到注入指令影响的程度，并评估其区分原始用户指令和注入指令的能力。

    

    大型语言模型（LLM）在遵循指令方面表现出卓越的能力，使其在面向客户的应用中具有重要价值。然而，它们的出色能力也引发了对由第三方攻击者注入模型输入的对抗性指令的风险放大的担忧，这些指令可能操纵LLM的原始指令并导致意外的行为和内容。因此，了解LLM准确辨别要遵循的指令的能力对于确保它们在现实场景中的安全部署至关重要。在本文中，我们提出了一个开创性的基准，用于自动评估注入的对抗性指令对LLM指令跟随鲁棒性的影响。该基准的目标是量化LLM受注入的对抗性指令影响的程度，并评估其区分这些注入的对抗性指令和原始用户指令的能力。

    Large Language Models (LLMs) have shown remarkable proficiency in following instructions, making them valuable in customer-facing applications. However, their impressive capabilities also raise concerns about the amplification of risks posed by adversarial instructions, which can be injected into the model input by third-party attackers to manipulate LLMs' original instructions and prompt unintended actions and content. Therefore, it is crucial to understand LLMs' ability to accurately discern which instructions to follow to ensure their safe deployment in real-world scenarios. In this paper, we propose a pioneering benchmark for automatically evaluating the robustness of instruction-following LLMs against adversarial instructions injected in the prompt. The objective of this benchmark is to quantify the extent to which LLMs are influenced by injected adversarial instructions and assess their ability to differentiate between these injected adversarial instructions and original user ins
    
[^63]: 面向矿山勘测任务的自主无人机的概率因果发现、推理和解释

    Towards Probabilistic Causal Discovery, Inference & Explanations for Autonomous Drones in Mine Surveying Tasks. (arXiv:2308.10047v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2308.10047](http://arxiv.org/abs/2308.10047)

    本文针对无人机在盐矿勘测任务中面临的因果挑战，提出了一个概率因果框架，包括因果规划、在线适应和事后解释，以解决混淆变量、非平稳性和建模困难等问题。

    

    因果建模为自主代理提供了解析其与世界互动的数据生成过程的潜力。这些模型捕捉了正式的知识和概率性的噪声和不确定性表示，这些特征在实际环境中的自主机器人遇到是常见的。因此，因果关系可以帮助自主代理进行决策和解释结果，但是以这种方式应用因果关系会引入新的挑战。本文在盐矿中无人机系统的环境中，识别了与因果关系相关的挑战。这样的环境对自主代理来说是具有挑战性的，因为存在混淆变量、非平稳性，并且难以提前构建完整的因果模型。为了解决这些问题，我们提出了一个概率因果框架，包括：基于因果的POMDP规划、在线SCM适应和事后因果推断解释。此外，我们还概述了计划中的研究方向。

    Causal modelling offers great potential to provide autonomous agents the ability to understand the data-generation process that governs their interactions with the world. Such models capture formal knowledge as well as probabilistic representations of noise and uncertainty typically encountered by autonomous robots in real-world environments. Thus, causality can aid autonomous agents in making decisions and explaining outcomes, but deploying causality in such a manner introduces new challenges. Here we identify challenges relating to causality in the context of a drone system operating in a salt mine. Such environments are challenging for autonomous agents because of the presence of confounders, non-stationarity, and a difficulty in building complete causal models ahead of time. To address these issues, we propose a probabilistic causal framework consisting of: causally-informed POMDP planning, online SCM adaptation, and post-hoc counterfactual explanations. Further, we outline planned
    
[^64]: Ada-QPacknet -- 自适应剪枝与位宽缩减作为一种高效的继续学习方法，不会遗忘的算法

    Ada-QPacknet -- adaptive pruning with bit width reduction as an efficient continual learning method without forgetting. (arXiv:2308.07939v1 [cs.LG])

    [http://arxiv.org/abs/2308.07939](http://arxiv.org/abs/2308.07939)

    Ada-QPacknet是一种自适应剪枝与位宽缩减的高效继续学习方法，通过剪枝和量化技术生成任务子网络，在动态和复杂环境中实现了与浮点数子网络相似的准确性。

    

    继续学习（CL）是一个过程，其中人类和深度学习模型之间的效率仍存在巨大差距。最近设计了许多CL算法，大部分都存在在动态和复杂环境中学习的问题。本文描述了一种基于新架构的方法Ada-QPacknet。它通过剪枝提取每个任务的子网络。基于架构的CL方法的关键是容量。在提出的方法中，通过高效的线性和非线性量化方法减小了模型的规模。该方法减小了权重格式的位宽。实验结果显示，混合8位和4位量化在著名的CL场景上实现了与浮点数子网络相似的准确性。据我们所知，这是第一个将剪枝和量化这两种压缩技术应用于生成任务子网络的CL策略。该算法在著名的情节组合上进行了测试。

    Continual Learning (CL) is a process in which there is still huge gap between human and deep learning model efficiency. Recently, many CL algorithms were designed. Most of them have many problems with learning in dynamic and complex environments. In this work new architecture based approach Ada-QPacknet is described. It incorporates the pruning for extracting the sub-network for each task. The crucial aspect in architecture based CL methods is theirs capacity. In presented method the size of the model is reduced by efficient linear and nonlinear quantisation approach. The method reduces the bit-width of the weights format. The presented results shows that hybrid 8 and 4-bit quantisation achieves similar accuracy as floating-point sub-network on a well-know CL scenarios. To our knowledge it is the first CL strategy which incorporates both compression techniques pruning and quantisation for generating task sub-networks. The presented algorithm was tested on well-known episode combination
    
[^65]: 使用大型语言模型将地面操纵器的原始任务转换为可执行动作

    Ground Manipulator Primitive Tasks to Executable Actions using Large Language Models. (arXiv:2308.06810v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2308.06810](http://arxiv.org/abs/2308.06810)

    本文提出了一种利用大型语言模型将操纵器的原始任务转换为机器人的低层动作的方法，通过设计类似程序函数的提示，实现了对位置/力的设定点的生成，从而实现了混合控制

    

    分层结构在机器人系统中被广泛使用，但是大多数机器人系统在规划和执行功能之间缺乏直接的方式。为了解决这个挑战，我们提出了一种新颖的方法，利用大型语言模型（LLMs）将操纵器的原始任务转换为机器人的低层动作。我们设计了一个类似于程序函数的提示，基于任务框架形式主义。通过这种方式，我们使得LLMs能够生成位置/力的设定点进行混合控制。我们还对几种最先进的LLMs进行了评估。

    Layered architectures have been widely used in robot systems. The majority of them implement planning and execution functions in separate layers. However, there still lacks a straightforward way to transit high-level tasks in the planning layer to the low-level motor commands in the execution layer. In order to tackle this challenge, we propose a novel approach to ground the manipulator primitive tasks to robot low-level actions using large language models (LLMs). We designed a program-function-like prompt based on the task frame formalism. In this way, we enable LLMs to generate position/force set-points for hybrid control. Evaluations over several state-of-the-art LLMs are provided.
    
[^66]: Adv-Inpainting:通过注意力引导的特征融合生成自然且可迁移的对抗性贴纸

    Adv-Inpainting: Generating Natural and Transferable Adversarial Patch via Attention-guided Feature Fusion. (arXiv:2308.05320v1 [cs.CV])

    [http://arxiv.org/abs/2308.05320](http://arxiv.org/abs/2308.05320)

    本文提出了一种称为Adv-Inpainting的创新攻击框架，通过注意力引导的特征融合生成自然且可迁移的对抗性贴纸，相比于传统的对抗性贴纸方法，该方法在生成图案和边界方面更加自然，并具有更强的迁移性能。

    

    最初的对抗性攻击利用加性噪声攻击人脸识别模型。然而，由于在实际环境中操作整个脸部是不切实际的，大多数现实世界中的人脸识别攻击都基于对抗性贴纸，将扰动限制在一个较小的区域内。先前的对抗性贴纸攻击常常导致不自然的图案和明显的边界，容易被察觉。我们认为生成带有合理内容的对抗性贴纸会比使用加性噪声或直接从潜在空间进行采样更具有更强的迁移性。为了生成自然且高度可迁移的对抗性贴纸，我们提出了一种创新的两阶段粗到精的攻击框架，称为Adv-Inpainting。在第一阶段中，我们提出了一种注意力引导的StyleGAN（Att-StyleGAN），根据注意力图自适应地结合纹理和身份特征，生成高度可迁移和自然的对抗性贴纸。

    The rudimentary adversarial attacks utilize additive noise to attack facial recognition (FR) models. However, because manipulating the total face is impractical in the physical setting, most real-world FR attacks are based on adversarial patches, which limit perturbations to a small area. Previous adversarial patch attacks often resulted in unnatural patterns and clear boundaries that were easily noticeable. In this paper, we argue that generating adversarial patches with plausible content can result in stronger transferability than using additive noise or directly sampling from the latent space. To generate natural-looking and highly transferable adversarial patches, we propose an innovative two-stage coarse-to-fine attack framework called Adv-Inpainting. In the first stage, we propose an attention-guided StyleGAN (Att-StyleGAN) that adaptively combines texture and identity features based on the attention map to generate high-transferable and natural adversarial patches. In the second
    
[^67]: FLIPS: 使用智能参与者选择的联邦学习

    FLIPS: Federated Learning using Intelligent Participant Selection. (arXiv:2308.03901v1 [cs.LG])

    [http://arxiv.org/abs/2308.03901](http://arxiv.org/abs/2308.03901)

    本文介绍了FLIPS，这是一个用于管理联邦学习中数据和参与者异质性的中间件系统。FLIPS通过标签分布聚类和智能参与者选择，并使用可信执行环境来确保隐私保护。实证评估表明，FLIPS相比随机方法有更好的性能。

    

    本文介绍了FLIPS的设计和实现，这是一个用于管理联邦学习中数据和参与者异质性的中间件系统。特别地，我们研究了标签分布聚类在联邦学习中参与者选择中的好处。FLIPS根据数据的标签分布预先对参与FL训练作业的各方进行聚类，并在FL训练期间确保每个聚类在被选中的参与者中公平地表示。FLIPS可以支持最常见的FL算法，包括FedAvg，FedProx，FedDyn，FedOpt和FedYogi。为了管理平台的异构性和动态资源可用性，FLIPS还结合了一种处理分布式智能社区应用中容量变化的拖累管理机制。标签分布、聚类和参与者选择的隐私通过可信执行环境(TEE)来确保。我们全面的实证评估将FLIPS与随机方法进行了比较。

    This paper presents the design and implementation of FLIPS, a middleware system to manage data and participant heterogeneity in federated learning (FL) training workloads. In particular, we examine the benefits of label distribution clustering on participant selection in federated learning. FLIPS clusters parties involved in an FL training job based on the label distribution of their data apriori, and during FL training, ensures that each cluster is equitably represented in the participants selected. FLIPS can support the most common FL algorithms, including FedAvg, FedProx, FedDyn, FedOpt and FedYogi. To manage platform heterogeneity and dynamic resource availability, FLIPS incorporates a straggler management mechanism to handle changing capacities in distributed, smart community applications. Privacy of label distributions, clustering and participant selection is ensured through a trusted execution environment (TEE). Our comprehensive empirical evaluation compares FLIPS with random p
    
[^68]: Relation-Oriented: 迈向与知识对准的因果人工智能

    Relation-Oriented: Toward Knowledge-Aligned Causal AI. (arXiv:2307.16387v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2307.16387](http://arxiv.org/abs/2307.16387)

    本研究从创新的关系导向视角出发，探讨了当前的建模范式中的观察模型与实际理解的不对齐问题，并提出了关系定义的表示学习方法作为实现关系导向建模的实践方法。

    

    在机器学习中，我们自然地应用一个观察导向的原则，其中观察变量先存在并为构建关系奠定基础。虽然对于传统模型来说足够了，但是人工智能与大数据的整合暴露了观察模型与我们的实际理解之间的不对齐。相反，人类塑造了由关系定义的认知实体，使我们能够跨越时间和超维度空间制定知识，而不是被限制在观察构建中。从一种创新的关系导向的视角出发，本研究通过来自计算机视觉和健康信息学的直观例子，分析了在我们当前的建模范式中这种不对齐的根源。我们还介绍了关系定义的表示学习方法作为关系导向建模的一种实际实施，支持广泛的实验验证。

    In machine learning, we naturally apply an Observation-Oriented principle, in which observational variables preexist and set the stage for constructing relationships. While sufficient for traditional models, the integration of AI with big data exposes the misalignment between the observational models and our actual comprehension. Contrarily, humans shape cognitive entities defined by relationships, enabling us to formulate knowledge across temporal and hyper-dimensional spaces, rather than being confined to observational constructs. From an innovative Relation-Oriented perspective, this study examines the roots of this misalignment within our current modeling paradigm, illuminated by intuitive examples from computer vision and health informatics. We also introduce the relation-defined representation learning methodology as a practical implementation of Relation-Oriented modeling, supported by extensive experimental validation.
    
[^69]: 一种具有规划、长期上下文理解和程序合成能力的现实世界WebAgent

    A Real-World WebAgent with Planning, Long Context Understanding, and Program Synthesis. (arXiv:2307.12856v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2307.12856](http://arxiv.org/abs/2307.12856)

    这篇论文介绍了一种名为WebAgent的LLM驱动代理，通过自我经验学习，在真实网站上完成任务。该方法通过规划、总结和生成代码来提高在真实网站上的成功率。

    

    最近，预训练的大型语言模型（LLMs）在自主Web自动化方面取得了更好的泛化性能和样本效率。然而，在真实世界的网站上，性能仍然受到三个方面的限制：开放领域性、有限的上下文长度和对HTML的归纳偏差的缺乏。我们介绍了一种名为WebAgent的LLM驱动代理，它通过自我经验学习，在遵循自然语言指令的前提下，在真实网站上完成任务。WebAgent通过将指令分解为规范的子指令，将长HTML文档总结为与任务相关的片段，并通过从中生成的Python程序对网站进行操作来提前进行规划。我们使用Flan-U-PaLM设计了WebAgent，用于生成有根代码，并使用HTML-T5进行预训练LLMs，利用局部和全局注意机制以及混合长跨度去噪目标来进行规划和总结。我们通过实验证明，我们的模块化方法提高了在真实网站上的成功率。

    Pre-trained large language models (LLMs) have recently achieved better generalization and sample efficiency in autonomous web automation. However, the performance on real-world websites has still suffered from (1) open domainness, (2) limited context length, and (3) lack of inductive bias on HTML. We introduce WebAgent, an LLM-driven agent that learns from self-experience to complete tasks on real websites following natural language instructions. WebAgent plans ahead by decomposing instructions into canonical sub-instructions, summarizes long HTML documents into task-relevant snippets, and acts on websites via Python programs generated from those. We design WebAgent with Flan-U-PaLM, for grounded code generation, and HTML-T5, new pre-trained LLMs for long HTML documents using local and global attention mechanisms and a mixture of long-span denoising objectives, for planning and summarization. We empirically demonstrate that our modular recipe improves the success on real websites by ov
    
[^70]: 一个用于道路段推荐维护的决策框架

    A decision making framework for recommended maintenance of road segments. (arXiv:2307.10085v1 [cs.AI])

    [http://arxiv.org/abs/2307.10085](http://arxiv.org/abs/2307.10085)

    这项研究提出了一个决策框架，通过整合多种人工智能决策技术和历史数据，为道路管理部门提供科学决策工具和证据，以解决道路维护的问题。

    

    随着全球道路交通的快速发展，各国已完成了道路网络的建设。然而，随之而来的挑战在于现有道路的维护。众所周知，各国在道路维护项目上的预算有限，道路管理部门在进行科学决策方面面临困难。因此，将各种人工智能决策技术与历史维护数据相结合，以适应道路维护科学决策的背景，成为一个迫切的问题。这种整合旨在为道路管理部门提供更科学的工具和证据，以进行决策。本文提出的框架主要解决以下四个问题：1）预测各路线的路面性能，2）确定维护路线的优先级，3）基于评估标准制定维护决策。

    With the rapid development of global road transportation, countries worldwide have completed the construction of road networks. However, the ensuing challenge lies in the maintenance of existing roads. It is well-known that countries allocate limited budgets to road maintenance projects, and road management departments face difficulties in making scientifically informed maintenance decisions. Therefore, integrating various artificial intelligence decision-making techniques to thoroughly explore historical maintenance data and adapt them to the context of road maintenance scientific decision-making has become an urgent issue. This integration aims to provide road management departments with more scientific tools and evidence for decision-making. The framework proposed in this paper primarily addresses the following four issues: 1) predicting the pavement performance of various routes, 2) determining the prioritization of maintenance routes, 3) making maintenance decisions based on the e
    
[^71]: 通过单向流进行对抗性似然估计

    Adversarial Likelihood Estimation with One-way Flows. (arXiv:2307.09882v1 [cs.LG])

    [http://arxiv.org/abs/2307.09882](http://arxiv.org/abs/2307.09882)

    本文提出了一种通过单向流进行对抗性似然估计的方法，并使用重要性采样解决了Wasserstein GAN中分区函数有偏估计的问题。同时，通过最大化生成器的熵，提高了模式覆盖效果。这种方法通过计算生成样本的密度来实现对分区函数的无偏估计和生成器熵的计算。

    

    生成对抗网络（GAN）能够产生高质量的样本，但无法提供样本周围的概率密度估计。然而，已经注意到在能量模型的设置中，最大化对数似然可以导致判别器提供非归一化的密度（通常称为能量）的对抗性框架。我们进一步发展了这一观点，结合重要性采样，并展示了以下内容：1）Wasserstein GAN对分区函数进行了有偏估计，我们提出使用无偏估计方法；2）在最优化似然时，必须最大化生成器的熵。这被假设会提供更好的模式覆盖。与以前的工作不同，我们明确计算了生成样本的密度。这是设计无偏估计分区函数以及计算生成器熵的关键因素。生成密度是通过一种新型的流网络来获得的，称为单向流网络。

    Generative Adversarial Networks (GANs) can produce high-quality samples, but do not provide an estimate of the probability density around the samples. However, it has been noted that maximizing the log-likelihood within an energy-based setting can lead to an adversarial framework where the discriminator provides unnormalized density (often called energy). We further develop this perspective, incorporate importance sampling, and show that 1) Wasserstein GAN performs a biased estimate of the partition function, and we propose instead to use an unbiased estimator; 2) when optimizing for likelihood, one must maximize generator entropy. This is hypothesized to provide a better mode coverage. Different from previous works, we explicitly compute the density of the generated samples. This is the key enabler to designing an unbiased estimator of the partition function and computation of the generator entropy term. The generator density is obtained via a new type of flow network, called one-way 
    
[^72]: 通过损失函数曲率视角揭示记忆化过程

    Memorization Through the Lens of Curvature of Loss Function Around Samples. (arXiv:2307.05831v1 [cs.LG])

    [http://arxiv.org/abs/2307.05831](http://arxiv.org/abs/2307.05831)

    本研究通过对损失函数曲率进行分析，研究了神经网络在不同样本上的泛化与记忆化特性。我们发现高曲率的样本通常是具有标签错误或冲突的长尾样本，并在CIFAR100数据集上发现了一种新的失败模型。通过对部分样本进行随机标签错误，我们展示了曲率排序可以有效识别出这些样本。

    

    神经网络参数过多，很容易过拟合训练数据。极端情况下，它们可以完全记忆训练集，即使标签是随机的。我们提议使用训练样本周围的损失函数曲率作为记忆化程度的度量，对所有训练轮次进行平均。我们利用这个度量来研究常见图像数据集中不同样本的泛化与记忆化特性。我们可视化具有最高损失曲率的样本，发现它们通常是长尾样本、标签错误或冲突样本。这种分析帮助我们在CIFAR100数据集上发现了一种新的失败模型，即具有不同标签的重复图像。我们还通过随机错误化少量样本的标签来人为地给数据集引入标签错误，并展示了按曲率排序可以高效地识别出标签错误样本的高AUROC值。

    Neural networks are overparametrized and easily overfit the datasets they train on. In the extreme case, it is shown that they can memorize a training set with fully randomized labels. We propose using the curvature of loss function around the training sample as a measure of its memorization, averaged over all training epochs. We use this to study the generalization versus memorization properties of different samples in popular image datasets. We visualize samples with the highest curvature of loss around them, and show that these visually correspond to long-tailed, mislabeled or conflicting samples. This analysis helps us find a, to the best of our knowledge, novel failure model on the CIFAR100 dataset, that of duplicated images with different labels. We also synthetically mislabel a proportion of the dataset by randomly corrupting the labels of a few samples, and show that sorting by curvature yields high AUROC values for identifying the mislabeled samples.
    
[^73]: 在高峰小时序列预测中缩小性能差距: Seq2Peak框架

    Bridge the Performance Gap in Peak-hour Series Forecasting: The Seq2Peak Framework. (arXiv:2307.01597v1 [cs.LG])

    [http://arxiv.org/abs/2307.01597](http://arxiv.org/abs/2307.01597)

    本文提出了Seq2Peak框架，针对高峰小时序列预测任务，该框架通过解决高度非平稳性和性能评估问题，成功缩小了在常规时间序列预测模型中观察到的性能差距。

    

    高峰小时序列预测（PHSF）是各个领域中一个重要但未被充分探索的任务。虽然最先进的深度学习模型在常规时间序列预测（TSF）中表现出色，但在PHSF中却难以达到可比较的结果。这可能归因于高峰小时序列中高度非平稳性的挑战，使得直接预测比标准的TSF更加困难。此外，手动从常规预测结果中提取最大值会导致性能不佳，因为模型会最小化平均差。为了解决这些问题，本文提出了Seq2Peak，一个专为PHSF任务而设计的新颖框架，以弥合在TSF模型中观察到的性能差距。Seq2Peak具有两个关键组件：CyclicNorm流程来减轻非平稳性问题，以及一个简单而有效的可训练参数自由峰值小时解码器，采用混合损失函数来利用原始序列和高峰小时序列作为监督信号。

    Peak-Hour Series Forecasting (PHSF) is a crucial yet underexplored task in various domains. While state-of-the-art deep learning models excel in regular Time Series Forecasting (TSF), they struggle to achieve comparable results in PHSF. This can be attributed to the challenges posed by the high degree of non-stationarity in peak-hour series, which makes direct forecasting more difficult than standard TSF. Additionally, manually extracting the maximum value from regular forecasting results leads to suboptimal performance due to models minimizing the mean deficit. To address these issues, this paper presents Seq2Peak, a novel framework designed specifically for PHSF tasks, bridging the performance gap observed in TSF models. Seq2Peak offers two key components: the CyclicNorm pipeline to mitigate the non-stationarity issue, and a simple yet effective trainable-parameter-free peak-hour decoder with a hybrid loss function that utilizes both the original series and peak-hour series as superv
    
[^74]: RefSAM：高效适应任何模型的指代视频对象分割

    RefSAM: Efficiently Adapting Segmenting Anything Model for Referring Video Object Segmentation. (arXiv:2307.00997v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2307.00997](http://arxiv.org/abs/2307.00997)

    本文介绍了RefSAM模型，该模型通过在线方式从不同时间戳的多视图信息中加入SAM的潜力，探索其在指代视频对象分割（RVOS）中的应用。通过使用跨模态MLP和分层稠密注意模块，我们改进了SAM模型，实现了对不同形态的精确理解，并取得了令人印象深刻的性能表现。

    

    Segment Anything Model (SAM)因其在图像分割中出色的性能而引起了广泛关注。然而，在指代视频对象分割（RVOS）方面，由于需要精确的用户交互提示以及对语言和视觉等不同形态的有限理解能力，SAM缺乏熟练度。本文提出了RefSAM模型，通过在线方式从不同时间戳的多视图信息中加入SAM的潜力，探索其在RVOS中的应用。我们的方法对原始SAM模型进行了适应，通过使用轻量级的跨模态MLP将指代表达的文本嵌入投影为稀疏和密集嵌入，作为用户交互提示，以增强跨模态学习。此外，我们还引入了分层稠密注意模块，以将分层视觉语义信息与稀疏嵌入融合，以获得细粒度的密集嵌入。

    The Segment Anything Model (SAM) has gained significant attention for its impressive performance in image segmentation. However, it lacks proficiency in referring video object segmentation (RVOS) due to the need for precise user-interactive prompts and a limited understanding of different modalities, such as language and vision. This paper presents the RefSAM model, which explores the potential of SAM for RVOS by incorporating multi-view information from diverse modalities and successive frames at different timestamps in an online manner. Our proposed approach adapts the original SAM model to enhance cross-modality learning by employing a lightweight Cross-Modal MLP that projects the text embedding of the referring expression into sparse and dense embeddings, serving as user-interactive prompts. Additionally, we have introduced the hierarchical dense attention module to fuse hierarchical visual semantic information with sparse embeddings in order to obtain fine-grained dense embeddings
    
[^75]: DoReMi: 通过检测和修复计划执行不一致来实现语言模型的基础

    DoReMi: Grounding Language Model by Detecting and Recovering from Plan-Execution Misalignment. (arXiv:2307.00329v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2307.00329](http://arxiv.org/abs/2307.00329)

    DoReMi是一种新颖的语言模型基础架构，通过检测和修复计划与执行之间的不一致性来实现语言模型的基础。该架构利用视觉问答模型检查约束条件以发现不一致，并调用语言模型进行重新规划以实现恢复。

    

    大型语言模型包含大量的语义知识，并具备出色的理解和推理能力。先前的研究已经探索了如何将语言模型与机器人任务相结合，以确保语言模型生成的序列在逻辑上正确且可执行。然而，由于环境扰动或控制器设计的不完善，底层执行可能会偏离高级计划。在本文中，我们提出了一种名为DoReMi的新型语言模型基础架构，该架构能够及时检测和修复计划与执行之间的不一致性。具体而言，我们利用LLM进行规划，并生成计划步骤的约束条件。这些约束条件可以指示计划与执行之间的不一致性，并且我们使用视觉问答（VQA）模型在低层技能执行过程中检查约束条件。如果发生特定的不一致，我们的方法将调用语言模型重新规划以从中恢复。

    Large language models encode a vast amount of semantic knowledge and possess remarkable understanding and reasoning capabilities. Previous research has explored how to ground language models in robotic tasks to ensure that the sequences generated by the language model are both logically correct and practically executable. However, low-level execution may deviate from the high-level plan due to environmental perturbations or imperfect controller design. In this paper, we propose DoReMi, a novel language model grounding framework that enables immediate Detection and Recovery from Misalignments between plan and execution. Specifically, LLMs are leveraged for both planning and generating constraints for planned steps. These constraints can indicate plan-execution misalignments and we use a vision question answering (VQA) model to check constraints during low-level skill execution. If certain misalignment occurs, our method will call the language model to re-plan in order to recover from mi
    
[^76]: 无监督的多色彩神经表示法用于CT金属伪影减少

    Unsupervised Polychromatic Neural Representation for CT Metal Artifact Reduction. (arXiv:2306.15203v1 [eess.IV])

    [http://arxiv.org/abs/2306.15203](http://arxiv.org/abs/2306.15203)

    本文提出了一种新颖的多色彩神经表示法（Polyner），用于解决CT成像中存在金属伪影的挑战性问题。Polyner通过建模非线性反问题，准确模拟CT采集过程，并利用无监督训练的神经网络架构恢复原始物体信息。实验证明Polyner在金属伪影减少方面的有效性。

    

    新兴的基于层析术的神经重建技术（如NeRF，NeAT和NeRP）在医学成像方面已经展示出独特的能力。在本文中，我们提出了一种新颖的多色彩神经表示法（Polyner）来解决CT成像中存在人体金属植入物时的挑战性问题。金属伪影是由于X射线能谱不同能量级金属的衰减系数剧烈变化而产生的，导致CT测量中的非线性金属效应。因此，从受金属影响的测量中重建CT图像是一个复杂的非线性反问题，先前的金属伪影减少（MAR）方法中采用的经验模型导致信号损失和强烈的混叠重建。Polyner从非线性反问题的角度对MAR问题进行建模。具体而言，我们首先推导出多色彩正演模型，以准确模拟非线性CT采集过程。然后，我们将据此设计一个无监督训练的神经网络架构，用于从金属伪影的CT投影图中恢复出原始的物体信息。我们通过对实际CT数据集进行了广泛实验证明了Polyner的有效性。

    Emerging neural reconstruction techniques based on tomography (e.g., NeRF, NeAT, and NeRP) have started showing unique capabilities in medical imaging. In this work, we present a novel Polychromatic neural representation (Polyner) to tackle the challenging problem of CT imaging when metallic implants exist within the human body. The artifacts arise from the drastic variation of metal's attenuation coefficients at various energy levels of the X-ray spectrum, leading to a nonlinear metal effect in CT measurements. Reconstructing CT images from metal-affected measurements hence poses a complicated nonlinear inverse problem where empirical models adopted in previous metal artifact reduction (MAR) approaches lead to signal loss and strongly aliased reconstructions. Polyner instead models the MAR problem from a nonlinear inverse problem perspective. Specifically, we first derive a polychromatic forward model to accurately simulate the nonlinear CT acquisition process. Then, we incorporate ou
    
[^77]: 线性时间逻辑规则的时间点解释框架

    Pointwise-in-Time Explanation for Linear Temporal Logic Rules. (arXiv:2306.13956v1 [cs.AI])

    [http://arxiv.org/abs/2306.13956](http://arxiv.org/abs/2306.13956)

    本文提出了一个可以评估给定路径规划中特定时间点上的单个线性时间逻辑(LTL)约束的相关性和状态的框架，可以用于在离散时间、离散空间中执行有限计划的代理任务中，为用户提供时间点解释和规则参数状态的洞察力。

    

    本文介绍了一个框架来评估给定路径规划中特定时间点上的单个线性时间逻辑(LTL)约束的相关性，这个任务被我们称为“时间点解释”。我们开发了一个包含状态评估算法的框架，适用于在Kripke结构可表达的离散时间、离散空间中执行有限计划的代理。在给定的结构上和已知约束代理的一组LTL规则的计划中，该算法针对两种类型的用户查询响应地生成解释。对于所选的查询时间，解释识别哪些规则是活动的，哪些规则刚刚被满足，哪些规则是不活动的，其中框架状态标准是正式和直观地定义的。解释还可以包括单个规则参数的状态，以提供进一步的洞察力。在本文中，我们系统地介绍了这个新颖的框架，并提供了其实现的示例。

    This work introduces a framework to assess the relevance of individual linear temporal logic (LTL) constraints at specific times in a given path plan, a task we refer to as "pointwise-in-time" explanation. We develop this framework, featuring a status assessment algorithm, for agents which execute finite plans in a discrete-time, discrete-space setting expressible via a Kripke structure. Given a plan on this structure and a set of LTL rules which are known to constrain the agent, the algorithm responds to two types of user queries to produce explanation. For the selected query time, explanations identify which rules are active, which have just been satisfied, and which are inactive, where the framework status criteria are formally and intuitively defined. Explanations may also include the status of individual rule arguments to provide further insight. In this paper, we systematically present this novel framework and provide an example of its implementation.
    
[^78]: SPRINT：通过语言指令 relabeling 实现可扩展的策略预训练

    SPRINT: Scalable Policy Pre-Training via Language Instruction Relabeling. (arXiv:2306.11886v1 [cs.RO])

    [http://arxiv.org/abs/2306.11886](http://arxiv.org/abs/2306.11886)

    SPRINT 提出了一种离线策略预训练方法，通过指令重标记及离线强化学习实现可扩展的预训练任务，大大减少了预训练所需的人力，同时使机器人能够获取更丰富的技能库，相较于之前的预训练方法，能够更快地学习新的长时间跨度任务。

    

    预训练机器人策略并赋予丰富的技能集合可以大大加速下游任务的学习。先前的研究通过自然语言指令定义预训练任务，但这需要人为地注释数十万个指令。因此，我们提出了 SPRINT，这是一种可扩展的离线策略预训练方法，可大大减少预训练多样的技能所需的人力。我们的方法使用两个核心想法来自动扩展基础预训练任务：通过大型语言模型来进行指令重标记和通过离线强化学习进行交叉轨迹技能链接。因此，SPRINT 预训练可以为机器人装备更丰富的技能库。在家庭模拟器和真实机器人厨房操作任务中的实验结果表明，SPRINT 相对于之前的预训练方法能够更快地学习新的长时间跨度任务。

    Pre-training robot policies with a rich set of skills can substantially accelerate the learning of downstream tasks. Prior works have defined pre-training tasks via natural language instructions, but doing so requires tedious human annotation of hundreds of thousands of instructions. Thus, we propose SPRINT, a scalable offline policy pre-training approach which substantially reduces the human effort needed for pre-training a diverse set of skills. Our method uses two core ideas to automatically expand a base set of pre-training tasks: instruction relabeling via large language models and cross-trajectory skill chaining through offline reinforcement learning. As a result, SPRINT pre-training equips robots with a much richer repertoire of skills. Experimental results in a household simulator and on a real robot kitchen manipulation task show that SPRINT leads to substantially faster learning of new long-horizon tasks than previous pre-training approaches. Website at https://clvrai.com/spr
    
[^79]: 教科书是你需要的全部。 (arXiv:2306.11644v2 [cs.CL] UPDATED)

    Textbooks Are All You Need. (arXiv:2306.11644v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2306.11644](http://arxiv.org/abs/2306.11644)

    phi-1是一个新的大型代码语言模型，通过精心训练和优化，尽管规模相对较小，但在准确率和新的性质方面表现出了令人惊讶的结果。

    

    我们介绍了一个新的大型代码语言模型phi-1，其体积明显小于竞争模型：phi-1是一个基于Transformer的模型，拥有13亿个参数，在8个A100上进行了4天的训练，使用了来自网络的“教科书质量”数据（60亿个标记）和使用GPT-3.5合成生成的教科书和练习（10亿个标记）。尽管规模小，phi-1在HumanEval上的pass@1准确率为50.6％，在MBPP上为55.5％。与我们在编码练习数据集上进行微调之前的模型 phi-1-base 和具有相同流程的350M参数的较小模型 phi-1-small 相比，它还展现了令人惊讶的新的性质，phi-1-small 在 HumanEval 上仍达到45％的准确率。

    We introduce phi-1, a new large language model for code, with significantly smaller size than competing models: phi-1 is a Transformer-based model with 1.3B parameters, trained for 4 days on 8 A100s, using a selection of ``textbook quality" data from the web (6B tokens) and synthetically generated textbooks and exercises with GPT-3.5 (1B tokens). Despite this small scale, phi-1 attains pass@1 accuracy 50.6% on HumanEval and 55.5% on MBPP. It also displays surprising emergent properties compared to phi-1-base, our model before our finetuning stage on a dataset of coding exercises, and phi-1-small, a smaller model with 350M parameters trained with the same pipeline as phi-1 that still achieves 45% on HumanEval.
    
[^80]: 超bolic活跃学习在域转移下的语义分割中的应用

    Hyperbolic Active Learning for Semantic Segmentation under Domain Shift. (arXiv:2306.11180v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2306.11180](http://arxiv.org/abs/2306.11180)

    这项研究首次在Poincaré双曲球模型中运用超bolic活跃学习方法，利用区域内像素嵌入的半径变化作为新的数据获取策略，以提升域转移下语义分割的性能。

    

    对于域转移下的语义分割任务，基于图像区域和伪标签的主动学习获取策略是最先进的。在区域内存在不同类别的伪标签可以识别出不同类别之间的像素，这是一种高效的主动学习数据获取策略。然而，由于设计限制，伪标签的变化仅限于选择类别的轮廓，限制了最终的主动学习性能。我们首次在Poincaré双曲球模型中使用超bolic方法来进行语义分割的主动学习，并利用区域内像素嵌入的半径变化作为一种新的数据获取策略。这源于一种无层次约束训练的超bolic空间的新颖几何特性，我们通过实验证明了这一点。也就是说，类别被映射到具有相当内类半径方差的紧凑超bolic区域，因为模型将难以解释的类别放置在更密集的超bolic区域内。

    For the task of semantic segmentation (SS) under domain shift, active learning (AL) acquisition strategies based on image regions and pseudo labels are state-of-the-art (SoA). The presence of diverse pseudo-labels within a region identifies pixels between different classes, which is a labeling efficient active learning data acquisition strategy. However, by design, pseudo-label variations are limited to only select the contours of classes, limiting the final AL performance. We approach AL for SS in the Poincar\'e hyperbolic ball model for the first time and leverage the variations of the radii of pixel embeddings within regions as a novel data acquisition strategy. This stems from a novel geometric property of a hyperbolic space trained without enforced hierarchies, which we experimentally prove. Namely, classes are mapped into compact hyperbolic areas with a comparable intra-class radii variance, as the model places classes of increasing explainable difficulty at denser hyperbolic are
    
[^81]: 揭示视频问答模型中联合多模态理解的幻象

    Revealing the Illusion of Joint Multimodal Understanding in VideoQA Models. (arXiv:2306.08889v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2306.08889](http://arxiv.org/abs/2306.08889)

    通过设计轻量级探针QUAG和替代方法QUAG-attention，发现视频问答模型在多模态理解方面存在幻象，即使在多模态损伤下仍能保持高性能，且用更少的计算量实现相似的性能。

    

    尽管视频问答（VideoQA）Transformer模型在标准基准测试中表现出竞争力，但其成功原因尚未完全理解。这些模型是否能够共同捕捉和利用视频和文本中丰富的多模态结构和动态性？或者它们仅仅是利用了捷径来获得高分？因此，我们设计了一个轻量级且非参数化的探针“QUAG”（QUadrant AveraGe），以对多模态表示进行批判性分析。QUAG通过在推理过程中系统地消除模型的耦合多模态理解来促进联合数据集-模型研究。令人惊讶的是，它表明即使在多模态损伤下，模型仍能保持高性能。我们将QUAG扩展为“QUAG-attention”，这是一个简化且表达能力较弱的自注意力替代方法。我们发现，带有QUAG-attention的模型在没有任何微调的情况下能够达到类似的性能，而且计算量显著减少。这些发现表明当前的VideoQA模型在理解多模态信息时存在一定的幻象。

    While VideoQA Transformer models demonstrate competitive performance on standard benchmarks, the reasons behind their success are not fully understood. Do these models jointly capture and leverage the rich multimodal structures and dynamics from video and text? Or are they merely exploiting shortcuts to achieve high scores? Hence, we design $\textit{QUAG}$ (QUadrant AveraGe), a lightweight and non-parametric probe, to critically analyze multimodal representations. QUAG facilitates combined dataset-model study by systematic ablation of model's coupled multimodal understanding during inference. Surprisingly, it demonstrates that the models manage to maintain high performance even under multimodal impairment. We extend QUAG to design "QUAG-attention", a simplistic and less-expressive replacement of self-attention. We find that the models with QUAG-attention achieve similar performance with significantly less mulops without any finetuning. These findings indicate that the current VideoQA b
    
[^82]: Mol-Instructions: 一个大规模生物分子指令数据集，为大语言模型提供支持

    Mol-Instructions: A Large-Scale Biomolecular Instruction Dataset for Large Language Models. (arXiv:2306.08018v1 [q-bio.QM])

    [http://arxiv.org/abs/2306.08018](http://arxiv.org/abs/2306.08018)

    Mol-Instructions是一个专门为生物分子领域设计的综合指令数据集，可以显著提高大语言模型在生物领域中的适应能力和认知敏锐度。

    

    大语言模型（LLM）以其卓越的任务处理能力和创新的输出，在许多领域推动了重大进展。然而，它们在生物分子研究等专业领域的熟练应用还受到限制。为了解决这个挑战，我们介绍了Mol-Instructions，这是一个经过精心策划、专门针对生物分子领域设计的综合指令数据集。Mol-Instructions由三个关键组成部分组成：分子导向指令、蛋白质导向指令和生物分子文本指令，每个部分都被策划用于增强LLM对生物分子特性和行为的理解和预测能力。通过对代表性LLM的广泛指令调整实验，我们强调了Mol-Instructions在增强大模型在生物分子研究复杂领域内的适应能力和认知敏锐度方面的潜力，从而促进生物分子领域的进一步发展。

    Large Language Models (LLMs), with their remarkable task-handling capabilities and innovative outputs, have catalyzed significant advancements across a spectrum of fields. However, their proficiency within specialized domains such as biomolecular studies remains limited. To address this challenge, we introduce Mol-Instructions, a meticulously curated, comprehensive instruction dataset expressly designed for the biomolecular realm. Mol-Instructions is composed of three pivotal components: molecule-oriented instructions, protein-oriented instructions, and biomolecular text instructions, each curated to enhance the understanding and prediction capabilities of LLMs concerning biomolecular features and behaviors. Through extensive instruction tuning experiments on the representative LLM, we underscore the potency of Mol-Instructions to enhance the adaptability and cognitive acuity of large models within the complex sphere of biomolecular studies, thereby promoting advancements in the biomol
    
[^83]: Push: 并发概率编程用于贝叶斯深度学习

    Push: Concurrent Probabilistic Programming for Bayesian Deep Learning. (arXiv:2306.06528v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2306.06528](http://arxiv.org/abs/2306.06528)

    Push是一个并发概率编程库，用于贝叶斯深度学习（BDL），可以在多GPU硬件上执行BDL推理算法。该库通过将神经网络表示为粒子，并允许粒子之间的异步通信和各种参数更新，简化了BDL实验和扩展粒子操作的过程。

    

    我们介绍了一个名为Push的库，采用概率编程的方法来进行贝叶斯深度学习（BDL）。该库可在多GPU硬件上并发执行BDL推理算法，用于神经网络（NN）模型。为实现这一目标，Push引入了一种抽象，将输入NN表示为一个粒子。Push使得创建粒子变得容易，以便于复制和粒子之间的异步通信，以实现各种参数更新，包括常见的BDL算法。我们希望通过Push降低进行BDL实验的门槛，通过简化在多GPU上扩展粒子的操作。我们评估了利用单节点多GPU设备进行视觉和科学机器学习（SciML）任务时的粒子扩展行为。

    We introduce a library called Push that takes a probabilistic programming approach to Bayesian deep learning (BDL). This library enables concurrent execution of BDL inference algorithms on multi-GPU hardware for neural network (NN) models. To accomplish this, Push introduces an abstraction that represents an input NN as a particle. Push enables easy creation of particles so that an input NN can be replicated and particles can communicate asynchronously so that a variety of parameter updates can be expressed, including common BDL algorithms. Our hope is that Push lowers the barrier to experimenting with BDL by streamlining the scaling of particles across GPUs. We evaluate the scaling behavior of particles on single-node multi-GPU devices on vision and scientific machine learning (SciML) tasks.
    
[^84]: Gode -- 将生物化学知识图谱集成到分子图神经网络的预训练中

    Gode -- Integrating Biochemical Knowledge Graph into Pre-training Molecule Graph Neural Network. (arXiv:2306.01631v1 [cs.LG])

    [http://arxiv.org/abs/2306.01631](http://arxiv.org/abs/2306.01631)

    本研究提出了一种新的方法，在分子结构和生物医学知识图谱中集成多个领域信息，通过自我监督策略预先训练更广泛和更强大的表示，并在化学属性预测任务上展示出出色的性能。

    

    分子属性的准确预测对于促进创新治疗方法的发展和理解化学物质和生物系统之间复杂的相互作用至关重要。本研究提出了一种新的方法，将单个分子结构的图表示与生物医学知识图谱 (KG) 的多个领域信息进行集成。通过集成两个级别的信息，我们可以使用自我监督策略预先训练更广泛和更强大的表示，用于分子级和 KG 级预测任务。在性能评估方面，我们在 11 个具有挑战性的化学属性预测任务上微调我们预先训练的模型。我们的框架的结果表明，我们微调的模型优于现有的最先进的模型。

    The precise prediction of molecular properties holds paramount importance in facilitating the development of innovative treatments and comprehending the intricate interplay between chemicals and biological systems. In this study, we propose a novel approach that integrates graph representations of individual molecular structures with multi-domain information from biomedical knowledge graphs (KGs). Integrating information from both levels, we can pre-train a more extensive and robust representation for both molecule-level and KG-level prediction tasks with our novel self-supervision strategy. For performance evaluation, we fine-tune our pre-trained model on 11 challenging chemical property prediction tasks. Results from our framework demonstrate our fine-tuned models outperform existing state-of-the-art models.
    
[^85]: 多窗口本地-全局注意力的掩码自编码器是更好的音频学习器

    Masked Autoencoders with Multi-Window Local-Global Attention Are Better Audio Learners. (arXiv:2306.00561v2 [cs.SD] UPDATED)

    [http://arxiv.org/abs/2306.00561](http://arxiv.org/abs/2306.00561)

    多窗口本地-全局注意力的掩码自编码器（MW-MAE）在音频学习任务中表现出更好的性能和通用表示能力，并具有更好的可扩展性。

    

    在这项工作中，我们提出了一种配备了新型多窗口多头注意力模块的多窗口掩码自编码器（MW-MAE），通过几个不同的本地和全局窗口的注意力头，有助于在每个解码器变压器块中对局部-全局交互进行建模。在十个下游音频任务的实证结果表明，MW-MAEs在整体性能上始终优于标准MAEs，并学习到更好的通用音频表示，同时显示出更好的可扩展性。通过研究注意距离和熵，发现MW-MAE编码器学习到具有更宽广的本地和全局注意力的头部。通过投影加权典型相关分析（PWCCA）分析注意头特征表示，显示MW-MAE的解码器层中具有相同窗口大小的注意力头学习到相关的特征表示，这使得每个块能够独立捕捉到本地和全局信息。

    In this work, we propose a Multi-Window Masked Autoencoder (MW-MAE) fitted with a novel Multi-Window Multi-Head Attention (MW-MHA) module that facilitates the modelling of local-global interactions in every decoder transformer block through attention heads of several distinct local and global windows. Empirical results on ten downstream audio tasks show that MW-MAEs consistently outperform standard MAEs in overall performance and learn better general-purpose audio representations, along with demonstrating considerably better scaling characteristics. Investigating attention distances and entropies reveals that MW-MAE encoders learn heads with broader local and global attention. Analyzing attention head feature representations through Projection Weighted Canonical Correlation Analysis (PWCCA) shows that attention heads with the same window sizes across the decoder layers of the MW-MAE learn correlated feature representations which enables each block to independently capture local and glo
    
[^86]: 稳健的各向异性正则化

    Stable Anisotropic Regularization. (arXiv:2305.19358v1 [cs.CL])

    [http://arxiv.org/abs/2305.19358](http://arxiv.org/abs/2305.19358)

    本文提出了一种新颖的正则化方法I-STAR，可以增加模型的稳定性，提高性能，并改善自然语言处理中的组合表示问题。

    

    鉴于大型语言模型（LLMs）的成功，研究模型激活的属性已引起了相当大的兴趣。文献普遍认为LLMs表示由少数具有极高方差和幅度的“异常维度”主导。自然语言处理（NLP）中的几项研究试图减轻这些异常维度的影响，并迫使LLMs成为各向同性（即在嵌入空间中所有维度具有均匀方差）的。各向同性被认为是LLMs的一种理想属性，可以提高模型性能并更加贴近人类直觉的文本表示。然而，关于NLP中各向同性的许多观点都是基于嵌入的平均余弦相似度，最近已经表明这是一种有缺陷的各向同性度量。在本文中，我们提出了I-STAR：基于IsoScore$^{\star}$的稳定各向异性正则化，这是一种新颖的正则化方法，可以用于增加模型的稳定性并提高性能。

    Given the success of Large Language Models (LLMs), there has been considerable interest in studying the properties of model activations. The literature overwhelmingly agrees that LLM representations are dominated by a few ``outlier dimensions'' with exceedingly high variance and magnitude. Several studies in Natural Language Processing (NLP) have sought to mitigate the impact of such outlier dimensions and force LLMs to be isotropic (i.e., have uniform variance across all dimensions in embedding space). Isotropy is thought to be a desirable property for LLMs that improves model performance and more closely aligns textual representations with human intuition. However, many of the claims regarding isotropy in NLP have been based on the average cosine similarity of embeddings, which has recently been shown to be a flawed measure of isotropy. In this paper, we propose I-STAR: IsoScore$^{\star}$-based STable Anisotropic Regularization, a novel regularization method that can be used to incre
    
[^87]: 如何有效地在强化学习中进行人类反馈查询？

    How to Query Human Feedback Efficiently in RL?. (arXiv:2305.18505v1 [cs.LG])

    [http://arxiv.org/abs/2305.18505](http://arxiv.org/abs/2305.18505)

    该论文提出了一种针对强化学习中人类反馈查询的有效采样方法，以在最少的人类反馈下学习最佳策略，并可应用于具有线性参数化和未知过渡的偏好模型，并引入了基于行动比较反馈的RLHF。

    

    人类反馈强化学习（RLHF）是一种范例，在此范例下，RL代理学习使用对轨迹的成对优先级反馈来最优化任务，而不是使用明确的奖励信号。尽管RLHF在微调语言模型方面已经取得了实用成功，但现有的实证研究并未解决如何高效采样轨迹对以查询人类反馈的挑战。在本研究中，我们提出了一种有效的采样方法，用于获取探索性轨迹，在收集任何人类反馈之前，使学习隐藏的奖励函数更加准确。理论分析表明，与现有文献相比，我们的算法在线性参数化和未知过渡的基于偏好模型下学习最优策略所需的人类反馈更少。具体而言，我们的框架可以纳入线性和低秩MDPs。此外，我们研究了使用基于行动比较的反馈的RLHF，并介绍了一种高效的采样方法，以在优化具有有限反馈的任务时获得探索性轨迹。

    Reinforcement Learning with Human Feedback (RLHF) is a paradigm in which an RL agent learns to optimize a task using pair-wise preference-based feedback over trajectories, rather than explicit reward signals. While RLHF has demonstrated practical success in fine-tuning language models, existing empirical work does not address the challenge of how to efficiently sample trajectory pairs for querying human feedback. In this study, we propose an efficient sampling approach to acquiring exploratory trajectories that enable accurate learning of hidden reward functions before collecting any human feedback. Theoretical analysis demonstrates that our algorithm requires less human feedback for learning the optimal policy under preference-based models with linear parameterization and unknown transitions, compared to the existing literature. Specifically, our framework can incorporate linear and low-rank MDPs. Additionally, we investigate RLHF with action-based comparison feedback and introduce an
    
[^88]: C-MCTS: 安全规划与蒙特卡洛树搜索

    C-MCTS: Safe Planning with Monte Carlo Tree Search. (arXiv:2305.16209v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2305.16209](http://arxiv.org/abs/2305.16209)

    C-MCTS 提出了一种解决有约束的决策问题的方法，通过训练安全评判器进行成本估计，并在部署期间通过剪枝不安全轨迹来限制探索，实现了更高的奖励和更高效的规划步骤。

    

    有约束的马尔可夫决策过程（CMDP）可以解决受约束的安全决策问题。尽管CMDP在强化学习的文献中得到了广泛研究，但对于使用MCTS等基于采样的规划算法来解决CMDP的研究却很少。以往的方法在成本方面保守行事，通过使用蒙特卡洛成本估计来避免违反约束，但这种估计存在高方差。我们提出了约束MCTS（C-MCTS），它使用先前在代理部署之前通过时间差分学习训练的安全评判器来估计成本。在部署期间，评判器通过剪枝不安全轨迹来限制探索。C-MCTS满足成本约束，但操作接近约束边界，比以往的工作获得更高的奖励。作为一个很好的副产品，这个规划器在规划步骤方面更加高效。最重要的是，在模型下，

    The Constrained Markov Decision Process (CMDP) formulation allows to solve safety-critical decision making tasks that are subject to constraints. While CMDPs have been extensively studied in the Reinforcement Learning literature, little attention has been given to sampling-based planning algorithms such as MCTS for solving them. Previous approaches perform conservatively with respect to costs as they avoid constraint violations by using Monte Carlo cost estimates that suffer from high variance. We propose Constrained MCTS (C-MCTS), which estimates cost using a safety critic that is trained with Temporal Difference learning in an offline phase prior to agent deployment. The critic limits exploration by pruning unsafe trajectories within MCTS during deployment. C-MCTS satisfies cost constraints but operates closer to the constraint boundary, achieving higher rewards than previous work. As a nice byproduct, the planner is more efficient w.r.t. planning steps. Most importantly, under model
    
[^89]: 在智能体和语言模型中被动学习主动因果策略

    Passive learning of active causal strategies in agents and language models. (arXiv:2305.16183v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2305.16183](http://arxiv.org/abs/2305.16183)

    通过被动学习，在智能体和语言模型中可以学习到一般化的主动因果策略，用于确定和使用因果关系结构。通过模仿专家数据进行训练的智能体能够在测试时推断和使用从未出现的因果链接，并将实验策略推广到从未观察到的新变量集。

    

    通过被动数据，我们能够学习到关于因果关系和实验的什么信息？鉴于被动训练的语言模型在工具使用等交互领域的最新成功，这个问题变得很重要。被动学习本质上是有限的。然而，我们展示了纯粹的被动学习实际上能够让智能体学习到一般化的策略，用于确定和使用因果关系结构，只要智能体能够在测试时干预。我们在形式上说明了首先进行实验，然后寻求目标的策略能够原则上使被动学习实现一般化。然后，我们从经验上展示了通过模仿专家数据进行训练的智能体在测试时能够推断和使用训练数据中从未出现的因果链接；这些智能体还能够将实验策略推广到从未在训练中观察到的新变量集。然后，我们展示了从被动数据中一般化因果干预和利用策略。

    What can be learned about causality and experimentation from passive data? This question is salient given recent successes of passively-trained language models in interactive domains such as tool use. Passive learning is inherently limited. However, we show that purely passive learning can in fact allow an agent to learn generalizable strategies for determining and using causal structures, as long as the agent can intervene at test time. We formally illustrate that learning a strategy of first experimenting, then seeking goals, can allow generalization from passive learning in principle. We then show empirically that agents trained via imitation on expert data can indeed generalize at test time to infer and use causal links which are never present in the training data; these agents can also generalize experimentation strategies to novel variable sets never observed in training. We then show that strategies for causal intervention and exploitation can be generalized from passive data ev
    
[^90]: 大型语言模型的自相矛盾幻觉：评估、检测和缓解

    Self-contradictory Hallucinations of Large Language Models: Evaluation, Detection and Mitigation. (arXiv:2305.15852v1 [cs.CL])

    [http://arxiv.org/abs/2305.15852](http://arxiv.org/abs/2305.15852)

    本文对大型语言模型的自相矛盾幻觉进行了评估、检测和缓解，探究了这一幻觉形式的普遍存在性。通过设计框架有效触发自相矛盾，发现不同语言模型中这种现象都频繁出现。ChatGPT和GPT-4能够准确识别自相矛盾，而Vicuna-13B则有些困难。

    

    大型语言模型容易产生幻想的文本。自相矛盾是一种重要的幻觉形式，指的是语言模型在同一语境中生成两个矛盾的句子。本文针对最先进、经过指导的语言模型，对自相矛盾进行了全面的分析、评估、检测和缓解。我们设计了一个框架来有效地触发自相矛盾，评估结果表明，无论是对于著名的还是不太出名的话题，不同的语言模型中自相矛盾都经常发生。

    Large language models (large LMs) are susceptible to producing text with hallucinated content. Self-contradiction, where the LM generates two contradictory sentences within the same context, is an important form of hallucination. In this work, we present a comprehensive analysis on self-contradiction for state-of-the-art, instruction-tuned LMs, including evaluation, detection, and mitigation. To effectively trigger self-contradictions, we design a framework that constrains LMs to generate appropriate sentence pairs. Our evaluation on these sentence pairs reveals that self-contradictions occur frequently across different LMs for both famous and lesser-known topics. Next, we prompt the LMs to detect self-contradictions. Our results indicate that ChatGPT and GPT-4 are able to accurately identify self-contradictions, while Vicuna-13B struggles to do so. For example, with our best prompting method, ChatGPT achieves 91.0% precision and 80.5% recall on the sentence pairs generated by itself. 
    
[^91]: 基于谱角度剖析生物数据中图神经网络的尺寸可泛化性：观点和实践

    Size Generalizability of Graph Neural Networks on Biological Data: Insights and Practices from the Spectral Perspective. (arXiv:2305.15611v1 [cs.LG])

    [http://arxiv.org/abs/2305.15611](http://arxiv.org/abs/2305.15611)

    本文通过谱角度的方法，研究了GNNs的尺寸可泛化性问题，并在真实生物数据集上进行了实验，发现GNNs在度分布和谱分布偏移时均表现敏感，在同一数据集的大图上的性能仍然下降，揭示了 GNNs的尺寸可泛化性问题。

    

    本文探讨了图神经网络 (GNNs) 是否具有从小图中学习的知识可推广到同一领域的大图中。之前的研究表明，不同大小的图之间的分布偏移，尤其是度分布，可能会导致图分类任务的性能下降。然而，在生物数据集中，度数是有界的，因此度分布的偏移很小。即使度分布偏移很小，我们观察到GNNs在同一数据集的大图上的性能仍然下降，暗示有其他原因。事实上，以往对于真实数据集中各种图尺寸引起的分布偏移类型和属性的探索不足。此外，以前的尺寸可泛化性分析大多集中在空间领域。为填补这些空白，我们采用谱角度去研究GNNs在生物图数据上的尺寸可泛化性。我们首先提出一个新框架来模拟各种类型的度分布偏移，并利用它来测试GNNs 在真实生物数据集上的尺寸可泛化性。我们的实验表明，除了度分布偏移外，GNNs 还对图大小变化引起的谱分布偏移很敏感。我们进一步分析了不同的GNN模型的影响，并表明，一些模型比其他模型更具有尺寸泛化性。本文展示了关于GNNs尺寸可泛化性问题的新观点和实践，并为该领域的未来研究提供了有益的洞察和建议。

    We investigate the question of whether the knowledge learned by graph neural networks (GNNs) from small graphs is generalizable to large graphs in the same domain. Prior works suggest that the distribution shift, particularly in the degree distribution, between graphs of different sizes can lead to performance degradation in the graph classification task. However, this may not be the case for biological datasets where the degrees are bounded and the distribution shift of degrees is small. Even with little degree distribution shift, our observations show that GNNs' performance on larger graphs from the same datasets still degrades, suggesting other causes. In fact, there has been a lack of exploration in real datasets to understand the types and properties of distribution shifts caused by various graph sizes. Furthermore, previous analyses of size generalizability mostly focus on the spatial domain.  To fill these gaps, we take the spectral perspective and study the size generalizabilit
    
[^92]: STAR: 利用大型语言模型通过结构到文本数据生成改进低资源信息抽取

    STAR: Improving Low-Resource Information Extraction by Structure-to-Text Data Generation with Large Language Models. (arXiv:2305.15090v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.15090](http://arxiv.org/abs/2305.15090)

    STAR是一种利用大型语言模型合成数据实例的数据生成方法，用于改进低资源信息抽取，为实际应用提供了需要最少人工标注的解决方案。

    

    信息抽取任务，如事件抽取，需要对输出结构和子任务依赖进行深入理解。为了获得合理的性能，它们严重依赖于以（段落，目标结构）对的形式的任务特定训练数据。然而，通过人工注释获得这样的数据是昂贵的，因此对于实际应用，我们迫切需要需要最少人工标注的低资源信息抽取方法。使用合成训练数据对监督模型进行微调可能是一种通用方法，但现有的数据生成方法要么仍然依赖于大规模的真实数据，要么由于性能差而无法应用于复杂的信息抽取任务。为了解决这些挑战，我们提出了STAR，一种利用大型语言模型（LLMs）根据有限的种子示例合成数据实例，从而提高低资源信息抽取性能的数据生成方法。

    Information extraction tasks such as event extraction require an in-depth understanding of the output structure and sub-task dependencies. They heavily rely on task-specific training data in the form of (passage, target structure) pairs to obtain reasonable performance. However, obtaining such data through human annotation is costly, leading to a pressing need for low-resource information extraction approaches that require minimal human labeling for real-world applications. Fine-tuning supervised models with synthesized training data would be a generalizable method, but the existing data generation methods either still rely on large-scale ground-truth data or cannot be applied to complicated IE tasks due to their poor performance. To address these challenges, we propose STAR, a data generation method that leverages Large Language Models (LLMs) to synthesize data instances given limited seed demonstrations, thereby boosting low-resource information extraction performance. Our approach i
    
[^93]: 利用强化学习训练扩散模型

    Training Diffusion Models with Reinforcement Learning. (arXiv:2305.13301v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2305.13301](http://arxiv.org/abs/2305.13301)

    本文研究了利用强化学习方法直接优化扩散模型以实现下游对象的问题，并提出一种称之为去噪扩散策略优化（DDPO）的有效策略梯度算法，能够适应难以通过提示表达的图像压缩等目标，以及通过人类反馈得出的美学质量等目标。

    

    扩散模型是一类灵活的生成模型，采用对数似然目标的近似训练。然而，大多数扩散模型的使用案例并不关注似然，而是关注人类感知的图像质量或药物效力等下游目标。本文研究利用强化学习方法直接优化扩散模型以实现此类目标。我们描述了将去噪视为多步决策问题的方法，并提出称之为去噪扩散策略优化（DDPO）的一类策略梯度算法，相对于替代的奖励加权似然方法更为有效。在实证研究中，DDPO能够适应难以通过提示表达的图像压缩等目标，以及通过人类反馈得出的美学质量等目标。最后，我们展示DDPO可以利用来自反馈的提示-图像对齐方式来进行优化。

    Diffusion models are a class of flexible generative models trained with an approximation to the log-likelihood objective. However, most use cases of diffusion models are not concerned with likelihoods, but instead with downstream objectives such as human-perceived image quality or drug effectiveness. In this paper, we investigate reinforcement learning methods for directly optimizing diffusion models for such objectives. We describe how posing denoising as a multi-step decision-making problem enables a class of policy gradient algorithms, which we refer to as denoising diffusion policy optimization (DDPO), that are more effective than alternative reward-weighted likelihood approaches. Empirically, DDPO is able to adapt text-to-image diffusion models to objectives that are difficult to express via prompting, such as image compressibility, and those derived from human feedback, such as aesthetic quality. Finally, we show that DDPO can improve prompt-image alignment using feedback from a 
    
[^94]: GraphCare: 使用个性化知识图谱提升医疗预测能力

    GraphCare: Enhancing Healthcare Predictions with Personalized Knowledge Graphs. (arXiv:2305.12788v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2305.12788](http://arxiv.org/abs/2305.12788)

    本论文提出了一种名为GraphCare的框架，通过使用个性化知识图谱来改进基于电子健康记录的医疗预测，并通过在两个公共数据集上的实验证明了其有效性。

    

    临床预测模型通常依赖于患者的电子健康记录(EHR)，但将医学知识整合到预测和决策中以提高效果具有挑战性。这是因为个性化预测需要个性化的知识图谱(KG)，而从患者EHR数据中生成个性化知识图谱很困难。为了解决这个问题，我们提出了一个名为\textsc{GraphCare}的开放式框架，它使用外部知识图谱来改进基于EHR的预测。我们的方法从大规模语言模型(LLM)和外部生物医学知识图谱中提取知识，构建个体化的患者知识图谱，然后使用我们提出的Bi-attention AugmenTed (BAT)图神经网络(GNN)进行医疗预测训练。在两个公共数据集MIMIC-III和MIMIC-IV上，\textsc{GraphCare}在四个关键的医疗预测任务上均超过了基准线：死亡率、再入院率、住院天数和药物推荐。在MIMIC-III上，它将AUROC提高了17.6%和6.6%，将F1得分提高了7.9%。

    Clinical predictive models often rely on patients' electronic health records (EHR), but integrating medical knowledge to enhance predictions and decision-making is challenging. This is because personalized predictions require personalized knowledge graphs (KGs), which are difficult to generate from patient EHR data. To address this, we propose \textsc{GraphCare}, an open-world framework that uses external KGs to improve EHR-based predictions. Our method extracts knowledge from large language models (LLMs) and external biomedical KGs to build patient-specific KGs, which are then used to train our proposed Bi-attention AugmenTed (BAT) graph neural network (GNN) for healthcare predictions. On two public datasets, MIMIC-III and MIMIC-IV, \textsc{GraphCare} surpasses baselines in four vital healthcare prediction tasks: mortality, readmission, length of stay (LOS), and drug recommendation. On MIMIC-III, it boosts AUROC by 17.6\% and 6.6\% for mortality and readmission, and F1-score by 7.9\% 
    
[^95]: 使用指令微调基础模型的多模态 Web 导航。

    Multimodal Web Navigation with Instruction-Finetuned Foundation Models. (arXiv:2305.11854v1 [cs.LG])

    [http://arxiv.org/abs/2305.11854](http://arxiv.org/abs/2305.11854)

    本文研究使用视觉语言基础模型进行数据驱动离线训练的 Web 代理，提出了一个指令跟随多模态代理WebGUM，将微调指令微调语言模型和视觉转换器，能够有效提高代理的基于视觉感知、HTML 理解和多步推理的能力。

    

    自主 Web 导航的进展受到了依赖数十亿次在线强化学习的探索性交互和具有领域特定模型设计的影响，这使得难以利用来自丰富领域外数据的泛化。在本工作中，我们研究了基于数据驱动的脱机训练，用于使用视觉语言基础模型的 Web 代理。我们提出了一个指令跟随多模态代理， WebGUM，它观察了网页截图和 HTML 页面，并输出 Web 导航操作，如单击和输入。WebGUM 是通过联合微调指令微调语言模型和视觉转换器在大量的演示语料库上训练的。我们凭经验证明，这种方法可以提高代理的基于视觉感知、HTML 理解和多步推理的能力，明显优于之前的工作。在 MiniWoB 基准测试中，我们超过之前最佳脱机方法 31.9% 以上，接近实现在线交互的表现。

    The progress of autonomous web navigation has been hindered by the dependence on billions of exploratory interactions via online reinforcement learning, and domain-specific model designs that make it difficult to leverage generalization from rich out-of-domain data. In this work, we study data-driven offline training for web agents with vision-language foundation models. We propose an instruction-following multimodal agent, WebGUM, that observes both webpage screenshots and HTML pages and outputs web navigation actions, such as click and type. WebGUM is trained by jointly finetuning an instruction-finetuned language model and a vision transformer on a large corpus of demonstrations. We empirically demonstrate this recipe improves the agent's ability of grounded visual perception, HTML comprehension and multi-step reasoning, outperforming prior works by a significant margin. On the MiniWoB benchmark, we improve over the previous best offline methods by more than 31.9%, being close to re
    
[^96]: CRITIC：大型语言模型可以通过工具交互批评进行自我校正

    CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing. (arXiv:2305.11738v1 [cs.CL])

    [http://arxiv.org/abs/2305.11738](http://arxiv.org/abs/2305.11738)

    本文提出了一个名为CRITIC的框架，使得大型语言模型可以通过与工具的交互校正自己的错误，从而避免生成出现不一致和问题行为的结果。

    

    近年来，大型语言模型的发展非常引人注目。然而，这些模型有时会出现不一致和问题行为，例如出现幻觉事实，生成有缺陷的代码或创建冒犯和有害的内容。与这些模型不同，人类通常使用外部工具来交叉检查和精炼他们的初步内容，例如使用搜索引擎进行事实检查或使用代码解释器进行调试。受这一观察的启发，我们引入了一个名为CRITIC的框架，允许LLMs（实质上是“黑盒子”）以类似于人类与工具交互的方式验证和逐步修正自己的输出。更具体地说，从初始输出开始，CRITIC与适当的工具交互以评估文本的某些方面，然后根据在此验证过程中获得的反馈修改输出。涉及自由形式问答、数学程序综合和毒性检测的全面评估表明，我们的框架使LLMs能够从错误中学习并纠正自己的错误。

    Recent developments in large language models (LLMs) have been impressive. However, these models sometimes show inconsistencies and problematic behavior, such as hallucinating facts, generating flawed code, or creating offensive and toxic content. Unlike these models, humans typically utilize external tools to cross-check and refine their initial content, like using a search engine for fact-checking, or a code interpreter for debugging. Inspired by this observation, we introduce a framework called CRITIC that allows LLMs, which are essentially "black boxes" to validate and progressively amend their own outputs in a manner similar to human interaction with tools. More specifically, starting with an initial output, CRITIC interacts with appropriate tools to evaluate certain aspects of the text, and then revises the output based on the feedback obtained during this validation process. Comprehensive evaluations involving free-form question answering, mathematical program synthesis, and toxi
    
[^97]: 扩散模型的结构剪枝

    Structural Pruning for Diffusion Models. (arXiv:2305.10924v1 [cs.LG])

    [http://arxiv.org/abs/2305.10924](http://arxiv.org/abs/2305.10924)

    本文提出了一种名为Diff-Pruning的高效压缩方法，通过一个Taylor展开过程来识别重要权重，从而从预先存在的模型中学习轻量级扩散模型，性能稳定，并在训练效率上显著提高。

    

    生成建模最近取得了显著的进展，主要是因为扩散概率模型（DPM）的转型意义。然而，这些模型的令人印象深刻的能力通常涉及到显著的计算开销，在训练和推理期间都是如此。为了应对这一挑战，我们提出了Diff-Pruning，一种专为从预先存在的模型中学习轻量级扩散模型而设计的高效压缩方法，无需进行大量的重新训练。Diff-Pruning的本质是通过剪枝时间步长的Taylor展开，在过滤掉无贡献扩散步骤和整合有信息的梯度来识别重要权重的过程。我们在四个不同数据集上进行的实证评估突出了我们所提出方法的两个主要优点：1）效率：它可以以原始训练投入的仅10％到20％的代价实现约50％的FLOPs减少; 2）一致性: 剪枝后的扩散模型产生的效果与原始模型相当，不会影响生成建模的质量。

    Generative modeling has recently undergone remarkable advancements, primarily propelled by the transformative implications of Diffusion Probabilistic Models (DPMs). The impressive capability of these models, however, often entails significant computational overhead during both training and inference. To tackle this challenge, we present Diff-Pruning, an efficient compression method tailored for learning lightweight diffusion models from pre-existing ones, without the need for extensive re-training. The essence of Diff-Pruning is encapsulated in a Taylor expansion over pruned timesteps, a process that disregards non-contributory diffusion steps and ensembles informative gradients to identify important weights. Our empirical assessment, undertaken across four diverse datasets highlights two primary benefits of our proposed method: 1) Efficiency: it enables approximately a 50% reduction in FLOPs at a mere 10% to 20% of the original training expenditure; 2) Consistency: the pruned diffusio
    
[^98]: 多智能体强化学习中的语义对齐任务分解

    Semantically Aligned Task Decomposition in Multi-Agent Reinforcement Learning. (arXiv:2305.10865v1 [cs.LG])

    [http://arxiv.org/abs/2305.10865](http://arxiv.org/abs/2305.10865)

    该论文提出了一种多智能体强化学习中的新方法SAMA，通过提前训练的语言模型和任务分解来解决ASG方法存在的样本效率问题和生成非实际任务奖励的子目标的问题。

    

    合作型MARL中的奖励稀疏问题着重于适当的信用分配。自动子目标生成（ASG）是最近出现的一种可行的MARL方法，其灵感来自于在内在驱动的增强学习中利用子目标。然而，从稀疏奖励中进行复杂任务规划的端到端学习无疑需要大量的培训样本。为了解决这个问题，我们提出了一种新的"解耦"决策方法，即在MARL中的语义对齐任务分解（SAMA），受到解耦表示学习的启发。

    The difficulty of appropriately assigning credit is particularly heightened in cooperative MARL with sparse reward, due to the concurrent time and structural scales involved. Automatic subgoal generation (ASG) has recently emerged as a viable MARL approach inspired by utilizing subgoals in intrinsically motivated reinforcement learning. However, end-to-end learning of complex task planning from sparse rewards without prior knowledge, undoubtedly requires massive training samples. Moreover, the diversity-promoting nature of existing ASG methods can lead to the "over-representation" of subgoals, generating numerous spurious subgoals of limited relevance to the actual task reward and thus decreasing the sample efficiency of the algorithm. To address this problem and inspired by the disentangled representation learning, we propose a novel "disentangled" decision-making method, Semantically Aligned task decomposition in MARL (SAMA), that prompts pretrained language models with chain-of-thou
    
[^99]: ConvXAI：通过对话提供异构的AI解释，支持人机科技写作

    ConvXAI: Delivering Heterogeneous AI Explanations via Conversations to Support Human-AI Scientific Writing. (arXiv:2305.09770v1 [cs.HC])

    [http://arxiv.org/abs/2305.09770](http://arxiv.org/abs/2305.09770)

    ConvXAI是一个基于对话的XAI系统，它集成了多种XAI类型，并将实际用户需求嵌入设计中，以提高实用性。

    

    尽管已经提出了各种各样的人工智能解释（XAI）方法来解释AI系统，但目前的方法是否对人类实用仍存在不一致的发现。为了改善XAI方法的实用性，一系列研究确定了现实世界中多样化和动态的用户需求与现有XAI方法之间的差距。虽然之前的研究设想将多种XAI方法集成到通用XAI界面（例如，基于对话或GUI的XAI系统）中以减轻这些差距，但缺少针对这些系统如何设计以满足实际用户需求的研究。在本研究中，我们提出了ConvXAI，这是一个基于对话的XAI系统，它结合了多种XAI类型，并赋予用户通过通用的XAI对话界面提出各种XAI问题的能力。特别地，我们创新地将实际用户需求（即，基于格式研究的四个原则）嵌入ConvXAI设计中，以提高实用性。

    While various AI explanation (XAI) methods have been proposed to interpret AI systems, whether the state-of-the-art XAI methods are practically useful for humans remains inconsistent findings. To improve the usefulness of XAI methods, a line of studies identifies the gaps between the diverse and dynamic real-world user needs with the status quo of XAI methods. Although prior studies envision mitigating these gaps by integrating multiple XAI methods into the universal XAI interfaces (e.g., conversational or GUI-based XAI systems), there is a lack of work investigating how these systems should be designed to meet practical user needs. In this study, we present ConvXAI, a conversational XAI system that incorporates multiple XAI types, and empowers users to request a variety of XAI questions via a universal XAI dialogue interface. Particularly, we innovatively embed practical user needs (i.e., four principles grounding on the formative study) into ConvXAI design to improve practical useful
    
[^100]: 评估ChatGPT的工作记忆容量

    Assessing Working Memory Capacity of ChatGPT. (arXiv:2305.03731v1 [cs.AI])

    [http://arxiv.org/abs/2305.03731](http://arxiv.org/abs/2305.03731)

    本文评估了最先进语言模型ChatGPT的工作记忆容量，结果显示其在N-back任务的行为表现与人类参与者相似，这为设计具有人类级认知能力的人工智能系统提供了关键洞察。

    

    工作记忆是人类智能和人工智能的关键方面，它作为信息临时存储和操作的工作空间。本文通过检查ChatGPT在N-back任务上的表现，调查了这一最先进语言模型的工作记忆容量。我们首先讨论了工作记忆对人类和人工智能的重要性，接着介绍了评估ChatGPT工作记忆容量的方法。研究比较了ChatGPT在言语和空间N- back任务上的行为表现与文献报道的人类参与者的表现，发现了显著的相似之处。我们的发现为设计具有人类级认知能力的人工智能系统的当前进展提供了关键洞察，并为通过人工智能模型理解人类工作记忆的未来努力提供了前景。

    Working memory is a critical aspect of both human intelligence and artificial intelligence (AI), serving as a workspace for the temporary storage and manipulation of information. This paper investigates working memory capacity of ChatGPT, a state-of-the-art language model, by examining its performance on N-back tasks. We begin by discussing the importance of working memory to humans and AI, followed by the methods employed to assess working memory capacity of ChatGPT. Our study compares behavioral performance of ChatGPT on verbal and spatial N-back tasks to that of human participants reported in the literature, revealing notable similarities. Our findings offer crucial insights into the current progress in designing AI systems with human-level cognitive abilities and hold promise for informing future endeavors aimed at enhancing AI working memory and understanding human working memory through AI models.
    
[^101]: 基于半监督学习的高维贝叶斯优化及优化无标签数据采样

    High-dimensional Bayesian Optimization via Semi-supervised Learning with Optimized Unlabeled Data Sampling. (arXiv:2305.02614v1 [cs.LG])

    [http://arxiv.org/abs/2305.02614](http://arxiv.org/abs/2305.02614)

    本文提出基于半监督学习的高维贝叶斯优化方法，利用特定的未标记数据采样、参数化采样分布的优化及动态选择无标记数据等策略，解决了高维贝叶斯优化难以处理的问题。

    

    贝叶斯优化（BO）是一种寻找黑箱函数全局最优解的强大工具。虽然黑箱函数的评估成本往往很高，但减少昂贵标记数据的使用是理想的。本文首次提出了一种教师-学生模型，利用半监督学习在BO环境下利用大量未标记的数据。其中，关键在于选择验证和未标记数据以提高BO的表现。为了优化无标签数据的采样，我们采用黑箱参数化采样分布，将其优化为所采用双层优化框架的一部分。更进一步，通过从动态适应的极值分布中选择未标签数据，我们证明了BO的性能可以进一步提高。我们的BO方法在学习后的低维潜在空间中运行，使其可扩展到高维问题。

    Bayesian optimization (BO) is a powerful tool for seeking the global optimum of black-box functions. While evaluations of the black-box functions can be highly costly, it is desirable to reduce the use of expensive labeled data. For the first time, we introduce a teacher-student model to exploit semi-supervised learning that can make use of large amounts of unlabelled data under the context of BO. Importantly, we show that the selection of the validation and unlabeled data is key to the performance of BO. To optimize the sampling of unlabeled data, we employ a black-box parameterized sampling distribution optimized as part of the employed bi-level optimization framework. Taking one step further, we demonstrate that the performance of BO can be further improved by selecting unlabeled data from a dynamically fitted extreme value distribution. Our BO method operates in a learned latent space with reduced dimensionality, making it scalable to high-dimensional problems. The proposed approac
    
[^102]: 差分隐私下的上下文学习

    Differentially Private In-Context Learning. (arXiv:2305.01639v1 [cs.LG])

    [http://arxiv.org/abs/2305.01639](http://arxiv.org/abs/2305.01639)

    本文提出了DP-ICL，实现了在隐私保证下对新任务的适应性。经过四个基准测试，发现其性能与非私有ICL相当。

    

    在部署大型语言模型（LLM）时，一个重要的问题是如何使用私有数据增强LLM。我们提出了"DP-ICL"来实现对新任务的适应性，同时保持隐私保证。DP-ICL通过使用"report-noisy-max"机制在示例集合上建立嘈杂一致性来进行私有推断。我们在四个基准测试上评估了DP-ICL，发现其与非私有ICL相比具有可比性的性能(<2%降级)。

    An important question in deploying large language models (LLMs) is how to augment LLMs with private data. We propose Differentially Private In-context Learning (DP-ICL) to enable LLMs to adapt to new tasks while maintaining privacy guarantees. DP-ICL performs private inference by establishing noisy consensus over an ensemble of exemplars using the Report-Noisy-Max mechanism. We evaluate DP-ICL on four benchmarks and find that it achieves comparable performance (<2\% degradation) with non-private ICL.
    
[^103]: 面向遥感时序数据的轻量级预训练Transformer

    Lightweight, Pre-trained Transformers for Remote Sensing Timeseries. (arXiv:2304.14065v1 [cs.CV])

    [http://arxiv.org/abs/2304.14065](http://arxiv.org/abs/2304.14065)

    设计针对远程传感器数据的自监督学习模型和训练技术，可以得到表现更好且更小的模型。预训练的遥感时间序列Transformer（Presto）在几个遥感基准测试中实现了最先进的结果。

    

    远程传感数据的机器学习算法在社会相关应用方面具有广泛的应用，但用于训练这些算法的标签可能很难或不可能获得。这个挑战已经推动了自监督学习领域的研究，旨在通过遥感数据解锁在标记数据集较小的地理位置或应用领域中使用机器学习。我们展示了为遥感数据设计模型和自监督训练技术可以得到更小、更优秀的模型。我们介绍了Remote Sensing Transformer（Presto），它是一种基于Transformer的模型，使用新颖的自监督目标对遥感时间序列数据进行预训练。我们的实验表明，与在自然图像上训练的可比模型相比，Presto在几个遥感基准测试中实现了最先进的结果，同时需要数量级更少的参数。

    Machine learning algorithms for parsing remote sensing data have a wide range of societally relevant applications, but labels used to train these algorithms can be difficult or impossible to acquire. This challenge has spurred research into self-supervised learning for remote sensing data aiming to unlock the use of machine learning in geographies or application domains where labelled datasets are small. Current self-supervised learning approaches for remote sensing data draw significant inspiration from techniques applied to natural images. However, remote sensing data has important differences from natural images -- for example, the temporal dimension is critical for many tasks and data is collected from many complementary sensors. We show that designing models and self-supervised training techniques specifically for remote sensing data results in both smaller and more performant models. We introduce the Pretrained Remote Sensing Transformer (Presto), a transformer-based model pre-tr
    
[^104]: 基于样本效率的模型驱动量子控制强化学习

    Sample-efficient Model-based Reinforcement Learning for Quantum Control. (arXiv:2304.09718v1 [quant-ph])

    [http://arxiv.org/abs/2304.09718](http://arxiv.org/abs/2304.09718)

    本论文提出了一种基于模型的强化学习方法，通过受到神经常微分方程进展的启发，这个方法采用自动微分的ODE表达由可学习的汉密尔顿安排参数化的模型来近似环境，在门控制和汉密尔顿参数的学习中通过系统交互解决问题。该方法在样本复杂度方面比标准基于模型自由的强化学习方法具有一个数量级的优势，适用于噪声时变门优化。

    

    我们提出了一种基于模型的强化学习方法，用于噪声时变门优化，其样本复杂度优于基于模型自由的强化学习。样本复杂度是控制器与物理系统交互的次数。借助一个归纳偏置，受最近神经常微分方程的进展启发，我们使用可微的ODE，其由可学习的汉密尔顿安排参数化，以表示模型近似环境，其时变部分（包括控制）完全已知。控制器和连续时域独立参数的汉密尔顿学习是通过与系统的交互来解决的。在真实数值实验中，我们展示了使用我们方法在准备一些标准单量子门的闭合和开放系统动态时，在样本复杂度方面与标准模型自由强化学习相比，具有一个数量级的优势，这包括单次测量、任意希尔伯特空间截断和不确定性等。

    We propose a model-based reinforcement learning (RL) approach for noisy time-dependent gate optimization with improved sample complexity over model-free RL. Sample complexity is the number of controller interactions with the physical system. Leveraging an inductive bias, inspired by recent advances in neural ordinary differential equations (ODEs), we use an auto-differentiable ODE parametrised by a learnable Hamiltonian ansatz to represent the model approximating the environment whose time-dependent part, including the control, is fully known. Control alongside Hamiltonian learning of continuous time-independent parameters is addressed through interactions with the system. We demonstrate an order of magnitude advantage in the sample complexity of our method over standard model-free RL in preparing some standard unitary gates with closed and open system dynamics, in realistic numerical experiments incorporating single shot measurements, arbitrary Hilbert space truncations and uncertaint
    
[^105]: 使用深度学习在眼部图像中准确定位角膜反射

    Precise localization of corneal reflections in eye images using deep learning trained on synthetic data. (arXiv:2304.05673v1 [cs.CV])

    [http://arxiv.org/abs/2304.05673](http://arxiv.org/abs/2304.05673)

    该论文提出了一种使用深度学习在眼部图像中准确定位角膜反射的方法，无需对真实眼部图像进行注释，仅使用模拟数据进行训练，该方法表现出色且提供了一种可行的解决方案。

    

    我们提出了一种深度学习方法，用于准确地定位单个眼部图像中角膜反射的中心。与以往的方法不同，我们使用了一个纯粹使用模拟数据训练的卷积神经网络（CNN）。使用只有模拟数据的方法的好处是完全避开了需要对真实眼部图像进行监督训练的繁琐注释过程。为了系统地评估我们方法的准确性，我们首先对放置在不同背景中和嵌入不同噪声水平的图像进行了测试。其次，我们对从真实眼睛中拍摄的高质量视频测试了该方法。我们的方法在真实眼部图像上表现出色，在空间精度方面降低了35％，并在模拟图像方面以空间准确性与最先进的方法相当。我们得出结论，我们的方法提供了一种精确的角膜反射中心定位方法，并提供了一种解决方案。

    We present a deep learning method for accurately localizing the center of a single corneal reflection (CR) in an eye image. Unlike previous approaches, we use a convolutional neural network (CNN) that was trained solely using simulated data. Using only simulated data has the benefit of completely sidestepping the time-consuming process of manual annotation that is required for supervised training on real eye images. To systematically evaluate the accuracy of our method, we first tested it on images with simulated CRs placed on different backgrounds and embedded in varying levels of noise. Second, we tested the method on high-quality videos captured from real eyes. Our method outperformed state-of-the-art algorithmic methods on real eye images with a 35% reduction in terms of spatial precision, and performed on par with state-of-the-art on simulated images in terms of spatial accuracy.We conclude that our method provides a precise method for CR center localization and provides a solutio
    
[^106]: 面向类别不均问题的集成学习和数据增强模型综述：组合、实现和评估

    A review of ensemble learning and data augmentation models for class imbalanced problems: combination, implementation and evaluation. (arXiv:2304.02858v1 [cs.LG])

    [http://arxiv.org/abs/2304.02858](http://arxiv.org/abs/2304.02858)

    本文研究了集成学习和数据增强方法的应用，针对类别不平衡问题，通过计算评估，找到了最有效的组合。

    

    分类问题中的类别不平衡（CI）是指属于一个类的观测值数量低于其他类的数量。集成学习结合数据增强方法已被广泛应用于解决类别不平衡问题。在过去的十年里，一些策略已经被应用于增强集成学习和数据增强方法，同时还开发了一些新方法，如生成对抗网络（GAN）。本文对用于解决基准CI问题的数据增强和集成学习方法进行计算评估。我们提出了一个评估CI问题的10个数据增强方法和10个集成学习方法的通用框架。我们的目标是识别提高分类效果最有效的组合。

    Class imbalance (CI) in classification problems arises when the number of observations belonging to one class is lower than the other classes. Ensemble learning that combines multiple models to obtain a robust model has been prominently used with data augmentation methods to address class imbalance problems. In the last decade, a number of strategies have been added to enhance ensemble learning and data augmentation methods, along with new methods such as generative adversarial networks (GANs). A combination of these has been applied in many studies, but the true rank of different combinations would require a computational review. In this paper, we present a computational review to evaluate data augmentation and ensemble learning methods used to address prominent benchmark CI problems. We propose a general framework that evaluates 10 data augmentation and 10 ensemble learning methods for CI problems. Our objective was to identify the most effective combination for improving classificat
    
[^107]: 基于徒弟学习的面向主题的文本到图像生成器

    Subject-driven Text-to-Image Generation via Apprenticeship Learning. (arXiv:2304.00186v1 [cs.CV])

    [http://arxiv.org/abs/2304.00186](http://arxiv.org/abs/2304.00186)

    该论文提出了一种基于徒弟学习的面向主题的文本到图像生成器SuTI，能够通过将大量基于主题的专家模型的数据输入徒弟模型，学习并推断出新主题的最佳专家模型，从而生成高品质的自定义图像，且速度比传统方法更快。

    

    最近的文本到图像生成模型（如DreamBooth）在通过针对目标主题微调“专家模型”，生成高度自定义的图像方面取得了显著进展。然而，这个过程很昂贵，因为每个主题都必须学习一个新的专家模型。在本文中，我们提出了一个替代主题特定微调的面向主题的文本到图像生成器SuTI。给定一个新主题的少量演示，SuTI可以即时生成不同场景中主题的新版本，而无需进行任何主题特定的优化。SuTI由“徒弟学习”驱动，其中从大量基于主题的专家模型生成的数据中学习单个的徒弟模型。具体而言，我们从互联网挖掘了数百万个图像簇，每个图像簇都聚焦于一个特定的视觉主题。我们采用这些簇来训练大量专门针对不同视觉主题的专家模型。徒弟模型通过推断基于其文本描述的新主题的最佳专家模型并生成图像来学习。我们在各种基准数据集上展示了SuTI的有效性，表明它可以生成高品质的不同主题的图像，同时比基于微调的方法快得多。

    Recent text-to-image generation models like DreamBooth have made remarkable progress in generating highly customized images of a target subject, by fine-tuning an ``expert model'' for a given subject from a few examples. However, this process is expensive, since a new expert model must be learned for each subject. In this paper, we present SuTI, a Subject-driven Text-to-Image generator that replaces subject-specific fine tuning with \emph{in-context} learning. Given a few demonstrations of a new subject, SuTI can instantly generate novel renditions of the subject in different scenes, without any subject-specific optimization. SuTI is powered by {\em apprenticeship learning}, where a single apprentice model is learned from data generated by massive amount of subject-specific expert models. Specifically, we mine millions of image clusters from the Internet, each centered around a specific visual subject. We adopt these clusters to train massive amount of expert models specialized on diff
    
[^108]: 分析人工通用智能的语境缺陷

    Analyzing the Contextual Shortcomings of Artificial General Intelligence. (arXiv:2304.00002v1 [cs.AI])

    [http://arxiv.org/abs/2304.00002](http://arxiv.org/abs/2304.00002)

    论文讨论了人工智能(AI)专家对于人工通用智能(AGI)的决策所需技能并不了解的问题。 虽然当前的机器可以模拟特定的人类属性，但 AGI 是在这样的前提下开发出来的：这可以很容易地扩展到一般智能水平。这会分散当前研究的注意力，远离相关问题。

    

    即使是最尖端的人工通用智能(AGI)项目，人与机器之间的差异也十分明显。尽管这种差异本质上将每个人的能力划分开来，但人类级别的智能(HLI)已经是AGI几十年来的目标。本文反对图灵测试的二元论，即将其作为潜在智能机器的基础和原始建立的意图。它讨论了AI专家如何误解模仿游戏作为对计算机系统进行拟人化的手段，并断言HLI是一个转移注意力、使当前研究远离相关问题的错误方向。尽管对AGI应用的潜在设计进行了广泛研究，但却很少考虑这样一个系统如何以类似人类的水平访问和摄取数据。尽管当前的机器可能模拟特定的人类属性，但AGI是在这样的前提下开发出来的：这可以很容易地扩展到一般智能水平。

    Even in the most cutting-edge Artificial General Intelligence (AGI) endeavors, the disparity between humans and artificial systems is extremely apparent. Although this difference fundamentally divides the capabilities of each, human-level intelligence (HLI) has remained the aim of AGI for decades. This paper opposes the binarity of the Turing Test, the foundation of this intention and original establishment of a potentially intelligent machine. It discusses how AI experts misinterpreted the Imitation Game as a means to anthropomorphize computer systems and asserts that HLI is a red herring that distracts current research from relevant problems. Despite the extensive research on the potential design of an AGI application, there has been little consideration of how such a system will access and ingest data at a human-like level. Although current machines may emulate specific human attributes, AGI is developed under the pretense that this can be easily scaled up to a general intelligence 
    
[^109]: 使用Tube MPC引导的数据增强，高效学习鲁棒性的自适应策略的深度学习（arXiv:2303.15688v1 [cs.RO]）

    Efficient Deep Learning of Robust, Adaptive Policies using Tube MPC-Guided Data Augmentation. (arXiv:2303.15688v1 [cs.RO])

    [http://arxiv.org/abs/2303.15688](http://arxiv.org/abs/2303.15688)

    本论文提出了一种高效的深度学习算法，可以学习具有鲁棒性和自适应能力的策略，通过引导数据增强，使用修改后的IL过程，并在学习适应性位置和姿态控制策略方面进行应用。

    

    在具有挑战性的非结构化环境中部署敏捷自主系统需要适应能力和对不确定性的鲁棒性。现有的鲁棒和自适应控制器，如基于MPC的控制器，可以在在线运行计算量庞大的情况下实现令人印象深刻的性能。出现了有效地从MPC学习鲁棒且可在机载设备上部署的策略的策略，但它们仍然缺乏基本适应能力。在这项工作中，我们扩展了一种现有的高效IL算法，用于鲁棒性策略从MPC学习，具有学习适应具有挑战性模型/环境不确定性的策略的能力。我们方法的关键思想是通过在学习的低维模型/环境表示上对策略进行调整，从而修改IL过程，这可以在在线状态下高效地估计。我们将我们的方法定制为学习自适应位置和姿态控制策略以在具有挑战性的干扰下跟踪轨迹。

    The deployment of agile autonomous systems in challenging, unstructured environments requires adaptation capabilities and robustness to uncertainties. Existing robust and adaptive controllers, such as the ones based on MPC, can achieve impressive performance at the cost of heavy online onboard computations. Strategies that efficiently learn robust and onboard-deployable policies from MPC have emerged, but they still lack fundamental adaptation capabilities. In this work, we extend an existing efficient IL algorithm for robust policy learning from MPC with the ability to learn policies that adapt to challenging model/environment uncertainties. The key idea of our approach consists in modifying the IL procedure by conditioning the policy on a learned lower-dimensional model/environment representation that can be efficiently estimated online. We tailor our approach to the task of learning an adaptive position and attitude control policy to track trajectories under challenging disturbances
    
[^110]: 行为健康个性化介入的政策优化

    Policy Optimization for Personalized Interventions in Behavioral Health. (arXiv:2303.12206v1 [cs.LG])

    [http://arxiv.org/abs/2303.12206](http://arxiv.org/abs/2303.12206)

    研究如何通过数字平台传递的行为健康介入最大化健康结果和治疗成本，提出了一个名为DecompPI的新算法，从离线数据进行预测任务，减轻了在线实验的需要，并在理论上证明了该算法的可扩展性和渐近收敛性。

    

    问题定义：通过数字平台传递的行为健康介入，通过教育，激励，提醒和外展，有望显着改善健康结果。我们研究了在介入具有成本和能力限制的情况下，优化患者个性化介入以最大化某种长期结果的问题。方法/结果：本文提供了一种无模型方法来解决这个问题。我们发现，来自增强学习文献的通用无模型方法对于医疗应用来说过于数据密集，而更简单的赌臂问题方法取得了进展，但忽略了长期患者动态。我们提出了一种新算法，称为DecompPI，它近似于一步政策迭代。实现DecompPI只需从离线数据进行预测任务，减轻了在线实验的需要。在理论上，我们展示了在一种自然的结构假设下，DecompPI可以获得算法复杂度的渐近收敛性，同时保持一个可扩展的模型.

    Problem definition: Behavioral health interventions, delivered through digital platforms, have the potential to significantly improve health outcomes, through education, motivation, reminders, and outreach. We study the problem of optimizing personalized interventions for patients to maximize some long-term outcome, in a setting where interventions are costly and capacity-constrained.  Methodology/results: This paper provides a model-free approach to solving this problem. We find that generic model-free approaches from the reinforcement learning literature are too data intensive for healthcare applications, while simpler bandit approaches make progress at the expense of ignoring long-term patient dynamics. We present a new algorithm we dub DecompPI that approximates one step of policy iteration. Implementing DecompPI simply consists of a prediction task from offline data, alleviating the need for online experimentation. Theoretically, we show that under a natural set of structural assu
    
[^111]: IFAN：面向人类和NLP模型的可解释性交互框架

    IFAN: An Explainability-Focused Interaction Framework for Humans and NLP Models. (arXiv:2303.03124v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2303.03124](http://arxiv.org/abs/2303.03124)

    IFAN是一个面向人类和NLP模型的可解释性交互框架，通过用户的实时反馈和适配器层的对齐，有效地减轻了偏见的仇恨言论分类器。

    

    可解释性和人类监督是将复杂NLP模型应用于实际应用的基本支柱。然而，应用解释性和人机交互方法需要技术熟练。尽管存在用于模型理解和分析的工具包，但集成人类反馈的选项仍然有限。我们提出了IFAN，一种用于与NLP模型进行实时基于解释的交互的框架。通过IFAN的界面，用户可以对选择的模型解释提供反馈，然后通过适配器层将其与人类的理性进行对齐。我们展示了该系统在最小影响性能的情况下，对减轻偏见的仇恨言论分类器十分有效。IFAN还提供了一个可视化的管理系统和API，用于管理模型（和数据集）以及控制访问权限。演示地址：https://ifan.ml。

    Interpretability and human oversight are fundamental pillars of deploying complex NLP models into real-world applications. However, applying explainability and human-in-the-loop methods requires technical proficiency. Despite existing toolkits for model understanding and analysis, options to integrate human feedback are still limited. We propose IFAN, a framework for real-time explanation-based interaction with NLP models. Through IFAN's interface, users can provide feedback to selected model explanations, which is then integrated through adapter layers to align the model with human rationale. We show the system to be effective in debiasing a hate speech classifier with minimal impact on performance. IFAN also offers a visual admin system and API to manage models (and datasets) as well as control access rights. A demo is live at https://ifan.ml.
    
[^112]: EvoPrompting: 适用于代码级神经架构搜索的语言模型

    EvoPrompting: Language Models for Code-Level Neural Architecture Search. (arXiv:2302.14838v1 [cs.NE] CROSS LISTED)

    [http://arxiv.org/abs/2302.14838](http://arxiv.org/abs/2302.14838)

    EvoPrompting利用语言模型作为自适应变异和交叉操作符来进行神经架构搜索，在MNIST-1D数据集和CLRS算法推理基准上都取得了比人类设计的架构更好的性能表现。

    

    鉴于语言模型（LM）在代码生成方面的最新成就，我们探索将LM作为进化神经架构搜索（NAS）算法的自适应变异和交叉操作符的使用。尽管NAS仍然过于困难，以至于仅仅通过提示就难以成功，但我们发现进化提示工程与软提示调整的组合，一种我们称之为EvoPrompting的方法，始终可以发现多样化且性能高的模型。我们首先证明EvoPrompting在MNIST-1D数据集上是有效的，其中EvoPrompting产生的卷积架构变体在准确率和模型大小方面均优于人类专家设计的架构和天真的少数先导提示。然后，我们将我们的方法应用于在CLRS算法推理基准上搜索图神经网络，其中EvoPrompting能够设计出比当前最先进的模型更好的新颖结构。

    Given the recent impressive accomplishments of language models (LMs) for code generation, we explore the use of LMs as adaptive mutation and crossover operators for an evolutionary neural architecture search (NAS) algorithm. While NAS still proves too difficult a task for LMs to succeed at solely through prompting, we find that the combination of evolutionary prompt engineering with soft prompt-tuning, a method we term EvoPrompting, consistently finds diverse and high performing models. We first demonstrate that EvoPrompting is effective on the computationally efficient MNIST-1D dataset, where EvoPrompting produces convolutional architecture variants that outperform both those designed by human experts and naive few-shot prompting in terms of accuracy and model size. We then apply our method to searching for graph neural networks on the CLRS Algorithmic Reasoning Benchmark, where EvoPrompting is able to design novel architectures that outperform current state-of-the-art models on 21 ou
    
[^113]: 构造数：如何建立一个图形？

    Construction numbers: How to build a graph?. (arXiv:2302.13186v2 [math.CO] UPDATED)

    [http://arxiv.org/abs/2302.13186](http://arxiv.org/abs/2302.13186)

    论文研究了计算偏序的线性扩展数量问题，并研究了由包含关系确定的图形的顶点和边的偏序，找到了路径、环、星形图、双星形图和完全图的构造序列数量，并提出了公式，同时研究了结构和应用。

    

    约50年前，斯坦利考虑了计算偏序的线性扩展数量问题。对于由包含关系确定的图形的顶点和边的偏序，我们称这样的线性扩展为图形的“构造序列”，因为每个边都遵循其两个端点。我们找到了路径、环、星形图、双星形图和完全图的此类序列数量。对于路径，我们认同斯坦利的想法（切线数），并得到了其他类型的公式。此外还研究了结构和应用。

    Counting the number of linear extensions of a partial order was considered by Stanley about 50 years ago. For the partial order on the vertices and edges of a graph determined by inclusion, we call such linear extensions {\it construction sequences} for the graph as each edge follows both of its endpoints. The number of such sequences for paths, cycles, stars, double-stars, and complete graphs is found. For paths, we agree with Stanley (the Tangent numbers) and get formulas for the other classes. Structure and applications are also studied.
    
[^114]: HUST轴承：一个实用的球轴承故障诊断数据集

    HUST bearing: a practical dataset for ball bearing fault diagnosis. (arXiv:2302.12533v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.12533](http://arxiv.org/abs/2302.12533)

    HUST轴承是一个实用的球轴承故障诊断数据集，其中包含90个带有6种故障类型（内部裂纹、外部裂纹、球体裂纹和它们的2种组合）的5种不同类型轴承的振动数据。研究者使用经典机器学习分类方法以及先进的非监督迁移学习算法对该数据集进行了评估，实验结果表明在分类任务上准确率可达到100%，在非监督迁移学习任务上准确率为60-80%。

    

    在这项工作中，我们介绍了一个名为HUST轴承的实用数据集，该数据集提供了一组不同球轴承的振动数据。该数据集包括5种类型轴承的90个原始振动数据，其中包括内部裂纹、外部裂纹、球体裂纹以及它们的2种组合在内的6种缺陷类型，以及3个工作条件下的采样率为51,200次/秒。我们在引入的数据集上建立了包络分析和阶次跟踪分析，以进行数据的初步评估。使用不同域中的特征，我们使用了多种经典的机器学习分类方法来识别数据集中的轴承故障。同时，我们还使用了典型的先进非监督迁移学习算法，以观察数据集各部分之间的知识可迁移性。在数据集上经过实验的方法在分类任务上获得了达到100%的不同准确率，并在非监督迁移学习任务上获得了60-80%的准确率。

    In this work, we introduce a practical dataset named HUST bearing, that provides a large set of vibration data on different ball bearings. This dataset contains 90 raw vibration data of 6 types of defects (inner crack, outer crack, ball crack, and their 2-combinations) on 5 types of bearing at 3 working conditions with the sample rate of 51,200 samples per second. We established the envelope analysis and order tracking analysis on the introduced dataset to allow an initial evaluation of the data. A number of classical machine learning classification methods are used to identify bearing faults of the dataset using features in different domains. The typical advanced unsupervised transfer learning algorithms also perform to observe the transferability of knowledge among parts of the dataset. The experimental results of examined methods on the dataset gain divergent accuracy up to 100% on classification task and 60-80% on unsupervised transfer learning task.
    
[^115]: 长时间尺度温度缩放

    Long Horizon Temperature Scaling. (arXiv:2302.03686v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.03686](http://arxiv.org/abs/2302.03686)

    提出了一种长时间尺度温度缩放（LHTS）方法，用于从温度缩放的联合分布中采样。LHTS可以优化样本的长时间尺度似然，并且在图像扩散模型和字符/语言自回归模型上展示了优势。

    

    温度缩放是一种调节模型分布锐度的常用技术。它广泛应用于采样可能的生成物和校准模型不确定性，甚至在许多大型语言模型的部署中作为可控参数。然而，自回归模型依赖于贪婪地优化下一个标记的短视温度缩放。为了解决这个问题，我们提出了一种新颖的方法，即长时间尺度温度缩放（LHTS），用于从温度缩放的联合分布中采样。LHTS与所有基于似然的模型兼容，并优化样本的长时间尺度似然。我们推导了一个温度相关的LHTS目标，并展示了在一系列温度上微调模型可以产生一个具有可控长时间尺度温度参数的单一模型。我们在图像扩散模型和字符/语言自回归模型上进行了LHTS实验，证明了相比于短视温度缩放的优势。

    Temperature scaling is a popular technique for tuning the sharpness of a model distribution. It is used extensively for sampling likely generations and calibrating model uncertainty, and even features as a controllable parameter to many large language models in deployment. However, autoregressive models rely on myopic temperature scaling that greedily optimizes the next token. To address this, we propose Long Horizon Temperature Scaling (LHTS), a novel approach for sampling from temperature-scaled joint distributions. LHTS is compatible with all likelihood-based models, and optimizes for the long horizon likelihood of samples. We derive a temperature-dependent LHTS objective, and show that finetuning a model on a range of temperatures produces a single model capable of generation with a controllable long horizon temperature parameter. We experiment with LHTS on image diffusion models and character/language autoregressive models, demonstrating advantages over myopic temperature scaling 
    
[^116]: 领域无关的分子生成与自我反馈

    Domain-Agnostic Molecular Generation with Self-feedback. (arXiv:2301.11259v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.11259](http://arxiv.org/abs/2301.11259)

    MolGen是一个专注于分子生成的预训练语言模型，使用了领域无关的分子前缀调整和自我反馈的范式，实现了化学有效性、多样性、新颖性和复杂性的突破，在分子生成领域表现出了出色的性能。

    

    分子的生成已经受到极大的关注，其革新了科学家设计分子结构的方式，并为化学和药物设计提供了宝贵的支持。然而，尽管在分子生成中使用语言模型具有潜力，但它们面临着许多挑战，比如生成语法或化学存在缺陷的分子，狭窄的领域专注以及由于缺乏注释数据或外部分子数据库而限制了生成多样性和可行性。因此，我们引入了MolGen，它是一个专门用于分子生成的预训练分子语言模型。MolGen通过重构一亿多个分子SELFIES获得了固有的结构和语法概念，并通过领域无关的分子前缀调整促进了不同领域之间的知识传递。此外，我们提出了一种自我反馈范式，启发预训练模型与最终下游目标对齐，有助于更稳健和高效的分子生成。我们在基准数据集上的实验表明，MolGen在化学有效性，多样性，新颖性和复杂性方面优于现有技术。

    The generation of molecules with desired properties has gained tremendous popularity, revolutionizing the way scientists design molecular structures and providing valuable support for chemical and drug design. However, despite the potential of language models in molecule generation, they face numerous challenges such as the generation of syntactically or chemically flawed molecules, narrow domain focus, and limitations in creating diverse and directionally feasible molecules due to a dearth of annotated data or external molecular databases. To this end, we introduce MolGen, a pre-trained molecular language model tailored specifically for molecule generation. MolGen acquires intrinsic structural and grammatical insights by reconstructing over 100 million molecular SELFIES, while facilitating knowledge transfer between different domains through domain-agnostic molecular prefix tuning. Moreover, we present a self-feedback paradigm that inspires the pre-trained model to align with the ulti
    
[^117]: 基于对抗生成网络的短SSVEP数据扩展框架

    Short-length SSVEP data extension by a novel generative adversarial networks based framework. (arXiv:2301.05599v3 [q-bio.NC] UPDATED)

    [http://arxiv.org/abs/2301.05599](http://arxiv.org/abs/2301.05599)

    本文提出了一种基于GAN的端到端信号转化网络TEGAN，可以将短SSVEP信号转换成长的人工SSVEP信号，并显著提高BCI系统的效率和准确性。

    

    基于SSVEP的脑机接口因其高信息传输速率和目标数量可用性而受到广泛关注。然而，频率识别方法的性能在很大程度上取决于用户校准数据的数量和数据长度，这限制了它在实际应用中的部署。最近，基于生成对抗网络（GANs）的数据生成方法已被广泛采用来创建合成的脑电数据，有望解决这些问题。本文提出了一种基于GANs的端到端信号转化网络TEGAN，用于数据长度扩展。TEGAN可以将短SSVEP信号转换成长的人工SSVEP信号。通过将一个新颖的U型生成器架构和一个辅助分类器加入到网络结构中，TEGAN可以在合成数据中产生有条件的特征。此外，我们实现并比较了两种最先进的频率识别方法，以评估TEGAN生成数据的有效性。实验结果表明，所提出的TEGAN方法优于传统的线性插值方法和最先进的基于深度学习的方法。所提出的TEGAN方法可以显著提高BCI系统的效率，减少所需的校准时间并改善分类的准确性。

    Steady-state visual evoked potentials (SSVEPs) based brain-computer interface (BCI) has received considerable attention due to its high information transfer rate (ITR) and available quantity of targets. However, the performance of frequency identification methods heavily hinges on the amount of user calibration data and data length, which hinders the deployment in real-world applications. Recently, generative adversarial networks (GANs)-based data generation methods have been widely adopted to create synthetic electroencephalography (EEG) data, holds promise to address these issues. In this paper, we proposed a GAN-based end-to-end signal transformation network for data length extension, termed as TEGAN. TEGAN transforms short-length SSVEP signals into long-length artificial SSVEP signals. By incorporating a novel U-Net generator architecture and an auxiliary classifier into the network architecture, the TEGAN could produce conditioned features in the synthetic data. Additionally, we i
    
[^118]: 默默杀手: 一种隐蔽的、无标签的、黑盒子后门攻击

    Silent Killer: A Stealthy, Clean-Label, Black-Box Backdoor Attack. (arXiv:2301.02615v2 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2301.02615](http://arxiv.org/abs/2301.02615)

    默默杀手是一种隐蔽的、无标签的、黑盒子后门攻击，它使用了隐蔽的毒物和触发器，在无标签攻击中使用通用对抗扰动作为触发器，通过渐变对齐来提高成功率，并在MNIST、CIFAR10和ImageNet数据集上取得了最新的成果。

    

    后门污染攻击对神经网络构成了众所周知的风险。然而，大多数研究都集中在宽松的威胁模型上。我们引入了一种名为默默杀手的新型攻击，在无标签的黑盒子环境中运行，使用隐蔽的毒物和触发器，并且胜过现有的方法。我们研究了在无标签攻击中使用通用对抗扰动作为触发器的方法，在毒标签设置下的成功案例之后。我们分析了一个天真的适应方法的成功情况，并发现需要渐变对齐以确保高成功率。我们对MNIST、CIFAR10和一个缩小版的ImageNet进行了彻底的实验，并取得了最新的成果。

    Backdoor poisoning attacks pose a well-known risk to neural networks. However, most studies have focused on lenient threat models. We introduce Silent Killer, a novel attack that operates in clean-label, black-box settings, uses a stealthy poison and trigger and outperforms existing methods. We investigate the use of universal adversarial perturbations as triggers in clean-label attacks, following the success of such approaches under poison-label settings. We analyze the success of a naive adaptation and find that gradient alignment for crafting the poison is required to ensure high success rates. We conduct thorough experiments on MNIST, CIFAR10, and a reduced version of ImageNet and achieve state-of-the-art results.
    
[^119]: 在约束规划求解器内学习通用的值选择启发式

    Learning a Generic Value-Selection Heuristic Inside a Constraint Programming Solver. (arXiv:2301.01913v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2301.01913](http://arxiv.org/abs/2301.01913)

    本论文提出了一种通用学习过程，用于在约束规划求解器内获取一个值选择启发式方法，以解决当前通用值选择启发式方法较为稀缺的问题。

    

    约束规划被认为是解决组合问题的一种高效方法。求解器中的重要设计选择是分支启发式，它们旨在在最短的时间内寻找最佳解决方案。然而，开发这些启发式需要耗费大量时间，并需要问题特定的专业知识。这一观察结果激发了许多使用机器学习自动学习高效启发式的努力，而无需专家干预。据我们所知，这仍然是一个开放的研究问题。尽管文献中有几种通用的变量选择启发式方法，但对于通用的值选择启发式方法的选择却较少。在本文中，我们提出通过引入一种通用的学习过程来解决这个问题，该过程可以用于在约束规划求解器内获得一个值选择启发式方法。这得益于深度Q学习算法和一个...

    Constraint programming is known for being an efficient approach for solving combinatorial problems. Important design choices in a solver are the branching heuristics, which are designed to lead the search to the best solutions in a minimum amount of time. However, developing these heuristics is a time-consuming process that requires problem-specific expertise. This observation has motivated many efforts to use machine learning to automatically learn efficient heuristics without expert intervention. To the best of our knowledge, it is still an open research question. Although several generic variable-selection heuristics are available in the literature, the options for a generic value-selection heuristic are more scarce. In this paper, we propose to tackle this issue by introducing a generic learning procedure that can be used to obtain a value-selection heuristic inside a constraint programming solver. This has been achieved thanks to the combination of a deep Q-learning algorithm, a t
    
[^120]: 通过寻找基于任务的平坦区域来改进多任务学习

    Improving Multi-task Learning via Seeking Task-based Flat Regions. (arXiv:2211.13723v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.13723](http://arxiv.org/abs/2211.13723)

    通过寻找基于任务的平坦区域，可以改进多任务学习并提高模型性能，但需要正确使用正则化技术以避免次优解。

    

    多任务学习（MTL）是一种广泛使用且强大的学习范式，用于训练深度神经网络，可以通过单个骨干学习多个目标。与单独训练任务相比，MTL显着降低了计算成本，提高了数据效率，并通过利用任务之间的知识来潜在地提高模型性能。因此，它已经被应用于各种应用领域，从计算机视觉到自然语言处理和语音识别。其中，MTL的一个新兴研究方向集中在操纵任务梯度以推导出对所有任务有益的最终梯度下降方向。尽管在许多基准测试上取得了令人印象深刻的结果，但是在实际问题上直接应用这些方法而不使用适当的正则化技术可能会导致次优解。特别是，标准训练在训练数据上最小化经验损失，很容易遭受过拟合问题。

    Multi-Task Learning (MTL) is a widely-used and powerful learning paradigm for training deep neural networks that allows learning more than one objective by a single backbone. Compared to training tasks separately, MTL significantly reduces computational costs, improves data efficiency, and potentially enhances model performance by leveraging knowledge across tasks. Hence, it has been adopted in a variety of applications, ranging from computer vision to natural language processing and speech recognition. Among them, there is an emerging line of work in MTL that focuses on manipulating the task gradient to derive an ultimate gradient descent direction to benefit all tasks. Despite achieving impressive results on many benchmarks, directly applying these approaches without using appropriate regularization techniques might lead to suboptimal solutions on real-world problems. In particular, standard training that minimizes the empirical loss on the training data can easily suffer from overfi
    
[^121]: L-MAE：遮罩自编码器用于语义分割数据增强

    L-MAE: Masked Autoencoders are Semantic Segmentation Datasets Augmenter. (arXiv:2211.11242v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2211.11242](http://arxiv.org/abs/2211.11242)

    本文提出了一种新的标签遮罩自编码器（L-MAE）方法，用于生成完整的语义分割标签。该方法首次将遮罩自编码器应用于下游任务，并采用了标签和图像的融合策略。这种方法能够解决大型模型和专业领域数据集中存在的数据标注不准确问题。

    

    在生成语义分割数据集方面，特别是在大型模型或专业领域（如医学影像或遥感）的背景下，一直以来都是耗时费力的。具体来说，大型模型需要大量的数据，而专业领域的数据集通常需要领域专家的参与。这两种情况都容易导致数据标注的不准确，这可能会严重影响训练模型的最终性能。本文提出了一种简单有效的像素级标签完成方法，即L-MAE（标签遮罩自编码器），它充分利用标签中的现有信息生成完整的标签。该文首次将遮罩自编码器应用于下游任务。具体来说，L-MAE采用了堆叠标签和相应图像的融合策略，即融合映射。此外，由于遮罩融合映射时会丢失一些图像信息，因此该方法直接进行重建。

    Generating semantic segmentation datasets has consistently been laborious and time-consuming, particularly in the context of large models or specialized domains(i.e. Medical Imaging or Remote Sensing). Specifically, large models necessitate a substantial volume of data, while datasets in professional domains frequently require the involvement of domain experts. Both scenarios are susceptible to inaccurate data labeling, which can significantly affect the ultimate performance of the trained model. This paper proposes a simple and effective label pixel-level completion method, \textbf{Label Mask AutoEncoder} (L-MAE), which fully uses the existing information in the label to generate the complete label. The proposed model are the first to apply the Mask Auto-Encoder to downstream tasks. In detail, L-MAE adopts the fusion strategy that stacks the label and the corresponding image, namely fuse map. Moreover, since some of the image information is lost when masking the fuse map, direct recon
    
[^122]: 语言模型的整体评估

    Holistic Evaluation of Language Models. (arXiv:2211.09110v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2211.09110](http://arxiv.org/abs/2211.09110)

    我们提出了语言模型的整体评估（HELM），通过对潜在场景和度量进行分类并采用多度量方法，提高语言模型的透明度和可信度。

    

    语言模型（LMs）正在成为几乎所有主要语言技术的基础，但它们的能力、限制和风险并不被很好地理解。我们提出了语言模型的整体评估（HELM），以提高语言模型的透明度。首先，我们对感兴趣的潜在场景（即用例）和度量（即期望）的广阔空间进行分类。然后，我们选择了一个宽泛的子集，基于覆盖范围和可行性，注意到了缺失或未充分代表的内容（例如，为被忽视的英语方言进行问答，用于可信度的度量）。其次，我们采用多度量方法：我们分别针对每个核心场景测量了准确度、校准度、鲁棒性、公平性、偏见、有毒性和效率这7个度量指标（在87.5%的时间内）。这确保了准确度以外的度量不会被忽视，并且权衡清晰。我们还进行了7个针对性评估，基于26个针对性场景，以分析特定场景下的性能。

    Language models (LMs) are becoming the foundation for almost all major language technologies, but their capabilities, limitations, and risks are not well understood. We present Holistic Evaluation of Language Models (HELM) to improve the transparency of language models. First, we taxonomize the vast space of potential scenarios (i.e. use cases) and metrics (i.e. desiderata) that are of interest for LMs. Then we select a broad subset based on coverage and feasibility, noting what's missing or underrepresented (e.g. question answering for neglected English dialects, metrics for trustworthiness). Second, we adopt a multi-metric approach: We measure 7 metrics (accuracy, calibration, robustness, fairness, bias, toxicity, and efficiency) for each of 16 core scenarios when possible (87.5% of the time). This ensures metrics beyond accuracy don't fall to the wayside, and that trade-offs are clearly exposed. We also perform 7 targeted evaluations, based on 26 targeted scenarios, to analyze speci
    
[^123]: 胸部X光深度学习基础模型中的偏倚风险

    Risk of Bias in Chest Radiography Deep Learning Foundation Models. (arXiv:2209.02965v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2209.02965](http://arxiv.org/abs/2209.02965)

    该研究分析了一种最近发布的胸部X光基础模型中的偏倚风险，并发现在生物性别和种族之间存在亚组性能差距。

    

    目的：分析最近发布的胸部X光基础模型是否存在偏倚，可能导致在生物性别和种族之间存在亚组性能差距。材料和方法：本回顾性研究使用CheXpert数据集中自2002年10月至2017年7月期间收集的42,884名患者（年龄平均为63岁，标准偏差为17岁；男性23,623人，女性19,261人）的127,118张胸部X光。使用降维方法和两样本Kolmogorov-Smirnov检验检测胸部X光基础模型和基础深度学习模型生成的特征中的分布偏差，以确定是否存在偏倚。然后进行全面的疾病检测性能分析，将特征中的任何偏倚与患者亚组的分类性能差异相关联。

    Purpose: To analyze a recently published chest radiography foundation model for the presence of biases that could lead to subgroup performance disparities across biological sex and race.  Materials and Methods: This retrospective study used 127,118 chest radiographs from 42,884 patients (mean age, 63 [SD] 17 years; 23,623 male, 19,261 female) from the CheXpert dataset collected between October 2002 and July 2017. To determine the presence of bias in features generated by a chest radiography foundation model and baseline deep learning model, dimensionality reduction methods together with two-sample Kolmogorov-Smirnov tests were used to detect distribution shifts across sex and race. A comprehensive disease detection performance analysis was then performed to associate any biases in the features to specific disparities in classification performance across patient subgroups.  Results: Ten out of twelve pairwise comparisons across biological sex and race showed statistically significant di
    
[^124]: 通过分散社会制裁的出现，分工的形成

    The emergence of division of labor through decentralized social sanctioning. (arXiv:2208.05568v4 [cs.MA] UPDATED)

    [http://arxiv.org/abs/2208.05568](http://arxiv.org/abs/2208.05568)

    本研究通过引入社会规范模型，展示了分散社会制裁的出现模式能够解决以自利为导向的终身学习个体中的分工问题。

    

    人类生态成功依赖于我们的独特能力，即灵活自组织成合作社会群体，其中最成功的群体采用了大量的专业化和分工。与大多数其他动物不同，人类通过一生的试错中学习自己要扮演的角色。然而，当某些关键角色比其他角色更具吸引力，并且个体是自利的时，就会出现社会困境：每个个体都希望其他人扮演关键但无报酬的角色，这样他们可以自由选择一个报酬更高的角色。但是，如果每个人都这样行事，且一个关键角色缺乏填补，就会发生灾难。在这种情况下，学习最佳角色分配可能是不可能的。因此，一个基本问题是：如何在一群以自利为导向的终身学习个体中形成分工呢？在这里，我们展示了通过引入社会规范模型（我们将其视为分散社会制裁的出现模式）可以解决这个问题。

    Human ecological success relies on our characteristic ability to flexibly self-organize into cooperative social groups, the most successful of which employ substantial specialization and division of labor. Unlike most other animals, humans learn by trial and error during their lives what role to take on. However, when some critical roles are more attractive than others, and individuals are self-interested, then there is a social dilemma: each individual would prefer others take on the critical-but-unremunerative roles so they may remain free to take one that pays better. But disaster occurs if all act thusly and a critical role goes unfilled. In such situations learning an optimum role distribution may not be possible. Consequently, a fundamental question is: how can division of labor emerge in groups of self-interested lifetime-learning individuals? Here we show that by introducing a model of social norms, which we regard as emerging patterns of decentralized social sanctioning, it be
    
[^125]: NSGA-II的非支配排序遗传算法II的近似保证

    Approximation Guarantees for the Non-Dominated Sorting Genetic Algorithm II (NSGA-II). (arXiv:2203.02693v3 [cs.NE] UPDATED)

    [http://arxiv.org/abs/2203.02693](http://arxiv.org/abs/2203.02693)

    最近的研究发现，当种群规模足够大时，NSGA-II可以高效地计算完整的帕累托前沿。但当种群规模较小时，NSGA-II对帕累托前沿的逼近效果较差，特别是存在较大的间隙。这篇论文通过数学证明，揭示了这一问题的原因，并提出了两种改进方法。

    

    最近的理论研究表明，当种群规模足够大时，NSGA-II可以高效地计算出完整的帕累托前沿。本文研究了当种群规模较小时，它对帕累托前沿的逼近程度。对于OneMinMax基准测试，我们指出在某些情况下，父代和后代很好地覆盖了帕累托前沿，但是下一代的帕累托前沿存在较大的间隙。我们的数学证明表明，造成这种不希望的行为的原因是NSGA-II在选择阶段只计算了拥挤距离一次，然后删除具有最小拥挤距离的个体，而没有考虑到删除会增加某些个体的拥挤距离。然后，我们分析了两种不容易出现这个问题的变体。对于在每次删除后更新拥挤距离的NSGA-II（Kukkonen and Deb（2006））和稳态NSGA-II（Nebro and Durillo（2009）），我们证明了帕累托前沿的间隙永远不会超过一个很小的值。

    Recent theoretical works have shown that the NSGA-II efficiently computes the full Pareto front when the population size is large enough. In this work, we study how well it approximates the Pareto front when the population size is smaller.  For the OneMinMax benchmark, we point out situations in which the parents and offspring cover well the Pareto front, but the next population has large gaps on the Pareto front. Our mathematical proofs suggest as reason for this undesirable behavior that the NSGA-II in the selection stage computes the crowding distance once and then removes individuals with smallest crowding distance without considering that a removal increases the crowding distance of some individuals.  We then analyze two variants not prone to this problem. For the NSGA-II that updates the crowding distance after each removal (Kukkonen and Deb (2006)) and the steady-state NSGA-II (Nebro and Durillo (2009)), we prove that the gaps in the Pareto front are never more than a small cons
    
[^126]: DEBOSH: 深度贝叶斯形状优化

    DEBOSH: Deep Bayesian Shape Optimization. (arXiv:2109.13337v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2109.13337](http://arxiv.org/abs/2109.13337)

    本论文提出了一种基于不确定性的方法，针对形状优化，在利用图神经网络预测工业设计性能时，解决了形状偏离训练集时预测不可靠的问题，并通过有效的贝叶斯优化提高了结果形状的质量。

    

    图神经网络（GNNs）可以快速准确地预测工业设计的性能，并用于有效优化其形状。然而，为了充分探索形状空间，通常需要考虑与训练集明显偏离的形状。对于这些情况，GNN的预测变得不可靠，但这通常被忽视。针对依赖高斯过程的优化技术，贝叶斯优化（BO）通过利用其评估自身精度的能力来解决这个问题。然而，当使用神经网络时，估计其不确定性的标准方法往往会导致计算量大和模型准确性降低。因此，我们提出了一种针对形状优化的新颖基于不确定性的方法。它实现了有效的BO，并提高了结果形状的质量，超过了最先进的方法。

    Graph Neural Networks (GNNs) can predict the performance of an industrial design quickly and accurately and be used to optimize its shape effectively. However, to fully explore the shape space, one must often consider shapes deviating significantly from the training set. For these, GNN predictions become unreliable, something that is often ignored. For optimization techniques relying on Gaussian Processes, Bayesian Optimization (BO) addresses this issue by exploiting their ability to assess their own accuracy. Unfortunately, this is harder to do when using neural networks because standard approaches to estimating their uncertainty can entail high computational loads and reduced model accuracy. Hence, we propose a novel uncertainty-based method tailored to shape optimization. It enables effective BO and increases the quality of the resulting shapes beyond that of state-of-the-art approaches.
    
[^127]: FUTURE-AI:医学影像中值得信赖的人工智能的指导原则和共识建议

    FUTURE-AI: Guiding Principles and Consensus Recommendations for Trustworthy Artificial Intelligence in Medical Imaging. (arXiv:2109.09658v4 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2109.09658](http://arxiv.org/abs/2109.09658)

    本论文介绍了一系列从经验、共识和最佳实践中提炼出的指导原则，旨在引领医学影像中值得信赖的人工智能的发展，提高信任、安全性和应用水平。

    

    近年来，人工智能和临床系统生成的大量数据的结合，推动了医学影像领域整个价值链上的成像人工智能解决方案的发展，包括图像重建、医学图像分割、基于图像的诊断和治疗规划。尽管医学影像中的人工智能取得了成功并有着巨大的潜力，但许多利益相关者担心成像人工智能解决方案的潜在风险和伦理问题，认为其复杂、不透明、难以理解、难以应用和难以在关键临床应用中建立信任。尽管存在这些担忧和风险，但目前尚没有具体的指导原则和最佳实践来引导未来医学影像中人工智能的发展以增加信任、安全性和采用。为了弥补这一空白，本文提出了从积累的经验、共识和最佳实践中精选出的指导原则。

    The recent advancements in artificial intelligence (AI) combined with the extensive amount of data generated by today's clinical systems, has led to the development of imaging AI solutions across the whole value chain of medical imaging, including image reconstruction, medical image segmentation, image-based diagnosis and treatment planning. Notwithstanding the successes and future potential of AI in medical imaging, many stakeholders are concerned of the potential risks and ethical implications of imaging AI solutions, which are perceived as complex, opaque, and difficult to comprehend, utilise, and trust in critical clinical applications. Despite these concerns and risks, there are currently no concrete guidelines and best practices for guiding future AI developments in medical imaging towards increased trust, safety and adoption. To bridge this gap, this paper introduces a careful selection of guiding principles drawn from the accumulated experiences, consensus, and best practices f
    
[^128]: 约束资源下神经模块专业化的动力学研究

    Dynamics of specialization in neural modules under resource constraints. (arXiv:2106.02626v2 [q-bio.NC] UPDATED)

    [http://arxiv.org/abs/2106.02626](http://arxiv.org/abs/2106.02626)

    本研究使用人工神经网络模拟实验，发现结构模块化并不一定能够确保功能专业化，在特定环境和资源限制下，才能够出现专业化现象。

    

    长期以来，人们一直认为大脑在结构和功能上高度模块化，但最近的证据使一些人对两种模块化的程度产生了怀疑。我们使用人工神经网络来测试结构模块化是否足以保证功能专业化，并发现一般情况下，并不一定成立，除非在极端水平上。然后，我们系统地测试了环境和网络的哪些特征会导致专业化的出现。我们使用了一个简单的玩具环境、任务和网络，以精确控制条件，并表明在这个设置中，几个不同的专业化度量指标给出了类似的结果。我们进一步发现，（1）专业化只能在环境中那些可以明确分离的特征存在的情况下出现，（2）专业化更容易在网络资源受到强烈限制的情况下出现，（3）这些发现在 qualitatively 上相似。

    It has long been believed that the brain is highly modular both in terms of structure and function, although recent evidence has led some to question the extent of both types of modularity. We used artificial neural networks to test the hypothesis that structural modularity is sufficient to guarantee functional specialization, and find that in general, this doesn't necessarily hold except at extreme levels. We then systematically tested which features of the environment and network do lead to the emergence of specialization. We used a simple toy environment, task and network, allowing us precise control, and show that in this setup, several distinct measures of specialization give qualitatively similar results. We further find that (1) specialization can only emerge in environments where features of that environment are meaningfully separable, (2) specialization preferentially emerges when the network is strongly resource-constrained, and (3) these findings are qualitatively similar ac
    
[^129]: iCORPP: 机器人上的交替常识推理与概率规划

    iCORPP: Interleaved Commonsense Reasoning and Probabilistic Planning on Robots. (arXiv:2004.08672v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2004.08672](http://arxiv.org/abs/2004.08672)

    iCORPP提出了一种新型算法，能够在机器人的决策过程中同时推理世界状态、动态和构建任务导向的控制器。

    

    在真实世界中，机器人的顺序决策是一个挑战，因为它要求机器人同时推理当前世界状态和动态，同时规划行动以完成复杂任务。一方面，声明性语言和推理算法能够良好地支持表示和处理常识知识。但是这些算法在规划未指定时间跨度下的最大累积奖励的行动时表现不佳。另一方面，概率规划框架（如马尔可夫决策过程（MDPs）和部分可观察到的MDPs（POMDPs））能够很好地支持在不确定性条件下规划实现长期目标。但是它们不能很好地表示或推理与行动无直接关联的知识。在本文中，我们提出了一种新颖的算法称为iCORPP，以同时估计当前世界状态，推理世界动态和构建任务导向的控制器。

    Robot sequential decision-making in the real world is a challenge because it requires the robots to simultaneously reason about the current world state and dynamics, while planning actions to accomplish complex tasks. On the one hand, declarative languages and reasoning algorithms well support representing and reasoning with commonsense knowledge. But these algorithms are not good at planning actions toward maximizing cumulative reward over a long, unspecified horizon. On the other hand, probabilistic planning frameworks, such as Markov decision processes (MDPs) and partially observable MDPs (POMDPs), well support planning to achieve long-term goals under uncertainty. But they are ill-equipped to represent or reason about knowledge that is not directly related to actions.  In this article, we present a novel algorithm, called iCORPP, to simultaneously estimate the current world state, reason about world dynamics, and construct task-oriented controllers. In this process, robot decision-
    

