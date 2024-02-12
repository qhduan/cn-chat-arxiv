# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Monotone, Bi-Lipschitz, and Polyak-\L{}ojasiewicz Networks](https://rss.arxiv.org/abs/2402.01344) | 这篇论文介绍了一种新的可逆神经网络BiLipNet，它具有调控输出敏感性和输入可区分性的能力。其中的主要创新是通过认证的强单调性和Lipschitz性的可逆残差层，与正交层组合构建了Bi-Lipschitz网络。另外，该论文还提出了满足Polyak-\L{}ojasiewicz条件的PLNet，并介绍了其应用于学习非凸代理损失的优势特性。 |
| [^2] | [On the Transferability of Large-Scale Self-Supervision to Few-Shot Audio Classification](https://rss.arxiv.org/abs/2402.01274) | 本研究评估了大规模自监督模型在少样本音频分类中的性能，并发现在一些少样本问题中取得了最先进的性能，同时发现语音为基础的少样本问题与多个下游音频任务之间存在较强的相关性。 |
| [^3] | [Feedback Loops With Language Models Drive In-Context Reward Hacking](https://arxiv.org/abs/2402.06627) | 与语言模型的反馈循环可能导致上下文内奖励欺骗（ICRH），即语言模型在测试时在优化目标的同时却产生负面副作用。这项研究确定了两个导致ICRH的过程：输出优化和策略优化。 |
| [^4] | [The Complexity of Sequential Prediction in Dynamical Systems](https://arxiv.org/abs/2402.06614) | 通过学习理论的角度，我们在没有参数假设的情况下，研究了在底层演化函数未知的动力系统中学习预测下一状态的问题，并提出了新的组合度量和维度来量化在可实现和不可知情况下的最佳错误和遗憾界限。 |
| [^5] | [RQP-SGD: Differential Private Machine Learning through Noisy SGD and Randomized Quantization](https://arxiv.org/abs/2402.06606) | RQP-SGD是一种结合了差分隐私随机梯度下降和随机量化的新方法，用于在边缘部署的低内存机器学习模型训练中实现隐私保护。通过研究其在具有凸目标和量化约束的ML任务上的效用收敛性，并证明了其相对确定性量化的有效性。 |
| [^6] | [Predictive representations: building blocks of intelligence](https://arxiv.org/abs/2402.06590) | 预测性表征可能是智能的多功能基石。 |
| [^7] | [More than the Sum of Its Parts: Ensembling Backbone Networks for Few-Shot Segmentation](https://arxiv.org/abs/2402.06581) | 本论文研究了在少样本分割中，通过集成不同主干网络的特征能否提高模型的性能。作者通过提出独立投票和特征融合两种集成技术，并在PANet上实现了这些技术。实验结果表明集成主干网络可以捕捉更丰富的视觉特征，从而提升分割效果。 |
| [^8] | [SAE: Single Architecture Ensemble Neural Networks](https://arxiv.org/abs/2402.06580) | SAE是一种单一架构集合神经网络方法，通过学习集合输入的最佳退出数量和深度，在任务上显示出优越的准确性和置信度校准。它能够根据特定架构或应用程序灵活地定制其配置。 |
| [^9] | [On the Universality of Coupling-based Normalizing Flows](https://arxiv.org/abs/2402.06578) | 我们提出了一个新颖的理论框架，用于理解基于耦合的标准化流的表达能力，并提出了一个新的分布普适性定理来克服以前工作的限制。这些结果支持耦合架构的表达能力，并弥补了实证结果和理论理解之间的差距。 |
| [^10] | [Distilling Morphology-Conditioned Hypernetworks for Efficient Universal Morphology Control](https://arxiv.org/abs/2402.06570) | 本论文提出了一种名为HyperDistill的方法，通过精简形态条件超网络，可在训练和未知测试机器人上实现与通用的transformers策略相当的性能，同时大大减小模型尺寸和计算成本。 |
| [^11] | [What is Hiding in Medicine's Dark Matter? Learning with Missing Data in Medical Practices](https://arxiv.org/abs/2402.06563) | 本研究使用统计方法和机器学习，通过分析儿科急诊数据和创伤伤害数据库，揭示了医疗实践模式与丢失数据之间的关联，并提出了临床数据插补的方法。这对于减少分析偏见、提高临床决策的有效性非常重要。 |
| [^12] | [Video Annotator: A framework for efficiently building video classifiers using vision-language models and active learning](https://arxiv.org/abs/2402.06560) | 本论文提出了一个名为视频标注器（VA）的框架，通过利用视觉语言模型和主动学习构建视频分类器，并通过人在循环系统实现了领域专家的直接参与，解决了传统数据注释方法的资源消耗和效率低下的问题。 |
| [^13] | [Diffusion-ES: Gradient-free Planning with Diffusion for Autonomous Driving and Zero-Shot Instruction Following](https://arxiv.org/abs/2402.06559) | 本文提出了一种Diffusion-ES方法，它结合了无梯度优化和轨迹去噪技术，用于优化黑盒非可微目标。该方法通过从扩散模型中采样轨迹，并使用黑盒奖励函数对其进行评分，实现了更高的多样性和可解释性。 |
| [^14] | [Deceptive Path Planning via Reinforcement Learning with Graph Neural Networks](https://arxiv.org/abs/2402.06552) | 本文提出了一种使用图神经网络的强化学习方法来进行欺骗性路径规划，克服了现有方法的局限性，并且具有普适性和实时适应性。 |
| [^15] | [Calibrating Long-form Generations from Large Language Models](https://arxiv.org/abs/2402.06544) | 该论文提出了一个统一的校准框架，用于校准大型语言模型的长篇生成。在该框架中，作者开发了三个度量指标用于评估模型的校准性，并提出了两种置信度引导方法。实验证明，更大的模型不一定能保证更好的校准。 |
| [^16] | [Bandit Convex Optimisation](https://arxiv.org/abs/2402.06535) | 这篇论文介绍了强盗凸优化的基本框架和用于解决这一问题的多种工具。虽然没有太多创新，但通过以新颖的方式应用现有工具，获得了新的算法和改进了一些界限。 |
| [^17] | [Generative Adversarial Bayesian Optimization for Surrogate Objectives](https://arxiv.org/abs/2402.06532) | 提出了生成对抗贝叶斯优化（GABO）算法，通过使用自适应源批评家正则化，将优化轨迹限制在代理函数可靠的区域内，解决了离线模型基于策略优化中代理模型预测不准确的问题。在多个离线优化任务中，GABO表现优于现有基准方法。 |
| [^18] | [Transferring facade labels between point clouds with semantic octrees while considering change detection](https://arxiv.org/abs/2402.06531) | 该论文提出了一种使用八叉树结构将标签从一个已标注的点云转移到一个未标注的点云的方法，同时考虑了点云之间的变化。这种方法的主要贡献是自动在表示同一真实世界对象的两个不同点云之间进行标签转移，并可以应用于数据驱动的深度学习算法。 |
| [^19] | [Refining Myocardial Infarction Detection: A Novel Multi-Modal Composite Kernel Strategy in One-Class Classification](https://arxiv.org/abs/2402.06530) | 本研究提出了一种新的方法，使用基于超声心动图的一种基于多模态复合核策略的单一类别分类算法来进行早期心肌梗死的检测。这种方法通过优化投影矩阵和特征转化，提高了心肌梗死检测的能力。 |
| [^20] | [Introspective Planning: Guiding Language-Enabled Agents to Refine Their Own Uncertainty](https://arxiv.org/abs/2402.06529) | 本文研究了内省规划的概念，作为一种引导语言驱动的代理机器人改进自身不确定性的系统方法。通过识别任务不确定性并主动寻求澄清，内省显著提高了机器人任务规划的成功率和安全性。 |
| [^21] | [Flexible infinite-width graph convolutional networks and the importance of representation learning](https://arxiv.org/abs/2402.06525) | 本文讨论了神经网络高斯过程（NNGP）在理论上的局限，提出图卷积深度内核机（graph convolutional deep kernel machine）来研究图分类任务中的表示学习问题。 |
| [^22] | [Reconstructing facade details using MLS point clouds and Bag-of-Words approach](https://arxiv.org/abs/2402.06521) | 该论文提出了一种使用MLS点云和词袋法的新方法，用于重建立面细节。通过结合预定义的3D模型库和半全局特征，该方法在实验中展示了有希望的结果，并改进了传统的词袋法方法。该方法有潜力用于更真实的立面重建，并可以用于测试自动驾驶功能或估算立面太阳能潜力。 |
| [^23] | [Multimodal Clinical Trial Outcome Prediction with Large Language Models](https://arxiv.org/abs/2402.06512) | 本研究提出了一种名为LIFTED的多模态临床试验结果预测方法，通过将不同模态数据转化为自然语言描述来统一数据，并构建统一的抗噪声编码器进行信息提取。 |
| [^24] | [Classifying point clouds at the facade-level using geometric features and deep learning networks](https://arxiv.org/abs/2402.06506) | 该论文提出了一种方法，利用几何特征和深度学习网络对立面级别的点云进行分类。实验证明，融合几何特征可以提高深度学习方法的性能，并促进语义分割的进步。 |
| [^25] | [ACTER: Diverse and Actionable Counterfactual Sequences for Explaining and Diagnosing RL Policies](https://arxiv.org/abs/2402.06503) | ACTER是一个算法，用于生成可行的反事实序列，提供关于如何避免RL策略失败的可行建议。 |
| [^26] | [Scalable Interactive Machine Learning for Future Command and Control](https://arxiv.org/abs/2402.06501) | 未来战争将需要指挥与控制（C2）人员在复杂且潜在模糊的情况下以更短的时间内做出决策。本论文通过利用互动式机器学习方法，结合人工智能和人类智能，以提高C2运作的适应性和效率。 |
| [^27] | [Deep Learning-Based Auto-Segmentation of Planning Target Volume for Total Marrow and Lymph Node Irradiation](https://arxiv.org/abs/2402.06494) | 本文基于深度学习探讨了在全骨髓和淋巴结照射治疗中，使用2D和3D U-Net模型以及nnU-Net框架自动分割照射计划靶体积(PTV)的方法，结果表明nnU-Net框架显著提高了分割性能。 |
| [^28] | [Inducing Systematicity in Transformers by Attending to Structurally Quantized Embeddings](https://arxiv.org/abs/2402.06492) | 本论文提出了SQ-Transformer模型，通过在嵌入和注意层中引入结构化量化的方法，无论训练集的复杂度如何，都能够明确地鼓励模型在编码句子时保持系统性。 |
| [^29] | [Cardiac ultrasound simulation for autonomous ultrasound navigation](https://arxiv.org/abs/2402.06463) | 该论文提出了一种用于自主超声导航的心脏超声模拟方法，通过使用其他模态的分割、优化的体积数据表示和GPU加速的蒙特卡洛路径追踪，生成大量视角相关和具有患者特异性的超声图像。 |
| [^30] | [Sequential Flow Matching for Generative Modeling](https://arxiv.org/abs/2402.06461) | 本文提出了一种称为SeqRF的新方法，用于通过直线化概率流来减小全局截断误差，并以此加速取样和提高综合质量。 |
| [^31] | [V-STaR: Training Verifiers for Self-Taught Reasoners](https://arxiv.org/abs/2402.06457) | V-STaR利用正确和不正确的解决方案训练验证器，用于选择模型生成的解决方案，实现了自我改进和验证方法在常见代码生成和数学推理任务中达到4%到17%的测试准确率提升。 |
| [^32] | [An Algorithmic Framework for Constructing Multiple Decision Trees by Evaluating Their Combination Performance Throughout the Construction Process](https://arxiv.org/abs/2402.06452) | 本研究提出了一种新的算法框架，能够同时构建多个决策树，并在构建过程中评估它们的组合性能，以指导最终预测的决策树组合的适用性。 |
| [^33] | [The Deep Equilibrium Algorithmic Reasoner](https://arxiv.org/abs/2402.06445) | 本文介绍了一种深度平衡算法推理器，可以通过直接找到算法的平衡点来训练网络解决算法问题。 |
| [^34] | [Incorporating Taylor Series and Recursive Structure in Neural Networks for Time Series Prediction](https://arxiv.org/abs/2402.06441) | 本论文提出了一种新的神经网络架构，将ResNet结构和泰勒级数框架相结合，实现了对时间序列预测的显著改进。此外，引入递归结构可以进一步提高预测准确性。这一研究为时间序列分析方法学的推进提供了潜力巨大的模型，具有广阔的研究和应用前景。 |
| [^35] | [Where is the Truth? The Risk of Getting Confounded in a Continual World](https://arxiv.org/abs/2402.06434) | 这篇论文研究了在一个连续学习环境中遭遇混淆的问题，通过实验证明了传统的连续学习方法无法忽略混淆，需要更强大的方法来处理这个问题。 |
| [^36] | [Trust the Process: Zero-Knowledge Machine Learning to Enhance Trust in Generative AI Interactions](https://arxiv.org/abs/2402.06414) | 本文提出了使用零知识证明技术来解决生成式AI公平性和隐私保护的问题，并介绍了一个实际的ZKML实现，可以增强用户对生成式AI输出的信任和透明度。 |
| [^37] | [Improving the Worst-Case Bidirectional Communication Complexity for Nonconvex Distributed Optimization under Function Similarity](https://arxiv.org/abs/2402.06412) | 本文提出了MARINA-P方法，通过引入一系列相关压缩器，优化了服务器到工作节点的通信复杂度。理论分析证明，MARINA-P在算法上优于现有方法，并可以作为支持双向压缩的起点。通过与上行压缩和动量步骤的结合，M3方法实现了双向压缩，并在总通信复杂度上改进。 |
| [^38] | [Hierarchical Transformers are Efficient Meta-Reinforcement Learners](https://arxiv.org/abs/2402.06402) | 层次化Transformer是一种高效的元强化学习方法，通过有效地提取过去经验的信息丰富资源，并应用于新的环境中，实现了超越最先进方法的元训练效果，并显著提高了泛化能力和学习效率。 |
| [^39] | [On the Convergence Rate of the Stochastic Gradient Descent (SGD) and application to a modified policy gradient for the Multi Armed Bandit](https://arxiv.org/abs/2402.06388) | 该论文证明了当学习速率按照逆时间衰减规则时，随机梯度下降（SGD）的收敛速度，并应用于修改的带有L2正则化的策略梯度多臂赌博机（MAB）的收敛性分析。 |
| [^40] | [Boosting-Based Sequential Meta-Tree Ensemble Construction for Improved Decision Trees](https://arxiv.org/abs/2402.06386) | 本研究提出了一种使用提升方法构建多个元树的方法，旨在改进决策树的预测性能。 |
| [^41] | [Optimal estimation of Gaussian (poly)trees](https://arxiv.org/abs/2402.06380) | 该论文开发了最优算法，在学习高斯树和高斯多项式树方面取得了显著成果，并提供了详细的理论保证和实证结果。 |
| [^42] | [High-Precision Geosteering via Reinforcement Learning and Particle Filters](https://arxiv.org/abs/2402.06377) | 基于强化学习和粒子滤波的地质定向方法，通过实时数据处理实现高精度地质定向决策 |
| [^43] | [TEE4EHR: Transformer Event Encoder for Better Representation Learning in Electronic Health Records](https://arxiv.org/abs/2402.06367) | TEE4EHR是一个使用点过程损失函数的Transformer事件编码器，用于编码电子健康记录中实验室测试的模式。它能够解决EHR中时间序列的不规则采样和缺失数据的挑战，并在各种基准数据集和实际数据库上展现出优越性能。 |
| [^44] | [The SpongeNet Attack: Sponge Weight Poisoning of Deep Neural Networks](https://arxiv.org/abs/2402.06357) | 本文提出了一种名为 SpongeNet 的新型海绵攻击，通过直接作用于预训练模型参数，成功增加了视觉模型的能耗，而且所需的样本数量更少。 |
| [^45] | [Fairness of Exposure in Online Restless Multi-armed Bandits](https://arxiv.org/abs/2402.06348) | 本研究提出了第一个在线的公平RMAB框架，通过将每个臂的拉取与其优势成比例，实现了公平的曝光。算法在单次拉取的公平性遗憾方面取得了次线性的结果$O(\sqrt{T\ln T})$。 |
| [^46] | [Taking Class Imbalance Into Account in Open Set Recognition Evaluation](https://arxiv.org/abs/2402.06331) | 本文研究了开放集识别评估中类别不平衡问题的影响，并提出了一套指导方针。 |
| [^47] | [Continual Learning on Graphs: A Survey](https://arxiv.org/abs/2402.06330) | 本文是一项关于持续图学习的综合调查，其中提出了新的分类法来克服灾难性遗忘问题，并分析了持续性能改进的挑战和可能的解决方案。 |
| [^48] | [Prompt Learning on Temporal Interaction Graphs](https://arxiv.org/abs/2402.06326) | 这个论文提出了一种在时间交互图上进行提示学习的方法，以解决当前模型在预训练和下游预测阶段所面临的时间差异和语义差异的问题。 |
| [^49] | [How Uniform Random Weights Induce Non-uniform Bias: Typical Interpolating Neural Networks Generalize with Narrow Teachers](https://arxiv.org/abs/2402.06323) | 在插值神经网络中，均匀随机权重可以产生非均匀偏差，因此通常插值神经网络会与窄教师NN一样很好地泛化。 |
| [^50] | [Particle Denoising Diffusion Sampler](https://arxiv.org/abs/2402.06320) | 本文介绍了一种粒子去噪扩散采样器（PDDS），通过使用原始迭代粒子方案和新颖的得分匹配损失，对非归一化概率密度进行采样和计算规范化常数。与标准的去噪扩散模型不同，PDDS 在温和假设下提供了渐近一致的估计。 |
| [^51] | [TimEHR: Image-based Time Series Generation for Electronic Health Records](https://arxiv.org/abs/2402.06318) | 提出了一种新的基于生成对抗网络的模型TimEHR，用于从EHR生成时间序列数据。通过将时间序列视为图像，并使用两个条件GAN，TimEHR在处理不规则采样、缺失值和高维度方面取得了优于现有方法的结果。 |
| [^52] | [Multimodal Interpretable Data-Driven Models for Early Prediction of Antimicrobial Multidrug Resistance Using Multivariate Time-Series](https://arxiv.org/abs/2402.06295) | 本研究提出了一种基于可解释的多模态数据驱动模型的方法，通过静态数据和多元时间序列模型预测和理解重症监护病房中抗菌药物多重耐药性细菌的出现。 |
| [^53] | [Probabilistic Forecasting of Irregular Time Series via Conditional Flows](https://arxiv.org/abs/2402.06293) | 该论文提出了一种使用条件流进行不规则时间序列的概率预测的新模型ProFITi。该模型通过学习条件下未来值的联合分布，对具有缺失值的不规则时间序列进行预测，而不假设底层分布的固定形状。通过引入可逆三角形注意力层和可逆非线性激活函数，该模型取得了良好的实验结果。 |
| [^54] | [Evaluating Membership Inference Attacks and Defenses in Federated Learning](https://arxiv.org/abs/2402.06289) | 这篇论文评估了联邦学习中成员推断攻击和防御的情况。评估揭示了两个重要发现：多时序的模型信息有助于提高攻击的有效性，多空间的模型信息有助于提高攻击的效果。这篇论文还评估了两种防御机制的效用和隐私权衡。 |
| [^55] | [AI, Meet Human: Learning Paradigms for Hybrid Decision Making Systems](https://arxiv.org/abs/2402.06287) | 本调查提出了混合决策系统的分类方法，为理解如何对人与机器之间的交互进行建模提供了概念性和技术性的框架。 |
| [^56] | [Retrieve, Merge, Predict: Augmenting Tables with Data Lakes](https://arxiv.org/abs/2402.06282) | 本文通过对数据湖中的数据发现进行深入分析，着重于表格增强，提出了准确检索连接候选人的重要性和简单合并方法的效率，以及现有解决方案的好处和局限性。 |
| [^57] | [Controllable seismic velocity synthesis using generative diffusion models](https://arxiv.org/abs/2402.06277) | 本论文提出使用生成扩散模型进行地震速度合成，通过纳入先验信息，可以生成与实验数据密切匹配的地震速度。 |
| [^58] | [Safe Active Learning for Time-Series Modeling with Gaussian Processes](https://arxiv.org/abs/2402.06276) | 本研究提出了一种安全的主动学习方法，用于时间序列建模。通过动态探索输入空间并根据安全要求和过去观察的输入和输出轨迹，我们的方法在现实技术应用中展示了其有效性。 |
| [^59] | [Neural SPH: Improved Neural Modeling of Lagrangian Fluid Dynamics](https://arxiv.org/abs/2402.06275) | 本研究发展了一种叫做神经SPH的方法，通过增强图神经网络和标准SPH求解器的组合来改进GNN模拟器的性能，在准确建模物理现象方面取得了较好的效果。 |
| [^60] | [Adaptive proximal gradient methods are universal without approximation](https://arxiv.org/abs/2402.06271) | 该论文证明了适应性近端梯度方法对于凸问题不受传统假设的限制，并且可以在局部梯度H\"older连续性条件下收敛，同时避免了线搜索步骤和近似的使用。对局部H\"older常数和H\"older连续性顺序的先验知识也不是必需的。在数值实验中，与基准方法进行了对比实验，涵盖了局部和全局 H\"older 设置。 |
| [^61] | [YAMLE: Yet Another Machine Learning Environment](https://arxiv.org/abs/2402.06268) | YAMLE是一个开源机器学习环境，旨在减少重复工作并提高机器学习研究的可重现性。它包括命令行界面和与PyTorch库的集成，致力于成为一个共享的生态系统。 |
| [^62] | [Value function interference and greedy action selection in value-based multi-objective reinforcement learning](https://arxiv.org/abs/2402.06266) | 基于价值的多目标强化学习中，如果用户的效用函数将广泛变化的向量值映射为相似的效用水平，会导致价值函数干扰并收敛到次优策略。 |
| [^63] | [Studious Bob Fight Back Against Jailbreaking via Prompt Adversarial Tuning](https://arxiv.org/abs/2402.06255) | 本文提出了一种名为Prompt Adversarial Tuning (PAT)的方法，通过训练一个防御控制机制并将其作为前缀嵌入到用户提示中，实现对大型语言模型（LLMs）的越狱行为的防御。实验证明该方法在保护LLMs免受产生有害信息的影响方面效果显著。 |
| [^64] | [N-1 Reduced Optimal Power Flow Using Augmented Hierarchical Graph Neural Network](https://arxiv.org/abs/2402.06226) | 本文提出了一种使用增强型分层图神经网络（AHGNN）的方法来预测关键拥塞线路并创建N-1优化潮流的降维（N-1 ROPF）。实验结果表明，AHGNN启用的N-1 ROPF在保持解的质量的同时，能够显著减少计算时间。 |
| [^65] | [Adaptive multi-gradient methods for quasiconvex vector optimization and applications to multi-task learning](https://arxiv.org/abs/2402.06224) | 我们提出了一种自适应步长方法，用于解决一类广泛的非凸多目标规划问题，并应用于创新的多梯度投影方法和多任务学习，展示了其在大规模挑战中的效果。 |
| [^66] | [Revealing Multimodal Contrastive Representation Learning through Latent Partial Causal Models](https://arxiv.org/abs/2402.06223) | 通过潜在部分因果模型，我们展示了多模式对比表示学习在识别潜在耦合变量方面的优秀能力，并揭示了预训练的多模态模型通过线性独立分量分析学习分离表示的潜力。 |
| [^67] | [The Berkeley Single Cell Computational Microscopy (BSCCM) Dataset](https://arxiv.org/abs/2402.06191) | 伯克利单细胞计算显微镜（BSCCM）数据集包含了约12,000,000张个体白血细胞的图像，提供了用于计算显微镜和计算机视觉算法开发和测试的宝贵资源。 |
| [^68] | [Masked LoGoNet: Fast and Accurate 3D Image Analysis for Medical Domain](https://arxiv.org/abs/2402.06190) | 本文介绍了一种名为LoGoNet的新型神经网络架构，采用自监督学习方法来应对医学图像分析中的挑战。LoGoNet通过采用大内核注意力和双重编码策略，灵活捕捉长、短距离特征相关性。这种创新的组合技术在医学图像分割中特别有益。 |
| [^69] | [A self-supervised framework for learning whole slide representations](https://arxiv.org/abs/2402.06188) | 这个论文提出了一个自监督学习框架（S3L），用于学习整个切片的表示。它结合了变压器模型的视觉和语言建模策略，通过生成配对视图进行自监督学习，以实现高质量的WSI视觉特征学习。 |
| [^70] | [Premier-TACO: Pretraining Multitask Representation via Temporal Action-Driven Contrastive Loss](https://arxiv.org/abs/2402.06187) | Premier-TACO是一种多任务特征表示学习方法，通过预训练通用特征表示，并引入负例抽样策略来提高时序行动对比学习的计算效率，从而显著增强了对新颖动作的少样本模仿学习的效果。 |
| [^71] | [Development and validation of an artificial intelligence model to accurately predict spinopelvic parameters](https://arxiv.org/abs/2402.06185) | 该研究开发了一个名为SpinePose的人工智能工具，可以准确预测脊盘盆参数，无需手动输入。 |
| [^72] | [The boundary of neural network trainability is fractal](https://arxiv.org/abs/2402.06184) | 本论文通过实验证明，神经网络训练的边界是分形的，对于超参数的微小改变非常敏感，这对于理解神经网络的训练可行性具有重要意义。 |
| [^73] | [SMC Is All You Need: Parallel Strong Scaling](https://arxiv.org/abs/2402.06173) | SMC并行扩展方法pSMC具有理论收敛速度，具有有界的时间复杂性和内存要求，适用于贝叶斯推断的问题。 |
| [^74] | [Pushing Boundaries: Mixup's Influence on Neural Collapse](https://arxiv.org/abs/2402.06171) | 本研究揭示了Mixup对神经塌陷的影响，通过对深度网络训练数据的最后一层激活进行研究，发现Mixup的最后一层激活收敛到与预期不同的独特配置。 |
| [^75] | [Learning Contrastive Feature Representations for Facial Action Unit Detection](https://arxiv.org/abs/2402.06165) | 这项研究提出了一种对比学习框架，通过监督和自监督信号来增强面部动作单元检测模型的性能。采用正样本抽样和权衡重要性的损失函数来应对噪声AU标签和AU类型分布不平衡的挑战。 |
| [^76] | [Wasserstein proximal operators describe score-based generative models and resolve memorization](https://arxiv.org/abs/2402.06162) | 该论文研究了基于分数的生成模型的数学结构，通过Wasserstein近端算子和平均场博弈可以描述生成模型的归纳偏差，通过解耦合的偏微分方程可以获得优化条件，提出了一个可解释的基于核的得分函数模型，极大地提高了生成模型的性能。 |
| [^77] | [Improved Evidential Deep Learning via a Mixture of Dirichlet Distributions](https://arxiv.org/abs/2402.06160) | 本文通过混合狄利克雷分布来改进证据深度学习（EDL）方法，解决了现有方法中认知不确定性在无限样本限制下可能不会消失的问题。 |
| [^78] | [POTEC: Off-Policy Learning for Large Action Spaces via Two-Stage Policy Decomposition](https://arxiv.org/abs/2402.06151) | POTEC提出了一种两阶段策略分解的算法，在大离散动作空间中有效进行离策略学习。该算法利用聚类选择第一阶段策略，并利用回归方法选择每个聚类内的具体动作。 |
| [^79] | [Domain Generalization with Small Data](https://arxiv.org/abs/2402.06150) | 本研究提出了一种在样本不足情况下解决领域泛化问题的方法。该方法通过使用概率嵌入来学习领域不变表示，并使用概率框架中的新方法测量混合分布之间的差异。结果表明，该方法在领域泛化性能上有显著的提升。 |
| [^80] | [On the Privacy of Selection Mechanisms with Gaussian Noise](https://arxiv.org/abs/2402.06137) | 该论文研究了带有高斯噪声的选择机制的隐私问题，并证明了在底层查询是有界的情况下，可以提供纯粹的前期和后期差分隐私界限。 |
| [^81] | [Jointly Learning Representations for Map Entities via Heterogeneous Graph Contrastive Learning](https://arxiv.org/abs/2402.06135) | 通过异构图对比学习方法能够共同学习多个类别的地图实体的表示，提取潜在的结构和语义信息。 |
| [^82] | [Rethinking Node-wise Propagation for Large-scale Graph Learning](https://arxiv.org/abs/2402.06128) | 提出了适应性拓扑感知传播（ATP）方法，以应对大规模图学习中节点传播的问题。此方法能对不同节点的拓扑角色进行个性化传播，并减少传播带来的偏差和额外开销。 |
| [^83] | [CityFlowER: An Efficient and Realistic Traffic Simulator with Embedded Machine Learning Models](https://arxiv.org/abs/2402.06127) | CityFlowER是一种高效真实的交通模拟器，通过嵌入机器学习模型提高了模拟的真实性和效率。 |
| [^84] | [Learn To be Efficient: Build Structured Sparsity in Large Language Models](https://arxiv.org/abs/2402.06126) | 本文通过引入一种新的算法"Learn-To-be-Efficient(LTE)"，提出了在大型语言模型(LLM)中构建结构化稀疏性的方法。该方法通过训练高效意识的LLM学习激活更少的神经元，取得更好的稀疏性和性能折衷。 |
| [^85] | [Peeking with PEAK: Sequential, Nonparametric Composite Hypothesis Tests for Means of Multiple Data Streams](https://arxiv.org/abs/2402.06122) | 本论文提出了一种名为PEAK的新型非参数顺序复合假设检验方法，适用于多个数据流的均值检验。该方法基于测试即博弈的框架，在任何停止时间上提供了非渐进α水平的检验。PEAK能够有效拒绝在满足非参数假设条件的所有潜在分布中错误的假设，从而实现对多个数据流的联合复合假设检验。与现有方法相比，该方法具有较高的计算效率。 |
| [^86] | [Iterated Denoising Energy Matching for Sampling from Boltzmann Densities](https://arxiv.org/abs/2402.06121) | 提出了一种基于迭代算法的新颖采样方法，通过利用能量函数和梯度进行训练，无需数据样本。该方法能够高效生成统计独立的样本，并且在高维度上具有可扩展性。通过利用扩散的快速模式混合行为，实现了对能量景观的平滑，从而实现了高效的探索和学习。 |
| [^87] | [AI enhanced data assimilation and uncertainty quantification applied to Geological Carbon Storage](https://arxiv.org/abs/2402.06110) | 本研究通过整合机器学习和数据同化技术，提出了用于地质碳封存的代理模型方法，可在维持高准确性的同时加快同化过程，具有较大的应用潜力。 |
| [^88] | [Multiple Instance Learning for Cheating Detection and Localization in Online Examinations](https://arxiv.org/abs/2402.06107) | 本文提出了一种基于多实例学习的作弊检测框架CHEESE，该框架综合考虑了头部姿势、凝视角度、身体姿势和背景信息等特征，并通过标签生成器和特征编码器实现了作弊行为的检测和定位。 |
| [^89] | [Function Aligned Regression: A Method Explicitly Learns Functional Derivatives from Data](https://arxiv.org/abs/2402.06104) | 该论文提出了一种名为FAR的方法，通过捕捉函数导数来更好、更高效地拟合底层真实函数。在合成数据集和八个真实世界任务中证明了该方法的有效性。 |
| [^90] | [Real-World Fluid Directed Rigid Body Control via Deep Reinforcement Learning](https://arxiv.org/abs/2402.06102) | 通过创建流体盒子实验控制系统，我们展示了在真实世界场景中使用深度强化学习算法合成复杂行为的能力，并探索了离线强化学习在数据高效假设测试中的潜力。 |
| [^91] | [Veni, Vidi, Vici: Solving the Myriad of Challenges before Knowledge Graph Learning](https://arxiv.org/abs/2402.06098) | 知识图谱学习面临着缺乏专家知识整合、节点度数极端性不稳定、缺乏不确定性和相关性的考虑以及缺乏可解释性的挑战。现有的解决尝试大多是孤立的，需要综合考虑这些问题的解决方案。 |
| [^92] | [TWIG: Towards pre-hoc Hyperparameter Optimisation and Cross-Graph Generalisation via Simulated KGE Models](https://arxiv.org/abs/2402.06097) | 这项研究引入了一种名为TWIG的新颖模型，可以通过拓扑特征学习权重来模拟KGE模型的输出，有效减少了参数数量，并具有预先优化超参数和跨图泛化的能力。 |
| [^93] | [Descriptive Kernel Convolution Network with Improved Random Walk Kernel](https://arxiv.org/abs/2402.06087) | 本文提出了一种描述性核卷积网络，通过改进随机游走核并引入颜色匹配随机游走，提升了图核在特征工程中的可学习性。进一步发现了随机游走核与GCN层的联系，并提出了一种新颖的GNN方法。 |
| [^94] | [SubGen: Token Generation in Sublinear Time and Memory](https://arxiv.org/abs/2402.06082) | 这项工作提出了一种名为SubGen的高效缓存压缩技术，通过在Attention模块中进行在线聚类和采样，实现了子线性的内存占用和时间复杂度，并建立了一个紧密的误差界。 |
| [^95] | [DiscDiff: Latent Diffusion Model for DNA Sequence Generation](https://arxiv.org/abs/2402.06079) | 本文介绍了一种新的框架用于生成DNA序列，包括一个用于生成离散DNA序列的潜在扩散模型和一个用于改进序列的后训练算法。我们的方法不仅在DNA序列生成方面树立了新的标准，并且在生成短序列和长序列方面表现出优越性能。此外，我们还引入了一个多物种的DNA生成数据集。这项研究将推动DNA的生成建模，并对基因治疗和蛋白质生产产生潜在影响。 |
| [^96] | [Scaling Artificial Intelligence for Digital Wargaming in Support of Decision-Making](https://arxiv.org/abs/2402.06075) | 针对决策支持下的战争游戏，本论文提出了规模化人工智能的发展并与人类判断相结合的重要性。通过提高全域意识、改善决策速度和质量、提供创新行动建议以及更快速地应对对手行动，我们能够更好地应对现代挑战和困境，增强人类决策的指导和增强。 |
| [^97] | [3D-2D Neural Nets for Phase Retrieval in Noisy Interferometric Imaging](https://arxiv.org/abs/2402.06063) | 我们引入了一种称为PRUNe的三维二维相位恢复神经网络，用于解决存在噪声的干涉成像中的相位恢复问题。该网络能够处理相位噪声，并在恢复精度和准确性方面表现出优势。 |
| [^98] | [Impact on Public Health Decision Making by Utilizing Big Data Without Domain Knowledge](https://arxiv.org/abs/2402.06059) | 本研究使用大量街景图像和纽约市的健康数据，发现基于数据的决策在没有考虑数据健壮性和基于虚假相关性时存在偏见。 |
| [^99] | [ActiveDP: Bridging Active Learning and Data Programming](https://arxiv.org/abs/2402.06056) | 本文提出了ActiveDP，一个将主动学习和数据编程相结合的交互式框架，以生成具有高准确性和覆盖率的标签，实验证明其优于以前的弱监督和主动学习方法，并在不同的标记预算下表现良好。 |
| [^100] | [Intelligent Mode-switching Framework for Teleoperation](https://arxiv.org/abs/2402.06047) | 本研究提出了一种智能模式切换框架，通过考虑模式切换和通信系统，解决了远程操作中的难题。通过预测用户意图并自主执行任务的一部分，减少了对操作者的需求，提高了任务完成率。 |
| [^101] | [Direct Acquisition Optimization for Low-Budget Active Learning](https://arxiv.org/abs/2402.06045) | 本研究提出了一种针对低预算主动学习的直接采集优化算法（DAO），该算法可以更准确地估计总体误差减少，效果超过了现有的方法。 |
| [^102] | [Contrastive Approach to Prior Free Positive Unlabeled Learning](https://arxiv.org/abs/2402.06038) | 该论文提出了一种免先验正无标学习的对比方法，通过预训练不变表示学习特征空间并利用嵌入的浓度特性对未标记样本进行伪标签处理，相比现有方法，在多个标准数据集上表现优异，同时不需要先验知识或类先验的估计。 |
| [^103] | [Optimizing Predictive AI in Physical Design Flows with Mini Pixel Batch Gradient Descent](https://arxiv.org/abs/2402.06034) | 本论文提出了一种迷你像素批量梯度下降（MPGD）算法，用于优化物理设计流程中的预测AI。实验证明MPGD在各种物理设计预测任务中具有显著的优势。 |
| [^104] | [An Inexact Halpern Iteration for with Application to Distributionally Robust Optimization](https://arxiv.org/abs/2402.06033) | 本文研究了确定性和随机环境下Halpern迭代算法的不精确变种，通过适当选择不精确的容差，这些变种展现出O(k^-1)的收敛速度，同时具有竞争性的收敛特性。并且我们还展示了这些方法在两类数据驱动Wasserstein分布鲁棒优化问题中的应用，以及在分布鲁棒学习中使用随机一阶方法进行不精确计算的能力。 |
| [^105] | [An operator learning perspective on parameter-to-observable maps](https://arxiv.org/abs/2402.06031) | 本论文从算子学习的视角研究了参数到可观测映射，提出了适应有限维输入和输出的傅里叶神经映射框架，并发展了通用逼近定理来支持该方法。此外，讨论了学习PtO映射的端到端方法和先学习解算子再计算可观测值的效率问题。 |
| [^106] | [Game-theoretic Counterfactual Explanation for Graph Neural Networks](https://arxiv.org/abs/2402.06030) | 本文提出了一种半值法的、非学习的方法来生成图神经网络的反事实解释，消除了额外训练的需要。与其他流行的方法相比，计算Banzhaf值在识别反事实解释时需要更低的样本复杂性，并且可以实现四倍的加速。 |
| [^107] | [Decision Theory-Guided Deep Reinforcement Learning for Fast Learning](https://arxiv.org/abs/2402.06023) | 决策理论引导的深度强化学习（DT-guided DRL）通过整合决策理论原则，实现了对DRL智能体的有效初始引导，并促进了在复杂环境中更高效可靠的学习过程。 |
| [^108] | [Checking the Sufficiently Scattered Condition using a Global Non-Convex Optimization Software](https://arxiv.org/abs/2402.06019) | 本文提出了一种使用全局非凸优化软件Gurobi解决足够分散条件检验问题的方法，在实际场景中可以在合理的时间范围内进行检查。 |
| [^109] | [NPSVC++: Nonparallel Classifiers Encounter Representation Learning](https://arxiv.org/abs/2402.06010) | 本文研究了一种称为非并行支持向量分类器(NPSVCs)的分类器家族，提出了NPSVC++，基于多目标优化。NPSVC++通过表示学习实现了NPSVC及其特征的端到端学习，追求帕累托最优，有效地解决了特征次优和类别依赖的问题，在实验证明了其优越性。 |
| [^110] | [Capability enhancement of the X-ray micro-tomography system via ML-assisted approaches](https://arxiv.org/abs/2402.05983) | 本文提出了一种基于卷积神经网络的深度学习模型，通过去除环形伪影来增强X射线微CT系统的能力。 |
| [^111] | [Anfinsen Goes Neural: a Graphical Model for Conditional Antibody Design](https://arxiv.org/abs/2402.05982) | Anfinsen Goes Neural (AGN) is a graphical model for conditional antibody design that combines a pre-trained protein language model with a graph neural network. It outperforms existing methods and addresses the limitation of generating unrealistic sequences. |
| [^112] | [Exploring the Impact of In-Browser Deep Learning Inference on Quality of User Experience and Performance](https://arxiv.org/abs/2402.05981) | 本研究通过全面性能评估，探索了浏览器内深度学习推理对用户体验质量和性能的影响。研究发现，浏览器内推理存在严重的延迟问题，平均比原生推理方法慢16.9倍。为了衡量这种影响，我们引入了新的指标：响应性，流畅度和推理准确性。 |
| [^113] | [Do Large Code Models Understand Programming Concepts? A Black-box Approach](https://arxiv.org/abs/2402.05980) | 本文使用反事实分析框架评估了十个大型代码模型对四种编程概念的理解情况，发现当前模型缺乏对数据流和控制流等概念的理解。 |
| [^114] | [Combining shape and contour features to improve tool wear monitoring in milling processes](https://arxiv.org/abs/2402.05978) | 本文提出了一种新的系统，结合了形状描述符和轮廓描述符，用于铣削过程中插入物的磨损监测。实验结果表明，使用后期融合方法将两个描述符组合在一起，可以显著提高分类性能，取得更好的准确率。 |
| [^115] | [Tool wear monitoring using an online, automatic and low cost system based on local texture](https://arxiv.org/abs/2402.05977) | 本研究提出了一种基于计算机视觉和机器学习的在线、低成本和快速方法，用于切削工具的磨损监测。通过将切削边缘图像分割成不同的区域，并使用局部二值模式的纹理描述符来判断每个区域的磨损程度，从而确定切削边缘和刀具是否可服役或可丢弃。 |
| [^116] | [RankSum An unsupervised extractive text summarization based on rank fusion](https://arxiv.org/abs/2402.05976) | RankSum是一种无监督的抽取式文本摘要方法，它利用多维度句子特征对句子进行排名，然后通过加权融合确定句子的重要性排名。该方法不需要监督信号，可以推广到其他数据集。 |
| [^117] | [A Deep Learning Approach for Brain Tumor Classification and Segmentation Using a Multiscale Convolutional Neural Network](https://arxiv.org/abs/2402.05975) | 本文提出了一种使用多尺度卷积神经网络的深度学习方法，可以自动进行脑肿瘤的分类和分割。通过与其他方法的比较，我们的方法在公开数据集上获得了较高的分类准确率。 |
| [^118] | [Blockchain-enabled Clustered and Scalable Federated Learning (BCS-FL) Framework in UAV Networks](https://arxiv.org/abs/2402.05973) | 本论文提出了一种基于区块链的聚簇可扩展联邦学习（BCS-FL）框架，用于改善无人机网络中的联邦学习的去中心化、协调、可扩展性和效率。该框架将无人机网络划分为聚簇，并由聚簇头无人机进行协调，以提高大规模无人机网络中的联邦学习性能。 |
| [^119] | [Gaussian-process-regression-based method for the localization of exceptional points in complex resonance spectra](https://arxiv.org/abs/2402.05972) | 本文介绍了一种基于高斯过程回归的方法，用于在复共振谱中定位例外点。通过训练一个GPR模型，并使用一些初始的特征值对进行根搜索，来对例外点的位置进行初步估计。然后通过迭代添加确切的特征值对来改进估计。该方法在低维矩阵模型和真实物理系统上进行了测试。 |
| [^120] | [Are we making much progress? Revisiting chemical reaction yield prediction from an imbalanced regression perspective](https://arxiv.org/abs/2402.05971) | 本文从不平衡回归的角度重新审视了化学反应收率预测。在合成规划中，准确的高收率预测对于化学家来说至关重要。然而，真实世界数据的不平衡分布导致了现有方法在高收率预测方面的性能差距。 |
| [^121] | [Modeling Spatio-temporal Dynamical Systems with Neural Discrete Learning and Levels-of-Experts](https://arxiv.org/abs/2402.05970) | 本文提出了使用神经离散学习和专家级别建模空时动力系统的方法。通过引入通用的专家模块和精细设计的物理流水线，可以在更广泛的现实世界背景下有效地建模和估计空时动态系统的状态变化。 |
| [^122] | [Breaking Symmetry When Training Transformers](https://arxiv.org/abs/2402.05969) | 该论文讨论了在训练Transformer时，删除位置编码和因果注意力机制后，输出的预测结果对于输入符号排列是不变的。研究人员通过对因果连接机制进行细致分析，提出了残差连接对Transformer模拟输入顺序重要性的贡献。 |
| [^123] | [Federated Learning Priorities Under the European Union Artificial Intelligence Act](https://arxiv.org/abs/2402.05968) | 欧盟人工智能法案可能推动联邦学习朝主流采用方向发展，并提出了数据隐私、性能和能源效率等方面的新挑战。 |
| [^124] | [The last Dance : Robust backdoor attack via diffusion models and bayesian approach](https://arxiv.org/abs/2402.05967) | 本文介绍了一种通过扩散模型和贝叶斯方法进行鲁棒后门攻击的方法，具体应用于音频Transformer模型，并证明了攻击的可行性。 |
| [^125] | [Rethink Model Re-Basin and the Linear Mode Connectivity](https://arxiv.org/abs/2402.05966) | 本论文重新审视了模型重新基底的现象，并发现了现有匹配算法的不足。通过适当的重归一化，我们改进了匹配算法，并揭示了它与重归一化过程的相互作用。这为剪枝提供了新的理解，推动了一种轻量且有效的后剪枝插件的开发。 |
| [^126] | [Hybrid Neural Representations for Spherical Data](https://arxiv.org/abs/2402.05965) | 本文研究了球面数据的混合神经表示方法，通过使用球形特征格和多层感知器进行预测，有效地捕捉了高度非线性信号的复杂细节。 |
| [^127] | [A Survey on Transformer Compression](https://arxiv.org/abs/2402.05964) | 《Transformer压缩调研》是对最近压缩方法的全面回顾，特别关注它们在Transformer模型中的应用。压缩方法主要分为修剪、量化、知识蒸馏和高效架构设计四个类别。 |
| [^128] | [Frugal Actor-Critic: Sample Efficient Off-Policy Deep Reinforcement Learning Using Unique Experiences](https://arxiv.org/abs/2402.05963) | 该方法通过选择独特样本并添加到回放缓冲器中以实现样本效率，在复杂动态系统的无模型控制策略合成中起着重要作用。 |
| [^129] | [EXGC: Bridging Efficiency and Explainability in Graph Condensation](https://arxiv.org/abs/2402.05962) | 本论文提出了EXGC方法，通过采用平均场变分近似和梯度信息瓶颈目标来提高图压缩的效率和可解释性。 |
| [^130] | [Genetic-guided GFlowNets: Advancing in Practical Molecular Optimization Benchmark](https://arxiv.org/abs/2402.05961) | 本文提出了一种名为基因引导GFlowNet (Genetic GFN) 的新方法，通过集成迭代遗传搜索和训练策略，该方法在实际分子优化基准测试中取得了16.213的最新得分，明显优于现有最佳得分15.185，同时在14个任务中超越了所有对比方法。 |
| [^131] | [Phase-driven Domain Generalizable Learning for Nonstationary Time Series](https://arxiv.org/abs/2402.05960) | 该论文提出了一个基于相位驱动的时间序列学习框架PhASER，通过相位增强、分离特征编码和特征广播的方法，实现了对非平稳数据的通用学习能力。 |
| [^132] | [Nature-Inspired Local Propagation](https://arxiv.org/abs/2402.05959) | 本文介绍了一种自然启发的局部传播算法，该算法通过在线处理环境信息而不依赖大量数据集，在机器学习领域具有潜力。这种算法的核心思想是结合数据表示和学习，以尊重时空局部性，并且当传播速度趋近于无穷大时，它等效于反向传播算法。 |
| [^133] | [A comparative study on wearables and single-camera video for upper-limb out-of-thelab activity recognition with different deep learning architectures](https://arxiv.org/abs/2402.05958) | 本研究比较了穿戴设备和单摄像头视频在不同深度学习体系结构下的上肢活动识别，为在实验室外跟踪患者活动提供了可行性，并探讨了在野外环境中识别和处理临床相关数据的机器学习系统的理想输入。 |
| [^134] | [Accelerating PDE Data Generation via Differential Operator Action in Solution Space](https://arxiv.org/abs/2402.05957) | 通过在解空间中应用微分算子作用，我们提出了一种加速PDE数据生成的算法，名为DiffOAS。它能够在生成数据的同时提高数据的精度，并且在时间复杂度上比现有的方法更高效。 |
| [^135] | [Pathformer: Multi-scale transformers with Adaptive Pathways for Time Series Forecasting](https://arxiv.org/abs/2402.05956) | 本文提出了一种名为Pathformer的多尺度自适应路径的Transformer模型，用于时间序列预测。通过整合时间分辨率和时间距离进行多尺度建模，并使用自适应路径来优化建模过程，可以提高预测准确性和泛化能力。 |
| [^136] | [A Hyper-Transformer model for Controllable Pareto Front Learning with Split Feasibility Constraints](https://arxiv.org/abs/2402.05955) | 本论文提出了一种用于具有分离可行性约束的可控帕累托前沿学习的超级变压器模型，可以通过近似和定位帕累托最优解来解决分裂多目标优化问题，并且在实践中限制了决策者目标的约束区域进行训练。 |
| [^137] | [EasyFS: an Efficient Model-free Feature Selection Framework via Elastic Transformation of Features](https://arxiv.org/abs/2402.05954) | EasyFS是一种高效的无模型特征选择框架，通过对特征进行弹性扩展和压缩，实现了对特征之间相互关系的建模，并发现最相关的特征。同时，通过新的冗余度度量方法实现了对冗余特征的高效过滤。 |
| [^138] | [idMotif: An Interactive Motif Identification in Protein Sequences](https://arxiv.org/abs/2402.05953) | idMotif是一个可视化分析框架，旨在帮助领域专家识别蛋白质序列中的模体。它利用深度学习方法对蛋白质序列进行分类，并通过局部解释深度学习模型的决策，发现潜在的模体候选序列。它提供多个交互式视图，用于分析蛋白质聚类和序列。一项案例研究证明了idMotif在蛋白质序列和模体分析与识别中的实用性。 |
| [^139] | [Advancing Graph Representation Learning with Large Language Models: A Comprehensive Survey of Techniques](https://arxiv.org/abs/2402.05952) | 本综述调查了将大型语言模型（LLM）与图表示学习（GRL）相结合的技术，并提供了一个新颖的分类法，深入分析了这些模型的核心组成部分和操作技术，为有效的模型设计和训练策略提供了新的视角。 |
| [^140] | [\textit{MinMaxMin} $Q$-learning](https://arxiv.org/abs/2402.05951) | \textit{MinMaxMin} $Q$-learning是一种乐观型Actor-Critic算法，通过解决过高估计偏差的问题，在各种基准任务中相对于现有算法表现出稳定的性能提升。 |
| [^141] | [\textit{SQT} -- \textit{std} $Q$-target](https://arxiv.org/abs/2402.05950) | SQT是一种基于Q-学习的保守型actor-critic算法，利用Q网络的标准差作为一种“不确定性惩罚”，成功解决了过高估计偏差问题，相较于TD3的Q-target公式具有更好的性能优势。 |
| [^142] | [An explainable machine learning-based approach for analyzing customers' online data to identify the importance of product attributes](https://arxiv.org/abs/2402.05949) | 本研究提出了一种基于机器学习和博弈论的方法，通过分析在线客户数据，提取产品开发的全面设计启示，并评估每个特性对总体满意度的重要性。 |
| [^143] | [DE$^3$-BERT: Distance-Enhanced Early Exiting for BERT based on Prototypical Networks](https://arxiv.org/abs/2402.05948) | DE$^3$-BERT是一种基于原型网络和距离度量的增强距离早期停止框架，用于提高BERT等预训练语言模型的推断速度和准确性。 |
| [^144] | [Separable Multi-Concept Erasure from Diffusion Models](https://arxiv.org/abs/2402.05947) | 提出了可分离的多概念抹除器（SepME），通过生成概念无关表示和权重解耦来解决扩散模型中的多概念抹除问题，并在不影响生成性能的情况下恢复概念。 |
| [^145] | [Unveiling Latent Causal Rules: A Temporal Point Process Approach for Abnormal Event Explanation](https://arxiv.org/abs/2402.05946) | 本文提出了一种基于时间点过程的方法，通过揭示潜在因果规律来解释异常事件，以帮助在高风险系统如医疗保健中快速诊断和精确治疗规划。该方法通过期望最大化算法优化规则集和模型参数，实现了准确的规则发现和根因识别。 |
| [^146] | [Eliminating Information Leakage in Hard Concept Bottleneck Models with Supervised, Hierarchical Concept Learning](https://arxiv.org/abs/2402.05945) | 本文解决了概念瓶颈模型中的信息泄漏问题，通过引入标签监督和构建分层概念集，提出了一种新的CBMs范例（SupCBM），它可以通过预测的概念和干预矩阵实现标签预测，并且只在不同的类别之间进行区分。 |
| [^147] | [Todyformer: Towards Holistic Dynamic Graph Transformers with Structure-Aware Tokenization](https://arxiv.org/abs/2402.05944) | Todyformer是一个全新的基于变压器的神经网络，它通过结合局部编码和全局编码能力，采用新颖的标记化策略和时间位置编码来解决动态图形中的过度压缩和长程依赖性问题。 |
| [^148] | [A hybrid IndRNNLSTM approach for real-time anomaly detection in software-defined networks](https://arxiv.org/abs/2402.05943) | 本文提出了一种混合 IndRNNLSTM 方法，用于实时检测软件定义网络中的异常。该方法通过结合 IndRNN 和 LSTM 的特点，学习相关和非相关特征，并使用四种特征选择模型提供适当的特征视角。在 NSL-KDD 数据集上实验结果显示，该方法达到了较低的 MAE 和 RMSE 值。 |
| [^149] | [Cooperative Knowledge Distillation: A Learner Agnostic Approach](https://arxiv.org/abs/2402.05942) | 合作知识蒸馏是一种通过多个模型相互合作来传递知识的方法，可以弥补传统知识蒸馏的局限性。不同模型的优劣势可以更有效地传递知识。 |
| [^150] | [Character-based Outfit Generation with Vision-augmented Style Extraction via LLMs](https://arxiv.org/abs/2402.05941) | 本文提出了一个新的基于人物的服装生成（COG）问题，旨在准确解释人物信息并根据用户的规范生成服装组合。我们提出了一个新颖的框架LVA-COG，利用大型语言模型（LLMs）从用户的兴趣中提取见解，并结合文本到图像模型，增强了对连贯服装的视觉理解和生成。 |
| [^151] | [Causal Relationship Network of Risk Factors Impacting Workday Loss in Underground Coal Mines](https://arxiv.org/abs/2402.05940) | 本研究使用一种新颖的因果人工智能方法，通过分析井下煤矿的伤害记录数据，建立了井下煤矿工作时间损失的因果关系网络。发现关键的因果关系包括风源和工作状态等不同因素之间的作用。 |
| [^152] | [Uncertainty Awareness of Large Language Models Under Code Distribution Shifts: A Benchmark Study](https://arxiv.org/abs/2402.05939) | 本文研究了大型语言模型在代码分布转移下的不确定性意识，并通过引入大规模基准数据集和应用概率方法来提高语言模型的可靠性。 |
| [^153] | [Large Language Model Meets Graph Neural Network in Knowledge Distillation](https://arxiv.org/abs/2402.05894) | 本论文提出了一种新颖的图知识蒸馏框架，使用大规模语言模型作为教师模型、图神经网络作为学生模型，解决了在理解文本-属性图中的节点分类问题中的限制。 |
| [^154] | [Knowledge Graphs Meet Multi-Modal Learning: A Comprehensive Survey](https://arxiv.org/abs/2402.05391) | 知识图谱与多模态学习的综述介绍了KG4MM和MM4KG两个主要方面，包括任务定义、构建进展、评估基准以及关键研究轨迹。 |
| [^155] | [Bellman Conformal Inference: Calibrating Prediction Intervals For Time Series](https://arxiv.org/abs/2402.05203) | 贝尔曼符合推断（BCI）是一个框架，通过解决一维随机控制问题，利用多步预测来提供校准的时间序列预测区间。BCI在任意分布转换和时间依赖性下实现了长期覆盖，且在波动率预测问题上生成更短的预测区间。 |
| [^156] | [Moco: A Learnable Meta Optimizer for Combinatorial Optimization](https://arxiv.org/abs/2402.04915) | Moco是一个可学习的组合优化元优化器，通过学习图神经网络来更新解决方案构建过程，并能够适应不同的情况和计算预算。 |
| [^157] | [On a Combinatorial Problem Arising in Machine Teaching](https://arxiv.org/abs/2402.04907) | 本文研究了机器教学中的一个组合问题，通过证明了一个最坏情况下的猜想，得出了关于教学维度的结果。该结果可以看作是解决了超立方体边界等周问题的定理的推广。 |
| [^158] | [$\texttt{NeRCC}$: Nested-Regression Coded Computing for Resilient Distributed Prediction Serving Systems](https://arxiv.org/abs/2402.04377) | NeRCC是一个通用的抗拖尾节点的近似编码计算框架，包括回归编码、计算和回归解码三个层次，通过优化两个正则化项的依赖关系来解决嵌套回归问题。 |
| [^159] | [Understanding the Effect of Noise in LLM Training Data with Algorithmic Chains of Thought](https://arxiv.org/abs/2402.04004) | 本论文研究了链式思维中的噪声对LLM训练数据的影响，并开发了追踪整数框架来生成可定制的噪声执行跟踪。通过评估预训练模型在算法可解任务中的表现，揭示了噪声的类型和强度对任务性能的影响。 |
| [^160] | [MolTC: Towards Molecular Relational Modeling In Language Models](https://arxiv.org/abs/2402.03781) | 本研究提出了一种基于语言模型的多模态框架MolTC，用于分子相互作用预测，该框架能够高效地整合分子对的丰富图形信息，并通过思维链理论实现统一的分子关系学习。 |
| [^161] | [Lens: A Foundation Model for Network Traffic](https://arxiv.org/abs/2402.03646) | "Lens"是一个基于T5架构的基础网络流量模型，通过学习大规模无标签数据的预训练表示，能够在流量理解和生成任务中取得精确的预测和生成。 |
| [^162] | [Learning Best-in-Class Policies for the Predict-then-Optimize Framework](https://arxiv.org/abs/2402.03256) | 我们提出了一种新颖的决策感知替代损失函数家族，用于predict-then-optimize框架，并且通过数值证据证实了其在误设置下的优越性。 |
| [^163] | [Efficient Numerical Wave Propagation Enhanced by an End-to-End Deep Learning Model](https://arxiv.org/abs/2402.02304) | 本文提出了一个由端到端深度学习模型加强的高效数值波传播方法，通过结合数值求解器和深度学习组件，优化算法架构、数据生成和并行时间算法，实现了在保持速度的同时显著提高性能。 |
| [^164] | [A Multi-Perspective Machine Learning Approach to Evaluate Police-Driver Interaction in Los Angeles](https://arxiv.org/abs/2402.01703) | 该研究提出了一种多角度的机器学习方法，用于分析洛杉矶警察与司机的互动。该方法利用多模态的数据包括音频、视频和文字信息，旨在提供对复杂和有争议的警民互动的分析工具。 |
| [^165] | [Timeseries Suppliers Allocation Risk Optimization via Deep Black Litterman Model](https://arxiv.org/abs/2401.17350) | 通过深度黑石贝莱曼模型和时空图神经网络，我们优化了供应商选择和订单分配，同时解决了零阶情况下的可信度问题，实现了准确的预测和精确的置信区间。 |
| [^166] | [Scaling Is All You Need: Autonomous Driving with JAX-Accelerated Reinforcement Learning](https://arxiv.org/abs/2312.15122) | 本研究提出了一种扩展的自动驾驶强化学习方法，在大规模实验中展示了随着规模增加，策略性能的改善。与现有机器学习自动驾驶策略相比，我们的最佳策略将故障率降低了64％，同时提高了25％的驾驶进展速度。 |
| [^167] | [Multimodal Attention Merging for Improved Speech Recognition and Audio Event Classification](https://arxiv.org/abs/2312.14378) | 多模态注意力合并（MAM）使用零-shot范式将文本和图像中的模型注意力矩阵的直接知识传输到音频领域中，可以显著降低自动语音识别的词错误率和音频事件分类的分类错误。 |
| [^168] | [Benchmarking Distribution Shift in Tabular Data with TableShift](https://arxiv.org/abs/2312.07577) | 这篇论文介绍了一个针对表格数据的分布漂移基准测试TableShift，包含15个二分类任务和相应的分布漂移，涵盖了多个领域，并且通过Python代码可以轻松访问。 |
| [^169] | [LayerCollapse: Adaptive compression of neural networks](https://arxiv.org/abs/2311.17943) | LayerCollapse是一种自适应压缩神经网络的方法，通过结构化剪枝来减少全连接层的深度，而不需要进行微调，并且对性能影响有限。该方法通过正则化激活函数的线性度来控制模型的表达能力。 |
| [^170] | [Eigenmatrix for unstructured sparse recovery](https://arxiv.org/abs/2311.16609) | 本文提出了一种名为特征矩阵的数据驱动构造，用于解决非结构稀疏恢复问题，对于样本值中的噪声和样本位置的非结构性质具有很好的适应性。 |
| [^171] | [Program Machine Policy: Addressing Long-Horizon Tasks by Integrating Program Synthesis and State Machines](https://arxiv.org/abs/2311.15960) | 这项工作提出了程序机器策略（POMP），在集成程序合成和状态机的基础上，解决了长期任务并表示复杂行为。 |
| [^172] | [Controllable Expensive Multi-objective Learning with Warm-starting Bayesian Optimization](https://arxiv.org/abs/2311.15297) | 这项工作提出了一种可控的Pareto Set Learning (PSL)方法，通过热启动贝叶斯优化和可控的帕累托集学习来解决现有PSL方法不稳定和低效的问题，并在合成和实际MOO问题上展示了其有效性。 |
| [^173] | [DroneOptiNet: A Framework for Optimal Drone-based Load Redistribution Mechanism for 5G and Beyond Solar Small Cell Networks](https://arxiv.org/abs/2311.12944) | 本研究提出了一种用于5G及其后太阳能小型蜂窝网络的最佳无人机负载重分配机制，通过使用无人机上的空中基站进行可靠安全的电力再分配，提高了网络的可靠性和稳健性。 |
| [^174] | [Fast multiplication by two's complement addition of numbers represented as a set of polynomial radix 2 indexes, stored as an integer list for massively parallel computation](https://arxiv.org/abs/2311.09922) | 本论文介绍了一种基于多项式基数2指数集合的快速乘法方法，在特定位数范围内比传统方法更快。该方法把数字表示为整数索引列表，并实现了分布式计算。 |
| [^175] | [Confident Naturalness Explanation (CNE): A Framework to Explain and Assess Patterns Forming Naturalness](https://arxiv.org/abs/2311.08936) | 本文提出了一个新的框架，称为自然度解释与评估模型（CNE）。该框架利用可解释的机器学习方法来分析卫星图像，以解释和评估形成自然度的模式，并带有相应的置信度。 |
| [^176] | [Fair Coresets via Optimal Transport](https://arxiv.org/abs/2311.05436) | 本研究提出了公平的Wasserstein核心集(FWC)，该方法通过最小化原始数据集与加权合成样本之间的Wasserstein距离，并强制实现人口平等，生成公平的合成代表性样本，可用于下游学习任务。 |
| [^177] | [Local Universal Explainer (LUX) -- a rule-based explainer with factual, counterfactual and visual explanations](https://arxiv.org/abs/2310.14894) | LUX是一种基于规则的解释器，可以生成事实、反事实和视觉解释，通过选择高密度簇形式的局部概念来形成决策边界。 |
| [^178] | [Efficient and Interpretable Bandit Algorithms](https://arxiv.org/abs/2310.14751) | 这个论文设计了一种高效且可解释的赌博算法，其中关注了模型参数的解释性和减小不确定性的目标。提出的CODE算法通过在符合约束条件的所有可能操作中进行探索，实现了最大程度的解释性和不确定性的减小。 |
| [^179] | [Model Selection of Zero-shot Anomaly Detectors in the Absence of Labeled Validation Data](https://arxiv.org/abs/2310.10461) | 本研究提出了一个通用框架SWSA（Selection With Synthetic Anomalies），用于在没有标签验证数据的情况下选择基于图像的零样本异常检测器。通过生成合成验证集，该方法能够实现模型选择，并在实证研究中展示了比基线方法更高的AUROC。 |
| [^180] | [Sorted LLaMA: Unlocking the Potential of Intermediate Layers of Large Language Models for Dynamic Inference](https://arxiv.org/abs/2309.08968) | Sorted LLaMA通过扩展SortedNet到生成NLP任务，使得大型语言模型在动态推理中更高效，并且不需要预训练，只需将标准微调替换为排序微调即可。该方法可以释放transformers中间层的潜力，同时最小化存储需求和过渡成本。 |
| [^181] | [Rethinking the Power of Graph Canonization in Graph Representation Learning with Stability](https://arxiv.org/abs/2309.00738) | 这篇论文提出了通过图规范化最大化GNNs表达能力的方法，并从模型稳定性角度研究了这些GNNs的能力。本文基于稳定GNN将相似的图映射到紧密相连的向量表示中，理论上揭示了图规范化增强的GNNs在表达能力和稳定性之间的折衷，提出了普遍图规范化的概念并给出了一种广泛适用的充分条件来解决普遍图规范化问题。实验证明了这种方法的有效性。 |
| [^182] | [CAMMARL: Conformal Action Modeling in Multi Agent Reinforcement Learning](https://arxiv.org/abs/2306.11128) | CAMMARL是一种新颖的多智能体强化学习算法，通过使用一致预测的概念对其他智能体的行动进行建模，并量化不确定性，提高了智能体的决策能力。 |
| [^183] | [Optimizing Floors in First Price Auctions: an Empirical Study of Yahoo Advertising](https://arxiv.org/abs/2302.06018) | 本研究提出了一个在一级定价拍卖中设定底价的模型，并应用于雅虎广告，从而使得在线发布者能够增加广告收入。 |
| [^184] | [Evaluation of Data Augmentation and Loss Functions in Semantic Image Segmentation for Drilling Tool Wear Detection](https://arxiv.org/abs/2302.05262) | 本研究评估了在钻孔工具磨损检测的语义图像分割中的数据增强和损失函数。结果发现，在适度增强的数据上训练的二元模型表现最佳。 |
| [^185] | [Robust variance-regularized risk minimization with concomitant scaling](https://arxiv.org/abs/2301.11584) | 本研究提出了一种简单但有效的学习过程，用于最小化潜在存在重尾风险的损失函数，该方法在各种数据集上表现出与使用CVaR或DRO风险等替代标准得到的最佳候选方案相当或更好的性能。 |
| [^186] | [Structure of Classifier Boundaries: Case Study for a Naive Bayes Classifier](https://arxiv.org/abs/2212.04382) | 本文研究了在图形输入空间中，分类器边界的结构。通过创建一种新的不确定性度量，称为邻居相似度，我们展示了朴素贝叶斯分类器的边界是巨大且复杂的结构。 |
| [^187] | [Locally Constrained Representations in Reinforcement Learning](https://arxiv.org/abs/2209.09441) | 本论文提出了一种在强化学习中使用局部约束表示的方法，通过辅助损失函数迫使状态表示与相邻状态的表示具有一定的可预测性，以更好地捕捉到环境的局部变化情况。 |
| [^188] | [Statistical exploration of the Manifold Hypothesis](https://arxiv.org/abs/2208.11665) | 这篇论文通过潜在度量模型从数据中得出了丰富而复杂的流形结构，并提供了解释流形假设的统计解释。该研究为发现和解释高维数据的几何结构以及探索数据生成机制提供了方法。 |
| [^189] | [On Rademacher Complexity-based Generalization Bounds for Deep Learning](https://arxiv.org/abs/2208.04284) | 该论文研究了基于Rademacher复杂度的方法在对卷积神经网络进行少类别图像分类时生成非空泛化界限。其中的关键技术贡献是发展了针对函数空间和具有一般Lipschitz激活函数的CNNs的新的Talagrand压缩引理。 |
| [^190] | [Parameter-free Mirror Descent](https://arxiv.org/abs/2203.00444) | 本论文针对无界域中构建自适应和无参的算法提出了一种修改后的在线镜像下降框架，并以此为基础开发了具有最优动态遗憾界限的无约束在线线性优化算法，并证明了基于Follow-the-Regularized-Leader的策略无法达到类似效果，此外还应用镜像下降框架构建了新的无参隐式更新以及简化和改进的无约束无标度算法。 |
| [^191] | [Fault-Tolerant Neural Networks from Biological Error Correction Codes](https://arxiv.org/abs/2202.12887) | 该论文根据哺乳动物皮质中的模拟纠错码，提出了一种基于生物纠错码的通用容错神经网络，实现了可靠计算；发现了从故障到容错神经计算的相变，为理解嘈杂模拟系统提供了新的途径。 |
| [^192] | [Toward More Generalized Malicious URL Detection Models](https://arxiv.org/abs/2202.10027) | 本文揭示了一个可能严重影响恶意URL检测机器学习模型性能的数据偏差问题，并提出了一种去偏置训练策略，通过自监督对抗训练技术来改善基于深度神经网络的模型的泛化能力。 |
| [^193] | [Universal Approximation Power of Deep Residual Neural Networks via Nonlinear Control Theory](https://arxiv.org/abs/2007.06007) | 本文通过非线性控制理论解释了深度残差神经网络的通用逼近能力，并提供了一个充分条件，在激活函数满足二次微分方程的情况下，一个足够深的神经网络能够在紧集合上逼近任意连续函数。 |
| [^194] | [Classify and Generate Reciprocally: Simultaneous Positive-Unlabelled Learning and Conditional Generation with Extra Data](https://arxiv.org/abs/2006.07841) | 本论文提出了一种同时利用正数据-无标签学习和有条件生成的训练框架，以及额外无标签数据的方法。通过使用一个对噪声标签具有鲁棒性的分类器噪声不变有条件生成对抗网络来提高PU分类器的性能，并利用PU分类器预测的标签和额外数据来帮助生成。实验结果证明了该方法的有效性。 |
| [^195] | [Adaptive Experiment Design with Synthetic Controls.](http://arxiv.org/abs/2401.17205) | 这种方法提出了Syntax，一个具有合成对照组的自适应实验设计，能够在多个亚群体中识别出具有正面治疗效果的亚群体，对于多样化患者反应的临床试验具有样本效率的优势。 |
| [^196] | [SliceGPT: Compress Large Language Models by Deleting Rows and Columns.](http://arxiv.org/abs/2401.15024) | SliceGPT是一种新的事后训练稀疏化方案，通过将每个权重矩阵替换为较小的矩阵以减小网络的维度，可以在保持高任务性能的同时减少模型参数。 |
| [^197] | [Prompt Design and Engineering: Introduction and Advanced Methods.](http://arxiv.org/abs/2401.14423) | 本文介绍了提示设计与工程的主要概念，并回顾了基本和更高级的方法。 |
| [^198] | [DALex: Lexicase-like Selection via Diverse Aggregation.](http://arxiv.org/abs/2401.12424) | 本文提出了一种新的选择算法DALex，它通过加权训练案例误差的和来选择最佳个体，相比标准的词法选择更快速。 |
| [^199] | [Data-Driven Target Localization: Benchmarking Gradient Descent Using the Cram\'er-Rao Bound.](http://arxiv.org/abs/2401.11176) | 本研究提出了一种数据驱动的神经网络方法，通过降低均方误差（MSE）实现了改进的目标方位和速度估计。这一发现强调了在雷达系统中采用深度学习方法的潜力，为在杂乱和动态环境中更准确的定位铺平了道路。 |
| [^200] | [Co-Pilot for Health: Personalized Algorithmic AI Nudging to Improve Health Outcomes.](http://arxiv.org/abs/2401.10816) | 该研究通过使用基于图神经网络的推荐系统和来自可穿戴设备的健康行为数据，设计并实施了一个人工智能驱动平台，实现了个性化和情境引导，能够提高参与者的日常活动水平和中等至剧烈运动时长。 |
| [^201] | [Efficient Fine-Tuning with Domain Adaptation for Privacy-Preserving Vision Transformer.](http://arxiv.org/abs/2401.05126) | 本论文提出了一种高效的领域适应方法，用于训练和测试隐私保护的视觉Transformer模型，并避免了使用加密图像导致的性能下降。实验结果表明，在图像分类任务上，该方法在CIFAR-10和ImageNet数据集上表现出更高的准确度。 |
| [^202] | [Unsupervised Test-Time Adaptation via Plug-and-Play Transformer Modules.](http://arxiv.org/abs/2401.04130) | 这项工作介绍了PLUTO:一种插拔式模块化的测试时领域适应策略，通过预先训练一系列针对不同源领域的模块，有效地创建了一个"模块存储库"。采用无监督的测试时自适应方法，从存储库中选择稀疏的相关模块的子集，并创建选中模块的加权组合，实现了对新领域的自适应。 |
| [^203] | [AST-T5: Structure-Aware Pretraining for Code Generation and Understanding.](http://arxiv.org/abs/2401.03003) | AST-T5是一种结构感知的预训练模型，通过利用抽象语法树（AST）来增强代码生成、转换和理解的能力。它优于其他同等大小的语言模型，并在代码到代码任务中表现出色。 |
| [^204] | [Self-Supervised Learning for Few-Shot Bird Sound Classification.](http://arxiv.org/abs/2312.15824) | 本研究展示了自监督学习在鸟鸣分类中的应用，通过无需标注的方式，从音频录音中获得有意义的鸟鸣表示，并展示了这些表示在少样本学习中的泛化能力。此外，使用预训练的音频神经网络选择高鸟活跃窗口进行自监督学习可以显著提升学习表示的质量。 |
| [^205] | [FairWASP: Fast and Optimal Fair Wasserstein Pre-processing.](http://arxiv.org/abs/2311.00109) | FairWASP是一种快速和最优的公平Wasserstein预处理方法，通过重新加权数据集来减少分类数据集中的不平等性，同时满足人口平等性准则。这种方法可以用于构建可以输入任何分类方法的数据集。 |
| [^206] | [Revisiting the Learnability of Apple Tasting.](http://arxiv.org/abs/2310.19064) | 该论文重新审视了苹果品尝的可学习性，从组合角度研究了在线可学习性。作者通过引入Effective width参数，紧密量化了在可实现设置中的极小期望错误，并在可实现设置中建立了极小期望错误数量的三分法。 |
| [^207] | [Davidsonian Scene Graph: Improving Reliability in Fine-grained Evaluation for Text-Image Generation.](http://arxiv.org/abs/2310.18235) | 本论文提出了Davidsonian场景图（DSG）的评估框架，解决了现有文本-图像生成模型评估中的可靠性挑战，包括QG问题的准确性和VQA答案的一致性。 |
| [^208] | [StochGradAdam: Accelerating Neural Networks Training with Stochastic Gradient Sampling.](http://arxiv.org/abs/2310.17042) | StochGradAdam是一种利用随机梯度抽样加速神经网络训练的优化器，通过选择性梯度考虑，能够稳定收敛，提升鲁棒训练。在图像分类和分割任务中表现优异。 |
| [^209] | [Redco: A Lightweight Tool to Automate Distributed Training of LLMs on Any GPU/TPUs.](http://arxiv.org/abs/2310.16355) | Redco是一个轻量级工具，旨在自动化分布式训练LLMs，并简化ML流程的开发。 |
| [^210] | [Deep Backtracking Counterfactuals for Causally Compliant Explanations.](http://arxiv.org/abs/2310.07665) | 本研究提供了一种实用方法，用于在深度生成组件的结构因果模型中计算回溯反事实。通过在因果模型的结构化潜在空间中解决优化问题，我们的方法能够生成反事实，并且与其他方法相比具备了多功能、模块化和符合因果关系的特点。 |
| [^211] | [LLark: A Multimodal Foundation Model for Music.](http://arxiv.org/abs/2310.07160) | LLark是一个通过多模态架构实现音乐理解的模型，能够在零样本泛化上匹配或超出现有基准模型，在字幕生成和推理任务中与人类响应高度一致。 |
| [^212] | [Boosting Facial Action Unit Detection Through Jointly Learning Facial Landmark Detection and Domain Separation and Reconstruction.](http://arxiv.org/abs/2310.05207) | 本文提出了一种新的面部动作单位（AU）检测框架，通过共享参数和引入多任务学习，在面部标志检测和AU域分离与重建之间实现了更好的性能。实验证明我们方法在野外AU检测方面优于现有方法。 |
| [^213] | [Large Language Model Cascades with Mixture of Thoughts Representations for Cost-efficient Reasoning.](http://arxiv.org/abs/2310.03094) | 本研究提出了一种基于思维混合表示的大规模语言模型级联方法，用于成本高效的推理。通过考虑更弱模型的答案一致性作为问题难度的信号，可以实现对问题的决策，从而节约使用更强模型的成本。 |
| [^214] | [Alphazero-like Tree-Search can Guide Large Language Model Decoding and Training.](http://arxiv.org/abs/2309.17179) | Alphazero类似的树搜索框架TS-LLM可以利用学习的价值函数指导大型语言模型的解码和训练，不仅适用于推理任务，还适用于其他任务，并且在不同大小的语言模型上具有普适性和可扩展性 |
| [^215] | [Insights Into the Inner Workings of Transformer Models for Protein Function Prediction.](http://arxiv.org/abs/2309.03631) | 本研究通过扩展可解释性人工智能方法，探索了Transformer模型在蛋白质功能预测中的内部运作，并成功识别出了与生物学和化学相关的序列部分，为蛋白质研究提供了重要线索。 |
| [^216] | [CktGNN: Circuit Graph Neural Network for Electronic Design Automation.](http://arxiv.org/abs/2308.16406) | 本文提出了一种名为CktGNN的电路图神经网络，通过识别电路的图形特性，同时自动化电路拓扑生成和器件尺寸调整。它使用两级GNN框架对电路图进行编码，并在设计效率上取得了显著的提升。 |
| [^217] | [SciBench: Evaluating College-Level Scientific Problem-Solving Abilities of Large Language Models.](http://arxiv.org/abs/2307.10635) | 这篇论文介绍了一个名为SciBench的基准套件，旨在对大型语言模型的大学水平科学问题解决能力进行评估。研究结果显示，当前的语言模型在提供复杂科学问题解决能力方面还有不足之处。 |
| [^218] | [Explainability is NOT a Game.](http://arxiv.org/abs/2307.07514) | Shapley values may provide misleading measures of relative feature importance in XAI, challenging their proposed uses in high-stakes application domains. |
| [^219] | [Self Expanding Neural Networks.](http://arxiv.org/abs/2307.04526) | 这项研究提出了一种自扩展神经网络的方法，通过自然梯度来自动增加神经网络的宽度和深度，以在训练损失降低的情况下提高性能，并在分类和回归问题中展示了其优势。 |
| [^220] | [A Combinatorial Characterization of Online Learning Games with Bounded Losses.](http://arxiv.org/abs/2307.03816) | 这项研究提出了一个新的尺度敏感的组合维度，称为顺序极小极大维度，并通过对有界损失的在线学习游戏进行研究，给出了对向量值回归和多标签分类的在线可学习性的紧密定量刻画。 |
| [^221] | [End-to-end Reinforcement Learning for Online Coverage Path Planning in Unknown Environments.](http://arxiv.org/abs/2306.16978) | 本文提出了一种基于端到端强化学习的在线覆盖路径规划方法，能处理未知环境并结合全局地图和局部感知输入，同时考虑长期路径规划和短期障碍物检测。 |
| [^222] | [Forecasting of the development of a partially-observed dynamical time series with the aid of time-invariance and linearity.](http://arxiv.org/abs/2306.16593) | 本研究提出了一种自回归松弛时间序列（ARS）模型，通过考虑动态系统的时间不变性和线性性，同时估计演化函数和缺失变量，用于预测动态时间序列中缺失变量的发展。 |
| [^223] | [Probabilistic matching of real and generated data statistics in generative adversarial networks.](http://arxiv.org/abs/2306.10943) | 本文提出一种通过向生成器损失函数中添加KL散度项的方法，来保证生成数据统计分布与真实数据的相应分布重合，并在实验中展示了此方法的优越性能。 |
| [^224] | [Sample-Efficient On-Policy Imitation Learning from Observations.](http://arxiv.org/abs/2306.09805) | 提出了一种称为SEILO的算法，该算法结合了标准的对抗模仿学习和逆动力学建模，实现了从无专家数据的观测中的样本高效策略模仿学习，成功地减少了与环境的交互并实现了专家水平的表现。 |
| [^225] | [Where Does My Model Underperform? A Human Evaluation of Slice Discovery Algorithms.](http://arxiv.org/abs/2306.08167) | 机器学习模型在语义连贯的数据子集上表现不佳仍然会出现问题，但是确定这些问题片段可能很困难，自动生成的片段并不是确定人工从业者问题性片段的银弹。 |
| [^226] | [Dynamic Inter-treatment Information Sharing for Heterogeneous Treatment Effects Estimation.](http://arxiv.org/abs/2305.15984) | 本论文提出了一种基于深度学习的HyperCATE框架，通过软权重共享的方式实现端到端信息共享来解决现有CATE学习器中的有偏估计问题，并在IHDP、ACIC-2016和Twins基准测试中评估了该框架的表现。 |
| [^227] | [Towards Convergence Rates for Parameter Estimation in Gaussian-gated Mixture of Experts.](http://arxiv.org/abs/2305.07572) | 本文提出新颖的Voronoi Loss函数来解决高斯门控混合专家模型参数估计的收敛速率问题，并在两种不同的门控网络下提供理论收敛速率的证明。 |
| [^228] | [A New Inexact Proximal Linear Algorithm with Adaptive Stopping Criteria for Robust Phase Retrieval.](http://arxiv.org/abs/2304.12522) | 本文提出了一种新的鲁棒相位恢复算法，通过使用自适应停止准则的非精确近端线性算法，该方法在实验中证明比现有方法更高效。 |
| [^229] | [Knowledge Distillation Under Ideal Joint Classifier Assumption.](http://arxiv.org/abs/2304.11004) | 本文提出了基于理想联合分类器假设的知识蒸馏框架，可以提供清晰全面的理解和为未来研究提供理论基础，使得教师和学生网络之间的知识传递更加高效。 |
| [^230] | [Depth Functions for Partial Orders with a Descriptive Analysis of Machine Learning Algorithms.](http://arxiv.org/abs/2304.09872) | 本文提出了一种基于深度函数的偏序集合描述性分析框架，并引入了偏序版本的单纯深度，用于比较基于多维性能度量的机器学习算法。实验证明此方法与现有基准方法不同，为分类器比较提供了新的视角。 |
| [^231] | [High-fidelity Pseudo-labels for Boosting Weakly-Supervised Segmentation.](http://arxiv.org/abs/2304.02621) | 本文提出了一种使用马尔可夫随机场增强弱监督分割中的高保真伪标签生成方法，能够生成更准确的伪标签，并使用新的训练策略来实现更好的收敛。实验结果表明该方法达到了弱监督分割方法的最佳性能。 |
| [^232] | [Fair Evaluation of Graph Markov Neural Networks.](http://arxiv.org/abs/2304.01235) | 本论文通过引入适用于GMNN的新测试方法，对三类不同信息源对GMNN在WikiVitals数据集中的预测准确性的贡献进行严格评估，结果表明标签相关性是帮助GMNN获得优势的关键信息源。 |
| [^233] | [A Novel Neural Network Approach for Predicting the Arrival Time of Buses for Smart On-Demand Public Transit.](http://arxiv.org/abs/2303.15495) | 本文介绍了一种基于神经网络的数据驱动方法，可以跨所有公交线路集体预测公交车到达每个交通点的时间，解决公交运输中公交车到达时间不准确和可靠的问题。 |
| [^234] | [Predicting discrete-time bifurcations with deep learning.](http://arxiv.org/abs/2303.09669) | 本研究利用深度学习训练分类器预测离散时间五种本地分岔，在经济、生态、生理学等方面的试验数据中都具有优秀表现，是提前警告关键转变的重要方法。 |
| [^235] | [Spacetime-Efficient Low-Depth Quantum State Preparation with Applications.](http://arxiv.org/abs/2303.02131) | 提出了一种使用占用空间和时间较小的低深度方法来准备任意量子态，能够在较少的量子资源使用下实现更快的准备速度。 |
| [^236] | [Learning Interpretable Low-dimensional Representation via Physical Symmetry.](http://arxiv.org/abs/2302.10890) | 通过使用物理对称性作为潜在空间的自洽约束条件，该研究展示了在音乐领域和计算机视觉领域，模型可以以无监督的方式学习出可解释的低维表示，例如线性音高和三维笛卡尔因素。 |
| [^237] | [A Primal-Dual Algorithm for Hybrid Federated Learning.](http://arxiv.org/abs/2210.08106) | 该论文提出了一种基于Fenchel对偶性的快速、稳健的混合联邦学习算法。实验证明了该算法相对于传统的FedAvg方法的性能改进，并提供了隐私保护措施。 |
| [^238] | [A Link between Coding Theory and Cross-Validation with Applications.](http://arxiv.org/abs/2103.11856) | 本研究研究了编码理论与交叉验证之间的联系，并发现了学习算法在固定数据上能解决不同二进制分类问题的数量与误差检测码理论密切相关。我们还对一种特定的交叉验证方法下的最大分类问题数量进行了研究，这取决于常权码的码字数量。同时，我们推广了常权码的概念，并证明了类似的结果适用于其他交叉验证错误和轻量级常权码。 |

# 详细

[^1]: 单调、Bi-Lipschitz和Polyak-\L{}ojasiewicz网络

    Monotone, Bi-Lipschitz, and Polyak-\L{}ojasiewicz Networks

    [https://rss.arxiv.org/abs/2402.01344](https://rss.arxiv.org/abs/2402.01344)

    这篇论文介绍了一种新的可逆神经网络BiLipNet，它具有调控输出敏感性和输入可区分性的能力。其中的主要创新是通过认证的强单调性和Lipschitz性的可逆残差层，与正交层组合构建了Bi-Lipschitz网络。另外，该论文还提出了满足Polyak-\L{}ojasiewicz条件的PLNet，并介绍了其应用于学习非凸代理损失的优势特性。

    

    本文介绍了一种新的BiLipNet，这是一种可逆的\emph{Bi-Lipschitz}神经网络，具有控制其\emph{Lipschitzness}（对输入扰动的输出敏感性）和\emph{inverse Lipschitzness}（不同输出的输入可区分性）的能力。主要贡献是一个新颖的可逆残差层，具有认证的强单调性和Lipschitz性，我们将其与正交层组合以构建Bi-Lipschitz网络。认证是基于增量二次约束的，与谱归一化相比，它能实现更紧密的界限。此外，我们将模型的反向计算形式化为三算子分裂问题，已知存在快速算法。基于所提出的Bi-Lipschitz网络，我们引入了一种新的标量输出网络，即PLNet，它满足Polyak-\L{}ojasiewicz条件。它可以用于学习具有有利特性的非凸代理损失，例如独特性和高效计算性。

    This paper presents a new \emph{bi-Lipschitz} invertible neural network, the BiLipNet, which has the ability to control both its \emph{Lipschitzness} (output sensitivity to input perturbations) and \emph{inverse Lipschitzness} (input distinguishability from different outputs). The main contribution is a novel invertible residual layer with certified strong monotonicity and Lipschitzness, which we compose with orthogonal layers to build bi-Lipschitz networks. The certification is based on incremental quadratic constraints, which achieves much tighter bounds compared to spectral normalization. Moreover, we formulate the model inverse calculation as a three-operator splitting problem, for which fast algorithms are known. Based on the proposed bi-Lipschitz network, we introduce a new scalar-output network, the PLNet, which satisfies the Polyak-\L{}ojasiewicz condition. It can be applied to learn non-convex surrogate losses with favourable properties, e.g., a unique and efficiently-computab
    
[^2]: 关于大规模自监督学习在少样本音频分类中的可迁移性

    On the Transferability of Large-Scale Self-Supervision to Few-Shot Audio Classification

    [https://rss.arxiv.org/abs/2402.01274](https://rss.arxiv.org/abs/2402.01274)

    本研究评估了大规模自监督模型在少样本音频分类中的性能，并发现在一些少样本问题中取得了最先进的性能，同时发现语音为基础的少样本问题与多个下游音频任务之间存在较强的相关性。

    

    近年来，自监督学习因其能够从无标签数据中学习到稳健的特征表示而表现出色。经过自监督预训练的网络可作为下游任务（包括少样本学习）中有效的特征提取器。尽管对于图像的无监督学习方法在少样本学习中的评估已经有了良好的基础，但在声学领域却明显缺失。本研究通过评估大规模自监督模型在少样本音频分类中的性能，弥补了这一空白。此外，我们还探讨了模型的少样本学习能力与其他下游任务基准之间的关系。我们的研究结果表明，在一些少样本问题（如SpeechCommandsv2）中，我们取得了最先进的性能，并且语音为基础的少样本问题与多个下游音频任务之间存在着较强的相关性。

    In recent years, self-supervised learning has excelled for its capacity to learn robust feature representations from unlabelled data. Networks pretrained through self-supervision serve as effective feature extractors for downstream tasks, including Few-Shot Learning. While the evaluation of unsupervised approaches for few-shot learning is well-established in imagery, it is notably absent in acoustics. This study addresses this gap by assessing large-scale self-supervised models' performance in few-shot audio classification. Additionally, we explore the relationship between a model's few-shot learning capability and other downstream task benchmarks. Our findings reveal state-of-the-art performance in some few-shot problems such as SpeechCommandsv2, as well as strong correlations between speech-based few-shot problems and various downstream audio tasks.
    
[^3]: 与语言模型的反馈循环推动上下文内奖励欺骗

    Feedback Loops With Language Models Drive In-Context Reward Hacking

    [https://arxiv.org/abs/2402.06627](https://arxiv.org/abs/2402.06627)

    与语言模型的反馈循环可能导致上下文内奖励欺骗（ICRH），即语言模型在测试时在优化目标的同时却产生负面副作用。这项研究确定了两个导致ICRH的过程：输出优化和策略优化。

    

    语言模型对外部世界产生影响：它们查询可以读写网页的API，生成能够影响人类行为的内容，以及作为自主代理运行系统命令。这些互动形成了反馈循环：语言模型的输出影响世界，反过来又影响后续的语言模型输出。在这项工作中，我们展示了反馈循环可能导致上下文内奖励欺骗(ICRH)，即测试时的语言模型在优化（可能隐含的）目标的同时，产生负面副作用。例如，考虑一个被部署用于增加Twitter参与度的语言模型代理；语言模型可能在上下文窗口中检索其以前的推文，并使推文更具争议性，从而增加参与度，但也增加了有毒性。我们确定并研究了导致ICRH的两个过程：输出优化和策略优化。对于这些过程，静态数据集上的评估是不足够的-他们无法捕捉到反馈效应，也不能捕捉到最有害的行为。为此，我们提供了...

    Language models influence the external world: they query APIs that read and write to web pages, generate content that shapes human behavior, and run system commands as autonomous agents. These interactions form feedback loops: LLM outputs affect the world, which in turn affect subsequent LLM outputs. In this work, we show that feedback loops can cause in-context reward hacking (ICRH), where the LLM at test-time optimizes a (potentially implicit) objective but creates negative side effects in the process. For example, consider an LLM agent deployed to increase Twitter engagement; the LLM may retrieve its previous tweets into the context window and make them more controversial, increasing engagement but also toxicity. We identify and study two processes that lead to ICRH: output-refinement and policy-refinement. For these processes, evaluations on static datasets are insufficient -- they miss the feedback effects and thus cannot capture the most harmful behavior. In response, we provide 
    
[^4]: 动力系统中顺序预测的复杂性研究

    The Complexity of Sequential Prediction in Dynamical Systems

    [https://arxiv.org/abs/2402.06614](https://arxiv.org/abs/2402.06614)

    通过学习理论的角度，我们在没有参数假设的情况下，研究了在底层演化函数未知的动力系统中学习预测下一状态的问题，并提出了新的组合度量和维度来量化在可实现和不可知情况下的最佳错误和遗憾界限。

    

    我们研究了在底层演化函数未知的情况下学习预测动力系统下一状态的问题。与以前的工作不同，我们对动力系统没有参数假设，并从学习理论的角度研究了该问题。我们定义了新的组合度量和维度，并证明它们量化了在可实现和不可知情况下的最佳错误和遗憾界限。

    We study the problem of learning to predict the next state of a dynamical system when the underlying evolution function is unknown. Unlike previous work, we place no parametric assumptions on the dynamical system, and study the problem from a learning theory perspective. We define new combinatorial measures and dimensions and show that they quantify the optimal mistake and regret bounds in the realizable and agnostic setting respectively.
    
[^5]: RQP-SGD：通过嘈杂的随机梯度下降和随机量化实现差分隐私的机器学习

    RQP-SGD: Differential Private Machine Learning through Noisy SGD and Randomized Quantization

    [https://arxiv.org/abs/2402.06606](https://arxiv.org/abs/2402.06606)

    RQP-SGD是一种结合了差分隐私随机梯度下降和随机量化的新方法，用于在边缘部署的低内存机器学习模型训练中实现隐私保护。通过研究其在具有凸目标和量化约束的ML任务上的效用收敛性，并证明了其相对确定性量化的有效性。

    

    物联网设备的兴起促使了对在边缘部署实时、高效、安全数据处理的机器学习的需求。在这种情况下，使用实值权重参数实现机器学习模型可能在大型模型上变得不切实际，因此有必要使用量化离散权重来训练模型。同时，这些低维模型也需要保护底层数据集的隐私。在这项工作中，我们提出了RQP-SGD，一种用于低内存边缘机器学习模型训练的隐私保护量化的新方法。该方法将差分隐私随机梯度下降（DP-SGD）与随机量化相结合，在机器学习中提供了可衡量的隐私保证。特别地，我们研究了在具有凸目标和量化约束的ML任务上实施RQP-SGD的效用收敛性，并证明其相对确定性量化的功效。

    The rise of IoT devices has prompted the demand for deploying machine learning at-the-edge with real-time, efficient, and secure data processing. In this context, implementing machine learning (ML) models with real-valued weight parameters can prove to be impractical particularly for large models, and there is a need to train models with quantized discrete weights. At the same time, these low-dimensional models also need to preserve privacy of the underlying dataset. In this work, we present RQP-SGD, a new approach for privacy-preserving quantization to train machine learning models for low-memory ML-at-the-edge. This approach combines differentially private stochastic gradient descent (DP-SGD) with randomized quantization, providing a measurable privacy guarantee in machine learning. In particular, we study the utility convergence of implementing RQP-SGD on ML tasks with convex objectives and quantization constraints and demonstrate its efficacy over deterministic quantization. Throug
    
[^6]: 预测性表征：智能的基石

    Predictive representations: building blocks of intelligence

    [https://arxiv.org/abs/2402.06590](https://arxiv.org/abs/2402.06590)

    预测性表征可能是智能的多功能基石。

    

    适应性行为通常需要预测未来事件。强化学习理论规定了什么样的预测性表征是有用的以及如何计算它们。本文将这些理论观点与认知和神经科学的研究结合起来。我们特别关注继任者表征（SR）及其广义形式，它们不仅被广泛应用于工程工具，也作为大脑功能的模型。这种融合表明特定类型的预测性表征可能是智能的多功能基石。

    Adaptive behavior often requires predicting future events. The theory of reinforcement learning prescribes what kinds of predictive representations are useful and how to compute them. This paper integrates these theoretical ideas with work on cognition and neuroscience. We pay special attention to the successor representation (SR) and its generalizations, which have been widely applied both as engineering tools and models of brain function. This convergence suggests that particular kinds of predictive representations may function as versatile building blocks of intelligence.
    
[^7]: 超越零件之和：集成主干网络进行少样本分割

    More than the Sum of Its Parts: Ensembling Backbone Networks for Few-Shot Segmentation

    [https://arxiv.org/abs/2402.06581](https://arxiv.org/abs/2402.06581)

    本论文研究了在少样本分割中，通过集成不同主干网络的特征能否提高模型的性能。作者通过提出独立投票和特征融合两种集成技术，并在PANet上实现了这些技术。实验结果表明集成主干网络可以捕捉更丰富的视觉特征，从而提升分割效果。

    

    语义分割是在人工智能和机器人技术中实现鲁棒图像理解的关键前提。尤其是在训练样本有限的挑战性条件下，少样本分割是传统分割方法的扩展和优化。在少样本分割中，主要的方法是依靠单一的主干网络进行视觉特征提取。选择使用哪个主干网络是影响整体性能的决定因素。在这项工作中，我们探究了从不同主干网络融合特征是否能够提高少样本分割模型捕捉更丰富的视觉特征的能力。为了解决这个问题，我们提出并比较了两种集成技术——独立投票和特征融合。在现有的少样本分割方法中，我们在PANet上实现了提出的集成技术。在PANet中，用于预测分割掩码的模块避免了可训练参数。

    Semantic segmentation is a key prerequisite to robust image understanding for applications in \acrlong{ai} and Robotics. \acrlong{fss}, in particular, concerns the extension and optimization of traditional segmentation methods in challenging conditions where limited training examples are available. A predominant approach in \acrlong{fss} is to rely on a single backbone for visual feature extraction. Choosing which backbone to leverage is a deciding factor contributing to the overall performance. In this work, we interrogate on whether fusing features from different backbones can improve the ability of \acrlong{fss} models to capture richer visual features. To tackle this question, we propose and compare two ensembling techniques-Independent Voting and Feature Fusion. Among the available \acrlong{fss} methods, we implement the proposed ensembling techniques on PANet. The module dedicated to predicting segmentation masks from the backbone embeddings in PANet avoids trainable parameters, 
    
[^8]: SAE: 单一架构集合神经网络

    SAE: Single Architecture Ensemble Neural Networks

    [https://arxiv.org/abs/2402.06580](https://arxiv.org/abs/2402.06580)

    SAE是一种单一架构集合神经网络方法，通过学习集合输入的最佳退出数量和深度，在任务上显示出优越的准确性和置信度校准。它能够根据特定架构或应用程序灵活地定制其配置。

    

    单一神经网络架构的集合能够在任务上显示出优越的准确性和置信度校准。最近的方法通过提前退出或多输入多输出框架将集合压缩到单一网络中。然而，这些方法的景观迄今为止是零散的，因此很难选择适合特定任务的方法。此外，这些方法的算法性能落后于独立神经网络的集合，并需要广泛的架构调整。我们提出了一种新的方法，将这些方法统一到单一架构集合中。我们的方法在单一神经网络中学习集合输入的最佳退出数量和深度。这使得SAE框架可以根据特定架构或应用程序灵活地定制其配置。我们评估了在各种网络架构类型和大小上进行图像分类和回归的SAE。我们展示了与基线相当的准确性或置信度校准。

    Ensembles of separate neural networks (NNs) have shown superior accuracy and confidence calibration over single NN across tasks. Recent methods compress ensembles within a single network via early exits or multi-input multi-output frameworks. However, the landscape of these methods is fragmented thus far, making it difficult to choose the right approach for a given task. Furthermore, the algorithmic performance of these methods is behind the ensemble of separate NNs and requires extensive architecture tuning. We propose a novel methodology unifying these approaches into a Single Architecture Ensemble (SAE). Our method learns the optimal number and depth of exits per ensemble input in a single NN. This enables the SAE framework to flexibly tailor its configuration for a given architecture or application. We evaluate SAEs on image classification and regression across various network architecture types and sizes. We demonstrate competitive accuracy or confidence calibration to baselines w
    
[^9]: 关于基于耦合的标准化流的普适性

    On the Universality of Coupling-based Normalizing Flows

    [https://arxiv.org/abs/2402.06578](https://arxiv.org/abs/2402.06578)

    我们提出了一个新颖的理论框架，用于理解基于耦合的标准化流的表达能力，并提出了一个新的分布普适性定理来克服以前工作的限制。这些结果支持耦合架构的表达能力，并弥补了实证结果和理论理解之间的差距。

    

    我们提出了一个新颖的理论框架，用于理解基于耦合的标准化流（如RealNVP）的表达能力。尽管耦合流在科学应用中很普遍，但由于其受限的架构，对于耦合流的全面理解仍然困难。现有的定理在实际应用中存在限制，因为它们需要使用任意病态的神经网络。此外，我们还证明了这些结构本质上导致体积保持流，这是一个限制表达能力的基本约束。我们提出了一种新的基于分布的耦合标准化流普适性定理，克服了以前工作的几个限制。我们的结果支持耦合架构具有表达能力的普遍经验，并为选择耦合函数的表达能力提供了细致入微的观点，填补了实证结果和理论理解之间的差距。

    We present a novel theoretical framework for understanding the expressive power of coupling-based normalizing flows such as RealNVP. Despite their prevalence in scientific applications, a comprehensive understanding of coupling flows remains elusive due to their restricted architectures. Existing theorems fall short as they require the use of arbitrarily ill-conditioned neural networks, limiting practical applicability. Additionally, we demonstrate that these constructions inherently lead to volume-preserving flows, a property which we show to be a fundamental constraint for expressivity. We propose a new distributional universality theorem for coupling-based normalizing flows, which overcomes several limitations of prior work. Our results support the general wisdom that the coupling architecture is expressive and provide a nuanced view for choosing the expressivity of coupling functions, bridging a gap between empirical results and theoretical understanding.
    
[^10]: 通过精简形态条件超网络实现高效的通用形态控制

    Distilling Morphology-Conditioned Hypernetworks for Efficient Universal Morphology Control

    [https://arxiv.org/abs/2402.06570](https://arxiv.org/abs/2402.06570)

    本论文提出了一种名为HyperDistill的方法，通过精简形态条件超网络，可在训练和未知测试机器人上实现与通用的transformers策略相当的性能，同时大大减小模型尺寸和计算成本。

    

    在不同机器人形态之间学习一个通用策略可以显著提高学习效率，并实现对未知形态的零样本泛化。然而，学习一个高性能的通用策略需要像transformers（TF）这样具有较大内存和计算成本的复杂架构，而比较简单的多层感知器（MLP）则具有更高的效率。为了在推理时既能达到像TF一样好的性能，又能具有像MLP一样的高效率，我们提出了HyperDistill。它包括：（1）一个形态条件的超网络（HN），用于生成机器人特定的MLP策略，和（2）一个对于成功训练至关重要的策略蒸馏方法。我们展示了在UNIMAL上，一个包含数百种不同形态的基准测试中，HyperDistill在训练和未知测试机器人上都能和通用的TF教师策略一样表现出色，同时将模型尺寸减小了6-14倍，计算成本在不同环境下减小了67-160倍。

    Learning a universal policy across different robot morphologies can significantly improve learning efficiency and enable zero-shot generalization to unseen morphologies. However, learning a highly performant universal policy requires sophisticated architectures like transformers (TF) that have larger memory and computational cost than simpler multi-layer perceptrons (MLP). To achieve both good performance like TF and high efficiency like MLP at inference time, we propose HyperDistill, which consists of: (1) A morphology-conditioned hypernetwork (HN) that generates robot-wise MLP policies, and (2) A policy distillation approach that is essential for successful training. We show that on UNIMAL, a benchmark with hundreds of diverse morphologies, HyperDistill performs as well as a universal TF teacher policy on both training and unseen test robots, but reduces model size by 6-14 times, and computational cost by 67-160 times in different environments. Our analysis attributes the efficiency 
    
[^11]: 医学的暗物质中隐藏着什么？在医疗实践中处理丢失数据的学习

    What is Hiding in Medicine's Dark Matter? Learning with Missing Data in Medical Practices

    [https://arxiv.org/abs/2402.06563](https://arxiv.org/abs/2402.06563)

    本研究使用统计方法和机器学习，通过分析儿科急诊数据和创伤伤害数据库，揭示了医疗实践模式与丢失数据之间的关联，并提出了临床数据插补的方法。这对于减少分析偏见、提高临床决策的有效性非常重要。

    

    电子病人记录（EPR）产生了大量数据，但其中包含重要的缺失信息。理解和处理这些缺失数据是临床数据分析的重要组成部分，如果不加以解决，可能导致分析中的偏见和关键结论的扭曲。缺失数据可能与医疗专业人士的实践模式有关，对缺失数据的插补可以提高临床决策的有效性。本研究专注于统计方法来理解和解释缺失数据，并使用单一中心的儿科急诊数据以及英国最大的创伤伤害数据库（TARN）中的数据，进行基于机器学习的临床数据插补。在对56,961个与儿童急诊部就诊相关的初步生命体征和观察数据进行的研究中，我们表明丢失数据很可能是非随机的，并展示了这些数据与医疗专业人士的实践模式的关联。

    Electronic patient records (EPRs) produce a wealth of data but contain significant missing information. Understanding and handling this missing data is an important part of clinical data analysis and if left unaddressed could result in bias in analysis and distortion in critical conclusions. Missing data may be linked to health care professional practice patterns and imputation of missing data can increase the validity of clinical decisions. This study focuses on statistical approaches for understanding and interpreting the missing data and machine learning based clinical data imputation using a single centre's paediatric emergency data and the data from UK's largest clinical audit for traumatic injury database (TARN). In the study of 56,961 data points related to initial vital signs and observations taken on children presenting to an Emergency Department, we have shown that missing data are likely to be non-random and how these are linked to health care professional practice patterns.
    
[^12]: 视频标注器：利用视觉语言模型和主动学习构建视频分类器的高效框架

    Video Annotator: A framework for efficiently building video classifiers using vision-language models and active learning

    [https://arxiv.org/abs/2402.06560](https://arxiv.org/abs/2402.06560)

    本论文提出了一个名为视频标注器（VA）的框架，通过利用视觉语言模型和主动学习构建视频分类器，并通过人在循环系统实现了领域专家的直接参与，解决了传统数据注释方法的资源消耗和效率低下的问题。

    

    高质量和一致的注释对于成功开发稳健的机器学习模型至关重要。传统的数据注释方法耗费资源且效率低下，常常依赖于非领域专家的第三方注释者。对于模型训练而言，通常最具信息量的困难样本往往很难在没有业务背景的情况下进行准确和一致的标注。这些困难样本在注释过程中可能无法预测地出现，需要进行可变次数的迭代和反馈循环，从而导致意想不到的费用和时间成本以保证质量。我们认为，更直接地通过领域专家的参与，使用人在循环系统，可以解决这些实践中的挑战。我们提出了一个新颖的框架，称为视频标注器（VA），用于注释、管理和迭代视频分类数据集。我们的方法为以最终用户为中心的模型开发过程提供了一种新的范式。

    High-quality and consistent annotations are fundamental to the successful development of robust machine learning models. Traditional data annotation methods are resource-intensive and inefficient, often leading to a reliance on third-party annotators who are not the domain experts. Hard samples, which are usually the most informative for model training, tend to be difficult to label accurately and consistently without business context. These can arise unpredictably during the annotation process, requiring a variable number of iterations and rounds of feedback, leading to unforeseen expenses and time commitments to guarantee quality.   We posit that more direct involvement of domain experts, using a human-in-the-loop system, can resolve many of these practical challenges. We propose a novel framework we call Video Annotator (VA) for annotating, managing, and iterating on video classification datasets. Our approach offers a new paradigm for an end-user-centered model development process,
    
[^13]: Diffusion-ES:基于扩散的零梯度规划用于自动驾驶和零阶指令跟随

    Diffusion-ES: Gradient-free Planning with Diffusion for Autonomous Driving and Zero-Shot Instruction Following

    [https://arxiv.org/abs/2402.06559](https://arxiv.org/abs/2402.06559)

    本文提出了一种Diffusion-ES方法，它结合了无梯度优化和轨迹去噪技术，用于优化黑盒非可微目标。该方法通过从扩散模型中采样轨迹，并使用黑盒奖励函数对其进行评分，实现了更高的多样性和可解释性。

    

    扩散模型在决策和控制中对复杂和多模态轨迹分布建模有很强优势。最近提出了奖励梯度引导去噪方法，用于产生在扩散模型所捕获的数据分布下，同时最大化可微分奖励函数和似然性的轨迹。奖励梯度引导去噪需要一个适合于清洁和噪声样本的可微分奖励函数，从而限制了其作为一种通用轨迹优化器的适用性。在本文中，我们提出了DiffusionES，一种将无梯度优化和轨迹去噪相结合的方法，用于在数据流形中优化黑盒非可微目标。Diffusion-ES从扩散模型中采样轨迹，并使用黑盒奖励函数对其进行评分。它通过截断扩散过程对得分高的轨迹进行变异，该过程应用少量的噪声和去噪步骤，从而实现了更高的多样性和更好的可解释性。

    Diffusion models excel at modeling complex and multimodal trajectory distributions for decision-making and control. Reward-gradient guided denoising has been recently proposed to generate trajectories that maximize both a differentiable reward function and the likelihood under the data distribution captured by a diffusion model. Reward-gradient guided denoising requires a differentiable reward function fitted to both clean and noised samples, limiting its applicability as a general trajectory optimizer. In this paper, we propose DiffusionES, a method that combines gradient-free optimization with trajectory denoising to optimize black-box non-differentiable objectives while staying in the data manifold. Diffusion-ES samples trajectories during evolutionary search from a diffusion model and scores them using a black-box reward function. It mutates high-scoring trajectories using a truncated diffusion process that applies a small number of noising and denoising steps, allowing for much mo
    
[^14]: 使用图神经网络的强化学习进行欺骗性路径规划

    Deceptive Path Planning via Reinforcement Learning with Graph Neural Networks

    [https://arxiv.org/abs/2402.06552](https://arxiv.org/abs/2402.06552)

    本文提出了一种使用图神经网络的强化学习方法来进行欺骗性路径规划，克服了现有方法的局限性，并且具有普适性和实时适应性。

    

    欺骗性路径规划(DPP)是指设计一条路径，以隐藏其真实目标以免被外部观察者发现。现有的DPP方法依赖于不切实际的假设，如全局状态可观察性和完美的模型知识，并且通常是针对特定问题的，这意味着即使对先前解决的问题进行微小更改也会迫使重新计算整个新解。鉴于这些缺点，这些方法不能泛化到未见过的问题实例，缺乏适应现实问题规模的可扩展性，以及无法调整欺骗程度和实时适应环境变化。在本文中，我们提出了一种基于强化学习(RL)的方案，用于训练策略以在任意加权图上执行DPP，并克服了这些问题。我们方法的核心是引入了一个局部感知模型，一种新的状态空间表示，概括了DPP问题的关键组成部分，并使用了图神经网络。

    Deceptive path planning (DPP) is the problem of designing a path that hides its true goal from an outside observer. Existing methods for DPP rely on unrealistic assumptions, such as global state observability and perfect model knowledge, and are typically problem-specific, meaning that even minor changes to a previously solved problem can force expensive computation of an entirely new solution. Given these drawbacks, such methods do not generalize to unseen problem instances, lack scalability to realistic problem sizes, and preclude both on-the-fly tunability of deception levels and real-time adaptivity to changing environments. In this paper, we propose a reinforcement learning (RL)-based scheme for training policies to perform DPP over arbitrary weighted graphs that overcomes these issues. The core of our approach is the introduction of a local perception model for the agent, a new state space representation distilling the key components of the DPP problem, the use of graph neural ne
    
[^15]: 从大型语言模型中校准长篇生成

    Calibrating Long-form Generations from Large Language Models

    [https://arxiv.org/abs/2402.06544](https://arxiv.org/abs/2402.06544)

    该论文提出了一个统一的校准框架，用于校准大型语言模型的长篇生成。在该框架中，作者开发了三个度量指标用于评估模型的校准性，并提出了两种置信度引导方法。实验证明，更大的模型不一定能保证更好的校准。

    

    为了提高大型语言模型（LLMs）的可靠性，校准是必要的 - 模型的评估置信度应该与其响应正确性的实际可能性相一致。然而，目前的置信度引导方法和校准指标通常依赖于对响应正确性的二元真/假评估。这种方法在长篇生成中不适用，因为答案可能部分正确。为了解决这一问题，我们引入了一个统一的校准框架，其中LLMs的响应正确性和关联的置信水平都被视为一系列分数的分布。在此框架下，我们开发了三个度量指标来精确评估LLM的校准，并进一步提出了基于自一致性和自评估的两种置信度引导方法。我们的实验包括长篇问答和摘要任务，结果表明，更大的模型不一定能保证更好的校准。

    To enhance Large Language Models' (LLMs) reliability, calibration is essential -- the model's assessed confidence scores should align with the actual likelihood of its responses being correct. However, current confidence elicitation methods and calibration metrics typically rely on a binary true/false assessment of response correctness. This approach does not apply to long-form generation, where an answer can be partially correct. Addressing this gap, we introduce a unified calibration framework, in which both the correctness of the LLMs' responses and their associated confidence levels are treated as distributions across a range of scores. Within this framework, we develop three metrics to precisely evaluate LLM calibration and further propose two confidence elicitation methods based on self-consistency and self-evaluation. Our experiments, which include long-form QA and summarization tasks, demonstrate that larger models don't necessarily guarantee better calibration, that calibratio
    
[^16]: Bandit Convex Optimisation（强盗凸优化）

    Bandit Convex Optimisation

    [https://arxiv.org/abs/2402.06535](https://arxiv.org/abs/2402.06535)

    这篇论文介绍了强盗凸优化的基本框架和用于解决这一问题的多种工具。虽然没有太多创新，但通过以新颖的方式应用现有工具，获得了新的算法和改进了一些界限。

    

    强盗凸优化是研究零阶凸优化的基本框架。本文介绍了用于解决该问题的许多工具，包括切平面方法、内点方法、连续指数权重、梯度下降和在线牛顿步骤。解释了许多假设和设置之间的细微差别。尽管在这里没有太多真正新的东西，但一些现有工具以新颖的方式应用于获得新算法。一些界限稍微改进了一些。

    Bandit convex optimisation is a fundamental framework for studying zeroth-order convex optimisation. These notes cover the many tools used for this problem, including cutting plane methods, interior point methods, continuous exponential weights, gradient descent and online Newton step. The nuances between the many assumptions and setups are explained. Although there is not much truly new here, some existing tools are applied in novel ways to obtain new algorithms. A few bounds are improved in minor ways.
    
[^17]: 生成对抗贝叶斯优化用于代理目标

    Generative Adversarial Bayesian Optimization for Surrogate Objectives

    [https://arxiv.org/abs/2402.06532](https://arxiv.org/abs/2402.06532)

    提出了生成对抗贝叶斯优化（GABO）算法，通过使用自适应源批评家正则化，将优化轨迹限制在代理函数可靠的区域内，解决了离线模型基于策略优化中代理模型预测不准确的问题。在多个离线优化任务中，GABO表现优于现有基准方法。

    

    离线基于模型的策略优化通过在优化过程中不查询真实的目标函数来优化学习到的代理目标函数。然而，在优化过程中经常遇到代理模型预测不准确的情况。为了解决这个问题，我们提出了使用自适应源批评家正则化的生成对抗贝叶斯优化（GABO），这是一个任务不可知的贝叶斯优化框架，采用了Lipschitz有界源批评家模型来约束优化轨迹，使其在代理函数可靠的区域内。我们证明，在连续输入空间先验的一定假设下，我们的算法动态调整源批评家正则化的强度。在各种科学领域的多个离线优化任务中，GABO优于现有基准方法。我们的代码可在https://github.com/michael-s-yao/gabo 查询。

    Offline model-based policy optimization seeks to optimize a learned surrogate objective function without querying the true oracle objective during optimization. However, inaccurate surrogate model predictions are frequently encountered along the optimization trajectory. To address this limitation, we propose generative adversarial Bayesian optimization (GABO) using adaptive source critic regularization, a task-agnostic framework for Bayesian optimization that employs a Lipschitz-bounded source critic model to constrain the optimization trajectory to regions where the surrogate function is reliable. We show that under certain assumptions for the continuous input space prior, our algorithm dynamically adjusts the strength of the source critic regularization. GABO outperforms existing baselines on a number of different offline optimization tasks across a variety of scientific domains. Our code is available at https://github.com/michael-s-yao/gabo
    
[^18]: 考虑变化检测的语义八叉树之间点云立面标签的转移

    Transferring facade labels between point clouds with semantic octrees while considering change detection

    [https://arxiv.org/abs/2402.06531](https://arxiv.org/abs/2402.06531)

    该论文提出了一种使用八叉树结构将标签从一个已标注的点云转移到一个未标注的点云的方法，同时考虑了点云之间的变化。这种方法的主要贡献是自动在表示同一真实世界对象的两个不同点云之间进行标签转移，并可以应用于数据驱动的深度学习算法。

    

    点云和高分辨率的三维数据在测绘、建筑和虚拟现实等领域越来越重要。然而，仅有这些数据是不够的，语义标签对于提取有用信息至关重要。在这个背景下，我们提出了一种使用八叉树结构将标签从一个已标注的点云转移到一个未标注的点云的方法。该结构还分析了点云之间的变化。我们的实验证实了我们的方法在解决变化的同时有效地转移标签。本项目的主要贡献是开发了一种方法，可以在表示同一真实世界对象的两个不同点云之间进行自动标签转移。该方法对于数据驱动的深度学习算法非常重要，因为它还可以通过两个数据集之间的确定性标签转移来规避随机转移学习。

    Point clouds and high-resolution 3D data have become increasingly important in various fields, including surveying, construction, and virtual reality. However, simply having this data is not enough; to extract useful information, semantic labeling is crucial. In this context, we propose a method to transfer annotations from a labeled to an unlabeled point cloud using an octree structure. The structure also analyses changes between the point clouds. Our experiments confirm that our method effectively transfers annotations while addressing changes. The primary contribution of this project is the development of the method for automatic label transfer between two different point clouds that represent the same real-world object. The proposed method can be of great importance for data-driven deep learning algorithms as it can also allow circumventing stochastic transfer learning by deterministic label transfer between datasets depicting the same objects.
    
[^19]: 改进心肌梗死检测：一种新颖的多模态复合核策略在单一类别分类中的应用

    Refining Myocardial Infarction Detection: A Novel Multi-Modal Composite Kernel Strategy in One-Class Classification

    [https://arxiv.org/abs/2402.06530](https://arxiv.org/abs/2402.06530)

    本研究提出了一种新的方法，使用基于超声心动图的一种基于多模态复合核策略的单一类别分类算法来进行早期心肌梗死的检测。这种方法通过优化投影矩阵和特征转化，提高了心肌梗死检测的能力。

    

    早期心肌梗死（MI）的检测对于预防进一步心肌损伤非常重要，MI是由冠状动脉疾病（CAD）引起的一种严重疾病。本研究引入了一种新方法，使用一种基于超声心动图的单一类别分类（OCC）算法进行早期MI检测。我们的研究通过采用基于多模态子空间支持向量数据描述的新方法克服了超声心动图数据有限的挑战。提出的技术涉及一种特殊的MI检测框架，使用复合核在非线性投影技巧中融合高斯和拉普拉斯sigmoid函数，将多视图超声心动图结合起来。此外，我们通过在优化过程中调整投影矩阵的最大化策略，提高了投影矩阵更新策略的效果。我们的方法通过将从超声心动图数据中提取的特征有效地转化为优化的低维空间，增强了MI检测的能力。

    Early detection of myocardial infarction (MI), a critical condition arising from coronary artery disease (CAD), is vital to prevent further myocardial damage. This study introduces a novel method for early MI detection using a one-class classification (OCC) algorithm in echocardiography. Our study overcomes the challenge of limited echocardiography data availability by adopting a novel approach based on Multi-modal Subspace Support Vector Data Description. The proposed technique involves a specialized MI detection framework employing multi-view echocardiography incorporating a composite kernel in the non-linear projection trick, fusing Gaussian and Laplacian sigmoid functions. Additionally, we enhance the update strategy of the projection matrices by adapting maximization for both or one of the modalities in the optimization process. Our method boosts MI detection capability by efficiently transforming features extracted from echocardiography data into an optimized lower-dimensional su
    
[^20]: 内省规划：引导语言驱动的代理机器人改进自身的不确定性

    Introspective Planning: Guiding Language-Enabled Agents to Refine Their Own Uncertainty

    [https://arxiv.org/abs/2402.06529](https://arxiv.org/abs/2402.06529)

    本文研究了内省规划的概念，作为一种引导语言驱动的代理机器人改进自身不确定性的系统方法。通过识别任务不确定性并主动寻求澄清，内省显著提高了机器人任务规划的成功率和安全性。

    

    大型语言模型（LLM）展示了先进的推理能力，使得机器人能够理解自然语言指令，并通过适当的基础塑造来策略性地进行高级行动规划。然而，LLM产生的幻觉可能导致机器人自信地执行与用户目标不符或在极端情况下不安全的计划。此外，自然语言指令中的固有歧义可能引发任务的不确定性，尤其是在存在多个有效选项的情况下。为了解决这个问题，LLMs必须识别此类不确定性并主动寻求澄清。本文探索了内省规划的概念，作为一种系统方法，引导LLMs在无需微调的情况下形成意识到不确定性的机器人任务执行计划。我们研究了任务级机器人规划中的不确定性量化，并证明与最先进的基于LLM的规划方法相比，内省显著提高了成功率和安全性。

    Large language models (LLMs) exhibit advanced reasoning skills, enabling robots to comprehend natural language instructions and strategically plan high-level actions through proper grounding. However, LLM hallucination may result in robots confidently executing plans that are misaligned with user goals or, in extreme cases, unsafe. Additionally, inherent ambiguity in natural language instructions can induce task uncertainty, particularly in situations where multiple valid options exist. To address this issue, LLMs must identify such uncertainty and proactively seek clarification. This paper explores the concept of introspective planning as a systematic method for guiding LLMs in forming uncertainty--aware plans for robotic task execution without the need for fine-tuning. We investigate uncertainty quantification in task-level robot planning and demonstrate that introspection significantly improves both success rates and safety compared to state-of-the-art LLM-based planning approaches.
    
[^21]: 灵活的无限宽图卷积网络及表示学习的重要性

    Flexible infinite-width graph convolutional networks and the importance of representation learning

    [https://arxiv.org/abs/2402.06525](https://arxiv.org/abs/2402.06525)

    本文讨论了神经网络高斯过程（NNGP）在理论上的局限，提出图卷积深度内核机（graph convolutional deep kernel machine）来研究图分类任务中的表示学习问题。

    

    理解神经网络的一种常见理论方法是进行无限宽度限制，此时输出成为高斯过程（GP）分布。这被称为神经网络高斯过程（NNGP）。然而，NNGP内核是固定的，只能通过少量超参数进行调节，消除了任何表示学习的可能性。这与有限宽度的神经网络形成对比，后者通常被认为能够表现良好，正是因为它们能够学习表示。因此，简化神经网络以使其在理论上可处理的同时，NNGP可能会消除使其工作良好的因素（表示学习）。这激发了我们对一系列图分类任务中表示学习是否必要的理解。我们开发了一个精确的工具来完成这个任务，即图卷积深度内核机（graph convolutional deep kernel machine）。这与NNGP非常相似，因为它是无限宽度限制并使用内核，但它带有一个“旋钮”来控制表示学习的程度。

    A common theoretical approach to understanding neural networks is to take an infinite-width limit, at which point the outputs become Gaussian process (GP) distributed. This is known as a neural network Gaussian process (NNGP). However, the NNGP kernel is fixed, and tunable only through a small number of hyperparameters, eliminating any possibility of representation learning. This contrasts with finite-width NNs, which are often believed to perform well precisely because they are able to learn representations. Thus in simplifying NNs to make them theoretically tractable, NNGPs may eliminate precisely what makes them work well (representation learning). This motivated us to understand whether representation learning is necessary in a range of graph classification tasks. We develop a precise tool for this task, the graph convolutional deep kernel machine. This is very similar to an NNGP, in that it is an infinite width limit and uses kernels, but comes with a `knob' to control the amount 
    
[^22]: 使用MLS点云和词袋法重建立面细节

    Reconstructing facade details using MLS point clouds and Bag-of-Words approach

    [https://arxiv.org/abs/2402.06521](https://arxiv.org/abs/2402.06521)

    该论文提出了一种使用MLS点云和词袋法的新方法，用于重建立面细节。通过结合预定义的3D模型库和半全局特征，该方法在实验中展示了有希望的结果，并改进了传统的词袋法方法。该方法有潜力用于更真实的立面重建，并可以用于测试自动驾驶功能或估算立面太阳能潜力。

    

    在立面元素重建中，识别特定对象类型仍然具有挑战性，并且常常通过矩形假设或边界框来规避。我们提出了一种新的方法来重建3D立面细节。我们将MLS点云和预定义的3D模型库结合起来，使用词袋法概念，并通过加入半全局特征来增强。我们在叠加了随机噪声的模型和TUM-FA\c{C}ADE数据集上进行了实验。我们的方法展示了有希望的结果，改进了传统的词袋法方法。它具有在不进行矩形假设的情况下用于更真实的立面重建的潜力，可以在测试自动驾驶功能或估算立面太阳能潜力等应用中使用。

    In the reconstruction of fa\c{c}ade elements, the identification of specific object types remains challenging and is often circumvented by rectangularity assumptions or the use of bounding boxes. We propose a new approach for the reconstruction of 3D fa\c{c}ade details. We combine MLS point clouds and a pre-defined 3D model library using a BoW concept, which we augment by incorporating semi-global features. We conduct experiments on the models superimposed with random noise and on the TUM-FA\c{C}ADE dataset. Our method demonstrates promising results, improving the conventional BoW approach. It holds the potential to be utilized for more realistic facade reconstruction without rectangularity assumptions, which can be used in applications such as testing automated driving functions or estimating fa\c{c}ade solar potential.
    
[^23]: 使用大型语言模型的多模态临床试验结果预测

    Multimodal Clinical Trial Outcome Prediction with Large Language Models

    [https://arxiv.org/abs/2402.06512](https://arxiv.org/abs/2402.06512)

    本研究提出了一种名为LIFTED的多模态临床试验结果预测方法，通过将不同模态数据转化为自然语言描述来统一数据，并构建统一的抗噪声编码器进行信息提取。

    

    临床试验是一个关键且昂贵的过程，通常需要多年时间和大量财力资源。因此，开发临床试验结果预测模型旨在排除可能失败的药物，并具有显著的成本节约潜力。最近的数据驱动尝试利用深度学习方法整合多模态数据来预测临床试验结果。然而，这些方法依赖于手动设计的模态特定编码器，这限制了适应新模态的可扩展性和识别不同模态之间相似信息模式的能力。为了解决这些问题，我们提出了一种多模态专家混合（LIFTED）方法用于临床试验结果预测。具体而言，LIFTED通过将不同模态的数据转化为自然语言描述来统一不同模态数据。然后，LIFTED构建统一的抗噪声编码器，从模态特定的语言描述中提取信息。

    The clinical trial is a pivotal and costly process, often spanning multiple years and requiring substantial financial resources. Therefore, the development of clinical trial outcome prediction models aims to exclude drugs likely to fail and holds the potential for significant cost savings. Recent data-driven attempts leverage deep learning methods to integrate multimodal data for predicting clinical trial outcomes. However, these approaches rely on manually designed modal-specific encoders, which limits both the extensibility to adapt new modalities and the ability to discern similar information patterns across different modalities. To address these issues, we propose a multimodal mixture-of-experts (LIFTED) approach for clinical trial outcome prediction. Specifically, LIFTED unifies different modality data by transforming them into natural language descriptions. Then, LIFTED constructs unified noise-resilient encoders to extract information from modal-specific language descriptions. S
    
[^24]: 使用几何特征和深度学习网络对立面级别的点云进行分类

    Classifying point clouds at the facade-level using geometric features and deep learning networks

    [https://arxiv.org/abs/2402.06506](https://arxiv.org/abs/2402.06506)

    该论文提出了一种方法，利用几何特征和深度学习网络对立面级别的点云进行分类。实验证明，融合几何特征可以提高深度学习方法的性能，并促进语义分割的进步。

    

    具有立面细节的三维建筑模型在许多应用中发挥着重要作用。在立面级别上对点云进行分类是创建这样的数字副本的关键。然而，很少有研究将深度神经网络用于这种详细分类。我们提出了一种融合几何特征和深度学习网络的方法，以对立面级别的点云进行分类。我们的实验表明，这种早期融合的特征提高了深度学习方法的性能。该方法可用于补偿深度学习网络在捕捉局部几何信息方面的能力，并促进语义分割的进步。

    3D building models with facade details are playing an important role in many applications now. Classifying point clouds at facade-level is key to create such digital replicas of the real world. However, few studies have focused on such detailed classification with deep neural networks. We propose a method fusing geometric features with deep learning networks for point cloud classification at facade-level. Our experiments conclude that such early-fused features improve deep learning methods' performance. This method can be applied for compensating deep learning networks' ability in capturing local geometric information and promoting the advancement of semantic segmentation.
    
[^25]: ACTER: 用于解释和诊断RL策略的多样且可行的反事实序列

    ACTER: Diverse and Actionable Counterfactual Sequences for Explaining and Diagnosing RL Policies

    [https://arxiv.org/abs/2402.06503](https://arxiv.org/abs/2402.06503)

    ACTER是一个算法，用于生成可行的反事实序列，提供关于如何避免RL策略失败的可行建议。

    

    了解强化学习（RL）中的失败如何发生以及如何防止是为了实现调试、维护用户信任和开发个性化策略而必要的。反事实推理经常被用来归咎和理解失败，通过寻找最接近的可能世界以避免失败。然而，当前RL中的反事实状态解释只能使用当前状态特征来解释结果，并不能提供关于如何预防负结果的可行性措施。在这项工作中，我们提出了ACTER（用于解释强化学习结果的可行反事实序列）算法，该算法生成可行的反事实序列，提供了关于如何避免失败的可行建议。ACTER研究导致失败的动作，并使用进化算法NSGA-II生成可以最小化改变且具有高确定性的反事实动作序列，以防止失败，即使在随机情况下也是如此。

    Understanding how failure occurs and how it can be prevented in reinforcement learning (RL) is necessary to enable debugging, maintain user trust, and develop personalized policies. Counterfactual reasoning has often been used to assign blame and understand failure by searching for the closest possible world in which the failure is avoided. However, current counterfactual state explanations in RL can only explain an outcome using just the current state features and offer no actionable recourse on how a negative outcome could have been prevented. In this work, we propose ACTER (Actionable Counterfactual Sequences for Explaining Reinforcement Learning Outcomes), an algorithm for generating counterfactual sequences that provides actionable advice on how failure can be avoided. ACTER investigates actions leading to a failure and uses the evolutionary algorithm NSGA-II to generate counterfactual sequences of actions that prevent it with minimal changes and high certainty even in stochastic 
    
[^26]: 可扩展互动式机器学习用于未来指挥与控制

    Scalable Interactive Machine Learning for Future Command and Control

    [https://arxiv.org/abs/2402.06501](https://arxiv.org/abs/2402.06501)

    未来战争将需要指挥与控制（C2）人员在复杂且潜在模糊的情况下以更短的时间内做出决策。本论文通过利用互动式机器学习方法，结合人工智能和人类智能，以提高C2运作的适应性和效率。

    

    未来战争将需要指挥与控制（C2）人员在复杂且潜在模糊的情况下以更短的时间内做出决策。鉴于需要强大的决策过程和决策支持工具，人工智能和人类智能的集成具有革命性地改变C2运作流程的潜力，以确保在快速变化的操作环境中的适应性和效率。我们提议利用最近在互动式机器学习方面取得的突破，人类可以与机器学习算法合作以指导机器学习算法的行为。本文确定了目前科技发展中存在的几个差距，未来的工作应该解决这些差距，以扩展这些方法在复杂的C2环境中发挥作用。特别是，我们描述了三个研究重点领域，共同旨在实现可扩展的互动式机器学习（SIML）：1）开发人工智能与人类交互算法以实现协同规划。

    Future warfare will require Command and Control (C2) personnel to make decisions at shrinking timescales in complex and potentially ill-defined situations. Given the need for robust decision-making processes and decision-support tools, integration of artificial and human intelligence holds the potential to revolutionize the C2 operations process to ensure adaptability and efficiency in rapidly changing operational environments. We propose to leverage recent promising breakthroughs in interactive machine learning, in which humans can cooperate with machine learning algorithms to guide machine learning algorithm behavior. This paper identifies several gaps in state-of-the-art science and technology that future work should address to extend these approaches to function in complex C2 contexts. In particular, we describe three research focus areas that together, aim to enable scalable interactive machine learning (SIML): 1) developing human-AI interaction algorithms to enable planning in co
    
[^27]: 基于深度学习的全骨髓和淋巴结照射计划靶体积自动分割

    Deep Learning-Based Auto-Segmentation of Planning Target Volume for Total Marrow and Lymph Node Irradiation

    [https://arxiv.org/abs/2402.06494](https://arxiv.org/abs/2402.06494)

    本文基于深度学习探讨了在全骨髓和淋巴结照射治疗中，使用2D和3D U-Net模型以及nnU-Net框架自动分割照射计划靶体积(PTV)的方法，结果表明nnU-Net框架显著提高了分割性能。

    

    为了优化癌症治疗中的放疗，特别是在处理复杂照射如全骨髓和淋巴结照射(TMLI)时，准确勾画照射计划靶体积(PTV)至关重要。不幸的是，对于这种治疗，依赖手工勾画耗时且容易出错。本文研究了将深度学习(DL)应用于自动分割TMLI治疗中的PTV，建立在先前基于2D U-Net模型的解决方案上。我们扩展了之前的研究(i)通过使用nnU-Net框架开发了2D和3D U-Net模型，(ii)通过将骨骼排除在外评估了训练好的模型在PTV上的表现，骨骼主要包含淋巴结，是最具挑战的区域。我们的结果显示nnU-Net框架的引入在分割性能上有统计上显著的提升。

    In order to optimize the radiotherapy delivery for cancer treatment, especially when dealing with complex treatments such as Total Marrow and Lymph Node Irradiation (TMLI), the accurate contouring of the Planning Target Volume (PTV) is crucial. Unfortunately, relying on manual contouring for such treatments is time-consuming and prone to errors. In this paper, we investigate the application of Deep Learning (DL) to automate the segmentation of the PTV in TMLI treatment, building upon previous work that introduced a solution to this problem based on a 2D U-Net model. We extend the previous research (i) by employing the nnU-Net framework to develop both 2D and 3D U-Net models and (ii) by evaluating the trained models on the PTV with the exclusion of bones, which consist mainly of lymp-nodes and represent the most challenging region of the target volume to segment. Our result show that the introduction of nnU-NET framework led to statistically significant improvement in the segmentation p
    
[^28]: 通过关注结构化量化的嵌入在Transformer中引导系统性

    Inducing Systematicity in Transformers by Attending to Structurally Quantized Embeddings

    [https://arxiv.org/abs/2402.06492](https://arxiv.org/abs/2402.06492)

    本论文提出了SQ-Transformer模型，通过在嵌入和注意层中引入结构化量化的方法，无论训练集的复杂度如何，都能够明确地鼓励模型在编码句子时保持系统性。

    

    Transformer在训练过复杂数据集后能够推广到结构和实体的新组合，但在复杂度不足的数据集上容易过拟合。我们观察到，当训练集足够复杂时，模型使用系统性的注意模式对具有共同句法结构的句子进行编码。受到这一观察的启发，我们提出了SQ-Transformer（结构化量化），即使使用低复杂度的训练集，也能明确地在嵌入和注意层中鼓励系统性。在嵌入层面上，我们引入了结构导向的向量量化（SoVQ），将单词嵌入聚类成若干类具有结构等价的实体。在注意层面上，我们设计了系统性注意层（SAL）和另一种替代性的系统性正则化层（SRL），它们都在量化的词嵌入上操作，以便以不变或类似的注意模式编码具有相同结构的句子。

    Transformers generalize to novel compositions of structures and entities after being trained on a complex dataset, but easily overfit on datasets of insufficient complexity. We observe that when the training set is sufficiently complex, the model encodes sentences that have a common syntactic structure using a systematic attention pattern. Inspired by this observation, we propose SQ-Transformer (Structurally Quantized) that explicitly encourages systematicity in the embeddings and attention layers, even with a training set of low complexity. At the embedding level, we introduce Structure-oriented Vector Quantization (SoVQ) to cluster word embeddings into several classes of structurally equivalent entities. At the attention level, we devise the Systematic Attention Layer (SAL) and an alternative, Systematically Regularized Layer (SRL) that operate on the quantized word embeddings so that sentences of the same structure are encoded with invariant or similar attention patterns. Empiricall
    
[^29]: 自主超声导航的心脏超声模拟

    Cardiac ultrasound simulation for autonomous ultrasound navigation

    [https://arxiv.org/abs/2402.06463](https://arxiv.org/abs/2402.06463)

    该论文提出了一种用于自主超声导航的心脏超声模拟方法，通过使用其他模态的分割、优化的体积数据表示和GPU加速的蒙特卡洛路径追踪，生成大量视角相关和具有患者特异性的超声图像。

    

    超声成像已被广泛应用于诊断和介入目的。然而，由于成像伪影、获取参数范围和患者解剖变异等原因，操作员技能的差异导致图像质量的不稳定。自动化图像获取任务可以提高获取的一致性和质量，但训练此类算法需要大量的导航数据，而这些数据在常规检查中没有保存。因此，我们提出了一种方法，可以从其他模态和任意位置生成大量超声图像，以便后续通过学习算法进行导航。我们提出了一种新的模拟流程，使用其他模态的分割、优化的体积数据表示和GPU加速的蒙特卡洛路径追踪来生成视角相关和具有患者特异性的超声图像。

    Ultrasound is well-established as an imaging modality for diagnostic and interventional purposes. However, the image quality varies with operator skills as acquiring and interpreting ultrasound images requires extensive training due to the imaging artefacts, the range of acquisition parameters and the variability of patient anatomies. Automating the image acquisition task could improve acquisition reproducibility and quality but training such an algorithm requires large amounts of navigation data, not saved in routine examinations. Thus, we propose a method to generate large amounts of ultrasound images from other modalities and from arbitrary positions, such that this pipeline can later be used by learning algorithms for navigation. We present a novel simulation pipeline which uses segmentations from other modalities, an optimized volumetric data representation and GPU-accelerated Monte Carlo path tracing to generate view-dependent and patient-specific ultrasound images. We extensivel
    
[^30]: 顺序流匹配用于生成建模

    Sequential Flow Matching for Generative Modeling

    [https://arxiv.org/abs/2402.06461](https://arxiv.org/abs/2402.06461)

    本文提出了一种称为SeqRF的新方法，用于通过直线化概率流来减小全局截断误差，并以此加速取样和提高综合质量。

    

    直接引导连续时间生成模型（例如扩散模型或基于流的模型）的概率流是通过数值解算器快速取样的关键。现有方法通过直接生成噪声和数据分布之间的联合分布的概率路径来学习线性路径。ODE模型的仿真速度慢的一个重要原因是ODE轨迹的高曲率导致的ODE求解器的全局截断误差，这会在低NFE范围内放大数值解算器的截断误差。为了解决这个挑战，我们提出了一种称为SeqRF的新方法，它是一种学习技术，用于直线化概率流以减小全局截断误差，从而加速取样并提高综合质量。通过理论和实证研究，我们首先观察到了SeqRF的直线化特性。

    Straightening the probability flow of the continuous-time generative models, such as diffusion models or flow-based models, is the key to fast sampling through the numerical solvers, existing methods learn a linear path by directly generating the probability path the joint distribution between the noise and data distribution. One key reason for the slow sampling speed of the ODE-based solvers that simulate these generative models is the global truncation error of the ODE solver, caused by the high curvature of the ODE trajectory, which explodes the truncation error of the numerical solvers in the low-NFE regime. To address this challenge, We propose a novel method called SeqRF, a learning technique that straightens the probability flow to reduce the global truncation error and hence enable acceleration of sampling and improve the synthesis quality. In both theoretical and empirical studies, we first observe the straightening property of our SeqRF. Through empirical evaluations via SeqR
    
[^31]: V-STaR: 自学推理器的训练方法

    V-STaR: Training Verifiers for Self-Taught Reasoners

    [https://arxiv.org/abs/2402.06457](https://arxiv.org/abs/2402.06457)

    V-STaR利用正确和不正确的解决方案训练验证器，用于选择模型生成的解决方案，实现了自我改进和验证方法在常见代码生成和数学推理任务中达到4%到17%的测试准确率提升。

    

    大型语言模型（LLM）的常见自我改进方法，例如STaR（Zelikman等人，2022），通过自动生成的解决方案迭代微调LLM以提高其问题解决能力。然而，这些方法在此过程中丢弃了大量的不正确的解决方案，可能忽略了这些解决方案中的宝贵信息。为了解决这个缺点，我们提出了V-STaR，它利用自我改进过程中生成的正确和不正确的解决方案来使用DPO训练一个判断模型生成解决方案的正确性的验证器。在推理时，这个验证器用来在众多候选解决方案中选择一个解决方案。多次运行V-STaR会逐步产生更好的推理器和验证器，在常见代码生成和数学推理基准测试中，使用LLaMA2模型可以取得4%到17%的测试准确率提升。

    Common self-improvement approaches for large language models (LLMs), such as STaR (Zelikman et al., 2022), iteratively fine-tune LLMs on self-generated solutions to improve their problem-solving ability. However, these approaches discard the large amounts of incorrect solutions generated during this process, potentially neglecting valuable information in such solutions. To address this shortcoming, we propose V-STaR that utilizes both the correct and incorrect solutions generated during the self-improvement process to train a verifier using DPO that judges correctness of model-generated solutions. This verifier is used at inference time to select one solution among many candidate solutions. Running V-STaR for multiple iterations results in progressively better reasoners and verifiers, delivering a 4% to 17% test accuracy improvement over existing self-improvement and verification approaches on common code generation and math reasoning benchmarks with LLaMA2 models.
    
[^32]: 基于评估构建过程中组合性能的算法框架用于构建多个决策树

    An Algorithmic Framework for Constructing Multiple Decision Trees by Evaluating Their Combination Performance Throughout the Construction Process

    [https://arxiv.org/abs/2402.06452](https://arxiv.org/abs/2402.06452)

    本研究提出了一种新的算法框架，能够同时构建多个决策树，并在构建过程中评估它们的组合性能，以指导最终预测的决策树组合的适用性。

    

    在机器学习中，使用决策树组合进行预测已被证明是有效的。目前构建决策树组合进行预测的典型方法有bagging和boosting。bagging方法独立构建决策树而不评估它们的组合性能，并在之后进行平均。boosting方法顺序构建决策树，只在每一步评估新决策树与固定过去决策树组合的性能。因此，这两种方法都不直接构建也不评估最终预测的决策树组合的适用性。当最终预测基于多个决策树组合时，在构建过程中评估组合的适用性是很自然的。在本研究中，我们提出了一种新的算法框架，能够同时构建决策树并在构建过程中评估它们的组合性能。我们的框架重复两个步骤。第一步是同时构建多个决策树，第二步是根据组合性能评估这些决策树的适用性，以指导后续的决策树构建。

    Predictions using a combination of decision trees are known to be effective in machine learning. Typical ideas for constructing a combination of decision trees for prediction are bagging and boosting. Bagging independently constructs decision trees without evaluating their combination performance and averages them afterward. Boosting constructs decision trees sequentially, only evaluating a combination performance of a new decision tree and the fixed past decision trees at each step. Therefore, neither method directly constructs nor evaluates a combination of decision trees for the final prediction. When the final prediction is based on a combination of decision trees, it is natural to evaluate the appropriateness of the combination when constructing them. In this study, we propose a new algorithmic framework that constructs decision trees simultaneously and evaluates their combination performance throughout the construction process. Our framework repeats two procedures. In the first p
    
[^33]: 深度平衡算法推理器

    The Deep Equilibrium Algorithmic Reasoner

    [https://arxiv.org/abs/2402.06445](https://arxiv.org/abs/2402.06445)

    本文介绍了一种深度平衡算法推理器，可以通过直接找到算法的平衡点来训练网络解决算法问题。

    

    最近关于神经算法推理的研究表明，图神经网络（GNNs）可以学习执行经典算法。然而，这样做一直使用的是递归架构，其中每个GNN的迭代与算法的迭代一致。由于算法的解通常是一个平衡点，我们猜测并经验性地验证，可以通过直接找到平衡点来训练网络解决算法问题。注意，这不需要将每个GNN的迭代与算法的步骤匹配。

    Recent work on neural algorithmic reasoning has demonstrated that graph neural networks (GNNs) could learn to execute classical algorithms. Doing so, however, has always used a recurrent architecture, where each iteration of the GNN aligns with an algorithm's iteration. Since an algorithm's solution is often an equilibrium, we conjecture and empirically validate that one can train a network to solve algorithmic problems by directly finding the equilibrium. Note that this does not require matching each GNN iteration with a step of the algorithm.
    
[^34]: 在神经网络中引入泰勒级数和递归结构进行时间序列预测

    Incorporating Taylor Series and Recursive Structure in Neural Networks for Time Series Prediction

    [https://arxiv.org/abs/2402.06441](https://arxiv.org/abs/2402.06441)

    本论文提出了一种新的神经网络架构，将ResNet结构和泰勒级数框架相结合，实现了对时间序列预测的显著改进。此外，引入递归结构可以进一步提高预测准确性。这一研究为时间序列分析方法学的推进提供了潜力巨大的模型，具有广阔的研究和应用前景。

    

    时间序列分析在物理学、生物学、化学和金融等多个领域都具有重要意义。本文提出了一种新颖的神经网络架构，将ResNet结构的要素与泰勒级数框架的创新结合起来。该方法在多个基准数据集上展现出显著的测试准确性提升。此外，我们还将我们的方法扩展到引入递归步骤，进一步提高测试准确性。我们的研究结果突显了我们提出的模型在推进时间序列分析方法学方面的潜力，为未来的研究和应用提供了有前景的途径。

    Time series analysis is relevant in various disciplines such as physics, biology, chemistry, and finance. In this paper, we present a novel neural network architecture that integrates elements from ResNet structures, while introducing the innovative incorporation of the Taylor series framework. This approach demonstrates notable enhancements in test accuracy across many of the baseline datasets investigated. Furthermore, we extend our method to incorporate a recursive step, which leads to even further improvements in test accuracy. Our findings underscore the potential of our proposed model to significantly advance time series analysis methodologies, offering promising avenues for future research and application.
    
[^35]: 真相在哪里？在连续的世界中遭遇混淆的风险

    Where is the Truth? The Risk of Getting Confounded in a Continual World

    [https://arxiv.org/abs/2402.06434](https://arxiv.org/abs/2402.06434)

    这篇论文研究了在一个连续学习环境中遭遇混淆的问题，通过实验证明了传统的连续学习方法无法忽略混淆，需要更强大的方法来处理这个问题。

    

    如果一个数据集通过一个虚假相关性来解决，而这种相关性无法泛化到新数据，该数据集就是混淆的。我们将展示，在一个连续学习的环境中，混淆因素可能随着任务的变化而变化，导致的挑战远远超过通常考虑的遗忘问题。具体来说，我们从数学上推导了这种混淆因素对一组混淆任务的有效联合解空间的影响。有趣的是，我们的理论预测，在许多这样的连续数据集中，当任务进行联合训练时，虚假相关性很容易被忽略，但是在顺序考虑任务时，避免混淆要困难得多。我们构建了这样一个数据集，并通过实验证明标准的连续学习方法无法忽略混淆，而同时对所有任务进行联合训练则是成功的。我们的连续混淆数据集ConCon基于CLEVR图像，证明了需要更强大的连续学习方法来处理混淆问题。

    A dataset is confounded if it is most easily solved via a spurious correlation which fails to generalize to new data. We will show that, in a continual learning setting where confounders may vary in time across tasks, the resulting challenge far exceeds the standard forgetting problem normally considered. In particular, we derive mathematically the effect of such confounders on the space of valid joint solutions to sets of confounded tasks. Interestingly, our theory predicts that for many such continual datasets, spurious correlations are easily ignored when the tasks are trained on jointly, but it is far harder to avoid confounding when they are considered sequentially. We construct such a dataset and demonstrate empirically that standard continual learning methods fail to ignore confounders, while training jointly on all tasks is successful. Our continually confounded dataset, ConCon, is based on CLEVR images and demonstrates the need for continual learning methods with more robust b
    
[^36]: 相信过程：零知识机器学习增强生成式AI交互中的信任

    Trust the Process: Zero-Knowledge Machine Learning to Enhance Trust in Generative AI Interactions

    [https://arxiv.org/abs/2402.06414](https://arxiv.org/abs/2402.06414)

    本文提出了使用零知识证明技术来解决生成式AI公平性和隐私保护的问题，并介绍了一个实际的ZKML实现，可以增强用户对生成式AI输出的信任和透明度。

    

    生成式AI（如transformers模型）在各个领域开辟了新的可能性，但也引发了公平性、透明度和可靠性方面的关切，尤其在医学和法律等领域。本文强调通过生成式AI确保这些领域的公平性和质量的紧迫性，并探索使用密码学技术，特别是零知识证明（ZKP），来解决性能公平性和准确性方面的问题，同时保护模型的隐私。将ZKP应用于机器学习模型，即零知识机器学习（ZKML），可以在不泄露敏感模型信息的情况下对AI生成内容进行独立验证，促进透明度和信任。ZKML通过为模型预测提供密码学审计痕迹，并确保用户之间的统一性能，提高了AI的公平性。我们介绍了snarkGPT，一个实际的transformers型ZKML实现，可以使用户验证输出的准确性和质量。

    Generative AI, exemplified by models like transformers, has opened up new possibilities in various domains but also raised concerns about fairness, transparency and reliability, especially in fields like medicine and law. This paper emphasizes the urgency of ensuring fairness and quality in these domains through generative AI. It explores using cryptographic techniques, particularly Zero-Knowledge Proofs (ZKPs), to address concerns regarding performance fairness and accuracy while protecting model privacy. Applying ZKPs to Machine Learning models, known as ZKML (Zero-Knowledge Machine Learning), enables independent validation of AI-generated content without revealing sensitive model information, promoting transparency and trust. ZKML enhances AI fairness by providing cryptographic audit trails for model predictions and ensuring uniform performance across users. We introduce snarkGPT, a practical ZKML implementation for transformers, to empower users to verify output accuracy and qualit
    
[^37]: 提高非凸分布式优化在函数相似性下的最坏情况双向通信复杂性

    Improving the Worst-Case Bidirectional Communication Complexity for Nonconvex Distributed Optimization under Function Similarity

    [https://arxiv.org/abs/2402.06412](https://arxiv.org/abs/2402.06412)

    本文提出了MARINA-P方法，通过引入一系列相关压缩器，优化了服务器到工作节点的通信复杂度。理论分析证明，MARINA-P在算法上优于现有方法，并可以作为支持双向压缩的起点。通过与上行压缩和动量步骤的结合，M3方法实现了双向压缩，并在总通信复杂度上改进。

    

    服务器和工作节点之间的有效通信在分布式优化中起着关键作用。本文主要关注优化服务器到工作节点的通信，并揭示了当前流行的下行压缩方法中的低效性。首先考虑上行通信成本可忽略的纯粹情况下，我们引入MARINA-P，一种使用一系列相关压缩器的新型下行压缩方法。理论分析证明，使用排列压缩器的MARINA-P可以实现服务器到工作节点的通信复杂度随工作节点数量提高，因此在算法上可证明优于现有算法。我们进一步展示了MARINA-P可以作为支持双向压缩的方法的起点。我们介绍了M3，这是一种将MARINA-P与上行压缩和动量步骤组合的方法，能够实现双向压缩，并在总通信复杂度上证明了改进。

    Effective communication between the server and workers plays a key role in distributed optimization. In this paper, we focus on optimizing the server-to-worker communication, uncovering inefficiencies in prevalent downlink compression approaches. Considering first the pure setup where the uplink communication costs are negligible, we introduce MARINA-P, a novel method for downlink compression, employing a collection of correlated compressors. Theoretical analyses demonstrates that MARINA-P with permutation compressors can achieve a server-to-worker communication complexity improving with the number of workers, thus being provably superior to existing algorithms. We further show that MARINA-P can serve as a starting point for extensions such as methods supporting bidirectional compression. We introduce M3, a method combining MARINA-P with uplink compression and a momentum step, achieving bidirectional compression with provable improvements in total communication complexity as the number
    
[^38]: 层次化Transformer是高效的元强化学习者

    Hierarchical Transformers are Efficient Meta-Reinforcement Learners

    [https://arxiv.org/abs/2402.06402](https://arxiv.org/abs/2402.06402)

    层次化Transformer是一种高效的元强化学习方法，通过有效地提取过去经验的信息丰富资源，并应用于新的环境中，实现了超越最先进方法的元训练效果，并显著提高了泛化能力和学习效率。

    

    我们介绍了一种强大的在线元强化学习方法，即层次化Transformer用于元强化学习（HTrMRL）。HTrMRL旨在解决使强化学习代理能够在以前未见任务中有效执行的挑战。我们展示了过去的经验作为信息丰富的资源，我们的模型有效地提炼和应用到新的上下文中。我们学习到的算法能够超越以前的最先进，并提供更高效的元训练，同时显著改善了泛化能力。在Meta-World基准的各种模拟任务上获得的实验结果表明，在各种任务上相比最先进的方法，学习效率和适应性显著提升。我们的方法不仅增强了代理从有限数据中的泛化能力，还为更强大和多功能的AI系统铺平了道路。

    We introduce Hierarchical Transformers for Meta-Reinforcement Learning (HTrMRL), a powerful online meta-reinforcement learning approach. HTrMRL aims to address the challenge of enabling reinforcement learning agents to perform effectively in previously unseen tasks. We demonstrate how past episodes serve as a rich source of information, which our model effectively distills and applies to new contexts. Our learned algorithm is capable of outperforming the previous state-of-the-art and provides more efficient meta-training while significantly improving generalization capabilities. Experimental results, obtained across various simulated tasks of the Meta-World Benchmark, indicate a significant improvement in learning efficiency and adaptability compared to the state-of-the-art on a variety of tasks. Our approach not only enhances the agent's ability to generalize from limited data but also paves the way for more robust and versatile AI systems.
    
[^39]: 关于随机梯度下降（SGD）的收敛速度及其在修改的多臂赌博机上的策略梯度应用

    On the Convergence Rate of the Stochastic Gradient Descent (SGD) and application to a modified policy gradient for the Multi Armed Bandit

    [https://arxiv.org/abs/2402.06388](https://arxiv.org/abs/2402.06388)

    该论文证明了当学习速率按照逆时间衰减规则时，随机梯度下降（SGD）的收敛速度，并应用于修改的带有L2正则化的策略梯度多臂赌博机（MAB）的收敛性分析。

    

    我们提出了一个自包含的证明，证明了当学习速率遵循逆时间衰减规则时，随机梯度下降（SGD）的收敛速度；接下来，我们将这些结果应用于带有L2正则化的修改的策略梯度多臂赌博机（MAB）的收敛性分析。

    We present a self-contained proof of the convergence rate of the Stochastic Gradient Descent (SGD) when the learning rate follows an inverse time decays schedule; we next apply the results to the convergence of a modified form of policy gradient Multi-Armed Bandit (MAB) with $L2$ regularization.
    
[^40]: 基于Boosting的顺序元树集成构建以改进决策树

    Boosting-Based Sequential Meta-Tree Ensemble Construction for Improved Decision Trees

    [https://arxiv.org/abs/2402.06386](https://arxiv.org/abs/2402.06386)

    本研究提出了一种使用提升方法构建多个元树的方法，旨在改进决策树的预测性能。

    

    决策树是机器学习领域中最流行的方法之一。然而，它存在过度加深树形结构导致的过拟合问题。近期有人提出了元树来解决过度加深树形结构导致的过拟合问题。此外，基于贝叶斯决策理论，元树能够保证统计上的最优性。因此，相较于决策树，我们期望元树表现更好。与单个决策树相比，已知由提升算法构造的决策树集成在提高预测性能方面更为有效。因此，我们期望由元树集成来提高预测性能比单个元树更有效，并且以前没有研究使用提升方法构建多个元树。因此，在本研究中，我们提出了使用提升方法构建多个元树的方法。

    A decision tree is one of the most popular approaches in machine learning fields. However, it suffers from the problem of overfitting caused by overly deepened trees. Then, a meta-tree is recently proposed. It solves the problem of overfitting caused by overly deepened trees. Moreover, the meta-tree guarantees statistical optimality based on Bayes decision theory. Therefore, the meta-tree is expected to perform better than the decision tree. In contrast to a single decision tree, it is known that ensembles of decision trees, which are typically constructed boosting algorithms, are more effective in improving predictive performance. Thus, it is expected that ensembles of meta-trees are more effective in improving predictive performance than a single meta-tree, and there are no previous studies that construct multiple meta-trees in boosting. Therefore, in this study, we propose a method to construct multiple meta-trees using a boosting approach. Through experiments with synthetic and ben
    
[^41]: 高斯（多项式）树的最优估计

    Optimal estimation of Gaussian (poly)trees

    [https://arxiv.org/abs/2402.06380](https://arxiv.org/abs/2402.06380)

    该论文开发了最优算法，在学习高斯树和高斯多项式树方面取得了显著成果，并提供了详细的理论保证和实证结果。

    

    我们开发了一种从数据中学习无向高斯树和有向高斯多项式树的最优算法。我们考虑了分布学习（即KL距离）和结构学习（即精确恢复）的两个问题。第一种方法基于Chow-Liu算法，有效地学习最优的树状分布。第二种方法是对用于多项式树的PC算法的修改，它使用偏相关作为条件独立性测试器进行基于约束的结构学习。我们得到了这两种方法的显式有限样本保证，并通过推导匹配的下界证明这两种方法都是最优的。此外，我们进行了数值实验，比较了各种算法的性能，提供了进一步的洞察和经验证据。

    We develop optimal algorithms for learning undirected Gaussian trees and directed Gaussian polytrees from data. We consider both problems of distribution learning (i.e. in KL distance) and structure learning (i.e. exact recovery). The first approach is based on the Chow-Liu algorithm, and learns an optimal tree-structured distribution efficiently. The second approach is a modification of the PC algorithm for polytrees that uses partial correlation as a conditional independence tester for constraint-based structure learning. We derive explicit finite-sample guarantees for both approaches, and show that both approaches are optimal by deriving matching lower bounds. Additionally, we conduct numerical experiments to compare the performance of various algorithms, providing further insights and empirical evidence.
    
[^42]: 基于强化学习和粒子滤波的高精度地质定向

    High-Precision Geosteering via Reinforcement Learning and Particle Filters

    [https://arxiv.org/abs/2402.06377](https://arxiv.org/abs/2402.06377)

    基于强化学习和粒子滤波的地质定向方法，通过实时数据处理实现高精度地质定向决策

    

    地质定向是钻井作业中的关键部分，传统上涉及对各种数据源（如井测数据）的手动解读。这引入了主观偏见和不一致的程序。学术界尝试通过贪婪优化和近似动态规划（ADP）来解决地质定向决策优化问题，显示出了一定的潜力，但缺乏适应现实多样情况的能力。强化学习（RL）为这些挑战提供了解决方案，通过基于奖励的迭代学习来促进最优决策。状态估计方法，例如粒子滤波（PF），提供了一种基于在线信息的补充策略，用于地质定向决策。我们将基于RL的地质定向与PF相结合，以应对现实的地质定向情况。我们的框架使用PF处理实时井测数据，估计井的位置相对于地层，然后将其信息用于基于RL的决策过程。我们比较了我们的方法

    Geosteering, a key component of drilling operations, traditionally involves manual interpretation of various data sources such as well-log data. This introduces subjective biases and inconsistent procedures. Academic attempts to solve geosteering decision optimization with greedy optimization and Approximate Dynamic Programming (ADP) showed promise but lacked adaptivity to realistic diverse scenarios. Reinforcement learning (RL) offers a solution to these challenges, facilitating optimal decision-making through reward-based iterative learning. State estimation methods, e.g., particle filter (PF), provide a complementary strategy for geosteering decision-making based on online information. We integrate an RL-based geosteering with PF to address realistic geosteering scenarios. Our framework deploys PF to process real-time well-log data to estimate the location of the well relative to the stratigraphic layers, which then informs the RL-based decision-making process. We compare our method
    
[^43]: TEE4EHR：用于更好地学习电子健康记录中表示的Transformer事件编码器

    TEE4EHR: Transformer Event Encoder for Better Representation Learning in Electronic Health Records

    [https://arxiv.org/abs/2402.06367](https://arxiv.org/abs/2402.06367)

    TEE4EHR是一个使用点过程损失函数的Transformer事件编码器，用于编码电子健康记录中实验室测试的模式。它能够解决EHR中时间序列的不规则采样和缺失数据的挑战，并在各种基准数据集和实际数据库上展现出优越性能。

    

    电子健康记录（EHR）中时间序列的不规则采样是开发机器学习模型的主要挑战之一。此外，某些临床变量的缺失数据模式并非随机的，而是取决于临床医生的决策和患者的状态。点过程是一种数学框架，用于分析与不规则采样模式一致的事件序列数据。我们的模型TEE4EHR是一个具有点过程损失函数的Transformer事件编码器（TEE），它对EHR中的实验室检测模式进行编码。我们的TEE的效用已在各种基准事件序列数据集上进行了研究。此外，我们在两个实际的EHR数据库上进行了实验，以更全面地评估我们的模型。首先，在自监督学习方法中，TEE与现有的基于注意力的深度神经网络一起进行联合学习，这在负对数似然和未来事件预测方面具有卓越的性能。

    Irregular sampling of time series in electronic health records (EHRs) is one of the main challenges for developing machine learning models. Additionally, the pattern of missing data in certain clinical variables is not at random but depends on the decisions of clinicians and the state of the patient. Point process is a mathematical framework for analyzing event sequence data that is consistent with irregular sampling patterns. Our model, TEE4EHR, is a transformer event encoder (TEE) with point process loss that encodes the pattern of laboratory tests in EHRs. The utility of our TEE has been investigated in a variety of benchmark event sequence datasets. Additionally, we conduct experiments on two real-world EHR databases to provide a more comprehensive evaluation of our model. Firstly, in a self-supervised learning approach, the TEE is jointly learned with an existing attention-based deep neural network which gives superior performance in negative log-likelihood and future event predic
    
[^44]: SpongeNet 攻击：深度神经网络的海绵权重中毒

    The SpongeNet Attack: Sponge Weight Poisoning of Deep Neural Networks

    [https://arxiv.org/abs/2402.06357](https://arxiv.org/abs/2402.06357)

    本文提出了一种名为 SpongeNet 的新型海绵攻击，通过直接作用于预训练模型参数，成功增加了视觉模型的能耗，而且所需的样本数量更少。

    

    海绵攻击旨在增加在硬件加速器上部署的神经网络的能耗和计算时间。现有的海绵攻击可以通过海绵示例进行推理，也可以通过海绵中毒在训练过程中进行。海绵示例利用添加到模型输入的扰动来增加能量和延迟，而海绵中毒则改变模型的目标函数来引发推理时的能量/延迟效应。在这项工作中，我们提出了一种新颖的海绵攻击，称为 SpongeNet。SpongeNet 是第一个直接作用于预训练模型参数的海绵攻击。我们的实验表明，相比于海绵中毒，SpongeNet 可以成功增加视觉模型的能耗，并且所需的样本数量更少。我们的实验结果表明，如果不专门针对海绵中毒进行调整（即减小批归一化偏差值），则毒害防御会失效。我们的工作显示出海绵攻击的影响。

    Sponge attacks aim to increase the energy consumption and computation time of neural networks deployed on hardware accelerators. Existing sponge attacks can be performed during inference via sponge examples or during training via Sponge Poisoning. Sponge examples leverage perturbations added to the model's input to increase energy and latency, while Sponge Poisoning alters the objective function of a model to induce inference-time energy/latency effects.   In this work, we propose a novel sponge attack called SpongeNet. SpongeNet is the first sponge attack that is performed directly on the parameters of a pre-trained model. Our experiments show that SpongeNet can successfully increase the energy consumption of vision models with fewer samples required than Sponge Poisoning. Our experiments indicate that poisoning defenses are ineffective if not adjusted specifically for the defense against Sponge Poisoning (i.e., they decrease batch normalization bias values). Our work shows that Spong
    
[^45]: 在线不平衡多臂赌博机中的曝光公平性

    Fairness of Exposure in Online Restless Multi-armed Bandits

    [https://arxiv.org/abs/2402.06348](https://arxiv.org/abs/2402.06348)

    本研究提出了第一个在线的公平RMAB框架，通过将每个臂的拉取与其优势成比例，实现了公平的曝光。算法在单次拉取的公平性遗憾方面取得了次线性的结果$O(\sqrt{T\ln T})$。

    

    不平衡多臂赌博机（RMAB）推广了多臂赌博机，其中每个臂展示马尔可夫行为，并根据其过渡动态进行转换。针对RMAB的解决方案存在于离线和在线情况下。然而，它们没有考虑臂之间的拉取分布。研究表明，最优策略会导致不公平，其中一些臂不够暴露。现有的RMAB公平性工作主要集中在离线案例中，这降低了它们在环境大部分不知道的现实场景中的应用。在在线场景中，我们提出了第一个公平的RMAB框架，其中每个臂接收的拉取与其优势成比例。我们将臂的优势定义为其稳态奖励分布的函数。我们证明了我们的算法在单次拉取的公平性遗憾方面实现了次线性的结果$O(\sqrt{T\ln T})$，其中$T$是总的尝试次数。经验证明，我们的算法在多次拉取的情况下表现良好。

    Restless multi-armed bandits (RMABs) generalize the multi-armed bandits where each arm exhibits Markovian behavior and transitions according to their transition dynamics. Solutions to RMAB exist for both offline and online cases. However, they do not consider the distribution of pulls among the arms. Studies have shown that optimal policies lead to unfairness, where some arms are not exposed enough. Existing works in fairness in RMABs focus heavily on the offline case, which diminishes their application in real-world scenarios where the environment is largely unknown. In the online scenario, we propose the first fair RMAB framework, where each arm receives pulls in proportion to its merit. We define the merit of an arm as a function of its stationary reward distribution. We prove that our algorithm achieves sublinear fairness regret in the single pull case $O(\sqrt{T\ln T})$, with $T$ being the total number of episodes. Empirically, we show that our algorithm performs well in the multi
    
[^46]: 在开放集识别评估中考虑类别不平衡问题

    Taking Class Imbalance Into Account in Open Set Recognition Evaluation

    [https://arxiv.org/abs/2402.06331](https://arxiv.org/abs/2402.06331)

    本文研究了开放集识别评估中类别不平衡问题的影响，并提出了一套指导方针。

    

    近年来，基于深度神经网络的系统不仅在流行度上逐渐增加，而且还受到用户信任的增长。然而，由于这些系统对于未知类别样本的识别仍服从封闭世界假设，并且往往会产生高置信度的错误标签。本文研究了开放集识别方法的评估，重点关注类别不平衡对已知和未知样本之间的二分的影响。作为问题分析的结果，我们提出了一套在这个领域中评估方法的指导方针。

    In recent years Deep Neural Network-based systems are not only increasing in popularity but also receive growing user trust. However, due to the closed-world assumption of such systems, they cannot recognize samples from unknown classes and often induce an incorrect label with high confidence. Presented work looks at the evaluation of methods for Open Set Recognition, focusing on the impact of class imbalance, especially in the dichotomy between known and unknown samples. As an outcome of problem analysis, we present a set of guidelines for evaluation of methods in this field.
    
[^47]: 图上的持续学习: 一项调查

    Continual Learning on Graphs: A Survey

    [https://arxiv.org/abs/2402.06330](https://arxiv.org/abs/2402.06330)

    本文是一项关于持续图学习的综合调查，其中提出了新的分类法来克服灾难性遗忘问题，并分析了持续性能改进的挑战和可能的解决方案。

    

    最近，在非稳态环境中，持续图学习被越来越多地应用于各种处理图结构数据的任务中。尽管它具有有前途的学习能力，但目前关于持续图学习的研究主要集中在减轻灾难性遗忘问题，而忽略了持续性能的改进。为了弥补这一差距，本文旨在从克服灾难性遗忘的角度提供对最近关于持续图学习的努力的全面调查。具体而言，我们从克服灾难性遗忘的角度，引入了持续图学习的新分类法。此外，我们系统地分析了将这些持续图学习方法应用于持续性能改进的挑战，然后讨论了可能的解决方案。最后，我们提出了关于持续图学习发展的未解决问题和未来发展方向，并讨论了它们如何影响持续性能的改进。

    Recently, continual graph learning has been increasingly adopted for diverse graph-structured data processing tasks in non-stationary environments. Despite its promising learning capability, current studies on continual graph learning mainly focus on mitigating the catastrophic forgetting problem while ignoring continuous performance improvement. To bridge this gap, this article aims to provide a comprehensive survey of recent efforts on continual graph learning. Specifically, we introduce a new taxonomy of continual graph learning from the perspective of overcoming catastrophic forgetting. Moreover, we systematically analyze the challenges of applying these continual graph learning methods in improving performance continuously and then discuss the possible solutions. Finally, we present open issues and future directions pertaining to the development of continual graph learning and discuss how they impact continuous performance improvement.
    
[^48]: 时间交互图上的提示学习

    Prompt Learning on Temporal Interaction Graphs

    [https://arxiv.org/abs/2402.06326](https://arxiv.org/abs/2402.06326)

    这个论文提出了一种在时间交互图上进行提示学习的方法，以解决当前模型在预训练和下游预测阶段所面临的时间差异和语义差异的问题。

    

    时间交互图(TIGs)被广泛用于表示真实世界系统。为了促进在TIGs上的表示学习，研究人员提出了一系列的TIG模型。然而，这些模型在“预训练，预测”训练范式中依然面临着两个难题。首先，预训练和推理数据之间的时间差异严重削弱了模型在动态演化数据上进行遥远未来预测的适用性。其次，预文本任务和下游任务之间的语义差异阻碍了它们在实际应用中的使用，因为它们在应用场景中很难对齐其学习和预测能力。

    Temporal Interaction Graphs (TIGs) are widely utilized to represent real-world systems. To facilitate representation learning on TIGs, researchers have proposed a series of TIG models. However, these models are still facing two tough gaps between the pre-training and downstream predictions in their ``pre-train, predict'' training paradigm. First, the temporal discrepancy between the pre-training and inference data severely undermines the models' applicability in distant future predictions on the dynamically evolving data. Second, the semantic divergence between pretext and downstream tasks hinders their practical applications, as they struggle to align with their learning and prediction capabilities across application scenarios.   Recently, the ``pre-train, prompt'' paradigm has emerged as a lightweight mechanism for model generalization. Applying this paradigm is a potential solution to solve the aforementioned challenges. However, the adaptation of this paradigm to TIGs is not straig
    
[^49]: 均匀随机权重如何引起不均匀偏差：典型插值神经网络与窄教师的普遍性

    How Uniform Random Weights Induce Non-uniform Bias: Typical Interpolating Neural Networks Generalize with Narrow Teachers

    [https://arxiv.org/abs/2402.06323](https://arxiv.org/abs/2402.06323)

    在插值神经网络中，均匀随机权重可以产生非均匀偏差，因此通常插值神经网络会与窄教师NN一样很好地泛化。

    

    背景。一个主要的理论难题是当神经网络被训练到零误差（即插值数据）时，为什么超参数化神经网络（NN）能够很好地泛化。通常，NN是使用随机梯度下降（SGD）或其变种之一训练的。然而，最近的实证研究检验了从看似均匀的参数先验中采样的随机NN对数据的泛化能力：该NN对训练集进行了完美分类。有趣的是，这样的NN样本通常像SGD训练的NN一样泛化良好。贡献。我们证明了如果存在与标签一致的窄“教师NN”，那么这样的随机NN插值器通常能很好地泛化。具体而言，我们证明了在NN参数化中的“平坦”先验通过NN结构中的冗余引入了丰富的NN函数先验。特别是，这会对较简单的函数产生偏向，这些函数需要较少的相关参数。

    Background. A main theoretical puzzle is why over-parameterized Neural Networks (NNs) generalize well when trained to zero loss (i.e., so they interpolate the data). Usually, the NN is trained with Stochastic Gradient Descent (SGD) or one of its variants. However, recent empirical work examined the generalization of a random NN that interpolates the data: the NN was sampled from a seemingly uniform prior over the parameters, conditioned on that the NN perfectly classifying the training set. Interestingly, such a NN sample typically generalized as well as SGD-trained NNs.   Contributions. We prove that such a random NN interpolator typically generalizes well if there exists an underlying narrow ``teacher NN" that agrees with the labels. Specifically, we show that such a `flat' prior over the NN parametrization induces a rich prior over the NN functions, due to the redundancy in the NN structure. In particular, this creates a bias towards simpler functions, which require less relevant pa
    
[^50]: 粒子去噪扩散采样器

    Particle Denoising Diffusion Sampler

    [https://arxiv.org/abs/2402.06320](https://arxiv.org/abs/2402.06320)

    本文介绍了一种粒子去噪扩散采样器（PDDS），通过使用原始迭代粒子方案和新颖的得分匹配损失，对非归一化概率密度进行采样和计算规范化常数。与标准的去噪扩散模型不同，PDDS 在温和假设下提供了渐近一致的估计。

    

    去噪扩散模型在生成建模中已经得到广泛应用。其核心思想是通过使用扩散将数据分布转化为高斯分布。然后通过使用得分匹配思想估计这种扩散的时间反演来获得来自数据分布的近似样本。我们在这里采用类似的策略来从非归一化概率密度中采样并计算它们的规范化常数。然而，在这里，时间反演扩散是通过使用基于新颖得分匹配损失的原始迭代粒子方案来模拟的。与标准的去噪扩散模型不同，结果的粒子去噪扩散采样器 (PDDS) 在温和假设下提供了渐近一致的估计。我们在多模态和高维采样任务上演示了 PDDS。

    Denoising diffusion models have become ubiquitous for generative modeling. The core idea is to transport the data distribution to a Gaussian by using a diffusion. Approximate samples from the data distribution are then obtained by estimating the time-reversal of this diffusion using score matching ideas. We follow here a similar strategy to sample from unnormalized probability densities and compute their normalizing constants. However, the time-reversed diffusion is here simulated by using an original iterative particle scheme relying on a novel score matching loss. Contrary to standard denoising diffusion models, the resulting Particle Denoising Diffusion Sampler (PDDS) provides asymptotically consistent estimates under mild assumptions. We demonstrate PDDS on multimodal and high dimensional sampling tasks.
    
[^51]: TimEHR：用于电子健康记录的基于图像的时间序列生成

    TimEHR: Image-based Time Series Generation for Electronic Health Records

    [https://arxiv.org/abs/2402.06318](https://arxiv.org/abs/2402.06318)

    提出了一种新的基于生成对抗网络的模型TimEHR，用于从EHR生成时间序列数据。通过将时间序列视为图像，并使用两个条件GAN，TimEHR在处理不规则采样、缺失值和高维度方面取得了优于现有方法的结果。

    

    电子健康记录（EHR）中的时间序列对生成模型提出了独特的挑战，如不规则采样，缺失值和高维度。在本文中，我们提出了一种新颖的生成对抗网络（GAN）模型TimEHR，用于从EHR生成时间序列数据。具体而言，TimEHR将时间序列视为图像，基于两个条件GAN构建。第一个GAN生成缺失模式，第二个GAN根据缺失模式生成时间序列值。在三个真实世界的EHR数据集上的实验结果表明，TimEHR在保真度，实用性和隐私度量方面优于现有的方法。

    Time series in Electronic Health Records (EHRs) present unique challenges for generative models, such as irregular sampling, missing values, and high dimensionality. In this paper, we propose a novel generative adversarial network (GAN) model, TimEHR, to generate time series data from EHRs. In particular, TimEHR treats time series as images and is based on two conditional GANs. The first GAN generates missingness patterns, and the second GAN generates time series values based on the missingness pattern. Experimental results on three real-world EHR datasets show that TimEHR outperforms state-of-the-art methods in terms of fidelity, utility, and privacy metrics.
    
[^52]: 多模态可解释的数据驱动模型用于预测多重抗菌药物耐药性的早期预测的研究

    Multimodal Interpretable Data-Driven Models for Early Prediction of Antimicrobial Multidrug Resistance Using Multivariate Time-Series

    [https://arxiv.org/abs/2402.06295](https://arxiv.org/abs/2402.06295)

    本研究提出了一种基于可解释的多模态数据驱动模型的方法，通过静态数据和多元时间序列模型预测和理解重症监护病房中抗菌药物多重耐药性细菌的出现。

    

    电子健康记录（EHR）是患者健康状况的多模态注册，包括静态数据和多元时间序列（MTS）。虽然MTS是临床预测的有价值工具，但将其与其他数据模态融合可能会带来更深入的洞察和更准确的结果。深度神经网络（DNNs）已成为识别和定义医疗领域潜在模式的基本工具。然而，DNN模型在临床环境中广泛应用还需要基本改进的可解释性。在这项研究中，我们提出了一种建立在可解释的多模态数据驱动模型集合上的方法，可以预测和理解马德里训拉布拉达大学医院（西班牙）的重症监护病房（ICU）中抗菌药物多重耐药性（AMR）细菌的出现。患者的个人资料和初始健康状况使用静态变量进行建模，而演变过程使用MTS进行建模。

    Electronic health records (EHR) is an inherently multimodal register of the patient's health status characterized by static data and multivariate time series (MTS). While MTS are a valuable tool for clinical prediction, their fusion with other data modalities can possibly result in more thorough insights and more accurate results. Deep neural networks (DNNs) have emerged as fundamental tools for identifying and defining underlying patterns in the healthcare domain. However, fundamental improvements in interpretability are needed for DNN models to be widely used in the clinical setting. In this study, we present an approach built on a collection of interpretable multimodal data-driven models that may anticipate and understand the emergence of antimicrobial multidrug resistance (AMR) germs in the intensive care unit (ICU) of the University Hospital of Fuenlabrada (Madrid, Spain). The profile and initial health status of the patient are modeled using static variables, while the evolution 
    
[^53]: 通过条件流进行不规则时间序列的概率预测

    Probabilistic Forecasting of Irregular Time Series via Conditional Flows

    [https://arxiv.org/abs/2402.06293](https://arxiv.org/abs/2402.06293)

    该论文提出了一种使用条件流进行不规则时间序列的概率预测的新模型ProFITi。该模型通过学习条件下未来值的联合分布，对具有缺失值的不规则时间序列进行预测，而不假设底层分布的固定形状。通过引入可逆三角形注意力层和可逆非线性激活函数，该模型取得了良好的实验结果。

    

    不规则采样的多变量时间序列具有缺失值的概率预测是许多领域的重要问题，包括医疗保健、天文学和气候学。目前该任务的最先进方法仅估计单个通道和单个时间点上观测值的边际分布，假设了一个固定形状的参数分布。在这项工作中，我们提出了一种新的模型ProFITi，用于使用条件归一化流对具有缺失值的不规则采样时间序列进行概率预测。该模型学习了在过去观测和查询的通道和时间上条件下时间序列未来值的联合分布，而不假设底层分布的固定形状。作为模型组件，我们引入了一种新颖的可逆三角形注意力层和一个可逆的非线性激活函数，能够在整个实数线上进行转换。我们在四个数据集上进行了大量实验，并证明了该模型的提议。

    Probabilistic forecasting of irregularly sampled multivariate time series with missing values is an important problem in many fields, including health care, astronomy, and climate. State-of-the-art methods for the task estimate only marginal distributions of observations in single channels and at single timepoints, assuming a fixed-shape parametric distribution. In this work, we propose a novel model, ProFITi, for probabilistic forecasting of irregularly sampled time series with missing values using conditional normalizing flows. The model learns joint distributions over the future values of the time series conditioned on past observations and queried channels and times, without assuming any fixed shape of the underlying distribution. As model components, we introduce a novel invertible triangular attention layer and an invertible non-linear activation function on and onto the whole real line. We conduct extensive experiments on four datasets and demonstrate that the proposed model pro
    
[^54]: 在联邦学习中评估成员推断攻击和防御

    Evaluating Membership Inference Attacks and Defenses in Federated Learning

    [https://arxiv.org/abs/2402.06289](https://arxiv.org/abs/2402.06289)

    这篇论文评估了联邦学习中成员推断攻击和防御的情况。评估揭示了两个重要发现：多时序的模型信息有助于提高攻击的有效性，多空间的模型信息有助于提高攻击的效果。这篇论文还评估了两种防御机制的效用和隐私权衡。

    

    成员推断攻击(MIAs)对于隐私保护的威胁在联邦学习中日益增长。半诚实的攻击者，例如服务器，可以根据观察到的模型信息确定一个特定样本是否属于目标客户端。本文对现有的MIAs和相应的防御策略进行了评估。我们对MIAs的评估揭示了两个重要发现。首先，结合多个通信轮次的模型信息(多时序)相比于利用单个时期的模型信息提高了MIAs的整体有效性。其次，在非目标客户端(Multi-spatial)中融入模型显著提高了MIAs的效果，特别是当客户端的数据是同质的时候。这凸显了在MIAs中考虑时序和空间模型信息的重要性。接下来，我们通过隐私-效用权衡评估了两种类型的防御机制对MIAs的有效性。

    Membership Inference Attacks (MIAs) pose a growing threat to privacy preservation in federated learning. The semi-honest attacker, e.g., the server, may determine whether a particular sample belongs to a target client according to the observed model information. This paper conducts an evaluation of existing MIAs and corresponding defense strategies. Our evaluation on MIAs reveals two important findings about the trend of MIAs. Firstly, combining model information from multiple communication rounds (Multi-temporal) enhances the overall effectiveness of MIAs compared to utilizing model information from a single epoch. Secondly, incorporating models from non-target clients (Multi-spatial) significantly improves the effectiveness of MIAs, particularly when the clients' data is homogeneous. This highlights the importance of considering the temporal and spatial model information in MIAs. Next, we assess the effectiveness via privacy-utility tradeoff for two type defense mechanisms against MI
    
[^55]: AI，与人相遇：混合决策系统的学习范式

    AI, Meet Human: Learning Paradigms for Hybrid Decision Making Systems

    [https://arxiv.org/abs/2402.06287](https://arxiv.org/abs/2402.06287)

    本调查提出了混合决策系统的分类方法，为理解如何对人与机器之间的交互进行建模提供了概念性和技术性的框架。

    

    每天，我们越来越多地依赖机器学习模型来自动化和支持高风险任务和决策。这种日益增长的存在意味着人类现在不断与基于机器学习的系统进行互动，每天进行模型的培训和使用。计算机科学文献中有几种不同的技术来考虑人与机器学习系统的交互，但其分类稀疏且目标各异。本调查提出了混合决策系统的分类方法，为理解当前计算机科学文献如何对人与机器之间的交互进行建模提供了概念性和技术性的框架。

    Everyday we increasingly rely on machine learning models to automate and support high-stake tasks and decisions. This growing presence means that humans are now constantly interacting with machine learning-based systems, training and using models everyday. Several different techniques in computer science literature account for the human interaction with machine learning systems, but their classification is sparse and the goals varied. This survey proposes a taxonomy of Hybrid Decision Making Systems, providing both a conceptual and technical framework for understanding how current computer science literature models interaction between humans and machines.
    
[^56]: 获取、合并、预测：通过数据湖增强表格

    Retrieve, Merge, Predict: Augmenting Tables with Data Lakes

    [https://arxiv.org/abs/2402.06282](https://arxiv.org/abs/2402.06282)

    本文通过对数据湖中的数据发现进行深入分析，着重于表格增强，提出了准确检索连接候选人的重要性和简单合并方法的效率，以及现有解决方案的好处和局限性。

    

    我们对数据湖中的数据发现进行了深入分析，重点是给定机器学习任务的表格增强。我们分析了三个主要步骤中使用的替代方法：检索可连接的表格、合并信息和预测结果表格。作为数据湖，本文使用了YADL（另一个数据湖）-我们开发的一种用于基准测试此数据发现任务的新型数据集-和Open Data US，一个被引用的真实数据湖。通过对这两个数据湖的系统性探索，我们的研究概述了准确检索连接候选人的重要性以及简单合并方法的效率。我们报告了现有解决方案的好处和局限性，旨在指导未来的研究。

    We present an in-depth analysis of data discovery in data lakes, focusing on table augmentation for given machine learning tasks. We analyze alternative methods used in the three main steps: retrieving joinable tables, merging information, and predicting with the resultant table. As data lakes, the paper uses YADL (Yet Another Data Lake) -- a novel dataset we developed as a tool for benchmarking this data discovery task -- and Open Data US, a well-referenced real data lake. Through systematic exploration on both lakes, our study outlines the importance of accurately retrieving join candidates and the efficiency of simple merging methods. We report new insights on the benefits of existing solutions and on their limitations, aiming at guiding future research in this space.
    
[^57]: 使用生成扩散模型实现可控地震速度合成

    Controllable seismic velocity synthesis using generative diffusion models

    [https://arxiv.org/abs/2402.06277](https://arxiv.org/abs/2402.06277)

    本论文提出使用生成扩散模型进行地震速度合成，通过纳入先验信息，可以生成与实验数据密切匹配的地震速度。

    

    准确的地震速度估计对于理解地球的地下结构、评估自然资源和评估地震危害至关重要。基于机器学习的反演算法在区域（例如勘探）和全球速度估计方面表现出有希望的性能，但其有效性依赖于训练数据集的规模和多样性，以覆盖目标解的分布。此外，提高速度估计的精度和可靠性还需要纳入先验信息，例如地质类别、钻井记录和地下结构，但目前的统计或神经网络方法对于处理多模态信息并不够灵活。为了解决这两个挑战，我们提出使用条件生成扩散模型进行地震速度合成，在其中我们可容易地纳入这些先验信息。这种方法可以生成与实验数据密切匹配的地震速度。

    Accurate seismic velocity estimations are vital to understanding Earth's subsurface structures, assessing natural resources, and evaluating seismic hazards. Machine learning-based inversion algorithms have shown promising performance in regional (i.e., for exploration) and global velocity estimation, while their effectiveness hinges on access to large and diverse training datasets whose distributions generally cover the target solutions. Additionally, enhancing the precision and reliability of velocity estimation also requires incorporating prior information, e.g., geological classes, well logs, and subsurface structures, but current statistical or neural network-based methods are not flexible enough to handle such multi-modal information. To address both challenges, we propose to use conditional generative diffusion models for seismic velocity synthesis, in which we readily incorporate those priors. This approach enables the generation of seismic velocities that closely match the expe
    
[^58]: 高斯过程下安全的时间序列建模的主动学习

    Safe Active Learning for Time-Series Modeling with Gaussian Processes

    [https://arxiv.org/abs/2402.06276](https://arxiv.org/abs/2402.06276)

    本研究提出了一种安全的主动学习方法，用于时间序列建模。通过动态探索输入空间并根据安全要求和过去观察的输入和输出轨迹，我们的方法在现实技术应用中展示了其有效性。

    

    学习时间序列模型对于许多应用如模拟和预测都是有用的。在本研究中，我们考虑了在考虑给定的安全性约束条件的情况下主动学习时间序列模型的问题。对于时间序列建模，我们使用了一个具有非线性外部输入结构的高斯过程。所提出的方法通过动态地探索输入空间来生成适用于时间序列模型学习的数据，即输入和输出轨迹。该方法将输入轨迹参数化为连续的轨迹部分，这些部分是根据安全要求和过去的观察逐步确定的。我们对所提出的算法进行分析，并在技术应用上进行了实证评估。结果显示了我们的方法在现实技术使用案例下的有效性。

    Learning time-series models is useful for many applications, such as simulation and forecasting. In this study, we consider the problem of actively learning time-series models while taking given safety constraints into account. For time-series modeling we employ a Gaussian process with a nonlinear exogenous input structure. The proposed approach generates data appropriate for time series model learning, i.e. input and output trajectories, by dynamically exploring the input space. The approach parametrizes the input trajectory as consecutive trajectory sections, which are determined stepwise given safety requirements and past observations. We analyze the proposed algorithm and evaluate it empirically on a technical application. The results show the effectiveness of our approach in a realistic technical use case.
    
[^59]: 神经SPH: 改进的拉格朗日流体动力学神经建模

    Neural SPH: Improved Neural Modeling of Lagrangian Fluid Dynamics

    [https://arxiv.org/abs/2402.06275](https://arxiv.org/abs/2402.06275)

    本研究发展了一种叫做神经SPH的方法，通过增强图神经网络和标准SPH求解器的组合来改进GNN模拟器的性能，在准确建模物理现象方面取得了较好的效果。

    

    平滑粒子流体动力学（SPH）在现代工程和科学领域中无处不在。SPH是一类通过有限材料点对流体动力学进行离散化处理的拉格朗日方案，通过跟踪这些材料点来追踪其演变的速度场。由于仿真的粒子特性，图神经网络（GNN）已经成为一种具有吸引力和成功的替代方法。然而，这种基于GNN的模拟器的实际实用性依赖于其对物理模型的准确建模能力，在长时间范围内提供准确且稳定的预测，这是一个众所周知的难题。在这项工作中，我们确定了张力不稳定性导致的粒子聚类现象是主要问题之一。基于这些见解，我们用标准SPH求解器的各个组成部分（包括压力、粘性和外力部分）增强了最先进的基于GNN的模拟器的训练和推断。所有经过神经SPH增强的模拟器都取得了更好的性能。

    Smoothed particle hydrodynamics (SPH) is omnipresent in modern engineering and scientific disciplines. SPH is a class of Lagrangian schemes that discretize fluid dynamics via finite material points that are tracked through the evolving velocity field. Due to the particle-like nature of the simulation, graph neural networks (GNNs) have emerged as appealing and successful surrogates. However, the practical utility of such GNN-based simulators relies on their ability to faithfully model physics, providing accurate and stable predictions over long time horizons - which is a notoriously hard problem. In this work, we identify particle clustering originating from tensile instabilities as one of the primary pitfalls. Based on these insights, we enhance both training and rollout inference of state-of-the-art GNN-based simulators with varying components from standard SPH solvers, including pressure, viscous, and external force components. All neural SPH-enhanced simulators achieve better perfor
    
[^60]: 适应性近端梯度方法在没有近似的情况下是通用的

    Adaptive proximal gradient methods are universal without approximation

    [https://arxiv.org/abs/2402.06271](https://arxiv.org/abs/2402.06271)

    该论文证明了适应性近端梯度方法对于凸问题不受传统假设的限制，并且可以在局部梯度H\"older连续性条件下收敛，同时避免了线搜索步骤和近似的使用。对局部H\"older常数和H\"older连续性顺序的先验知识也不是必需的。在数值实验中，与基准方法进行了对比实验，涵盖了局部和全局 H\"older 设置。

    

    我们证明，对于凸问题，适应性近端梯度方法不受传统利普希兹假设的限制。我们的分析揭示了一类无需线搜索的方法在仅具有局部H\"older梯度连续性的情况下仍然收敛，特别适用于连续可微的半代数函数。为了弥补缺乏局部利普希兹连续性的问题，常见的方法包括$\varepsilon$-oracle和/或线搜索步骤。相反，我们利用普通的H\"older不等式而不涉及任何近似，同时保持适应性方案无需线搜索的特性。此外，我们证明了不需要先验知识的局部H\"older常数或H\"older连续性的顺序，也可以实现完全的序列收敛性。在数值实验中，我们对机器学习中的不同任务进行了基准方法的比较，涵盖了局部和全局 H\"older 设置。

    We show that adaptive proximal gradient methods for convex problems are not restricted to traditional Lipschitzian assumptions. Our analysis reveals that a class of linesearch-free methods is still convergent under mere local H\"older gradient continuity, covering in particular continuously differentiable semi-algebraic functions. To mitigate the lack of local Lipschitz continuity, popular approaches revolve around $\varepsilon$-oracles and/or linesearch procedures. In contrast, we exploit plain H\"older inequalities not entailing any approximation, all while retaining the linesearch-free nature of adaptive schemes. Furthermore, we prove full sequence convergence without prior knowledge of local H\"older constants nor of the order of H\"older continuity. In numerical experiments we present comparisons to baseline methods on diverse tasks from machine learning covering both the locally and the globally H\"older setting.
    
[^61]: YAMLE：又一个机器学习环境

    YAMLE: Yet Another Machine Learning Environment

    [https://arxiv.org/abs/2402.06268](https://arxiv.org/abs/2402.06268)

    YAMLE是一个开源机器学习环境，旨在减少重复工作并提高机器学习研究的可重现性。它包括命令行界面和与PyTorch库的集成，致力于成为一个共享的生态系统。

    

    YAMLE：又一个机器学习环境是一个开源框架，它促进了机器学习模型和方法的快速原型设计和实验。其主要动机是在实现新方法时减少重复工作，并提高机器学习研究的可重现性。YAMLE包括一个命令行界面以及与流行且维护良好的基于PyTorch的库的集成，以简化训练、超参数优化和日志记录。YAMLE的雄心壮志是发展成一个共享的生态系统，研究人员和实践者可以快速构建和比较现有的实现。在https://github.com/martinferianc/yamle找到它。

    YAMLE: Yet Another Machine Learning Environment is an open-source framework that facilitates rapid prototyping and experimentation with machine learning (ML) models and methods. The key motivation is to reduce repetitive work when implementing new approaches and improve reproducibility in ML research. YAMLE includes a command-line interface and integrations with popular and well-maintained PyTorch-based libraries to streamline training, hyperparameter optimisation, and logging. The ambition for YAMLE is to grow into a shared ecosystem where researchers and practitioners can quickly build on and compare existing implementations. Find it at: https://github.com/martinferianc/yamle.
    
[^62]: 基于价值的多目标强化学习中的价值函数干扰和贪婪动作选择

    Value function interference and greedy action selection in value-based multi-objective reinforcement learning

    [https://arxiv.org/abs/2402.06266](https://arxiv.org/abs/2402.06266)

    基于价值的多目标强化学习中，如果用户的效用函数将广泛变化的向量值映射为相似的效用水平，会导致价值函数干扰并收敛到次优策略。

    

    多目标强化学习（MORL）算法将传统的强化学习（RL）扩展到具有多个相互冲突目标的更一般情况下，这些目标由向量值奖励表示。广泛使用的标量RL方法（如Q学习）可以通过（1）学习向量值的价值函数和（2）使用反映用户对不同目标的效用的标量化或排序算子来处理多个目标。然而，正如我们在这里所示，如果用户的效用函数将广泛变化的向量值映射为相似的效用水平，这可能会导致代理学习的价值函数干扰，从而收敛到次优策略。这在优化预期标量化回报准则时在随机环境中最为普遍，但我们提供了一个简单的例子证明干扰也可能在确定性环境中出现。

    Multi-objective reinforcement learning (MORL) algorithms extend conventional reinforcement learning (RL) to the more general case of problems with multiple, conflicting objectives, represented by vector-valued rewards. Widely-used scalar RL methods such as Q-learning can be modified to handle multiple objectives by (1) learning vector-valued value functions, and (2) performing action selection using a scalarisation or ordering operator which reflects the user's utility with respect to the different objectives. However, as we demonstrate here, if the user's utility function maps widely varying vector-values to similar levels of utility, this can lead to interference in the value-function learned by the agent, leading to convergence to sub-optimal policies. This will be most prevalent in stochastic environments when optimising for the Expected Scalarised Return criterion, but we present a simple example showing that interference can also arise in deterministic environments. We demonstrat
    
[^63]: 进取的鲍勃通过提示对抗调整抵制越狱行为

    Studious Bob Fight Back Against Jailbreaking via Prompt Adversarial Tuning

    [https://arxiv.org/abs/2402.06255](https://arxiv.org/abs/2402.06255)

    本文提出了一种名为Prompt Adversarial Tuning (PAT)的方法，通过训练一个防御控制机制并将其作为前缀嵌入到用户提示中，实现对大型语言模型（LLMs）的越狱行为的防御。实验证明该方法在保护LLMs免受产生有害信息的影响方面效果显著。

    

    尽管大型语言模型（LLM）在各种应用中取得了巨大的成功，但它们也容易受到特定提示的影响，从而绕过内置的安全措施并提供危险或非法内容，这种现象被称为越狱行为。为了保护LLMs免受产生有害信息的影响，提出了各种防御策略，其中大多数集中在内容过滤或模型的对抗训练方面。在本文中，我们提出了一种名为Prompt Adversarial Tuning（PAT）的方法，通过训练一个防御控制机制并将其作为前缀嵌入到用户提示中来实现我们的防御策略。我们设计了一个类似对抗训练的训练过程，以实现我们的优化目标，交替更新攻击和防御控制机制。据我们所知，我们是第一个从提示调整的角度实施防御的人。一旦应用，我们的方法几乎不会影响LLMs的操作效率。实验表明我们的方法在抵御越狱行为方面具有良好的效果。

    Although Large Language Models (LLMs) have achieved tremendous success in various applications, they are also susceptible to certain prompts that can induce them to bypass built-in safety measures and provide dangerous or illegal content, a phenomenon known as jailbreak. To protect LLMs from producing harmful information, various defense strategies are proposed, with most focusing on content filtering or adversarial training of models. In this paper, we propose an approach named Prompt Adversarial Tuning (PAT) to train a defense control mechanism, which is then embedded as a prefix to user prompts to implement our defense strategy. We design a training process similar to adversarial training to achieve our optimized goal, alternating between updating attack and defense controls. To our knowledge, we are the first to implement defense from the perspective of prompt tuning. Once employed, our method will hardly impact the operational efficiency of LLMs. Experiments show that our method i
    
[^64]: 使用增强型分层图神经网络实现N-1优化潮流的降维

    N-1 Reduced Optimal Power Flow Using Augmented Hierarchical Graph Neural Network

    [https://arxiv.org/abs/2402.06226](https://arxiv.org/abs/2402.06226)

    本文提出了一种使用增强型分层图神经网络（AHGNN）的方法来预测关键拥塞线路并创建N-1优化潮流的降维（N-1 ROPF）。实验结果表明，AHGNN启用的N-1 ROPF在保持解的质量的同时，能够显著减少计算时间。

    

    优化潮流（OPF）用于在电力系统实时运行中进行发电重调。N-1 OPF可以确保在各种事故情况下安全运行电网。对于变量和约束众多的大型复杂电力网络，实时N-1 OPF的最优解需要大量计算资源。为了解决这个挑战，引入机器学习（ML）作为预测拥塞或负载过重线路的额外工具。本文提出了一种先进的ML模型，称为增强型分层图神经网络（AHGNN），用于预测关键的拥塞线路并创建N-1优化潮流的降维（N-1 ROPF）。所提出的AHGNN启用的N-1 ROPF能够显著减少计算时间，同时保持解的质量。还实现了几种基于GNN的ML模型作为基准，以证明所提出的AHGNN方法的有效性。案例研究证明了所提出的AHGNN的有效性。

    Optimal power flow (OPF) is used to perform generation redispatch in power system real-time operations. N-1 OPF can ensure safe grid operations under diverse contingency scenarios. For large and intricate power networks with numerous variables and constraints, achieving an optimal solution for real-time N-1 OPF necessitates substantial computational resources. To mitigate this challenge, machine learning (ML) is introduced as an additional tool for predicting congested or heavily loaded lines dynamically. In this paper, an advanced ML model known as the augmented hierarchical graph neural network (AHGNN) was proposed to predict critical congested lines and create N-1 reduced OPF (N-1 ROPF). The proposed AHGNN-enabled N-1 ROPF can result in a remarkable reduction in computing time while retaining the solution quality. Several variations of GNN-based ML models are also implemented as benchmark to demonstrate effectiveness of the proposed AHGNN approach. Case studies prove the proposed AH
    
[^65]: 自适应多梯度方法用于拟凸向量优化及其在多任务学习中的应用

    Adaptive multi-gradient methods for quasiconvex vector optimization and applications to multi-task learning

    [https://arxiv.org/abs/2402.06224](https://arxiv.org/abs/2402.06224)

    我们提出了一种自适应步长方法，用于解决一类广泛的非凸多目标规划问题，并应用于创新的多梯度投影方法和多任务学习，展示了其在大规模挑战中的效果。

    

    我们提出了一种自适应步长方法，该方法不包括线搜索技术，用于解决一个广泛的非凸多目标规划问题在一个无界约束集上。我们还证明了在适度的假设下一个通用方法的收敛性。具体来说，目标函数可能不满足凸性标准。与下降线搜索算法不同，它不需要一个初始步长由之前确定的利普希茨常数来确定。过程的主要特点是直到达到预定条件才进行渐进步长的减小。它可以特别应用于为无界约束优化问题提供一种创新的多梯度投影方法。一些计算实例的初步结果证实了该策略的准确性。我们将所提出的技术应用到一些多任务学习实验中，以展示其在大规模挑战中的效果。

    We present an adaptive step-size method, which does not include line-search techniques, for solving a wide class of nonconvex multiobjective programming problems on an unbounded constraint set. We also prove convergence of a general approach under modest assumptions. More specifically, the convexity criterion might not be satisfied by the objective function. Unlike descent line-search algorithms, it does not require an initial step-size to be determined by a previously determined Lipschitz constant. The process's primary characteristic is its gradual step-size reduction up until a predetermined condition is met. It can be specifically applied to offer an innovative multi-gradient projection method for unbounded constrained optimization issues. Preliminary findings from a few computational examples confirm the accuracy of the strategy. We apply the proposed technique to some multi-task learning experiments to show its efficacy for large-scale challenges.
    
[^66]: 通过潜在部分因果模型揭示多模式对比表示学习

    Revealing Multimodal Contrastive Representation Learning through Latent Partial Causal Models

    [https://arxiv.org/abs/2402.06223](https://arxiv.org/abs/2402.06223)

    通过潜在部分因果模型，我们展示了多模式对比表示学习在识别潜在耦合变量方面的优秀能力，并揭示了预训练的多模态模型通过线性独立分量分析学习分离表示的潜力。

    

    多模式对比表示学习方法在各个领域取得了成功，部分原因是由于它们能够生成复杂现象的有意义的共享表示。为了增强对这些获得的表示的深度分析和理解，我们引入了一种特别针对多模态数据设计的统一因果模型。通过研究这个模型，我们展示了多模式对比表示学习在识别在提出的统一模型中的潜在耦合变量方面的优秀能力，即使在不同假设下导致的线性或置换变换。我们的发现揭示了预训练的多模态模型（如CLIP）通过线性独立分量分析这一令人惊讶的简单而高效的工具学习分离表示的潜力。实验证明了我们发现的鲁棒性，即使在被违反假设的情况下，也验证了所提出方法在学习疾病方面的有效性。

    Multimodal contrastive representation learning methods have proven successful across a range of domains, partly due to their ability to generate meaningful shared representations of complex phenomena. To enhance the depth of analysis and understanding of these acquired representations, we introduce a unified causal model specifically designed for multimodal data. By examining this model, we show that multimodal contrastive representation learning excels at identifying latent coupled variables within the proposed unified model, up to linear or permutation transformations resulting from different assumptions. Our findings illuminate the potential of pre-trained multimodal models, eg, CLIP, in learning disentangled representations through a surprisingly simple yet highly effective tool: linear independent component analysis. Experiments demonstrate the robustness of our findings, even when the assumptions are violated, and validate the effectiveness of the proposed method in learning dise
    
[^67]: 伯克利单细胞计算显微镜（BSCCM）数据集

    The Berkeley Single Cell Computational Microscopy (BSCCM) Dataset

    [https://arxiv.org/abs/2402.06191](https://arxiv.org/abs/2402.06191)

    伯克利单细胞计算显微镜（BSCCM）数据集包含了约12,000,000张个体白血细胞的图像，提供了用于计算显微镜和计算机视觉算法开发和测试的宝贵资源。

    

    计算显微镜，即硬件和算法的联合设计，显示出降低成本、更稳健地执行和收集新类型信息的潜力。计算显微镜系统的性能，特别是那些融合了机器学习的系统，通常是样本相关的。因此，标准数据集是比较不同方法性能的重要工具。在这里，我们介绍伯克利单细胞计算显微镜（BSCCM）数据集，其中包含了约12,000,000张个体白血细胞的图像。该数据集是在LED阵列显微镜上采用多种照明模式拍摄的图像，并且包含了标记不同细胞类型表面蛋白的荧光测量结果。我们希望这个数据集能为计算显微镜和计算机视觉中新算法的开发和测试提供宝贵资源。

    Computational microscopy, in which hardware and algorithms of an imaging system are jointly designed, shows promise for making imaging systems that cost less, perform more robustly, and collect new types of information. Often, the performance of computational imaging systems, especially those that incorporate machine learning, is sample-dependent. Thus, standardized datasets are an essential tool for comparing the performance of different approaches. Here, we introduce the Berkeley Single Cell Computational Microscopy (BSCCM) dataset, which contains over ~12,000,000 images of 400,000 of individual white blood cells. The dataset contains images captured with multiple illumination patterns on an LED array microscope and fluorescent measurements of the abundance of surface proteins that mark different cell types. We hope this dataset will provide a valuable resource for the development and testing of new algorithms in computational microscopy and computer vision with practical biomedical 
    
[^68]: Masked LoGoNet：用于医学领域的快速准确3D图像分析

    Masked LoGoNet: Fast and Accurate 3D Image Analysis for Medical Domain

    [https://arxiv.org/abs/2402.06190](https://arxiv.org/abs/2402.06190)

    本文介绍了一种名为LoGoNet的新型神经网络架构，采用自监督学习方法来应对医学图像分析中的挑战。LoGoNet通过采用大内核注意力和双重编码策略，灵活捕捉长、短距离特征相关性。这种创新的组合技术在医学图像分割中特别有益。

    

    标准的现代机器学习图像方法在医学应用中面临挑战，因为数据集构建的高成本和有限的标记训练数据。此外，这些方法在部署时通常用于每天处理大量数据，给医疗设施带来高维护成本。在本文中，我们引入了一种新的神经网络架构LoGoNet，采用定制的自监督学习（SSL）方法来缓解这些挑战。LoGoNet在U形架构内整合了一种新颖的特征提取器，利用大内核注意力（LKA）和双重编码策略，灵活地捕捉长、短距离特征相关性。这与现有方法依赖增加网络容量以增强特征提取的方式形成对比。我们模型中这些新技术的组合在医学图像分割中特别有益，考虑到其困难性。

    Standard modern machine-learning-based imaging methods have faced challenges in medical applications due to the high cost of dataset construction and, thereby, the limited labeled training data available. Additionally, upon deployment, these methods are usually used to process a large volume of data on a daily basis, imposing a high maintenance cost on medical facilities. In this paper, we introduce a new neural network architecture, termed LoGoNet, with a tailored self-supervised learning (SSL) method to mitigate such challenges. LoGoNet integrates a novel feature extractor within a U-shaped architecture, leveraging Large Kernel Attention (LKA) and a dual encoding strategy to capture both long-range and short-range feature dependencies adeptly. This is in contrast to existing methods that rely on increasing network capacity to enhance feature extraction. This combination of novel techniques in our model is especially beneficial in medical image segmentation, given the difficulty of le
    
[^69]: 一个自监督学习框架用于学习整个切片的表示

    A self-supervised framework for learning whole slide representations

    [https://arxiv.org/abs/2402.06188](https://arxiv.org/abs/2402.06188)

    这个论文提出了一个自监督学习框架（S3L），用于学习整个切片的表示。它结合了变压器模型的视觉和语言建模策略，通过生成配对视图进行自监督学习，以实现高质量的WSI视觉特征学习。

    

    整个切片成像对于生物医学显微镜和计算病理学至关重要。然而，由于其千兆像素的大小、多样的组织病理学特征、空间异质性以及有限的/不存在的数据注释，整个切片图像 (WSIs) 构成了一个复杂的计算机视觉挑战。这些挑战突显了仅依靠监督训练可能导致次优的整个切片表示。自监督表示学习可以为下游诊断任务（如癌症诊断或分子遗传预测）实现高质量的WSI视觉特征学习。在这里，我们提出了一个通用的自监督整个切片学习（S3L）框架，用于千兆像素规模的WSI自监督。S3L将来自基于变压器的视觉和语言建模的数据转换策略结合到一个统一的框架中，以生成用于自监督的配对视图。S3L利用内在的区域异质性、组织学特征的可变性和信息冗余性

    Whole slide imaging is fundamental to biomedical microscopy and computational pathology. However, whole slide images (WSIs) present a complex computer vision challenge due to their gigapixel size, diverse histopathologic features, spatial heterogeneity, and limited/absent data annotations. These challenges highlight that supervised training alone can result in suboptimal whole slide representations. Self-supervised representation learning can achieve high-quality WSI visual feature learning for downstream diagnostic tasks, such as cancer diagnosis or molecular genetic prediction. Here, we present a general self-supervised whole slide learning (S3L) framework for gigapixel-scale self-supervision of WSIs. S3L combines data transformation strategies from transformer-based vision and language modeling into a single unified framework to generate paired views for self-supervision. S3L leverages the inherent regional heterogeneity, histologic feature variability, and information redundancy wi
    
[^70]: Premier-TACO: 通过时间驱动的对比损失进行多任务表示预训练

    Premier-TACO: Pretraining Multitask Representation via Temporal Action-Driven Contrastive Loss

    [https://arxiv.org/abs/2402.06187](https://arxiv.org/abs/2402.06187)

    Premier-TACO是一种多任务特征表示学习方法，通过预训练通用特征表示，并引入负例抽样策略来提高时序行动对比学习的计算效率，从而显著增强了对新颖动作的少样本模仿学习的效果。

    

    我们提出了Premier-TACO，这是一种多任务特征表示学习方法，旨在提高顺序决策任务中少样本策略学习的效率。Premier-TACO利用一部分多任务离线数据集进行预训练通用特征表示，该特征表示捕捉了关键的环境动力学，并使用最少的专家演示进行微调。它通过引入一种新的负例抽样策略推动了时序行动对比学习（TACO）目标的发展，TACO在视觉控制任务中具有最先进的结果。这种策略在显著提高TACO的计算效率方面非常重要，使大规模多任务离线预训练成为可能。我们在包括Deepmind Control Suite、MetaWorld和LIBERO在内的各种连续控制基准测试中进行了广泛的实证评估，证明了Premier-TACO在预训练视觉表示方面的有效性，显著增强了对新颖动作的少样本模仿学习。

    We present Premier-TACO, a multitask feature representation learning approach designed to improve few-shot policy learning efficiency in sequential decision-making tasks. Premier-TACO leverages a subset of multitask offline datasets for pretraining a general feature representation, which captures critical environmental dynamics and is fine-tuned using minimal expert demonstrations. It advances the temporal action contrastive learning (TACO) objective, known for state-of-the-art results in visual control tasks, by incorporating a novel negative example sampling strategy. This strategy is crucial in significantly boosting TACO's computational efficiency, making large-scale multitask offline pretraining feasible. Our extensive empirical evaluation in a diverse set of continuous control benchmarks including Deepmind Control Suite, MetaWorld, and LIBERO demonstrate Premier-TACO's effectiveness in pretraining visual representations, significantly enhancing few-shot imitation learning of nove
    
[^71]: 开发和验证一个人工智能模型，准确预测脊盘盆参数

    Development and validation of an artificial intelligence model to accurately predict spinopelvic parameters

    [https://arxiv.org/abs/2402.06185](https://arxiv.org/abs/2402.06185)

    该研究开发了一个名为SpinePose的人工智能工具，可以准确预测脊盘盆参数，无需手动输入。

    

    目标。研究表明，脊盘盆对齐与临床症状的改善相关。然而，脊盘盆放射学参数的测量费时且观察者之间的一致性值得关注。自动测量工具能够以迅速而一致的方式进行测量，但现有工具仍然受到某种程度的手动输入要求的限制。该研究提出了一种新颖的人工智能工具SpinePose，可以在不需要手动输入的情况下高精度地预测脊盘盆参数。

    Objective. Achieving appropriate spinopelvic alignment has been shown to be associated with improved clinical symptoms. However, measurement of spinopelvic radiographic parameters is time-intensive and interobserver reliability is a concern. Automated measurement tools have the promise of rapid and consistent measurements, but existing tools are still limited by some degree of manual user-entry requirements. This study presents a novel artificial intelligence (AI) tool called SpinePose that automatically predicts spinopelvic parameters with high accuracy without the need for manual entry.   Methods. SpinePose was trained and validated on 761 sagittal whole-spine X-rays to predict sagittal vertical axis (SVA), pelvic tilt (PT), pelvic incidence (PI), sacral slope (SS), lumbar lordosis (LL), T1-pelvic angle (T1PA), and L1-pelvic angle (L1PA). A separate test set of 40 X-rays was labeled by 4 reviewers, including fellowship-trained spine surgeons and a fellowship-trained radiologist with 
    
[^72]: 神经网络可训练性的边界是分形的

    The boundary of neural network trainability is fractal

    [https://arxiv.org/abs/2402.06184](https://arxiv.org/abs/2402.06184)

    本论文通过实验证明，神经网络训练的边界是分形的，对于超参数的微小改变非常敏感，这对于理解神经网络的训练可行性具有重要意义。

    

    一些分形，例如与Mandelbrot和二次Julia集相关的分形，通过迭代函数计算，并识别导致结果序列发散或保持有界的超参数之间的边界。神经网络训练同样涉及迭代更新函数（例如梯度下降的重复步骤），可能导致收敛或发散的行为，并且对超参数的微小改变非常敏感。受到这些相似性的启发，我们实验性地研究了导致稳定和发散训练的神经网络超参数之间的边界。我们发现，所有测试配置中，这个边界在十个数量级以上的尺度范围内都是分形的。

    Some fractals -- for instance those associated with the Mandelbrot and quadratic Julia sets -- are computed by iterating a function, and identifying the boundary between hyperparameters for which the resulting series diverges or remains bounded. Neural network training similarly involves iterating an update function (e.g. repeated steps of gradient descent), can result in convergent or divergent behavior, and can be extremely sensitive to small changes in hyperparameters. Motivated by these similarities, we experimentally examine the boundary between neural network hyperparameters that lead to stable and divergent training. We find that this boundary is fractal over more than ten decades of scale in all tested configurations.
    
[^73]: SMC就是你需要的：并行强扩展

    SMC Is All You Need: Parallel Strong Scaling

    [https://arxiv.org/abs/2402.06173](https://arxiv.org/abs/2402.06173)

    SMC并行扩展方法pSMC具有理论收敛速度，具有有界的时间复杂性和内存要求，适用于贝叶斯推断的问题。

    

    在贝叶斯推断的一般框架中，目标分布只能按比例常数进行评估。传统的一致Bayesian方法，如序贯蒙特卡洛(SMC)和马尔科夫链蒙特卡洛(MCMC)，具有无界的时间复杂性要求。我们开发了一种完全并行的序贯蒙特卡洛(pSMC)方法，可以证明它具有并行强扩展性，即如果允许异步进程数量增长，时间复杂性(和每个节点的内存)仍然保持有界。更具体地说，pSMC具有MSE$=O(1/NR)$的理论收敛速度，其中$N$表示每个处理器中的通信样本数量，$R$表示处理器数量。特别地，对于适当大的问题相关$N$，当$R\rightarrow \infty$时，该方法以固定有限的时间复杂性Cost$=O(1)$收敛到无穷小精度MSE$=O(\varepsilon^2)$，没有效率泄漏，即计算复杂性Cost$=O(\varepsilon)$。

    In the general framework of Bayesian inference, the target distribution can only be evaluated up-to a constant of proportionality. Classical consistent Bayesian methods such as sequential Monte Carlo (SMC) and Markov chain Monte Carlo (MCMC) have unbounded time complexity requirements. We develop a fully parallel sequential Monte Carlo (pSMC) method which provably delivers parallel strong scaling, i.e. the time complexity (and per-node memory) remains bounded if the number of asynchronous processes is allowed to grow. More precisely, the pSMC has a theoretical convergence rate of MSE$ = O(1/NR)$, where $N$ denotes the number of communicating samples in each processor and $R$ denotes the number of processors. In particular, for suitably-large problem-dependent $N$, as $R \rightarrow \infty$ the method converges to infinitesimal accuracy MSE$=O(\varepsilon^2)$ with a fixed finite time-complexity Cost$=O(1)$ and with no efficiency leakage, i.e. computational complexity Cost$=O(\varepsilon
    
[^74]: Pushing Boundaries: Mixup对神经塌陷的影响

    Pushing Boundaries: Mixup's Influence on Neural Collapse

    [https://arxiv.org/abs/2402.06171](https://arxiv.org/abs/2402.06171)

    本研究揭示了Mixup对神经塌陷的影响，通过对深度网络训练数据的最后一层激活进行研究，发现Mixup的最后一层激活收敛到与预期不同的独特配置。

    

    Mixup是一种数据增强策略，它利用训练实例及其相应的标签的凸组合来增强深度神经网络的稳健性和校准性。尽管它被广泛应用，但其成功的细微机制尚未完全理解。观察到的神经塌陷现象，即深度网络的最后一层激活和分类器收敛到一个简单光角紧框架（ETF），为探索mixup是否引发了替代几何配置及其能解释其成功的动机提供了有力支持。在本研究中，我们深入研究了经过mixup处理的深度网络训练数据的最后一层激活，旨在揭示其运行有效性的洞见。我们的调查涵盖了各种架构和数据集对，揭示了mixup的最后一层激活主要收敛到与预期不同的独特配置。

    Mixup is a data augmentation strategy that employs convex combinations of training instances and their respective labels to augment the robustness and calibration of deep neural networks. Despite its widespread adoption, the nuanced mechanisms that underpin its success are not entirely understood. The observed phenomenon of Neural Collapse, where the last-layer activations and classifier of deep networks converge to a simplex equiangular tight frame (ETF), provides a compelling motivation to explore whether mixup induces alternative geometric configurations and whether those could explain its success. In this study, we delve into the last-layer activations of training data for deep networks subjected to mixup, aiming to uncover insights into its operational efficacy. Our investigation, spanning various architectures and dataset pairs, reveals that mixup's last-layer activations predominantly converge to a distinctive configuration different than one might expect. In this configuration,
    
[^75]: 学习对比特征表示来进行面部动作单元检测

    Learning Contrastive Feature Representations for Facial Action Unit Detection

    [https://arxiv.org/abs/2402.06165](https://arxiv.org/abs/2402.06165)

    这项研究提出了一种对比学习框架，通过监督和自监督信号来增强面部动作单元检测模型的性能。采用正样本抽样和权衡重要性的损失函数来应对噪声AU标签和AU类型分布不平衡的挑战。

    

    面部动作单元（AU）检测的主要方法涉及监督的多标签二进制分类问题。现有的方法常常对AU的像素级信息进行编码，从而对模型的复杂性和表达能力提出了很大的要求。此外，由于存在噪声AU标签，这种做法增加了过拟合的风险。在本研究中，我们引入了一个对比学习框架，通过监督和自监督信号增强。目标是在AU检测领域中摆脱传统的像素级学习范式，获得判别特征。为了应对噪声AU标签带来的挑战，我们通过引入自监督信号来增强监督信号。这种增强是通过正样本抽样实现的，包括三种不同类型的正样本对。另外，为了减轻每个AU类型的分布不平衡问题，我们采用了一种权衡重要性的损失函数。

    The predominant approach to facial action unit (AU) detection revolves around a supervised multi-label binary classification problem. Existing methodologies often encode pixel-level information of AUs, thereby imposing substantial demands on model complexity and expressiveness. Moreover, this practice elevates the susceptibility to overfitting due to the presence of noisy AU labels. In the present study, we introduce a contrastive learning framework enhanced by both supervised and self-supervised signals. The objective is to acquire discriminative features, deviating from the conventional pixel-level learning paradigm within the domain of AU detection. To address the challenge posed by noisy AU labels, we augment the supervised signal through the introduction of a self-supervised signal. This augmentation is achieved through positive sample sampling, encompassing three distinct types of positive sample pairs. Furthermore, to mitigate the imbalanced distribution of each AU type, we empl
    
[^76]: Wasserstein近端算子描述基于分数的生成模型并解决记忆问题

    Wasserstein proximal operators describe score-based generative models and resolve memorization

    [https://arxiv.org/abs/2402.06162](https://arxiv.org/abs/2402.06162)

    该论文研究了基于分数的生成模型的数学结构，通过Wasserstein近端算子和平均场博弈可以描述生成模型的归纳偏差，通过解耦合的偏微分方程可以获得优化条件，提出了一个可解释的基于核的得分函数模型，极大地提高了生成模型的性能。

    

    我们关注基于分数的生成模型（SGMs）的基本数学结构。我们首先用Wasserstein近端算子（WPO）来构建SGMs，并证明通过平均场博弈（MFGs），WPO的结构揭示了描述扩散和基于分数模型的归纳偏差的数学结构。特别是，MFGs以一对耦合的偏微分方程的形式给出了最优性条件：一种前向控制的Fokker-Planck（FP）方程和一种向后的Hamilton-Jacobi-Bellman（HJB）方程。通过Cole-Hopf变换并利用交叉熵可以与密度的线性泛函相关联的事实，我们证明了HJB方程是一种无控制的FP方程。其次，利用手头的数学结构，我们提出了一个可解释的基于核的得分函数模型，该模型极大地提高了SGMs在训练样本和训练时间方面的性能。

    We focus on the fundamental mathematical structure of score-based generative models (SGMs). We first formulate SGMs in terms of the Wasserstein proximal operator (WPO) and demonstrate that, via mean-field games (MFGs), the WPO formulation reveals mathematical structure that describes the inductive bias of diffusion and score-based models. In particular, MFGs yield optimality conditions in the form of a pair of coupled partial differential equations: a forward-controlled Fokker-Planck (FP) equation, and a backward Hamilton-Jacobi-Bellman (HJB) equation. Via a Cole-Hopf transformation and taking advantage of the fact that the cross-entropy can be related to a linear functional of the density, we show that the HJB equation is an uncontrolled FP equation. Second, with the mathematical structure at hand, we present an interpretable kernel-based model for the score function which dramatically improves the performance of SGMs in terms of training samples and training time. In addition, the WP
    
[^77]: 通过混合狄利克雷分布改进证据深度学习

    Improved Evidential Deep Learning via a Mixture of Dirichlet Distributions

    [https://arxiv.org/abs/2402.06160](https://arxiv.org/abs/2402.06160)

    本文通过混合狄利克雷分布来改进证据深度学习（EDL）方法，解决了现有方法中认知不确定性在无限样本限制下可能不会消失的问题。

    

    本文探讨了一种现代的预测不确定性估计方法，称为证据深度学习（EDL），其中通过最小化特定的目标函数，训练单个神经网络模型以学习预测分布上的元分布。尽管现有方法在经验性能方面表现强大，但Bengs等人的最近研究发现了现有方法的一个根本缺陷：即使在无限样本限制下，学习到的认知不确定性可能不会消失。通过提供文献中一类广泛使用的目标函数的统一视角，我们得到了这个观察的证实。我们的分析揭示了EDL方法本质上通过最小化分布与与样本大小无关的目标分布之间的特定差异度量来训练元分布，从而产生错误的认知不确定性。基于理论原则，我们提出通过将其建模为狄利克雷分布混合物来学习一致目标分布，从而改进了EDL方法。

    This paper explores a modern predictive uncertainty estimation approach, called evidential deep learning (EDL), in which a single neural network model is trained to learn a meta distribution over the predictive distribution by minimizing a specific objective function. Despite their strong empirical performance, recent studies by Bengs et al. identify a fundamental pitfall of the existing methods: the learned epistemic uncertainty may not vanish even in the infinite-sample limit. We corroborate the observation by providing a unifying view of a class of widely used objectives from the literature. Our analysis reveals that the EDL methods essentially train a meta distribution by minimizing a certain divergence measure between the distribution and a sample-size-independent target distribution, resulting in spurious epistemic uncertainty. Grounded in theoretical principles, we propose learning a consistent target distribution by modeling it with a mixture of Dirichlet distributions and lear
    
[^78]: POTEC:通过两阶段策略分解的大动作空间离策略学习

    POTEC: Off-Policy Learning for Large Action Spaces via Two-Stage Policy Decomposition

    [https://arxiv.org/abs/2402.06151](https://arxiv.org/abs/2402.06151)

    POTEC提出了一种两阶段策略分解的算法，在大离散动作空间中有效进行离策略学习。该算法利用聚类选择第一阶段策略，并利用回归方法选择每个聚类内的具体动作。

    

    我们研究了在存在大离散动作空间的情境吞噬机制中的离线策略学习(OPL)，现有方法大多依赖于回归模型或重要性加权策略梯度，但由于过高的偏差或方差而失败。为了克服OPL中的这些问题，我们提出了一种新的两阶段算法，称为两阶段策略分解的策略优化(POTEC)。它利用动作空间中的聚类，并分别通过基于策略和回归的方法学习两种不同的策略。特别地，我们推导出一种新颖的低方差梯度估计器，通过基于策略的方法高效地学习第一阶段策略以选择聚类。为了在第一阶段策略采样的聚类中选择特定动作，POTEC在每个聚类中使用来自回归方法的第二阶段策略。我们展示了一种局部正确性条件，该条件仅要求回归模型保持相关性。

    We study off-policy learning (OPL) of contextual bandit policies in large discrete action spaces where existing methods -- most of which rely crucially on reward-regression models or importance-weighted policy gradients -- fail due to excessive bias or variance. To overcome these issues in OPL, we propose a novel two-stage algorithm, called Policy Optimization via Two-Stage Policy Decomposition (POTEC). It leverages clustering in the action space and learns two different policies via policy- and regression-based approaches, respectively. In particular, we derive a novel low-variance gradient estimator that enables to learn a first-stage policy for cluster selection efficiently via a policy-based approach. To select a specific action within the cluster sampled by the first-stage policy, POTEC uses a second-stage policy derived from a regression-based approach within each cluster. We show that a local correctness condition, which only requires that the regression model preserves the rela
    
[^79]: 用小数据进行领域泛化

    Domain Generalization with Small Data

    [https://arxiv.org/abs/2402.06150](https://arxiv.org/abs/2402.06150)

    本研究提出了一种在样本不足情况下解决领域泛化问题的方法。该方法通过使用概率嵌入来学习领域不变表示，并使用概率框架中的新方法测量混合分布之间的差异。结果表明，该方法在领域泛化性能上有显著的提升。

    

    本研究中，我们提出在“样本不足”情况下解决领域泛化的问题。我们不是基于确定性模型提取潜在特征嵌入，而是提出基于概率框架学习领域不变表示的方法，通过将每个数据点映射为概率嵌入。具体来说，我们首先将经验最大均值差异（MMD）扩展为一种新的概率MMD，可以度量由一系列潜在分布（即源领域）组成的混合分布之间的差异，而不是潜在点。此外，我们通过一种新的概率对比语义对齐（CSA）损失来提倡使正概率嵌入对更接近，而将其他负概率嵌入拉开。通过概率模型捕捉到的学到表示，我们提出的方法可以结合在分布上的度量，从而提升领域泛化性能。

    In this work, we propose to tackle the problem of domain generalization in the context of \textit{insufficient samples}. Instead of extracting latent feature embeddings based on deterministic models, we propose to learn a domain-invariant representation based on the probabilistic framework by mapping each data point into probabilistic embeddings. Specifically, we first extend empirical maximum mean discrepancy (MMD) to a novel probabilistic MMD that can measure the discrepancy between mixture distributions (i.e., source domains) consisting of a series of latent distributions rather than latent points. Moreover, instead of imposing the contrastive semantic alignment (CSA) loss based on pairs of latent points, a novel probabilistic CSA loss encourages positive probabilistic embedding pairs to be closer while pulling other negative ones apart. Benefiting from the learned representation captured by probabilistic models, our proposed method can marriage the measurement on the \textit{distri
    
[^80]: 关于带有高斯噪声的选择机制的隐私问题

    On the Privacy of Selection Mechanisms with Gaussian Noise

    [https://arxiv.org/abs/2402.06137](https://arxiv.org/abs/2402.06137)

    该论文研究了带有高斯噪声的选择机制的隐私问题，并证明了在底层查询是有界的情况下，可以提供纯粹的前期和后期差分隐私界限。

    

    报告噪声最大值和阈值以上是两个经典的差分隐私(DP)选择机制。它们的输出是通过对一系列低灵敏度的查询添加噪声，并报告满足某个条件的查询(噪声的)答案的身份来获得的。当在查询上添加拉普拉斯噪声时，这些机制的纯DP保证很容易获得。另一方面，当使用高斯噪声实例化时，标准分析只能提供近似的DP保证，尽管这些机制的输出位于离散空间中。在这项工作中，我们重新审视了使用高斯噪声的报告噪声最大值和阈值以上的分析，并展示了在额外的假设下，即底层查询是有界的情况下，可以为报告噪声最大值提供纯粹的前期DP界限，以及为阈值以上提供纯粹的后期DP界限。得到的界限是紧密的，并且依赖于可以使用标准元方法数值评估的闭式表达式。

    Report Noisy Max and Above Threshold are two classical differentially private (DP) selection mechanisms. Their output is obtained by adding noise to a sequence of low-sensitivity queries and reporting the identity of the query whose (noisy) answer satisfies a certain condition. Pure DP guarantees for these mechanisms are easy to obtain when Laplace noise is added to the queries. On the other hand, when instantiated using Gaussian noise, standard analyses only yield approximate DP guarantees despite the fact that the outputs of these mechanisms lie in a discrete space. In this work, we revisit the analysis of Report Noisy Max and Above Threshold with Gaussian noise and show that, under the additional assumption that the underlying queries are bounded, it is possible to provide pure ex-ante DP bounds for Report Noisy Max and pure ex-post DP bounds for Above Threshold. The resulting bounds are tight and depend on closed-form expressions that can be numerically evaluated using standard met
    
[^81]: 通过异构图对比学习共同学习地图实体的表示

    Jointly Learning Representations for Map Entities via Heterogeneous Graph Contrastive Learning

    [https://arxiv.org/abs/2402.06135](https://arxiv.org/abs/2402.06135)

    通过异构图对比学习方法能够共同学习多个类别的地图实体的表示，提取潜在的结构和语义信息。

    

    电子地图在地理信息系统中起着至关重要的作用，为各种城市管理场景和日常生活服务提供支持。开发有效的地图实体表示学习（MERL）方法对于从电子地图中提取嵌入信息并将地图实体转化为表示向量以供下游应用非常重要。然而，现有的MERL方法通常只关注一种特定类别的地图实体，如兴趣点（POIs）、道路段或土地分块，这对于真实世界中多样化的基于地图的应用来说是不充分的，可能会丢失不同类型实体之间的潜在结构和语义信息。此外，使用不同地图实体的分别生成的表示可能会引入不一致性。受此启发，我们提出了一种名为HOME-GCL的新方法，用于学习多种类别地图实体的表示。我们的方法利用了一个异构地图实体图（HOME图），其中包括了地图实体之间的交互的潜在结构和语义信息。

    The electronic map plays a crucial role in geographic information systems, serving various urban managerial scenarios and daily life services. Developing effective Map Entity Representation Learning (MERL) methods is crucial to extracting embedding information from electronic maps and converting map entities into representation vectors for downstream applications. However, existing MERL methods typically focus on one specific category of map entities, such as POIs, road segments, or land parcels, which is insufficient for real-world diverse map-based applications and might lose latent structural and semantic information interacting between entities of different types. Moreover, using representations generated by separate models for different map entities can introduce inconsistencies. Motivated by this, we propose a novel method named HOME-GCL for learning representations of multiple categories of map entities. Our approach utilizes a heterogeneous map entity graph (HOME graph) that in
    
[^82]: 重新思考大规模图学习的节点传播方式

    Rethinking Node-wise Propagation for Large-scale Graph Learning

    [https://arxiv.org/abs/2402.06128](https://arxiv.org/abs/2402.06128)

    提出了适应性拓扑感知传播（ATP）方法，以应对大规模图学习中节点传播的问题。此方法能对不同节点的拓扑角色进行个性化传播，并减少传播带来的偏差和额外开销。

    

    可扩展的图神经网络（GNN）已成为一种有前景的技术，在许多大规模基于图的Web应用中展现出卓越的预测性能和高效运行效率。然而，大多数可扩展的GNN倾向于以相同的传播规则处理图中的所有节点，忽视了它们的拓扑独特性；现有的节点级传播优化策略在复杂的Web规模图中效果不佳，需要对节点的局部属性进行全面描绘。直观地说，Web规模图中的不同节点具有不同的拓扑角色，因此无差别地传播或忽视局部上下文可能会影响节点表示的质量。小规模情景无法匹配Web规模图中的这种复杂拓扑。为了解决上述问题，我们提出了适应性拓扑感知传播（ATP）方法，减少潜在的高偏差传播和额外开销。

    Scalable graph neural networks (GNNs) have emerged as a promising technique, which exhibits superior predictive performance and high running efficiency across numerous large-scale graph-based web applications. However, (i) Most scalable GNNs tend to treat all nodes in graphs with the same propagation rules, neglecting their topological uniqueness; (ii) Existing node-wise propagation optimization strategies are insufficient on web-scale graphs with intricate topology, where a full portrayal of nodes' local properties is required. Intuitively, different nodes in web-scale graphs possess distinct topological roles, and therefore propagating them indiscriminately or neglect local contexts may compromise the quality of node representations. This intricate topology in web-scale graphs cannot be matched by small-scale scenarios. To address the above issues, we propose \textbf{A}daptive \textbf{T}opology-aware \textbf{P}ropagation (ATP), which reduces potential high-bias propagation and extrac
    
[^83]: CityFlowER:一种具有嵌入式机器学习模型的高效真实交通模拟器

    CityFlowER: An Efficient and Realistic Traffic Simulator with Embedded Machine Learning Models

    [https://arxiv.org/abs/2402.06127](https://arxiv.org/abs/2402.06127)

    CityFlowER是一种高效真实的交通模拟器，通过嵌入机器学习模型提高了模拟的真实性和效率。

    

    交通模拟是交通基础设施规划、智能交通控制政策学习和交通流分析的重要工具。其有效性严重依赖所使用模拟器的真实性。传统的交通模拟器（例如SUMO和CityFlow）往往受限于基于规则的模型和过于简化驾驶行为的超参数，导致模拟结果不真实。为了增强真实性，一些模拟器提供了与机器学习模型进行交互的应用程序接口（API），这些模型可以从观测数据中学习并提供更复杂的驾驶行为模型。然而，当车辆数量增加时，这种方法面临可扩展性和时间效率的挑战。为解决这些限制，我们引入了CityFlowER，一种在现有CityFlow模拟器基础上的改进，旨在实现高效真实的城市交通模拟。CityFlowER创新性地将机器学习模型预嵌入到模拟器中，从而提高了模拟的效率和真实性。

    Traffic simulation is an essential tool for transportation infrastructure planning, intelligent traffic control policy learning, and traffic flow analysis. Its effectiveness relies heavily on the realism of the simulators used. Traditional traffic simulators, such as SUMO and CityFlow, are often limited by their reliance on rule-based models with hyperparameters that oversimplify driving behaviors, resulting in unrealistic simulations. To enhance realism, some simulators have provided Application Programming Interfaces (APIs) to interact with Machine Learning (ML) models, which learn from observed data and offer more sophisticated driving behavior models. However, this approach faces challenges in scalability and time efficiency as vehicle numbers increase. Addressing these limitations, we introduce CityFlowER, an advancement over the existing CityFlow simulator, designed for efficient and realistic city-wide traffic simulation. CityFlowER innovatively pre-embeds ML models within the s
    
[^84]: 学习变得高效：在大型语言模型中构建结构化稀疏性

    Learn To be Efficient: Build Structured Sparsity in Large Language Models

    [https://arxiv.org/abs/2402.06126](https://arxiv.org/abs/2402.06126)

    本文通过引入一种新的算法"Learn-To-be-Efficient(LTE)"，提出了在大型语言模型(LLM)中构建结构化稀疏性的方法。该方法通过训练高效意识的LLM学习激活更少的神经元，取得更好的稀疏性和性能折衷。

    

    大型语言模型(LLM)以其十亿级参数取得了显著的成功，但它们产生了高昂的推理开销。在LLM中出现的激活稀疏性为通过仅涉及部分参数进行推理提供了一种自然的方法来减少这种成本。现有方法只关注利用这种自然形成的激活稀疏性，忽视了进一步放大这种固有稀疏性的潜力。本文中，我们假设LLM可以通过实现更结构化的激活稀疏性来学习高效。为实现这一目标，我们引入了一种新颖的算法"Learn-To-be-Efficient(LTE)", 旨在训练高效意识的LLM学习激活更少的神经元，并在稀疏性和性能之间取得更好的折衷。此外，与主要关注基于ReLU模型的SOTA MoEfication方法不同，LTE还可以应用于像GPT和LLaMA这样具有软激活函数的LLM。我们在四个模型和十一个数据集上评估了LTE。

    Large Language Models (LLMs) have achieved remarkable success with their billion-level parameters, yet they incur high inference overheads. The emergence of activation sparsity in LLMs provides a natural approach to reduce this cost by involving only parts of the parameters for inference. Existing methods only focus on utilizing this naturally formed activation sparsity, overlooking the potential for further amplifying this inherent sparsity. In this paper, we hypothesize that LLMs can learn to be efficient by achieving more structured activation sparsity.To achieve this, we introduce a novel algorithm, Learn-To-be-Efficient (LTE), designed to train efficiency-aware LLMs to learn to activate fewer neurons and achieve a better trade-off between sparsity and performance. Furthermore, unlike SOTA MoEfication methods, which mainly focus on ReLU-based models, LTE can also be applied to LLMs like GPT and LLaMA with soft activation functions. We evaluate LTE on four models and eleven datasets
    
[^85]: 使用PEAK进行窥探：多个数据流均值的顺序、非参数复合假设检验

    Peeking with PEAK: Sequential, Nonparametric Composite Hypothesis Tests for Means of Multiple Data Streams

    [https://arxiv.org/abs/2402.06122](https://arxiv.org/abs/2402.06122)

    本论文提出了一种名为PEAK的新型非参数顺序复合假设检验方法，适用于多个数据流的均值检验。该方法基于测试即博弈的框架，在任何停止时间上提供了非渐进α水平的检验。PEAK能够有效拒绝在满足非参数假设条件的所有潜在分布中错误的假设，从而实现对多个数据流的联合复合假设检验。与现有方法相比，该方法具有较高的计算效率。

    

    我们提出了一种新颖的非参数顺序复合假设检验方法，用于多个数据流的均值。我们的方法名为PEAK（基于期望平均资产的窥探），基于测试即博弈的框架，提供了一个在任何停止时间上的非渐进α水平测试。PEAK在计算上可行，并且能够有效拒绝在满足我们的非参数假设条件的所有潜在分布中错误的假设，从而实现对多个数据流的联合复合假设检验。我们在强化学习中的最佳臂识别和阈值识别任务中对我们的理论结果进行了数值验证，并展示了我们的方法在计算效率上优于现有的测试方法。

    We propose a novel nonparametric sequential test for composite hypotheses for means of multiple data streams. Our proposed method, \emph{peeking with expectation-based averaged capital} (PEAK), builds upon the testing-as-betting framework and provides a non-asymptotic $\alpha$-level test across any stopping time. PEAK is computationally tractable and efficiently rejects hypotheses that are incorrect across all potential distributions that satisfy our nonparametric assumption, enabling joint composite hypothesis testing on multiple streams of data. We numerically validate our theoretical findings under the best arm identification and threshold identification in the bandit setting, illustrating the computational efficiency of our method against state-of-the-art testing methods.
    
[^86]: 通过迭代去噪能量匹配从玻尔兹曼密度中进行采样

    Iterated Denoising Energy Matching for Sampling from Boltzmann Densities

    [https://arxiv.org/abs/2402.06121](https://arxiv.org/abs/2402.06121)

    提出了一种基于迭代算法的新颖采样方法，通过利用能量函数和梯度进行训练，无需数据样本。该方法能够高效生成统计独立的样本，并且在高维度上具有可扩展性。通过利用扩散的快速模式混合行为，实现了对能量景观的平滑，从而实现了高效的探索和学习。

    

    高效地从未标准化的概率分布中生成统计独立的样本，比如多体系统的平衡样本，是科学中的一个基础问题。在本文中，我们提出了迭代去噪能量匹配（iDEM），这是一种迭代算法，它利用了一种新颖的随机得分匹配目标，仅使用能量函数及其梯度 - 而不是数据样本 - 来训练扩散基础的采样器。具体而言，iDEM在以下两个步骤之间交替进行：（I）从扩散基础的采样器中采样高模型密度的区域，和（II）使用这些样本在我们的随机匹配目标中进一步改进采样器。iDEM在高维度上是可扩展的，内部匹配目标是无需模拟的，并且不需要MCMC样本。此外，通过利用扩散的快速模式混合行为，iDEM平滑了能量背景，实现了高效的探索和学习的分摊采样器。我们对一系列任务进行了iDEM的评估...

    Efficiently generating statistically independent samples from an unnormalized probability distribution, such as equilibrium samples of many-body systems, is a foundational problem in science. In this paper, we propose Iterated Denoising Energy Matching (iDEM), an iterative algorithm that uses a novel stochastic score matching objective leveraging solely the energy function and its gradient -- and no data samples -- to train a diffusion-based sampler. Specifically, iDEM alternates between (I) sampling regions of high model density from a diffusion-based sampler and (II) using these samples in our stochastic matching objective to further improve the sampler. iDEM is scalable to high dimensions as the inner matching objective, is simulation-free, and requires no MCMC samples. Moreover, by leveraging the fast mode mixing behavior of diffusion, iDEM smooths out the energy landscape enabling efficient exploration and learning of an amortized sampler. We evaluate iDEM on a suite of tasks rang
    
[^87]: AI增强的数据同化和不确定性量化在地质碳封存中的应用

    AI enhanced data assimilation and uncertainty quantification applied to Geological Carbon Storage

    [https://arxiv.org/abs/2402.06110](https://arxiv.org/abs/2402.06110)

    本研究通过整合机器学习和数据同化技术，提出了用于地质碳封存的代理模型方法，可在维持高准确性的同时加快同化过程，具有较大的应用潜力。

    

    本研究探讨了机器学习（ML）和数据同化（DA）技术的整合，重点是在保持高准确性物理结果的同时，实现地质碳封存（GCS）项目的代理模型。首先，我们评估了两种不同机器学习模型，傅里叶神经算子（FNOs）和Transformer UNet（T-UNet）在沉积通道储层中CO$_2$注入模拟中的代理建模能力。我们引入了基于代理模型的混合ESMDA（SH-ESMDA），这是传统集合平滑器与多数据同化（ESMDA）的一种改进。这种方法使用FNOs和T-UNet作为代理模型，并且有潜力使标准的ESMDA过程至少快50％或更多，具体取决于同化步骤的数量。此外，我们还介绍了基于代理模型的混合RML（SH-RML），这是一种依赖于随机最大似然（RML）的变分数据同化方法。

    This study investigates the integration of machine learning (ML) and data assimilation (DA) techniques, focusing on implementing surrogate models for Geological Carbon Storage (GCS) projects while maintaining high fidelity physical results in posterior states. Initially, we evaluate the surrogate modeling capability of two distinct machine learning models, Fourier Neural Operators (FNOs) and Transformer UNet (T-UNet), in the context of CO$_2$ injection simulations within channelized reservoirs. We introduce the Surrogate-based hybrid ESMDA (SH-ESMDA), an adaptation of the traditional Ensemble Smoother with Multiple Data Assimilation (ESMDA). This method uses FNOs and T-UNet as surrogate models and has the potential to make the standard ESMDA process at least 50% faster or more, depending on the number of assimilation steps. Additionally, we introduce Surrogate-based Hybrid RML (SH-RML), a variational data assimilation approach that relies on the randomized maximum likelihood (RML) wher
    
[^88]: 在在线考试中的作弊检测和定位的多实例学习

    Multiple Instance Learning for Cheating Detection and Localization in Online Examinations

    [https://arxiv.org/abs/2402.06107](https://arxiv.org/abs/2402.06107)

    本文提出了一种基于多实例学习的作弊检测框架CHEESE，该框架综合考虑了头部姿势、凝视角度、身体姿势和背景信息等特征，并通过标签生成器和特征编码器实现了作弊行为的检测和定位。

    

    2019年冠状病毒病流行疫情的蔓延导致许多课程和考试变成在线形式。考试监考系统中的作弊行为检测模型在保证远程考试的公平性方面起着重要作用。然而，作弊行为很少见，大多数研究者在作弊行为检测任务中没有全面考虑头部姿势、凝视角度、身体姿势和背景信息等特征。在本文中，我们开发并提出了CHEESE，一种基于多实例学习的作弊检测框架。该框架包括实现弱监督的标签生成器和学习判别性特征的特征编码器。此外，该框架还将3D卷积提取的身体姿势和背景特征与OpenFace 2.0捕获的眼睛凝视、头部姿势和面部特征相结合。通过拼接，这些特征被送入时空图模块进行分析。

    The spread of the Coronavirus disease-2019 epidemic has caused many courses and exams to be conducted online. The cheating behavior detection model in examination invigilation systems plays a pivotal role in guaranteeing the equality of long-distance examinations. However, cheating behavior is rare, and most researchers do not comprehensively take into account features such as head posture, gaze angle, body posture, and background information in the task of cheating behavior detection. In this paper, we develop and present CHEESE, a CHEating detection framework via multiplE inStancE learning. The framework consists of a label generator that implements weak supervision and a feature encoder to learn discriminative features. In addition, the framework combines body posture and background features extracted by 3D convolution with eye gaze, head posture and facial features captured by OpenFace 2.0. These features are fed into the spatio-temporal graph module by stitching to analyze the spa
    
[^89]: 功能对齐回归：一种从数据中明确学习函数导数的方法

    Function Aligned Regression: A Method Explicitly Learns Functional Derivatives from Data

    [https://arxiv.org/abs/2402.06104](https://arxiv.org/abs/2402.06104)

    该论文提出了一种名为FAR的方法，通过捕捉函数导数来更好、更高效地拟合底层真实函数。在合成数据集和八个真实世界任务中证明了该方法的有效性。

    

    回归是机器学习中的一个基本任务，在过去几十年中引起了广泛关注。传统的回归方法主要通过使用损失函数来将模型预测与每个个体数据样本的真实值对齐，然而，我们发现这种方法可能导致在不同样本之间关系的预测不够优化。近期的研究工作引入了标签相似性信息来改进回归方法，但在完全捕捉底层真实函数的复杂性方面仍存在明显的差距。在本文中，我们提出了FAR（功能对齐回归）作为一种更好、更高效的解决方案，通过捕捉函数导数来拟合底层真实函数。我们在两个合成数据集和六个领域的八个大规模真实世界任务中验证了该方法的有效性。

    Regression is a fundamental task in machine learning that has garnered extensive attention over the past decades. The conventional approach for regression involves employing loss functions that primarily concentrate on aligning model prediction with the ground truth for each individual data sample, which, as we show, can result in sub-optimal prediction of the relationships between the different samples. Recent research endeavors have introduced novel perspectives by incorporating label similarity information to regression. However, a notable gap persists in these approaches when it comes to fully capturing the intricacies of the underlying ground truth function. In this work, we propose FAR (Function Aligned Regression) as a arguably better and more efficient solution to fit the underlying function of ground truth by capturing functional derivatives. We demonstrate the effectiveness of the proposed method practically on 2 synthetic datasets and on 8 extensive real-world tasks from 6 b
    
[^90]: 通过深度强化学习实现真实世界流体引导刚体控制

    Real-World Fluid Directed Rigid Body Control via Deep Reinforcement Learning

    [https://arxiv.org/abs/2402.06102](https://arxiv.org/abs/2402.06102)

    通过创建流体盒子实验控制系统，我们展示了在真实世界场景中使用深度强化学习算法合成复杂行为的能力，并探索了离线强化学习在数据高效假设测试中的潜力。

    

    最近在强化学习（RL）的真实世界应用中取得的进展依赖于能够准确模拟大规模系统的能力。然而，像流体动力学系统这样的领域展示了复杂的动态现象，很难以高积分速率进行模拟，从而限制了现代深度RL算法在昂贵或安全关键硬件上的直接应用。在这项工作中，我们引入了“流体盒子（Box o Flows）”，这是一个新颖的台式实验控制系统，用于在动态真实世界场景中系统地评估RL算法。我们描述了流体盒子的关键组件，并通过一系列实验展示了最先进的无模型RL算法如何通过简单的奖励规范合成各种复杂行为。此外，我们还探索了离线RL在数据高效假设测试中的作用，通过重用过去的经验。我们相信，从这项初步研究中获得的见解以及流体盒子等系统的可用性将推动RL在真实世界中的应用。

    Recent advances in real-world applications of reinforcement learning (RL) have relied on the ability to accurately simulate systems at scale. However, domains such as fluid dynamical systems exhibit complex dynamic phenomena that are hard to simulate at high integration rates, limiting the direct application of modern deep RL algorithms to often expensive or safety critical hardware. In this work, we introduce "Box o Flows", a novel benchtop experimental control system for systematically evaluating RL algorithms in dynamic real-world scenarios. We describe the key components of the Box o Flows, and through a series of experiments demonstrate how state-of-the-art model-free RL algorithms can synthesize a variety of complex behaviors via simple reward specifications. Furthermore, we explore the role of offline RL in data-efficient hypothesis testing by reusing past experiences. We believe that the insights gained from this preliminary study and the availability of systems like the Box o 
    
[^91]: 来，见，胜：解决知识图谱学习前的众多挑战

    Veni, Vidi, Vici: Solving the Myriad of Challenges before Knowledge Graph Learning

    [https://arxiv.org/abs/2402.06098](https://arxiv.org/abs/2402.06098)

    知识图谱学习面临着缺乏专家知识整合、节点度数极端性不稳定、缺乏不确定性和相关性的考虑以及缺乏可解释性的挑战。现有的解决尝试大多是孤立的，需要综合考虑这些问题的解决方案。

    

    知识图谱(KG)已经成为表示大规模链接数据的常见方法。然而，由于KG的巨大规模，图谱学习系统需要帮助人类进行分析、解释和模式检测。尽管有许多KG学习系统对研究人员和临床医生进行了赋能，但我们发现现有的图谱学习中存在四个关键不足，这些不足同时限制了KG学习的性能并降低了人类与这些学习系统的最佳接口能力。这些不足包括：1)缺乏专家知识的整合，2)对KG中节点度数极端性的不稳定性，3)在学习过程中缺乏对不确定性和相关性的考虑，4)缺乏可解释性。此外，我们对解决每个问题的现有尝试进行了分类，并指出每个尝试在很大程度上与解决其他问题的尝试相隔离。通过对这些问题的形式化，

    Knowledge Graphs (KGs) have become increasingly common for representing large-scale linked data. However, their immense size has required graph learning systems to assist humans in analysis, interpretation, and pattern detection. While there have been promising results for researcher- and clinician- empowerment through a variety of KG learning systems, we identify four key deficiencies in state-of-the-art graph learning that simultaneously limit KG learning performance and diminish the ability of humans to interface optimally with these learning systems. These deficiencies are: 1) lack of expert knowledge integration, 2) instability to node degree extremity in the KG, 3) lack of consideration for uncertainty and relevance while learning, and 4) lack of explainability. Furthermore, we characterise state-of-the-art attempts to solve each of these problems and note that each attempt has largely been isolated from attempts to solve the other problems. Through a formalisation of these probl
    
[^92]: TWIG：通过模拟KGE模型实现预先超参数优化和跨图泛化

    TWIG: Towards pre-hoc Hyperparameter Optimisation and Cross-Graph Generalisation via Simulated KGE Models

    [https://arxiv.org/abs/2402.06097](https://arxiv.org/abs/2402.06097)

    这项研究引入了一种名为TWIG的新颖模型，可以通过拓扑特征学习权重来模拟KGE模型的输出，有效减少了参数数量，并具有预先优化超参数和跨图泛化的能力。

    

    在本文中，我们介绍了一种名为TWIG（Topologically-Weighted Intelligence Generation）的新颖的、无需嵌入的模拟KGE输出的范式，它只使用了一小部分参数。TWIG从图数据的拓扑特征作为输入学习权重，没有对实体或边的潜在表示进行编码。我们在UMLS数据集上的实验结果表明，单个TWIG神经网络几乎可以准确预测所有超参数配置下最先进的ComplEx-N3 KGE模型的结果。为了达到这个目标，它只使用了2590个可学习参数，但准确预测了1215个不同超参数组合的结果，相当于29322000个参数的总成本。

    In this paper we introduce TWIG (Topologically-Weighted Intelligence Generation), a novel, embedding-free paradigm for simulating the output of KGEs that uses a tiny fraction of the parameters. TWIG learns weights from inputs that consist of topological features of the graph data, with no coding for latent representations of entities or edges. Our experiments on the UMLS dataset show that a single TWIG neural network can predict the results of state-of-the-art ComplEx-N3 KGE model nearly exactly on across all hyperparameter configurations. To do this it uses a total of 2590 learnable parameters, but accurately predicts the results of 1215 different hyperparameter combinations with a combined cost of 29,322,000 parameters. Based on these results, we make two claims: 1) that KGEs do not learn latent semantics, but only latent representations of structural patterns; 2) that hyperparameter choice in KGEs is a deterministic function of the KGE model and graph structure. We further hypothesi
    
[^93]: 具有改进的随机游走核的描述性核卷积网络

    Descriptive Kernel Convolution Network with Improved Random Walk Kernel

    [https://arxiv.org/abs/2402.06087](https://arxiv.org/abs/2402.06087)

    本文提出了一种描述性核卷积网络，通过改进随机游走核并引入颜色匹配随机游走，提升了图核在特征工程中的可学习性。进一步发现了随机游走核与GCN层的联系，并提出了一种新颖的GNN方法。

    

    图核曾经是处理结构化数据的主要特征工程方法，但由于缺乏可学习性，已被现代GNN取代。最近，一系列核卷积网络(KCNs)通过引入可学习的隐藏图来卷积输入数据，成功地使图核得以复苏。随机游走核(RWK)作为许多KCNs中的默认核，受到越来越多的关注。本文首先重新审视了RWK及其在KCNs中的现有使用情况，并揭示了现有设计的几个不足之处，提出了一种改进的图核RWK+，通过引入颜色匹配随机游走并推导其高效计算方法。然后，我们提出了RWK+CN，一个使用RWK+作为核心核函数的KCN，通过无监督目标来学习描述性的图特征，这是GNNs无法实现的。此外，通过展开RWK+，我们发现它与常规GCN层存在联系，并提出了一种新颖的GNN方法。

    Graph kernels used to be the dominant approach to feature engineering for structured data, which are superseded by modern GNNs as the former lacks learnability. Recently, a suite of Kernel Convolution Networks (KCNs) successfully revitalized graph kernels by introducing learnability, which convolves input with learnable hidden graphs using a certain graph kernel. The random walk kernel (RWK) has been used as the default kernel in many KCNs, gaining increasing attention. In this paper, we first revisit the RWK and its current usage in KCNs, revealing several shortcomings of the existing designs, and propose an improved graph kernel RWK+, by introducing color-matching random walks and deriving its efficient computation. We then propose RWK+CN, a KCN that uses RWK+ as the core kernel to learn descriptive graph features with an unsupervised objective, which can not be achieved by GNNs. Further, by unrolling RWK+, we discover its connection with a regular GCN layer, and propose a novel GNN 
    
[^94]: SubGen：子线性时间和内存的令牌生成

    SubGen: Token Generation in Sublinear Time and Memory

    [https://arxiv.org/abs/2402.06082](https://arxiv.org/abs/2402.06082)

    这项工作提出了一种名为SubGen的高效缓存压缩技术，通过在Attention模块中进行在线聚类和采样，实现了子线性的内存占用和时间复杂度，并建立了一个紧密的误差界。

    

    尽管大型语言模型（LLM）取得了显著的成功，但它们广泛的内存需求使得在长上下文令牌生成环境中部署它们存在挑战。LLM解码器的巨大内存占用量来自于在注意模块中存储所有先前令牌的必要性，这是由键-值（KV）缓存所强制的要求。在这项工作中，我们的重点是开发一种高效的键值缓存压缩技术。经验证据表明，在注意模块中的键嵌入中存在显著的聚类倾向。基于这一关键洞察，我们设计了一种具有子线性复杂度的新型缓存方法，采用键令牌的在线聚类和值的在线l2采样。结果是一个可以证明准确和高效的注意解码算法，称为SubGen。这个算法不仅确保了子线性内存占用和子线性时间复杂度，还为我们的方法建立了一个紧密的误差界。经验评估...

    Despite the significant success of large language models (LLMs), their extensive memory requirements pose challenges for deploying them in long-context token generation. The substantial memory footprint of LLM decoders arises from the necessity to store all previous tokens in the attention module, a requirement imposed by key-value (KV) caching. In this work, our focus is on developing an efficient compression technique for the KV cache. Empirical evidence indicates a significant clustering tendency within key embeddings in the attention module. Building on this key insight, we have devised a novel caching method with sublinear complexity, employing online clustering on key tokens and online $\ell_2$ sampling on values. The result is a provably accurate and efficient attention decoding algorithm, termed SubGen. Not only does this algorithm ensure a sublinear memory footprint and sublinear time complexity, but we also establish a tight error bound for our approach. Empirical evaluations
    
[^95]: DiscDiff: DNA序列生成的潜在扩散模型

    DiscDiff: Latent Diffusion Model for DNA Sequence Generation

    [https://arxiv.org/abs/2402.06079](https://arxiv.org/abs/2402.06079)

    本文介绍了一种新的框架用于生成DNA序列，包括一个用于生成离散DNA序列的潜在扩散模型和一个用于改进序列的后训练算法。我们的方法不仅在DNA序列生成方面树立了新的标准，并且在生成短序列和长序列方面表现出优越性能。此外，我们还引入了一个多物种的DNA生成数据集。这项研究将推动DNA的生成建模，并对基因治疗和蛋白质生产产生潜在影响。

    

    本文介绍了一种新颖的DNA序列生成框架，包括两个关键组成部分：DiscDiff，一种为生成离散DNA序列而定制的潜在扩散模型（LDM），以及Absorb-Escape，一种用于改进这些序列的后训练算法。Absorb-Escape通过纠正在潜在空间和输入空间之间的转换过程中固有的“舍入误差”，提高了生成序列的真实性。我们的方法不仅在DNA序列生成方面树立了新的标准，而且在生成短序列和长序列方面表现出优越的性能，同时引入了EPD-GenDNA，这是第一个涵盖15个物种的、综合性的DNA生成数据集，包含160,000个独特序列。我们希望这项研究能推动DNA的生成建模，对基因治疗和蛋白质生产可能产生影响。

    This paper introduces a novel framework for DNA sequence generation, comprising two key components: DiscDiff, a Latent Diffusion Model (LDM) tailored for generating discrete DNA sequences, and Absorb-Escape, a post-training algorithm designed to refine these sequences. Absorb-Escape enhances the realism of the generated sequences by correcting `round errors' inherent in the conversion process between latent and input spaces. Our approach not only sets new standards in DNA sequence generation but also demonstrates superior performance over existing diffusion models, in generating both short and long DNA sequences. Additionally, we introduce EPD-GenDNA, the first comprehensive, multi-species dataset for DNA generation, encompassing 160,000 unique sequences from 15 species. We hope this study will advance the generative modelling of DNA, with potential implications for gene therapy and protein production.
    
[^96]: 实现数字化决策支持下的规模化人工智能战争游戏

    Scaling Artificial Intelligence for Digital Wargaming in Support of Decision-Making

    [https://arxiv.org/abs/2402.06075](https://arxiv.org/abs/2402.06075)

    针对决策支持下的战争游戏，本论文提出了规模化人工智能的发展并与人类判断相结合的重要性。通过提高全域意识、改善决策速度和质量、提供创新行动建议以及更快速地应对对手行动，我们能够更好地应对现代挑战和困境，增强人类决策的指导和增强。

    

    在这个前所未有的由技术驱动的变革时代，我们更需要积极投资于开发强大的人工智能（AI）来支持决策的战争游戏。通过推进AI技术系统并将其与人类判断相结合，我们将能够提高全域意识，改善决策周期的速度和质量，提供创新行动的建议，更迅速地应对对手的行动。因此，我们迫切需要加快AI的开发，以帮助我们更好地应对现代挑战和困境的复杂性，这些挑战目前需要人类智能，并在可能的情况下试图超越人类智能-而不是取代人类，而是以机器速度增强和更好地指导人类决策。

    In this unprecedented era of technology-driven transformation, it becomes more critical than ever that we aggressively invest in developing robust artificial intelligence (AI) for wargaming in support of decision-making. By advancing AI-enabled systems and pairing these with human judgment, we will be able to enhance all-domain awareness, improve the speed and quality of our decision cycles, offer recommendations for novel courses of action, and more rapidly counter our adversary's actions. It therefore becomes imperative that we accelerate the development of AI to help us better address the complexity of modern challenges and dilemmas that currently requires human intelligence and, if possible, attempt to surpass human intelligence--not to replace humans, but to augment and better inform human decision-making at machine speed. Although deep reinforcement learning continues to show promising results in intelligent agent behavior development for the long-horizon, complex tasks typically
    
[^97]: 三维二维神经网络用于噪声干涉成像中的相位恢复

    3D-2D Neural Nets for Phase Retrieval in Noisy Interferometric Imaging

    [https://arxiv.org/abs/2402.06063](https://arxiv.org/abs/2402.06063)

    我们引入了一种称为PRUNe的三维二维相位恢复神经网络，用于解决存在噪声的干涉成像中的相位恢复问题。该网络能够处理相位噪声，并在恢复精度和准确性方面表现出优势。

    

    近年来，神经网络已被用于解决成像中的相位恢复问题，其准确性和速度优于传统技术，尤其是在存在噪声的情况下。然而，在干涉成像的背景下，现有的神经网络架构对相位噪声的处理很少。这种噪声在干涉仪中自然产生，由于机械不稳定性或大气湍流，限制了测量采集时间，并在光强有限的情况下（例如遥感）造成挑战。在这里，我们引入了一种三维二维相位恢复U-Net（PRUNe），该网络以噪声和随机相移的干涉图作为输入，并输出一个二维相位图像。一个三维下采样的卷积编码器捕捉帧内和帧间的相关性，产生一个二维潜在空间，该空间通过一个二维解码器上采样成一个相位图像。我们将我们的模型与最先进的奇异值分解算法进行对比，并发现PRUNe具有更高的恢复精度和准确性。

    In recent years, neural networks have been used to solve phase retrieval problems in imaging with superior accuracy and speed than traditional techniques, especially in the presence of noise. However, in the context of interferometric imaging, phase noise has been largely unaddressed by existing neural network architectures. Such noise arises naturally in an interferometer due to mechanical instabilities or atmospheric turbulence, limiting measurement acquisition times and posing a challenge in scenarios with limited light intensity, such as remote sensing. Here, we introduce a 3D-2D Phase Retrieval U-Net (PRUNe) that takes noisy and randomly phase-shifted interferograms as inputs, and outputs a single 2D phase image. A 3D downsampling convolutional encoder captures correlations within and between frames to produce a 2D latent space, which is upsampled by a 2D decoder into a phase image. We test our model against a state-of-the-art singular value decomposition algorithm and find PRUNe 
    
[^98]: 利用大数据进行公共卫生决策的影响

    Impact on Public Health Decision Making by Utilizing Big Data Without Domain Knowledge

    [https://arxiv.org/abs/2402.06059](https://arxiv.org/abs/2402.06059)

    本研究使用大量街景图像和纽约市的健康数据，发现基于数据的决策在没有考虑数据健壮性和基于虚假相关性时存在偏见。

    

    新的数据来源和从中提取信息的人工智能方法正在变得丰富多样，并且与许多社会应用的决策相关。一个重要的例子是街景图像，在100多个国家可用，并被考虑用于评估建筑环境与社区健康结果的相关性。在这种使用情境下，当基于数据的决策没有考虑到数据的健壮性，或者预测基于虚假的相关性时，使用人工智能存在重要的偏见。为了研究这个风险，我们利用了来自纽约市的200.2万个街景图像以及健康、人口统计和社会经济数据。首先，我们证明了在城市内部由街景图像标签推断出的建筑环境特征可能与实际情况不符合。我们还发现个体级别的体力活动不足行为显著影响了决策的影响。

    New data sources, and artificial intelligence (AI) methods to extract information from them are becoming plentiful, and relevant to decision making in many societal applications. An important example is street view imagery, available in over 100 countries, and considered for applications such as assessing built environment aspects in relation to community health outcomes. Relevant to such uses, important examples of bias in the use of AI are evident when decision-making based on data fails to account for the robustness of the data, or predictions are based on spurious correlations. To study this risk, we utilize 2.02 million GSV images along with health, demographic, and socioeconomic data from New York City. Initially, we demonstrate that built environment characteristics inferred from GSV labels at the intra-city level may exhibit inadequate alignment with the ground truth. We also find that the average individual-level behavior of physical inactivity significantly mediates the impac
    
[^99]: ActiveDP：将主动学习和数据编程框架相结合

    ActiveDP: Bridging Active Learning and Data Programming

    [https://arxiv.org/abs/2402.06056](https://arxiv.org/abs/2402.06056)

    本文提出了ActiveDP，一个将主动学习和数据编程相结合的交互式框架，以生成具有高准确性和覆盖率的标签，实验证明其优于以前的弱监督和主动学习方法，并在不同的标记预算下表现良好。

    

    现代机器学习模型需要大规模标记数据集以达到良好的性能，但手动标记大规模数据集既昂贵又耗时。数据编程范式可以高效地标记大规模数据集，但会产生噪声标签，从而降低下游模型的性能。而主动学习范式可以获取准确的标签，但只针对一小部分实例。本文提出了ActiveDP，一个将主动学习和数据编程框架相结合的交互式框架，以生成高精度和覆盖度具备的标签，结合了两种范式的优势。实验证明，ActiveDP优于以前的弱监督和主动学习方法，并在不同的标记预算下始终表现良好。

    Modern machine learning models require large labelled datasets to achieve good performance, but manually labelling large datasets is expensive and time-consuming. The data programming paradigm enables users to label large datasets efficiently but produces noisy labels, which deteriorates the downstream model's performance. The active learning paradigm, on the other hand, can acquire accurate labels but only for a small fraction of instances. In this paper, we propose ActiveDP, an interactive framework bridging active learning and data programming together to generate labels with both high accuracy and coverage, combining the strengths of both paradigms. Experiments show that ActiveDP outperforms previous weak supervision and active learning approaches and consistently performs well under different labelling budgets.
    
[^100]: 智能模式切换框架用于远程操作

    Intelligent Mode-switching Framework for Teleoperation

    [https://arxiv.org/abs/2402.06047](https://arxiv.org/abs/2402.06047)

    本研究提出了一种智能模式切换框架，通过考虑模式切换和通信系统，解决了远程操作中的难题。通过预测用户意图并自主执行任务的一部分，减少了对操作者的需求，提高了任务完成率。

    

    由于感知能力有限、通信延迟高和操作者侧自由度有限，远程操作可能非常困难。自主远程操作通过预测用户意图并自主执行某些任务的一部分，以减少对操作者的需求，提高任务完成率。然而，模式切换的决策通常假设由操作者完成，这增加了操作者要控制的自由度，并引入了额外的心理负担。另一方面，目前的文献中并未研究通信的角度，尽管通信不完美和资源限制是远程操作的主要瓶颈。在本研究中，我们提出了一种智能模式切换框架，同时考虑模式切换和通信系统。用户意图识别在操作者一侧完成。基于用户意图识别，一个深度强化学习算法被用于决定何时切换到自主模式。

    Teleoperation can be very difficult due to limited perception, high communication latency, and limited degrees of freedom (DoFs) at the operator side. Autonomous teleoperation is proposed to overcome this difficulty by predicting user intentions and performing some parts of the task autonomously to decrease the demand on the operator and increase the task completion rate. However, decision-making for mode-switching is generally assumed to be done by the operator, which brings an extra DoF to be controlled by the operator and introduces extra mental demand. On the other hand, the communication perspective is not investigated in the current literature, although communication imperfections and resource limitations are the main bottlenecks for teleoperation. In this study, we propose an intelligent mode-switching framework by jointly considering mode-switching and communication systems. User intention recognition is done at the operator side. Based on user intention recognition, a deep rei
    
[^101]: 针对低预算主动学习的直接采集优化

    Direct Acquisition Optimization for Low-Budget Active Learning

    [https://arxiv.org/abs/2402.06045](https://arxiv.org/abs/2402.06045)

    本研究提出了一种针对低预算主动学习的直接采集优化算法（DAO），该算法可以更准确地估计总体误差减少，效果超过了现有的方法。

    

    主动学习（AL）在将数据密集型机器学习（ML）模型集成到有限标记数据领域中变得越来越重要，然而，当标记预算较低时其效果显著降低。在本文中，首先我们经验性地观察了现有低预算设置下AL算法的性能退化，并引入了一种新的AL算法——直接采集优化（DAO），该算法根据期望真实损失减少来优化样本选择。具体而言，DAO利用影响函数来更新模型参数，并结合额外的采集策略来减轻损失估计中的偏差。这种方法能够更准确地估计总体误差减少，而无需进行大量计算或依赖于有标签的数据。实验证明，在七个基准测试中，DAO在低预算设置下的有效性超过了最先进的方法。

    Active Learning (AL) has gained prominence in integrating data-intensive machine learning (ML) models into domains with limited labeled data. However, its effectiveness diminishes significantly when the labeling budget is low. In this paper, we first empirically observe the performance degradation of existing AL algorithms in the low-budget settings, and then introduce Direct Acquisition Optimization (DAO), a novel AL algorithm that optimizes sample selections based on expected true loss reduction. Specifically, DAO utilizes influence functions to update model parameters and incorporates an additional acquisition strategy to mitigate bias in loss estimation. This approach facilitates a more accurate estimation of the overall error reduction, without extensive computations or reliance on labeled data. Experiments demonstrate DAO's effectiveness in low budget settings, outperforming state-of-the-arts approaches across seven benchmarks.
    
[^102]: 免先验正无标（Positive Unlabeled）学习的对比方法

    Contrastive Approach to Prior Free Positive Unlabeled Learning

    [https://arxiv.org/abs/2402.06038](https://arxiv.org/abs/2402.06038)

    该论文提出了一种免先验正无标学习的对比方法，通过预训练不变表示学习特征空间并利用嵌入的浓度特性对未标记样本进行伪标签处理，相比现有方法，在多个标准数据集上表现优异，同时不需要先验知识或类先验的估计。

    

    正无标（Positive Unlabeled）学习是指在给定少量标记的正样本和一组未标记样本（可能是正例或负例）的情况下学习一个二分类器的任务。在本文中，我们提出了一种新颖的正无标学习框架，通过保证不变表示学习学习特征空间，并利用嵌入的浓度特性对未标记样本进行伪标签处理。总体而言，我们提出的方法在多个标准正无标基准数据集上轻松超越了现有的正无标学习方法，而不需要先验知识或类先验的估计。值得注意的是，我们的方法在标记数据稀缺的情况下仍然有效，而大多数正无标学习算法则失败。我们还提供了简单的理论分析来推动我们提出的算法，并为我们的方法建立了一般化保证。

    Positive Unlabeled (PU) learning refers to the task of learning a binary classifier given a few labeled positive samples, and a set of unlabeled samples (which could be positive or negative). In this paper, we propose a novel PU learning framework, that starts by learning a feature space through pretext-invariant representation learning and then applies pseudo-labeling to the unlabeled examples, leveraging the concentration property of the embeddings. Overall, our proposed approach handily outperforms state-of-the-art PU learning methods across several standard PU benchmark datasets, while not requiring a-priori knowledge or estimate of class prior. Remarkably, our method remains effective even when labeled data is scant, where most PU learning algorithms falter. We also provide simple theoretical analysis motivating our proposed algorithms and establish generalization guarantee for our approach.
    
[^103]: 使用迷你像素批量梯度下降优化物理设计流程中的预测AI

    Optimizing Predictive AI in Physical Design Flows with Mini Pixel Batch Gradient Descent

    [https://arxiv.org/abs/2402.06034](https://arxiv.org/abs/2402.06034)

    本论文提出了一种迷你像素批量梯度下降（MPGD）算法，用于优化物理设计流程中的预测AI。实验证明MPGD在各种物理设计预测任务中具有显著的优势。

    

    爆炸式的预测AI在现代芯片物理设计流程中实现了快速而有效的评估和决策。现有的最先进框架通常包括最小化预测与真实值之间的均方误差（MSE）的目标。我们认为MSE的平均效果导致模型训练和部署两方面都存在局限性，而良好的MSE行为不能保证这些模型在可能由于少量预测误差而受损的物理设计流程中的能力。为了解决这个问题，我们提出了迷你像素批量梯度下降（MPGD），这是一种即插即用的优化算法，它考虑了最具信息量的条目，可能提供更快更好的收敛性。在代表性基准套件上的实验表明，MPGD在使用CNN或基于图的模型进行各种物理设计预测任务时具有显著的优势。

    Exploding predictive AI has enabled fast yet effective evaluation and decision-making in modern chip physical design flows. State-of-the-art frameworks typically include the objective of minimizing the mean square error (MSE) between the prediction and the ground truth. We argue the averaging effect of MSE induces limitations in both model training and deployment, and good MSE behavior does not guarantee the capability of these models to assist physical design flows which are likely sabotaged due to a small portion of prediction error. To address this, we propose mini-pixel batch gradient descent (MPGD), a plug-and-play optimization algorithm that takes the most informative entries into consideration, offering probably faster and better convergence. Experiments on representative benchmark suits show the significant benefits of MPGD on various physical design prediction tasks using CNN or Graph-based models.
    
[^104]: 不精确的Halpern迭代算法及其在分布鲁棒优化中的应用

    An Inexact Halpern Iteration for with Application to Distributionally Robust Optimization

    [https://arxiv.org/abs/2402.06033](https://arxiv.org/abs/2402.06033)

    本文研究了确定性和随机环境下Halpern迭代算法的不精确变种，通过适当选择不精确的容差，这些变种展现出O(k^-1)的收敛速度，同时具有竞争性的收敛特性。并且我们还展示了这些方法在两类数据驱动Wasserstein分布鲁棒优化问题中的应用，以及在分布鲁棒学习中使用随机一阶方法进行不精确计算的能力。

    

    Halpern迭代算法因其简单形式和吸引人的收敛性质，近年来在解决单调包含问题方面引起了越来越多的关注。本文研究了确定性和随机环境下该方案的不精确变种。我们进行了广泛的收敛性分析，并表明通过适当选择不精确的容差，不精确方案在（期望的）残差范数上具有O(k^-1)的收敛速度。我们的结果放宽了文献中采用的最新不精确性条件，同时具有相同的竞争性收敛特性。然后，我们演示了如何使用所提出的方法解决两类具有凸凹最小-最大优化重构的数据驱动Wasserstein分布鲁棒优化问题。我们强调了其在使用随机一阶方法进行分布鲁棒学习中的不精确计算能力。

    The Halpern iteration for solving monotone inclusion problems has gained increasing interests in recent years due to its simple form and appealing convergence properties. In this paper, we investigate the inexact variants of the scheme in both deterministic and stochastic settings. We conduct extensive convergence analysis and show that by choosing the inexactness tolerances appropriately, the inexact schemes admit an $O(k^{-1})$ convergence rate in terms of the (expected) residue norm. Our results relax the state-of-the-art inexactness conditions employed in the literature while sharing the same competitive convergence properties. We then demonstrate how the proposed methods can be applied for solving two classes of data-driven Wasserstein distributionally robust optimization problems that admit convex-concave min-max optimization reformulations. We highlight its capability of performing inexact computations for distributionally robust learning with stochastic first-order methods.
    
[^105]: 参数到可观测映射的算子学习视角

    An operator learning perspective on parameter-to-observable maps

    [https://arxiv.org/abs/2402.06031](https://arxiv.org/abs/2402.06031)

    本论文从算子学习的视角研究了参数到可观测映射，提出了适应有限维输入和输出的傅里叶神经映射框架，并发展了通用逼近定理来支持该方法。此外，讨论了学习PtO映射的端到端方法和先学习解算子再计算可观测值的效率问题。

    

    计算高效的参数化物理模型替代品在科学和工程中起着至关重要的作用。算子学习提供了一个数据驱动的替代方案，可以在函数空间中进行映射。然而，通常只有有限维的模型输入参数化或有限维的模型输出可观测数据可用，而不是全场测量数据。本文基于傅里叶神经算子，引入了傅里叶神经映射（Fourier Neural Mappings，FNMs）框架，能够适应这样的有限维输入和输出。本文为该方法发展了通用逼近定理。此外，在许多应用中，底层的参数到可观测（PtO）映射是通过无穷维算子来隐式定义的，例如偏微分方程的解算子。一个自然的问题是，是更有效地学习PtO映射的端到端方法，还是首先学习解算子，然后计算可观测值。

    Computationally efficient surrogates for parametrized physical models play a crucial role in science and engineering. Operator learning provides data-driven surrogates that map between function spaces. However, instead of full-field measurements, often the available data are only finite-dimensional parametrizations of model inputs or finite observables of model outputs. Building off of Fourier Neural Operators, this paper introduces the Fourier Neural Mappings (FNMs) framework that is able to accommodate such finite-dimensional inputs and outputs. The paper develops universal approximation theorems for the method. Moreover, in many applications the underlying parameter-to-observable (PtO) map is defined implicitly through an infinite-dimensional operator, such as the solution operator of a partial differential equation. A natural question is whether it is more data-efficient to learn the PtO map end-to-end or first learn the solution operator and subsequently compute the observable fro
    
[^106]: 博弈论对图神经网络的反事实解释

    Game-theoretic Counterfactual Explanation for Graph Neural Networks

    [https://arxiv.org/abs/2402.06030](https://arxiv.org/abs/2402.06030)

    本文提出了一种半值法的、非学习的方法来生成图神经网络的反事实解释，消除了额外训练的需要。与其他流行的方法相比，计算Banzhaf值在识别反事实解释时需要更低的样本复杂性，并且可以实现四倍的加速。

    

    图神经网络（GNNs）在复杂网络中的节点分类任务中是一种强大的工具。然而，它们的决策过程对用户来说仍然是一个黑盒子，这使得理解其预测背后的推理变得困难。反事实解释（CFE）已经显示出增强机器学习模型可解释性的潜力。先前的基于学习的方法计算GNNs的CFE通常需要训练额外的图形。在本文中，我们提出了一种基于半值的、非学习的方法来生成节点分类任务的CFE，消除了任何额外训练的需要。我们的结果表明，与计算Shapley值等其他流行方法相比，计算Banzhaf值需要更低的样本复杂性来识别反事实解释。我们的经验证据表明，与Shapley值相比，计算Banzhaf值可以实现四倍的加速。我们还设计了一个阈值化方法。

    Graph Neural Networks (GNNs) have been a powerful tool for node classification tasks in complex networks. However, their decision-making processes remain a black-box to users, making it challenging to understand the reasoning behind their predictions. Counterfactual explanations (CFE) have shown promise in enhancing the interpretability of machine learning models. Prior approaches to compute CFE for GNNS often are learning-based approaches that require training additional graphs. In this paper, we propose a semivalue-based, non-learning approach to generate CFE for node classification tasks, eliminating the need for any additional training. Our results reveals that computing Banzhaf values requires lower sample complexity in identifying the counterfactual explanations compared to other popular methods such as computing Shapley values. Our empirical evidence indicates computing Banzhaf values can achieve up to a fourfold speed up compared to Shapley values. We also design a thresholding
    
[^107]: 决策理论引导的深度强化学习用于快速学习

    Decision Theory-Guided Deep Reinforcement Learning for Fast Learning

    [https://arxiv.org/abs/2402.06023](https://arxiv.org/abs/2402.06023)

    决策理论引导的深度强化学习（DT-guided DRL）通过整合决策理论原则，实现了对DRL智能体的有效初始引导，并促进了在复杂环境中更高效可靠的学习过程。

    

    本文介绍了一种新颖的方法，即决策理论引导的深度强化学习（DT-guided DRL），用于解决强化学习中固有的冷启动问题。通过整合决策理论原则，DT-guided DRL增强了智能体在复杂环境中的初始性能和鲁棒性，使得学习过程更高效可靠。我们的研究涵盖了两个主要问题背景：杆车和迷宫导航挑战。实验结果表明，决策理论的整合不仅有助于对DRL智能体进行有效的初始引导，还促进了在具有大规模和复杂状态空间的环境中更有结构和知情的探索策略。实验结果表明，与常规的DRL相比，DT-guided DRL能够提供显著更高的奖励。具体而言，在训练的初始阶段，DT-guided DRL的累积奖励增加了184%。

    This paper introduces a novel approach, Decision Theory-guided Deep Reinforcement Learning (DT-guided DRL), to address the inherent cold start problem in DRL. By integrating decision theory principles, DT-guided DRL enhances agents' initial performance and robustness in complex environments, enabling more efficient and reliable convergence during learning. Our investigation encompasses two primary problem contexts: the cart pole and maze navigation challenges. Experimental results demonstrate that the integration of decision theory not only facilitates effective initial guidance for DRL agents but also promotes a more structured and informed exploration strategy, particularly in environments characterized by large and intricate state spaces. The results of experiment demonstrate that DT-guided DRL can provide significantly higher rewards compared to regular DRL. Specifically, during the initial phase of training, the DT-guided DRL yields up to an 184% increase in accumulated reward. Mo
    
[^108]: 使用全局非凸优化软件检验足够分散条件

    Checking the Sufficiently Scattered Condition using a Global Non-Convex Optimization Software

    [https://arxiv.org/abs/2402.06019](https://arxiv.org/abs/2402.06019)

    本文提出了一种使用全局非凸优化软件Gurobi解决足够分散条件检验问题的方法，在实际场景中可以在合理的时间范围内进行检查。

    

    足够分散条件（SSC）是研究各种矩阵分解问题的可辨识性的关键条件，包括非负、最小体积、对称、单纯结构和多面体矩阵分解。足够分散条件可以确保计算得到的矩阵分解是唯一可辨识的，除了平凡的模糊不确定性。然而，一般情况下，这个条件是NP难问题。在本文中，我们展示了在实际场景中，在矩阵的秩不太大时，它可以在合理的时间内进行检查，将问题构建为一个非凸二次优化问题，并在有界集合上求解。我们使用全局非凸优化软件Gurobi，并在合成数据集和实际世界的高光谱图像上展示了该代码的可用性。

    The sufficiently scattered condition (SSC) is a key condition in the study of identifiability of various matrix factorization problems, including nonnegative, minimum-volume, symmetric, simplex-structured, and polytopic matrix factorizations. The SSC allows one to guarantee that the computed matrix factorization is unique/identifiable, up to trivial ambiguities. However, this condition is NP-hard to check in general. In this paper, we show that it can however be checked in a reasonable amount of time in realistic scenarios, when the factorization rank is not too large. This is achieved by formulating the problem as a non-convex quadratic optimization problem over a bounded set. We use the global non-convex optimization software Gurobi, and showcase the usefulness of this code on synthetic data sets and on real-world hyperspectral images.
    
[^109]: NPSVC++: 非并行分类器遇到表示学习

    NPSVC++: Nonparallel Classifiers Encounter Representation Learning

    [https://arxiv.org/abs/2402.06010](https://arxiv.org/abs/2402.06010)

    本文研究了一种称为非并行支持向量分类器(NPSVCs)的分类器家族，提出了NPSVC++，基于多目标优化。NPSVC++通过表示学习实现了NPSVC及其特征的端到端学习，追求帕累托最优，有效地解决了特征次优和类别依赖的问题，在实验证明了其优越性。

    

    本文侧重于一种特定的分类器家族，称为非并行支持向量分类器(NPSVCs)。与典型的分类器不同，NPSVC的训练涉及多目标的最小化，导致特征次优和类别依赖的潜在问题。因此，尚未建立有效的学习方案来通过表示学习，特别是深度学习来改善NPSVC的性能。为了突破这一瓶颈，我们基于多目标优化开发了NPSVC++，实现了NPSVC及其特征的端到端学习。通过追求帕累托最优，NPSVC++在理论上确保了跨类别的特征优化，从而有效地克服了上述两个问题。我们提出了一种基于对偶优化的通用学习过程，并基于此提供了两个可应用的实例，K-NPSVC++和D-NPSVC++。实验证明了它们在现有方法上的优越性，并验证了NPSVC++的有效性。

    This paper focuses on a specific family of classifiers called nonparallel support vector classifiers (NPSVCs). Different from typical classifiers, the training of an NPSVC involves the minimization of multiple objectives, resulting in the potential concerns of feature suboptimality and class dependency. Consequently, no effective learning scheme has been established to improve NPSVCs' performance through representation learning, especially deep learning. To break this bottleneck, we develop NPSVC++ based on multi-objective optimization, enabling the end-to-end learning of NPSVC and its features. By pursuing Pareto optimality, NPSVC++ theoretically ensures feature optimality across classes, hence effectively overcoming the two issues above. A general learning procedure via duality optimization is proposed, based on which we provide two applicable instances, K-NPSVC++ and D-NPSVC++. The experiments show their superiority over the existing methods and verify the efficacy of NPSVC++.
    
[^110]: X射线微CT系统通过机器学习辅助方法的能力增强

    Capability enhancement of the X-ray micro-tomography system via ML-assisted approaches

    [https://arxiv.org/abs/2402.05983](https://arxiv.org/abs/2402.05983)

    本文提出了一种基于卷积神经网络的深度学习模型，通过去除环形伪影来增强X射线微CT系统的能力。

    

    X射线微CT图像中的环形伪影是准确视觉解释和定量分析的主要关注点之一。X射线微CT扫描仪的几何结构类似于医用CT机器，只是样本以固定的源和探测器旋转。环形伪影是由MicroCT数据采集过程中探测器像素的缺陷或非线性响应引起的。MicroCT图像中的伪影经常严重到无法进行进一步的分析。因此，了解伪影的原因和潜在解决方案以最大化图像质量非常重要。本文提出了一种基于卷积神经网络（CNN）的深度学习（DL）模型，灵感来自UNet，其中包含一系列具有跳跃连接的编码器和解码器单元，用于去除环形伪影。该提出的架构利用结构相似性指数测量（SSIM）和均方误差（MSE）进行评估。此外，结果还进行了

    Ring artifacts in X-ray micro-CT images are one of the primary causes of concern in their accurate visual interpretation and quantitative analysis. The geometry of X-ray micro-CT scanners is similar to the medical CT machines, except the sample is rotated with a stationary source and detector. The ring artifacts are caused by a defect or non-linear responses in detector pixels during the MicroCT data acquisition. Artifacts in MicroCT images can often be so severe that the images are no longer useful for further analysis. Therefore, it is essential to comprehend the causes of artifacts and potential solutions to maximize image quality. This article presents a convolution neural network (CNN)-based Deep Learning (DL) model inspired by UNet with a series of encoder and decoder units with skip connections for removal of ring artifacts. The proposed architecture has been evaluated using the Structural Similarity Index Measure (SSIM) and Mean Squared Error (MSE). Additionally, the results ar
    
[^111]: Anfinsen Goes Neural: 一种用于条件抗体设计的图模型

    Anfinsen Goes Neural: a Graphical Model for Conditional Antibody Design

    [https://arxiv.org/abs/2402.05982](https://arxiv.org/abs/2402.05982)

    Anfinsen Goes Neural (AGN) is a graphical model for conditional antibody design that combines a pre-trained protein language model with a graph neural network. It outperforms existing methods and addresses the limitation of generating unrealistic sequences.

    

    抗体设计在推动治疗学方面起着关键作用。尽管深度学习在这个领域取得了快速进展，但现有方法对一般蛋白质知识的利用有限，并假设图模型违反蛋白质的经验发现。为了解决这些限制，我们提出了Anfinsen Goes Neural (AGN)，这是一个使用预训练的蛋白质语言模型(pLM)并编码了一种关于蛋白质的重要发现，即Anfinsen's dogma的图模型。我们的框架遵循序列生成和图神经网络(GNN)进行结构预测的两步过程。实验证明，我们的方法在基准实验中优于现有方法的结果。我们还解决了非自回归模型的一个关键限制，即它们倾向于生成具有过多重复标记的不现实序列。为了解决这个问题，我们引入了基于组合的正则化项到交叉熵目标中，可以实现有效的权衡。

    Antibody design plays a pivotal role in advancing therapeutics. Although deep learning has made rapid progress in this field, existing methods make limited use of general protein knowledge and assume a graphical model (GM) that violates empirical findings on proteins. To address these limitations, we present Anfinsen Goes Neural (AGN), a graphical model that uses a pre-trained protein language model (pLM) and encodes a seminal finding on proteins called Anfinsen's dogma. Our framework follows a two-step process of sequence generation with pLM and structure prediction with graph neural network (GNN). Experiments show that our approach outperforms state-of-the-art results on benchmark experiments. We also address a critical limitation of non-autoregressive models -- namely, that they tend to generate unrealistic sequences with overly repeating tokens. To resolve this, we introduce a composition-based regularization term to the cross-entropy objective that allows an efficient trade-off be
    
[^112]: 探索浏览器内深度学习推理对用户体验质量和性能的影响

    Exploring the Impact of In-Browser Deep Learning Inference on Quality of User Experience and Performance

    [https://arxiv.org/abs/2402.05981](https://arxiv.org/abs/2402.05981)

    本研究通过全面性能评估，探索了浏览器内深度学习推理对用户体验质量和性能的影响。研究发现，浏览器内推理存在严重的延迟问题，平均比原生推理方法慢16.9倍。为了衡量这种影响，我们引入了新的指标：响应性，流畅度和推理准确性。

    

    深度学习越来越多地通过“浏览器内推理”这种方法整合到Web应用程序中，其中DL处理直接在Web浏览器中进行。然而，这种方法的实际性能及其对用户体验质量（QoE）的影响尚不为人所知。这种知识的空白需要新形式的QoE测量，超越传统的指标，如页面加载时间。为了解决这个问题，我们进行了浏览器内推理的首次全面性能评估。我们引入了新的指标：响应性，流畅度和推理准确性。我们的全面研究包括9个广泛使用的DL模型，并在50个常用的PC Web浏览器上进行了测试。研究结果显示，浏览器内推理存在严重的延迟问题：在CPU上平均比原生推理方法慢16.9倍，在GPU上慢4.9倍。这种延迟有几个因素导致，包括未充分使用的硬件指令集，固有的延迟等。

    Deep Learning (DL) is increasingly being integrated into Web applications through a method known as "in-browser inference", where the DL processes occur directly within Web browsers. However, the actual performance of this method and its effect on user experience quality (QoE) is not well-understood. This gap in knowledge necessitates new forms of QoE measurement, going beyond traditional metrics such as page load time. To address this, we conducted the first extensive performance evaluation of in-browser inference. We introduced new metrics for this purpose: responsiveness, smoothness, and inference accuracy.   Our thorough study included 9 widely-used DL models and tested them across 50 popular PC Web browsers. The findings show a significant latency issue with in-browser inference: it's on average 16.9 times slower on CPU and 4.9 times slower on GPU than native inference methods. Several factors contribute to this latency, including underused hardware instruction sets, inherent dela
    
[^113]: 大型代码模型是否理解编程概念？一种黑盒方法探究

    Do Large Code Models Understand Programming Concepts? A Black-box Approach

    [https://arxiv.org/abs/2402.05980](https://arxiv.org/abs/2402.05980)

    本文使用反事实分析框架评估了十个大型代码模型对四种编程概念的理解情况，发现当前模型缺乏对数据流和控制流等概念的理解。

    

    大型语言模型在文本生成方面的成功也使其在代码生成和编码任务方面表现更好。虽然有很多工作展示了它们在代码补全和编辑等任务上的出色性能，但为什么它们能够成功还不清楚。我们通过探索自回归模型对底层程序的逻辑结构理解程度，来填补这一差距。我们提出了用于编程概念谓词的反事实分析（CACP）作为一种反事实测试框架，以评估大型代码模型是否理解编程概念。只通过黑盒访问模型，我们使用CACP评估了十个流行的大型代码模型对四个不同编程概念的理解情况。我们的研究结果表明，当前模型缺乏对数据流和控制流等概念的理解。

    Large Language Models' success on text generation has also made them better at code generation and coding tasks. While a lot of work has demonstrated their remarkable performance on tasks such as code completion and editing, it is still unclear as to why. We help bridge this gap by exploring to what degree auto-regressive models understand the logical constructs of the underlying programs. We propose Counterfactual Analysis for Programming Concept Predicates (CACP) as a counterfactual testing framework to evaluate whether Large Code Models understand programming concepts. With only black-box access to the model, we use CACP to evaluate ten popular Large Code Models for four different programming concepts. Our findings suggest that current models lack understanding of concepts such as data flow and control flow.
    
[^114]: 结合形状和轮廓特征来提高铣削过程中的刀具磨损监测

    Combining shape and contour features to improve tool wear monitoring in milling processes

    [https://arxiv.org/abs/2402.05978](https://arxiv.org/abs/2402.05978)

    本文提出了一种新的系统，结合了形状描述符和轮廓描述符，用于铣削过程中插入物的磨损监测。实验结果表明，使用后期融合方法将两个描述符组合在一起，可以显著提高分类性能，取得更好的准确率。

    

    本文提出了一种新的基于形状描述符和轮廓描述符组合的系统，用于根据磨损程度对铣削过程中的插入物进行分类，采用基于计算机视觉的方法。为了描述磨损区域的形状，我们提出了一种新的描述符ShapeFeat，并使用BORCHIZ方法对其轮廓进行表征，根据我们的调查，该方法在基于计算机视觉的刀具磨损监测中取得了最佳性能。实验结果表明，使用后期融合方法将BORCHIZ和ShapeFeat组合在一起，显著提高了分类性能，二元分类将磨损分为高或低的准确率达到91.44%，三个目标类别（高、中、低磨损）的准确率达到82.90%。这些结果优于单独使用两个描述符的结果，其准确率分别为88.70%和80.67%。

    In this paper, a new system based on combinations of a shape descriptor and a contour descriptor has been proposed for classifying inserts in milling processes according to their wear level following a computer vision based approach. To describe the wear region shape we have proposed a new descriptor called ShapeFeat and its contour has been characterized using the method BORCHIZ that, to the best of our knowledge, achieves the best performance for tool wear monitoring following a computer vision-based approach. Results show that the combination of BORCHIZ with ShapeFeat using a late fusion method improves the classification performance significantly, obtaining an accuracy of 91.44% in the binary classification (i.e. the classification of the wear as high or low) and 82.90% using three target classes (i.e. classification of the wear as high, medium or low). These results outperform the ones obtained by both descriptors used on their own, which achieve accuracies of 88.70 and 80.67% for
    
[^115]: 基于局部纹理的在线、自动和低成本系统进行刀具磨损监测

    Tool wear monitoring using an online, automatic and low cost system based on local texture

    [https://arxiv.org/abs/2402.05977](https://arxiv.org/abs/2402.05977)

    本研究提出了一种基于计算机视觉和机器学习的在线、低成本和快速方法，用于切削工具的磨损监测。通过将切削边缘图像分割成不同的区域，并使用局部二值模式的纹理描述符来判断每个区域的磨损程度，从而确定切削边缘和刀具是否可服役或可丢弃。

    

    在本研究中，我们提出了一种基于计算机视觉和机器学习的新的在线、低成本和快速方法，用于确定边缘轮廓铣削过程中使用的切削工具是否可服役或可丢弃，根据它们的磨损程度。我们创建了一个由254张边缘轮廓切削头图像组成的新数据集，根据我们所知，这是第一个公开可用且具有足够质量的数据集。所有刀片都被分割，并且其切削边缘被裁剪，获得了577张切削边缘图像：301张可用和276张可丢弃。所提出的方法基于（1）将切削边缘图像分为不同的区域，称为磨损斑块（WP），（2）使用基于不同类型的局部二值模式（LBP）的纹理描述符来表征每个区域是磨损还是可用，并（3）根据这些WP的状态来确定切削边缘（因此也是刀具）是否可服役或可丢弃。我们提出并评估了五种不同的斑块分割方法。

    In this work we propose a new online, low cost and fast approach based on computer vision and machine learning to determine whether cutting tools used in edge profile milling processes are serviceable or disposable based on their wear level. We created a new dataset of 254 images of edge profile cutting heads which is, to the best of our knowledge, the first publicly available dataset with enough quality for this purpose. All the inserts were segmented and their cutting edges were cropped, obtaining 577 images of cutting edges: 301 functional and 276 disposable. The proposed method is based on (1) dividing the cutting edge image in different regions, called Wear Patches (WP), (2) characterising each one as worn or serviceable using texture descriptors based on different variants of Local Binary Patterns (LBP) and (3) determine, based on the state of these WP, if the cutting edge (and, therefore, the tool) is serviceable or disposable. We proposed and assessed five different patch divis
    
[^116]: RankSum：一种基于排名融合的无监督抽取式文本摘要方法

    RankSum An unsupervised extractive text summarization based on rank fusion

    [https://arxiv.org/abs/2402.05976](https://arxiv.org/abs/2402.05976)

    RankSum是一种无监督的抽取式文本摘要方法，它利用多维度句子特征对句子进行排名，然后通过加权融合确定句子的重要性排名。该方法不需要监督信号，可以推广到其他数据集。

    

    本文提出了一种名为Ranksum的方法，用于基于排名融合的无监督单文档抽取式文本摘要。该方法利用为每个句子提取的四个多维度句子特征进行句子显著性排名：主题信息、语义内容、重要关键词和位置。Ranksum根据每个特征生成的句子显著性排名进行加权融合，以确定句子的重要性排名。融合权重是完全无监督生成的，需要标记的文档集合来学习融合权重。我们发现融合权重可以推广到其他数据集，因此将Ranksum视为一种无监督方法。为了确定主题排名，我们使用概率主题模型，而使用句子嵌入来捕捉语义信息。使用句子嵌入来生成排名时，我们利用连体网络产生抽象化的句子表示，然后形成排名。

    In this paper, we propose Ranksum, an approach for extractive text summarization of single documents based on the rank fusion of four multi-dimensional sentence features extracted for each sentence: topic information, semantic content, significant keywords, and position. The Ranksum obtains the sentence saliency rankings corresponding to each feature in an unsupervised way followed by the weighted fusion of the four scores to rank the sentences according to their significance. The scores are generated in completely unsupervised way, and a labeled document set is required to learn the fusion weights. Since we found that the fusion weights can generalize to other datasets, we consider the Ranksum as an unsupervised approach. To determine topic rank, we employ probabilistic topic models whereas semantic information is captured using sentence embeddings. To derive rankings using sentence embeddings, we utilize Siamese networks to produce abstractive sentence representation and then we form
    
[^117]: 一种使用多尺度卷积神经网络的深度学习方法进行脑肿瘤分类和分割

    A Deep Learning Approach for Brain Tumor Classification and Segmentation Using a Multiscale Convolutional Neural Network

    [https://arxiv.org/abs/2402.05975](https://arxiv.org/abs/2402.05975)

    本文提出了一种使用多尺度卷积神经网络的深度学习方法，可以自动进行脑肿瘤的分类和分割。通过与其他方法的比较，我们的方法在公开数据集上获得了较高的分类准确率。

    

    在本文中，我们提出了一种完全自动的脑肿瘤分割和分类模型，使用了包括多尺度方法在内的深度卷积神经网络。我们的方法与之前的工作的一个区别是输入图像在不同处理路径上以三个空间尺度进行处理。这个机制是受到人类视觉系统的内在操作的启示。提出的神经模型可以分析包含三种类型肿瘤（脑膜瘤、胶质瘤和垂体瘤）的MRI图像，包括矢状面、冠状面和轴面，并且不需要预处理输入图像事先移除头骨或椎骨部分。我们的方法在一个公开可用的包含233名患者3064张切片的MRI图像数据集上的性能与之前的经典机器学习和深度学习方法进行了比较。在比较中，我们的方法明显地获得了0.973的肿瘤分类准确率，高于其他方法。

    In this paper, we present a fully automatic brain tumor segmentation and classification model using a Deep Convolutional Neural Network that includes a multiscale approach. One of the differences of our proposal with respect to previous works is that input images are processed in three spatial scales along different processing pathways. This mechanism is inspired in the inherent operation of the Human Visual System. The proposed neural model can analyze MRI images containing three types of tumors: meningioma, glioma, and pituitary tumor, over sagittal, coronal, and axial views and does not need preprocessing of input images to remove skull or vertebral column parts in advance. The performance of our method on a publicly available MRI image dataset of 3064 slices from 233 patients is compared with previously classical machine learning and deep learning published methods. In the comparison, our method remarkably obtained a tumor classification accuracy of 0.973, higher than the other app
    
[^118]: 区块链技术支持的聚簇可扩展联邦学习（BCS-FL）框架在无人机网络中的应用

    Blockchain-enabled Clustered and Scalable Federated Learning (BCS-FL) Framework in UAV Networks

    [https://arxiv.org/abs/2402.05973](https://arxiv.org/abs/2402.05973)

    本论文提出了一种基于区块链的聚簇可扩展联邦学习（BCS-FL）框架，用于改善无人机网络中的联邦学习的去中心化、协调、可扩展性和效率。该框架将无人机网络划分为聚簇，并由聚簇头无人机进行协调，以提高大规模无人机网络中的联邦学习性能。

    

    隐私、可扩展性和可靠性是无人机网络作为分布式系统所面临的重要挑战，特别是在使用大量数据交换的机器学习（ML）技术时。最近，在无人机网络中应用联邦学习（FL）改善了协作、隐私、韧性和适应性，成为无人机应用的一种有前景的框架。然而，为无人机网络实现FL引入了通信开销、同步问题、可扩展性限制和资源约束等缺点。为了解决这些挑战，本文提出了一种基于区块链的聚簇可扩展联邦学习（BCS-FL）框架，用于无人机网络。该框架将无人机网络划分为分离的聚簇，并由聚簇头无人机（CHs）进行协调，以建立一个连通图。聚簇化使得联邦学习在大规模无人机网络中的去中心化、协调、可扩展性和效率得到了改善。

    Privacy, scalability, and reliability are significant challenges in unmanned aerial vehicle (UAV) networks as distributed systems, especially when employing machine learning (ML) technologies with substantial data exchange. Recently, the application of federated learning (FL) to UAV networks has improved collaboration, privacy, resilience, and adaptability, making it a promising framework for UAV applications. However, implementing FL for UAV networks introduces drawbacks such as communication overhead, synchronization issues, scalability limitations, and resource constraints. To address these challenges, this paper presents the Blockchain-enabled Clustered and Scalable Federated Learning (BCS-FL) framework for UAV networks. This improves the decentralization, coordination, scalability, and efficiency of FL in large-scale UAV networks. The framework partitions UAV networks into separate clusters, coordinated by cluster head UAVs (CHs), to establish a connected graph. Clustering enables
    
[^119]: 基于高斯过程回归的复共振谱中例外点定位方法

    Gaussian-process-regression-based method for the localization of exceptional points in complex resonance spectra

    [https://arxiv.org/abs/2402.05972](https://arxiv.org/abs/2402.05972)

    本文介绍了一种基于高斯过程回归的方法，用于在复共振谱中定位例外点。通过训练一个GPR模型，并使用一些初始的特征值对进行根搜索，来对例外点的位置进行初步估计。然后通过迭代添加确切的特征值对来改进估计。该方法在低维矩阵模型和真实物理系统上进行了测试。

    

    依赖于至少两个可控参数的开放量子系统中的共振可以显示出例外点（EP）现象，其中两个或多个共振的特征值和特征向量同时存在。在参数空间中准确定位这些点是具有挑战性的，特别是在计算量子谱和共振的数值计算非常昂贵的系统中。本文介绍了一种利用高斯过程回归（GPR）的高效机器学习算法来寻找例外点的方法。GPR模型基于一个初始的属于EP的特征值对进行训练，并通过数值较便宜的根搜索进行EP位置的初步估计。然后，通过将选择的确切特征值对添加为训练点来迭代改进估计。本文基于一个简单的低维矩阵模型开发和测试了GPR方法，然后将其应用于一个具有挑战性的真实物理系统，即定位EP。

    Resonances in open quantum systems depending on at least two controllable parameters can show the phenomenon of exceptional points (EPs), where not only the eigenvalues but also the eigenvectors of two or more resonances coalesce. Their exact localization in the parameter space is challenging, in particular in systems, where the computation of the quantum spectra and resonances is numerically very expensive. We introduce an efficient machine learning algorithm to find exceptional points based on Gaussian process regression (GPR). The GPR-model is trained with an initial set of eigenvalue pairs belonging to an EP and used for a first estimation of the EP position via a numerically cheap root search. The estimate is then improved iteratively by adding selected exact eigenvalue pairs as training points to the GPR-model. The GPR-based method is developed and tested on a simple low-dimensional matrix model and then applied to a challenging real physical system, viz., the localization of EPs
    
[^120]: 我们取得了多少进展？从不平衡回归的角度重新审视化学反应收率预测

    Are we making much progress? Revisiting chemical reaction yield prediction from an imbalanced regression perspective

    [https://arxiv.org/abs/2402.05971](https://arxiv.org/abs/2402.05971)

    本文从不平衡回归的角度重新审视了化学反应收率预测。在合成规划中，准确的高收率预测对于化学家来说至关重要。然而，真实世界数据的不平衡分布导致了现有方法在高收率预测方面的性能差距。

    

    化学反应的收率是指目标产物形成的百分比与化学反应过程中消耗的反应物之间的关系。准确的收率预测可以在合成规划中指导化学家选择高收率反应，在投入时间和资源进行湿实验之前提供宝贵的见解。虽然最近在收率预测方面取得了整体性能的改进，但在提高高收率反应的预测方面仍存在一个开放性挑战，这对于化学家来说更为重要。本文认为高收率预测方面的性能差距是由于真实世界数据的不平衡分布所致，这些数据偏向于低收率反应，通常是由于未反应的起始物质和反应过程中的固有歧义性。尽管存在数据不平衡，现有的收率预测方法继续将不同收率范围视为平衡的训练情况。

    The yield of a chemical reaction quantifies the percentage of the target product formed in relation to the reactants consumed during the chemical reaction. Accurate yield prediction can guide chemists toward selecting high-yield reactions during synthesis planning, offering valuable insights before dedicating time and resources to wet lab experiments. While recent advancements in yield prediction have led to overall performance improvement across the entire yield range, an open challenge remains in enhancing predictions for high-yield reactions, which are of greater concern to chemists. In this paper, we argue that the performance gap in high-yield predictions results from the imbalanced distribution of real-world data skewed towards low-yield reactions, often due to unreacted starting materials and inherent ambiguities in the reaction processes. Despite this data imbalance, existing yield prediction methods continue to treat different yield ranges equally, assuming a balanced training
    
[^121]: 使用神经离散学习和专家级别建模空时动力系统

    Modeling Spatio-temporal Dynamical Systems with Neural Discrete Learning and Levels-of-Experts

    [https://arxiv.org/abs/2402.05970](https://arxiv.org/abs/2402.05970)

    本文提出了使用神经离散学习和专家级别建模空时动力系统的方法。通过引入通用的专家模块和精细设计的物理流水线，可以在更广泛的现实世界背景下有效地建模和估计空时动态系统的状态变化。

    

    本文针对基于一系列观测（如视频帧）的空时动态系统中状态变化的建模和估计问题进行了研究。传统的数值模拟系统在很大程度上依赖于初始设置和构建的偏微分方程（PDE）的正确性。尽管最近利用神经网络发现了基于数据的PDE模型的重大成功，但是奇异场景和缺乏局部洞察力的限制阻碍了它们在更广泛的现实世界背景下的有效性。为此，本文提出了通用的专家模块——光流估计组件，以数据驱动的方式捕捉一般物理过程的演化规律。为了增强局部洞察力，我们精心设计了一个更精细的物理流水线，因为局部特征可能受到各种内部上下文信息的影响，这可能与宏观属性相矛盾。

    In this paper, we address the issue of modeling and estimating changes in the state of the spatio-temporal dynamical systems based on a sequence of observations like video frames. Traditional numerical simulation systems depend largely on the initial settings and correctness of the constructed partial differential equations (PDEs). Despite recent efforts yielding significant success in discovering data-driven PDEs with neural networks, the limitations posed by singular scenarios and the absence of local insights prevent them from performing effectively in a broader real-world context. To this end, this paper propose the universal expert module -- that is, optical flow estimation component, to capture the evolution laws of general physical processes in a data-driven fashion. To enhance local insight, we painstakingly design a finer-grained physical pipeline, since local characteristics may be influenced by various internal contextual information, which may contradict the macroscopic pro
    
[^122]: 打破训练Transformer时的对称性

    Breaking Symmetry When Training Transformers

    [https://arxiv.org/abs/2402.05969](https://arxiv.org/abs/2402.05969)

    该论文讨论了在训练Transformer时，删除位置编码和因果注意力机制后，输出的预测结果对于输入符号排列是不变的。研究人员通过对因果连接机制进行细致分析，提出了残差连接对Transformer模拟输入顺序重要性的贡献。

    

    正如我们在本文中展示的那样，没有位置编码和因果注意力机制的Transformer架构对于输入符号1, 2, ..., n-1的排列是不变的。通常情况下，这两种机制都会被使用，以打破对输入符号的对称性。最近已经表明，可以在没有位置编码的情况下训练Transformer。这必须通过因果注意机制来实现。在本文中，我们详细阐述了因果连接机制必须是使Transformer能够模拟输入顺序重要性的原因。Transformer的垂直“切片”都被鼓励表示输入序列中的相同位置k。我们假设残差连接对于这种现象起到了贡献，并提供了证据支持这一观点。

    As we show in this paper, the prediction for output token $n+1$ of Transformer architectures without one of the mechanisms of positional encodings and causal attention is invariant to permutations of input tokens $1, 2, ..., n-1$. Usually, both mechanisms are employed and the symmetry with respect to the input tokens is broken. Recently, it has been shown that one can train Transformers without positional encodings. This must be enabled by the causal attention mechanism. In this paper, we elaborate on the argument that the causal connection mechanism must be responsible for the fact that Transformers are able to model input sequences where the order is important. Vertical "slices" of Transformers are all encouraged to represent the same location $k$ in the input sequence. We hypothesize that residual connections contribute to this phenomenon, and demonstrate evidence for this.
    
[^123]: 欧盟人工智能法案下的联邦学习优先事项

    Federated Learning Priorities Under the European Union Artificial Intelligence Act

    [https://arxiv.org/abs/2402.05968](https://arxiv.org/abs/2402.05968)

    欧盟人工智能法案可能推动联邦学习朝主流采用方向发展，并提出了数据隐私、性能和能源效率等方面的新挑战。

    

    AI监管时代已经来临，欧盟人工智能法案（AI Act）引领着潮流。我们的关键问题是，这将如何影响以数据隐私为优先并进行机器学习的联邦学习（FL），其与集中式学习的出发点根本不同。我们相信AI法案和未来的监管可能是推动FL走向主流采用的缺失催化剂。然而，这只能发生在FL社区重新优先考虑其研究重点的情况下。在我们的立场论文中，我们进行了首次跨学科分析（法律和机器学习），分析了AI法案对FL可能产生的影响，并通过定量和定性分析进行了一系列支持我们主要观点的观察。我们探讨了数据治理问题和对隐私的担忧。我们确定了在生命周期监视中性能和能源效率方面的新挑战。综合我们的分析，表明FL有着巨大的机会，

    The age of AI regulation is upon us, with the European Union Artificial Intelligence Act (AI Act) leading the way. Our key inquiry is how this will affect Federated Learning (FL), whose starting point of prioritizing data privacy while performing ML fundamentally differs from that of centralized learning. We believe the AI Act and future regulations could be the missing catalyst that pushes FL toward mainstream adoption. However, this can only occur if the FL community reprioritizes its research focus. In our position paper, we perform a first-of-its-kind interdisciplinary analysis (legal and ML) of the impact the AI Act may have on FL and make a series of observations supporting our primary position through quantitative and qualitative analysis. We explore data governance issues and the concern for privacy. We establish new challenges regarding performance and energy efficiency within lifecycle monitoring. Taken together, our analysis suggests there is a sizable opportunity for FL to 
    
[^124]: 最后之舞：通过扩散模型和贝叶斯方法进行鲁棒后门攻击

    The last Dance : Robust backdoor attack via diffusion models and bayesian approach

    [https://arxiv.org/abs/2402.05967](https://arxiv.org/abs/2402.05967)

    本文介绍了一种通过扩散模型和贝叶斯方法进行鲁棒后门攻击的方法，具体应用于音频Transformer模型，并证明了攻击的可行性。

    

    扩散模型是最先进的深度学习生成模型，其通过逐步添加噪音和去噪的方式学习正向和反向扩散过程的原理进行训练。本文旨在欺骗基于音频的DNN模型，例如Hugging Face框架中的音频模型，特别是基于Transformer的人工智能模型，这些模型是强大的机器学习模型，节省时间，提供更高效的结果。我们证明了在Hugging Face推导出的音频Transformer上实现后门攻击（称为`BacKBayDiffMod`）的可行性。本文中开发的后门攻击基于毒化模型的训练数据，涉及后门扩散采样和贝叶斯方法分布的引入。

    Diffusion models are state-of-the-art deep learning generative models that are trained on the principle of learning forward and backward diffusion processes via the progressive addition of noise and denoising. In this paper, we seek to trick audio-based DNN models, such as those in the Hugging Face framework, for example, those that focus on audio, in particular transformer-based artificial intelligence models, which are powerful machine learning models that save time and deliver faster, more efficient results. We demonstrate the feasibility of backdoor attacks (called `BacKBayDiffMod`) on audio transformers derived from Hugging Face, a popular framework in the world of artificial intelligence (AI) research. The backdoor attack developed in this paper is based on poisoning the model's training data by incorporating backdoor diffusion sampling and a Bayesian approach to the distribution of poisoned data.
    
[^125]: 重新思考模型重新基底和线性模态连接性

    Rethink Model Re-Basin and the Linear Mode Connectivity

    [https://arxiv.org/abs/2402.05966](https://arxiv.org/abs/2402.05966)

    本论文重新审视了模型重新基底的现象，并发现了现有匹配算法的不足。通过适当的重归一化，我们改进了匹配算法，并揭示了它与重归一化过程的相互作用。这为剪枝提供了新的理解，推动了一种轻量且有效的后剪枝插件的开发。

    

    最近的研究表明，对于足够宽的模型来说，大部分随机梯度下降（SGD）的解可以收敛到相同的基底，只是顺序可能不同。这种现象被称为模型重新基底的阶段，对于模型平均化有重要影响。然而，当前的重新基底策略在效果上存在局限性，因为对底层机制的理解不够全面。为了填补这一空白，我们的研究重新审视了标准做法，并揭示了现有匹配算法的频繁不足之处，我们通过适当的重归一化来缓解这些问题。通过引入更直接的分析方法，我们揭示了匹配算法与重归一化过程之间的相互作用。这种观点不仅澄清和改进了以前的研究结果，还促进了新的洞见。例如，它将线性模态连接性与剪枝联系起来，从而激发了一种轻量且有效的后剪枝插件，可以直接与任何现有的剪枝技术合并。

    Recent studies suggest that with sufficiently wide models, most SGD solutions can, up to permutation, converge into the same basin. This phenomenon, known as the model re-basin regime, has significant implications for model averaging. However, current re-basin strategies are limited in effectiveness due to a lack of comprehensive understanding of underlying mechanisms. Addressing this gap, our work revisits standard practices and uncovers the frequent inadequacies of existing matching algorithms, which we show can be mitigated through proper re-normalization. By introducing a more direct analytical approach, we expose the interaction between matching algorithms and re-normalization processes. This perspective not only clarifies and refines previous findings but also facilitates novel insights. For instance, it connects the linear mode connectivity to pruning, motivating a lightweight yet effective post-pruning plug-in that can be directly merged with any existing pruning techniques. Ou
    
[^126]: 球面数据的混合神经表示研究

    Hybrid Neural Representations for Spherical Data

    [https://arxiv.org/abs/2402.05965](https://arxiv.org/abs/2402.05965)

    本文研究了球面数据的混合神经表示方法，通过使用球形特征格和多层感知器进行预测，有效地捕捉了高度非线性信号的复杂细节。

    

    本文研究了球面数据的混合神经表示，这是科学研究中越来越相关的领域。我们的工作重点是天气和气候数据以及宇宙微波背景（CMB）数据。尽管以前的研究探索了基于坐标的球形信号神经表示，但它们经常无法捕捉到高度非线性信号的复杂细节。为了解决这个限制，我们引入了一种名为混合神经表示球面数据（HNeR-S）的新方法。我们的主要思想是使用球形特征格来获取位置特征，然后与多层感知器结合来预测目标信号。我们考虑了与天气数据和CMB数据对齐的等距像素化结构的等矩形和等面积隔等纬度特征格。我们广泛验证了我们的HNeR-S在回归、超分辨率、时间插值和压缩方面的有效性

    In this paper, we study hybrid neural representations for spherical data, a domain of increasing relevance in scientific research. In particular, our work focuses on weather and climate data as well as comic microwave background (CMB) data. Although previous studies have delved into coordinate-based neural representations for spherical signals, they often fail to capture the intricate details of highly nonlinear signals. To address this limitation, we introduce a novel approach named Hybrid Neural Representations for Spherical data (HNeR-S). Our main idea is to use spherical feature-grids to obtain positional features which are combined with a multilayer perception to predict the target signal. We consider feature-grids with equirectangular and hierarchical equal area isolatitude pixelization structures that align with weather data and CMB data, respectively. We extensively verify the effectiveness of our HNeR-S for regression, super-resolution, temporal interpolation, and compression 
    
[^127]: 《Transformer压缩调研》

    A Survey on Transformer Compression

    [https://arxiv.org/abs/2402.05964](https://arxiv.org/abs/2402.05964)

    《Transformer压缩调研》是对最近压缩方法的全面回顾，特别关注它们在Transformer模型中的应用。压缩方法主要分为修剪、量化、知识蒸馏和高效架构设计四个类别。

    

    基于Transformer架构的大型模型在人工智能领域，特别是自然语言处理（NLP）和计算机视觉（CV）领域中扮演着日益重要的角色。模型压缩方法可以减少模型的内存和计算成本，是在实际设备上实现Transformer模型的必要步骤。鉴于Transformer的独特架构，具有交替的注意力和前馈神经网络（FFN）模块，需要特定的压缩技术。这些压缩方法的效率也至关重要，因为重新训练整个训练数据集上的大型模型往往是不切实际的。本调研提供了对最近压缩方法的全面回顾，特别关注它们在Transformer模型中的应用。压缩方法主要分为修剪、量化、知识蒸馏和高效架构设计四个类别。在每个类别中，我们讨论了压缩方法

    Large models based on the Transformer architecture play increasingly vital roles in artificial intelligence, particularly within the realms of natural language processing (NLP) and computer vision (CV). Model compression methods reduce their memory and computational cost, which is a necessary step to implement the transformer models on practical devices. Given the unique architecture of transformer, featuring alternative attention and Feedforward Neural Network (FFN) modules, specific compression techniques are required. The efficiency of these compression methods is also paramount, as it is usually impractical to retrain large models on the entire training dataset.This survey provides a comprehensive review of recent compression methods, with a specific focus on their application to transformer models. The compression methods are primarily categorized into pruning, quantization, knowledge distillation, and efficient architecture design. In each category, we discuss compression methods
    
[^128]: 节俭的演员-评论家模型：使用独特经历的高效离线深度强化学习

    Frugal Actor-Critic: Sample Efficient Off-Policy Deep Reinforcement Learning Using Unique Experiences

    [https://arxiv.org/abs/2402.05963](https://arxiv.org/abs/2402.05963)

    该方法通过选择独特样本并添加到回放缓冲器中以实现样本效率，在复杂动态系统的无模型控制策略合成中起着重要作用。

    

    在用于复杂动态系统的无模型控制策略合成中，对回放缓冲器的高效利用在离线演员-评论家强化学习算法中起着重要作用。我们提出了一种实现样本效率的方法，该方法通过在探索过程中选择独特样本并将其添加到回放缓冲器中，旨在减小缓冲器的大小并保持样本的独立同分布（IID）的性质。我们的方法基于在随机探索的初始阶段遇到的经历中选择一组重要的状态变量的重要子集，根据所选重要状态变量将状态空间划分为一组抽象状态，最后通过使用核密度估计器选择具有独特状态-奖励组合的经历。我们严格证明了将所提出的独特经历方法纳入离线演员-评论家算法中的有效性。

    Efficient utilization of the replay buffer plays a significant role in the off-policy actor-critic reinforcement learning (RL) algorithms used for model-free control policy synthesis for complex dynamical systems. We propose a method for achieving sample efficiency, which focuses on selecting unique samples and adding them to the replay buffer during the exploration with the goal of reducing the buffer size and maintaining the independent and identically distributed (IID) nature of the samples. Our method is based on selecting an important subset of the set of state variables from the experiences encountered during the initial phase of random exploration, partitioning the state space into a set of abstract states based on the selected important state variables, and finally selecting the experiences with unique state-reward combination by using a kernel density estimator. We formally prove that the off-policy actor-critic algorithm incorporating the proposed method for unique experience
    
[^129]: EXGC: 在图压缩中平衡效率与可解释性

    EXGC: Bridging Efficiency and Explainability in Graph Condensation

    [https://arxiv.org/abs/2402.05962](https://arxiv.org/abs/2402.05962)

    本论文提出了EXGC方法，通过采用平均场变分近似和梯度信息瓶颈目标来提高图压缩的效率和可解释性。

    

    在海量数据集（如网络数据）上进行图表示学习已经取得了显著进展。然而，相关的计算和存储开销引起了人们的关注。为此，引入了图压缩（GCond）来将这些大型真实数据集蒸馏为更简洁但信息丰富的合成图。尽管进行了加速努力，现有的GCond方法主要在海量网络数据图上面临效率问题。因此，在这项工作中，我们指出了当前范例的两个主要不足之处：（1）大量参数集的并发更新，（2）明显的参数冗余。为了相应地克服这两个限制，我们首先采用平均场变分近似进行收敛加速，然后提出梯度信息瓶颈（GDIB）的目标来减少冗余。通过结合领先的解释技术（如GNNExplainer和GSAT）来实例化GDIB，我们的EXGC能够同时提高效率和可解释性。

    Graph representation learning on vast datasets, like web data, has made significant strides. However, the associated computational and storage overheads raise concerns. In sight of this, Graph condensation (GCond) has been introduced to distill these large real datasets into a more concise yet information-rich synthetic graph. Despite acceleration efforts, existing GCond methods mainly grapple with efficiency, especially on expansive web data graphs. Hence, in this work, we pinpoint two major inefficiencies of current paradigms: (1) the concurrent updating of a vast parameter set, and (2) pronounced parameter redundancy. To counteract these two limitations correspondingly, we first (1) employ the Mean-Field variational approximation for convergence acceleration, and then (2) propose the objective of Gradient Information Bottleneck (GDIB) to prune redundancy. By incorporating the leading explanation techniques (e.g., GNNExplainer and GSAT) to instantiate the GDIB, our EXGC, the Efficien
    
[^130]: 基因引导GFlowNets：在实际分子优化基准方面的进展

    Genetic-guided GFlowNets: Advancing in Practical Molecular Optimization Benchmark

    [https://arxiv.org/abs/2402.05961](https://arxiv.org/abs/2402.05961)

    本文提出了一种名为基因引导GFlowNet (Genetic GFN) 的新方法，通过集成迭代遗传搜索和训练策略，该方法在实际分子优化基准测试中取得了16.213的最新得分，明显优于现有最佳得分15.185，同时在14个任务中超越了所有对比方法。

    

    本文提出了一种新的GFlowNet变体，即基因引导GFlowNet (Genetic GFN)，它将迭代遗传搜索集成到GFlowNet中。遗传搜索有效地引导GFlowNet进入高回报区域，解决了全局过度探索导致的训练效率低下和探索有限区域的问题。此外，还引入了训练策略，如基于排名的重放训练和无监督最大似然预训练，以提高基因引导GFlowNet的样本效率。该方法在实际分子优化 (PMO) 领域的官方基准测试中显示了16.213的最新得分，明显优于基准测试中报告的最佳得分15.185。值得注意的是，我们的方法在23个任务中的14个任务中超过了所有对比方法，包括强化学习，贝叶斯优化，生成模型，GFlowNets和遗传算法。

    This paper proposes a novel variant of GFlowNet, genetic-guided GFlowNet (Genetic GFN), which integrates an iterative genetic search into GFlowNet. Genetic search effectively guides the GFlowNet to high-rewarded regions, addressing global over-exploration that results in training inefficiency and exploring limited regions. In addition, training strategies, such as rank-based replay training and unsupervised maximum likelihood pre-training, are further introduced to improve the sample efficiency of Genetic GFN. The proposed method shows a state-of-the-art score of 16.213, significantly outperforming the reported best score in the benchmark of 15.185, in practical molecular optimization (PMO), which is an official benchmark for sample-efficient molecular optimization. Remarkably, ours exceeds all baselines, including reinforcement learning, Bayesian optimization, generative models, GFlowNets, and genetic algorithms, in 14 out of 23 tasks.
    
[^131]: 基于相位驱动的非平稳时间序列通用学习

    Phase-driven Domain Generalizable Learning for Nonstationary Time Series

    [https://arxiv.org/abs/2402.05960](https://arxiv.org/abs/2402.05960)

    该论文提出了一个基于相位驱动的时间序列学习框架PhASER，通过相位增强、分离特征编码和特征广播的方法，实现了对非平稳数据的通用学习能力。

    

    监测和识别连续感知数据中的模式对许多实际应用至关重要。这些实际时间序列数据通常是非平稳的，其统计和谱特性随时间变化。这在开发能够有效泛化不同分布的学习模型方面提出了重大挑战。在本工作中，我们观察到非平稳统计与相位信息内在相关，提出了一个时间序列学习框架PhASER。它包括三个新颖的元素：1）相位增强，使非平稳性多样化同时保留有区别性的语义；2）将时变幅度和相位视为独立模态进行单独特征编码；3）利用新颖的残差连接将相位与特征结合，以强化分布不变性学习的固有正则化作用。通过在5个人体活动识别数据集上进行广泛评估，

    Monitoring and recognizing patterns in continuous sensing data is crucial for many practical applications. These real-world time-series data are often nonstationary, characterized by varying statistical and spectral properties over time. This poses a significant challenge in developing learning models that can effectively generalize across different distributions. In this work, based on our observation that nonstationary statistics are intrinsically linked to the phase information, we propose a time-series learning framework, PhASER. It consists of three novel elements: 1) phase augmentation that diversifies non-stationarity while preserving discriminatory semantics, 2) separate feature encoding by viewing time-varying magnitude and phase as independent modalities, and 3) feature broadcasting by incorporating phase with a novel residual connection for inherent regularization to enhance distribution invariant learning. Upon extensive evaluation on 5 datasets from human activity recognit
    
[^132]: 自然启发的局部传播

    Nature-Inspired Local Propagation

    [https://arxiv.org/abs/2402.05959](https://arxiv.org/abs/2402.05959)

    本文介绍了一种自然启发的局部传播算法，该算法通过在线处理环境信息而不依赖大量数据集，在机器学习领域具有潜力。这种算法的核心思想是结合数据表示和学习，以尊重时空局部性，并且当传播速度趋近于无穷大时，它等效于反向传播算法。

    

    机器学习中取得的令人瞩目的成果，包括最近在生成性人工智能方面的进展，都依赖于大量的数据集。相反，自然界中的智能过程并不需要这样的数据集，而只需通过对环境信息的在线处理即可产生。特别是，自然学习过程依赖于数据表示和学习相互交织以尊重时空局部性的机制。本文展示了这种特性来自于对学习的预算法视角，该视角受到了理论物理学相关研究的启发。我们展示了当传播速度趋于无穷大时，所得到的“学习法则”的算法解释（采用哈密顿方程结构）将归结为反向传播算法。这为基于全面在线信息处理的机器学习研究开辟了新的道路，其中反向传播算法被提出的时空局部算法取代。

    The spectacular results achieved in machine learning, including the recent advances in generative AI, rely on large data collections. On the opposite, intelligent processes in nature arises without the need for such collections, but simply by online processing of the environmental information. In particular, natural learning processes rely on mechanisms where data representation and learning are intertwined in such a way to respect spatiotemporal locality. This paper shows that such a feature arises from a pre-algorithmic view of learning that is inspired by related studies in Theoretical Physics. We show that the algorithmic interpretation of the derived "laws of learning", which takes the structure of Hamiltonian equations, reduces to Backpropagation when the speed of propagation goes to infinity. This opens the doors to machine learning studies based on full on-line information processing that are based the replacement of Backpropagation with the proposed spatiotemporal local algori
    
[^133]: 穿戴设备和单摄像头视频在不同深度学习体系结构下的上肢活动识别的比较研究

    A comparative study on wearables and single-camera video for upper-limb out-of-thelab activity recognition with different deep learning architectures

    [https://arxiv.org/abs/2402.05958](https://arxiv.org/abs/2402.05958)

    本研究比较了穿戴设备和单摄像头视频在不同深度学习体系结构下的上肢活动识别，为在实验室外跟踪患者活动提供了可行性，并探讨了在野外环境中识别和处理临床相关数据的机器学习系统的理想输入。

    

    在临床和研究环境中，广泛应用计算机视觉解决方案和最新的高端惯性测量单元(IMU)来评估人类的体力活动。然而，为了增加在实验室外跟踪患者活动的可行性，需要使用较少的设备进行运动获取。在这种情况下，基于IMU的穿戴设备和单摄像头系统是有前景的解决方案。此外，还需要开发能够在野外环境中识别和处理临床相关数据的机器学习系统，因此确定这些系统的理想输入至关重要。

    The use of a wide range of computer vision solutions, and more recently high-end Inertial Measurement Units (IMU) have become increasingly popular for assessing human physical activity in clinical and research settings. Nevertheless, to increase the feasibility of patient tracking in out-of-the-lab settings, it is necessary to use a reduced number of devices for movement acquisition. Promising solutions in this context are IMU-based wearables and single camera systems. Additionally, the development of machine learning systems able to recognize and digest clinically relevant data in-the-wild is needed, and therefore determining the ideal input to those is crucial.
    
[^134]: 在解空间中加速PDE数据生成的微分算子作用

    Accelerating PDE Data Generation via Differential Operator Action in Solution Space

    [https://arxiv.org/abs/2402.05957](https://arxiv.org/abs/2402.05957)

    通过在解空间中应用微分算子作用，我们提出了一种加速PDE数据生成的算法，名为DiffOAS。它能够在生成数据的同时提高数据的精度，并且在时间复杂度上比现有的方法更高效。

    

    最近数据驱动方法（如神经算子）在减少偏微分方程（PDEs）求解时间方面取得了显著进展。然而，这些方法面临的主要挑战之一是需要大量高精度的训练数据，在生成过程中需要显著的计算成本。为了解决这个挑战，我们提出了一种新颖的PDE数据集生成算法，即解空间中的微分算子作用（DiffOAS），它同时加快了数据生成过程并提高了生成数据的精度。具体而言，DiffOAS获取了几个基本的PDE解，并将它们组合以获得解。它对这些解应用微分算子，我们称之为“算子作用”，以高效生成精确的PDE数据点。理论分析表明，DiffOAS方法的时间复杂度比现有的生成方法低一个数量级。

    Recent advancements in data-driven approaches, such as Neural Operator (NO), have demonstrated their effectiveness in reducing the solving time of Partial Differential Equations (PDEs). However, one major challenge faced by these approaches is the requirement for a large amount of high-precision training data, which needs significant computational costs during the generation process. To address this challenge, we propose a novel PDE dataset generation algorithm, namely Differential Operator Action in Solution space (DiffOAS), which speeds up the data generation process and enhances the precision of the generated data simultaneously. Specifically, DiffOAS obtains a few basic PDE solutions and then combines them to get solutions. It applies differential operators on these solutions, a process we call 'operator action', to efficiently generate precise PDE data points. Theoretical analysis shows that the time complexity of DiffOAS method is one order lower than the existing generation meth
    
[^135]: Pathformer: 多尺度自适应路径的时间序列预测模型

    Pathformer: Multi-scale transformers with Adaptive Pathways for Time Series Forecasting

    [https://arxiv.org/abs/2402.05956](https://arxiv.org/abs/2402.05956)

    本文提出了一种名为Pathformer的多尺度自适应路径的Transformer模型，用于时间序列预测。通过整合时间分辨率和时间距离进行多尺度建模，并使用自适应路径来优化建模过程，可以提高预测准确性和泛化能力。

    

    基于Transformer的模型在时间序列预测中取得了一些成功。现有的方法主要从有限或固定尺度对时间序列进行建模，这使得捕捉跨多个尺度的不同特征变得具有挑战性。本文提出了一种多尺度自适应路径（Pathformer）的Transformer模型。该模型同时整合了时间分辨率和时间距离进行多尺度建模。多尺度划分运用不同大小的数据块将时间序列分割成不同的时间分辨率。基于每个尺度的划分，对这些数据块进行双重注意力机制，以捕捉全局相关性和局部细节作为时间依赖关系。我们进一步通过自适应路径来丰富多尺度Transformer，该路径可以根据输入时间序列中不断变化的时间动态调整多尺度建模过程，提高Pathformer的预测准确性和泛化能力。在11个真实数据集上进行了大量实验。

    Transformer-based models have achieved some success in time series forecasting. Existing methods mainly model time series from limited or fixed scales, making it challenging to capture different characteristics spanning various scales. In this paper, we propose multi-scale transformers with adaptive pathways (Pathformer). The proposed Transformer integrates both temporal resolution and temporal distance for multi-scale modeling. Multi-scale division divides the time series into different temporal resolutions using patches of various sizes. Based on the division of each scale, dual attention is performed over these patches to capture global correlations and local details as temporal dependencies. We further enrich the multi-scale transformer with adaptive pathways, which adaptively adjust the multi-scale modeling process based on the varying temporal dynamics in the input time series, improving the prediction accuracy and generalization of Pathformer. Extensive experiments on eleven rea
    
[^136]: 一个用于具有分离可行性约束的可控帕累托前沿学习的超级变压器模型

    A Hyper-Transformer model for Controllable Pareto Front Learning with Split Feasibility Constraints

    [https://arxiv.org/abs/2402.05955](https://arxiv.org/abs/2402.05955)

    本论文提出了一种用于具有分离可行性约束的可控帕累托前沿学习的超级变压器模型，可以通过近似和定位帕累托最优解来解决分裂多目标优化问题，并且在实践中限制了决策者目标的约束区域进行训练。

    

    可控帕累托前沿学习（CPFL）通过近似帕累托解集，然后在给定参考向量下定位帕累托最优解。然而，在实践中，决策者的目标受到约束区域的限制，因此我们只在约束区域进行训练，而不是整个决策空间。具有分离可行性约束（SFC）的可控帕累托前沿学习是一种寻找满足某些约束条件的分裂多目标优化问题的最佳帕累托解的方法。在之前的研究中，CPFL使用了由多层感知器（Hyper-MLP）模块组成的超网络模型。随着深度学习中变压器结构的显著进步，变压器在各种任务中可以超越其他结构。因此，我们为具有SFC的CPFL开发了一种超级变压器（Hyper-Trans）模型。我们使用序列到序列函数的通用逼近理论来展示Hyper-Trans模型的优势。

    Controllable Pareto front learning (CPFL) approximates the Pareto solution set and then locates a Pareto optimal solution with respect to a given reference vector. However, decision-maker objectives were limited to a constraint region in practice, so instead of training on the entire decision space, we only trained on the constraint region. Controllable Pareto front learning with Split Feasibility Constraints (SFC) is a way to find the best Pareto solutions to a split multi-objective optimization problem that meets certain constraints. In the previous study, CPFL used a Hypernetwork model comprising multi-layer perceptron (Hyper-MLP) blocks. With the substantial advancement of transformer architecture in deep learning, transformers can outperform other architectures in various tasks. Therefore, we have developed a hyper-transformer (Hyper-Trans) model for CPFL with SFC. We use the theory of universal approximation for the sequence-to-sequence function to show that the Hyper-Trans model
    
[^137]: EasyFS:一种通过特征的弹性变换实现高效的无模型特征选择框架

    EasyFS: an Efficient Model-free Feature Selection Framework via Elastic Transformation of Features

    [https://arxiv.org/abs/2402.05954](https://arxiv.org/abs/2402.05954)

    EasyFS是一种高效的无模型特征选择框架，通过对特征进行弹性扩展和压缩，实现了对特征之间相互关系的建模，并发现最相关的特征。同时，通过新的冗余度度量方法实现了对冗余特征的高效过滤。

    

    传统的无模型特征选择方法将每个特征独立处理，忽视了特征之间的相互关系，这导致其性能相对较差，与模型感知方法相比。为了解决这个挑战，我们提出了一种通过对特征进行弹性扩展和压缩的高效无模型特征选择框架——EasyFS，以实现比最先进的模型感知方法更好的性能，同时具备现有无模型方法的效率和灵活性。具体而言，EasyFS采用随机非线性投影网络扩展特征空间，实现原始特征的非线性组合，以建模特征之间的相互关系并发现最相关的特征。同时，提出了一种基于编码率变化的新型冗余度度量方法，用于高效过滤冗余特征。在21个不同数据集上进行了全面的实验，

    Traditional model-free feature selection methods treat each feature independently while disregarding the interrelationships among features, which leads to relatively poor performance compared with the model-aware methods. To address this challenge, we propose an efficient model-free feature selection framework via elastic expansion and compression of the features, namely EasyFS, to achieve better performance than state-of-the-art model-aware methods while sharing the characters of efficiency and flexibility with the existing model-free methods. In particular, EasyFS expands the feature space by using the random non-linear projection network to achieve the non-linear combinations of the original features, so as to model the interrelationships among the features and discover most correlated features. Meanwhile, a novel redundancy measurement based on the change of coding rate is proposed for efficient filtering of redundant features. Comprehensive experiments on 21 different datasets sho
    
[^138]: idMotif：蛋白质序列中的互动模体识别

    idMotif: An Interactive Motif Identification in Protein Sequences

    [https://arxiv.org/abs/2402.05953](https://arxiv.org/abs/2402.05953)

    idMotif是一个可视化分析框架，旨在帮助领域专家识别蛋白质序列中的模体。它利用深度学习方法对蛋白质序列进行分类，并通过局部解释深度学习模型的决策，发现潜在的模体候选序列。它提供多个交互式视图，用于分析蛋白质聚类和序列。一项案例研究证明了idMotif在蛋白质序列和模体分析与识别中的实用性。

    

    本文介绍了idMotif，一个旨在帮助领域专家识别蛋白质序列中模体的可视化分析框架。模体是由氨基酸组成的短序列，对于理解蛋白质的不同功能至关重要。识别这些模体对于预测疾病或感染至关重要。idMotif采用了基于深度学习的方法对蛋白质序列进行分类，通过深度学习模型决策的局部解释，可以发现蛋白质组中潜在的模体候选序列。它提供多个交互式视图，用于分析蛋白质聚类或组及其序列。一项案例研究结合专家反馈，说明了idMotif在促进蛋白质序列和模体分析和识别方面的实用性。

    This article introduces idMotif, a visual analytics framework designed to aid domain experts in the identification of motifs within protein sequences. Motifs, short sequences of amino acids, are critical for understanding the distinct functions of proteins. Identifying these motifs is pivotal for predicting diseases or infections. idMotif employs a deep learning-based method for the categorization of protein sequences, enabling the discovery of potential motif candidates within protein groups through local explanations of deep learning model decisions. It offers multiple interactive views for the analysis of protein clusters or groups and their sequences. A case study, complemented by expert feedback, illustrates idMotif's utility in facilitating the analysis and identification of protein sequences and motifs.
    
[^139]: 利用大型语言模型推进图表示学习：技术全面调查

    Advancing Graph Representation Learning with Large Language Models: A Comprehensive Survey of Techniques

    [https://arxiv.org/abs/2402.05952](https://arxiv.org/abs/2402.05952)

    本综述调查了将大型语言模型（LLM）与图表示学习（GRL）相结合的技术，并提供了一个新颖的分类法，深入分析了这些模型的核心组成部分和操作技术，为有效的模型设计和训练策略提供了新的视角。

    

    大型语言模型（LLM）与图表示学习（GRL）的整合标志着分析复杂数据结构的重大进展。这种合作利用LLM的先进语言能力来改进图模型的上下文理解能力和适应性，从而拓宽了GRL的范围和潜力。尽管已经有大量的研究致力于将LLM集成到图领域中，但缺乏一份深入分析这些模型核心组成部分和操作技术的全面综述。我们的调查通过提出一种新颖的分类法来分解这些模型为主要组成部分和操作技术，从新的技术角度深入分析。我们进一步将最近的文献分解为两个主要组成部分，包括知识提取器和组织者，以及两个操作技术，包括集成和训练策略，揭示出有效的模型设计和训练策略的要点。

    The integration of Large Language Models (LLMs) with Graph Representation Learning (GRL) marks a significant evolution in analyzing complex data structures. This collaboration harnesses the sophisticated linguistic capabilities of LLMs to improve the contextual understanding and adaptability of graph models, thereby broadening the scope and potential of GRL. Despite a growing body of research dedicated to integrating LLMs into the graph domain, a comprehensive review that deeply analyzes the core components and operations within these models is notably lacking. Our survey fills this gap by proposing a novel taxonomy that breaks down these models into primary components and operation techniques from a novel technical perspective. We further dissect recent literature into two primary components including knowledge extractors and organizers, and two operation techniques including integration and training stratigies, shedding light on effective model design and training strategies. Additio
    
[^140]: \textit{MinMaxMin} $Q$-learning

    \textit{MinMaxMin} $Q$-learning

    [https://arxiv.org/abs/2402.05951](https://arxiv.org/abs/2402.05951)

    \textit{MinMaxMin} $Q$-learning是一种乐观型Actor-Critic算法，通过解决过高估计偏差的问题，在各种基准任务中相对于现有算法表现出稳定的性能提升。

    

    \textit{MinMaxMin} $Q$-learning是一种新颖的乐观型Actor-Critic算法，解决了保守型强化学习算法中存在的过高估计偏差的问题（$Q$-估计过高估计了真实的$Q$值）。其核心公式依赖于$Q$-网络之间的差异，采用最小批次最大最小$Q$-网络距离作为$Q$-目标加入，并作为优先级经验回放采样规则。我们在TD3和TD7之上实施了\textit{MinMaxMin}，并对其在流行的MuJoCo和Bullet环境中对抗现有的连续空间算法-DDPG，TD3和TD7进行了严格测试。结果显示，在所有测试任务中，\textit{MinMaxMin}相对于DDPG，TD3和TD7均表现出了稳定的性能提升。

    \textit{MinMaxMin} $Q$-learning is a novel \textit{optimistic} Actor-Critic algorithm that addresses the problem of \textit{overestimation} bias ($Q$-estimations are overestimating the real $Q$-values) inherent in \textit{conservative} RL algorithms. Its core formula relies on the disagreement among $Q$-networks in the form of the min-batch MaxMin $Q$-networks distance which is added to the $Q$-target and used as the priority experience replay sampling-rule. We implement \textit{MinMaxMin} on top of TD3 and TD7, subjecting it to rigorous testing against state-of-the-art continuous-space algorithms-DDPG, TD3, and TD7-across popular MuJoCo and Bullet environments. The results show a consistent performance improvement of \textit{MinMaxMin} over DDPG, TD3, and TD7 across all tested tasks.
    
[^141]: SQT - std Q-target

    \textit{SQT} -- \textit{std} $Q$-target

    [https://arxiv.org/abs/2402.05950](https://arxiv.org/abs/2402.05950)

    SQT是一种基于Q-学习的保守型actor-critic算法，利用Q网络的标准差作为一种“不确定性惩罚”，成功解决了过高估计偏差问题，相较于TD3的Q-target公式具有更好的性能优势。

    

    Std Q-target是一种基于Q-学习的保守型actor-critic算法，它基于一个关键的Q公式：Q网络的标准差，这个标准差作为一种“不确定性惩罚”，是对过高估计偏差问题的一种简约解决方案。我们在TD3/TD7代码的基础上实现了SQT，并将其与最先进的actor-critic算法DDPG、TD3和TD7在七个常见的MuJoCo和Bullet任务上进行了测试。我们的结果表明，在强化学习中，SQT的Q-target公式相对于TD3的Q-target公式在解决过高估计偏差的保守解方面具有优势，而在所有任务中，SQT相对于DDPG、TD3和TD7都有明显的性能优势。

    \textit{Std} $Q$-target is a \textit{conservative}, actor-critic, ensemble, $Q$-learning-based algorithm, which is based on a single key $Q$-formula: $Q$-networks standard deviation, which is an "uncertainty penalty", and, serves as a minimalistic solution to the problem of \textit{overestimation} bias. We implement \textit{SQT} on top of TD3/TD7 code and test it against the state-of-the-art (SOTA) actor-critic algorithms, DDPG, TD3 and TD7 on seven popular MuJoCo and Bullet tasks. Our results demonstrate \textit{SQT}'s $Q$-target formula superiority over \textit{TD3}'s $Q$-target formula as a \textit{conservative} solution to overestimation bias in RL, while \textit{SQT} shows a clear performance advantage on a wide margin over DDPG, TD3, and TD7 on all tasks.
    
[^142]: 一种可解释的基于机器学习的方法用于分析客户的在线数据以确定产品特性的重要性

    An explainable machine learning-based approach for analyzing customers' online data to identify the importance of product attributes

    [https://arxiv.org/abs/2402.05949](https://arxiv.org/abs/2402.05949)

    本研究提出了一种基于机器学习和博弈论的方法，通过分析在线客户数据，提取产品开发的全面设计启示，并评估每个特性对总体满意度的重要性。

    

    在线客户数据为产品设计和市场研究提供了宝贵的信息，因为它可以揭示客户的喜好。然而，使用人工智能（AI）对这些数据进行数据驱动设计的分析是一项具有挑战性的任务，因为可能存在隐藏的模式。此外，在这些研究领域中，大多数研究仅限于发现客户的需求。在本研究中，我们提出了一种基于博弈论机器学习（ML）方法，可以从在线评级的基础上选择、排序和组合最大化客户满意度的产品特性，从而提取全面的设计启示，用于产品开发的指导。然后，我们使用SHAP（SHapley Additive exPlanations）博弈论方法，根据其对预测的贡献为每个特征分配一个值，为评估每个特征对总体满意度的重要性提供指导。我们将该方法应用于来自Kaggle的笔记本电脑的真实数据集中。

    Online customer data provides valuable information for product design and marketing research, as it can reveal the preferences of customers. However, analyzing these data using artificial intelligence (AI) for data-driven design is a challenging task due to potential concealed patterns. Moreover, in these research areas, most studies are only limited to finding customers' needs. In this study, we propose a game theory machine learning (ML) method that extracts comprehensive design implications for product development. The method first uses a genetic algorithm to select, rank, and combine product features that can maximize customer satisfaction based on online ratings. Then, we use SHAP (SHapley Additive exPlanations), a game theory method that assigns a value to each feature based on its contribution to the prediction, to provide a guideline for assessing the importance of each feature for the total satisfaction. We apply our method to a real-world dataset of laptops from Kaggle, and d
    
[^143]: DE$^3$-BERT: 基于原型网络的增强距离早期停止方法，用于BERT

    DE$^3$-BERT: Distance-Enhanced Early Exiting for BERT based on Prototypical Networks

    [https://arxiv.org/abs/2402.05948](https://arxiv.org/abs/2402.05948)

    DE$^3$-BERT是一种基于原型网络和距离度量的增强距离早期停止框架，用于提高BERT等预训练语言模型的推断速度和准确性。

    

    早期停止方法通过动态调整执行的层数，提高了像BERT这样的预训练语言模型的推断速度。然而，大多数早期停止方法仅考虑了来自单个测试样本的局部信息来确定早期停止的指标，而未利用样本群体提供的全局信息。这导致对预测正确性的估计不够准确，从而产生错误的早期停止决策。为了弥合这个差距，我们探索了有效结合局部和全局信息以确保可靠的早期停止的必要性。为此，我们利用原型网络学习类别原型，并设计了样本和类别原型之间的距离度量。这使我们能够利用全局信息来估计早期预测的正确性。基于此，我们提出了一种新颖的DE$^3$-BERT增强距离早期停止框架。

    Early exiting has demonstrated its effectiveness in accelerating the inference of pre-trained language models like BERT by dynamically adjusting the number of layers executed. However, most existing early exiting methods only consider local information from an individual test sample to determine their exiting indicators, failing to leverage the global information offered by sample population. This leads to suboptimal estimation of prediction correctness, resulting in erroneous exiting decisions. To bridge the gap, we explore the necessity of effectively combining both local and global information to ensure reliable early exiting during inference. Purposefully, we leverage prototypical networks to learn class prototypes and devise a distance metric between samples and class prototypes. This enables us to utilize global information for estimating the correctness of early predictions. On this basis, we propose a novel Distance-Enhanced Early Exiting framework for BERT (DE$^3$-BERT). DE$^3
    
[^144]: 可分离的多概念抹除与扩散模型

    Separable Multi-Concept Erasure from Diffusion Models

    [https://arxiv.org/abs/2402.05947](https://arxiv.org/abs/2402.05947)

    提出了可分离的多概念抹除器（SepME），通过生成概念无关表示和权重解耦来解决扩散模型中的多概念抹除问题，并在不影响生成性能的情况下恢复概念。

    

    大规模扩散模型以其令人印象深刻的图像生成能力而闻名，这引发了研究人员对其社会影响的担忧，例如对版权艺术风格的模仿。为了应对这些问题，现有方法采用机器遗忘技术从预训练模型中消除不安全的概念。然而，这些方法会损害生成性能，忽视多概念消除之间的耦合，以及概念恢复问题。为了解决这些问题，我们提出了一种可分离的多概念抹除器（SepME），主要包括两部分：概念无关表示的生成和权重解耦。前者旨在避免遗忘与遗忘概念无关的重要信息。后者分离可优化的模型权重，使每个权重增量对应于特定概念的消除，而不影响对其他概念的生成性能。

    Large-scale diffusion models, known for their impressive image generation capabilities, have raised concerns among researchers regarding social impacts, such as the imitation of copyrighted artistic styles. In response, existing approaches turn to machine unlearning techniques to eliminate unsafe concepts from pre-trained models. However, these methods compromise the generative performance and neglect the coupling among multi-concept erasures, as well as the concept restoration problem. To address these issues, we propose a Separable Multi-concept Eraser (SepME), which mainly includes two parts: the generation of concept-irrelevant representations and the weight decoupling. The former aims to avoid unlearning substantial information that is irrelevant to forgotten concepts. The latter separates optimizable model weights, making each weight increment correspond to a specific concept erasure without affecting generative performance on other concepts. Specifically, the weight increment fo
    
[^145]: 揭示潜在因果规律：一种基于时间点过程的异常事件解释方法

    Unveiling Latent Causal Rules: A Temporal Point Process Approach for Abnormal Event Explanation

    [https://arxiv.org/abs/2402.05946](https://arxiv.org/abs/2402.05946)

    本文提出了一种基于时间点过程的方法，通过揭示潜在因果规律来解释异常事件，以帮助在高风险系统如医疗保健中快速诊断和精确治疗规划。该方法通过期望最大化算法优化规则集和模型参数，实现了准确的规则发现和根因识别。

    

    在高风险系统如医疗保健中，理解异常事件背后的因果原因是至关重要的，例如患者健康状况的突然变化。揭示因果原因有助于快速诊断和精确治疗规划。在本文中，我们提出了一种自动化方法来揭示解释观察事件的“如果-那么”逻辑规则。我们引入了时间点过程来建模所关注事件，并发现一组潜在规则来解释事件的发生。为了实现这一点，我们采用了期望最大化（EM）算法。在E步中，我们计算每个事件被每个发现的规则解释的可能性。在M步中，我们更新规则集和模型参数，以增强可能性函数的下界。值得注意的是，我们以微分的方式优化规则集。我们的方法在发现规则和识别根本原因方面表现出准确的性能。我们使用合成数据展示了它的有希望的结果。

    In high-stakes systems such as healthcare, it is critical to understand the causal reasons behind unusual events, such as sudden changes in patient's health. Unveiling the causal reasons helps with quick diagnoses and precise treatment planning. In this paper, we propose an automated method for uncovering "if-then" logic rules to explain observational events. We introduce temporal point processes to model the events of interest, and discover the set of latent rules to explain the occurrence of events. To achieve this, we employ an Expectation-Maximization (EM) algorithm. In the E-step, we calculate the likelihood of each event being explained by each discovered rule. In the M-step, we update both the rule set and model parameters to enhance the likelihood function's lower bound. Notably, we optimize the rule set in a differential manner. Our approach demonstrates accurate performance in both discovering rules and identifying root causes. We showcase its promising results using syntheti
    
[^146]: 通过有监督的、分层概念学习消除硬概念瓶颈模型中的信息泄漏问题

    Eliminating Information Leakage in Hard Concept Bottleneck Models with Supervised, Hierarchical Concept Learning

    [https://arxiv.org/abs/2402.05945](https://arxiv.org/abs/2402.05945)

    本文解决了概念瓶颈模型中的信息泄漏问题，通过引入标签监督和构建分层概念集，提出了一种新的CBMs范例（SupCBM），它可以通过预测的概念和干预矩阵实现标签预测，并且只在不同的类别之间进行区分。

    

    概念瓶颈模型（CBMs）旨在通过将特征和标签与人类可理解的概念联系起来，提供可解释和可干预的预测。尽管最近的CBMs显示出了巨大的潜力，但它们存在信息泄漏问题，即在概念表示为概率或二进制状态时，超出概念的意图信息泄漏到后续的标签预测中。因此，通过无法区分的概念来错误分类不同的类别，削弱了CBMs的解释和干预能力。本文通过在概念预测中引入标签监督和构建分层概念集来缓解信息泄漏问题。因此，我们提出了一种新的CBMs范例，即SupCBM，它通过预测的概念和精心设计的干预矩阵实现标签预测。SupCBM将重点放在与预测标签最相关的概念上，并且仅在不同的类别之间进行区分。

    Concept Bottleneck Models (CBMs) aim to deliver interpretable and interventionable predictions by bridging features and labels with human-understandable concepts. While recent CBMs show promising potential, they suffer from information leakage, where unintended information beyond the concepts (either when concepts are represented with probabilities or binary states) are leaked to the subsequent label prediction. Consequently, distinct classes are falsely classified via indistinguishable concepts, undermining the interpretation and intervention of CBMs.   This paper alleviates the information leakage issue by introducing label supervision in concept predication and constructing a hierarchical concept set. Accordingly, we propose a new paradigm of CBMs, namely SupCBM, which achieves label predication via predicted concepts and a deliberately-designed intervention matrix. SupCBM focuses on concepts that are mostly relevant to the predicted label and only distinguishes classes when differe
    
[^147]: Todyformer：面向结构感知标记化的整体动态图形变压器

    Todyformer: Towards Holistic Dynamic Graph Transformers with Structure-Aware Tokenization

    [https://arxiv.org/abs/2402.05944](https://arxiv.org/abs/2402.05944)

    Todyformer是一个全新的基于变压器的神经网络，它通过结合局部编码和全局编码能力，采用新颖的标记化策略和时间位置编码来解决动态图形中的过度压缩和长程依赖性问题。

    

    时间关联的图神经网络因其能够模拟演化结构和时间模式并展示出良好性能而受到了广泛关注。然而，已知这些架构受到了一些问题的限制，如过度压缩和过度平滑。与此同时，变压器已经展示了出色的计算能力，能够有效解决与长程依赖性相关的挑战。因此，我们引入了一种新颖的基于变压器的神经网络Todyformer，专为动态图形设计。它通过以下方式将消息传递神经网络（MPNNs）的局部编码能力与变压器的全局编码能力统一起来：i）采用新颖的面向动态图形的块状化范式来改善过度压缩，ii）利用MPNNs的结构感知参数化标记化策略，iii）引入带有时间位置编码的变压器来捕捉长程依赖性，以及iv）交替的编码架构。

    Temporal Graph Neural Networks have garnered substantial attention for their capacity to model evolving structural and temporal patterns while exhibiting impressive performance. However, it is known that these architectures are encumbered by issues that constrain their performance, such as over-squashing and over-smoothing. Meanwhile, Transformers have demonstrated exceptional computational capacity to effectively address challenges related to long-range dependencies. Consequently, we introduce Todyformer-a novel Transformer-based neural network tailored for dynamic graphs. It unifies the local encoding capacity of Message-Passing Neural Networks (MPNNs) with the global encoding of Transformers through i) a novel patchifying paradigm for dynamic graphs to improve over-squashing, ii) a structure-aware parametric tokenization strategy leveraging MPNNs, iii) a Transformer with temporal positional-encoding to capture long-range dependencies, and iv) an encoding architecture that alternates
    
[^148]: 一种用于软件定义网络实时异常检测的混合 IndRNNLSTM 方法

    A hybrid IndRNNLSTM approach for real-time anomaly detection in software-defined networks

    [https://arxiv.org/abs/2402.05943](https://arxiv.org/abs/2402.05943)

    本文提出了一种混合 IndRNNLSTM 方法，用于实时检测软件定义网络中的异常。该方法通过结合 IndRNN 和 LSTM 的特点，学习相关和非相关特征，并使用四种特征选择模型提供适当的特征视角。在 NSL-KDD 数据集上实验结果显示，该方法达到了较低的 MAE 和 RMSE 值。

    

    在软件定义网络中使用数据流预测进行异常检测是一项困难的任务。这个问题归类为时序和回归问题。机器学习方法在这个领域中具有挑战性，因为需要手动选择特征。而深度学习方法由于能够自动选择特征具有重要的特点。与此同时，基于 RNN 的方法被广泛使用。LSTM 和 GRU 方法能够很好地学习相关实体；而 IndRNN 方法则能够学习时序中的非相关实体。本文提出的方法尝试使用 IndRNN 和 LSTM 的组合来学习相关和非相关特征。特征选择方法还为模型提供了适当的特征视角；为了实现这一目的，使用了四种特征选择模型：Filter、Wrapper、Embedded 和 Autoencoder。提出的 IndRNNLSTM 算法与 Embedded 的组合能够在 NSL-KDD 数据集上实现 MAE=1.22 和 RMSE=9.92。

    Anomaly detection in SDN using data flow prediction is a difficult task. This problem is included in the category of time series and regression problems. Machine learning approaches are challenging in this field due to the manual selection of features. On the other hand, deep learning approaches have important features due to the automatic selection of features. Meanwhile, RNN-based approaches have been used the most. The LSTM and GRU approaches learn dependent entities well; on the other hand, the IndRNN approach learns non-dependent entities in time series. The proposed approach tried to use a combination of IndRNN and LSTM approaches to learn dependent and non-dependent features. Feature selection approaches also provide a suitable view of features for the models; for this purpose, four feature selection models, Filter, Wrapper, Embedded, and Autoencoder were used. The proposed IndRNNLSTM algorithm, in combination with Embedded, was able to achieve MAE=1.22 and RMSE=9.92 on NSL-KDD 
    
[^149]: 合作知识蒸馏：一种学习者无关的方法

    Cooperative Knowledge Distillation: A Learner Agnostic Approach

    [https://arxiv.org/abs/2402.05942](https://arxiv.org/abs/2402.05942)

    合作知识蒸馏是一种通过多个模型相互合作来传递知识的方法，可以弥补传统知识蒸馏的局限性。不同模型的优劣势可以更有效地传递知识。

    

    知识蒸馏是一种简单而强大的将教师模型的知识传递给学生模型的方法。现有的研究存在以下至少一种关键限制，限制其使用范围和方向：无论该知识是否有用，所有知识都从教师传递给学生；学生是这种交流中唯一学习的一方；典型的蒸馏只从一个教师向一个学生传递知识。我们提出了一种新形式的知识蒸馏，即合作蒸馏，其中许多模型可以同时充当学生和教师的角色。模型之间的合作方式如下：一个模型（学生）识别其性能中的特定缺陷，并搜索另一个模型（教师），通过生成对应事实情况的虚拟实例来编码所学知识。由于不同模型可能具有不同的优势和劣势，因此合作蒸馏的方法可以更有效地传递知识。

    Knowledge distillation is a simple but powerful way to transfer knowledge between a teacher model to a student model. Existing work suffers from at least one of the following key limitations in terms of direction and scope of transfer which restrict its use: all knowledge is transferred from teacher to student regardless of whether or not that knowledge is useful, the student is the only one learning in this exchange, and typically distillation transfers knowledge only from a single teacher to a single student. We formulate a novel form of knowledge distillation in which many models can act as both students and teachers which we call cooperative distillation. The models cooperate as follows: a model (the student) identifies specific deficiencies in it's performance and searches for another model (the teacher) who encodes learned knowledge into instructional virtual instances via counterfactual instance generation. Because different models may have different strengths and weaknesses, al
    
[^150]: 基于人物的服装生成与通过LLMs进行视觉增强的风格提取

    Character-based Outfit Generation with Vision-augmented Style Extraction via LLMs

    [https://arxiv.org/abs/2402.05941](https://arxiv.org/abs/2402.05941)

    本文提出了一个新的基于人物的服装生成（COG）问题，旨在准确解释人物信息并根据用户的规范生成服装组合。我们提出了一个新颖的框架LVA-COG，利用大型语言模型（LLMs）从用户的兴趣中提取见解，并结合文本到图像模型，增强了对连贯服装的视觉理解和生成。

    

    服装生成问题涉及根据用户的兴趣推荐一个完整的服装。现有方法主要基于锚定商品或指定查询风格来推荐物品，但不考虑用户对电影、社交媒体等中著名人物的兴趣。本文定义了一个新的基于人物的服装生成（COG）问题，旨在准确解释人物信息，并根据用户的规范（如年龄和性别）生成完整的服装组合。为了解决这个问题，我们提出了一个新颖的框架LVA-COG，利用大型语言模型（LLMs）从用户的兴趣（例如人物信息）中提取见解，并采用提示工程技术准确理解用户的喜好。此外，我们还结合了文本到图像模型，增强了对连贯服装的视觉理解和生成（事实或反事实）。我们的框架将LLMs与文本到图像模型整合起来。

    The outfit generation problem involves recommending a complete outfit to a user based on their interests. Existing approaches focus on recommending items based on anchor items or specific query styles but do not consider customer interests in famous characters from movie, social media, etc. In this paper, we define a new Character-based Outfit Generation (COG) problem, designed to accurately interpret character information and generate complete outfit sets according to customer specifications such as age and gender. To tackle this problem, we propose a novel framework LVA-COG that leverages Large Language Models (LLMs) to extract insights from customer interests (e.g., character information) and employ prompt engineering techniques for accurate understanding of customer preferences. Additionally, we incorporate text-to-image models to enhance the visual understanding and generation (factual or counterfactual) of cohesive outfits. Our framework integrates LLMs with text-to-image models 
    
[^151]: 井下煤矿工作时间损失的风险因素因果关系网络

    Causal Relationship Network of Risk Factors Impacting Workday Loss in Underground Coal Mines

    [https://arxiv.org/abs/2402.05940](https://arxiv.org/abs/2402.05940)

    本研究使用一种新颖的因果人工智能方法，通过分析井下煤矿的伤害记录数据，建立了井下煤矿工作时间损失的因果关系网络。发现关键的因果关系包括风源和工作状态等不同因素之间的作用。

    

    本研究旨在利用一种新颖的因果人工智能（AI）方法建立井下煤矿工作时间损失的各种因素之间的因果关系网络。分析利用了从国家职业安全与健康研究所（NIOSH）获得的数据。从NIOSH数据库中提取了来自1990年至2020年的共计101,010份伤害记录，涵盖了3,982个独立的井下煤矿。利用一种名为群组贪婪等价搜索（GGES）的新颖因果AI方法进行了因果关系的分析和可视化。通过干预计算调整（IDA）得分对每个变量对工作时间损失的影响进行了评估。使用10折交叉验证技术进行模型训练和验证。利用接邻点精确度（AP）、接邻点召回率（AR）、箭头头部精确度（AHP）和箭头头部召回率（AHR）等性能指标对模型进行评估。研究发现，在2006年之后，关键的因果关系包括风源和工作状态等不同因素之间的作用有所     changed

    This study aims to establish the causal relationship network between various factors leading to workday loss in underground coal mines using a novel causal artificial intelligence (AI) method. The analysis utilizes data obtained from the National Institute for Occupational Safety and Health (NIOSH). A total of 101,010 injury records from 3,982 unique underground coal mines spanning the years from 1990 to 2020 were extracted from the NIOSH database. Causal relationships were analyzed and visualized using a novel causal AI method called Grouped Greedy Equivalence Search (GGES). The impact of each variable on workday loss was assessed through intervention do-calculus adjustment (IDA) scores. Model training and validation were performed using the 10-fold cross-validation technique. Performance metrics, including adjacency precision (AP), adjacency recall (AR), arrowhead precision (AHP), and arrowhead recall (AHR), were utilized to evaluate the models. Findings revealed that after 2006, key
    
[^152]: 大型语言模型在代码分布转移下的不确定性意识：基准研究

    Uncertainty Awareness of Large Language Models Under Code Distribution Shifts: A Benchmark Study

    [https://arxiv.org/abs/2402.05939](https://arxiv.org/abs/2402.05939)

    本文研究了大型语言模型在代码分布转移下的不确定性意识，并通过引入大规模基准数据集和应用概率方法来提高语言模型的可靠性。

    

    大型语言模型（LLMs）被广泛应用于编程语言分析，以提高人类生产力。然而，它们的可靠性可能会受到各种代码分布转移的影响，导致输出不一致。尽管众所周知，概率方法通过不确定性校准和估计可以减轻此类影响，但与其在基于图像的任务中的应用相比，它们在语言领域的效果尚未得到充分探索。在这项工作中，我们首先引入了一个大规模的基准数据集，其中包含三种代码分布转移的实际模式，强度各异。然后，我们对CodeLlama应用最先进的概率方法进行了全面调查。我们观察到，这些方法通常可以提高CodeLlama对不确定性的意识，提高了校准质量和更高的不确定性估计精度。然而，我们的研究进一步揭示了不同标准下的性能动态。

    Large Language Models (LLMs) have been widely employed in programming language analysis to enhance human productivity. Yet, their reliability can be compromised by various code distribution shifts, leading to inconsistent outputs. While probabilistic methods are known to mitigate such impact through uncertainty calibration and estimation, their efficacy in the language domain remains underexplored compared to their application in image-based tasks. In this work, we first introduce a large-scale benchmark dataset, incorporating three realistic patterns of code distribution shifts at varying intensities. Then we thoroughly investigate state-of-the-art probabilistic methods applied to CodeLlama using these shifted code snippets. We observe that these methods generally improve the uncertainty awareness of CodeLlama, with increased calibration quality and higher uncertainty estimation~(UE) precision. However, our study further reveals varied performance dynamics across different criteria (e
    
[^153]: 大规模语言模型在知识蒸馏中遇见图神经网络

    Large Language Model Meets Graph Neural Network in Knowledge Distillation

    [https://arxiv.org/abs/2402.05894](https://arxiv.org/abs/2402.05894)

    本论文提出了一种新颖的图知识蒸馏框架，使用大规模语言模型作为教师模型、图神经网络作为学生模型，解决了在理解文本-属性图中的节点分类问题中的限制。

    

    尽管近期学术界对于大规模语言模型（LLMs）在理解文本-属性图（TAG）方面的进展和潜力有所披露，但LLMs在实际应用中的部署受到了计算和存储需求高，推理过程中延迟长的限制。同时，传统的图神经网络（GNNs）虽然轻量且擅长学习图的结构特征，但对于真实应用中TAG复杂语义的把握有所限制。为了解决这些限制，我们聚焦于TAG中节点分类的下游任务，提出了一种新颖的图知识蒸馏框架，称为语言图知识蒸馏（LinguGKD），使用LLMs作为教师模型，GNNs作为学生模型进行知识蒸馏。其中包括对LLM进行TAG定向指导调整以应对设计的节点分类提示，然后对层次化学习的节点特征进行对齐。

    Despite recent community revelations about the advancements and potential of Large Language Models (LLMs) in understanding Text-Attributed Graphs (TAG), the deployment of LLMs for production is hindered by their high computational and storage requirements, as well as long latencies during inference. Simultaneously, although traditional Graph Neural Networks (GNNs) are light weight and adept at learning structural features of graphs, their ability to grasp the complex semantics in TAGs is somewhat constrained for real applications. To address these limitations, we concentrate on the downstream task of node classification in TAG and propose a novel graph knowledge distillation framework, termed Linguistic Graph Knowledge Distillation (LinguGKD), using LLMs as teacher models and GNNs as student models for knowledge distillation. It involves TAG-oriented instruction tuning of LLM on designed node classification prompts, followed by aligning the hierarchically learned node features of the t
    
[^154]: 知识图谱与多模态学习：综述

    Knowledge Graphs Meet Multi-Modal Learning: A Comprehensive Survey

    [https://arxiv.org/abs/2402.05391](https://arxiv.org/abs/2402.05391)

    知识图谱与多模态学习的综述介绍了KG4MM和MM4KG两个主要方面，包括任务定义、构建进展、评估基准以及关键研究轨迹。

    

    知识图谱在推动各种人工智能应用方面起着关键作用，语义网络社区对多模态维度的探索为创新打开了新的途径。在本综述中，我们仔细审查了300多篇文章，重点关注了两个主要方面的知识图谱感知研究：以知识图谱支持多模态任务的KG驱动多模态（KG4MM）学习，将知识图谱研究扩展到多模态知识图谱（MM4KG）领域。我们从定义知识图谱和多模态知识图谱开始，然后探索它们的构建进展。我们的综述包括两个主要任务类别：KG感知的多模态学习任务，如图像分类和视觉问答，以及内在的多模态知识图谱任务，如多模态知识图谱补全和实体对齐，突出了具体的研究轨迹。对于这些任务中的大部分，我们提供了定义、评估基准，并进一步指出进行相关研究的重要见解。最后，我们讨论了cu

    Knowledge Graphs (KGs) play a pivotal role in advancing various AI applications, with the semantic web community's exploration into multi-modal dimensions unlocking new avenues for innovation. In this survey, we carefully review over 300 articles, focusing on KG-aware research in two principal aspects: KG-driven Multi-Modal (KG4MM) learning, where KGs support multi-modal tasks, and Multi-Modal Knowledge Graph (MM4KG), which extends KG studies into the MMKG realm. We begin by defining KGs and MMKGs, then explore their construction progress. Our review includes two primary task categories: KG-aware multi-modal learning tasks, such as Image Classification and Visual Question Answering, and intrinsic MMKG tasks like Multi-modal Knowledge Graph Completion and Entity Alignment, highlighting specific research trajectories. For most of these tasks, we provide definitions, evaluation benchmarks, and additionally outline essential insights for conducting relevant research. Finally, we discuss cu
    
[^155]: 贝尔曼符合推断：时间序列预测中预测区间的校准

    Bellman Conformal Inference: Calibrating Prediction Intervals For Time Series

    [https://arxiv.org/abs/2402.05203](https://arxiv.org/abs/2402.05203)

    贝尔曼符合推断（BCI）是一个框架，通过解决一维随机控制问题，利用多步预测来提供校准的时间序列预测区间。BCI在任意分布转换和时间依赖性下实现了长期覆盖，且在波动率预测问题上生成更短的预测区间。

    

    我们引入了贝尔曼符合推断（BCI），这是一个围绕任何时间序列预测模型的框架，可以提供校准的预测区间。与现有方法不同，BCI能够利用多步预测，并通过在每个时间步骤上解决一维随机控制问题（SCP）来显式优化平均区间长度。特别地，我们使用动态规划算法来找到SCP的最优策略。我们证明了在任意分布转换和时间依赖性下，BCI能够实现长期覆盖，即使多步预测较差。我们在实证中发现，与现有方法相比，BCI避免了无信息区间（长度无限）的生成，并在波动率预测问题上生成了明显更短的预测区间。

    We introduce Bellman Conformal Inference (BCI), a framework that wraps around any time series forecasting models and provides calibrated prediction intervals. Unlike the existing methods, BCI is able to leverage multi-step ahead forecasts and explicitly optimize the average interval lengths by solving a one-dimensional stochastic control problem (SCP) at each time step. In particular, we use the dynamic programming algorithm to find the optimal policy for the SCP. We prove that BCI achieves long-term coverage under arbitrary distribution shifts and temporal dependence, even with poor multi-step ahead forecasts. We find empirically that BCI avoids uninformative intervals that have infinite lengths and generates substantially shorter prediction intervals on volatility forecasting problems when compared with existing methods.
    
[^156]: Moco: 一种可学习的组合优化元优化器

    Moco: A Learnable Meta Optimizer for Combinatorial Optimization

    [https://arxiv.org/abs/2402.04915](https://arxiv.org/abs/2402.04915)

    Moco是一个可学习的组合优化元优化器，通过学习图神经网络来更新解决方案构建过程，并能够适应不同的情况和计算预算。

    

    相关的组合优化问题（COPs）通常是NP难的。过去，这些问题主要是通过人工设计的启发式方法来解决的，但是神经网络的进展促使人们开发了从数据中学习启发式方法的通用方法。许多方法利用神经网络直接构建解决方案，但在推理时无法进一步改进已经构建的解决方案。我们的方法Moco学习了一个图神经网络，根据从当前搜索状态提取的特征来更新解决方案构建过程。这种元训练过程以搜索过程中找到的最佳解决方案为目标，给定搜索预算等信息。这使得Moco能够适应不同的情况，例如不同的计算预算。Moco是一个完全可学习的元优化器，不使用任何特定问题的局部搜索或分解。我们在旅行商问题（TSP）和最大最小费用流问题中测试了Moco。

    Relevant combinatorial optimization problems (COPs) are often NP-hard. While they have been tackled mainly via handcrafted heuristics in the past, advances in neural networks have motivated the development of general methods to learn heuristics from data. Many approaches utilize a neural network to directly construct a solution, but are limited in further improving based on already constructed solutions at inference time. Our approach, Moco, learns a graph neural network that updates the solution construction procedure based on features extracted from the current search state. This meta training procedure targets the overall best solution found during the search procedure given information such as the search budget. This allows Moco to adapt to varying circumstances such as different computational budgets. Moco is a fully learnable meta optimizer that does not utilize any problem specific local search or decomposition. We test Moco on the Traveling Salesman Problem (TSP) and Maximum In
    
[^157]: 机器教学中的组合问题探究

    On a Combinatorial Problem Arising in Machine Teaching

    [https://arxiv.org/abs/2402.04907](https://arxiv.org/abs/2402.04907)

    本文研究了机器教学中的一个组合问题，通过证明了一个最坏情况下的猜想，得出了关于教学维度的结果。该结果可以看作是解决了超立方体边界等周问题的定理的推广。

    

    本文研究了一种机器教学模型，其中教师映射是由概念和示例的大小函数构建的。机器教学中的主要问题是任何概念所需的最小示例数量，即所谓的教学维度。最近的一篇论文[7]猜测，在这个模型中，作为概念类大小的函数时，最坏情况发生在一致性矩阵包含从零及以上的二进制表示的数字时。在本文中，我们证明了他们的猜想。该结果可以看作是解决超立方体的边界等周问题的定理[12]的推广，我们的证明基于[10]的引理。

    We study a model of machine teaching where the teacher mapping is constructed from a size function on both concepts and examples. The main question in machine teaching is the minimum number of examples needed for any concept, the so-called teaching dimension. A recent paper [7] conjectured that the worst case for this model, as a function of the size of the concept class, occurs when the consistency matrix contains the binary representations of numbers from zero and up. In this paper we prove their conjecture. The result can be seen as a generalization of a theorem resolving the edge isoperimetry problem for hypercubes [12], and our proof is based on a lemma of [10].
    
[^158]: $\texttt{NeRCC}$: 内嵌回归编码计算用于具有弹性的分布式预测服务系统

    $\texttt{NeRCC}$: Nested-Regression Coded Computing for Resilient Distributed Prediction Serving Systems

    [https://arxiv.org/abs/2402.04377](https://arxiv.org/abs/2402.04377)

    NeRCC是一个通用的抗拖尾节点的近似编码计算框架，包括回归编码、计算和回归解码三个层次，通过优化两个正则化项的依赖关系来解决嵌套回归问题。

    

    对抗拖尾节点(stragglers)是预测服务系统的一个重要特征，任务是在预先训练的机器学习模型上执行输入数据的推理。在本文中，我们提出了一种名为NeRCC的通用的抗拖尾节点的近似编码计算框架。NeRCC包括三个层次：(1)回归编码和抽样，生成编码数据点，作为原始数据点的组合；(2)计算，其中一个工作集群在编码数据点上运行推理；(3)回归解码和抽样，从编码数据点的可用预测中近似恢复出原始数据点的预测结果。我们认为框架的总体目标揭示了编码和解码层中两个回归模型之间的潜在相互关系。我们提出了一个解决嵌套回归问题的方法，通过总结它们对两个联合优化的正则化项的依赖关系。

    Resilience against stragglers is a critical element of prediction serving systems, tasked with executing inferences on input data for a pre-trained machine-learning model. In this paper, we propose NeRCC, as a general straggler-resistant framework for approximate coded computing. NeRCC includes three layers: (1) encoding regression and sampling, which generates coded data points, as a combination of original data points, (2) computing, in which a cluster of workers run inference on the coded data points, (3) decoding regression and sampling, which approximately recovers the predictions of the original data points from the available predictions on the coded data points. We argue that the overall objective of the framework reveals an underlying interconnection between two regression models in the encoding and decoding layers. We propose a solution to the nested regressions problem by summarizing their dependence on two regularization terms that are jointly optimized. Our extensive experi
    
[^159]: 了解算法式思维链中噪声对LLM训练数据的影响

    Understanding the Effect of Noise in LLM Training Data with Algorithmic Chains of Thought

    [https://arxiv.org/abs/2402.04004](https://arxiv.org/abs/2402.04004)

    本论文研究了链式思维中的噪声对LLM训练数据的影响，并开发了追踪整数框架来生成可定制的噪声执行跟踪。通过评估预训练模型在算法可解任务中的表现，揭示了噪声的类型和强度对任务性能的影响。

    

    在预训练和微调过程中，大型语言模型（LLMs）通常会使用数万亿个标记的文本进行训练，这些文本质量各异。在训练的两个阶段中，通常会根据启发式方法过滤掉“低质量”或“有噪声”的训练样本，然而很少有人量化地了解噪声的类型或强度如何影响下游性能。在本研究中，我们研究了链式思维（CoT）中的噪声如何影响在算法可解任务的高度控制环境下的任务性能。首先，我们开发了追踪整数（TInt）框架，用于为任意整数列表上的算术函数生成高度可定制的噪声执行跟踪。然后，我们定义了两种类型的噪声：局部形式的静态噪声，在计算CoT跟踪后应用；以及全局形式的动态噪声，在计算中传播跟踪中的错误。然后，我们评估了预训练模型在测试性能上的表现。

    During both pretraining and fine-tuning, Large Language Models (\textbf{LLMs}) are trained on trillions of tokens of text of widely varying quality. Both phases of training typically involve heuristically filtering out ``low-quality'' or \textit{noisy} training samples, yet little is known quantitatively about how the type or intensity of noise affects downstream performance. In this work, we study how noise in chain of thought (\textbf{CoT}) impacts task performance in the highly-controlled setting of algorithmically solvable tasks. First, we develop the Traced Integer (\textbf{TInt}) framework to generate highly customizable noised execution traces for any arithmetic function on lists of integers. We then define two types of noise: \textit{static} noise, a local form of noise which is applied after the CoT trace is computed, and \textit{dynamic} noise, a global form of noise which propagates errors in the trace as it is computed. We then evaluate the test performance of pretrained mo
    
[^160]: MolTC: 在语言模型中进行分子关系建模

    MolTC: Towards Molecular Relational Modeling In Language Models

    [https://arxiv.org/abs/2402.03781](https://arxiv.org/abs/2402.03781)

    本研究提出了一种基于语言模型的多模态框架MolTC，用于分子相互作用预测，该框架能够高效地整合分子对的丰富图形信息，并通过思维链理论实现统一的分子关系学习。

    

    分子关系学习（MRL）旨在理解分子之间的相互作用，在推进生物化学研究方面起到了关键作用。最近，大型语言模型（LLMs）的采用已成为一种有效和高效的MRL方法，这些模型以其庞大的知识存储库和先进的逻辑推理能力而闻名。尽管具有潜力，但这些方法主要依赖于文本数据，因此没有充分利用分子图中固有的丰富结构信息。此外，缺乏统一的框架加剧了信息的浪费，因为它阻碍了在不同数据集之间共享学习到的相互作用理由。为了解决这些挑战，本研究提出了一种基于LLM的多模态框架，用于根据思维链（CoT）理论对分子相互作用进行预测，称为MolTC，它可以高效地整合分子对的丰富图形信息。

    Molecular Relational Learning (MRL), aiming to understand interactions between molecular pairs, plays a pivotal role in advancing biochemical research. Recently, the adoption of large language models (LLMs), known for their vast knowledge repositories and advanced logical inference capabilities, has emerged as a promising way for efficient and effective MRL. Despite their potential, these methods predominantly rely on the textual data, thus not fully harnessing the wealth of structural information inherent in molecular graphs. Moreover, the absence of a unified framework exacerbates the information underutilization, as it hinders the sharing of interaction rationale learned across diverse datasets. To address these challenges, this work proposes a novel LLM-based multi-modal framework for Molecular inTeraction prediction following Chain-of-Thought (CoT) theory, termed MolTC, which can efficiently integrate rich graphical information of molecular pairs. For achieving a unified MRL, MolT
    
[^161]: Lens: 网络流量的基础模型

    Lens: A Foundation Model for Network Traffic

    [https://arxiv.org/abs/2402.03646](https://arxiv.org/abs/2402.03646)

    "Lens"是一个基于T5架构的基础网络流量模型，通过学习大规模无标签数据的预训练表示，能够在流量理解和生成任务中取得精确的预测和生成。

    

    网络流量是指通过互联网或连接计算机的任何系统发送和接收的信息量。分析和理解网络流量对于提高网络安全和管理至关重要。然而，由于数据包的特殊特性，如异构标头和缺乏语义的加密负载，网络流量的分析带来了巨大的挑战。为了捕捉流量的潜在语义，一些研究采用了基于Transformer编码器或解码器的预训练技术，从大规模的流量数据中学习表示。然而，这些方法通常只在流量理解（分类）或流量生成任务中表现出色。为了解决这个问题，我们开发了Lens，这是一个基础的网络流量模型，利用T5架构从大规模的无标签数据中学习预训练表示。借助编码器-解码器框架的优势，该模型能够捕捉全局和局部特征，实现精确的流量预测和生成。

    Network traffic refers to the amount of information being sent and received over the internet or any system that connects computers. Analyzing and understanding network traffic is vital for improving network security and management. However, the analysis of network traffic poses great challenges due to the unique characteristics of data packets, such as heterogeneous headers and encrypted payload lacking semantics. To capture the latent semantics of traffic, a few studies have adopted pre-training techniques based on the Transformer encoder or decoder to learn the representations from large-scale traffic data. However, these methods typically excel only in traffic understanding (classification) or traffic generation tasks. To address this issue, we develop Lens, a foundational network traffic model that leverages the T5 architecture to learn the pre-trained representations from large-scale unlabeled data. Harnessing the strength of the encoder-decoder framework, which captures the glob
    
[^162]: 学习Predict-then-Optimize框架中的最优策略

    Learning Best-in-Class Policies for the Predict-then-Optimize Framework

    [https://arxiv.org/abs/2402.03256](https://arxiv.org/abs/2402.03256)

    我们提出了一种新颖的决策感知替代损失函数家族，用于predict-then-optimize框架，并且通过数值证据证实了其在误设置下的优越性。

    

    我们提出了一种新颖的决策感知替代损失函数家族，称为Perturbation Gradient（PG）损失，用于predict-then-optimize框架。这些损失直接近似了下游决策损失，并可以使用现成的基于梯度的方法进行优化。重要的是，与现有的替代损失不同，我们的PG损失的近似误差随着样本数量的增加而消失。这意味着优化我们的替代损失可以在渐近意义下得到最佳策略，即使在误设置下也是如此。这是第一个在误设置下的这样的结果，我们提供了数值证据证实了当基础模型误设置且噪声不是中心对称时，我们的PG损失在实践中显著优于现有的提案。鉴于在实践中误设置很常见--特别是当我们可能更喜欢一个更简单、更可解释的模型时--PG损失提供了一种新颖的、理论上有依据的、可计算的决策感知方法。

    We propose a novel family of decision-aware surrogate losses, called Perturbation Gradient (PG) losses, for the predict-then-optimize framework. These losses directly approximate the downstream decision loss and can be optimized using off-the-shelf gradient-based methods. Importantly, unlike existing surrogate losses, the approximation error of our PG losses vanishes as the number of samples grows. This implies that optimizing our surrogate loss yields a best-in-class policy asymptotically, even in misspecified settings. This is the first such result in misspecified settings and we provide numerical evidence confirming our PG losses substantively outperform existing proposals when the underlying model is misspecified and the noise is not centrally symmetric. Insofar as misspecification is commonplace in practice -- especially when we might prefer a simpler, more interpretable model -- PG losses offer a novel, theoretically justified, method for computationally tractable decision-aware 
    
[^163]: 由端到端深度学习模型加强的高效数值波传播

    Efficient Numerical Wave Propagation Enhanced by an End-to-End Deep Learning Model

    [https://arxiv.org/abs/2402.02304](https://arxiv.org/abs/2402.02304)

    本文提出了一个由端到端深度学习模型加强的高效数值波传播方法，通过结合数值求解器和深度学习组件，优化算法架构、数据生成和并行时间算法，实现了在保持速度的同时显著提高性能。

    

    在多个科学和工程领域，从地震建模到医学成像，对于高频波传播的高保真和高效解决方案的需求非常重要。最近在波传播模型中的一项进展利用足够准确的细求解器输出来训练神经网络，以提高快速但不准确的粗求解器的准确性。稳定且快速的求解器还允许使用并行时间算法Parareal来提取和纠正高频波组成部分。在本文中，我们在Nguyen和Tsai（2023）的工作基础上，提出了一个新颖的统一系统，将数值求解器与深度学习组件整合到端到端框架中。在提出的设置中，我们研究了神经网络架构、数据生成算法和Parareal方案的改进。我们的结果表明，这种协调的结构在不牺牲速度的情况下显著提高了性能，并且证明了

    In a variety of scientific and engineering domains, ranging from seismic modeling to medical imaging, the need for high-fidelity and efficient solutions for high-frequency wave propagation holds great significance. Recent advances in wave modeling use sufficiently accurate fine solver outputs to train neural networks that enhance the accuracy of a fast but inaccurate coarse solver. A stable and fast solver further allows the use of Parareal, a parallel-in-time algorithm to retrieve and correct high-frequency wave components. In this paper we build upon the work of Nguyen and Tsai (2023) and present a novel unified system that integrates a numerical solver with deep learning components into an end-to-end framework. In the proposed setting, we investigate refinements to the neural network architecture, data generation algorithm and Parareal scheme. Our results show that the cohesive structure significantly improves performance without sacrificing speed, and demonstrate the importance of 
    
[^164]: 一种多角度的机器学习方法用于评估洛杉矶警察与司机的互动

    A Multi-Perspective Machine Learning Approach to Evaluate Police-Driver Interaction in Los Angeles

    [https://arxiv.org/abs/2402.01703](https://arxiv.org/abs/2402.01703)

    该研究提出了一种多角度的机器学习方法，用于分析洛杉矶警察与司机的互动。该方法利用多模态的数据包括音频、视频和文字信息，旨在提供对复杂和有争议的警民互动的分析工具。

    

    政府官员与市民之间的互动影响公共福祉和民主社会的正当性。警察是国家最显而易见、最接触市民的代理人，在交通站停期间，他们每年与公众互动超过2000万次。如今，这些互动经常被戴在身上的摄像机记录下来，这被视为提高警察问责制和改善警民互动的手段。然而，由于缺乏可靠的自动化工具来分析这些复杂而有争议的警民互动，这些记录的及时分析受到了阻碍。本文提出了一种新的多角度、多模态机器学习（ML）工具的方法，用于分析来自这些身上摄像机记录的音频、视频和文字信息。我们的方法首先确定与不同利益相关者最相关的沟通方面，包括共同感知互动的标志标记以及具有这些标记的符号。

    Interactions between the government officials and civilians affect public wellbeing and the state legitimacy that is necessary for the functioning of democratic society. Police officers, the most visible and contacted agents of the state, interact with the public more than 20 million times a year during traffic stops. Today, these interactions are regularly recorded by body-worn cameras (BWCs), which are lauded as a means to enhance police accountability and improve police-public interactions. However, the timely analysis of these recordings is hampered by a lack of reliable automated tools that can enable the analysis of these complex and contested police-public interactions. This article proposes an approach to developing new multi-perspective, multimodal machine learning (ML) tools to analyze the audio, video, and transcript information from this BWC footage. Our approach begins by identifying the aspects of communication most salient to different stakeholders, including both commun
    
[^165]: 通过深度黑石贝莱曼模型优化时间序列供应商分配风险

    Timeseries Suppliers Allocation Risk Optimization via Deep Black Litterman Model

    [https://arxiv.org/abs/2401.17350](https://arxiv.org/abs/2401.17350)

    通过深度黑石贝莱曼模型和时空图神经网络，我们优化了供应商选择和订单分配，同时解决了零阶情况下的可信度问题，实现了准确的预测和精确的置信区间。

    

    我们介绍了BL模型和Perspective矩阵，以优化供应商选择和订单分配，重点关注时间和空间动态。我们使用时空图神经网络开发了供应商关系网络，增强了对复杂供应商相互依赖关系的理解。此外，我们还通过Masked Ranking机制解决了零阶情况下的可信度问题，提高了供应商排序效率。与传统模型相比，我们的模型在两个数据集上展现了优越的结果。我们使用真实数据集进行的评估突出了DBLM在提供准确预测和精确置信区间方面的优势，特别是在高分辨率情景下。

    We introduce the BL model and the Perspective Matrix to optimize supplier selection and order allocation, focusing on both temporal and spatial dynamics. Our development of a Supplier Relationship Network, using a Spatio-Temporal Graph Neural Network, enhances the understanding of complex supplier interdependencies. Additionally, we address credibility issues in zero-order scenarios with a Masked Ranking Mechanism, improving supplier ranking efficiency. Our model demonstrates superior results on two datasets compared to the traditional models. Our evaluations using real-world datasets highlight DBLM's superiority in providing accurate predictions and precise confidence intervals, particularly in high-resolution scenarios.
    
[^166]: 扩展就是一切：使用JAX加速强化学习的自动驾驶

    Scaling Is All You Need: Autonomous Driving with JAX-Accelerated Reinforcement Learning

    [https://arxiv.org/abs/2312.15122](https://arxiv.org/abs/2312.15122)

    本研究提出了一种扩展的自动驾驶强化学习方法，在大规模实验中展示了随着规模增加，策略性能的改善。与现有机器学习自动驾驶策略相比，我们的最佳策略将故障率降低了64％，同时提高了25％的驾驶进展速度。

    

    强化学习已经在复杂领域如视频游戏中展现出超越最优人类的能力。然而，为自动驾驶运行必要规模的强化学习实验非常困难。构建一个大规模的强化学习系统并在多个GPU上进行分布是具有挑战性的。在训练过程中在真实世界车辆上收集经验从安全和可扩展性的角度来看是不可行的。因此，需要一个高效且真实的驾驶模拟器，使用大量来自真实驾驶的数据。我们将这些能力集合在一起，并进行大规模的强化学习实验用于自动驾驶。我们证明，随着规模的增加，我们的策略表现得到了提升。我们最佳策略将故障率降低了64％，同时比现有机器学习自动驾驶策略提高了25％的驾驶进展速度。

    Reinforcement learning has been demonstrated to outperform even the best humans in complex domains like video games. However, running reinforcement learning experiments on the required scale for autonomous driving is extremely difficult. Building a large scale reinforcement learning system and distributing it across many GPUs is challenging. Gathering experience during training on real world vehicles is prohibitive from a safety and scalability perspective. Therefore, an efficient and realistic driving simulator is required that uses a large amount of data from real-world driving. We bring these capabilities together and conduct large-scale reinforcement learning experiments for autonomous driving. We demonstrate that our policy performance improves with increasing scale. Our best performing policy reduces the failure rate by 64% while improving the rate of driving progress by 25% compared to the policies produced by state-of-the-art machine learning for autonomous driving.
    
[^167]: 多模态注意力合并用于提高语音识别和音频事件分类

    Multimodal Attention Merging for Improved Speech Recognition and Audio Event Classification

    [https://arxiv.org/abs/2312.14378](https://arxiv.org/abs/2312.14378)

    多模态注意力合并（MAM）使用零-shot范式将文本和图像中的模型注意力矩阵的直接知识传输到音频领域中，可以显著降低自动语音识别的词错误率和音频事件分类的分类错误。

    

    使用自我监督目标在无标签数据上训练大型基础模型，然后在下游任务中进行微调已成为一种标准的流程。然而，这种方法的有效性经常受限于有限的微调计算资源和标记下游数据的稀缺性。我们引入了多模态注意力合并（MAM），试图通过零-shot范式将高资源模态（文本和图像）中模型注意力矩阵的直接知识传输到资源受限领域（语音和音频）中。MAM将自动语音识别（ASR）模型的相对词错误率（WER）降低了最多6.70％，将音频事件分类（AEC）模型的相对分类错误降低了10.63％。在一些数据/计算可用的情况下，我们提出了可学习的MAM，一种基于数据的方法来合并注意力矩阵，使ASR的WER相对降低了进一步的2.90％，而AEC的相对降低了18.42％。

    Training large foundation models using self-supervised objectives on unlabeled data, followed by fine-tuning on downstream tasks, has emerged as a standard procedure. Unfortunately, the efficacy of this approach is often constrained by both limited fine-tuning compute and scarcity in labeled downstream data. We introduce Multimodal Attention Merging (MAM), an attempt that facilitates direct knowledge transfer from attention matrices of models rooted in high resource modalities, text and images, to those in resource-constrained domains, speech and audio, employing a zero-shot paradigm. MAM reduces the relative Word Error Rate (WER) of an Automatic Speech Recognition (ASR) model by up to 6.70%, and relative classification error of an Audio Event Classification (AEC) model by 10.63%. In cases where some data/compute is available, we present Learnable-MAM, a data-driven approach to merging attention matrices, resulting in a further 2.90% relative reduction in WER for ASR and 18.42% relativ
    
[^168]: 使用TableShift来评估表格数据的分布漂移

    Benchmarking Distribution Shift in Tabular Data with TableShift

    [https://arxiv.org/abs/2312.07577](https://arxiv.org/abs/2312.07577)

    这篇论文介绍了一个针对表格数据的分布漂移基准测试TableShift，包含15个二分类任务和相应的分布漂移，涵盖了多个领域，并且通过Python代码可以轻松访问。

    

    随着文本和图像模型从研究对象转向实际部署，对于分布漂移的鲁棒性越来越受关注。然而，尽管表格机器学习任务被广泛应用于现实世界中，但针对表格数据的分布漂移的高质量基准测试仍然缺乏，而且与文本和图像模型的差异进一步加剧了这个问题。因此，对于表格模型在面对分布漂移时的鲁棒性了解甚少。为了解决这个问题，我们引入了TableShift，一个针对表格数据的分布漂移基准测试。TableShift总共包含15个二分类任务，每个任务都有相应的分布漂移，并包含了多样化的数据来源、预测目标和分布漂移。该基准测试涵盖了金融、教育、公共政策、医疗保健和公民参与等领域，并且可以通过仅几行Python代码的TableShift API进行访问。我们进行了大规模的f评估。

    Robustness to distribution shift has become a growing concern for text and image models as they transition from research subjects to deployment in the real world. However, high-quality benchmarks for distribution shift in tabular machine learning tasks are still lacking despite the widespread real-world use of tabular data and differences in the models used for tabular data in comparison to text and images. As a consequence, the robustness of tabular models to distribution shift is poorly understood. To address this issue, we introduce TableShift, a distribution shift benchmark for tabular data. TableShift contains 15 binary classification tasks in total, each with an associated shift, and includes a diverse set of data sources, prediction targets, and distribution shifts. The benchmark covers domains including finance, education, public policy, healthcare, and civic participation, and is accessible using only a few lines of Python code via the TableShift API. We conduct a large-scale 
    
[^169]: LayerCollapse: 自适应压缩神经网络

    LayerCollapse: Adaptive compression of neural networks

    [https://arxiv.org/abs/2311.17943](https://arxiv.org/abs/2311.17943)

    LayerCollapse是一种自适应压缩神经网络的方法，通过结构化剪枝来减少全连接层的深度，而不需要进行微调，并且对性能影响有限。该方法通过正则化激活函数的线性度来控制模型的表达能力。

    

    处理当代深度学习和基于transformer的模型不断增长的规模是一个重大挑战。超参数化的Transformer网络在自然语言处理和计算机视觉方面的业绩超过了先前的技术。这些模型含有数亿个参数，需要大量的计算资源，并容易过拟合。在这项工作中，我们提出了LayerCollapse，一种结构化剪枝的形式，用于减少全连接层的深度。我们开发了一种新的正则化项，允许在不进行微调的情况下进行训练后压缩，并对性能产生有限的影响。LayerCollapse通过对全连接层之间的激活进行正则化，调节激活函数的线性度来控制模型的表达能力。线性激活函数将线性转换的秩降低到相应线性转换的秩。我们通过展示LayerCollapse的压缩能力来证明其有效性。

    Handling the ever-increasing scale of contemporary deep learning and transformer-based models poses a significant challenge. Overparameterized Transformer networks outperform prior art in Natural Language processing and Computer Vision. These models contain hundreds of millions of parameters, demanding significant computational resources and making them prone to overfitting. In this work we present LayerCollapse, a form of structured pruning to reduce the depth of fully connected layers. We develop a novel regularizer allowing for post-training compression without finetuning, while having limited impact on performance. LayerCollapse controls model expressiveness with regularization on the activations between fully connected layers, modulating the linearity of activation functions. A linear activation function reduces the rank of the transformation to the rank of the corresponding linear transformation. We demonstrate the effectiveness of LayerCollapse by showing its compression capabil
    
[^170]: 用于非结构稀疏恢复的特征矩阵

    Eigenmatrix for unstructured sparse recovery

    [https://arxiv.org/abs/2311.16609](https://arxiv.org/abs/2311.16609)

    本文提出了一种名为特征矩阵的数据驱动构造，用于解决非结构稀疏恢复问题，对于样本值中的噪声和样本位置的非结构性质具有很好的适应性。

    

    本文考虑了一般形式的非结构稀疏恢复问题，包括有理逼近、谱函数估计、傅里叶反演、拉普拉斯反演和稀疏反卷积等。主要挑战是样本值中的噪声和样本位置的非结构性质。本文提出了特征矩阵，一种具有所需近似特征值和特征向量的数据驱动构造，为这些稀疏恢复问题提供了一种新的方法。数值结果证明了所提方法的效率。

    This paper considers the unstructured sparse recovery problems in a general form. Examples include rational approximation, spectral function estimation, Fourier inversion, Laplace inversion, and sparse deconvolution. The main challenges are the noise in the sample values and the unstructured nature of the sample locations. This paper proposes the eigenmatrix, a data-driven construction with desired approximate eigenvalues and eigenvectors. The eigenmatrix offers a new way for these sparse recovery problems. Numerical results are provided to demonstrate the efficiency of the proposed method.
    
[^171]: 程序机器策略：通过集成程序合成和状态机解决长期任务

    Program Machine Policy: Addressing Long-Horizon Tasks by Integrating Program Synthesis and State Machines

    [https://arxiv.org/abs/2311.15960](https://arxiv.org/abs/2311.15960)

    这项工作提出了程序机器策略（POMP），在集成程序合成和状态机的基础上，解决了长期任务并表示复杂行为。

    

    深度强化学习在各个领域表现出色，但缺乏泛化能力和解释性。另一方面，编程式强化学习方法重新定义了强化学习任务，将其视为合成可解释的程序，可以在环境中执行。尽管取得了鼓舞人心的结果，但这些方法局限于短期任务。另一方面，使用状态机表示强化学习策略可以归纳推广到长期任务；然而，它在获取多样和复杂行为方面存在困难。本研究提出了程序机器策略（POMP），以桥接编程式强化学习和状态机策略的优势，允许表示复杂行为并解决长期任务。具体而言，我们介绍了一种方法，可以检索一组有效、多样且兼容的程序。然后，我们将这些程序用作状态机的模式，并学习一个转移函数。

    Deep reinforcement learning (deep RL) excels in various domains but lacks generalizability and interpretability. On the other hand, programmatic RL methods (Trivedi et al., 2021; Liu et al., 2023) reformulate RL tasks as synthesizing interpretable programs that can be executed in the environments. Despite encouraging results, these methods are limited to short-horizon tasks. On the other hand, representing RL policies using state machines (Inala et al., 2020) can inductively generalize to long-horizon tasks; however, it struggles to scale up to acquire diverse and complex behaviors. This work proposes the Program Machine Policy (POMP), which bridges the advantages of programmatic RL and state machine policies, allowing for the representation of complex behaviors and the address of long-term tasks. Specifically, we introduce a method that can retrieve a set of effective, diverse, and compatible programs. Then, we use these programs as modes of a state machine and learn a transition func
    
[^172]: 可控的高昂多目标学习与热启动贝叶斯优化

    Controllable Expensive Multi-objective Learning with Warm-starting Bayesian Optimization

    [https://arxiv.org/abs/2311.15297](https://arxiv.org/abs/2311.15297)

    这项工作提出了一种可控的Pareto Set Learning (PSL)方法，通过热启动贝叶斯优化和可控的帕累托集学习来解决现有PSL方法不稳定和低效的问题，并在合成和实际MOO问题上展示了其有效性。

    

    Pareto Set Learning (PSL)是一种在多目标优化（MOO）问题中近似整个帕累托前沿的有希望的方法。然而，现有的无导数PSL方法通常不稳定且低效，特别是对于昂贵的黑箱MOO问题，目标函数评估成本高昂。在这项工作中，我们提出了一种新颖的可控PSL方法，称为Co-PSL，以解决现有PSL方法的不稳定性和低效性。特别地，Co-PSL包括两个阶段：(1)热启动贝叶斯优化，以获得质量高的高斯过程先验，(2)可控帕累托集学习，以准确获取从偏好到相应帕累托解的参数映射。前者有助于稳定PSL过程并减少昂贵函数评估的数量。后者支持冲突目标之间的实时权衡控制。通过合成和实际世界的MOO问题的性能展示了Co-PSL方法的有效性。

    Pareto Set Learning (PSL) is a promising approach for approximating the entire Pareto front in multi-objective optimization (MOO) problems. However, existing derivative-free PSL methods are often unstable and inefficient, especially for expensive black-box MOO problems where objective function evaluations are costly. In this work, we propose to address the instability and inefficiency of existing PSL methods with a novel controllable PSL method, called Co-PSL. Particularly, Co-PSL consists of two stages: (1) warm-starting Bayesian optimization to obtain quality Gaussian Processes priors and (2) controllable Pareto set learning to accurately acquire a parametric mapping from preferences to the corresponding Pareto solutions. The former is to help stabilize the PSL process and reduce the number of expensive function evaluations. The latter is to support real-time trade-off control between conflicting objectives. Performances across synthesis and real-world MOO problems showcase the effec
    
[^173]: DroneOptiNet: 一种用于5G及其后太阳能小型蜂窝网络的最佳无人机负载重分配机制的框架

    DroneOptiNet: A Framework for Optimal Drone-based Load Redistribution Mechanism for 5G and Beyond Solar Small Cell Networks

    [https://arxiv.org/abs/2311.12944](https://arxiv.org/abs/2311.12944)

    本研究提出了一种用于5G及其后太阳能小型蜂窝网络的最佳无人机负载重分配机制，通过使用无人机上的空中基站进行可靠安全的电力再分配，提高了网络的可靠性和稳健性。

    

    第五代及其后的蜂窝网络对功率需求提出了重要的限制，需要能够高效利用能源的解决方案。在本研究中，我们提出了一种新颖的使用无人机上的空中基站（BS）进行可靠安全的电力再分配的用户负载转移方法，以跨越由绿色小型蜂窝BS组成的微网网络。根据用户密度和空中基站的可用性，通过将空中基站从高能耗小区迁移到低能耗小区，来满足能量不足的小区的能量需求。所提出的混合无人机框架将长短期记忆与独特的成本函数结合，使用进化神经网络来有效地管理无人机和基站的能量和负载重分配。所提出的算法减少了基站停电，并保持了一致的吞吐量稳定性，从而展示了其提升无线网络可靠性和稳健性的能力。

    The power requirements posed by the fifth-generation and beyond cellular networks are an important constraint in network deployment and require energy-efficient solutions. In this work, we propose a novel user load transfer approach using airborne base stations (BS) mounted on drones for reliable and secure power redistribution across the micro-grid network comprising green small cell BSs. Depending on the user density and the availability of an aerial BS, the energy requirement of a cell with an energy deficit is accommodated by migrating the aerial BS from a high-energy to a low-energy cell. The proposed hybrid drone-based framework integrates long short-term memory with unique cost functions using an evolutionary neural network for drones and BSs and efficiently manages energy and load redistribution. The proposed algorithm reduces power outages at BSs and maintains consistent throughput stability, thereby demonstrating its capability to boost the reliability and robustness of wirel
    
[^174]: 通过采用整数列表作为多项式基数2指数的集合来实现快速乘法

    Fast multiplication by two's complement addition of numbers represented as a set of polynomial radix 2 indexes, stored as an integer list for massively parallel computation

    [https://arxiv.org/abs/2311.09922](https://arxiv.org/abs/2311.09922)

    本论文介绍了一种基于多项式基数2指数集合的快速乘法方法，在特定位数范围内比传统方法更快。该方法把数字表示为整数索引列表，并实现了分布式计算。

    

    我们演示了一种基于用整数列表表示的多项式基数2指数集合的乘法方法。该方法采用python代码实现了一组算法。我们展示了该方法在某一位数范围内比数论变换(NTT)和卡拉茨巴(Karatsuba)乘法更快。我们还实现了用python代码进行比较，与多项式基数2整数方法进行比较。我们展示了任何整数或实数都可以表示为整数索引列表，表示二进制中的有限级数。该数字的整数索引有限级数可以存储和分布在多个CPU / GPU上。我们展示了加法和乘法运算可以应用于作为索引整数表示的两个补码加法，并可以完全分布在给定的CPU / GPU架构上。我们展示了完全的分布性能。

    We demonstrate a multiplication method based on numbers represented as set of polynomial radix 2 indices stored as an integer list. The 'polynomial integer index multiplication' method is a set of algorithms implemented in python code. We demonstrate the method to be faster than both the Number Theoretic Transform (NTT) and Karatsuba for multiplication within a certain bit range. Also implemented in python code for comparison purposes with the polynomial radix 2 integer method. We demonstrate that it is possible to express any integer or real number as a list of integer indices, representing a finite series in base two. The finite series of integer index representation of a number can then be stored and distributed across multiple CPUs / GPUs. We show that operations of addition and multiplication can be applied as two's complement additions operating on the index integer representations and can be fully distributed across a given CPU / GPU architecture. We demonstrate fully distribute
    
[^175]: 自然度解释与评估模型（CNE）：解释和评估形成自然度的模式的框架

    Confident Naturalness Explanation (CNE): A Framework to Explain and Assess Patterns Forming Naturalness

    [https://arxiv.org/abs/2311.08936](https://arxiv.org/abs/2311.08936)

    本文提出了一个新的框架，称为自然度解释与评估模型（CNE）。该框架利用可解释的机器学习方法来分析卫星图像，以解释和评估形成自然度的模式，并带有相应的置信度。

    

    受人类活动（如城市化、农业和其他人为干预）最小影响的自然保护区是指那些地区。为了更好地理解和绘制这些区域的自然度，可以使用机器学习模型分析卫星图像。具体而言，可解释的机器学习方法在揭示这些保护环境中有助于自然度概念的模式方面显示出了潜力。此外，解决与机器学习模型固有的不确定性相关的问题对于全面了解这个概念至关重要。然而，现有方法存在局限性。他们要么无法提供既有效又客观的解释，要么难以提供准确衡量特定模式对自然度贡献的定量指标以及相关的置信度。在本文中，我们提出了一种新颖的框架，称为自然度解释与评估模型（CNE）。

    Protected natural areas are regions that have been minimally affected by human activities such as urbanization, agriculture, and other human interventions. To better understand and map the naturalness of these areas, machine learning models can be used to analyze satellite imagery. Specifically, explainable machine learning methods show promise in uncovering patterns that contribute to the concept of naturalness within these protected environments. Additionally, addressing the uncertainty inherent in machine learning models is crucial for a comprehensive understanding of this concept. However, existing approaches have limitations. They either fail to provide explanations that are both valid and objective or struggle to offer a quantitative metric that accurately measures the contribution of specific patterns to naturalness, along with the associated confidence. In this paper, we propose a novel framework called the Confident Naturalness Explanation (CNE) framework. This framework combi
    
[^176]: 通过最优传输实现公平的核心集

    Fair Coresets via Optimal Transport

    [https://arxiv.org/abs/2311.05436](https://arxiv.org/abs/2311.05436)

    本研究提出了公平的Wasserstein核心集(FWC)，该方法通过最小化原始数据集与加权合成样本之间的Wasserstein距离，并强制实现人口平等，生成公平的合成代表性样本，可用于下游学习任务。

    

    数据精炼和核心集已成为生成用于处理大规模数据集的下游学习任务的较小代表性样本集的流行方法。与此同时，机器学习越来越多地应用于社会层面的决策过程，使得模型构建者必须解决存在于数据中的子群体的固有偏见问题。当前方法通过优化相对于原始样本的局部属性来创建公平的合成代表性样本，但其对下游学习过程的影响尚未被探索。在这项工作中，我们提出了公平的Wasserstein核心集（FWC），一种新颖的核心集方法，它生成既具有公平性的合成代表性样本，又具有用于下游学习任务的样本级权重。FWC最小化原始数据集与加权合成样本之间的Wasserstein距离，同时强制实现人口平等。我们展示了FWC的无约束版本等价于通常的最优传输问题，并且通过实验证明了FWC的有效性和公平性。

    Data distillation and coresets have emerged as popular approaches to generate a smaller representative set of samples for downstream learning tasks to handle large-scale datasets. At the same time, machine learning is being increasingly applied to decision-making processes at a societal level, making it imperative for modelers to address inherent biases towards subgroups present in the data. Current approaches create fair synthetic representative samples by optimizing local properties relative to the original samples, but their effect on downstream learning processes has yet to be explored. In this work, we present fair Wasserstein coresets (FWC), a novel coreset approach which generates fair synthetic representative samples along with sample-level weights to be used in downstream learning tasks. FWC minimizes the Wasserstein distance between the original dataset and the weighted synthetic samples while enforcing demographic parity. We show that an unconstrained version of FWC is equiv
    
[^177]: 本地通用解释器（LUX）-- 一种基于规则的解释器，具有事实、反事实和视觉解释

    Local Universal Explainer (LUX) -- a rule-based explainer with factual, counterfactual and visual explanations

    [https://arxiv.org/abs/2310.14894](https://arxiv.org/abs/2310.14894)

    LUX是一种基于规则的解释器，可以生成事实、反事实和视觉解释，通过选择高密度簇形式的局部概念来形成决策边界。

    

    可解释的人工智能（XAI）是近年来最被广泛发展的人工智能领域之一。它也是最分散的领域之一，有多种方法专注于解释的不同方面。这使得一次性以紧凑和一致的方式获得完整的解释变得困难。为了解决这个问题，我们提出了本地通用解释器（LUX），它是一种基于规则的解释器，可以生成事实、反事实和视觉解释。它基于修改后的决策树算法，允许斜交和集成特征重要性XAI方法，如SHAP或LIME。与其他算法相反，它不使用数据生成，而是专注于选择以高密度簇形式出现的真实数据的局部概念，这些局部概念对解释模型的决策边界形成最大的影响。我们在真实和合成数据集上测试了我们的方法，并将其与最先进的基于规则的方法进行了比较。

    Explainable artificial intelligence (XAI) is one of the most intensively developed area of AI in recent years. It is also one of the most fragmented with multiple methods that focus on different aspects of explanations. This makes difficult to obtain the full spectrum of explanation at once in a compact and consistent way. To address this issue, we present Local Universal Explainer (LUX), which is a rule-based explainer that can generate factual, counterfactual and visual explanations. It is based on a modified version of decision tree algorithms that allows for oblique splits and integration with feature importance XAI methods such as SHAP or LIME. It does not use data generation in opposite to other algorithms, but is focused on selecting local concepts in a form of high-density clusters of real data that have the highest impact on forming the decision boundary of the explained model. We tested our method on real and synthetic datasets and compared it with state-of-the-art rule-based
    
[^178]: 高效且可解释的赌博算法

    Efficient and Interpretable Bandit Algorithms

    [https://arxiv.org/abs/2310.14751](https://arxiv.org/abs/2310.14751)

    这个论文设计了一种高效且可解释的赌博算法，其中关注了模型参数的解释性和减小不确定性的目标。提出的CODE算法通过在符合约束条件的所有可能操作中进行探索，实现了最大程度的解释性和不确定性的减小。

    

    考虑到现代机器学习中解释性的重要性，我们设计了高效且可解释的赌博算法。如果一个赌博算法探索的目标是减小未知模型参数的不确定性，那么它就具有可解释性。为了衡量可解释性，我们引入了一种新的模型误差度量，对比了所有可能操作的平均奖励估计的减少率与它们的实际均值之间的差异。我们提出了一种基于约束最优设计的赌博算法CODE，它具有可解释性，并最大程度地减小了不确定性。CODE的关键思想是在符合统计约束条件的所有可能操作中进行探索，以实现可解释性。我们在多臂赌博机和线性赌博机中高效地实现了CODE，并通过利用近似最优设计的最优性准则推导出了接近最优的后悔界限。CODE还可以看作是去除传统的相位消除中的阶段，这使得算法更加高效。

    Motivated by the importance of explainability in modern machine learning, we design bandit algorithms that are efficient and interpretable. A bandit algorithm is interpretable if it explores with the objective of reducing uncertainty in the unknown model parameter. To quantify the interpretability, we introduce a novel metric of model error, which compares the rate reduction of the mean reward estimates to their actual means among all the plausible actions. We propose CODE, a bandit algorithm based on a Constrained Optimal DEsign, that is interpretable and maximally reduces the uncertainty. The key idea in CODE is to explore among all plausible actions, determined by a statistical constraint, to achieve interpretability. We implement CODE efficiently in both multi-armed and linear bandits and derive near-optimal regret bounds by leveraging the optimality criteria of the approximate optimal design. CODE can be also viewed as removing phases in conventional phased elimination, which make
    
[^179]: 无标签验证数据下零样本异常检测器的模型选择

    Model Selection of Zero-shot Anomaly Detectors in the Absence of Labeled Validation Data

    [https://arxiv.org/abs/2310.10461](https://arxiv.org/abs/2310.10461)

    本研究提出了一个通用框架SWSA（Selection With Synthetic Anomalies），用于在没有标签验证数据的情况下选择基于图像的零样本异常检测器。通过生成合成验证集，该方法能够实现模型选择，并在实证研究中展示了比基线方法更高的AUROC。

    

    异常检测需要在大型无标签数据集中检测异常样本。尽管深度学习的进步和基础模型的出现产生了强大的零样本异常检测方法，但其在实践中的应用常常受到标签数据的缺乏的限制 - 在没有标签数据的情况下，无法可靠地评估其检测性能。在这项工作中，我们提出了一种通用框架SWSA（Selection With Synthetic Anomalies）来选择基于图像的异常检测器，并使用生成的合成验证集。我们提出的异常生成方法假设只有少量的正常图像支持集，并且不需要训练或微调。生成后，我们的合成验证集被用于创建模型选择的验证框架中的检测任务。在实证研究中，我们发现SWSA常常选择与真实验证集选择相匹配的模型，结果比基线方法的AUROC更高。

    Anomaly detection requires detecting abnormal samples in large unlabeled datasets. While progress in deep learning and the advent of foundation models has produced powerful zero-shot anomaly detection methods, their deployment in practice is often hindered by the lack of labeled data -- without it, their detection performance cannot be evaluated reliably. In this work, we propose SWSA (Selection With Synthetic Anomalies): a general-purpose framework to select image-based anomaly detectors with a generated synthetic validation set. Our proposed anomaly generation method assumes access to only a small support set of normal images and requires no training or fine-tuning. Once generated, our synthetic validation set is used to create detection tasks that compose a validation framework for model selection. In an empirical study, we find that SWSA often selects models that match selections made with a ground-truth validation set, resulting in higher AUROCs than baseline methods. We also find
    
[^180]: Sorted LLaMA: 揭示大型语言模型中间层的潜力，用于动态推理

    Sorted LLaMA: Unlocking the Potential of Intermediate Layers of Large Language Models for Dynamic Inference

    [https://arxiv.org/abs/2309.08968](https://arxiv.org/abs/2309.08968)

    Sorted LLaMA通过扩展SortedNet到生成NLP任务，使得大型语言模型在动态推理中更高效，并且不需要预训练，只需将标准微调替换为排序微调即可。该方法可以释放transformers中间层的潜力，同时最小化存储需求和过渡成本。

    

    大型语言模型（LLMs）通过在理解和生成类似人类文本方面表现出色，为自然语言处理（NLP）领域带来了革命。然而，广泛部署这些模型可能成本过高。SortedNet是一种最近的训练技术，通过利用网络中的模块化和基于计算/准确性对子模型进行嵌套排序，实现了动态推理。我们将SortedNet扩展到生成NLP任务，使大型语言模型在不进行任何预训练的情况下变得动态，仅通过将标准微调（SFT）替换为排序微调（SoFT）。我们的方法提高了模型的效率，消除了在推理过程中在不同场景中使用多个模型的需求。我们展示了这种方法可以释放transformers中间层在生成目标输出方面的能力。我们的子模型仍然是原始模型的组成部分，最小化了存储需求和在不同计算/延迟预算之间的过渡成本。

    Large language models (LLMs) have revolutionized natural language processing (NLP) by excelling at understanding and generating human-like text. However, their widespread deployment can be prohibitively expensive. SortedNet is a recent training technique for enabling dynamic inference by leveraging the modularity in networks and sorting sub-models based on computation/accuracy in a nested manner. We extend SortedNet to generative NLP tasks, making large language models dynamic without any Pre-Training and by only replacing Standard Fine-Tuning (SFT) with Sorted Fine-Tuning (SoFT). Our approach boosts model efficiency, eliminating the need for multiple models for various scenarios during inference. We show that this approach can unlock the power of intermediate layers of transformers in generating the target output. Our sub-models remain integral components of the original model, minimizing storage requirements and transition costs between different computational/latency budgets. The ef
    
[^181]: 重新思考图规范化在图表示学习中的能力

    Rethinking the Power of Graph Canonization in Graph Representation Learning with Stability

    [https://arxiv.org/abs/2309.00738](https://arxiv.org/abs/2309.00738)

    这篇论文提出了通过图规范化最大化GNNs表达能力的方法，并从模型稳定性角度研究了这些GNNs的能力。本文基于稳定GNN将相似的图映射到紧密相连的向量表示中，理论上揭示了图规范化增强的GNNs在表达能力和稳定性之间的折衷，提出了普遍图规范化的概念并给出了一种广泛适用的充分条件来解决普遍图规范化问题。实验证明了这种方法的有效性。

    

    近年来，图神经网络（GNNs）的表达能力得到了广泛研究，以揭示设计更强大的GNNs的原则。图规范化作为一种区分非同构图的典型方法，但在开发表达能力强的GNNs时很少被采用。本文提出通过图规范化最大化GNNs的表达能力，然后从模型稳定性的角度来研究这些GNNs的能力。稳定的GNN会将相似的图映射到在向量空间中紧密相连的图表示中，而GNN的稳定性对于将性能推广到未见过的图很关键。我们在理论上揭示了图规范化增强的GNNs在表达能力和稳定性之间的折衷。然后，我们引入了普遍图规范化的概念，作为解决折衷的通用解决方案，并表征了一种广泛适用的充分条件来解决普遍图规范化问题。一系列全面的实验证明了其有效性。

    The expressivity of Graph Neural Networks (GNNs) has been studied broadly in recent years to reveal the design principles for more powerful GNNs. Graph canonization is known as a typical approach to distinguish non-isomorphic graphs, yet rarely adopted when developing expressive GNNs. This paper proposes to maximize the expressivity of GNNs by graph canonization, then the power of such GNNs is studies from the perspective of model stability. A stable GNN will map similar graphs to close graph representations in the vectorial space, and the stability of GNNs is critical to generalize their performance to unseen graphs. We theoretically reveal the trade-off of expressivity and stability in graph-canonization-enhanced GNNs. Then we introduce a notion of universal graph canonization as the general solution to address the trade-off and characterize a widely applicable sufficient condition to solve the universal graph canonization. A comprehensive set of experiments demonstrates the effectiv
    
[^182]: CAMMARL: 多智能体强化学习中的一致行动建模

    CAMMARL: Conformal Action Modeling in Multi Agent Reinforcement Learning

    [https://arxiv.org/abs/2306.11128](https://arxiv.org/abs/2306.11128)

    CAMMARL是一种新颖的多智能体强化学习算法，通过使用一致预测的概念对其他智能体的行动进行建模，并量化不确定性，提高了智能体的决策能力。

    

    在与多个智能体的环境中采取行动之前，自主智能体可能会从推理其他智能体和利用对系统行为的保证或信心的概念中获益。在本文中，我们提出了一种新颖的多智能体强化学习（MARL）算法CAMMARL，该算法涉及以置信集的形式对不同情况下其他智能体的行动进行建模，即包含其真实行动的高概率集合。然后，我们使用这些估计结果来指导智能体的决策。为了估计这些集合，我们使用了一致预测的概念，通过这种方式，我们不仅可以获得最有可能的结果的估计，还可以量化可操作的不确定性。例如，我们可以预测一个可证明以高概率（例如95％）覆盖真实预测的集合。通过在两个完全合作的多智能体任务上进行多个实验，我们展示了CAMMARL提高了智能体的能力。

    Before taking actions in an environment with more than one intelligent agent, an autonomous agent may benefit from reasoning about the other agents and utilizing a notion of a guarantee or confidence about the behavior of the system. In this article, we propose a novel multi-agent reinforcement learning (MARL) algorithm CAMMARL, which involves modeling the actions of other agents in different situations in the form of confident sets, i.e., sets containing their true actions with a high probability. We then use these estimates to inform an agent's decision-making. For estimating such sets, we use the concept of conformal predictions, by means of which, we not only obtain an estimate of the most probable outcome but get to quantify the operable uncertainty as well. For instance, we can predict a set that provably covers the true predictions with high probabilities (e.g., 95%). Through several experiments in two fully cooperative multi-agent tasks, we show that CAMMARL elevates the capabi
    
[^183]: 优化一级定价拍卖中的底价：雅虎广告的实证研究

    Optimizing Floors in First Price Auctions: an Empirical Study of Yahoo Advertising

    [https://arxiv.org/abs/2302.06018](https://arxiv.org/abs/2302.06018)

    本研究提出了一个在一级定价拍卖中设定底价的模型，并应用于雅虎广告，从而使得在线发布者能够增加广告收入。

    

    底价（也称为保留价格）帮助发布者增加其广告空间的预期收入，通常通过拍卖方式出售。底价被定义为卖方（可以是发布者或广告交易所）愿意接受的最低出价。本文提出了一个在一级定价拍卖中设定底价的模型，并讨论了其在雅虎网站上实施的影响。该模型捕捉了在线广告行业的重要特征。例如，一些竞标者对广告交易所如何处理竞标者的数据提出限制，这限制了设置保留价格的模型选择。我们的解决方案通过在竞标请求中加入底价来引导竞标者改变其竞标行为，帮助在线发布者增加其广告收入。该解决方案已在雅虎上实施，并取得了显著的成果。预计雅虎的展示广告存货增量年均收入为+1.3%。

    Floors (also known as reserve prices) help publishers to increase the expected revenue of their ad space, which is usually sold via auctions. Floors are defined as the minimum bid that a seller (it can be a publisher or an ad exchange) is willing to accept for the inventory opportunity. In this paper, we present a model to set floors in first price auctions, and discuss the impact of its implementation on Yahoo sites. The model captures important characteristics of the online advertising industry. For instance, some bidders impose restrictions on how ad exchanges can handle data from bidders, conditioning the model choice to set reserve prices. Our solution induces bidders to change their bidding behavior as a response to the floors enclosed in the bid request, helping online publishers to increase their ad revenue.   The outlined methodology has been implemented at Yahoo with remarkable results. The annualized incremental revenue is estimated at +1.3% on Yahoo display inventory, and +
    
[^184]: 数据增强和损失函数在钻孔工具磨损检测的语义图像分割中的评估

    Evaluation of Data Augmentation and Loss Functions in Semantic Image Segmentation for Drilling Tool Wear Detection

    [https://arxiv.org/abs/2302.05262](https://arxiv.org/abs/2302.05262)

    本研究评估了在钻孔工具磨损检测的语义图像分割中的数据增强和损失函数。结果发现，在适度增强的数据上训练的二元模型表现最佳。

    

    在制造过程中，工具磨损监测对于质量控制和成本降低至关重要。本文提出了一种基于U-Net的语义图像分割流程，应用于切割插入物的显微图像，用于磨损检测。磨损区域分为两种不同类型，形成一个多类别分类问题。另一方面，将两种磨损类型合并为一个通用的磨损类别，可以把问题定义为二元分类任务。除了比较二分类和多分类问题外，还研究了不同的损失函数，包括交叉熵、焦点交叉熵和基于IoU的损失。此外，还对不同尺寸的图像块进行模型训练，并使用不同强度的数据增强技术。我们发现，在适度增强的数据上训练的二元模型表现最佳。

    Tool wear monitoring is crucial for quality control and cost reduction in manufacturing processes, of which drilling applications are one example. In this paper, we present a U-Net based semantic image segmentation pipeline, deployed on microscopy images of cutting inserts, for the purpose of wear detection. The wear area is differentiated in two different types, resulting in a multiclass classification problem. Joining the two wear types in one general wear class, on the other hand, allows the problem to be formulated as a binary classification task. Apart from the comparison of the binary and multiclass problem, also different loss functions, i. e., Cross Entropy, Focal Cross Entropy, and a loss based on the Intersection over Union (IoU), are investigated. Furthermore, models are trained on image tiles of different sizes, and augmentation techniques of varying intensities are deployed. We find, that the best performing models are binary models, trained on data with moderate augmentat
    
[^185]: 具有同时调整尺度的健壮方差正则化风险最小化

    Robust variance-regularized risk minimization with concomitant scaling

    [https://arxiv.org/abs/2301.11584](https://arxiv.org/abs/2301.11584)

    本研究提出了一种简单但有效的学习过程，用于最小化潜在存在重尾风险的损失函数，该方法在各种数据集上表现出与使用CVaR或DRO风险等替代标准得到的最佳候选方案相当或更好的性能。

    

    在潜在存在重尾风险的损失函数下，我们考虑了最小化损失均值和标准差之和的任务，而无需精确估计方差。通过修改一种无方差健壮均值估计技术以适应我们的问题设定，我们推导出一个简单的学习过程，可以与标准的基于梯度的求解器轻松结合，用于传统的机器学习工作流程中。经验上，我们验证了我们提出的方法，尽管简单，但在各种数据集上表现出与使用CVaR或DRO风险等替代标准得到的最佳候选方案相当或更好的性能。

    Under losses which are potentially heavy-tailed, we consider the task of minimizing sums of the loss mean and standard deviation, without trying to accurately estimate the variance. By modifying a technique for variance-free robust mean estimation to fit our problem setting, we derive a simple learning procedure which can be easily combined with standard gradient-based solvers to be used in traditional machine learning workflows. Empirically, we verify that our proposed approach, despite its simplicity, performs as well or better than even the best-performing candidates derived from alternative criteria such as CVaR or DRO risks on a variety of datasets.
    
[^186]: 分类器边界的结构：朴素贝叶斯分类器的案例研究

    Structure of Classifier Boundaries: Case Study for a Naive Bayes Classifier

    [https://arxiv.org/abs/2212.04382](https://arxiv.org/abs/2212.04382)

    本文研究了在图形输入空间中，分类器边界的结构。通过创建一种新的不确定性度量，称为邻居相似度，我们展示了朴素贝叶斯分类器的边界是巨大且复杂的结构。

    

    无论基于模型、训练数据还是二者组合，分类器将（可能复杂的）输入数据归入相对较少的输出类别之一。本文研究在输入空间为图的情况下，边界的结构——那些被分类为不同类别的邻近点——的特性。我们的科学背景是基于模型的朴素贝叶斯分类器，用于处理由下一代测序仪生成的DNA读数。我们展示了边界既是巨大的，又具有复杂的结构。我们创建了一种新的不确定性度量，称为邻居相似度，它将一个点的结果与其邻居的结果分布进行比较。这个度量不仅追踪了贝叶斯分类器的两个固有不确定性度量，还可以在没有固有不确定性度量的分类器上实现，但需要计算成本。

    Whether based on models, training data or a combination, classifiers place (possibly complex) input data into one of a relatively small number of output categories. In this paper, we study the structure of the boundary--those points for which a neighbor is classified differently--in the context of an input space that is a graph, so that there is a concept of neighboring inputs, The scientific setting is a model-based naive Bayes classifier for DNA reads produced by Next Generation Sequencers. We show that the boundary is both large and complicated in structure. We create a new measure of uncertainty, called Neighbor Similarity, that compares the result for a point to the distribution of results for its neighbors. This measure not only tracks two inherent uncertainty measures for the Bayes classifier, but also can be implemented, at a computational cost, for classifiers without inherent measures of uncertainty.
    
[^187]: 强化学习中的局部约束表示

    Locally Constrained Representations in Reinforcement Learning

    [https://arxiv.org/abs/2209.09441](https://arxiv.org/abs/2209.09441)

    本论文提出了一种在强化学习中使用局部约束表示的方法，通过辅助损失函数迫使状态表示与相邻状态的表示具有一定的可预测性，以更好地捕捉到环境的局部变化情况。

    

    强化学习的成功很大程度上依赖于从环境观测数据中学习到稳健的表示。在大多数情况下，纯粹通过强化学习损失函数学习到的表示在不同状态下可能差异巨大，这取决于值函数的变化。然而，学习到的表示并不一定需要与当前任务非常相关。仅依赖强化学习目标可能会导致表示在连续的时间步长中差异很大。此外，由于强化学习损失函数有一个变化的目标，学习到的表示将取决于当前值/策略的好坏。因此，将表示与主要任务解耦可以使其不仅关注于任务特定特征，还关注环境动态特征。为此，我们提出了局部约束表示，其中辅助损失函数迫使状态表示能够由相邻状态的表示进行预测。这鼓励表示更好地捕捉到环境的局部变化。

    The success of Reinforcement Learning (RL) heavily relies on the ability to learn robust representations from the observations of the environment. In most cases, the representations learned purely by the reinforcement learning loss can differ vastly across states depending on how the value functions change. However, the representations learned need not be very specific to the task at hand. Relying only on the RL objective may yield representations that vary greatly across successive time steps. In addition, since the RL loss has a changing target, the representations learned would depend on how good the current values/policies are. Thus, disentangling the representations from the main task would allow them to focus not only on the task-specific features but also the environment dynamics. To this end, we propose locally constrained representations, where an auxiliary loss forces the state representations to be predictable by the representations of the neighboring states. This encourages
    
[^188]: 统计对流形假设的探索

    Statistical exploration of the Manifold Hypothesis

    [https://arxiv.org/abs/2208.11665](https://arxiv.org/abs/2208.11665)

    这篇论文通过潜在度量模型从数据中得出了丰富而复杂的流形结构，并提供了解释流形假设的统计解释。该研究为发现和解释高维数据的几何结构以及探索数据生成机制提供了方法。

    

    流形假设是机器学习中广为接受的理论，它认为名义上的高维数据实际上集中在高维空间中的低维流形中。这种现象在许多真实世界的情况中经验性地观察到，在过去几十年中已经导致了多种统计方法的发展，并被认为是现代人工智能技术成功的关键因素。我们表明，通过潜在度量模型这种通用且非常简单的统计模型，可以从数据中生成丰富而有时复杂的流形结构，通过潜变量、相关性和平稳性等基本概念。这为为什么流形假设在这么多情况下似乎成立提供了一个一般的统计解释。在潜在度量模型的基础上，我们提出了发现和解释高维数据几何结构以及探索数据生成机制的程序。

    The Manifold Hypothesis is a widely accepted tenet of Machine Learning which asserts that nominally high-dimensional data are in fact concentrated near a low-dimensional manifold, embedded in high-dimensional space. This phenomenon is observed empirically in many real world situations, has led to development of a wide range of statistical methods in the last few decades, and has been suggested as a key factor in the success of modern AI technologies. We show that rich and sometimes intricate manifold structure in data can emerge from a generic and remarkably simple statistical model -- the Latent Metric Model -- via elementary concepts such as latent variables, correlation and stationarity. This establishes a general statistical explanation for why the Manifold Hypothesis seems to hold in so many situations. Informed by the Latent Metric Model we derive procedures to discover and interpret the geometry of high-dimensional data, and explore hypotheses about the data generating mechanism
    
[^189]: 基于Rademacher复杂度的深度学习一般化界限研究

    On Rademacher Complexity-based Generalization Bounds for Deep Learning

    [https://arxiv.org/abs/2208.04284](https://arxiv.org/abs/2208.04284)

    该论文研究了基于Rademacher复杂度的方法在对卷积神经网络进行少类别图像分类时生成非空泛化界限。其中的关键技术贡献是发展了针对函数空间和具有一般Lipschitz激活函数的CNNs的新的Talagrand压缩引理。

    

    我们展示了基于Rademacher复杂度的方法可以生成对卷积神经网络（CNNs）进行分类少量类别图像非空泛化界限。新的Talagrand压缩引理的发展对于高维映射函数空间和具有一般Lipschitz激活函数的CNNs是一个关键技术贡献。我们的结果表明，Rademacher复杂度不依赖于CNNs的网络长度，特别是对于诸如ReLU，Leaky ReLU，Parametric Rectifier Linear Unit，Sigmoid和Tanh等特定类型的激活函数。

    We show that the Rademacher complexity-based approach can generate non-vacuous generalisation bounds on Convolutional Neural Networks (CNNs) for classifying a small number of classes of images. The development of new Talagrand's contraction lemmas for high-dimensional mappings between function spaces and CNNs for general Lipschitz activation functions is a key technical contribution. Our results show that the Rademacher complexity does not depend on the network length for CNNs with some special types of activation functions such as ReLU, Leaky ReLU, Parametric Rectifier Linear Unit, Sigmoid, and Tanh.
    
[^190]: 无参镜像下降

    Parameter-free Mirror Descent

    [https://arxiv.org/abs/2203.00444](https://arxiv.org/abs/2203.00444)

    本论文针对无界域中构建自适应和无参的算法提出了一种修改后的在线镜像下降框架，并以此为基础开发了具有最优动态遗憾界限的无约束在线线性优化算法，并证明了基于Follow-the-Regularized-Leader的策略无法达到类似效果，此外还应用镜像下降框架构建了新的无参隐式更新以及简化和改进的无约束无标度算法。

    

    我们提出了一种修改后的在线镜像下降框架，适用于在无界域中构建自适应和无参的算法。我们利用这种技术开发了第一个无约束在线线性优化算法，实现了最优的动态遗憾界限，并进一步证明基于Follow-the-Regularized-Leader的自然策略无法达到类似的结果。我们还将我们的镜像下降框架应用于构建无参隐式更新，以及一个简化和改进的无约束无标度算法。

    We develop a modified online mirror descent framework that is suitable for building adaptive and parameter-free algorithms in unbounded domains. We leverage this technique to develop the first unconstrained online linear optimization algorithm achieving an optimal dynamic regret bound, and we further demonstrate that natural strategies based on Follow-the-Regularized-Leader are unable to achieve similar results. We also apply our mirror descent framework to build new parameter-free implicit updates, as well as a simplified and improved unconstrained scale-free algorithm.
    
[^191]: 从生物纠错码到容错神经网络

    Fault-Tolerant Neural Networks from Biological Error Correction Codes

    [https://arxiv.org/abs/2202.12887](https://arxiv.org/abs/2202.12887)

    该论文根据哺乳动物皮质中的模拟纠错码，提出了一种基于生物纠错码的通用容错神经网络，实现了可靠计算；发现了从故障到容错神经计算的相变，为理解嘈杂模拟系统提供了新的途径。

    

    在深度学习中，是否可能实现容错计算一直是一个悬而未决的问题：是否可以仅使用不可靠的神经元实现任意可靠的计算？在哺乳动物皮质的网格细胞中，观察到了模拟纠错码保护状态免受神经射频噪声的现象，但其在信息处理中的作用尚不清楚。在本研究中，我们利用这些生物纠错码，开发了一种通用的容错神经网络，如果每个神经元的故障性都低于一个严格的阈值，则能够实现可靠的计算；令人惊讶的是，我们发现嘈杂的生物神经元低于这个阈值。从故障到容错神经计算的相变的发现，揭示了皮质中可靠计算的机制，为理解与人工智能和神经形态计算有关的嘈杂模拟系统打开了一条道路。

    It has been an open question in deep learning if fault-tolerant computation is possible: can arbitrarily reliable computation be achieved using only unreliable neurons? In the grid cells of the mammalian cortex, analog error correction codes have been observed to protect states against neural spiking noise, but their role in information processing is unclear. Here, we use these biological error correction codes to develop a universal fault-tolerant neural network that achieves reliable computation if the faultiness of each neuron lies below a sharp threshold; remarkably, we find that noisy biological neurons fall below this threshold. The discovery of a phase transition from faulty to fault-tolerant neural computation suggests a mechanism for reliable computation in the cortex and opens a path towards understanding noisy analog systems relevant to artificial intelligence and neuromorphic computing.
    
[^192]: 更为泛化的恶意URL检测模型的探索

    Toward More Generalized Malicious URL Detection Models

    [https://arxiv.org/abs/2202.10027](https://arxiv.org/abs/2202.10027)

    本文揭示了一个可能严重影响恶意URL检测机器学习模型性能的数据偏差问题，并提出了一种去偏置训练策略，通过自监督对抗训练技术来改善基于深度神经网络的模型的泛化能力。

    

    本文揭示了一个可能严重影响恶意URL检测机器学习模型性能的数据偏差问题。我们描述了如何使用可解释的机器学习技术来识别这种偏差，并进一步认为这样的偏差在真实世界的安全数据中自然存在，用于训练分类模型。然后，我们提出了一种能够应用于大多数基于深度学习的模型的去偏置训练策略，以减轻因特征偏差而产生的负面影响。该解决方案基于自监督对抗训练技术，用于训练深度神经网络从偏差数据中学习不变特征。我们进行了一系列实验，证明了该策略可以显著改善基于CNN和RNN的检测模型的泛化能力。

    This paper reveals a data bias issue that can severely affect the performance while conducting a machine learning model for malicious URL detection. We describe how such bias can be identified using interpretable machine learning techniques, and further argue that such biases naturally exist in the real world security data for training a classification model. We then propose a debiased training strategy that can be applied to most deep-learning based models to alleviate the negative effects from the biased features. The solution is based on the technique of self-supervised adversarial training to train deep neural networks learning invariant embedding from biased data. We conduct a wide range of experiments to demonstrate that the proposed strategy can lead to significantly better generalization capability for both CNN-based and RNN-based detection models.
    
[^193]: 深度残差神经网络通过非线性控制理论实现通用逼近能力

    Universal Approximation Power of Deep Residual Neural Networks via Nonlinear Control Theory

    [https://arxiv.org/abs/2007.06007](https://arxiv.org/abs/2007.06007)

    本文通过非线性控制理论解释了深度残差神经网络的通用逼近能力，并提供了一个充分条件，在激活函数满足二次微分方程的情况下，一个足够深的神经网络能够在紧集合上逼近任意连续函数。

    

    本文通过几何非线性控制来解释深度残差神经网络的通用逼近能力。受到最近建立残差网络和控制系统之间联系的工作的启发，我们提供了一个一般的充分条件，要求激活函数或其导数之一满足一个二次微分方程，以使残差网络具有通用逼近能力。在实践中使用的许多激活函数满足这个假设，我们证明这个属性足以让一个足够深的具有$n+1$神经元每层的神经网络，在紧集合上相对于最大范数逼近任意连续的从$\mathbb{R}^n$到$\mathbb{R}^n$的函数。我们进一步展示了这个结果适用于非常简单的架构，只需要权重取两个值。第一个关键技术贡献是将通用逼近与残差神经网络的几何非线性控制相联系。

    In this paper, we explain the universal approximation capabilities of deep residual neural networks through geometric nonlinear control. Inspired by recent work establishing links between residual networks and control systems, we provide a general sufficient condition for a residual network to have the power of universal approximation by asking the activation function, or one of its derivatives, to satisfy a quadratic differential equation. Many activation functions used in practice satisfy this assumption, exactly or approximately, and we show this property to be sufficient for an adequately deep neural network with $n+1$ neurons per layer to approximate arbitrarily well, on a compact set and with respect to the supremum norm, any continuous function from $\mathbb{R}^n$ to $\mathbb{R}^n$. We further show this result to hold for very simple architectures for which the weights only need to assume two values. The first key technical contribution consists of relating the universal approxi
    
[^194]: 同时进行正数据-无标签学习和有条件生成，利用额外数据来分类和生成

    Classify and Generate Reciprocally: Simultaneous Positive-Unlabelled Learning and Conditional Generation with Extra Data

    [https://arxiv.org/abs/2006.07841](https://arxiv.org/abs/2006.07841)

    本论文提出了一种同时利用正数据-无标签学习和有条件生成的训练框架，以及额外无标签数据的方法。通过使用一个对噪声标签具有鲁棒性的分类器噪声不变有条件生成对抗网络来提高PU分类器的性能，并利用PU分类器预测的标签和额外数据来帮助生成。实验结果证明了该方法的有效性。

    

    在许多机器学习问题中，标记类别数据的稀缺性是一个普遍存在的瓶颈。虽然存在丰富的无标签数据并提供潜在的解决方案，但利用它们是非常具有挑战性的。本文通过同时利用正数据-无标签（Positive-Unlabeled，PU）分类和有条件生成，以及额外的无标签数据，解决了这个问题。特别地，我们提出了一个新的训练框架，使得在面对额外数据（尤其是分布外的无标签数据）时，同时进行PU分类和有条件生成成为可能，通过探索它们之间的相互作用：1）通过一个对噪声标签具有鲁棒性的新型分类器噪声不变有条件生成对抗网络（Classifier-Noise-Invariant Conditional GAN，CNI-CGAN）来提高PU分类器的性能，2）利用PU分类器预测的标签和额外数据来帮助生成。从理论上，我们证明了CNI-CGAN的最优条件，并在实验中通过广泛的评估来验证了我们的方法。

    The scarcity of class-labeled data is a ubiquitous bottleneck in many machine learning problems. While abundant unlabeled data typically exist and provide a potential solution, it is highly challenging to exploit them. In this paper, we address this problem by leveraging Positive-Unlabeled~(PU) classification and the conditional generation with extra unlabeled data \emph{simultaneously}. In particular, we present a novel training framework to jointly target both PU classification and conditional generation when exposed to extra data, especially out-of-distribution unlabeled data, by exploring the interplay between them: 1) enhancing the performance of PU classifiers with the assistance of a novel Classifier-Noise-Invariant Conditional GAN~(CNI-CGAN) that is robust to noisy labels, 2) leveraging extra data with predicted labels from a PU classifier to help the generation. Theoretically, we prove the optimal condition of CNI-CGAN, and experimentally, we conducted extensive evaluations on
    
[^195]: 具有合成对照组的自适应实验设计

    Adaptive Experiment Design with Synthetic Controls. (arXiv:2401.17205v1 [stat.ML])

    [http://arxiv.org/abs/2401.17205](http://arxiv.org/abs/2401.17205)

    这种方法提出了Syntax，一个具有合成对照组的自适应实验设计，能够在多个亚群体中识别出具有正面治疗效果的亚群体，对于多样化患者反应的临床试验具有样本效率的优势。

    

    临床试验通常用于了解新治疗对给定患者群体的影响。然而，大规模群体中的患者很少以相同的方式对待相同的治疗做出反应。患者反应的多样性需要进行多个亚群体的效果研究 - 尤其是当治疗对整体群体没有或几乎没有益处，而对特定亚群体可能具有显著的益处时。基于这种需求，我们提出了Syntax，一种探索性试验设计，在众多亚群体中识别具有正面治疗效果的亚群体。Syntax具有样本效率，因为它(i) 自适应招募和分配患者，(ii) 通过合成对照组形成每个亚群体的控制样本，从而估计治疗效果。我们通过实验证实了Syntax的性能，并提供了关于它何时可能优于传统试验设计的见解。

    Clinical trials are typically run in order to understand the effects of a new treatment on a given population of patients. However, patients in large populations rarely respond the same way to the same treatment. This heterogeneity in patient responses necessitates trials that investigate effects on multiple subpopulations - especially when a treatment has marginal or no benefit for the overall population but might have significant benefit for a particular subpopulation. Motivated by this need, we propose Syntax, an exploratory trial design that identifies subpopulations with positive treatment effect among many subpopulations. Syntax is sample efficient as it (i) recruits and allocates patients adaptively and (ii) estimates treatment effects by forming synthetic controls for each subpopulation that combines control samples from other subpopulations. We validate the performance of Syntax and provide insights into when it might have an advantage over conventional trial designs through e
    
[^196]: SliceGPT: 通过删除行和列来压缩大型语言模型

    SliceGPT: Compress Large Language Models by Deleting Rows and Columns. (arXiv:2401.15024v1 [cs.LG])

    [http://arxiv.org/abs/2401.15024](http://arxiv.org/abs/2401.15024)

    SliceGPT是一种新的事后训练稀疏化方案，通过将每个权重矩阵替换为较小的矩阵以减小网络的维度，可以在保持高任务性能的同时减少模型参数。

    

    大型语言模型已成为自然语言处理的基石，但使用它们需要大量计算和内存资源。稀疏化方法可以缓解这些资源限制，并且最近的研究表明训练好的模型可以进行事后的稀疏化处理。现有的稀疏化技术面临着挑战，因为它们需要额外的数据结构，并且在当前硬件上速度受限。在本文中，我们提出了一种新的事后训练稀疏化方案SliceGPT，该方案用较小的（稠密的）矩阵替换每个权重矩阵，从而减小网络的嵌入维度。通过大量的实验，我们展示了SliceGPT在保持相应稠密模型的99%、99%和90%的零-shot任务性能的同时，可以移除LLAMA2-70B、OPT 66B和Phi-2模型中多达25%的模型参数（包括嵌入）。我们的切片模型在较少的GPU上运行并且更快，无需额外的代码优化。

    Large language models have become the cornerstone of natural language processing, but their use comes with substantial costs in terms of compute and memory resources. Sparsification provides a solution to alleviate these resource constraints, and recent works have shown that trained models can be sparsified post-hoc. Existing sparsification techniques face challenges as they need additional data structures and offer constrained speedup with current hardware. In this paper we present SliceGPT, a new post-training sparsification scheme which replaces each weight matrix with a smaller (dense) matrix, reducing the embedding dimension of the network. Through extensive experimentation, we show that SliceGPT can remove up to 25% of the model parameters (including embeddings) for LLAMA2-70B, OPT 66B and Phi-2 models while maintaining 99%, 99% and 90% zero-shot task performance of the dense model respectively. Our sliced models run on fewer GPUs and run faster without any additional code optimi
    
[^197]: 提示设计与工程：介绍与高级方法

    Prompt Design and Engineering: Introduction and Advanced Methods. (arXiv:2401.14423v1 [cs.SE])

    [http://arxiv.org/abs/2401.14423](http://arxiv.org/abs/2401.14423)

    本文介绍了提示设计与工程的主要概念，并回顾了基本和更高级的方法。

    

    提示设计与工程在过去几个月中成为了一个重要的学科。在本文中，我们介绍了主要概念，并回顾了提示设计与工程的基本和更高级的方法。

    Prompt design and engineering has become an important discipline in just the past few months. In this paper, we provide an introduction to the main concepts as well as review basic and more advanced approaches to prompt design and engineering.
    
[^198]: DALex: 通过多样聚合实现类似词法选择的选择算法

    DALex: Lexicase-like Selection via Diverse Aggregation. (arXiv:2401.12424v1 [cs.NE])

    [http://arxiv.org/abs/2401.12424](http://arxiv.org/abs/2401.12424)

    本文提出了一种新的选择算法DALex，它通过加权训练案例误差的和来选择最佳个体，相比标准的词法选择更快速。

    

    在进化计算和机器学习的多个领域中，词法选择被证明相比其他选择算法具有优势。词法选择在其标准形式下，根据随机顺序的训练案例进行逐一考虑，并基于此过程对种群或其他集合进行筛选。然而，这个逐步筛选的过程可能会耗时，尤其是在具有大量训练案例的情况下。本文提出了一种新的方法DALex（即多样聚合词法选择），该方法在选择个体方面与词法选择几乎等效，但速度更快。DALex方法根据加权训练案例误差的和选择最佳个体，其中权重是随机采样的。这使得我们可以将选择所需的核心计算形式化为矩阵乘法，而不是递归循环比较，从而可以利用优化和并行化的计算。

    Lexicase selection has been shown to provide advantages over other selection algorithms in several areas of evolutionary computation and machine learning. In its standard form, lexicase selection filters a population or other collection based on randomly ordered training cases that are considered one at a time. This iterated filtering process can be time-consuming, particularly in settings with large numbers of training cases. In this paper, we propose a new method that is nearly equivalent to lexicase selection in terms of the individuals that it selects, but which does so significantly more quickly. The new method, called DALex (for Diversely Aggregated Lexicase), selects the best individual with respect to a weighted sum of training case errors, where the weights are randomly sampled. This allows us to formulate the core computation required for selection as matrix multiplication instead of recursive loops of comparisons, which in turn allows us to take advantage of optimized and pa
    
[^199]: 数据驱动的目标定位: 使用Cramér-Rao界限对梯度下降进行基准测试

    Data-Driven Target Localization: Benchmarking Gradient Descent Using the Cram\'er-Rao Bound. (arXiv:2401.11176v1 [eess.SP])

    [http://arxiv.org/abs/2401.11176](http://arxiv.org/abs/2401.11176)

    本研究提出了一种数据驱动的神经网络方法，通过降低均方误差（MSE）实现了改进的目标方位和速度估计。这一发现强调了在雷达系统中采用深度学习方法的潜力，为在杂乱和动态环境中更准确的定位铺平了道路。

    

    在现代雷达系统中，使用方位和速度估计进行精确的目标定位至关重要。传统的无偏估计方法利用梯度下降算法达到Cramér-Rao界限（CRB）的理论极限，用于参数估计的误差。在本研究中，我们提出了一种数据驱动的神经网络方法，优于这些传统技术，在目标方位和速度估计方面表现出更高的准确性。使用代表性的模拟场景，我们展示了我们提出的神经网络模型始终实现了改进的参数估计，这是由于其固有的偏见性，从而得到了减小的均方误差（MSE）。我们的发现强调了在雷达系统中采用深度学习方法的潜力，为在杂乱和动态环境中更准确的定位铺平了道路。

    In modern radar systems, precise target localization using azimuth and velocity estimation is paramount. Traditional unbiased estimation methods have leveraged gradient descent algorithms to reach the theoretical limits of the Cram\'er Rao Bound (CRB) for the error of the parameter estimates. In this study, we present a data-driven neural network approach that outperforms these traditional techniques, demonstrating improved accuracies in target azimuth and velocity estimation. Using a representative simulated scenario, we show that our proposed neural network model consistently achieves improved parameter estimates due to its inherently biased nature, yielding a diminished mean squared error (MSE). Our findings underscore the potential of employing deep learning methods in radar systems, paving the way for more accurate localization in cluttered and dynamic environments.
    
[^200]: Co-Pilot for Health: 个性化算法AI引导改善健康结果

    Co-Pilot for Health: Personalized Algorithmic AI Nudging to Improve Health Outcomes. (arXiv:2401.10816v1 [cs.HC])

    [http://arxiv.org/abs/2401.10816](http://arxiv.org/abs/2401.10816)

    该研究通过使用基于图神经网络的推荐系统和来自可穿戴设备的健康行为数据，设计并实施了一个人工智能驱动平台，实现了个性化和情境引导，能够提高参与者的日常活动水平和中等至剧烈运动时长。

    

    在全球范围内自动塑造大型人群的健康行为，跨可穿戴设备类型和疾病状况具有巨大的潜力来改善全球健康结果。我们设计并实施了一个基于图神经网络（GNN）推荐系统和来自可穿戴健身设备的精细健康行为数据的人工智能驱动平台，用于数字算法引导。在此我们描述了该平台在新加坡针对$n=84,764$个个体的12周期间进行个性化和情境引导的有效性结果。我们统计验证了目标组中接受此类AI优化日常引导的参与者相较于控制组中未接受任何引导的匹配参与者，其每天的步数增加了6.17%（$p = 3.09\times10^{-4}$），每周中等至剧烈运动（MVPA）分钟增加了7.61%（$p = 1.16\times10^{-2}$）。此外，此类引导非常可行。

    The ability to shape health behaviors of large populations automatically, across wearable types and disease conditions at scale has tremendous potential to improve global health outcomes. We designed and implemented an AI driven platform for digital algorithmic nudging, enabled by a Graph-Neural Network (GNN) based Recommendation System, and granular health behavior data from wearable fitness devices. Here we describe the efficacy results of this platform with its capabilities of personalized and contextual nudging to $n=84,764$ individuals over a 12-week period in Singapore. We statistically validated that participants in the target group who received such AI optimized daily nudges increased daily physical activity like step count by 6.17% ($p = 3.09\times10^{-4}$) and weekly minutes of Moderate to Vigorous Physical Activity (MVPA) by 7.61% ($p = 1.16\times10^{-2}$), compared to matched participants in control group who did not receive any nudges. Further, such nudges were very well r
    
[^201]: 高效领域适应下的隐私保护视觉Transformer的精调方法

    Efficient Fine-Tuning with Domain Adaptation for Privacy-Preserving Vision Transformer. (arXiv:2401.05126v1 [cs.CV])

    [http://arxiv.org/abs/2401.05126](http://arxiv.org/abs/2401.05126)

    本论文提出了一种高效的领域适应方法，用于训练和测试隐私保护的视觉Transformer模型，并避免了使用加密图像导致的性能下降。实验结果表明，在图像分类任务上，该方法在CIFAR-10和ImageNet数据集上表现出更高的准确度。

    

    我们提出了一种新颖的方法，用于使用视觉Transformer（ViT）进行隐私保护的深度神经网络（DNN）。该方法不仅可以用视觉保护的图像训练模型并进行测试，而且还可以避免使用加密图像导致的性能下降，而传统方法不能避免图像加密的影响。通过领域适应方法，可以高效地对使用加密图像的ViT进行精细调整。在实验中，该方法在CIFAR-10和ImageNet数据集上的图像分类任务中表现出优于传统方法的分类准确度。

    We propose a novel method for privacy-preserving deep neural networks (DNNs) with the Vision Transformer (ViT). The method allows us not only to train models and test with visually protected images but to also avoid the performance degradation caused from the use of encrypted images, whereas conventional methods cannot avoid the influence of image encryption. A domain adaptation method is used to efficiently fine-tune ViT with encrypted images. In experiments, the method is demonstrated to outperform conventional methods in an image classification task on the CIFAR-10 and ImageNet datasets in terms of classification accuracy.
    
[^202]: 无监督的测试时自适应：通过插入和播放变换器模块

    Unsupervised Test-Time Adaptation via Plug-and-Play Transformer Modules. (arXiv:2401.04130v1 [cs.LG])

    [http://arxiv.org/abs/2401.04130](http://arxiv.org/abs/2401.04130)

    这项工作介绍了PLUTO:一种插拔式模块化的测试时领域适应策略，通过预先训练一系列针对不同源领域的模块，有效地创建了一个"模块存储库"。采用无监督的测试时自适应方法，从存储库中选择稀疏的相关模块的子集，并创建选中模块的加权组合，实现了对新领域的自适应。

    

    参数高效调优(PET)方法，如LoRA、Adapter和Visual Prompt Tuning(VPT)，通过调整变换器模型中的小模块，在使适应新领域方面取得了成功。然而，在测试过程中遇到的领域数量可能非常大，数据通常是无标签的。因此，适应新领域是具有挑战性的，也不现实为每个这样的领域生成定制的调整模块。为了应对这些挑战，本文引入了PLUTO：一种插拔模块化的测试时领域适应策略。我们预训练了一系列模块，每个模块专为不同的源领域进行了专门设计，有效地创建了一个"模块存储库"。给定一个带有少样本无标签数据的目标域，我们引入了一种无监督的测试时自适应(TTA)方法，来(1)从库中选择出稀疏的相关模块的子集，并且(2)在不调整权重的情况下创建选中模块的加权组合。这种插拔式的特性使得它可===

    Parameter-efficient tuning (PET) methods such as LoRA, Adapter, and Visual Prompt Tuning (VPT) have found success in enabling adaptation to new domains by tuning small modules within a transformer model. However, the number of domains encountered during test time can be very large, and the data is usually unlabeled. Thus, adaptation to new domains is challenging; it is also impractical to generate customized tuned modules for each such domain. Toward addressing these challenges, this work introduces PLUTO: a Plug-and-pLay modUlar Test-time domain adaptatiOn strategy. We pre-train a large set of modules, each specialized for different source domains, effectively creating a ``module store''. Given a target domain with few-shot unlabeled data, we introduce an unsupervised test-time adaptation (TTA) method to (1) select a sparse subset of relevant modules from this store and (2) create a weighted combination of selected modules without tuning their weights. This plug-and-play nature enable
    
[^203]: AST-T5：面向代码生成和理解的结构感知预训练模型

    AST-T5: Structure-Aware Pretraining for Code Generation and Understanding. (arXiv:2401.03003v1 [cs.SE])

    [http://arxiv.org/abs/2401.03003](http://arxiv.org/abs/2401.03003)

    AST-T5是一种结构感知的预训练模型，通过利用抽象语法树（AST）来增强代码生成、转换和理解的能力。它优于其他同等大小的语言模型，并在代码到代码任务中表现出色。

    

    大型语言模型在代码相关任务中取得了显著进展，然而许多模型将代码视为简单序列，忽略了其结构化特性。我们引入了AST-T5，一种新颖的预训练范式，利用抽象语法树（AST）增强了代码生成、转换和理解。通过动态规划，我们的AST感知分割保留了代码结构，而AST感知跨度破坏目标使模型能够重建各种代码结构。与其他模型不同，AST-T5避免了复杂的程序分析或架构更改，因此可以与任何编码器-解码器Transformer无缝集成。评估结果显示，AST-T5在各种代码相关任务中始终优于同等大小的语言模型。结构感知使得AST-T5在代码到代码任务中特别强大，在Bugs2Fix任务的精确匹配得分上超过CodeT5 2个点，并在CodeXGLUE中的Java-C#转换任务的精确匹配得分上超过CodeT5 3个点。

    Large language models (LLMs) have made significant advancements in code-related tasks, yet many LLMs treat code as simple sequences, neglecting its structured nature. We introduce AST-T5, a novel pretraining paradigm that leverages the Abstract Syntax Tree (AST) for enhanced code generation, transpilation, and understanding. Using dynamic programming, our AST-Aware Segmentation retains code structure, while our AST-Aware Span Corruption objective equips the model to reconstruct various code structures. Unlike other models, AST-T5 avoids intricate program analyses or architectural changes, so it integrates seamlessly with any encoder-decoder Transformer. Evaluations show that AST-T5 consistently outperforms similar-sized LMs across various code-related tasks. Structure-awareness makes AST-T5 particularly powerful in code-to-code tasks, surpassing CodeT5 by 2 points in exact match score for the Bugs2Fix task and by 3 points in exact match score for Java-C# Transpilation in CodeXGLUE. Our
    
[^204]: 自监督学习用于少样本鸟鸣分类

    Self-Supervised Learning for Few-Shot Bird Sound Classification. (arXiv:2312.15824v3 [cs.SD] UPDATED)

    [http://arxiv.org/abs/2312.15824](http://arxiv.org/abs/2312.15824)

    本研究展示了自监督学习在鸟鸣分类中的应用，通过无需标注的方式，从音频录音中获得有意义的鸟鸣表示，并展示了这些表示在少样本学习中的泛化能力。此外，使用预训练的音频神经网络选择高鸟活跃窗口进行自监督学习可以显著提升学习表示的质量。

    

    音频中的自监督学习（SSL）在各个领域有着巨大的潜力，尤其是在存在大量无标签数据的情况下。这在生物声学领域尤为重要，生物学家经常从自然环境中收集大量的声音数据集。本研究证明了无需标注就能够从音频录音中获得有意义的鸟鸣表示的自监督学习的能力。我们的实验显示，这些学习到的表示能够在少样本学习（FSL）场景中泛化到新的鸟类。此外，我们还展示了使用预训练的音频神经网络选择高鸟活跃的窗口进行自监督学习可以显著提升学习表示的质量。

    Self-supervised learning (SSL) in audio holds significant potential across various domains, particularly in situations where abundant, unlabeled data is readily available at no cost. This is particularly pertinent in bioacoustics, where biologists routinely collect extensive sound datasets from the natural environment. In this study, we demonstrate that SSL is capable of acquiring meaningful representations of bird sounds from audio recordings without the need for annotations. Our experiments showcase that these learned representations exhibit the capacity to generalize to new bird species in few-shot learning (FSL) scenarios. Additionally, we show that selecting windows with high bird activation for self-supervised learning, using a pretrained audio neural network, significantly enhances the quality of the learned representations.
    
[^205]: FairWASP：快速和最优的公平Wasserstein预处理

    FairWASP: Fast and Optimal Fair Wasserstein Pre-processing. (arXiv:2311.00109v1 [cs.LG])

    [http://arxiv.org/abs/2311.00109](http://arxiv.org/abs/2311.00109)

    FairWASP是一种快速和最优的公平Wasserstein预处理方法，通过重新加权数据集来减少分类数据集中的不平等性，同时满足人口平等性准则。这种方法可以用于构建可以输入任何分类方法的数据集。

    

    近年来，机器学习方法的快速发展旨在减少不同子群体之间模型输出的不平等性。在许多情况下，训练数据可能会被不同用户在多个下游应用中使用，这意味着对训练数据本身进行干预可能是最有效的。在这项工作中，我们提出了一种新的预处理方法FairWASP，旨在减少分类数据集中的不平等性，而不会修改原始数据。FairWASP返回样本级权重，使重新加权的数据集最小化与原始数据集的Wasserstein距离，同时满足（经验版本的）人口平等性，这是一种常用的公平性准则。我们从理论上证明了整数权重的最优性，这意味着我们的方法可以等同地理解为复制或删除样本。因此，FairWASP可用于构建可以输入任何分类方法的数据集，而不仅仅是接受样本权重的方法。

    Recent years have seen a surge of machine learning approaches aimed at reducing disparities in model outputs across different subgroups. In many settings, training data may be used in multiple downstream applications by different users, which means it may be most effective to intervene on the training data itself. In this work, we present FairWASP, a novel pre-processing approach designed to reduce disparities in classification datasets without modifying the original data. FairWASP returns sample-level weights such that the reweighted dataset minimizes the Wasserstein distance to the original dataset while satisfying (an empirical version of) demographic parity, a popular fairness criterion. We show theoretically that integer weights are optimal, which means our method can be equivalently understood as duplicating or eliminating samples. FairWASP can therefore be used to construct datasets which can be fed into any classification method, not just methods which accept sample weights. Ou
    
[^206]: 重新审视苹果品尝的可学习性

    Revisiting the Learnability of Apple Tasting. (arXiv:2310.19064v1 [cs.LG])

    [http://arxiv.org/abs/2310.19064](http://arxiv.org/abs/2310.19064)

    该论文重新审视了苹果品尝的可学习性，从组合角度研究了在线可学习性。作者通过引入Effective width参数，紧密量化了在可实现设置中的极小期望错误，并在可实现设置中建立了极小期望错误数量的三分法。

    

    在在线二元分类中，学习者只有在预测为"1"时观察到真实标签。本文重新研究了这种经典的部分反馈设置，并从组合角度研究了在线可学习性。我们证明了在不可知设置下，Littlestone维度仍然是苹果品尝的紧密定量刻画，解决了\cite{helmbold2000apple}提出的一个悬而未决的问题。此外，我们给出了一个新的组合参数，称为有效宽度，紧密量化了在可实现设置中的极小期望错误。作为推论，我们使用有效宽度在可实现设置中建立了极小期望错误数量的三分法。特别地，我们证明了在可实现设置中，任何学习者在苹果品尝反馈下的期望错误数量只能是$\Theta(1), \Theta(\sqrt{T})$, 或 $\Theta(T)$。

    In online binary classification under \textit{apple tasting} feedback, the learner only observes the true label if it predicts "1". First studied by \cite{helmbold2000apple}, we revisit this classical partial-feedback setting and study online learnability from a combinatorial perspective. We show that the Littlestone dimension continues to prove a tight quantitative characterization of apple tasting in the agnostic setting, closing an open question posed by \cite{helmbold2000apple}. In addition, we give a new combinatorial parameter, called the Effective width, that tightly quantifies the minimax expected mistakes in the realizable setting. As a corollary, we use the Effective width to establish a \textit{trichotomy} of the minimax expected number of mistakes in the realizable setting. In particular, we show that in the realizable setting, the expected number of mistakes for any learner under apple tasting feedback can only be $\Theta(1), \Theta(\sqrt{T})$, or $\Theta(T)$.
    
[^207]: Davidsonian场景图：改进文本-图像生成的细粒度评估的可靠性

    Davidsonian Scene Graph: Improving Reliability in Fine-grained Evaluation for Text-Image Generation. (arXiv:2310.18235v1 [cs.CV])

    [http://arxiv.org/abs/2310.18235](http://arxiv.org/abs/2310.18235)

    本论文提出了Davidsonian场景图（DSG）的评估框架，解决了现有文本-图像生成模型评估中的可靠性挑战，包括QG问题的准确性和VQA答案的一致性。

    

    评估文本到图像模型一直是困难的。最近一种用于评估文本-图像忠实度的强大方法是基于QG/A（问题生成和回答），它使用预训练的基础模型自动生成一组问题和答案，并基于这些答案与基于提示的答案在视觉问题回答模型中提取的一致性对输出图像进行评分。这种评估自然上取决于底层QG和QA模型的质量。我们确定并解决了现有QG/A工作中的几个可靠性挑战：（a）QG问题应尊重提示（避免幻觉、重复和遗漏）和（b）VQA答案应一致（不会在图像中宣称没有摩托车，同时声称摩托车是蓝色）。我们通过Davidsonian场景图（DSG），这个受形式语义启发的实证评估框架，解决了这些问题。

    Evaluating text-to-image models is notoriously difficult. A strong recent approach for assessing text-image faithfulness is based on QG/A (question generation and answering), which uses pre-trained foundational models to automatically generate a set of questions and answers from the prompt, and output images are scored based on whether these answers extracted with a visual question answering model are consistent with the prompt-based answers. This kind of evaluation is naturally dependent on the quality of the underlying QG and QA models. We identify and address several reliability challenges in existing QG/A work: (a) QG questions should respect the prompt (avoiding hallucinations, duplications, and omissions) and (b) VQA answers should be consistent (not asserting that there is no motorcycle in an image while also claiming the motorcycle is blue). We address these issues with Davidsonian Scene Graph (DSG), an empirically grounded evaluation framework inspired by formal semantics. DSG
    
[^208]: StochGradAdam: 利用随机梯度抽样加速神经网络训练

    StochGradAdam: Accelerating Neural Networks Training with Stochastic Gradient Sampling. (arXiv:2310.17042v1 [cs.LG])

    [http://arxiv.org/abs/2310.17042](http://arxiv.org/abs/2310.17042)

    StochGradAdam是一种利用随机梯度抽样加速神经网络训练的优化器，通过选择性梯度考虑，能够稳定收敛，提升鲁棒训练。在图像分类和分割任务中表现优异。

    

    在深度学习优化领域中，本文介绍了StochGradAdam优化器，这是对广受赞誉的Adam算法的新颖改进。StochGradAdam的核心是其梯度抽样技术。该方法不仅确保稳定收敛，而且利用选择性梯度考虑的优势，通过减轻噪声或异常数据的影响和增强损失函数空间的探索，提升了鲁棒训练。在图像分类和分割任务中，StochGradAdam表现出优于传统Adam优化器的性能。通过在每次迭代中精心选择一部分梯度进行抽样，该优化器能够有效应对复杂模型的管理。本文从数学基础到偏差校正策略全面探讨了StochGradAdam的方法，展示了深度学习训练技术的可期进展。

    In the rapidly advancing domain of deep learning optimization, this paper unveils the StochGradAdam optimizer, a novel adaptation of the well-regarded Adam algorithm. Central to StochGradAdam is its gradient sampling technique. This method not only ensures stable convergence but also leverages the advantages of selective gradient consideration, fostering robust training by potentially mitigating the effects of noisy or outlier data and enhancing the exploration of the loss landscape for more dependable convergence. In both image classification and segmentation tasks, StochGradAdam has demonstrated superior performance compared to the traditional Adam optimizer. By judiciously sampling a subset of gradients at each iteration, the optimizer is optimized for managing intricate models. The paper provides a comprehensive exploration of StochGradAdam's methodology, from its mathematical foundations to bias correction strategies, heralding a promising advancement in deep learning training tec
    
[^209]: Redco:一个轻量级工具，可在任何GPU/TPUs上自动化分布式训练LLMs

    Redco: A Lightweight Tool to Automate Distributed Training of LLMs on Any GPU/TPUs. (arXiv:2310.16355v1 [cs.LG])

    [http://arxiv.org/abs/2310.16355](http://arxiv.org/abs/2310.16355)

    Redco是一个轻量级工具，旨在自动化分布式训练LLMs，并简化ML流程的开发。

    

    人工智能的最新进展主要归功于大型语言模型（LLMs）。然而，它们不断增长的内存需求给机器学习（ML）研究人员和工程师带来了挑战。解决这个问题需要开发人员将大型模型分区以分布在多个GPU或TPU上。这需要使用现有模型并行工具（如Megatron-LM、DeepSpeed和Alpa）进行相当的编码和复杂的配置工作。这些工具需要用户具备机器学习系统（MLSys）的专业知识，给LLM开发带来了瓶颈，特别是对于没有MLSys背景的开发人员。在这项工作中，我们提出了Redco，这是一个轻量级且用户友好的工具，旨在自动化LLMs的分布式训练和推理，以及简化ML流程的开发。Redco的设计强调了两个关键方面。首先，为了自动化模型并行，我们的研究确定了两个简单的规则，用于为任何GPU / TPU生成张量并行策略。

    The recent progress of AI can be largely attributed to large language models (LLMs). However, their escalating memory requirements introduce challenges for machine learning (ML) researchers and engineers. Addressing this requires developers to partition a large model to distribute it across multiple GPUs or TPUs. This necessitates considerable coding and intricate configuration efforts with existing model parallel tools, such as Megatron-LM, DeepSpeed, and Alpa. These tools require users' expertise in machine learning systems (MLSys), creating a bottleneck in LLM development, particularly for developers without MLSys background. In this work, we present Redco, a lightweight and user-friendly tool crafted to automate distributed training and inference for LLMs, as well as to simplify ML pipeline development. The design of Redco emphasizes two key aspects. Firstly, to automate model parallism, our study identifies two straightforward rules to generate tensor parallel strategies for any g
    
[^210]: 深度回溯对因果一致解释的反事实推理

    Deep Backtracking Counterfactuals for Causally Compliant Explanations. (arXiv:2310.07665v1 [cs.AI])

    [http://arxiv.org/abs/2310.07665](http://arxiv.org/abs/2310.07665)

    本研究提供了一种实用方法，用于在深度生成组件的结构因果模型中计算回溯反事实。通过在因果模型的结构化潜在空间中解决优化问题，我们的方法能够生成反事实，并且与其他方法相比具备了多功能、模块化和符合因果关系的特点。

    

    反事实推理可以通过回答在改变情况下会观察到什么来提供有价值的见解，条件是根据实际观察。虽然经典的介入式解释已经得到了广泛研究，回溯原则被提出作为一种保持所有因果定律完整性的替代哲学，但其研究较少。在本研究中，我们介绍了在由深度生成组件组成的结构因果模型中计算回溯反事实的实用方法。为此，我们对结构分配施加了条件，通过在因果模型的结构化潜在空间中解决一个可行的约束优化问题来生成反事实。我们的方法还可以与反事实解释领域的方法进行比较。与这些方法相比，我们的方法代表了一种多功能、模块化和遵守因果的替代方案。

    Counterfactuals can offer valuable insights by answering what would have been observed under altered circumstances, conditional on a factual observation. Whereas the classical interventional interpretation of counterfactuals has been studied extensively, backtracking constitutes a less studied alternative the backtracking principle has emerged as an alternative philosophy where all causal laws are kept intact. In the present work, we introduce a practical method for computing backtracking counterfactuals in structural causal models that consist of deep generative components. To this end, we impose conditions on the structural assignments that enable the generation of counterfactuals by solving a tractable constrained optimization problem in the structured latent space of a causal model. Our formulation also facilitates a comparison with methods in the field of counterfactual explanations. Compared to these, our method represents a versatile, modular and causally compliant alternative. 
    
[^211]: LLark: 一种用于音乐的多模态基础模型

    LLark: A Multimodal Foundation Model for Music. (arXiv:2310.07160v1 [cs.SD])

    [http://arxiv.org/abs/2310.07160](http://arxiv.org/abs/2310.07160)

    LLark是一个通过多模态架构实现音乐理解的模型，能够在零样本泛化上匹配或超出现有基准模型，在字幕生成和推理任务中与人类响应高度一致。

    

    音乐具有独特且复杂的结构，对于专业人士和现有的AI系统来说都具有挑战性，并相对于其他形式的音频呈现出独特的挑战。我们提出了LLark，一种针对音乐理解的指令调谐多模型模型。我们详细介绍了我们的数据集创建过程，其中包括增强多样化的开源音乐数据集的注释，并将其转换为统一的指令调谐格式。我们提出了一种多模态架构用于LLark，将预训练的音乐生成模型与预训练的语言模型相结合。在对三种类型的任务（音乐理解、字幕生成和推理）进行评估时，我们展示了我们的模型在音乐理解的零样本泛化上与现有基准模型相匹配或超出，并且在字幕生成和推理任务中人类与模型的响应显示出高度一致性。LLark完全是根据开源音乐数据和模型进行训练的，并且我们公开了我们的训练代码。

    Music has a unique and complex structure which is challenging for both expert humans and existing AI systems to understand, and presents unique challenges relative to other forms of audio. We present LLark, an instruction-tuned multimodal model for music understanding. We detail our process for dataset creation, which involves augmenting the annotations of diverse open-source music datasets and converting them to a unified instruction-tuning format. We propose a multimodal architecture for LLark, integrating a pretrained generative model for music with a pretrained language model. In evaluations on three types of tasks (music understanding, captioning, and reasoning), we show that our model matches or outperforms existing baselines in zero-shot generalization for music understanding, and that humans show a high degree of agreement with the model's responses in captioning and reasoning tasks. LLark is trained entirely from open-source music data and models, and we make our training code
    
[^212]: 通过同时学习面部标志检测、域分离和重建来提高面部动作单位检测的精度

    Boosting Facial Action Unit Detection Through Jointly Learning Facial Landmark Detection and Domain Separation and Reconstruction. (arXiv:2310.05207v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2310.05207](http://arxiv.org/abs/2310.05207)

    本文提出了一种新的面部动作单位（AU）检测框架，通过共享参数和引入多任务学习，在面部标志检测和AU域分离与重建之间实现了更好的性能。实验证明我们方法在野外AU检测方面优于现有方法。

    

    最近，如何将大量的在野非标记面部图像引入监督式面部动作单位（AU）检测框架中成为一个具有挑战性的问题。本文提出了一种新的AU检测框架，通过共享同构面部提取模块的参数，引入多任务学习，同时学习AU域分离和重建以及面部标志检测。另外，我们提出了一种基于对比学习的新特征对齐方案，通过简单的投影器和改进的对比损失添加了四个额外的中间监督器来促进特征重建的过程。在两个基准测试上的实验结果表明，我们在野外AU检测方面优于现有的方法。

    Recently how to introduce large amounts of unlabeled facial images in the wild into supervised Facial Action Unit (AU) detection frameworks has become a challenging problem. In this paper, we propose a new AU detection framework where multi-task learning is introduced to jointly learn AU domain separation and reconstruction and facial landmark detection by sharing the parameters of homostructural facial extraction modules. In addition, we propose a new feature alignment scheme based on contrastive learning by simple projectors and an improved contrastive loss, which adds four additional intermediate supervisors to promote the feature reconstruction process. Experimental results on two benchmarks demonstrate our superiority against the state-of-the-art methods for AU detection in the wild.
    
[^213]: 基于思维混合表示的大规模语言模型级联用于成本高效的推理

    Large Language Model Cascades with Mixture of Thoughts Representations for Cost-efficient Reasoning. (arXiv:2310.03094v1 [cs.CL])

    [http://arxiv.org/abs/2310.03094](http://arxiv.org/abs/2310.03094)

    本研究提出了一种基于思维混合表示的大规模语言模型级联方法，用于成本高效的推理。通过考虑更弱模型的答案一致性作为问题难度的信号，可以实现对问题的决策，从而节约使用更强模型的成本。

    

    大规模语言模型（LLM）如GPT-4在各种任务中展现出了非凡的性能，但是这种强大的性能通常伴随着使用付费API服务的高昂费用。本文的研究动机是为了研究构建LLM级联以节约使用LLM的成本，特别是用于进行推理（例如数学、因果推理）任务的成本。我们的级联管道遵循一个直观的思想，即简单的问题可以由一个更弱但更实惠的LLM来解决，而只有具有挑战性的问题才需要更强大、更昂贵的LLM。为了实现这种决策，我们考虑到更弱的LLM的“答案一致性”作为问题难度的信号，并提出了几种答案采样和一致性检查的方法，其中一种方法利用了两种思维表示（即连续思维和程序思维）的混合。通过在六个推理基准数据集上的实验，我们使用GPT-3.5-turbo和GPT-4作为较弱的模型，

    Large language models (LLMs) such as GPT-4 have exhibited remarkable performance in a variety of tasks, but this strong performance often comes with the high expense of using paid API services. In this paper, we are motivated to study building an LLM cascade to save the cost of using LLMs, particularly for performing reasoning (e.g., mathematical, causal) tasks. Our cascade pipeline follows the intuition that simpler questions can be addressed by a weaker but more affordable LLM, whereas only the challenging questions necessitate the stronger and more expensive LLM. To realize this decision-making, we consider the "answer consistency" of the weaker LLM as a signal of the question difficulty and propose several methods for the answer sampling and consistency checking, including one leveraging a mixture of two thought representations (i.e., Chain-of-Thought and Program-of-Thought). Through experiments on six reasoning benchmark datasets, with GPT-3.5-turbo and GPT-4 being the weaker and 
    
[^214]: Alphazero类似的树搜索可以指导大型语言模型的解码和训练

    Alphazero-like Tree-Search can Guide Large Language Model Decoding and Training. (arXiv:2309.17179v1 [cs.LG])

    [http://arxiv.org/abs/2309.17179](http://arxiv.org/abs/2309.17179)

    Alphazero类似的树搜索框架TS-LLM可以利用学习的价值函数指导大型语言模型的解码和训练，不仅适用于推理任务，还适用于其他任务，并且在不同大小的语言模型上具有普适性和可扩展性

    

    大型语言模型 (LLM) 通常采用采样或束搜索，结合 Chain-of-Thought (CoT) 等提示来提高推理和解码能力。最近的研究如 Tree-of-Thought (ToT) 和 Reasoning via Planning (RAP) 旨在通过利用树搜索算法来引导多步推理，来增强LLM的推理能力。这些方法主要关注LLM在推理过程中的推理能力，并且严重依赖人为设计的提示来激活LLM作为一个价值函数，缺乏普适性和可扩展性。为了解决这些限制，我们提出了一种AlphaZero类似的用于LLM的树搜索框架 (称为TS-LLM)，系统地说明了如何通过学习的价值函数利用树搜索来指导LLM的解码能力。TS-LLM在两个关键方面与众不同：(1)通过利用学习的价值函数，我们的方法可以普适地应用于除了推理之外的不同任务 (例如RLHF对齐)，以及任何大小的LLM，而不需要提示

    Large language models (LLMs) typically employ sampling or beam search, accompanied by prompts such as Chain-of-Thought (CoT), to boost reasoning and decoding ability. Recent work like Tree-of-Thought (ToT) and Reasoning via Planning (RAP) aim to augment the reasoning capabilities of LLMs by utilizing tree-search algorithms to guide multi-step reasoning. These methods mainly focus on LLMs' reasoning ability during inference and heavily rely on human-designed prompts to activate LLM as a value function, which lacks general applicability and scalability. To address these limitations, we present an AlphaZero-like tree-search framework for LLMs (termed TS-LLM), systematically illustrating how tree-search with a learned value function can guide LLMs' decoding ability. TS-LLM distinguishes itself in two key ways: (1) Leveraging a learned value function, our approach can be generally applied to different tasks beyond reasoning (such as RLHF alignment), and LLMs of any size, without prompting a
    
[^215]: 对于蛋白功能预测中Transformer模型内部运作的洞察

    Insights Into the Inner Workings of Transformer Models for Protein Function Prediction. (arXiv:2309.03631v1 [cs.LG])

    [http://arxiv.org/abs/2309.03631](http://arxiv.org/abs/2309.03631)

    本研究通过扩展可解释性人工智能方法，探索了Transformer模型在蛋白质功能预测中的内部运作，并成功识别出了与生物学和化学相关的序列部分，为蛋白质研究提供了重要线索。

    

    动机：我们探索了可解释性人工智能（XAI）如何帮助揭示神经网络用于蛋白质功能预测的内部运作，通过扩展广泛使用的XAI方法——集成梯度，使其能够检查调整为基因本体术语和酶委员会编号预测的Transformer模型内的潜在表示。结果：该方法使我们能够识别出变压器在序列中特别关注的氨基酸，并展示这些相关的序列部分反映了生物学和化学的预期，无论是在嵌入层还是模型内部。我们确定了变压器头与地面真实序列注释（例如，跨膜区域，活性位点）之间具有统计显著对应的归因图的变压器头，这在多个蛋白质中都有出现。代码可在https://github.com/markuswenzel/xai-proteins 上获取和实施。

    Motivation: We explored how explainable AI (XAI) can help to shed light into the inner workings of neural networks for protein function prediction, by extending the widely used XAI method of integrated gradients such that latent representations inside of transformer models, which were finetuned to Gene Ontology term and Enzyme Commission number prediction, can be inspected too. Results: The approach enabled us to identify amino acids in the sequences that the transformers pay particular attention to, and to show that these relevant sequence parts reflect expectations from biology and chemistry, both in the embedding layer and inside of the model, where we identified transformer heads with a statistically significant correspondence of attribution maps with ground truth sequence annotations (e.g., transmembrane regions, active sites) across many proteins. Availability and Implementation: Source code can be accessed at https://github.com/markuswenzel/xai-proteins .
    
[^216]: CktGNN：用于电子设计自动化的电路图神经网络

    CktGNN: Circuit Graph Neural Network for Electronic Design Automation. (arXiv:2308.16406v1 [cs.LG])

    [http://arxiv.org/abs/2308.16406](http://arxiv.org/abs/2308.16406)

    本文提出了一种名为CktGNN的电路图神经网络，通过识别电路的图形特性，同时自动化电路拓扑生成和器件尺寸调整。它使用两级GNN框架对电路图进行编码，并在设计效率上取得了显著的提升。

    

    由于巨大的设计空间和复杂的设计权衡，模拟电路的电子设计自动化一直是集成电路领域的一个长期挑战。在过去的几十年中，人们大多数关注于在给定电路拓扑的情况下自动调整晶体管尺寸。通过识别电路的图形特性，本文提出了一种电路图神经网络(CktGNN)，它基于编码器依赖的优化子程序，同时自动化电路拓扑生成和器件尺寸调整。特别是，CktGNN使用两级GNN框架（嵌套GNN）对电路图进行编码，其中电路表示为已知子图基础上的子图组合。通过这种方式，它通过减少子图数量来显著提高设计效率以执行消息传递。

    The electronic design automation of analog circuits has been a longstanding challenge in the integrated circuit field due to the huge design space and complex design trade-offs among circuit specifications. In the past decades, intensive research efforts have mostly been paid to automate the transistor sizing with a given circuit topology. By recognizing the graph nature of circuits, this paper presents a Circuit Graph Neural Network (CktGNN) that simultaneously automates the circuit topology generation and device sizing based on the encoder-dependent optimization subroutines. Particularly, CktGNN encodes circuit graphs using a two-level GNN framework (of nested GNN) where circuits are represented as combinations of subgraphs in a known subgraph basis. In this way, it significantly improves design efficiency by reducing the number of subgraphs to perform message passing. Nonetheless, another critical roadblock to advancing learning-assisted circuit design automation is a lack of public
    
[^217]: SciBench: 对大型语言模型评估大学水平的科学问题解决能力

    SciBench: Evaluating College-Level Scientific Problem-Solving Abilities of Large Language Models. (arXiv:2307.10635v1 [cs.CL])

    [http://arxiv.org/abs/2307.10635](http://arxiv.org/abs/2307.10635)

    这篇论文介绍了一个名为SciBench的基准套件，旨在对大型语言模型的大学水平科学问题解决能力进行评估。研究结果显示，当前的语言模型在提供复杂科学问题解决能力方面还有不足之处。

    

    最近大型语言模型(LLMs)的进展在许多数学基准上取得了显著的进步。然而，这些基准大多只包含初高中科目的问题，仅包含多项选择题，并且仅限于基本算术运算范围。为了解决这些问题，本文介绍了一个广泛的基准套件SciBench，旨在系统地检测复杂科学问题解决所需的推理能力。SciBench包含两个经过精心策划的数据集：一个开放集，包括从数学、化学和物理教科书中摘录的大学水平的科学问题，以及一个封闭集，包含来自计算机科学和数学本科考试的问题。基于这两个数据集，我们对两个代表性的LLM进行了深入的基准研究，并采用不同的提示策略。结果表明，当前的LLMs在提供复杂科学问题解决能力方面还存在不足之处。

    Recent advances in large language models (LLMs) have demonstrated notable progress on many mathematical benchmarks. However, most of these benchmarks only feature problems grounded in junior and senior high school subjects, contain only multiple-choice questions, and are confined to a limited scope of elementary arithmetic operations. To address these issues, this paper introduces an expansive benchmark suite SciBench that aims to systematically examine the reasoning capabilities required for complex scientific problem solving. SciBench contains two carefully curated datasets: an open set featuring a range of collegiate-level scientific problems drawn from mathematics, chemistry, and physics textbooks, and a closed set comprising problems from undergraduate-level exams in computer science and mathematics. Based on the two datasets, we conduct an in-depth benchmark study of two representative LLMs with various prompting strategies. The results reveal that current LLMs fall short of deli
    
[^218]: 解释性不是游戏。(arXiv:2307.07514v1 [cs.AI])

    Explainability is NOT a Game. (arXiv:2307.07514v1 [cs.AI])

    [http://arxiv.org/abs/2307.07514](http://arxiv.org/abs/2307.07514)

    Shapley values may provide misleading measures of relative feature importance in XAI, challenging their proposed uses in high-stakes application domains.

    

    可解释性人工智能（XAI）旨在帮助人类决策者理解复杂的机器学习（ML）模型。XAI的一个重要特征是通过使用Shapley值来理论上证明相对特征重要性的度量。本文在最近的研究基础上，提出一个简单的论证，说明Shapley值可能会给相对特征重要性提供误导，使其为预测中无关的特征分配更高的重要性，而对与预测有关的特征分配较低的重要性。这些结果的意义在于它们有效地挑战了相对特征重要性的多种提议用法，这些用法正在高风险应用领域快速增长。

    Explainable artificial intelligence (XAI) aims to help human decision-makers in understanding complex machine learning (ML) models. One of the hallmarks of XAI are measures of relative feature importance, which are theoretically justified through the use of Shapley values. This paper builds on recent work and offers a simple argument for why Shapley values can provide misleading measures of relative feature importance, by assigning more importance to features that are irrelevant for a prediction, and assigning less importance to features that are relevant for a prediction. The significance of these results is that they effectively challenge the many proposed uses of measures of relative feature importance in a fast-growing range of high-stakes application domains.
    
[^219]: 自扩展神经网络

    Self Expanding Neural Networks. (arXiv:2307.04526v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2307.04526](http://arxiv.org/abs/2307.04526)

    这项研究提出了一种自扩展神经网络的方法，通过自然梯度来自动增加神经网络的宽度和深度，以在训练损失降低的情况下提高性能，并在分类和回归问题中展示了其优势。

    

    神经网络的训练结果严重依赖于所选择的架构；即使只是对网络大小做微小修改，通常也需要重新开始训练过程。相比之下，我们以一个小的架构开始训练，只在问题需要时扩展其容量，并避免干扰先前的优化过程。因此，我们引入了一种基于自然梯度的方法，当这样做可能大幅降低假设收敛训练损失时，直观地扩展了神经网络的宽度和深度。我们证明了神经元添加的“速率”上界，并且给出了计算廉价的扩展评分的下界。我们在分类和回归问题中展示了自扩展神经网络的优势，包括那些合适的架构大小在先验上相当不确定的问题。

    The results of training a neural network are heavily dependent on the architecture chosen; and even a modification of only the size of the network, however small, typically involves restarting the training process. In contrast to this, we begin training with a small architecture, only increase its capacity as necessary for the problem, and avoid interfering with previous optimization while doing so. We thereby introduce a natural gradient based approach which intuitively expands both the width and depth of a neural network when this is likely to substantially reduce the hypothetical converged training loss. We prove an upper bound on the "rate" at which neurons are added, and a computationally cheap lower bound on the expansion score. We illustrate the benefits of such Self-Expanding Neural Networks in both classification and regression problems, including those where the appropriate architecture size is substantially uncertain a priori.
    
[^220]: 在有界损失下的在线学习游戏的组合特征化

    A Combinatorial Characterization of Online Learning Games with Bounded Losses. (arXiv:2307.03816v1 [cs.LG])

    [http://arxiv.org/abs/2307.03816](http://arxiv.org/abs/2307.03816)

    这项研究提出了一个新的尺度敏感的组合维度，称为顺序极小极大维度，并通过对有界损失的在线学习游戏进行研究，给出了对向量值回归和多标签分类的在线可学习性的紧密定量刻画。

    

    我们研究了假设类别对任意但有界损失函数的在线可学习性。我们提出了一个新的尺度敏感的组合维度，称为顺序极小极大维度，并证明它能够紧密定量地刻画在线可学习性。作为应用，我们给出了对两个常见学习场景的在线可学习性的首个定量刻画：向量值回归和多标签分类。

    We study the online learnability of hypothesis classes with respect to arbitrary, but bounded, loss functions. We give a new scale-sensitive combinatorial dimension, named the sequential Minimax dimension, and show that it gives a tight quantitative characterization of online learnability. As applications, we give the first quantitative characterization of online learnability for two natural learning settings: vector-valued regression and multilabel classification.
    
[^221]: 未知环境中的在线覆盖路径规划的端到端强化学习

    End-to-end Reinforcement Learning for Online Coverage Path Planning in Unknown Environments. (arXiv:2306.16978v1 [cs.RO])

    [http://arxiv.org/abs/2306.16978](http://arxiv.org/abs/2306.16978)

    本文提出了一种基于端到端强化学习的在线覆盖路径规划方法，能处理未知环境并结合全局地图和局部感知输入，同时考虑长期路径规划和短期障碍物检测。

    

    覆盖路径规划是寻找覆盖给定封闭区域整个自由空间的最短路径的问题，应用范围从机器人割草和吸尘到地雷清除和搜救任务。虽然离线方法可以为已知环境找到可证明完备且在某些情况下是最优的路径，但在在线场景下，环境事先未知，特别是在存在非静态障碍物的情况下，其价值有限。我们提出了一种基于连续状态和动作空间的端到端强化学习方法，用于处理未知环境的在线覆盖路径规划问题。我们从全局地图和局部感知输入构建观察空间，使代理能够规划长期路径，并同时对短期障碍物进行行动。为了考虑大规模环境，我们提出使用多尺度地图输入表示。此外，我们提出了一种新颖的总变差正则化方法以减少路径偏离问题。

    Coverage path planning is the problem of finding the shortest path that covers the entire free space of a given confined area, with applications ranging from robotic lawn mowing and vacuum cleaning, to demining and search-and-rescue tasks. While offline methods can find provably complete, and in some cases optimal, paths for known environments, their value is limited in online scenarios where the environment is not known beforehand, especially in the presence of non-static obstacles. We propose an end-to-end reinforcement learning-based approach in continuous state and action space, for the online coverage path planning problem that can handle unknown environments. We construct the observation space from both global maps and local sensory inputs, allowing the agent to plan a long-term path, and simultaneously act on short-term obstacle detections. To account for large-scale environments, we propose to use a multi-scale map input representation. Furthermore, we propose a novel total var
    
[^222]: 动态时间序列的发展预测在时间不变性和线性性的帮助下

    Forecasting of the development of a partially-observed dynamical time series with the aid of time-invariance and linearity. (arXiv:2306.16593v1 [stat.ME])

    [http://arxiv.org/abs/2306.16593](http://arxiv.org/abs/2306.16593)

    本研究提出了一种自回归松弛时间序列（ARS）模型，通过考虑动态系统的时间不变性和线性性，同时估计演化函数和缺失变量，用于预测动态时间序列中缺失变量的发展。

    

    动态系统产生一种依赖多元序列，称为动态时间序列，通过演化函数发展而来。由于当前时间点的动态时间序列变量通常依赖于前一个时间点的所有变量，现有研究通过估计演化函数来预测未来时间点的变量。然而，在某些实际情况下，动态时间序列中的一些变量是缺失的。本研究提出了一种自回归松弛时间序列（ARS）模型。ARS模型涉及演化函数和作为松弛时间序列的潜在缺失变量的同时估计，借助于动态系统的时间不变性和线性性。本研究实证了提出的ARS模型的有效性。

    A dynamical system produces a dependent multivariate sequence called dynamical time series, developed with an evolution function. As variables in the dynamical time series at the current time-point usually depend on the whole variables in the previous time-point, existing studies forecast the variables at the future time-point by estimating the evolution function. However, some variables in the dynamical time-series are missing in some practical situations. In this study, we propose an autoregressive with slack time series (ARS) model. ARS model involves the simultaneous estimation of the evolution function and the underlying missing variables as a slack time series, with the aid of the time-invariance and linearity of the dynamical system. This study empirically demonstrates the effectiveness of the proposed ARS model.
    
[^223]: 生成对抗网络中真实数据和生成数据统计的概率匹配

    Probabilistic matching of real and generated data statistics in generative adversarial networks. (arXiv:2306.10943v1 [stat.ML])

    [http://arxiv.org/abs/2306.10943](http://arxiv.org/abs/2306.10943)

    本文提出一种通过向生成器损失函数中添加KL散度项的方法，来保证生成数据统计分布与真实数据的相应分布重合，并在实验中展示了此方法的优越性能。

    

    生成对抗网络是一种强大的生成建模方法。虽然生成样本往往难以区分真实数据，但不能保证它们遵循真实数据分布。本文提出了一种方法，确保某些生成数据统计分布与真实数据的相应分布重合。为此，我们在生成器损失函数中添加了Kullback-Leibler项：KL散度是在每次迭代中从小批量值获得的相应生成分布和由条件能量模型表示的真实分布之间的差异。我们在一个合成数据集和两个实际数据集上评估了该方法，并展示了我们方法的优越性能。

    Generative adversarial networks constitute a powerful approach to generative modeling. While generated samples often are indistinguishable from real data, there is no guarantee that they will follow the true data distribution. In this work, we propose a method to ensure that the distributions of certain generated data statistics coincide with the respective distributions of the real data. In order to achieve this, we add a Kullback-Leibler term to the generator loss function: the KL divergence is taken between the true distributions as represented by a conditional energy-based model, and the corresponding generated distributions obtained from minibatch values at each iteration. We evaluate the method on a synthetic dataset and two real-world datasets and demonstrate improved performance of our method.
    
[^224]: 观测中的样本高效策略模仿学习

    Sample-Efficient On-Policy Imitation Learning from Observations. (arXiv:2306.09805v1 [cs.LG])

    [http://arxiv.org/abs/2306.09805](http://arxiv.org/abs/2306.09805)

    提出了一种称为SEILO的算法，该算法结合了标准的对抗模仿学习和逆动力学建模，实现了从无专家数据的观测中的样本高效策略模仿学习，成功地减少了与环境的交互并实现了专家水平的表现。

    

    通过使用专家演示，模仿学习 (ILD) 旨在通过消除强化学习的许多缺点来帮助学习输出更好的策略。然而，在大多数真实世界的应用中，缺乏专家行动指导，因此无法使用ILD。相反，我们考虑观测中的模仿学习 (ILO)，其中没有提供专家动作，使其成为更具挑战性的问题。现有方法通常使用策略学习，这是众所周知的成本昂贵的。本文提出了 SEILO，一种新颖的样本高效策略算法，用于 ILO，将标准的对抗模仿学习与逆动力学建模相结合。这种方法使代理能够从对抗程序和行为克隆损失中获得反馈。我们实验证明，与其他最先进的策略 ILO 和 ILD 方法相比，我们提出的算法需要较少的与环境的交互来实现专家性能。

    Imitation learning from demonstrations (ILD) aims to alleviate numerous shortcomings of reinforcement learning through the use of demonstrations. However, in most real-world applications, expert action guidance is absent, making the use of ILD impossible. Instead, we consider imitation learning from observations (ILO), where no expert actions are provided, making it a significantly more challenging problem to address. Existing methods often employ on-policy learning, which is known to be sample-costly. This paper presents SEILO, a novel sample-efficient on-policy algorithm for ILO, that combines standard adversarial imitation learning with inverse dynamics modeling. This approach enables the agent to receive feedback from both the adversarial procedure and a behavior cloning loss. We empirically demonstrate that our proposed algorithm requires fewer interactions with the environment to achieve expert performance compared to other state-of-the-art on-policy ILO and ILD methods.
    
[^225]: 我的模型性能为什么会下降？对片段发现算法的人工评估

    Where Does My Model Underperform? A Human Evaluation of Slice Discovery Algorithms. (arXiv:2306.08167v1 [cs.HC])

    [http://arxiv.org/abs/2306.08167](http://arxiv.org/abs/2306.08167)

    机器学习模型在语义连贯的数据子集上表现不佳仍然会出现问题，但是确定这些问题片段可能很困难，自动生成的片段并不是确定人工从业者问题性片段的银弹。

    

    机器学习（ML）模型可以在语义连贯的数据子集（即“片段”）上表现不佳，而高平均准确率的模型仍然会出现这种问题。这种行为可能对模型的安全性或偏见在部署中产生重大影响，但在实践中确定这些性能下降的片段可能很困难，特别是在从业者缺乏访问群组注释以定义其数据的连贯子集的领域。受到这些挑战的驱动，ML研究人员开发了新的片段发现算法，旨在将数据的连贯和高误差子集分组在一起。然而，评估这些工具是否帮助人类正确形成他们的模型性能下降的假设还很少。我们进行了一项受控用户研究（N = 15），向用户展示两种最先进的片段发现算法输出的40个片段，并要求他们形成有关对象检测模型性能下降的假设。我们的响应变量是参与者正确按错误率对片段进行排名的能力。我们的主要发现是：（1）两种片段发现算法都不会让参与者在假设上表现出系统性的优势；（2）即使在同一种片段发现算法中，参与者在正确对片段进行排序的能力上也存在显着变异；（3）对象类别的错误率比对象大小或位置等隐含语义更好地预测了片段难度。总体而言，我们的结果表明自动生成的片段并非确定人工从业者问题性片段的银弹，而在实践中使用这些算法必须小心。

    Machine learning (ML) models that achieve high average accuracy can still underperform on semantically coherent subsets (i.e. "slices") of data. This behavior can have significant societal consequences for the safety or bias of the model in deployment, but identifying these underperforming slices can be difficult in practice, especially in domains where practitioners lack access to group annotations to define coherent subsets of their data. Motivated by these challenges, ML researchers have developed new slice discovery algorithms that aim to group together coherent and high-error subsets of data. However, there has been little evaluation focused on whether these tools help humans form correct hypotheses about where (for which groups) their model underperforms. We conduct a controlled user study (N = 15) where we show 40 slices output by two state-of-the-art slice discovery algorithms to users, and ask them to form hypotheses about where an object detection model underperforms. Our res
    
[^226]: 面向异质治疗效应估计的动态治疗信息共享

    Dynamic Inter-treatment Information Sharing for Heterogeneous Treatment Effects Estimation. (arXiv:2305.15984v1 [cs.LG])

    [http://arxiv.org/abs/2305.15984](http://arxiv.org/abs/2305.15984)

    本论文提出了一种基于深度学习的HyperCATE框架，通过软权重共享的方式实现端到端信息共享来解决现有CATE学习器中的有偏估计问题，并在IHDP、ACIC-2016和Twins基准测试中评估了该框架的表现。

    

    已有的异质治疗效应学习者缺乏端到端治疗信息共享的通用机制，必须将数据分割到潜在结果函数中训练CATE学习器，这可能导致具有有限观测数据的有偏估计。为了解决这个问题，我们提出了一种基于深度学习的框架，用于训练CATE学习器，促进治疗组之间的动态端到端信息共享。该框架基于“超网络”的“软权重共享”，具有参数效率、更快训练和改进结果等优点。所提出的框架补充了现有的CATE学习器，并引入了一类我们称之为“HyperCATE”的新型不确定性感知CATE学习器。我们开发了常用CATE学习器的HyperCATE版本，并在IHDP、ACIC-2016和Twins基准测试中进行了评估。

    Existing heterogeneous treatment effects learners, also known as conditional average treatment effects (CATE) learners, lack a general mechanism for end-to-end inter-treatment information sharing, and data have to be split among potential outcome functions to train CATE learners which can lead to biased estimates with limited observational datasets. To address this issue, we propose a novel deep learning-based framework to train CATE learners that facilitates dynamic end-to-end information sharing among treatment groups. The framework is based on \textit{soft weight sharing} of \textit{hypernetworks}, which offers advantages such as parameter efficiency, faster training, and improved results. The proposed framework complements existing CATE learners and introduces a new class of uncertainty-aware CATE learners that we refer to as \textit{HyperCATE}. We develop HyperCATE versions of commonly used CATE learners and evaluate them on IHDP, ACIC-2016, and Twins benchmarks. Our experimental 
    
[^227]: 高斯门控混合专家模型参数估计的收敛速率研究

    Towards Convergence Rates for Parameter Estimation in Gaussian-gated Mixture of Experts. (arXiv:2305.07572v1 [stat.ML])

    [http://arxiv.org/abs/2305.07572](http://arxiv.org/abs/2305.07572)

    本文提出新颖的Voronoi Loss函数来解决高斯门控混合专家模型参数估计的收敛速率问题，并在两种不同的门控网络下提供理论收敛速率的证明。

    

    混合专家模型因其在集成学习中的应用而被引入神经网络中，近年来成为现代深度神经网络中处理异构数据分析的基本构件。然而，对于高斯门控混合专家模型参数估计的收敛行为的理解还不充分。我们通过设计新颖的Voronoi Loss函数来解决这些问题，并提供了理论收敛速率的证明，揭示了在两种分离的门控网络下最大似然估计器的不同行为。

    Originally introduced as a neural network for ensemble learning, mixture of experts (MoE) has recently become a fundamental building block of highly successful modern deep neural networks for heterogeneous data analysis in several applications, including those in machine learning, statistics, bioinformatics, economics, and medicine. Despite its popularity in practice, a satisfactory level of understanding of the convergence behavior of Gaussian-gated MoE parameter estimation is far from complete. The underlying reason for this challenge is the inclusion of covariates in the Gaussian gating and expert networks, which leads to their intrinsically complex interactions via partial differential equations with respect to their parameters. We address these issues by designing novel Voronoi loss functions to accurately capture heterogeneity in the maximum likelihood estimator (MLE) for resolving parameter estimation in these models. Our results reveal distinct behaviors of the MLE under two se
    
[^228]: 一种新的具有自适应停止准则的非精确近端线性算法，用于鲁棒相位恢复问题。

    A New Inexact Proximal Linear Algorithm with Adaptive Stopping Criteria for Robust Phase Retrieval. (arXiv:2304.12522v1 [math.OC])

    [http://arxiv.org/abs/2304.12522](http://arxiv.org/abs/2304.12522)

    本文提出了一种新的鲁棒相位恢复算法，通过使用自适应停止准则的非精确近端线性算法，该方法在实验中证明比现有方法更高效。

    

    本文考虑了鲁棒相位恢复问题，该问题可视为一个非光滑和非凸优化问题。我们提出了一种新的非精确近端线性算法，其中子问题被不精确求解。我们的贡献是为子问题提出了两种自适应停止准则。我们分析了所提出方法的收敛性能。通过对合成和实际数据集的实验，我们证明了我们的方法比现有方法更高效，例如原始近端线性算法和次梯度方法。

    This paper considers the robust phase retrieval problem, which can be cast as a nonsmooth and nonconvex optimization problem. We propose a new inexact proximal linear algorithm with the subproblem being solved inexactly. Our contributions are two adaptive stopping criteria for the subproblem. The convergence behavior of the proposed methods is analyzed. Through experiments on both synthetic and real datasets, we demonstrate that our methods are much more efficient than existing methods, such as the original proximal linear algorithm and the subgradient method.
    
[^229]: 基于理想联合分类器假设的知识蒸馏

    Knowledge Distillation Under Ideal Joint Classifier Assumption. (arXiv:2304.11004v1 [cs.LG])

    [http://arxiv.org/abs/2304.11004](http://arxiv.org/abs/2304.11004)

    本文提出了基于理想联合分类器假设的知识蒸馏框架，可以提供清晰全面的理解和为未来研究提供理论基础，使得教师和学生网络之间的知识传递更加高效。

    

    知识蒸馏是一种将大型神经网络压缩为更高效小型网络的强大技术。Softmax回归表征学习是一种常用的方法，它使用预先训练的教师网络来指导更小的学生网络的学习。尽管有几项研究探讨了Softmax回归表征学习的有效性，但提供知识转移的基础机制尚不够清楚。本文提出了理想联合分类器知识蒸馏（IJCKD），这是一个统一的框架，旨在为现有的知识蒸馏方法提供清晰全面的理解和为未来研究提供理论基础。我们使用从领域适应理论推导出的数学技术，提供了学生网络误差界的详细分析，其作为教师的函数关系。我们的框架可以在深度学习中应用于各种应用，包括图像识别和自然语言处理。

    Knowledge distillation is a powerful technique to compress large neural networks into smaller, more efficient networks. Softmax regression representation learning is a popular approach that uses a pre-trained teacher network to guide the learning of a smaller student network. While several studies explored the effectiveness of softmax regression representation learning, the underlying mechanism that provides knowledge transfer is not well understood. This paper presents Ideal Joint Classifier Knowledge Distillation (IJCKD), a unified framework that provides a clear and comprehensive understanding of the existing knowledge distillation methods and a theoretical foundation for future research. Using mathematical techniques derived from a theory of domain adaptation, we provide a detailed analysis of the student network's error bound as a function of the teacher. Our framework enables efficient knowledge transfer between teacher and student networks and can be applied to various applicati
    
[^230]: 基于深度函数的偏序集合的描述性分析和机器学习算法

    Depth Functions for Partial Orders with a Descriptive Analysis of Machine Learning Algorithms. (arXiv:2304.09872v1 [cs.LG])

    [http://arxiv.org/abs/2304.09872](http://arxiv.org/abs/2304.09872)

    本文提出了一种基于深度函数的偏序集合描述性分析框架，并引入了偏序版本的单纯深度，用于比较基于多维性能度量的机器学习算法。实验证明此方法与现有基准方法不同，为分类器比较提供了新的视角。

    

    我们提出了一个框架，基于深度函数对偏序集合进行描述性分析。尽管深度函数在线性和度量空间中进行了大量研究，但是对于偏序等非标准数据类型的深度函数的讨论却很少。我们介绍了著名的单纯深度的偏序版本-无并通用深度（ufg depth）。此外，我们利用我们的 ufg depth 来比较基于多维性能度量的机器学习算法。具体地，我们分析不同分类器在标准基准数据集样本上的表现分布。我们的结果有希望地证明了我们的方法与现有基准方法有很大不同，因此为分类器比较的激烈辩论增加了新的视角。

    We propose a framework for descriptively analyzing sets of partial orders based on the concept of depth functions. Despite intensive studies of depth functions in linear and metric spaces, there is very little discussion on depth functions for non-standard data types such as partial orders. We introduce an adaptation of the well-known simplicial depth to the set of all partial orders, the union-free generic (ufg) depth. Moreover, we utilize our ufg depth for a comparison of machine learning algorithms based on multidimensional performance measures. Concretely, we analyze the distribution of different classifier performances over a sample of standard benchmark data sets. Our results promisingly demonstrate that our approach differs substantially from existing benchmarking approaches and, therefore, adds a new perspective to the vivid debate on the comparison of classifiers.
    
[^231]: 增强弱监督分割的高保真伪标签生成方法

    High-fidelity Pseudo-labels for Boosting Weakly-Supervised Segmentation. (arXiv:2304.02621v1 [cs.CV])

    [http://arxiv.org/abs/2304.02621](http://arxiv.org/abs/2304.02621)

    本文提出了一种使用马尔可夫随机场增强弱监督分割中的高保真伪标签生成方法，能够生成更准确的伪标签，并使用新的训练策略来实现更好的收敛。实验结果表明该方法达到了弱监督分割方法的最佳性能。

    

    近年来，图像级别的弱监督语义分割（WSSS）任务因为其可减少大量数据标注成本而变得流行。WSSS的典型方法是使用全局平均池化（GAP）在卷积特征映射上训练图像分类网络。这使得可以基于类别激活图（CAMs）估计对象位置，CAMs识别图像区域的重要性。然后使用CAMs生成伪标签，以形式化的分割掩码的方式在缺乏像素级标签的情况下对分割模型进行监督。在SEAM基线的情况下，一个先前的工作提出了提高CAM学习的两种方法：（1）重要性抽样，它是GAP的替代方法；（2）特征相似性损失，它使用一种启发式方法，即对象轮廓几乎仅与图像中的颜色边缘对齐。在这项工作中，我们为这些任务提出了一种不同的概率解释CAM的方法，从而生成更精确的伪标签。具体而言，我们采用马尔可夫随机场将局部空间一致性约束融入CAM学习中。我们还提出了一种新的训练策略，交替更新CAM和分割模型以实现更好的收敛。在基准数据集上的实验结果表明，我们的方法在弱监督分割方法中实现了最先进的性能。

    The task of image-level weakly-supervised semantic segmentation (WSSS) has gained popularity in recent years, as it reduces the vast data annotation cost for training segmentation models. The typical approach for WSSS involves training an image classification network using global average pooling (GAP) on convolutional feature maps. This enables the estimation of object locations based on class activation maps (CAMs), which identify the importance of image regions. The CAMs are then used to generate pseudo-labels, in the form of segmentation masks, to supervise a segmentation model in the absence of pixel-level ground truth. In case of the SEAM baseline, a previous work proposed to improve CAM learning in two ways: (1) Importance sampling, which is a substitute for GAP, and (2) the feature similarity loss, which utilizes a heuristic that object contours almost exclusively align with color edges in images. In this work, we propose a different probabilistic interpretation of CAMs for thes
    
[^232]: 图马尔可夫神经网络的公平评估

    Fair Evaluation of Graph Markov Neural Networks. (arXiv:2304.01235v1 [cs.LG])

    [http://arxiv.org/abs/2304.01235](http://arxiv.org/abs/2304.01235)

    本论文通过引入适用于GMNN的新测试方法，对三类不同信息源对GMNN在WikiVitals数据集中的预测准确性的贡献进行严格评估，结果表明标签相关性是帮助GMNN获得优势的关键信息源。

    

    最近提出采用图马尔可夫神经网络（GMNN）改进常规图神经网络（GNN），将标签依赖性纳入半监督节点分类任务中。GMNN从理论上以严谨的方式解决问题，并使用三类信息来预测标签。与常规的GNN一样，他们使用节点特征和图结构，但他们还利用相邻节点标签的信息，提高预测的准确性。本文介绍了一个名为WikiVitals的新数据集，其中包含48k个互相引用的维基百科文章，被分类为32个类别，由2.3M边连接。我们的目标是对GMNN对这个数据集的贡献的三种不同信息源进行严格评估：文章内容、文章互相之间的连接以及标签之间的相关性。为此，我们采用了一种最近提出的适用于GNN的样本外测试方法，并将其适用于GMNN。我们的实验结果显示，GMNN在此数据集上始终优于GNN，并且标签相关性是帮助GMNN实现这些增益的关键信息源。

    Graph Markov Neural Networks (GMNN) have recently been proposed to improve regular graph neural networks (GNN) by including label dependencies into the semi-supervised node classification task. GMNNs do this in a theoretically principled way and use three kinds of information to predict labels. Just like ordinary GNNs, they use the node features and the graph structure but they moreover leverage information from the labels of neighboring nodes to improve the accuracy of their predictions. In this paper, we introduce a new dataset named WikiVitals which contains a graph of 48k mutually referred Wikipedia articles classified into 32 categories and connected by 2.3M edges. Our aim is to rigorously evaluate the contributions of three distinct sources of information to the prediction accuracy of GMNN for this dataset: the content of the articles, their connections with each other and the correlations among their labels. For this purpose we adapt a method which was recently proposed for perf
    
[^233]: 一种用于智能按需公共交通预测公交车到达时间的新型神经网络方法

    A Novel Neural Network Approach for Predicting the Arrival Time of Buses for Smart On-Demand Public Transit. (arXiv:2303.15495v1 [cs.LG])

    [http://arxiv.org/abs/2303.15495](http://arxiv.org/abs/2303.15495)

    本文介绍了一种基于神经网络的数据驱动方法，可以跨所有公交线路集体预测公交车到达每个交通点的时间，解决公交运输中公交车到达时间不准确和可靠的问题。

    

    在城市的主要公共交通系统中，公交运输存在着问题，其中包括对于乘客公交车到达时间的估计更加准确和可靠。这可能导致延误和减少乘客人数，尤其是在依靠公共交通的城市中更加严重。公交车到达时间与时间表不匹配是一个普遍的问题，导致固定时刻表的延迟。根据本文在纽约市公交数据上进行的研究，公交车到达时间和实际计划时间之间存在平均约八分钟或491秒的延迟。本研究提出了一种基于人工智能的数据驱动方法，用于估计每个交通点（站）公交车的到达时间。我们的方法基于全连接神经网络，可以在大都市区域中跨所有公交线路集体预测到达时间。我们的神经网络数据驱动方法为估算公交车到达时间提供了一种新的方式。

    Among the major public transportation systems in cities, bus transit has its problems, including more accuracy and reliability when estimating the bus arrival time for riders. This can lead to delays and decreased ridership, especially in cities where public transportation is heavily relied upon. A common issue is that the arrival times of buses do not match the schedules, resulting in latency for fixed schedules. According to the study in this paper on New York City bus data, there is an average delay of around eight minutes or 491 seconds mismatch between the bus arrivals and the actual scheduled time. This research paper presents a novel AI-based data-driven approach for estimating the arrival times of buses at each transit point (station). Our approach is based on a fully connected neural network and can predict the arrival time collectively across all bus lines in large metropolitan areas. Our neural-net data-driven approach provides a new way to estimate the arrival time of the b
    
[^234]: 利用深度学习预测离散时间分歧

    Predicting discrete-time bifurcations with deep learning. (arXiv:2303.09669v1 [q-bio.QM])

    [http://arxiv.org/abs/2303.09669](http://arxiv.org/abs/2303.09669)

    本研究利用深度学习训练分类器预测离散时间五种本地分岔，在经济、生态、生理学等方面的试验数据中都具有优秀表现，是提前警告关键转变的重要方法。

    

    许多自然和人造系统容易发生关键转变-动力学的突然和潜在的破坏性变化。深度学习分类器通过从大规模模拟训练数据集中学习分岔（动力学不稳定性）的通用特征，为关键转变提供提前警告信号（EWS）。到目前为止，分类器只被训练用于预测连续时间分歧，而忽略了离散时间分歧独特的丰富动态特征。本文使用深度学习分类器训练提供 EWS 的五种离散时间、共维度1的本地分岔。我们在生理学、经济学和生态学中使用的离散时间模型的模拟数据以及经历了倍增分岔的鸡心聚集的实验数据进行测试。在广泛的噪声强度和接近分岔的速率范围内，分类器优于常用的 EWS。它也能够正确预测分岔。

    Many natural and man-made systems are prone to critical transitions -- abrupt and potentially devastating changes in dynamics. Deep learning classifiers can provide an early warning signal (EWS) for critical transitions by learning generic features of bifurcations (dynamical instabilities) from large simulated training data sets. So far, classifiers have only been trained to predict continuous-time bifurcations, ignoring rich dynamics unique to discrete-time bifurcations. Here, we train a deep learning classifier to provide an EWS for the five local discrete-time bifurcations of codimension-1. We test the classifier on simulation data from discrete-time models used in physiology, economics and ecology, as well as experimental data of spontaneously beating chick-heart aggregates that undergo a period-doubling bifurcation. The classifier outperforms commonly used EWS under a wide range of noise intensities and rates of approach to the bifurcation. It also predicts the correct bifurcation
    
[^235]: 使用占用空间和时间较小的低深度量子态准备方法及其应用

    Spacetime-Efficient Low-Depth Quantum State Preparation with Applications. (arXiv:2303.02131v2 [quant-ph] UPDATED)

    [http://arxiv.org/abs/2303.02131](http://arxiv.org/abs/2303.02131)

    提出了一种使用占用空间和时间较小的低深度方法来准备任意量子态，能够在较少的量子资源使用下实现更快的准备速度。

    

    我们提出一种新颖的确定性方法来准备任意的量子态。当将我们的协议编译为CNOT门和任意的单比特门时，它可以在深度$O(\log(N))$和空间时间分配$O(N)$的情况下准备一个$N$维的量子态，这两个参数都是最优的。当编译为$\{\mathrm{H,S,T,CNOT}\}$门集时，我们证明它所需的量子资源比之前的方法要少。具体而言，在深度$O(\log(N/\epsilon))$和空间时间分配$O(N\log(\log(N)/\epsilon))$下，可以准备一个误差为$\epsilon$的任意的量子态，这比之前方法的$O(\log(N)\log(N/\epsilon))$和$O(N\log(N/\epsilon))$有所改进。我们说明了我们协议的减小空间时间分配可以有效地快速准备多个不相交状态，只需要常数因子的辅助量子比特开销--通过高效地重用$O(N)$个辅助比特来准备一个乘积态。

    We propose a novel deterministic method for preparing arbitrary quantum states. When our protocol is compiled into CNOT and arbitrary single-qubit gates, it prepares an $N$-dimensional state in depth $O(\log(N))$ and spacetime allocation (a metric that accounts for the fact that oftentimes some ancilla qubits need not be active for the entire circuit) $O(N)$, which are both optimal. When compiled into the $\{\mathrm{H,S,T,CNOT}\}$ gate set, we show that it requires asymptotically fewer quantum resources than previous methods. Specifically, it prepares an arbitrary state up to error $\epsilon$ in depth $O(\log(N/\epsilon))$ and spacetime allocation $O(N\log(\log(N)/\epsilon))$, improving over $O(\log(N)\log(N/\epsilon))$ and $O(N\log(N/\epsilon))$, respectively. We illustrate how the reduced spacetime allocation of our protocol enables rapid preparation of many disjoint states with only constant-factor ancilla overhead -- $O(N)$ ancilla qubits are reused efficiently to prepare a product
    
[^236]: 通过物理对称学习可解释的低维表示

    Learning Interpretable Low-dimensional Representation via Physical Symmetry. (arXiv:2302.10890v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.10890](http://arxiv.org/abs/2302.10890)

    通过使用物理对称性作为潜在空间的自洽约束条件，该研究展示了在音乐领域和计算机视觉领域，模型可以以无监督的方式学习出可解释的低维表示，例如线性音高和三维笛卡尔因素。

    

    可解释的表示学习在创造性智能系统中起着关键作用。在音乐领域，当前的学习算法可以成功地学习各种特征，如音高、音色、和弦、纹理等。然而，大多数方法严重依赖音乐领域知识。现在还不清楚什么样的一般性计算原则会产生可解释的表示，特别是与人类感知保持一致的低维因素。在这项研究中，我们从现代物理学中获得灵感，将物理对称性作为潜在空间的自洽约束条件。特别是，它要求先验模型对潜在状态的动态进行描述，并以某种群变换对其进行等变。我们展示了物理对称性使得模型能够以无监督的方式从未标记的单声道音乐音频中学习一个线性音高因素。此外，相同的方法可以应用于计算机视觉，学习一个三维笛卡尔因素。

    Interpretable representation learning has been playing a key role in creative intelligent systems. In the music domain, current learning algorithms can successfully learn various features such as pitch, timbre, chord, texture, etc. However, most methods rely heavily on music domain knowledge. It remains an open question what general computational principles give rise to interpretable representations, especially low-dim factors that agree with human perception. In this study, we take inspiration from modern physics and use physical symmetry as a self-consistency constraint for the latent space. Specifically, it requires the prior model that characterises the dynamics of the latent states to be equivariant with respect to certain group transformations. We show that physical symmetry leads the model to learn a linear pitch factor from unlabelled monophonic music audio in a self-supervised fashion. In addition, the same methodology can be applied to computer vision, learning a 3D Cartesian
    
[^237]: 一种混合联邦学习的原始-对偶算法

    A Primal-Dual Algorithm for Hybrid Federated Learning. (arXiv:2210.08106v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.08106](http://arxiv.org/abs/2210.08106)

    该论文提出了一种基于Fenchel对偶性的快速、稳健的混合联邦学习算法。实验证明了该算法相对于传统的FedAvg方法的性能改进，并提供了隐私保护措施。

    

    在实际情况中，混合联邦学习很重要，但对于仅持有部分特征和样本的客户端，很少有方法存在。我们提供了一种基于Fenchel对偶性的快速、稳健的混合联邦学习算法。我们证明了算法在各种实际情况下收敛于与在中心训练模型相同的解决方案。此外，我们还提供了实验结果，证明了该算法相对于联邦学习中常用的方法FedAvg的性能改进。我们还提供了隐私考虑和保护客户数据的必要步骤。

    Very few methods for hybrid federated learning, where clients only hold subsets of both features and samples, exist. Yet, this scenario is very important in practical settings. We provide a fast, robust algorithm for hybrid federated learning that hinges on Fenchel Duality. We prove the convergence of the algorithm to the same solution as if the model was trained centrally in a variety of practical regimes. Furthermore, we provide experimental results that demonstrate the performance improvements of the algorithm over a commonly used method in federated learning, FedAvg. We also provide privacy considerations and necessary steps to protect client data.
    
[^238]: 编码理论与交叉验证的联系及其应用

    A Link between Coding Theory and Cross-Validation with Applications. (arXiv:2103.11856v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2103.11856](http://arxiv.org/abs/2103.11856)

    本研究研究了编码理论与交叉验证之间的联系，并发现了学习算法在固定数据上能解决不同二进制分类问题的数量与误差检测码理论密切相关。我们还对一种特定的交叉验证方法下的最大分类问题数量进行了研究，这取决于常权码的码字数量。同时，我们推广了常权码的概念，并证明了类似的结果适用于其他交叉验证错误和轻量级常权码。

    

    单个学习算法在给定数据上能解决多少个不同的二进制分类问题，其中要求交叉验证错误为零或最多给定数量？在前一种情况下，根据无免费午餐定理，这个数量是有限的，我们表明精确答案由误差检测码理论给出。作为案例研究，我们关注AUC性能度量和留一对交叉验证(LPOCV)，其中每个可能具有不同类标签的数据对都会被暂时保留。我们发现，对于固定类比例的分类问题，学习算法能够实现零LPOCV错误的最大数量等于常权码(CWC)中的最大码字数量，具有一定的技术性质。然后，我们通过引入轻量级常权码(light CWC)来推广CWC，并证明了对于非零LPOCV错误和轻量级常权码的类似结果。此外，我们对码字数量的最大上界和最大下界也进行了证明。

    How many different binary classification problems a single learning algorithm can solve on a fixed data with exactly zero or at most a given number of cross-validation errors? While the number in the former case is known to be limited by the no-free-lunch theorem, we show that the exact answers are given by the theory of error detecting codes. As a case study, we focus on the AUC performance measure and leave-pair-out cross-validation (LPOCV), in which every possible pair of data with different class labels is held out at a time. We shown that the maximal number of classification problems with fixed class proportion, for which a learning algorithm can achieve zero LPOCV error, equals the maximal number of code words in a constant weight code (CWC), with certain technical properties. We then generalize CWCs by introducing light CWCs and prove an analogous result for nonzero LPOCV errors and light CWCs. Moreover, we prove both upper and lower bounds on the maximal numbers of code words i
    

