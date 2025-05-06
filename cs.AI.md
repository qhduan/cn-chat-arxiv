# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CHOSEN: Contrastive Hypothesis Selection for Multi-View Depth Refinement](https://arxiv.org/abs/2404.02225) | CHOSEN是一个用于多视角深度细化的对比假设选择框架，通过对比学习和精心设计的假设特征，能够在多视角立体匹配中实现较高质量的深度和法线精度。 |
| [^2] | [Backpropagation through space, time, and the brain](https://arxiv.org/abs/2403.16933) | 提出了 Generalized Latent Equilibrium (GLE)，它是一种针对神经元网络的物理动态局部时空信用分配的计算框架。 |
| [^3] | [Learning with Noisy Foundation Models](https://arxiv.org/abs/2403.06869) | 本文首次全面了解和分析了预训练数据集中的噪声性质，有效减轻其对下游任务影响。 |
| [^4] | [DECIDER: A Rule-Controllable Decoding Strategy for Language Generation by Imitating Dual-System Cognitive Theory](https://arxiv.org/abs/2403.01954) | DECIDER是一种受双系统认知理论启发的规则可控解码策略，通过在预训练语言模型中引入逻辑推理器，有效地遵循给定规则以引导生成方向朝向目标。 |
| [^5] | [GenAINet: Enabling Wireless Collective Intelligence via Knowledge Transfer and Reasoning](https://arxiv.org/abs/2402.16631) | 该论文提出了基于GenAI的GenAINet框架，通过知识传输和推理实现无线集体智能，为6G时代铺平了通向人工通用智能的道路。 |
| [^6] | [REPLAY: Modeling Time-Varying Temporal Regularities of Human Mobility for Location Prediction over Sparse Trajectories](https://arxiv.org/abs/2402.16310) | 该论文提出了REPLAY模型，利用一般RNN架构来学习捕捉人类移动的时间变化规律，用于位置预测。 |
| [^7] | [Linear Dynamics-embedded Neural Network for Long-Sequence Modeling](https://arxiv.org/abs/2402.15290) | 提出了一种名为嵌入线性动力学的神经网络（LDNN），通过引入连续状态空间模型的属性和优化策略，实现了在长序列任务中具有少量参数、灵活推断和高效训练，最终在长距离竞技场上取得了有效且领先的性能。 |
| [^8] | [Clients Collaborate: Flexible Differentially Private Federated Learning with Guaranteed Improvement of Utility-Privacy Trade-off](https://arxiv.org/abs/2402.07002) | 本论文提出了一种名为FedCEO的新型联邦学习框架，在保护用户隐私的同时，通过让客户端相互协作，实现了对模型效用和隐私之间的权衡。通过高效的张量低秩近端优化，该框架能够恢复被打断的语义信息，并在效用-隐私权衡方面取得了显著的改进。 |
| [^9] | [SMUTF: Schema Matching Using Generative Tags and Hybrid Features](https://arxiv.org/abs/2402.01685) | SMUTF是一种用于大规模表格数据模式匹配的独特方法，通过结合基于规则的特征工程、预训练语言模型和生成式大语言模型，并使用生成标签提高匹配效果。同时，作者开发并开源了HDXSM数据集来解决现有数据集不足的问题。 |
| [^10] | [Joint Problems in Learning Multiple Dynamical Systems](https://arxiv.org/abs/2311.02181) | 聚类时间序列的新问题，提出联合划分轨迹集并学习每个部分的线性动态系统模型，以最小化所有模型的最大误差 |
| [^11] | [Advective Diffusion Transformers for Topological Generalization in Graph Learning.](http://arxiv.org/abs/2310.06417) | 本研究探索了在不同的图拓扑存在下，图扩散方程如何对GNN进行外推和概括，揭示了基于局部扩散的现有模型在概括能力上的不足，并提出了非局部扩散的潜力。 |
| [^12] | [A simple connection from loss flatness to compressed representations in neural networks.](http://arxiv.org/abs/2310.01770) | 该论文研究了深度神经网络中损失平坦性和神经表示压缩之间的关系，通过简单的数学关系，证明了损失平坦性与神经表示的压缩相关。 |
| [^13] | [Large Language Models at Work in China's Labor Market.](http://arxiv.org/abs/2308.08776) | 本文研究了大型语言模型（LLMs）对中国劳动市场的潜在影响，并分析了职业对LLM能力的暴露程度及其与工资水平和经验溢价之间的关系。研究结果表明，高薪和经验密集型工作可能面临更大的替代风险。此外，研究还开发了一个考虑行业暴露的经济增长模型，以量化AI采用对生产力和就业之间的权衡。这项研究为理解中国劳动市场中越来越强大的AI系统的影响提供了基础。 |
| [^14] | [When Foundation Model Meets Federated Learning: Motivations, Challenges, and Future Directions.](http://arxiv.org/abs/2306.15546) | 基础模型与联邦学习的交叉提供了解锁新可能性的独特机会，扩展了数据可用性，促进了协作式模型发展，并提高了性能和隐私保护。 |
| [^15] | [One Transformer for All Time Series: Representing and Training with Time-Dependent Heterogeneous Tabular Data.](http://arxiv.org/abs/2302.06375) | 本研究提出了一种Transformer架构，用于表示具有时间相关的异构表格数据，通过使用一组频率函数来表示数值特征，并采用唯一的损失函数进行统一训练。 |

# 详细

[^1]: CHOSEN：用于多视角深度细化的对比假设选择

    CHOSEN: Contrastive Hypothesis Selection for Multi-View Depth Refinement

    [https://arxiv.org/abs/2404.02225](https://arxiv.org/abs/2404.02225)

    CHOSEN是一个用于多视角深度细化的对比假设选择框架，通过对比学习和精心设计的假设特征，能够在多视角立体匹配中实现较高质量的深度和法线精度。

    

    我们提出了CHOSEN，这是一个简单而灵活、强大且有效的多视角深度细化框架。它可以应用于任何现有的多视角立体匹配流程中，具有针对不同多视角采集系统（如相机相对定位和镜头）的直观泛化能力。给定初始深度估计，CHOSEN迭代地重新采样并选择最佳假设，并自动适应由采集系统确定的不同度量或固有尺度。我们方法的关键在于在适当的解决方案空间中应用对比学习以及精心设计的假设特征，基于这些特征，可以有效地区分正负假设。将CHOSEN集成到简单的基线多视角立体匹配流程中，在深度和法线精度方面提供了令人印象深刻的质量，与许多当前基于深度学习的多视角立体匹配流程相比有所提高。

    arXiv:2404.02225v1 Announce Type: cross  Abstract: We propose CHOSEN, a simple yet flexible, robust and effective multi-view depth refinement framework. It can be employed in any existing multi-view stereo pipeline, with straightforward generalization capability for different multi-view capture systems such as camera relative positioning and lenses. Given an initial depth estimation, CHOSEN iteratively re-samples and selects the best hypotheses, and automatically adapts to different metric or intrinsic scales determined by the capture system. The key to our approach is the application of contrastive learning in an appropriate solution space and a carefully designed hypothesis feature, based on which positive and negative hypotheses can be effectively distinguished. Integrated in a simple baseline multi-view stereo pipeline, CHOSEN delivers impressive quality in terms of depth and normal accuracy compared to many current deep learning based multi-view stereo pipelines.
    
[^2]: 通过空间、时间和大脑进行反向传播

    Backpropagation through space, time, and the brain

    [https://arxiv.org/abs/2403.16933](https://arxiv.org/abs/2403.16933)

    提出了 Generalized Latent Equilibrium (GLE)，它是一种针对神经元网络的物理动态局部时空信用分配的计算框架。

    

    有效的神经网络学习需要根据它们对解决任务的相对贡献来调整单个突触。然而，无论是生物还是人工的物理神经系统都受到时空局限。这样的网络如何执行高效的信用分配，在很大程度上仍是一个悬而未决的问题。在机器学习中，错误的反向传播算法几乎普遍被空间（BP）和时间（BPTT）两种方式给出答案。然而，BP(TT)被广泛认为依赖于不具生物学意义的假设，特别是关于时空局限性，而正向传播模型，如实时递归学习（RTRL），则受到内存约束的限制。我们引入了广义潜在平衡（GLE），这是一个针对神经元物理动态网络完全局部时空信用分配的计算框架。我们从

    arXiv:2403.16933v1 Announce Type: cross  Abstract: Effective learning in neuronal networks requires the adaptation of individual synapses given their relative contribution to solving a task. However, physical neuronal systems -- whether biological or artificial -- are constrained by spatio-temporal locality. How such networks can perform efficient credit assignment, remains, to a large extent, an open question. In Machine Learning, the answer is almost universally given by the error backpropagation algorithm, through both space (BP) and time (BPTT). However, BP(TT) is well-known to rely on biologically implausible assumptions, in particular with respect to spatiotemporal (non-)locality, while forward-propagation models such as real-time recurrent learning (RTRL) suffer from prohibitive memory constraints. We introduce Generalized Latent Equilibrium (GLE), a computational framework for fully local spatio-temporal credit assignment in physical, dynamical networks of neurons. We start by 
    
[^3]: 在有噪声基础模型中学习

    Learning with Noisy Foundation Models

    [https://arxiv.org/abs/2403.06869](https://arxiv.org/abs/2403.06869)

    本文首次全面了解和分析了预训练数据集中的噪声性质，有效减轻其对下游任务影响。

    

    基础模型通常是在大规模数据集上进行预训练，然后通过调整来适应下游任务。然而，大规模预训练数据集往往无法获取或成本过高，可能包含标签噪声，这可能会对模型的泛化能力造成不利影响，并带来意想不到的风险。本文是首个全面了解和分析预训练数据集中噪声性质，并有效减轻其对下游任务影响的工作。具体而言，通过在合成有噪声的ImageNet-1K、YFCC15M和CC12M数据集上进行完全监督和图像-文本对比预训练的广泛实验，我们证明了，尽管预训练中的轻微噪声可以使同领域（ID）性能受益，即训练和测试数据共享类似分布，但它总是会破坏跨领域（OOD）性能，在那里训练和测试分布明显不同。

    arXiv:2403.06869v1 Announce Type: cross  Abstract: Foundation models are usually pre-trained on large-scale datasets and then adapted to downstream tasks through tuning. However, the large-scale pre-training datasets, often inaccessible or too expensive to handle, can contain label noise that may adversely affect the generalization of the model and pose unexpected risks. This paper stands out as the first work to comprehensively understand and analyze the nature of noise in pre-training datasets and then effectively mitigate its impacts on downstream tasks. Specifically, through extensive experiments of fully-supervised and image-text contrastive pre-training on synthetic noisy ImageNet-1K, YFCC15M, and CC12M datasets, we demonstrate that, while slight noise in pre-training can benefit in-domain (ID) performance, where the training and testing data share a similar distribution, it always deteriorates out-of-domain (OOD) performance, where training and testing distributions are signific
    
[^4]: DECIDERS：一种通过模仿双系统认知理论实现规则可控解码策略的语言生成方法

    DECIDER: A Rule-Controllable Decoding Strategy for Language Generation by Imitating Dual-System Cognitive Theory

    [https://arxiv.org/abs/2403.01954](https://arxiv.org/abs/2403.01954)

    DECIDER是一种受双系统认知理论启发的规则可控解码策略，通过在预训练语言模型中引入逻辑推理器，有效地遵循给定规则以引导生成方向朝向目标。

    

    词典约束解码方法旨在通过某些目标概念控制所生成文本的意义或风格。现有方法过于关注这些目标本身，导致缺乏关于如何实现这些目标的高层推理。然而，人类通常通过遵循某些规则来处理任务，这些规则不仅关注于目标本身，还关注于引发目标发生的语义相关概念。在这项工作中，我们提出了DECIDER，这是一种受到双系统认知理论启发的约束语言生成的规则可控解码策略。具体而言，在DECIDER中，一个预训练语言模型（PLM）配备了一个逻辑推理器，以高层规则作为输入。然后，DECIDER允许规则信号在每个解码步骤中流入PLM。广泛的实验结果表明，DECIDER能够有效地遵循给定的规则，引导生成方向朝向目标进行生成。

    arXiv:2403.01954v1 Announce Type: cross  Abstract: Lexicon-based constrained decoding approaches aim to control the meaning or style of the generated text through certain target concepts. Existing approaches over-focus the targets themselves, leading to a lack of high-level reasoning about how to achieve them. However, human usually tackles tasks by following certain rules that not only focuses on the targets but also on semantically relevant concepts that induce the occurrence of targets. In this work, we present DECIDER, a rule-controllable decoding strategy for constrained language generation inspired by dual-system cognitive theory. Specifically, in DECIDER, a pre-trained language model (PLM) is equiped with a logic reasoner that takes high-level rules as input. Then, the DECIDER allows rule signals to flow into the PLM at each decoding step. Extensive experimental results demonstrate that DECIDER can effectively follow given rules to guide generation direction toward the targets i
    
[^5]: GenAINet：通过知识传输和推理实现无线集体智能

    GenAINet: Enabling Wireless Collective Intelligence via Knowledge Transfer and Reasoning

    [https://arxiv.org/abs/2402.16631](https://arxiv.org/abs/2402.16631)

    该论文提出了基于GenAI的GenAINet框架，通过知识传输和推理实现无线集体智能，为6G时代铺平了通向人工通用智能的道路。

    

    arXiv:2402.16631v2 声明类型：替换 摘要：生成人工智能（GenAI）和通信网络被期望在6G中具有突破性的协同作用。通过无线网络连接GenAI代理可能会释放集体智能的力量，并为人工通用智能（AGI）铺平道路。然而，当前的无线网络设计为“数据管道”，并不适合容纳和利用GenAI的力量。在本文中，我们提出了GenAINet框架，其中分布式GenAI代理传达知识（高级概念或摘要）以完成任意任务。首先，我们提供一个网络架构，整合了GenAI能力，以管理网络协议和应用程序。在此基础上，我们通过提出一种语义本地化的GenAINet来研究有效的通信和推理问题。具体来说，GenAI代理从多模态原始数据中提取语义概念，构建一个知识库表示

    arXiv:2402.16631v2 Announce Type: replace  Abstract: Generative artificial intelligence (GenAI) and communication networks are expected to have groundbreaking synergies in 6G. Connecting GenAI agents over a wireless network can potentially unleash the power of collective intelligence and pave the way for artificial general intelligence (AGI). However, current wireless networks are designed as a "data pipe" and are not suited to accommodate and leverage the power of GenAI. In this paper, we propose the GenAINet framework in which distributed GenAI agents communicate knowledge (high-level concepts or abstracts) to accomplish arbitrary tasks. We first provide a network architecture integrating GenAI capabilities to manage both network protocols and applications. Building on this, we investigate effective communication and reasoning problems by proposing a semantic-native GenAINet. Specifically, GenAI agents extract semantic concepts from multi-modal raw data, build a knowledgebase represe
    
[^6]: REPLAY: 对稀疏轨迹进行位置预测的人类移动时间变化规律建模

    REPLAY: Modeling Time-Varying Temporal Regularities of Human Mobility for Location Prediction over Sparse Trajectories

    [https://arxiv.org/abs/2402.16310](https://arxiv.org/abs/2402.16310)

    该论文提出了REPLAY模型，利用一般RNN架构来学习捕捉人类移动的时间变化规律，用于位置预测。

    

    位置预测是根据历史用户移动轨迹来预测用户位置的技术。为了应对真实世界用户移动轨迹的固有稀疏问题，时空上下文被证明是非常有用的。现有的解决方案主要是将位置之间的时空距离纳入到移动轨迹中，要么通过将其作为附加输入提供给递归神经网络（RNNs），要么通过利用它们来寻找有信息的过去隐藏状态进行预测。然而，这种基于距离的方法未能捕捉人类移动的时间变化规律，例如，人类移动在早晨通常比其他时间更有规律；这暗示了实际时间戳的有用性。基于这一背景，我们提出了REPLAY，是一种通用的RNN架构，旨在捕捉时间变化的人类移动时间规律以进行位置预测。

    arXiv:2402.16310v1 Announce Type: cross  Abstract: Location prediction forecasts a user's location based on historical user mobility traces. To tackle the intrinsic sparsity issue of real-world user mobility traces, spatiotemporal contexts have been shown as significantly useful. Existing solutions mostly incorporate spatiotemporal distances between locations in mobility traces, either by feeding them as additional inputs to Recurrent Neural Networks (RNNs) or by using them to search for informative past hidden states for prediction. However, such distance-based methods fail to capture the time-varying temporal regularities of human mobility, where human mobility is often more regular in the morning than in other periods, for example; this suggests the usefulness of the actual timestamps besides the temporal distances. Against this background, we propose REPLAY, a general RNN architecture learning to capture the time-varying temporal regularities for location prediction. Specifically, 
    
[^7]: 嵌入线性动力学的神经网络用于长序列建模

    Linear Dynamics-embedded Neural Network for Long-Sequence Modeling

    [https://arxiv.org/abs/2402.15290](https://arxiv.org/abs/2402.15290)

    提出了一种名为嵌入线性动力学的神经网络（LDNN），通过引入连续状态空间模型的属性和优化策略，实现了在长序列任务中具有少量参数、灵活推断和高效训练，最终在长距离竞技场上取得了有效且领先的性能。

    

    由于现有模型在长序列建模中性能和计算效率之间的权衡成为瓶颈，受到控制理论中具有多输入多输出的连续状态空间模型（SSMs）启发，我们提出了一种名为嵌入线性动力学的神经网络（LDNN）的新型神经网络。 SSM的连续、离散和卷积属性使LDNN具有少量参数、灵活的推断和在长序列任务中高效训练的特点。 我们开发了两种有效策略，对角化和“解耦然后快速傅立叶变换（FFT）”，以将卷积的时间复杂度从$O(LNH\max\{L, N\})$降低到$O(LN\max\{H, \log L\})$。 我们通过双向非因果和多头设置进一步改进了LDNN，以适应更广泛的应用范围。 对长距离竞技场（LRA）的大量实验表明了LDNN的有效性和最先进的性能。

    arXiv:2402.15290v1 Announce Type: cross  Abstract: The trade-off between performance and computational efficiency in long-sequence modeling becomes a bottleneck for existing models. Inspired by the continuous state space models (SSMs) with multi-input and multi-output in control theory, we propose a new neural network called Linear Dynamics-embedded Neural Network (LDNN). SSMs' continuous, discrete, and convolutional properties enable LDNN to have few parameters, flexible inference, and efficient training in long-sequence tasks. Two efficient strategies, diagonalization and $'\text{Disentanglement then Fast Fourier Transform (FFT)}'$, are developed to reduce the time complexity of convolution from $O(LNH\max\{L, N\})$ to $O(LN\max \{H, \log L\})$. We further improve LDNN through bidirectional noncausal and multi-head settings to accommodate a broader range of applications. Extensive experiments on the Long Range Arena (LRA) demonstrate the effectiveness and state-of-the-art performance
    
[^8]: 客户端协作：具有保证隐私-效用权衡改进的灵活差分隐私联邦学习

    Clients Collaborate: Flexible Differentially Private Federated Learning with Guaranteed Improvement of Utility-Privacy Trade-off

    [https://arxiv.org/abs/2402.07002](https://arxiv.org/abs/2402.07002)

    本论文提出了一种名为FedCEO的新型联邦学习框架，在保护用户隐私的同时，通过让客户端相互协作，实现了对模型效用和隐私之间的权衡。通过高效的张量低秩近端优化，该框架能够恢复被打断的语义信息，并在效用-隐私权衡方面取得了显著的改进。

    

    为了防止用户数据的隐私泄漏，在联邦学习中广泛使用差分隐私，但它并不是免费的。噪声的添加会随机干扰模型的语义完整性，并且这种干扰会随着通信轮次的增加而累积。在本文中，我们引入了一种具有严格隐私保证的新型联邦学习框架，名为FedCEO，通过让客户端"相互协作"，旨在在模型效用和用户隐私之间找到一种权衡。具体而言，我们在服务器上对堆叠的本地模型参数进行了高效的张量低秩近端优化，展示了它在光谱空间中灵活截断高频组分的能力。这意味着我们的FedCEO能够通过平滑全局语义空间来有效恢复被打断的语义信息，以适应不同隐私设置和持续的训练过程。此外，我们将SOTA的效用-隐私权衡边界提高了一个数量级。

    To defend against privacy leakage of user data, differential privacy is widely used in federated learning, but it is not free. The addition of noise randomly disrupts the semantic integrity of the model and this disturbance accumulates with increased communication rounds. In this paper, we introduce a novel federated learning framework with rigorous privacy guarantees, named FedCEO, designed to strike a trade-off between model utility and user privacy by letting clients ''Collaborate with Each Other''. Specifically, we perform efficient tensor low-rank proximal optimization on stacked local model parameters at the server, demonstrating its capability to flexibly truncate high-frequency components in spectral space. This implies that our FedCEO can effectively recover the disrupted semantic information by smoothing the global semantic space for different privacy settings and continuous training processes. Moreover, we improve the SOTA utility-privacy trade-off bound by an order of $\sqr
    
[^9]: SMUTF：使用生成标签和混合特征的模式匹配方法

    SMUTF: Schema Matching Using Generative Tags and Hybrid Features

    [https://arxiv.org/abs/2402.01685](https://arxiv.org/abs/2402.01685)

    SMUTF是一种用于大规模表格数据模式匹配的独特方法，通过结合基于规则的特征工程、预训练语言模型和生成式大语言模型，并使用生成标签提高匹配效果。同时，作者开发并开源了HDXSM数据集来解决现有数据集不足的问题。

    

    我们引入了SMUTF，一种用于大规模表格数据模式匹配的独特方法，该方法假设在开放域任务中，监督学习不会影响性能，从而实现了有效的跨域匹配。这个系统独特地结合了基于规则的特征工程、预训练语言模型和生成式大语言模型。受人道主义交换语言的启发，我们使用“生成标签”为每个数据列部署了创新的适应性，提高了模式匹配的效果。SMUTF具有广泛的灵活性，可以与任何现有的预训练嵌入、分类方法和生成模型无缝配合使用。鉴于模式匹配缺乏广泛的公开数据集，我们已经创建并开源了HDXSM数据集，该数据集来自公共人道主义数据，我们相信这是目前最全面的模式匹配数据集。

    We introduce SMUTF, a unique approach for large-scale tabular data schema matching (SM), which assumes that supervised learning does not affect performance in open-domain tasks, thereby enabling effective cross-domain matching. This system uniquely combines rule-based feature engineering, pre-trained language models, and generative large language models. In an innovative adaptation inspired by the Humanitarian Exchange Language, we deploy 'generative tags' for each data column, enhancing the effectiveness of SM. SMUTF exhibits extensive versatility, working seamlessly with any pre-existing pre-trained embeddings, classification methods, and generative models.   Recognizing the lack of extensive, publicly available datasets for SM, we have created and open-sourced the HDXSM dataset from the public humanitarian data. We believe this to be the most exhaustive SM dataset currently available. In evaluations across various public datasets and the novel HDXSM dataset, SMUTF demonstrated excep
    
[^10]: 学习多个动态系统中的联合问题

    Joint Problems in Learning Multiple Dynamical Systems

    [https://arxiv.org/abs/2311.02181](https://arxiv.org/abs/2311.02181)

    聚类时间序列的新问题，提出联合划分轨迹集并学习每个部分的线性动态系统模型，以最小化所有模型的最大误差

    

    时间序列的聚类是一个经过充分研究的问题，其应用范围从通过代谢产物浓度获得的定量个性化代谢模型到量子信息理论中的状态判别。我们考虑了一个变种，即给定一组轨迹和一些部分，我们联合划分轨迹集并学习每个部分的线性动态系统（LDS）模型，以使得所有模型的最大误差最小化。我们提出了全局收敛的方法和EM启发式算法，并附上了有前景的计算结果。

    arXiv:2311.02181v2 Announce Type: replace-cross  Abstract: Clustering of time series is a well-studied problem, with applications ranging from quantitative, personalized models of metabolism obtained from metabolite concentrations to state discrimination in quantum information theory. We consider a variant, where given a set of trajectories and a number of parts, we jointly partition the set of trajectories and learn linear dynamical system (LDS) models for each part, so as to minimize the maximum error across all the models. We present globally convergent methods and EM heuristics, accompanied by promising computational results.
    
[^11]: 用于图学习中的拓扑概括的流动扩散变压器

    Advective Diffusion Transformers for Topological Generalization in Graph Learning. (arXiv:2310.06417v1 [cs.LG])

    [http://arxiv.org/abs/2310.06417](http://arxiv.org/abs/2310.06417)

    本研究探索了在不同的图拓扑存在下，图扩散方程如何对GNN进行外推和概括，揭示了基于局部扩散的现有模型在概括能力上的不足，并提出了非局部扩散的潜力。

    

    图扩散方程与图神经网络（GNN）密切相关，并且最近引起了人们的关注，作为分析GNN动力学、形式化其表达能力和证明架构选择的有原则的框架。图学习中的一个关键问题是GNN的概括能力。当前方法的一个主要限制在于假设训练集和测试集中的图拓扑来自相同的分布。本文通过探索图扩散方程在不同图拓扑存在下的外推和概括能力，迈出了解析GNN概括性的一步。我们首先展示了基于图上局部扩散的现有模型在概括能力上的不足，这是由于对拓扑变化的指数敏感性引起的。随后的分析揭示了非局部扩散的潜力，它倡导对完全连接的潜在图进行特征传播。

    Graph diffusion equations are intimately related to graph neural networks (GNNs) and have recently attracted attention as a principled framework for analyzing GNN dynamics, formalizing their expressive power, and justifying architectural choices. One key open questions in graph learning is the generalization capabilities of GNNs. A major limitation of current approaches hinges on the assumption that the graph topologies in the training and test sets come from the same distribution. In this paper, we make steps towards understanding the generalization of GNNs by exploring how graph diffusion equations extrapolate and generalize in the presence of varying graph topologies. We first show deficiencies in the generalization capability of existing models built upon local diffusion on graphs, stemming from the exponential sensitivity to topology variation. Our subsequent analysis reveals the promise of non-local diffusion, which advocates for feature propagation over fully-connected latent gr
    
[^12]: 损失平坦性与神经网络中压缩表示的简单联系

    A simple connection from loss flatness to compressed representations in neural networks. (arXiv:2310.01770v1 [cs.LG])

    [http://arxiv.org/abs/2310.01770](http://arxiv.org/abs/2310.01770)

    该论文研究了深度神经网络中损失平坦性和神经表示压缩之间的关系，通过简单的数学关系，证明了损失平坦性与神经表示的压缩相关。

    

    对深度神经网络的泛化能力进行研究的方法有很多种，包括至少两种不同的方法：一种基于参数空间中损失景观的形状，另一种基于特征空间中表示流形的结构（即单位活动的空间）。这两种方法相关但很少同时进行研究和明确关联。在这里，我们提出了一种简单的分析方法来建立这种联系。我们展示了在深度神经网络学习的最后阶段，神经表示流形的体积压缩与正在进行的参数优化所探索的最小值周围的损失平坦性相关。我们证明了这可以由一个相对简单的数学关系来预测：损失平坦性意味着神经表示的压缩。我们的结果与\citet{ma_linear_2021}的先前研究密切相关，该研究展示了平坦性（即小特征值）与表示流形的体积压缩之间的关系。

    Deep neural networks' generalization capacity has been studied in a variety of ways, including at least two distinct categories of approach: one based on the shape of the loss landscape in parameter space, and the other based on the structure of the representation manifold in feature space (that is, in the space of unit activities). These two approaches are related, but they are rarely studied together and explicitly connected. Here, we present a simple analysis that makes such a connection. We show that, in the last phase of learning of deep neural networks, compression of the volume of the manifold of neural representations correlates with the flatness of the loss around the minima explored by ongoing parameter optimization. We show that this is predicted by a relatively simple mathematical relationship: loss flatness implies compression of neural representations. Our results build closely on prior work of \citet{ma_linear_2021}, which shows how flatness (i.e., small eigenvalues of t
    
[^13]: 大型语言模型在中国劳动市场的应用

    Large Language Models at Work in China's Labor Market. (arXiv:2308.08776v1 [econ.GN])

    [http://arxiv.org/abs/2308.08776](http://arxiv.org/abs/2308.08776)

    本文研究了大型语言模型（LLMs）对中国劳动市场的潜在影响，并分析了职业对LLM能力的暴露程度及其与工资水平和经验溢价之间的关系。研究结果表明，高薪和经验密集型工作可能面临更大的替代风险。此外，研究还开发了一个考虑行业暴露的经济增长模型，以量化AI采用对生产力和就业之间的权衡。这项研究为理解中国劳动市场中越来越强大的AI系统的影响提供了基础。

    

    本文探讨了大型语言模型（LLMs）对中国劳动市场的潜在影响。我们通过结合人类专业知识和LLM分类，按照Eloundou等人（2023）的方法分析了职业对LLM能力的暴露程度。然后将职业暴露程度聚合到行业水平上，得到行业暴露得分。结果表明，职业暴露和工资水平/经验溢价之间存在正相关关系，表明高薪和经验密集型的工作可能面临着LLM驱动软件的更大替代风险。行业暴露得分与专家评估和经济直觉相一致。我们还开发了一个考虑行业暴露的经济增长模型，以量化AI采用带来的生产力和就业之间的权衡。总体来说，本研究为理解中国越来越强大的AI系统对劳动市场的影响提供了分析基础。主要创新包括职业水平的暴露情况。

    This paper explores the potential impacts of large language models (LLMs) on the Chinese labor market. We analyze occupational exposure to LLM capabilities by incorporating human expertise and LLM classifications, following Eloundou et al. (2023)'s methodology. We then aggregate occupation exposure to the industry level to obtain industry exposure scores. The results indicate a positive correlation between occupation exposure and wage levels/experience premiums, suggesting higher-paying and experience-intensive jobs may face greater displacement risks from LLM-powered software. The industry exposure scores align with expert assessments and economic intuitions. We also develop an economic growth model incorporating industry exposure to quantify the productivity-employment trade-off from AI adoption. Overall, this study provides an analytical basis for understanding the labor market impacts of increasingly capable AI systems in China. Key innovations include the occupation-level exposure
    
[^14]: 当基础模型遇到联邦学习：动机、挑战和未来方向

    When Foundation Model Meets Federated Learning: Motivations, Challenges, and Future Directions. (arXiv:2306.15546v1 [cs.LG])

    [http://arxiv.org/abs/2306.15546](http://arxiv.org/abs/2306.15546)

    基础模型与联邦学习的交叉提供了解锁新可能性的独特机会，扩展了数据可用性，促进了协作式模型发展，并提高了性能和隐私保护。

    

    基础模型（FM）与联邦学习（FL）的交叉提供了相互的好处，在AI研究中提供了解锁新可能性的独特机会，解决了AI和现实世界应用中的关键挑战。FL扩展了FM的数据可用性，并实现了计算共享，分散了训练过程，并减轻了FL参与者的负担。它促进了协作式FM发展，民主化了这一过程，促进了包容性和创新。另一方面，FM以其庞大的规模、预训练的知识和出色的性能，为FL提供了一个强大的起点，促进了在非独立同分布数据下更快的收敛和更好的性能。此外，利用FM生成合成数据可以丰富数据多样性，减少过拟合，保护隐私。通过研究FL和FM之间的相互作用，本文旨在加深对它们协同关系的理解，强调动机和挑战。

    The intersection of the Foundation Model (FM) and Federated Learning (FL) provides mutual benefits, presents a unique opportunity to unlock new possibilities in AI research, and address critical challenges in AI and real-world applications. FL expands the availability of data for FMs and enables computation sharing, distributing the training process and reducing the burden on FL participants. It promotes collaborative FM development, democratizing the process and fostering inclusivity and innovation. On the other hand, FM, with its enormous size, pre-trained knowledge, and exceptional performance, serves as a robust starting point for FL, facilitating faster convergence and better performance under non-iid data. Additionally, leveraging FM to generate synthetic data enriches data diversity, reduces overfitting, and preserves privacy. By examining the interplay between FL and FM, this paper aims to deepen the understanding of their synergistic relationship, highlighting the motivations,
    
[^15]: 一种适用于所有时间序列的Transformer：表示和训练具有时间相关的异构表格数据

    One Transformer for All Time Series: Representing and Training with Time-Dependent Heterogeneous Tabular Data. (arXiv:2302.06375v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.06375](http://arxiv.org/abs/2302.06375)

    本研究提出了一种Transformer架构，用于表示具有时间相关的异构表格数据，通过使用一组频率函数来表示数值特征，并采用唯一的损失函数进行统一训练。

    

    近年来，将深度学习技术应用于表格数据的兴趣日益增长，以复制其他人工智能领域在这一结构化领域的成功。特别有趣的是，表格数据具有时间依赖性，例如金融交易。然而，表格值的异质性，其中类别元素与数值项混合，使得这种适应变得困难。在本文中，我们提出了一种Transformer架构来表示异构的时间相关的表格数据，数值特征使用一组频率函数表示，并且整个网络使用唯一的损失函数进行统一训练。

    There is a recent growing interest in applying Deep Learning techniques to tabular data, in order to replicate the success of other Artificial Intelligence areas in this structured domain. Specifically interesting is the case in which tabular data have a time dependence, such as, for instance financial transactions. However, the heterogeneity of the tabular values, in which categorical elements are mixed with numerical items, makes this adaptation difficult. In this paper we propose a Transformer architecture to represent heterogeneous time-dependent tabular data, in which numerical features are represented using a set of frequency functions and the whole network is uniformly trained with a unique loss function.
    

