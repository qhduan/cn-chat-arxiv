# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Unsolvable Problem Detection: Evaluating Trustworthiness of Vision Language Models](https://arxiv.org/abs/2403.20331) | 本文提出了一个新颖且重要的挑战，即Unsolvable Problem Detection（UPD），用于评估视觉语言模型在视觉问答任务中能否在面对不可解问题时保持答案的能力，并通过广泛实验发现大多数模型存在改进的空间。 |
| [^2] | [Object-Centric Domain Randomization for 3D Shape Reconstruction in the Wild](https://arxiv.org/abs/2403.14539) | 提出了ObjectDR，利用对象-centric的域随机化合成单视图3D形状重建中缺乏的配对数据，通过条件生成模型和解耦框架来生成和保留对象轮廓以及广泛变化的数据，从而为培训模型捕捉域不变性几何形状。 |
| [^3] | [Feint in Multi-Player Games](https://arxiv.org/abs/2403.07932) | 该论文介绍了多人游戏中假动作的首次形式化、实现和定量评估，并证明其能够显著提高奖励收益、增加游戏多样性，且时间消耗方面开销很小。 |
| [^4] | [Artificial Intelligence for Complex Network: Potential, Methodology and Application](https://arxiv.org/abs/2402.16887) | 人工智能技术与丰富真实网络数据的存在开启了复杂网络科学研究的新时代，有望克服现存挑战。 |
| [^5] | [VideoPrism: A Foundational Visual Encoder for Video Understanding](https://arxiv.org/abs/2402.13217) | VideoPrism是一个通用的视频编码器，通过全局-局部语义视频嵌入的蒸馏和标记混洗方案，在多个视频理解任务上取得了最新技术水平的表现。 |
| [^6] | [Masked Attention is All You Need for Graphs](https://arxiv.org/abs/2402.10793) | 提出了一种在图上学习的简单替代方法，称为掩码注意力（MAG），其利用注意力矩阵来创建定制的注意力模式，在长距离任务上表现出色并胜过其他方法。 |
| [^7] | [Provably Sample Efficient RLHF via Active Preference Optimization](https://arxiv.org/abs/2402.10500) | 通过Active Preference Optimization算法，在Bradley-Terry-Luce偏好模型下实现了RLHF的样本效率提高，优化了对提示收集偏好数据的策略。 |
| [^8] | [Link Prediction with Relational Hypergraphs](https://arxiv.org/abs/2402.04062) | 本文提出了两种适用于关系超图的链接预测框架，并通过实证分析验证了这些框架在各种关系超图基准测试上的有效性。 |
| [^9] | [MacroSwarm: A Field-based Compositional Framework for Swarm Programming.](http://arxiv.org/abs/2401.10969) | MacroSwarm是一种基于场的群体编程框架，通过可组合的功能模块实现复杂的群体行为，通过将感知场映射为执行目标场，提供了一种系统化的设计和实现群体行为的方法。 |
| [^10] | [RSAM: Learning on manifolds with Riemannian Sharpness-aware Minimization.](http://arxiv.org/abs/2309.17215) | RSAM是一种在流形上学习的算法，通过将Sharpness-Aware Minimization (SAM)推广到Riemannian流形，引入了流形上sharpness的概念，并通过理论分析证明了其与泛化能力的关系。通过该算法的应用，我们展示了其在提升泛化能力方面的有效性。 |
| [^11] | [Dual-Modal Attention-Enhanced Text-Video Retrieval with Triplet Partial Margin Contrastive Learning.](http://arxiv.org/abs/2309.11082) | 本文提出了一种双模态注意增强的文本-视频检索方法，通过引入新颖的对比学习技术，能够准确衡量跨模态相似性和挖掘难负样本。 |
| [^12] | [Packet Header Recognition Utilizing an All-Optical Reservoir Based on Reinforcement-Learning-Optimized Double-Ring Resonator.](http://arxiv.org/abs/2308.13818) | 本文提出了一种基于双环谐振器的全光储备系统，通过使用深度强化学习算法，最大化延迟-带宽积参数，实现了快速、准确的光分组头识别。优化后的级联环可用于实现小尺寸、平顶延迟谱的光计算。 |
| [^13] | [A Deep RL Approach on Task Placement and Scaling of Edge Resources for Cellular Vehicle-to-Network Service Provisioning.](http://arxiv.org/abs/2305.09832) | 本文提出了一种基于深度强化学习的分散式方法，用于解决车联网服务提供中的任务部署和边缘资源的扩展问题。 |

# 详细

[^1]: 不可解问题检测：评估视觉语言模型的可信度

    Unsolvable Problem Detection: Evaluating Trustworthiness of Vision Language Models

    [https://arxiv.org/abs/2403.20331](https://arxiv.org/abs/2403.20331)

    本文提出了一个新颖且重要的挑战，即Unsolvable Problem Detection（UPD），用于评估视觉语言模型在视觉问答任务中能否在面对不可解问题时保持答案的能力，并通过广泛实验发现大多数模型存在改进的空间。

    

    本文介绍了一个新颖而重要的挑战，即Unsolvable Problem Detection（UPD），用于评估视觉语言模型（VLMs）在视觉问答（VQA）任务中面对不可解问题时保持答案的能力。UPD包括三个不同的设置：缺失答案检测（AAD）、不兼容答案集检测（IASD）和不兼容视觉问题检测（IVQD）。通过广泛的实验深入研究UPD问题表明，大多数VLMs，包括GPT-4V和LLaVA-Next-34B，在各种程度上都很难应对我们的基准测试，突显了改进的重要空间。为了解决UPD，我们探索了无需训练和基于训练的解决方案，提供了对其有效性和局限性的新见解。我们希望我们的见解，以及在提议的UPD设置内的未来努力，将增强对VLMs的更广泛理解和发展。

    arXiv:2403.20331v1 Announce Type: cross  Abstract: This paper introduces a novel and significant challenge for Vision Language Models (VLMs), termed Unsolvable Problem Detection (UPD). UPD examines the VLM's ability to withhold answers when faced with unsolvable problems in the context of Visual Question Answering (VQA) tasks. UPD encompasses three distinct settings: Absent Answer Detection (AAD), Incompatible Answer Set Detection (IASD), and Incompatible Visual Question Detection (IVQD). To deeply investigate the UPD problem, extensive experiments indicate that most VLMs, including GPT-4V and LLaVA-Next-34B, struggle with our benchmarks to varying extents, highlighting significant room for the improvements. To address UPD, we explore both training-free and training-based solutions, offering new insights into their effectiveness and limitations. We hope our insights, together with future efforts within the proposed UPD settings, will enhance the broader understanding and development of
    
[^2]: Object-Centric Domain Randomization用于野外3D形状重建

    Object-Centric Domain Randomization for 3D Shape Reconstruction in the Wild

    [https://arxiv.org/abs/2403.14539](https://arxiv.org/abs/2403.14539)

    提出了ObjectDR，利用对象-centric的域随机化合成单视图3D形状重建中缺乏的配对数据，通过条件生成模型和解耦框架来生成和保留对象轮廓以及广泛变化的数据，从而为培训模型捕捉域不变性几何形状。

    

    单视图3D形状在野外的重建面临的最大挑战之一是来自真实环境中的<3D形状，2D图像>-配对数据的稀缺性。受域随机化引人注目的成就的启发，我们提出了ObjectDR，通过对对象外观和背景的视觉变化进行随机仿真，合成这种配对数据。我们的数据合成框架利用条件生成模型（例如ControlNet）生成符合空间条件（例如2.5D草图）的图像，这些条件可以通过从对象集合（例如Objaverse-XL）的渲染过程获得3D形状。为了模拟多样化的变化同时保留嵌入空间条件中的对象轮廓，我们还引入了一个利用初始对象指导的解耦框架。

    arXiv:2403.14539v1 Announce Type: cross  Abstract: One of the biggest challenges in single-view 3D shape reconstruction in the wild is the scarcity of <3D shape, 2D image>-paired data from real-world environments. Inspired by remarkable achievements via domain randomization, we propose ObjectDR which synthesizes such paired data via a random simulation of visual variations in object appearances and backgrounds. Our data synthesis framework exploits a conditional generative model (e.g., ControlNet) to generate images conforming to spatial conditions such as 2.5D sketches, which are obtainable through a rendering process of 3D shapes from object collections (e.g., Objaverse-XL). To simulate diverse variations while preserving object silhouettes embedded in spatial conditions, we also introduce a disentangled framework which leverages an initial object guidance. After synthesizing a wide range of data, we pre-train a model on them so that it learns to capture a domain-invariant geometry p
    
[^3]: 多人游戏中的假动作

    Feint in Multi-Player Games

    [https://arxiv.org/abs/2403.07932](https://arxiv.org/abs/2403.07932)

    该论文介绍了多人游戏中假动作的首次形式化、实现和定量评估，并证明其能够显著提高奖励收益、增加游戏多样性，且时间消耗方面开销很小。

    

    这篇论文介绍了对多人游戏中的假动作进行了首次形式化、实现和定量评估。我们首先从多人游戏的角度对假动作进行了形式化，涉及到时间、空间和它们的集体影响。该形式化建立在非传递性主动马尔可夫游戏模型之上，其中假动作能够产生可观的影响。接下来，我们考虑了在多代理建模的最新进展下（即多智能体强化学习）在多人游戏中实施假动作的实际细节。最后，我们定量检验了我们设计的有效性，结果显示我们的假动作设计可以（1）显著提高游戏中的奖励收益；（2）显著提高多人游戏的多样性；以及（3）仅在时间消耗方面产生可忽略的开销。我们得出结论，我们的假动作设计是有效的。

    arXiv:2403.07932v1 Announce Type: cross  Abstract: This paper introduces the first formalization, implementation and quantitative evaluation of Feint in Multi-Player Games. Our work first formalizes Feint from the perspective of Multi-Player Games, in terms of the temporal, spatial, and their collective impacts. The formalization is built upon Non-transitive Active Markov Game Model, where Feint can have a considerable amount of impacts. Then, our work considers practical implementation details of Feint in Multi-Player Games, under the state-of-the-art progress of multi-agent modeling to date (namely Multi-Agent Reinforcement Learning). Finally, our work quantitatively examines the effectiveness of our design, and the results show that our design of Feint can (1) greatly improve the reward gains from the game; (2) significantly improve the diversity of Multi-Player Games; and (3) only incur negligible overheads in terms of time consumption. We conclude that our design of Feint is effec
    
[^4]: 复杂网络的人工智能：潜力、方法论和应用

    Artificial Intelligence for Complex Network: Potential, Methodology and Application

    [https://arxiv.org/abs/2402.16887](https://arxiv.org/abs/2402.16887)

    人工智能技术与丰富真实网络数据的存在开启了复杂网络科学研究的新时代，有望克服现存挑战。

    

    复杂网络存在于各种真实世界系统中，从自然环境到人类社会。这些网络的本质在于它们能够从微观混乱-其中网络拓扑和节点动态交织-转变和演化为具有特定集体行为的宏观秩序。在过去的二十年里，复杂网络科学显著增强了我们对真实世界网络潜在机制、结构和动态的理解。尽管取得了这些进展，但在探索更加真实系统和提升实际应用方面仍然存在着相当大的挑战。人工智能技术的出现，以及丰富多样的真实世界网络数据的存在，开启了复杂网络科学研究的新时代。本调查旨在系统地探讨人工智能在克服复杂网络科学研究所面临的挑战方面的潜在优势。

    arXiv:2402.16887v1 Announce Type: cross  Abstract: Complex networks pervade various real-world systems, from the natural environment to human societies. The essence of these networks is in their ability to transition and evolve from microscopic disorder-where network topology and node dynamics intertwine-to a macroscopic order characterized by certain collective behaviors. Over the past two decades, complex network science has significantly enhanced our understanding of the statistical mechanics, structures, and dynamics underlying real-world networks. Despite these advancements, there remain considerable challenges in exploring more realistic systems and enhancing practical applications. The emergence of artificial intelligence (AI) technologies, coupled with the abundance of diverse real-world network data, has heralded a new era in complex network science research. This survey aims to systematically address the potential advantages of AI in overcoming the lingering challenges of com
    
[^5]: VideoPrism: 用于视频理解的基础视觉编码器

    VideoPrism: A Foundational Visual Encoder for Video Understanding

    [https://arxiv.org/abs/2402.13217](https://arxiv.org/abs/2402.13217)

    VideoPrism是一个通用的视频编码器，通过全局-局部语义视频嵌入的蒸馏和标记混洗方案，在多个视频理解任务上取得了最新技术水平的表现。

    

    我们引入了VideoPrism，一个通用的视频编码器，使用单个冻结模型处理多样的视频理解任务。我们在包含3600万高质量视频标题对和58.2亿个带有嘈杂平行文本（如ASR转录）的视频剪辑的异构语料库上对VideoPrism进行预训练。预训练方法通过全局-局部语义视频嵌入的蒸馏和一个标记混洗方案改进了掩码自编码，使VideoPrism能够主要专注于视频模态同时利用与视频相关联的宝贵文本。我们在四个广泛的视频理解任务组上进行了对VideoPrism的广泛测试，从网络视频问答到科学CV， 在33个视频理解基准测试中的30个上实现了最新技术水平的性能。

    arXiv:2402.13217v1 Announce Type: cross  Abstract: We introduce VideoPrism, a general-purpose video encoder that tackles diverse video understanding tasks with a single frozen model. We pretrain VideoPrism on a heterogeneous corpus containing 36M high-quality video-caption pairs and 582M video clips with noisy parallel text (e.g., ASR transcripts). The pretraining approach improves upon masked autoencoding by global-local distillation of semantic video embeddings and a token shuffling scheme, enabling VideoPrism to focus primarily on the video modality while leveraging the invaluable text associated with videos. We extensively test VideoPrism on four broad groups of video understanding tasks, from web video question answering to CV for science, achieving state-of-the-art performance on 30 out of 33 video understanding benchmarks.
    
[^6]: 掩码注意力是图的关键

    Masked Attention is All You Need for Graphs

    [https://arxiv.org/abs/2402.10793](https://arxiv.org/abs/2402.10793)

    提出了一种在图上学习的简单替代方法，称为掩码注意力（MAG），其利用注意力矩阵来创建定制的注意力模式，在长距离任务上表现出色并胜过其他方法。

    

    图神经网络（GNNs）和消息传递算法的变种主要用于在图上学习，这在很大程度上归功于它们的灵活性、速度和令人满意的性能。然而，设计强大而通用的GNNs需要大量的研究工作，通常依赖于精心选择的手工制作的消息传递操作符。受此启发，我们提出了一种在图上学习的非常简单的替代方法，它完全依赖于注意力。图被表示为节点或边集，并通过掩码注意权重矩阵来强制它们的连接，有效地为每个图创建定制的注意力模式。尽管其简单性，用于图的掩码注意力（MAG）在长距离任务上表现出色，并在55多个节点和图级任务上优于强消息传递基线和更复杂的基于注意力的方法。

    arXiv:2402.10793v1 Announce Type: cross  Abstract: Graph neural networks (GNNs) and variations of the message passing algorithm are the predominant means for learning on graphs, largely due to their flexibility, speed, and satisfactory performance. The design of powerful and general purpose GNNs, however, requires significant research efforts and often relies on handcrafted, carefully-chosen message passing operators. Motivated by this, we propose a remarkably simple alternative for learning on graphs that relies exclusively on attention. Graphs are represented as node or edge sets and their connectivity is enforced by masking the attention weight matrix, effectively creating custom attention patterns for each graph. Despite its simplicity, masked attention for graphs (MAG) has state-of-the-art performance on long-range tasks and outperforms strong message passing baselines and much more involved attention-based methods on over 55 node and graph-level tasks. We also show significantly 
    
[^7]: 通过主动偏好优化实现经验证的样本效率的RLHF

    Provably Sample Efficient RLHF via Active Preference Optimization

    [https://arxiv.org/abs/2402.10500](https://arxiv.org/abs/2402.10500)

    通过Active Preference Optimization算法，在Bradley-Terry-Luce偏好模型下实现了RLHF的样本效率提高，优化了对提示收集偏好数据的策略。

    

    强化学习从人类反馈（RLHF）在将大型语言模型（LLMs）与人类偏好相一致方面至关重要。虽然这些对齐的生成模型已经在各种任务中展示出令人印象深刻的能力，但是依赖高质量的人类偏好数据在实际RLHF实施中构成了昂贵的瓶颈。因此，需要更好和自适应的数据收集策略。为此，我们将RLHF以上下文偏好赌博机问题的形式框定，其中提示作为上下文，并表明通过随机选择提示收集偏好数据的天真方式导致一个在奖励方面具有$\Omega(1)$次优性差距的策略。然后，我们提出了$\textit{Active Preference Optimization}$（$\texttt{APO}$）算法，该算法积极选择提示以收集偏好数据。在Bradley-Terry-Luce（BTL）偏好模型下，\texttt{APO}实现了样本效率，而不会妥协于polic

    arXiv:2402.10500v1 Announce Type: cross  Abstract: Reinforcement Learning from Human Feedback (RLHF) is pivotal in aligning Large Language Models (LLMs) with human preferences. While these aligned generative models have demonstrated impressive capabilities across various tasks, the dependence on high-quality human preference data poses a costly bottleneck in practical implementation of RLHF. Hence better and adaptive strategies for data collection is needed. To this end, we frame RLHF as a contextual preference bandit problem with prompts as contexts and show that the naive way of collecting preference data by choosing prompts uniformly at random leads to a policy that suffers an $\Omega(1)$ suboptimality gap in rewards. Then we propose $\textit{Active Preference Optimization}$ ($\texttt{APO}$), an algorithm that actively selects prompts to collect preference data. Under the Bradley-Terry-Luce (BTL) preference model, \texttt{APO} achieves sample efficiency without compromising on polic
    
[^8]: 使用关系超图进行链接预测

    Link Prediction with Relational Hypergraphs

    [https://arxiv.org/abs/2402.04062](https://arxiv.org/abs/2402.04062)

    本文提出了两种适用于关系超图的链接预测框架，并通过实证分析验证了这些框架在各种关系超图基准测试上的有效性。

    

    对于使用知识图谱进行链接预测已经进行了深入的研究，导致了具有成功应用的图神经网络体系结构的丰富景观。然而，将这些体系结构的成功转移到使用关系超图进行链接预测仍然具有挑战性。关系超边的存在使得链接预测成为在不同选择的k个节点之间的任务，这比使用知识图谱进行链接预测要困难得多，因为每个关系都是二进制的（k=2）。在本文中，我们提出了两个使用关系超图进行链接预测的框架，并通过相应的关系Weisfeiler-Leman算法以及一些自然逻辑形式对生成的模型体系结构的表达能力进行了彻底分析。通过广泛的实证分析，我们验证了提出的模型体系结构在各种关系超图基准测试上的能力。

    Link prediction with knowledge graphs has been thoroughly studied in graph machine learning, leading to a rich landscape of graph neural network architectures with successful applications. Nonetheless, it remains challenging to transfer the success of these architectures to link prediction with relational hypergraphs. The presence of relational hyperedges makes link prediction a task between $k$ nodes for varying choices of $k$, which is substantially harder than link prediction with knowledge graphs, where every relation is binary ($k=2$). In this paper, we propose two frameworks for link prediction with relational hypergraphs and conduct a thorough analysis of the expressive power of the resulting model architectures via corresponding relational Weisfeiler-Leman algorithms, and also via some natural logical formalisms. Through extensive empirical analysis, we validate the power of the proposed model architectures on various relational hypergraph benchmarks. The resulting model archit
    
[^9]: MacroSwarm: 一种基于场的组合框架用于群体编程

    MacroSwarm: A Field-based Compositional Framework for Swarm Programming. (arXiv:2401.10969v1 [cs.AI])

    [http://arxiv.org/abs/2401.10969](http://arxiv.org/abs/2401.10969)

    MacroSwarm是一种基于场的群体编程框架，通过可组合的功能模块实现复杂的群体行为，通过将感知场映射为执行目标场，提供了一种系统化的设计和实现群体行为的方法。

    

    群体行为工程是一项旨在研究协调简单智能体团体内计算和行动的方法和技术，以实现复杂的全局目标，如图案形成、集体移动、聚类和分布式感知。尽管在群体（无人机、机器人、车辆）分析和工程方面取得了一些进展，但仍然需要通用的设计和实现方法和工具，以系统化的方式定义复杂的群体行为。为了对此做出贡献，本文提出了一种新的基于场的协调方法，称为MacroSwarm，以可重用且完全可组合的功能模块为基础，嵌入集体计算和协调。基于集成计算的宏编程范式，MacroSwarm提出了将每个群体行为块表示为将感知场映射为执行目标场的纯函数的思路。

    Swarm behaviour engineering is an area of research that seeks to investigate methods and techniques for coordinating computation and action within groups of simple agents to achieve complex global goals like pattern formation, collective movement, clustering, and distributed sensing. Despite recent progress in the analysis and engineering of swarms (of drones, robots, vehicles), there is still a need for general design and implementation methods and tools that can be used to define complex swarm behaviour in a principled way. To contribute to this quest, this article proposes a new field-based coordination approach, called MacroSwarm, to design and program swarm behaviour in terms of reusable and fully composable functional blocks embedding collective computation and coordination. Based on the macroprogramming paradigm of aggregate computing, MacroSwarm builds on the idea of expressing each swarm behaviour block as a pure function mapping sensing fields into actuation goal fields, e.g.
    
[^10]: RSAM：使用Riemannian Sharpness-aware Minimization在流形上学习

    RSAM: Learning on manifolds with Riemannian Sharpness-aware Minimization. (arXiv:2309.17215v1 [cs.LG])

    [http://arxiv.org/abs/2309.17215](http://arxiv.org/abs/2309.17215)

    RSAM是一种在流形上学习的算法，通过将Sharpness-Aware Minimization (SAM)推广到Riemannian流形，引入了流形上sharpness的概念，并通过理论分析证明了其与泛化能力的关系。通过该算法的应用，我们展示了其在提升泛化能力方面的有效性。

    

    如今，了解损失函数空间的几何结构有望提升模型的泛化能力。在这项工作中，我们借鉴了之前将几何原理应用于优化的研究，并提出了一种改进受限制优化问题鲁棒性和泛化能力的新方法。事实上，本文旨在将Sharpness-Aware Minimization (SAM)优化器推广到Riemannian流形。为了支持流形上的“sharpness”概念，我们首先对流形上的sharpness引入了一种新的定义。为了证明这个概念的有效性，我们提出了一个理论分析来描述流形sharpness与泛化能力之间的关系，并呈现了一个更紧密的泛化缺口上限，这是之前未知的结果。受到这个分析的启发，我们提出了我们的算法，Riemannian Sharpness-Aware Minimization (RSAM)。为了展示RSAM在提升泛化能力方面的能力，我们在一个约束优化问题上评估和对比了我们的算法。

    Nowadays, understanding the geometry of the loss landscape shows promise in enhancing a model's generalization ability. In this work, we draw upon prior works that apply geometric principles to optimization and present a novel approach to improve robustness and generalization ability for constrained optimization problems. Indeed, this paper aims to generalize the Sharpness-Aware Minimization (SAM) optimizer to Riemannian manifolds. In doing so, we first extend the concept of sharpness and introduce a novel notion of sharpness on manifolds. To support this notion of sharpness, we present a theoretical analysis characterizing generalization capabilities with respect to manifold sharpness, which demonstrates a tighter bound on the generalization gap, a result not known before. Motivated by this analysis, we introduce our algorithm, Riemannian Sharpness-Aware Minimization (RSAM). To demonstrate RSAM's ability to enhance generalization ability, we evaluate and contrast our algorithm on a br
    
[^11]: 双模态注意增强的文本-视频检索与三元部分边际对比学习

    Dual-Modal Attention-Enhanced Text-Video Retrieval with Triplet Partial Margin Contrastive Learning. (arXiv:2309.11082v1 [cs.CV])

    [http://arxiv.org/abs/2309.11082](http://arxiv.org/abs/2309.11082)

    本文提出了一种双模态注意增强的文本-视频检索方法，通过引入新颖的对比学习技术，能够准确衡量跨模态相似性和挖掘难负样本。

    

    近年来，网络视频的爆炸性增长使得文本-视频检索对于视频过滤、推荐和搜索变得越来越重要和流行。文本-视频检索旨在将相关的文本/视频排在不相关的文本/视频之前。该任务的核心是准确衡量文本和视频之间的跨模态相似性。最近，对比学习方法在文本-视频检索方面显示出有希望的结果，其中大部分方法侧重于构建正负样本对以学习文本和视频表示。然而，他们在关注难负样本和模拟不同层次的语义相似性方面不够，存在两个问题。为了解决这两个问题，本文使用两个新方法改进了对比学习。首先，为了利用艰难的例子来提高鲁棒的判别能力，我们提出了一种新颖的双模态注意增强模块(DMAE)，从文本和视觉线索中挖掘难负样本。通过进一步引入一个负面样本筛选机制，该方法可以建模不同级别的语义相似性。

    In recent years, the explosion of web videos makes text-video retrieval increasingly essential and popular for video filtering, recommendation, and search. Text-video retrieval aims to rank relevant text/video higher than irrelevant ones. The core of this task is to precisely measure the cross-modal similarity between texts and videos. Recently, contrastive learning methods have shown promising results for text-video retrieval, most of which focus on the construction of positive and negative pairs to learn text and video representations. Nevertheless, they do not pay enough attention to hard negative pairs and lack the ability to model different levels of semantic similarity. To address these two issues, this paper improves contrastive learning using two novel techniques. First, to exploit hard examples for robust discriminative power, we propose a novel Dual-Modal Attention-Enhanced Module (DMAE) to mine hard negative pairs from textual and visual clues. By further introducing a Negat
    
[^12]: 基于强化学习优化的双环谐振器的全光储备的分组头识别

    Packet Header Recognition Utilizing an All-Optical Reservoir Based on Reinforcement-Learning-Optimized Double-Ring Resonator. (arXiv:2308.13818v1 [eess.SP])

    [http://arxiv.org/abs/2308.13818](http://arxiv.org/abs/2308.13818)

    本文提出了一种基于双环谐振器的全光储备系统，通过使用深度强化学习算法，最大化延迟-带宽积参数，实现了快速、准确的光分组头识别。优化后的级联环可用于实现小尺寸、平顶延迟谱的光计算。

    

    光分组头识别是光通信网络中重要的信号处理任务。本文提出了一个全光储备系统，其中集成了双环谐振器作为节点，用于快速准确地识别光分组头。由于节点的延迟-带宽积（DBP）是储备系统中的重要参数，我们采用深度强化学习算法，对不同类型的双环谐振器进行DBP的最大化，该算法具有全参数空间优化和快速收敛速度的优势。有趣的是，级联、并联和嵌入式配置的优化DBP达到了相同的最大值，被认为是全局最大值。最后，使用优化的级联环组成的全光储备系统进行了3位和6位分组头识别任务，该系统具有大大减小的芯片尺寸和所需的“平顶”延迟谱。利用这种光计算方案，字错误率

    Optical packet header recognition is an important signal processing task of optical communication networks. In this work, we propose an all-optical reservoir, consisting of integrated double-ring resonators (DRRs) as nodes, for fast and accurate optical packet header recognition. As the delay-bandwidth product (DBP) of the node is a key figure-of-merit in the reservoir, we adopt a deep reinforcement learning algorithm to maximize the DBPs for various types of DRRs, which has the advantage of full parameter space optimization and fast convergence speed. Intriguingly, the optimized DBPs of the DRRs in cascaded, parallel, and embedded configurations reach the same maximum value, which is believed to be the global maximum. Finally, 3-bit and 6-bit packet header recognition tasks are performed with the all-optical reservoir consisting of the optimized cascaded rings, which have greatly reduced chip size and the desired "flat-top" delay spectra. Using this optical computing scheme, word-erro
    
[^13]: 基于深度强化学习的边缘资源任务部署和扩展方法用于车载网络服务提供

    A Deep RL Approach on Task Placement and Scaling of Edge Resources for Cellular Vehicle-to-Network Service Provisioning. (arXiv:2305.09832v1 [cs.AI])

    [http://arxiv.org/abs/2305.09832](http://arxiv.org/abs/2305.09832)

    本文提出了一种基于深度强化学习的分散式方法，用于解决车联网服务提供中的任务部署和边缘资源的扩展问题。

    

    “车联网”正处于我们社会数字化转型的前沿。本文提出了一种分散式方法用于提供车辆通联网（C-V2N）服务，解决服务任务部署和边缘资源的扩展问题。我们证明了这个联合问题的复杂性，并提出了一个两个问题的联接方式，采用了基于贪心算法的关于任务部署的方法和基于 Deep Deterministic Policy Gradient (DDPG) 的扩展方法。本文还对我们的方法进行了基准测试，重点关注了扩展代理与多个状态下最先进的扩展方法的性能比较。

    Cellular-Vehicle-to-Everything (C-V2X) is currently at the forefront of the digital transformation of our society. By enabling vehicles to communicate with each other and with the traffic environment using cellular networks, we redefine transportation, improving road safety and transportation services, increasing efficiency of traffic flows, and reducing environmental impact. This paper proposes a decentralized approach for provisioning Cellular Vehicular-to-Network (C-V2N) services, addressing the coupled problems of service task placement and scaling of edge resources. We formalize the joint problem and prove its complexity. We propose an approach to tackle it, linking the two problems, employing decentralized decision-making using (i) a greedy approach for task placement and (ii) a Deep Deterministic Policy Gradient (DDPG) based approach for scaling. We benchmark the performance of our approach, focusing on the scaling agent, against several State-of-the-Art (SoA) scaling approaches
    

