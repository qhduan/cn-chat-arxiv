# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Tiny Models are the Computational Saver for Large Models](https://arxiv.org/abs/2403.17726) | TinySaver是一种动态模型压缩方法，通过使用小型模型来自适应地替换大型模型，从而提高计算效率。 |
| [^2] | [Advanced Artificial Intelligence Algorithms in Cochlear Implants: Review of Healthcare Strategies, Challenges, and Perspectives](https://arxiv.org/abs/2403.15442) | 人工智能在提高植入式听觉设备的语音质量方面具有前瞻性，并通过先进的信号处理技术以及应对多源语音和环境噪音挑战等方法来克服语音失真问题 |
| [^3] | [A minimal coalition logic](https://arxiv.org/abs/2403.14704) | 提出了一个基于一般并发博弈模型的最小联盟逻辑，不具备传统模型中的独立性、序列性和确定性假设，展示了其完备性并与传统模型进行了比较 |
| [^4] | [Are LLMs Good Cryptic Crossword Solvers?](https://arxiv.org/abs/2403.12094) | 本文建立了三种流行LLMs的基准结果，表明它们在难解填字游戏上的表现仍远远不及人类。 |
| [^5] | [Efficient Pruning of Large Language Model with Adaptive Estimation Fusion](https://arxiv.org/abs/2403.10799) | 提出了一种简单而高效的剪枝方法，能够自适应地模拟每个子结构的重要性，并根据多层结构的结果自适应地融合粗粒度和细粒度的估计。 |
| [^6] | [Content-aware Masked Image Modeling Transformer for Stereo Image Compression](https://arxiv.org/abs/2403.08505) | 提出了一种名为CAMSIC的立体图像压缩框架，通过引入面向内容感知的掩码图像建模（MIM）技术，使得无需额外Transformer解码器就能捕捉空间和视差依赖关系，实验结果表明实现了最先进的率失真结果。 |
| [^7] | [ACPO: AI-Enabled Compiler-Driven Program Optimization](https://arxiv.org/abs/2312.09982) | 该论文提出了ACPO框架，通过机器学习模型提供给LLVM简单全面的工具，以实现编译器驱动的程序优化。 |
| [^8] | [Self-Supervised Deconfounding Against Spatio-Temporal Shifts: Theory and Modeling](https://arxiv.org/abs/2311.12472) | 该论文针对时空数据中常见的分布变化问题提出了一种自监督去混淆方法并提出了名为DCA的理论解决方案。 |
| [^9] | [Towards Real-time Training of Physics-informed Neural Networks: Applications in Ultrafast Ultrasound Blood Flow Imaging.](http://arxiv.org/abs/2309.04755) | 本研究提出了一种实时训练基于物理信息的神经网络（PINN）的框架，用于解决Navier-Stokes方程，以实现超快速超声血流成像。该框架将Navier-Stokes方程离散化为稳态，并通过迁移学习顺序求解稳态方程。此外，采用平均恒定随机梯度下降作为初始化，并提出了一种并行训练方案，适用于所有时间戳。 |
| [^10] | [ChatGPT Needs SPADE (Sustainability, PrivAcy, Digital divide, and Ethics) Evaluation: A Review.](http://arxiv.org/abs/2305.03123) | 本文研究关注ChatGPT面临的可持续性、隐私、数字鸿沟和伦理问题，提出了SPADE评估的必要性，并给出了缓解和建议。 |

# 详细

[^1]: 小型模型是大型模型的计算节省者

    Tiny Models are the Computational Saver for Large Models

    [https://arxiv.org/abs/2403.17726](https://arxiv.org/abs/2403.17726)

    TinySaver是一种动态模型压缩方法，通过使用小型模型来自适应地替换大型模型，从而提高计算效率。

    

    本文介绍了TinySaver，一种类似于早期退出的动态模型压缩方法，它使用小型模型来自适应地替换大型模型。与传统的压缩技术不同，像TinySaver这样的动态方法可以利用难度差异，使得某些输入能够提前完成推理过程，从而节省计算资源。大多数现有的早期退出设计是通过向模型的骨干结构附加额外的网络分支来实现的。然而，我们的研究揭示了完全独立的小型模型可以在对性能影响最小的情况下替代较大模型的大部分工作。将它们作为第一个退出点可以显著提高计算效率。通过搜索并使用最合适的小型模型作为给定大型模型的计算节省者，所提出的方法作为一种新颖且通用的模型压缩方法。

    arXiv:2403.17726v1 Announce Type: new  Abstract: This paper introduces TinySaver, an early-exit-like dynamic model compression approach which employs tiny models to substitute large models adaptively. Distinct from traditional compression techniques, dynamic methods like TinySaver can leverage the difficulty differences to allow certain inputs to complete their inference processes early, thereby conserving computational resources. Most existing early exit designs are implemented by attaching additional network branches to the model's backbone. Our study, however, reveals that completely independent tiny models can replace a substantial portion of the larger models' job with minimal impact on performance. Employing them as the first exit can remarkably enhance computational efficiency. By searching and employing the most appropriate tiny model as the computational saver for a given large model, the proposed approaches work as a novel and generic method to model compression. This finding
    
[^2]: 人工智能在耳蜗植入装置中的先进算法：医疗策略、挑战和展望综述

    Advanced Artificial Intelligence Algorithms in Cochlear Implants: Review of Healthcare Strategies, Challenges, and Perspectives

    [https://arxiv.org/abs/2403.15442](https://arxiv.org/abs/2403.15442)

    人工智能在提高植入式听觉设备的语音质量方面具有前瞻性，并通过先进的信号处理技术以及应对多源语音和环境噪音挑战等方法来克服语音失真问题

    

    arXiv:2403.15442v1 公告类型: 跨领域 摘要: 自动语音识别（ASR）在我们的日常生活中发挥着至关重要的作用，不仅为与机器交互提供了便利，还为部分或完全听力受损的个体提供了沟通的机会。这一过程涉及以模拟形式接收语音信号，然后通过各种信号处理算法使其与容量有限的设备（如CI）兼容。然而，这些配备有有限数量电极的植入装置在合成过程中往往导致语音失真。尽管研究人员在使用各种最先进的信号处理技术改善接收到的语音质量方面做出了努力，但在涉及多个语音源、环境噪声和其他情况的场景中，挑战仍然存在。新人工智能（AI）方法的出现引入了先进的策略来解决这些限制。

    arXiv:2403.15442v1 Announce Type: cross  Abstract: Automatic speech recognition (ASR) plays a pivotal role in our daily lives, offering utility not only for interacting with machines but also for facilitating communication for individuals with either partial or profound hearing impairments. The process involves receiving the speech signal in analogue form, followed by various signal processing algorithms to make it compatible with devices of limited capacity, such as cochlear implants (CIs). Unfortunately, these implants, equipped with a finite number of electrodes, often result in speech distortion during synthesis. Despite efforts by researchers to enhance received speech quality using various state-of-the-art signal processing techniques, challenges persist, especially in scenarios involving multiple sources of speech, environmental noise, and other circumstances. The advent of new artificial intelligence (AI) methods has ushered in cutting-edge strategies to address the limitations
    
[^3]: 一个最小联盟逻辑

    A minimal coalition logic

    [https://arxiv.org/abs/2403.14704](https://arxiv.org/abs/2403.14704)

    提出了一个基于一般并发博弈模型的最小联盟逻辑，不具备传统模型中的独立性、序列性和确定性假设，展示了其完备性并与传统模型进行了比较

    

    联盟逻辑是战略推理研究中的一个中心逻辑。本文首先指出联盟逻辑模型，即并发博弈模型，存在三个过于强的假设。其一是代理的独立性；即，两个不同联盟的两个可用联合动作的合并总是可以合并成两个联盟的联盟。其二是序列性；即，联盟总是有可用的联合动作。其三是确定性，即，大联盟的合作动作总是有唯一结果。其次，我们提出了一个基于一般并发博弈模型的联盟逻辑，该模型不具备这三个假设。我们展示了这一逻辑的完备性，并与联盟逻辑进行了详细比较。在战略推理的背景下，这一逻辑似乎是最小的。

    arXiv:2403.14704v1 Announce Type: cross  Abstract: Coalition logic is a central logic in strategic reasoning studies. In this paper, we first argue that Coalition Logic models, concurrent game models, have three too-strong assumptions. The first one is the independence of agents; that is, the merge of two available joint actions of two disjoint coalitions is always available for the union of the two coalitions. The second one is seriality; that is, coalitions always have available joint actions. The third one is determinism, that is, the grand coalition's joint actions always have a unique outcome. Second, we present a coalition logic based on general concurrent game models, which do not have the three assumptions. We show the completeness of this logic and compare it with Coalition Logic in detail. This logic seems minimal in the context of strategic reasoning.
    
[^4]: LLMs是一个好的难解填字游戏求解器吗？

    Are LLMs Good Cryptic Crossword Solvers?

    [https://arxiv.org/abs/2403.12094](https://arxiv.org/abs/2403.12094)

    本文建立了三种流行LLMs的基准结果，表明它们在难解填字游戏上的表现仍远远不及人类。

    

    难解填字游戏是一种谜题，不仅依赖于一般知识，还依赖于求解者在不同层面上操纵语言并处理各种类型的文字游戏。先前的研究表明，即使对于现代NLP模型来说，解决这类谜题也是一项挑战。然而，大型语言模型（LLMs）的能力尚未在这一任务上进行测试。在本文中，我们为三种流行的LLMs -- LLaMA2、Mistral和ChatGPT建立了基准结果，显示它们在这一任务上的表现仍远远不及人类。

    arXiv:2403.12094v1 Announce Type: new  Abstract: Cryptic crosswords are puzzles that rely not only on general knowledge but also on the solver's ability to manipulate language on different levels and deal with various types of wordplay. Previous research suggests that solving such puzzles is a challenge even for modern NLP models. However, the abilities of large language models (LLMs) have not yet been tested on this task. In this paper, we establish the benchmark results for three popular LLMs -- LLaMA2, Mistral, and ChatGPT -- showing that their performance on this task is still far from that of humans.
    
[^5]: 使用自适应估计融合高效剪枝大型语言模型

    Efficient Pruning of Large Language Model with Adaptive Estimation Fusion

    [https://arxiv.org/abs/2403.10799](https://arxiv.org/abs/2403.10799)

    提出了一种简单而高效的剪枝方法，能够自适应地模拟每个子结构的重要性，并根据多层结构的结果自适应地融合粗粒度和细粒度的估计。

    

    大型语言模型（LLMs）已经成为许多生成性下游任务中至关重要的组成部分，这导致在资源受限设备上高效部署它们成为不可避免的趋势和重大挑战。结构化剪枝是解决这一挑战的广泛应用方法。然而，当处理多个解码器层的复杂结构时，通常的方法往往采用常见的估计方法进行剪枝。这些方法导致特定下游任务精度下降。本文介绍了一种简单而有效的方法，可自适应地模拟每个子结构的重要性。同时，它可以基于复杂和多层结构的结果，自适应地融合粗粒度和细粒度的估计。我们设计的所有方面都无缝集成到端到端的剪枝框架中。与主流数据集上的最先进方法相比，我们的实验结果表明

    arXiv:2403.10799v1 Announce Type: cross  Abstract: Large language models (LLMs) have become crucial for many generative downstream tasks, leading to an inevitable trend and significant challenge to deploy them efficiently on resource-constrained devices. Structured pruning is a widely used method to address this challenge. However, when dealing with the complex structure of the multiple decoder layers, general methods often employ common estimation approaches for pruning. These approaches lead to a decline in accuracy for specific downstream tasks. In this paper, we introduce a simple yet efficient method that adaptively models the importance of each substructure. Meanwhile, it can adaptively fuse coarse-grained and finegrained estimations based on the results from complex and multilayer structures. All aspects of our design seamlessly integrate into the endto-end pruning framework. Our experimental results, compared with state-of-the-art methods on mainstream datasets, demonstrate ave
    
[^6]: 面向内容感知的掩码图像建模变压器用于立体图像压缩

    Content-aware Masked Image Modeling Transformer for Stereo Image Compression

    [https://arxiv.org/abs/2403.08505](https://arxiv.org/abs/2403.08505)

    提出了一种名为CAMSIC的立体图像压缩框架，通过引入面向内容感知的掩码图像建模（MIM）技术，使得无需额外Transformer解码器就能捕捉空间和视差依赖关系，实验结果表明实现了最先进的率失真结果。

    

    现有基于学习的立体图像编解码器采用了复杂的转换方法，但在编码潜在表示时却采用了从单个图像编解码器导出的简单熵模型。然而，这些熵模型难以有效捕捉立体图像固有的空间-视差特征，导致亚最优的率失真结果。本文提出了一种名为CAMSIC的立体图像压缩框架。 CAMSIC 独立地将每个图像转换为潜在表示，并采用强大的无解码器变压器熵模型来捕捉空间和视差依赖关系，引入了一种新颖的面向内容感知的掩码图像建模（MIM）技术。我们的面向内容感知的MIM促进了先验信息与估计令牌之间的高效双向交互，自然地消除了额外的Transformer解码器的需求。实验证明，我们的立体图像编解码器实现了最先进的率失真结果。

    arXiv:2403.08505v1 Announce Type: cross  Abstract: Existing learning-based stereo image codec adopt sophisticated transformation with simple entropy models derived from single image codecs to encode latent representations. However, those entropy models struggle to effectively capture the spatial-disparity characteristics inherent in stereo images, which leads to suboptimal rate-distortion results. In this paper, we propose a stereo image compression framework, named CAMSIC. CAMSIC independently transforms each image to latent representation and employs a powerful decoder-free Transformer entropy model to capture both spatial and disparity dependencies, by introducing a novel content-aware masked image modeling (MIM) technique. Our content-aware MIM facilitates efficient bidirectional interaction between prior information and estimated tokens, which naturally obviates the need for an extra Transformer decoder. Experiments show that our stereo image codec achieves state-of-the-art rate-d
    
[^7]: ACPO: AI-Enabled Compiler-Driven Program Optimization

    ACPO: AI-Enabled Compiler-Driven Program Optimization

    [https://arxiv.org/abs/2312.09982](https://arxiv.org/abs/2312.09982)

    该论文提出了ACPO框架，通过机器学习模型提供给LLVM简单全面的工具，以实现编译器驱动的程序优化。

    

    该论文提出了ACPO：AI-Enabled Compiler-driven Program Optimization，这是一个新颖的框架，为LLVM提供简单全面的工具，以从应用机器学习模型来进行不同的优化通路中获益。首先展示了ACPO的高层视图、类层次结构和功能，然后通过将循环展开和函数内联传递的ML使能化，展示了ACPO的一些用例，描述了ACPO如何发挥作用。

    arXiv:2312.09982v2 Announce Type: replace-cross  Abstract: The key to performance optimization of a program is to decide correctly when a certain transformation should be applied by a compiler. This is an ideal opportunity to apply machine-learning models to speed up the tuning process; while this realization has been around since the late 90s, only recent advancements in ML enabled a practical application of ML to compilers as an end-to-end framework.   This paper presents ACPO: \textbf{\underline{A}}I-Enabled \textbf{\underline{C}}ompiler-driven \textbf{\underline{P}}rogram \textbf{\underline{O}}ptimization; a novel framework to provide LLVM with simple and comprehensive tools to benefit from employing ML models for different optimization passes. We first showcase the high-level view, class hierarchy, and functionalities of ACPO and subsequently, demonstrate a couple of use cases of ACPO by ML-enabling the Loop Unroll and Function Inlining passes and describe how ACPO can be leverage
    
[^8]: 针对时空偏移的自监督去混淆：理论与建模

    Self-Supervised Deconfounding Against Spatio-Temporal Shifts: Theory and Modeling

    [https://arxiv.org/abs/2311.12472](https://arxiv.org/abs/2311.12472)

    该论文针对时空数据中常见的分布变化问题提出了一种自监督去混淆方法并提出了名为DCA的理论解决方案。

    

    作为时空（ST）数据的重要应用，ST交通预测在提高城市出行效率和促进可持续发展中起着至关重要的作用。本文首先通过构建过去交通数据、未来交通数据和外部ST上下文的因果图，系统地阐明了过去艺术作品在OOD交通数据上的失败是由于ST上下文充当了混淆因素，即过去数据和未来数据的共同原因。然后，我们从因果角度提出了一种理论解决方案，称为Disentangled Contextual Adjustment（DCA）。

    arXiv:2311.12472v2 Announce Type: replace  Abstract: As an important application of spatio-temporal (ST) data, ST traffic forecasting plays a crucial role in improving urban travel efficiency and promoting sustainable development. In practice, the dynamics of traffic data frequently undergo distributional shifts attributed to external factors such as time evolution and spatial differences. This entails forecasting models to handle the out-of-distribution (OOD) issue where test data is distributed differently from training data. In this work, we first formalize the problem by constructing a causal graph of past traffic data, future traffic data, and external ST contexts. We reveal that the failure of prior arts in OOD traffic data is due to ST contexts acting as a confounder, i.e., the common cause for past data and future ones. Then, we propose a theoretical solution named Disentangled Contextual Adjustment (DCA) from a causal lens. It differentiates invariant causal correlations again
    
[^9]: 实时训练基于物理信息的神经网络：超快速超声血流成像的应用

    Towards Real-time Training of Physics-informed Neural Networks: Applications in Ultrafast Ultrasound Blood Flow Imaging. (arXiv:2309.04755v1 [cs.CE])

    [http://arxiv.org/abs/2309.04755](http://arxiv.org/abs/2309.04755)

    本研究提出了一种实时训练基于物理信息的神经网络（PINN）的框架，用于解决Navier-Stokes方程，以实现超快速超声血流成像。该框架将Navier-Stokes方程离散化为稳态，并通过迁移学习顺序求解稳态方程。此外，采用平均恒定随机梯度下降作为初始化，并提出了一种并行训练方案，适用于所有时间戳。

    

    基于物理信息的神经网络（PINN）是纳维-斯托克斯方程的最杰出求解器之一，而纳维-斯托克斯方程广泛应用于血流的控制方程。然而，目前的方法仅依赖于完整的纳维-斯托克斯方程，对于超快速多普勒超声，这一最新技术应用于\emph{体内}复杂血流动力学的展示，每秒获取数千帧（或时间戳），这是不切实际的。本文首先提出了一种新的PINN训练框架，通过将纳维-斯托克斯方程离散化为稳态，并通过迁移学习顺序求解稳态纳维-斯托克斯方程，为解决纳维-斯托克斯方程提供了新的训练框架，称为SeqPINN。在SeqPINN的成功基础上，我们采用了平均恒定随机梯度下降（SGD）作为初始化的思想，并提出了一种并行训练方案，适用于所有时间戳。为了确保良好的泛化初始化，我们借鉴了

    Physics-informed Neural Network (PINN) is one of the most preeminent solvers of Navier-Stokes equations, which are widely used as the governing equation of blood flow. However, current approaches, relying on full Navier-Stokes equations, are impractical for ultrafast Doppler ultrasound, the state-of-the-art technique for depiction of complex blood flow dynamics \emph{in vivo} through acquired thousands of frames (or, timestamps) per second. In this article, we first propose a novel training framework of PINN for solving Navier-Stokes equations by discretizing Navier-Stokes equations into steady state and sequentially solving steady-state Navier-Stokes equations with transfer learning. The novel training framework is coined as SeqPINN. Upon the success of SeqPINN, we adopt the idea of averaged constant stochastic gradient descent (SGD) as initialization and propose a parallel training scheme for all timestamps. To ensure an initialization that generalizes well, we borrow the concept of 
    
[^10]: ChatGPT 需要进行SPADE（可持续性、隐私、数字鸿沟和伦理）评估：一项综述。

    ChatGPT Needs SPADE (Sustainability, PrivAcy, Digital divide, and Ethics) Evaluation: A Review. (arXiv:2305.03123v1 [cs.CY])

    [http://arxiv.org/abs/2305.03123](http://arxiv.org/abs/2305.03123)

    本文研究关注ChatGPT面临的可持续性、隐私、数字鸿沟和伦理问题，提出了SPADE评估的必要性，并给出了缓解和建议。

    

    ChatGPT是另一个大型语言模型（LLM），由于其性能和有效的对话能力，在研究和工业界中得到了巨大的关注。最近，许多研究已经发表，以展示ChatGPT和其他LLMs的有效性、效率、集成和情感。相反，本研究关注的是大多数被忽视的重要方面，即可持续性、隐私、数字鸿沟和伦理，并建议不仅仅是ChatGPT，而是在对话机器人类别中的每一个后续入口都应该进行SPADE评估。本文详细讨论了关于ChatGPT的问题和关注点与上述特征一致。我们通过一些初步的数据收集和可视化以及假设的事实来支持我们的假设。我们还为每个问题提出了缓解和建议。此外，我们还提供了一些未来方向和开放问题的探讨。

    ChatGPT is another large language model (LLM) inline but due to its performance and ability to converse effectively, it has gained a huge popularity amongst research as well as industrial community. Recently, many studies have been published to show the effectiveness, efficiency, integration, and sentiments of chatGPT and other LLMs. In contrast, this study focuses on the important aspects that are mostly overlooked, i.e. sustainability, privacy, digital divide, and ethics and suggests that not only chatGPT but every subsequent entry in the category of conversational bots should undergo Sustainability, PrivAcy, Digital divide, and Ethics (SPADE) evaluation. This paper discusses in detail about the issues and concerns raised over chatGPT in line with aforementioned characteristics. We support our hypothesis by some preliminary data collection and visualizations along with hypothesized facts. We also suggest mitigations and recommendations for each of the concerns. Furthermore, we also s
    

