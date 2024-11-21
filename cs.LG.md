# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning the Market: Sentiment-Based Ensemble Trading Agents](https://rss.arxiv.org/abs/2402.01441) | 该论文提出了将情感分析和深度强化学习集成算法应用于股票交易的方法，设计了可以根据市场情绪动态调整的交易策略。实验结果表明，这种方法比传统的集成策略、单一智能体算法和市场指标更具盈利性、稳健性和风险最小化。相关研究还发现，传统的固定更换集成智能体的做法并不是最优的，而基于情感的动态框架可以显著提高交易智能体的性能。 |
| [^2] | [Neuron Patching: Neuron-level Model Editing on Code Generation and LLMs](https://rss.arxiv.org/abs/2312.05356) | 这项工作介绍了一种神经元层面的模型编辑方法，能够在编码任务中修补LLM模型，并且在API序列推荐、代码生成和伪代码到代码转换等任务中得到了验证和评估。 |
| [^3] | [PDE-CNNs: Axiomatic Derivations and Applications](https://arxiv.org/abs/2403.15182) | PDE-CNNs通过利用几何意义的演化PDE的求解器替代传统的组件，提供了更少的参数、固有的等变性、更好的性能、数据效率和几何可解释性。 |
| [^4] | [Select High-Level Features: Efficient Experts from a Hierarchical Classification Network](https://arxiv.org/abs/2403.05601) | 提出了一种新颖的专家生成方法，通过选择高级特征实现动态降低任务和计算复杂性，为未来设计轻量级和适应性强的网络铺平了道路 |
| [^5] | [Generating Visual Stimuli from EEG Recordings using Transformer-encoder based EEG encoder and GAN](https://arxiv.org/abs/2402.10115) | 本研究使用基于Transformer编码器的EEG编码器和GAN网络，通过合成图像从EEG信号中恢复出各种对象类别的图像，同时结合对抗损失和感知损失，提高生成图像的质量。 |
| [^6] | [RanDumb: A Simple Approach that Questions the Efficacy of Continual Representation Learning](https://arxiv.org/abs/2402.08823) | RanDumb是一种简单的方法，通过固定的随机变换嵌入原始像素并学习简单的线性分类器，质疑了持续表示学习的效果。 实验结果显示，RanDumb在众多持续学习基准测试中明显优于使用深度网络进行持续学习的表示学习。 |
| [^7] | [Operator learning without the adjoint](https://arxiv.org/abs/2401.17739) | 本论文提出了一种不需要探测伴随算子的算子学习方法，通过在Fourier基上进行投影来逼近一类非自伴随的无限维紧算子，并应用于恢复椭圆型偏微分算子的格林函数。这是第一个试图填补算子学习理论与实践差距的无需伴随算子分析。 |
| [^8] | [GDL-DS: A Benchmark for Geometric Deep Learning under Distribution Shifts.](http://arxiv.org/abs/2310.08677) | GDL-DS是一个基准测试，用于评估几何深度学习模型在具有分布转换的场景中的性能。它包括多个科学领域的评估数据集，并研究了不同级别的超出分布特征的信息访问。 |
| [^9] | [Delegating Data Collection in Decentralized Machine Learning.](http://arxiv.org/abs/2309.01837) | 这项研究在分散机器学习生态系统中研究了委托的数据收集问题，通过设计最优契约解决了模型质量评估的不确定性和对最优性能缺乏预先知识的挑战。 |
| [^10] | [Bandits with Deterministically Evolving States.](http://arxiv.org/abs/2307.11655) | 该论文提出了一种名为具有确定性演化状态的强盗模型，用于学习带有强盗反馈的推荐系统和在线广告。该模型考虑了状态演化的不同速率，能准确评估奖励与系统健康程度之间的关系。 |
| [^11] | [Smart Pressure e-Mat for Human Sleeping Posture and Dynamic Activity Recognition.](http://arxiv.org/abs/2305.11367) | 本文介绍了一种基于Velostat的智能压力电子垫系统，可用于识别人体姿势和运动，具有高精度。 |
| [^12] | [Deep quantum neural networks form Gaussian processes.](http://arxiv.org/abs/2305.09957) | 本文证明了基于Haar随机酉或正交深量子神经网络的某些模型的输出会收敛于高斯过程。然而，这种高斯过程不能用于通过贝叶斯统计学来有效预测QNN的输出。 |
| [^13] | [Stochastic Approximation Approaches to Group Distributionally Robust Optimization.](http://arxiv.org/abs/2302.09267) | 本文提出了一种随机逼近法，用于组分布式鲁棒优化，该算法利用在线学习技术，将每轮所需的样本数从$m$个降至$1$个，同时保持相同的样本复杂度。 |
| [^14] | [Generalization on the Unseen, Logic Reasoning and Degree Curriculum.](http://arxiv.org/abs/2301.13105) | 本文研究了在逻辑推理任务中对未知数据的泛化能力，提供了网络架构在该设置下的表现证据，发现了一类网络模型在未知数据上学习了最小度插值器，并对长度普通化现象提供了解释。 |
| [^15] | [Revisiting Discrete Soft Actor-Critic.](http://arxiv.org/abs/2209.10081) | 本研究重新审视了将连续动作空间的Soft Actor-Critic方法调整为离散动作空间的问题，并提出了解决Q值低估和性能不稳定的方法，验证了其在Atari游戏和大规模MOBA游戏中的有效性。 |

# 详细

[^1]: 学习市场：基于情感的集成交易智能体

    Learning the Market: Sentiment-Based Ensemble Trading Agents

    [https://rss.arxiv.org/abs/2402.01441](https://rss.arxiv.org/abs/2402.01441)

    该论文提出了将情感分析和深度强化学习集成算法应用于股票交易的方法，设计了可以根据市场情绪动态调整的交易策略。实验结果表明，这种方法比传统的集成策略、单一智能体算法和市场指标更具盈利性、稳健性和风险最小化。相关研究还发现，传统的固定更换集成智能体的做法并不是最优的，而基于情感的动态框架可以显著提高交易智能体的性能。

    

    我们提出了将情感分析和深度强化学习集成算法应用于股票交易，并设计了一种能够根据当前市场情绪动态调整所使用的智能体的策略。我们创建了一个简单但有效的方法来提取新闻情感，并将其与对现有作品的一般改进结合起来，从而得到有效考虑定性市场因素和定量股票数据的自动交易智能体。我们证明了我们的方法导致了一种盈利、稳健且风险最小的策略，优于传统的集成策略以及单一智能体算法和市场指标。我们的发现表明，传统的每隔固定月份更换集成智能体的做法并不是最优的，基于情感的动态框架极大地提升了这些智能体的性能。此外，由于我们的算法设计简单且...

    We propose the integration of sentiment analysis and deep-reinforcement learning ensemble algorithms for stock trading, and design a strategy capable of dynamically altering its employed agent given concurrent market sentiment. In particular, we create a simple-yet-effective method for extracting news sentiment and combine this with general improvements upon existing works, resulting in automated trading agents that effectively consider both qualitative market factors and quantitative stock data. We show that our approach results in a strategy that is profitable, robust, and risk-minimal -- outperforming the traditional ensemble strategy as well as single agent algorithms and market metrics. Our findings determine that the conventional practice of switching ensemble agents every fixed-number of months is sub-optimal, and that a dynamic sentiment-based framework greatly unlocks additional performance within these agents. Furthermore, as we have designed our algorithm with simplicity and
    
[^2]: Neuron Patching: 神经元层面的模型编辑与代码生成

    Neuron Patching: Neuron-level Model Editing on Code Generation and LLMs

    [https://rss.arxiv.org/abs/2312.05356](https://rss.arxiv.org/abs/2312.05356)

    这项工作介绍了一种神经元层面的模型编辑方法，能够在编码任务中修补LLM模型，并且在API序列推荐、代码生成和伪代码到代码转换等任务中得到了验证和评估。

    

    大型语言模型在软件工程中得到了成功应用，特别是在代码生成方面。更新这些模型的新知识非常昂贵，通常需要全面实现其价值。在本文中，我们提出了一种新颖有效的模型编辑方法MENT，用于在编码任务中修补LLM模型。基于生成式LLM的机制，MENT可以在预测下一个令牌时进行模型编辑，并进一步支持常见的编码任务。MENT具有高效、有效和可靠的特点。它可以通过修补1或2个神经元来纠正神经模型。作为神经元层面上生成模型编辑的先驱工作，我们规范了编辑过程并介绍了相关概念。此外，我们还引入了新的衡量方法来评估其泛化能力，并建立了一个用于进一步研究的基准。我们的方法在三个编码任务上进行了评估，包括API序列推荐、行级代码生成和伪代码到代码转换。

    Large Language Models are successfully adopted in software engineering, especially in code generation. Updating these models with new knowledge is very expensive, and is often required to fully realize their value. In this paper, we propose a novel and effective model editing approach, \textsc{MENT}, to patch LLMs in coding tasks. Based on the mechanism of generative LLMs, \textsc{MENT} enables model editing in next-token predictions, and further supports common coding tasks. \textsc{MENT} is effective, efficient, and reliable. It can correct a neural model by patching 1 or 2 neurons. As the pioneer work on neuron-level model editing of generative models, we formalize the editing process and introduce the involved concepts. Besides, we also introduce new measures to evaluate its generalization ability, and build a benchmark for further study. Our approach is evaluated on three coding tasks, including API-seq recommendation, line-level code generation, and pseudocode-to-code transaction
    
[^3]: PDE-CNNs：公理推导与应用

    PDE-CNNs: Axiomatic Derivations and Applications

    [https://arxiv.org/abs/2403.15182](https://arxiv.org/abs/2403.15182)

    PDE-CNNs通过利用几何意义的演化PDE的求解器替代传统的组件，提供了更少的参数、固有的等变性、更好的性能、数据效率和几何可解释性。

    

    基于偏微分方程组卷积神经网络（PDE-G-CNNs）利用具有几何意义的演化偏微分方程的求解器替代G-CNNs中常规组件。PDE-G-CNNs同时提供了几个关键优势：更少的参数、固有等变性、更好的性能、数据效率和几何可解释性。本文重点研究特征图在整个网络中为二维的欧几里德等变PDE-G-CNNs。我们将这个框架的变体称为PDE-CNN。我们列出了几个在实践中令人满意的公理，并从中推导出应在PDE-CNN中使用哪些PDE。在这里，我们通过经典线性和形态尺度空间理论的公理受启发，通过引入半域值信号对其进行推广。此外，我们通过实验证实，相对于小型网络，PDE-CNN提供了更少的参数、更好的性能和数据效率。

    arXiv:2403.15182v1 Announce Type: new  Abstract: PDE-based Group Convolutional Neural Networks (PDE-G-CNNs) utilize solvers of geometrically meaningful evolution PDEs as substitutes for the conventional components in G-CNNs. PDE-G-CNNs offer several key benefits all at once: fewer parameters, inherent equivariance, better performance, data efficiency, and geometric interpretability. In this article we focus on Euclidean equivariant PDE-G-CNNs where the feature maps are two dimensional throughout. We call this variant of the framework a PDE-CNN. We list several practically desirable axioms and derive from these which PDEs should be used in a PDE-CNN. Here our approach to geometric learning via PDEs is inspired by the axioms of classical linear and morphological scale-space theory, which we generalize by introducing semifield-valued signals. Furthermore, we experimentally confirm for small networks that PDE-CNNs offer fewer parameters, better performance, and data efficiency in compariso
    
[^4]: 选择高级特征：分层分类网络中的高效专家

    Select High-Level Features: Efficient Experts from a Hierarchical Classification Network

    [https://arxiv.org/abs/2403.05601](https://arxiv.org/abs/2403.05601)

    提出了一种新颖的专家生成方法，通过选择高级特征实现动态降低任务和计算复杂性，为未来设计轻量级和适应性强的网络铺平了道路

    

    这项研究介绍了一种新颖的专家生成方法，可以动态减少任务和计算复杂性，同时不影响预测性能。它基于一种新的分层分类网络拓扑结构，将通用低级特征的顺序处理与高级特征的并行处理和嵌套相结合。这种结构允许创新的特征提取技术：能够选择与任务相关类别的高级特征。在某些情况下，几乎可以跳过所有不必要的高级特征，这可以显著减少推理成本，在资源受限的条件下非常有益。我们相信这种方法为未来轻量级和可适应的网络设计铺平了道路，使其适用于从紧凑边缘设备到大型云端的各种应用。在动态推理方面，我们的方法可以实现排除

    arXiv:2403.05601v1 Announce Type: new  Abstract: This study introduces a novel expert generation method that dynamically reduces task and computational complexity without compromising predictive performance. It is based on a new hierarchical classification network topology that combines sequential processing of generic low-level features with parallelism and nesting of high-level features. This structure allows for the innovative extraction technique: the ability to select only high-level features of task-relevant categories. In certain cases, it is possible to skip almost all unneeded high-level features, which can significantly reduce the inference cost and is highly beneficial in resource-constrained conditions. We believe this method paves the way for future network designs that are lightweight and adaptable, making them suitable for a wide range of applications, from compact edge devices to large-scale clouds. In terms of dynamic inference our methodology can achieve an exclusion 
    
[^5]: 使用基于Transformer编码器的EEG编码器和GAN从EEG记录中生成视觉刺激

    Generating Visual Stimuli from EEG Recordings using Transformer-encoder based EEG encoder and GAN

    [https://arxiv.org/abs/2402.10115](https://arxiv.org/abs/2402.10115)

    本研究使用基于Transformer编码器的EEG编码器和GAN网络，通过合成图像从EEG信号中恢复出各种对象类别的图像，同时结合对抗损失和感知损失，提高生成图像的质量。

    

    在这项研究中，我们解决了感知性脑解码领域的一个现代研究挑战，即使用对抗式深度学习框架从EEG信号中合成图像。具体目标是利用主体观看图像时获得的EEG记录重新创建属于各种对象类别的图像。为了实现这一目标，我们使用基于Transformer编码器的EEG编码器生成EEG编码，然后将其作为GAN网络的生成器组件的输入。除了对抗损失之外，我们还采用了感知损失来提高生成图像的质量。

    arXiv:2402.10115v1 Announce Type: new  Abstract: In this study, we tackle a modern research challenge within the field of perceptual brain decoding, which revolves around synthesizing images from EEG signals using an adversarial deep learning framework. The specific objective is to recreate images belonging to various object categories by leveraging EEG recordings obtained while subjects view those images. To achieve this, we employ a Transformer-encoder based EEG encoder to produce EEG encodings, which serve as inputs to the generator component of the GAN network. Alongside the adversarial loss, we also incorporate perceptual loss to enhance the quality of the generated images.
    
[^6]: RanDumb: 一种质疑持续表示学习效果的简单方法

    RanDumb: A Simple Approach that Questions the Efficacy of Continual Representation Learning

    [https://arxiv.org/abs/2402.08823](https://arxiv.org/abs/2402.08823)

    RanDumb是一种简单的方法，通过固定的随机变换嵌入原始像素并学习简单的线性分类器，质疑了持续表示学习的效果。 实验结果显示，RanDumb在众多持续学习基准测试中明显优于使用深度网络进行持续学习的表示学习。

    

    我们提出了RanDumb来检验持续表示学习的效果。RanDumb将原始像素使用一个固定的随机变换嵌入，这个变换近似了RBF-Kernel，在看到任何数据之前初始化，并学习一个简单的线性分类器。我们提出了一个令人惊讶且一致的发现：在众多持续学习基准测试中，RanDumb在性能上明显优于使用深度网络进行持续学习的表示学习，这表明在这些情景下表示学习的性能较差。RanDumb不存储样本，并在数据上进行单次遍历，一次处理一个样本。它与GDumb相辅相成，在GDumb性能特别差的低样本情况下运行。当将RanDumb扩展到使用预训练模型替换随机变换的情景时，我们得出相同一致的结论。我们的调查结果既令人惊讶又令人担忧，因为表示学习在这些情况下表现糟糕。

    arXiv:2402.08823v1 Announce Type: cross Abstract: We propose RanDumb to examine the efficacy of continual representation learning. RanDumb embeds raw pixels using a fixed random transform which approximates an RBF-Kernel, initialized before seeing any data, and learns a simple linear classifier on top. We present a surprising and consistent finding: RanDumb significantly outperforms the continually learned representations using deep networks across numerous continual learning benchmarks, demonstrating the poor performance of representation learning in these scenarios. RanDumb stores no exemplars and performs a single pass over the data, processing one sample at a time. It complements GDumb, operating in a low-exemplar regime where GDumb has especially poor performance. We reach the same consistent conclusions when RanDumb is extended to scenarios with pretrained models replacing the random transform with pretrained feature extractor. Our investigation is both surprising and alarming as
    
[^7]: 不需要伴随算子的算子学习

    Operator learning without the adjoint

    [https://arxiv.org/abs/2401.17739](https://arxiv.org/abs/2401.17739)

    本论文提出了一种不需要探测伴随算子的算子学习方法，通过在Fourier基上进行投影来逼近一类非自伴随的无限维紧算子，并应用于恢复椭圆型偏微分算子的格林函数。这是第一个试图填补算子学习理论与实践差距的无需伴随算子分析。

    

    算子学习中存在一个谜团：如何在没有探测伴随算子的情况下从数据中恢复非自伴随算子？目前的实际方法表明，在仅使用由算子的正向作用生成的数据的情况下，可以准确地恢复算子，而不需要访问伴随算子。然而，以直观的方式看，似乎有必要采样伴随算子的作用。在本文中，我们部分解释了这个谜团，通过证明在不查询伴随算子的情况下，可以通过在Fourier基上进行投影来逼近一类非自伴随的无限维紧算子。然后，我们将该结果应用于恢复椭圆型偏微分算子的格林函数，并导出一个无需伴随算子的样本复杂度界限。虽然现有的理论证明了算子学习的低样本复杂度，但我们的是第一个试图填补理论与实践差距的无需伴随算子的分析。

    There is a mystery at the heart of operator learning: how can one recover a non-self-adjoint operator from data without probing the adjoint? Current practical approaches suggest that one can accurately recover an operator while only using data generated by the forward action of the operator without access to the adjoint. However, naively, it seems essential to sample the action of the adjoint. In this paper, we partially explain this mystery by proving that without querying the adjoint, one can approximate a family of non-self-adjoint infinite-dimensional compact operators via projection onto a Fourier basis. We then apply the result to recovering Green's functions of elliptic partial differential operators and derive an adjoint-free sample complexity bound. While existing theory justifies low sample complexity in operator learning, ours is the first adjoint-free analysis that attempts to close the gap between theory and practice.
    
[^8]: GDL-DS: 分布转换下几何深度学习的基准测试

    GDL-DS: A Benchmark for Geometric Deep Learning under Distribution Shifts. (arXiv:2310.08677v1 [cs.LG])

    [http://arxiv.org/abs/2310.08677](http://arxiv.org/abs/2310.08677)

    GDL-DS是一个基准测试，用于评估几何深度学习模型在具有分布转换的场景中的性能。它包括多个科学领域的评估数据集，并研究了不同级别的超出分布特征的信息访问。

    

    几何深度学习(GDL)在各个科学领域引起了广泛关注，主要是因为其擅长对具有复杂几何结构的数据进行建模。然而，很少有研究探索其在处理分布转换问题上的能力，这是许多相关应用中常见的挑战。为了弥补这一空白，我们提出了GDL-DS，这是一个全面的基准测试，旨在评估GDL模型在具有分布转换的场景中的性能。我们的评估数据集涵盖了从粒子物理学和材料科学到生物化学的不同科学领域，并包括各种分布转换，包括条件、协变和概念转换。此外，我们研究了来自超出分布的测试数据的信息访问的三个级别，包括没有超出分布的信息、只有带标签的超出分布特征和带有少数标签的超出分布特征。总体而言，我们的基准测试涉及30个不同的实验设置，并评估3种信息访问水平。

    Geometric deep learning (GDL) has gained significant attention in various scientific fields, chiefly for its proficiency in modeling data with intricate geometric structures. Yet, very few works have delved into its capability of tackling the distribution shift problem, a prevalent challenge in many relevant applications. To bridge this gap, we propose GDL-DS, a comprehensive benchmark designed for evaluating the performance of GDL models in scenarios with distribution shifts. Our evaluation datasets cover diverse scientific domains from particle physics and materials science to biochemistry, and encapsulate a broad spectrum of distribution shifts including conditional, covariate, and concept shifts. Furthermore, we study three levels of information access from the out-of-distribution (OOD) testing data, including no OOD information, only OOD features without labels, and OOD features with a few labels. Overall, our benchmark results in 30 different experiment settings, and evaluates 3 
    
[^9]: 委托分散机器学习中的数据收集

    Delegating Data Collection in Decentralized Machine Learning. (arXiv:2309.01837v1 [cs.LG])

    [http://arxiv.org/abs/2309.01837](http://arxiv.org/abs/2309.01837)

    这项研究在分散机器学习生态系统中研究了委托的数据收集问题，通过设计最优契约解决了模型质量评估的不确定性和对最优性能缺乏预先知识的挑战。

    

    受分散机器学习生态系统的出现的启发，我们研究了数据收集的委托问题。以契约理论为出发点，我们设计了解决两个基本机器学习挑战的最优和近似最优契约：模型质量评估的不确定性和对任何模型最优性能的缺乏知识。我们证明，通过简单的线性契约可以解决不确定性问题，即使委托人只有一个小的测试集，也能实现1-1/e的一等效用水平。此外，我们给出了委托人测试集大小的充分条件，可以达到对最优效用的逼近。为了解决对最优性能缺乏预先知识的问题，我们提出了一个凸问题，可以自适应和高效地计算最优契约。

    Motivated by the emergence of decentralized machine learning ecosystems, we study the delegation of data collection. Taking the field of contract theory as our starting point, we design optimal and near-optimal contracts that deal with two fundamental machine learning challenges: lack of certainty in the assessment of model quality and lack of knowledge regarding the optimal performance of any model. We show that lack of certainty can be dealt with via simple linear contracts that achieve 1-1/e fraction of the first-best utility, even if the principal has a small test set. Furthermore, we give sufficient conditions on the size of the principal's test set that achieves a vanishing additive approximation to the optimal utility. To address the lack of a priori knowledge regarding the optimal performance, we give a convex program that can adaptively and efficiently compute the optimal contract.
    
[^10]: 具有确定性演化状态的强盗模型

    Bandits with Deterministically Evolving States. (arXiv:2307.11655v1 [cs.LG])

    [http://arxiv.org/abs/2307.11655](http://arxiv.org/abs/2307.11655)

    该论文提出了一种名为具有确定性演化状态的强盗模型，用于学习带有强盗反馈的推荐系统和在线广告。该模型考虑了状态演化的不同速率，能准确评估奖励与系统健康程度之间的关系。

    

    我们提出了一种学习与强盗反馈结合的模型，同时考虑到确定性演化和不可观测的状态，我们称之为具有确定性演化状态的强盗模型。我们的模型主要应用于推荐系统和在线广告的学习。在这两种情况下，算法在每一轮获得的奖励是选择行动的短期奖励和系统的“健康”程度（即通过其状态测量）的函数。例如，在推荐系统中，平台从用户对特定类型内容的参与中获得的奖励不仅取决于具体内容的固有特征，还取决于用户与平台上其他类型内容互动后其偏好的演化。我们的通用模型考虑了状态演化的不同速率λ∈[0,1]（例如，用户的偏好因先前内容消费而快速变化）。

    We propose a model for learning with bandit feedback while accounting for deterministically evolving and unobservable states that we call Bandits with Deterministically Evolving States. The workhorse applications of our model are learning for recommendation systems and learning for online ads. In both cases, the reward that the algorithm obtains at each round is a function of the short-term reward of the action chosen and how ``healthy'' the system is (i.e., as measured by its state). For example, in recommendation systems, the reward that the platform obtains from a user's engagement with a particular type of content depends not only on the inherent features of the specific content, but also on how the user's preferences have evolved as a result of interacting with other types of content on the platform. Our general model accounts for the different rate $\lambda \in [0,1]$ at which the state evolves (e.g., how fast a user's preferences shift as a result of previous content consumption
    
[^11]: 智能压力电子垫用于人类睡眠姿势和动态活动识别

    Smart Pressure e-Mat for Human Sleeping Posture and Dynamic Activity Recognition. (arXiv:2305.11367v1 [cs.CV])

    [http://arxiv.org/abs/2305.11367](http://arxiv.org/abs/2305.11367)

    本文介绍了一种基于Velostat的智能压力电子垫系统，可用于识别人体姿势和运动，具有高精度。

    

    在强调医疗保健、早期教育和健身方面，越来越多的非侵入式测量和识别方法受到关注。压力感应由于其简单的结构、易于访问、可视化应用和无害性而得到广泛研究。本文介绍了一种基于压敏材料Velostat的智能压力电子垫(SP e-Mat)系统，用于人体监测应用，包括睡眠姿势、运动和瑜伽识别。在子系统扫描电子垫读数并处理信号后，它生成一个压力图像流。采用深度神经网络(DNNs)来拟合和训练压力图像流，并识别相应的人类行为。四种睡眠姿势和受Nintendo Switch Ring Fit Adventure(RFA)启发的五种动态活动被用作拟议的SPeM系统的初步验证。SPeM系统在两种应用中均达到了较高的准确性，这证明了其高精度和。

    With the emphasis on healthcare, early childhood education, and fitness, non-invasive measurement and recognition methods have received more attention. Pressure sensing has been extensively studied due to its advantages of simple structure, easy access, visualization application, and harmlessness. This paper introduces a smart pressure e-mat (SPeM) system based on a piezoresistive material Velostat for human monitoring applications, including sleeping postures, sports, and yoga recognition. After a subsystem scans e-mat readings and processes the signal, it generates a pressure image stream. Deep neural networks (DNNs) are used to fit and train the pressure image stream and recognize the corresponding human behavior. Four sleeping postures and five dynamic activities inspired by Nintendo Switch Ring Fit Adventure (RFA) are used as a preliminary validation of the proposed SPeM system. The SPeM system achieves high accuracies on both applications, which demonstrates the high accuracy and
    
[^12]: 深度量子神经网络对应高斯过程

    Deep quantum neural networks form Gaussian processes. (arXiv:2305.09957v1 [quant-ph])

    [http://arxiv.org/abs/2305.09957](http://arxiv.org/abs/2305.09957)

    本文证明了基于Haar随机酉或正交深量子神经网络的某些模型的输出会收敛于高斯过程。然而，这种高斯过程不能用于通过贝叶斯统计学来有效预测QNN的输出。

    

    众所周知，从独立同分布的先验条件开始初始化的人工神经网络在隐藏层神经元数目足够大的极限下收敛到高斯过程。本文证明了量子神经网络（QNNs）也存在类似的结果。特别地，我们证明了基于Haar随机酉或正交深QNNs的某些模型的输出在希尔伯特空间维度$d$足够大时会收敛于高斯过程。由于输入状态、测量的可观测量以及酉矩阵的元素不独立等因素的作用，本文对这一结果的推导比经典情形更加微妙。我们分析的一个重要后果是，这个结果得到的高斯过程不能通过贝叶斯统计学来有效地预测QNN的输出。此外，我们的定理表明，Haar随机QNNs中的测量现象比以前认为的要更严重，我们证明了演员的集中现象。

    It is well known that artificial neural networks initialized from independent and identically distributed priors converge to Gaussian processes in the limit of large number of neurons per hidden layer. In this work we prove an analogous result for Quantum Neural Networks (QNNs). Namely, we show that the outputs of certain models based on Haar random unitary or orthogonal deep QNNs converge to Gaussian processes in the limit of large Hilbert space dimension $d$. The derivation of this result is more nuanced than in the classical case due the role played by the input states, the measurement observable, and the fact that the entries of unitary matrices are not independent. An important consequence of our analysis is that the ensuing Gaussian processes cannot be used to efficiently predict the outputs of the QNN via Bayesian statistics. Furthermore, our theorems imply that the concentration of measure phenomenon in Haar random QNNs is much worse than previously thought, as we prove that ex
    
[^13]: 随机逼近法用于组分布式鲁棒优化

    Stochastic Approximation Approaches to Group Distributionally Robust Optimization. (arXiv:2302.09267v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.09267](http://arxiv.org/abs/2302.09267)

    本文提出了一种随机逼近法，用于组分布式鲁棒优化，该算法利用在线学习技术，将每轮所需的样本数从$m$个降至$1$个，同时保持相同的样本复杂度。

    

    本文研究组分布式鲁棒优化（GDRO），目的是学习一个能在$m$个不同分布上表现良好的模型。首先，我们将GDRO建模为随机凸凹鞍点问题，并证明使用$m$个样本的随机镜像下降法(SMD)，能够实现$O(m(\log m)/\epsilon ^2)$个样本的复杂度，以找到一个$\epsilon$-最优解，这与$\Omega(m/\epsilon ^2)$的下界想匹配，除了一个对数因子。接下来，我们利用在线学习技术，将每轮所需的样本数从$m$个降至$1$个，同时保持相同的样本复杂度。具体而言，我们将GDRO构造为一个双人博弈，其中一个玩家简单地执行SMD，另一个执行一种用于非明显多臂老虎机的在线算法。接下来，我们考虑一个更实际的情况，即可以从每个分布中绘制的样本数量不同，并提出一种新的公式。

    This paper investigates group distributionally robust optimization (GDRO), with the purpose to learn a model that performs well over $m$ different distributions. First, we formulate GDRO as a stochastic convex-concave saddle-point problem, and demonstrate that stochastic mirror descent (SMD), using $m$ samples in each iteration, achieves an $O(m (\log m)/\epsilon^2)$ sample complexity for finding an $\epsilon$-optimal solution, which matches the $\Omega(m/\epsilon^2)$ lower bound up to a logarithmic factor. Then, we make use of techniques from online learning to reduce the number of samples required in each round from $m$ to $1$, keeping the same sample complexity. Specifically, we cast GDRO as a two-players game where one player simply performs SMD and the other executes an online algorithm for non-oblivious multi-armed bandits. Next, we consider a more practical scenario where the number of samples that can be drawn from each distribution is different, and propose a novel formulation
    
[^14]: 对未知数据的泛化、逻辑推理和学位课程的概述

    Generalization on the Unseen, Logic Reasoning and Degree Curriculum. (arXiv:2301.13105v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.13105](http://arxiv.org/abs/2301.13105)

    本文研究了在逻辑推理任务中对未知数据的泛化能力，提供了网络架构在该设置下的表现证据，发现了一类网络模型在未知数据上学习了最小度插值器，并对长度普通化现象提供了解释。

    

    本文考虑了逻辑（布尔）函数的学习，重点在于对未知数据的泛化（GOTU）设定，这是一种强大的分布外泛化的案例。这是由于某些推理任务（例如算术/逻辑）中数据的丰富组合性质使得代表性数据采样具有挑战性，并且在GOTU下成功学习为第一个“推理”学习者展示了一个小插图。然后，我们研究了通过(S)GD训练的不同网络架构在GOTU下的表现，并提供了理论和实验证据，证明了一个类别的网络模型（包括Transformer的实例、随机特征模型和对角线线性网络）在未知数据上学习了最小度插值器。我们还提供了证据表明，其他具有更大学习速率或均场网络的实例达到了渗漏最小度解。这些发现带来了两个影响：（1）我们提供了对长度普通化的解释

    This paper considers the learning of logical (Boolean) functions with focus on the generalization on the unseen (GOTU) setting, a strong case of out-of-distribution generalization. This is motivated by the fact that the rich combinatorial nature of data in certain reasoning tasks (e.g., arithmetic/logic) makes representative data sampling challenging, and learning successfully under GOTU gives a first vignette of an 'extrapolating' or 'reasoning' learner. We then study how different network architectures trained by (S)GD perform under GOTU and provide both theoretical and experimental evidence that for a class of network models including instances of Transformers, random features models, and diagonal linear networks, a min-degree-interpolator is learned on the unseen. We also provide evidence that other instances with larger learning rates or mean-field networks reach leaky min-degree solutions. These findings lead to two implications: (1) we provide an explanation to the length genera
    
[^15]: 重新审视离散型Soft Actor-Critic方法

    Revisiting Discrete Soft Actor-Critic. (arXiv:2209.10081v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2209.10081](http://arxiv.org/abs/2209.10081)

    本研究重新审视了将连续动作空间的Soft Actor-Critic方法调整为离散动作空间的问题，并提出了解决Q值低估和性能不稳定的方法，验证了其在Atari游戏和大规模MOBA游戏中的有效性。

    

    本文研究将连续动作空间的Soft Actor-Critic方法（SAC）调整为离散动作空间。我们重新审视了经典的SAC方法，并深入理解了在离散设置下其Q值低估和性能不稳定的问题。因此，我们提出了熵惩罚和具有Q-clip的双平均Q-learning方法来解决这些问题。通过对包括Atari游戏和一个大规模MOBA游戏在内的典型基准问题进行广泛实验，验证了我们方法的有效性。我们的代码可在以下链接找到: https://github.com/coldsummerday/Revisiting-Discrete-SAC.

    We study the adaption of soft actor-critic (SAC) from continuous action space to discrete action space. We revisit vanilla SAC and provide an in-depth understanding of its Q value underestimation and performance instability issues when applied to discrete settings. We thereby propose entropy-penalty and double average Q-learning with Q-clip to address these issues. Extensive experiments on typical benchmarks with discrete action space, including Atari games and a large-scale MOBA game, show the efficacy of our proposed method. Our code is at:https://github.com/coldsummerday/Revisiting-Discrete-SAC.
    

