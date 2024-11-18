# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [BAdam: A Memory Efficient Full Parameter Training Method for Large Language Models](https://arxiv.org/abs/2404.02827) | BAdam提出了一种内存高效的全参数微调大型语言模型的方法，并在实验中展现出优越的收敛行为以及在性能评估中的优势。 |
| [^2] | [Swarm Characteristics Classification Using Neural Networks](https://arxiv.org/abs/2403.19572) | 本文研究了使用监督神经网络时间序列分类（NN TSC）预测军事背景下群体自主体的关键属性和战术，以及展示了NN TSC在快速推断攻击群体情报方面的有效性。 |
| [^3] | [A Survey on Deep Learning and State-of-the-arts Applications](https://arxiv.org/abs/2403.17561) | 深度学习是解决复杂问题的强大工具，本研究旨在全面审视深度学习模型及其应用的最新发展 |
| [^4] | [ThermoHands: A Benchmark for 3D Hand Pose Estimation from Egocentric Thermal Image](https://arxiv.org/abs/2403.09871) | ThermoHands提出了一个新的基准ThermoHands，旨在解决热图中主观视角3D手部姿势估计的挑战，介绍了一个具有双transformer模块的定制基线方法TheFormer，表明热成像在恶劣条件下实现稳健的3D手部姿势估计的有效性。 |
| [^5] | [DEEP-IoT: Downlink-Enhanced Efficient-Power Internet of Things](https://arxiv.org/abs/2403.00321) | DEEP-IoT通过“更多监听，更少传输”的策略，挑战和转变了传统的物联网通信模型，大幅降低能耗并提高设备寿命。 |
| [^6] | [Covariance-Adaptive Least-Squares Algorithm for Stochastic Combinatorial Semi-Bandits](https://arxiv.org/abs/2402.15171) | 提出了一种协方差自适应的最小二乘算法，利用在线估计协方差结构，相对于基于代理方差的算法获得改进的遗憾上界，特别在协方差系数全为非负时，能有效地利用半臂反馈，并在各种参数设置下表现优异。 |
| [^7] | [CLCE: An Approach to Refining Cross-Entropy and Contrastive Learning for Optimized Learning Fusion](https://arxiv.org/abs/2402.14551) | CLCE方法结合了标签感知对比学习与交叉熵损失，通过协同利用难例挖掘提高了性能表现 |
| [^8] | [ConSmax: Hardware-Friendly Alternative Softmax with Learnable Parameters](https://arxiv.org/abs/2402.10930) | ConSmax是一种硬件友好型Softmax替代方案，通过引入可学习参数，在不影响性能的情况下实现了对原Softmax关键任务的高效处理。 |
| [^9] | [Multi-View Symbolic Regression](https://arxiv.org/abs/2402.04298) | 多视角符号回归(MvSR)是一种同时考虑多个数据集的符号回归方法，能够找到一个参数化解来准确拟合所有数据集，解决了传统方法无法处理不同实验设置的问题。 |

# 详细

[^1]: BAdam：面向大型语言模型的内存高效全参数训练方法

    BAdam: A Memory Efficient Full Parameter Training Method for Large Language Models

    [https://arxiv.org/abs/2404.02827](https://arxiv.org/abs/2404.02827)

    BAdam提出了一种内存高效的全参数微调大型语言模型的方法，并在实验中展现出优越的收敛行为以及在性能评估中的优势。

    

    这项工作提出了BAdam，这是一种利用Adam作为内部求解器的块坐标优化框架的优化器。BAdam提供了一种内存高效的方法，用于对大型语言模型进行全参数微调，并且由于链式规则属性减少了反向过程的运行时间。在实验中，我们将BAdam应用于在Alpaca-GPT4数据集上使用单个RTX3090-24GB GPU进行指导微调的Llama 2-7B模型。结果表明，与LoRA和LOMO相比，BAdam展现出了优越的收敛行为。此外，我们通过使用MT-bench对指导微调模型进行下游性能评估，结果显示BAdam在适度超越LoRA的基础上更显著地优于LOMO。最后，我们将BAdam与Adam在中等任务上进行了比较，即在SuperGLUE基准上对RoBERTa-large进行微调。结果表明，BAdam能够缩小与Adam之间的性能差距。我们的代码

    arXiv:2404.02827v1 Announce Type: new  Abstract: This work presents BAdam, an optimizer that leverages the block coordinate optimization framework with Adam as the inner solver. BAdam offers a memory efficient approach to the full parameter finetuning of large language models and reduces running time of the backward process thanks to the chain rule property. Experimentally, we apply BAdam to instruction-tune the Llama 2-7B model on the Alpaca-GPT4 dataset using a single RTX3090-24GB GPU. The results indicate that BAdam exhibits superior convergence behavior in comparison to LoRA and LOMO. Furthermore, our downstream performance evaluation of the instruction-tuned models using the MT-bench shows that BAdam modestly surpasses LoRA and more substantially outperforms LOMO. Finally, we compare BAdam with Adam on a medium-sized task, i.e., finetuning RoBERTa-large on the SuperGLUE benchmark. The results demonstrate that BAdam is capable of narrowing the performance gap with Adam. Our code is
    
[^2]: 使用神经网络对群体特性进行分类

    Swarm Characteristics Classification Using Neural Networks

    [https://arxiv.org/abs/2403.19572](https://arxiv.org/abs/2403.19572)

    本文研究了使用监督神经网络时间序列分类（NN TSC）预测军事背景下群体自主体的关键属性和战术，以及展示了NN TSC在快速推断攻击群体情报方面的有效性。

    

    理解群体自主体的特性对于国防和安全应用至关重要。本文介绍了使用监督神经网络时间序列分类（NN TSC）来预测军事环境中群体自主体的关键属性和战术的研究。具体地，NN TSC被应用于推断两个二进制属性 - 通信和比例导航 - 这两者结合定义了四种互斥的群体战术。我们发现文献中对于使用神经网络进行群体分类存在一定的空白，并展示了NN TSC在快速推断有关攻击群体情报以指导反制动作方面的有效性。通过模拟的群体对战，我们评估了NN TSC在观察窗口要求、噪声鲁棒性和对群体规模的可扩展性方面的性能。关键发现显示NN能够使用较短的观察窗口以97%的准确率预测群体行为。

    arXiv:2403.19572v1 Announce Type: new  Abstract: Understanding the characteristics of swarming autonomous agents is critical for defense and security applications. This article presents a study on using supervised neural network time series classification (NN TSC) to predict key attributes and tactics of swarming autonomous agents for military contexts. Specifically, NN TSC is applied to infer two binary attributes - communication and proportional navigation - which combine to define four mutually exclusive swarm tactics. We identify a gap in literature on using NNs for swarm classification and demonstrate the effectiveness of NN TSC in rapidly deducing intelligence about attacking swarms to inform counter-maneuvers. Through simulated swarm-vs-swarm engagements, we evaluate NN TSC performance in terms of observation window requirements, noise robustness, and scalability to swarm size. Key findings show NNs can predict swarm behaviors with 97% accuracy using short observation windows of
    
[^3]: 深度学习及其最新应用综述

    A Survey on Deep Learning and State-of-the-arts Applications

    [https://arxiv.org/abs/2403.17561](https://arxiv.org/abs/2403.17561)

    深度学习是解决复杂问题的强大工具，本研究旨在全面审视深度学习模型及其应用的最新发展

    

    深度学习, 是人工智能的一个分支，是一种利用多层互连单元（神经元）从原始输入数据中直接学习复杂模式和表示的计算模型。受到这种学习能力的赋能，深度学习已成为解决复杂问题的强大工具，是许多突破性技术和创新的核心驱动力。构建深度学习模型是一项具有挑战性的任务，因为算法的复杂性和现实问题的动态性。有几项研究回顾了深度学习的概念和应用。然而，这些研究大多集中于深度学习模型类型和卷积神经网络架构，对深度学习模型及其在不同领域解决复杂问题的最新发展的覆盖面有限。因此，受到这些限制的启发，本研究旨在全面审视th

    arXiv:2403.17561v1 Announce Type: new  Abstract: Deep learning, a branch of artificial intelligence, is a computational model that uses multiple layers of interconnected units (neurons) to learn intricate patterns and representations directly from raw input data. Empowered by this learning capability, it has become a powerful tool for solving complex problems and is the core driver of many groundbreaking technologies and innovations. Building a deep learning model is a challenging task due to the algorithm`s complexity and the dynamic nature of real-world problems. Several studies have reviewed deep learning concepts and applications. However, the studies mostly focused on the types of deep learning models and convolutional neural network architectures, offering limited coverage of the state-of-the-art of deep learning models and their applications in solving complex problems across different domains. Therefore, motivated by the limitations, this study aims to comprehensively review th
    
[^4]: ThermoHands：一种用于从主观视角热图中估计3D手部姿势的基准

    ThermoHands: A Benchmark for 3D Hand Pose Estimation from Egocentric Thermal Image

    [https://arxiv.org/abs/2403.09871](https://arxiv.org/abs/2403.09871)

    ThermoHands提出了一个新的基准ThermoHands，旨在解决热图中主观视角3D手部姿势估计的挑战，介绍了一个具有双transformer模块的定制基线方法TheFormer，表明热成像在恶劣条件下实现稳健的3D手部姿势估计的有效性。

    

    在这项工作中，我们提出了ThermoHands，这是一个针对基于热图的主观视角3D手部姿势估计的新基准，旨在克服诸如光照变化和遮挡（例如手部穿戴物）等挑战。该基准包括来自28名主体进行手-物体和手-虚拟交互的多样数据集，经过自动化过程准确标注了3D手部姿势。我们引入了一个定制的基线方法TheFormer，利用双transformer模块在热图中实现有效的主观视角3D手部姿势估计。我们的实验结果突显了TheFormer的领先性能，并确认了热成像在实现恶劣条件下稳健的3D手部姿势估计方面的有效性。

    arXiv:2403.09871v1 Announce Type: cross  Abstract: In this work, we present ThermoHands, a new benchmark for thermal image-based egocentric 3D hand pose estimation, aimed at overcoming challenges like varying lighting and obstructions (e.g., handwear). The benchmark includes a diverse dataset from 28 subjects performing hand-object and hand-virtual interactions, accurately annotated with 3D hand poses through an automated process. We introduce a bespoken baseline method, TheFormer, utilizing dual transformer modules for effective egocentric 3D hand pose estimation in thermal imagery. Our experimental results highlight TheFormer's leading performance and affirm thermal imaging's effectiveness in enabling robust 3D hand pose estimation in adverse conditions.
    
[^5]: DEEP-IoT: 下行增强型高效能物联网

    DEEP-IoT: Downlink-Enhanced Efficient-Power Internet of Things

    [https://arxiv.org/abs/2403.00321](https://arxiv.org/abs/2403.00321)

    DEEP-IoT通过“更多监听，更少传输”的策略，挑战和转变了传统的物联网通信模型，大幅降低能耗并提高设备寿命。

    

    本文介绍了DEEP-IoT，这是一种具有革命意义的通信范例，旨在重新定义物联网设备之间的通信方式。通过开创性的“更多监听，更少传输”的策略，DEEP-IoT挑战和转变了传统的发送方（物联网设备）为中心的通信模型，将接收方（接入点）作为关键角色，从而降低能耗并延长设备寿命。我们不仅概念化了DEEP-IoT，还通过在窄带系统中集成深度学习增强的反馈信道编码来实现它。模拟结果显示，IoT单元的运行寿命显著提高，比使用Turbo和Polar编码的传统系统提高了最多52.71%。这一进展标志着一种变革。

    arXiv:2403.00321v1 Announce Type: cross  Abstract: At the heart of the Internet of Things (IoT) -- a domain witnessing explosive growth -- the imperative for energy efficiency and the extension of device lifespans has never been more pressing. This paper presents DEEP-IoT, a revolutionary communication paradigm poised to redefine how IoT devices communicate. Through a pioneering "listen more, transmit less" strategy, DEEP-IoT challenges and transforms the traditional transmitter (IoT devices)-centric communication model to one where the receiver (the access point) play a pivotal role, thereby cutting down energy use and boosting device longevity. We not only conceptualize DEEP-IoT but also actualize it by integrating deep learning-enhanced feedback channel codes within a narrow-band system. Simulation results show a significant enhancement in the operational lifespan of IoT cells -- surpassing traditional systems using Turbo and Polar codes by up to 52.71%. This leap signifies a paradi
    
[^6]: 用于随机组合半臂老虎机的协方差自适应最小二乘算法

    Covariance-Adaptive Least-Squares Algorithm for Stochastic Combinatorial Semi-Bandits

    [https://arxiv.org/abs/2402.15171](https://arxiv.org/abs/2402.15171)

    提出了一种协方差自适应的最小二乘算法，利用在线估计协方差结构，相对于基于代理方差的算法获得改进的遗憾上界，特别在协方差系数全为非负时，能有效地利用半臂反馈，并在各种参数设置下表现优异。

    

    我们解决了随机组合半臂老虎机问题，其中玩家可以从包含d个基本项的P个子集中进行选择。大多数现有算法（如CUCB、ESCB、OLS-UCB）需要对奖励分布有先验知识，比如子高斯代理-方差的上界，这很难准确估计。在这项工作中，我们设计了OLS-UCB的方差自适应版本，依赖于协方差结构的在线估计。在实际设置中，估计协方差矩阵的系数要容易得多，并且相对于基于代理方差的算法，导致改进的遗憾上界。当协方差系数全为非负时，我们展示了我们的方法有效地利用了半臂反馈，并且可以明显优于老虎机反馈方法，在指数级别P≫d以及P≤d的情况下，这一点并不来自大多数现有分析。

    arXiv:2402.15171v1 Announce Type: new  Abstract: We address the problem of stochastic combinatorial semi-bandits, where a player can select from P subsets of a set containing d base items. Most existing algorithms (e.g. CUCB, ESCB, OLS-UCB) require prior knowledge on the reward distribution, like an upper bound on a sub-Gaussian proxy-variance, which is hard to estimate tightly. In this work, we design a variance-adaptive version of OLS-UCB, relying on an online estimation of the covariance structure. Estimating the coefficients of a covariance matrix is much more manageable in practical settings and results in improved regret upper bounds compared to proxy variance-based algorithms. When covariance coefficients are all non-negative, we show that our approach efficiently leverages the semi-bandit feedback and provably outperforms bandit feedback approaches, not only in exponential regimes where P $\gg$ d but also when P $\le$ d, which is not straightforward from most existing analyses.
    
[^7]: CLCE：一种优化学习融合的改进交叉熵和对比学习方法

    CLCE: An Approach to Refining Cross-Entropy and Contrastive Learning for Optimized Learning Fusion

    [https://arxiv.org/abs/2402.14551](https://arxiv.org/abs/2402.14551)

    CLCE方法结合了标签感知对比学习与交叉熵损失，通过协同利用难例挖掘提高了性能表现

    

    最先进的预训练图像模型主要采用两阶段方法：在大规模数据集上进行初始无监督预训练，然后使用交叉熵损失（CE）进行特定任务的微调。然而，已经证明CE可能会损害模型的泛化性和稳定性。为了解决这些问题，我们引入了一种名为CLCE的新方法，该方法将标签感知对比学习与CE相结合。我们的方法不仅保持了两种损失函数的优势，而且以协同方式利用难例挖掘来增强性能。

    arXiv:2402.14551v1 Announce Type: cross  Abstract: State-of-the-art pre-trained image models predominantly adopt a two-stage approach: initial unsupervised pre-training on large-scale datasets followed by task-specific fine-tuning using Cross-Entropy loss~(CE). However, it has been demonstrated that CE can compromise model generalization and stability. While recent works employing contrastive learning address some of these limitations by enhancing the quality of embeddings and producing better decision boundaries, they often overlook the importance of hard negative mining and rely on resource intensive and slow training using large sample batches. To counter these issues, we introduce a novel approach named CLCE, which integrates Label-Aware Contrastive Learning with CE. Our approach not only maintains the strengths of both loss functions but also leverages hard negative mining in a synergistic way to enhance performance. Experimental results demonstrate that CLCE significantly outperf
    
[^8]: ConSmax: 具有可学习参数的硬件友好型Softmax替代方案

    ConSmax: Hardware-Friendly Alternative Softmax with Learnable Parameters

    [https://arxiv.org/abs/2402.10930](https://arxiv.org/abs/2402.10930)

    ConSmax是一种硬件友好型Softmax替代方案，通过引入可学习参数，在不影响性能的情况下实现了对原Softmax关键任务的高效处理。

    

    自注意机制将基于transformer的大型语言模型（LLM）与卷积和循环神经网络区分开来。尽管性能有所提升，但由于自注意中广泛使用Softmax，在硅上实现实时LLM推断仍具挑战性。为了解决这一挑战，我们提出了Constant Softmax（ConSmax），这是一种高效的Softmax替代方案，采用可微的规范化参数来消除Softmax中的最大搜索和分母求和，实现了大规模并行化。

    arXiv:2402.10930v1 Announce Type: cross  Abstract: The self-attention mechanism sets transformer-based large language model (LLM) apart from the convolutional and recurrent neural networks. Despite the performance improvement, achieving real-time LLM inference on silicon is challenging due to the extensively used Softmax in self-attention. Apart from the non-linearity, the low arithmetic intensity greatly reduces the processing parallelism, which becomes the bottleneck especially when dealing with a longer context. To address this challenge, we propose Constant Softmax (ConSmax), a software-hardware co-design as an efficient Softmax alternative. ConSmax employs differentiable normalization parameters to remove the maximum searching and denominator summation in Softmax. It allows for massive parallelization while performing the critical tasks of Softmax. In addition, a scalable ConSmax hardware utilizing a bitwidth-split look-up table (LUT) can produce lossless non-linear operation and 
    
[^9]: 多视角符号回归

    Multi-View Symbolic Regression

    [https://arxiv.org/abs/2402.04298](https://arxiv.org/abs/2402.04298)

    多视角符号回归(MvSR)是一种同时考虑多个数据集的符号回归方法，能够找到一个参数化解来准确拟合所有数据集，解决了传统方法无法处理不同实验设置的问题。

    

    符号回归(SR)搜索表示解释变量和响应变量之间关系的分析表达式。目前的SR方法假设从单个实验中提取的单个数据集。然而，研究人员经常面临来自不同设置的多个实验结果集。传统的SR方法可能无法找到潜在的表达式，因为每个实验的参数可能不同。在这项工作中，我们提出了多视角符号回归(MvSR)，它同时考虑多个数据集，模拟实验环境，并输出一个通用的参数化解。这种方法将评估的表达式适应每个独立数据集，并同时返回能够准确拟合所有数据集的参数函数族f(x; \theta)。我们使用从已知表达式生成的数据以及来自实际世界的数据来展示MvSR的有效性。

    Symbolic regression (SR) searches for analytical expressions representing the relationship between a set of explanatory and response variables. Current SR methods assume a single dataset extracted from a single experiment. Nevertheless, frequently, the researcher is confronted with multiple sets of results obtained from experiments conducted with different setups. Traditional SR methods may fail to find the underlying expression since the parameters of each experiment can be different. In this work we present Multi-View Symbolic Regression (MvSR), which takes into account multiple datasets simultaneously, mimicking experimental environments, and outputs a general parametric solution. This approach fits the evaluated expression to each independent dataset and returns a parametric family of functions f(x; \theta) simultaneously capable of accurately fitting all datasets. We demonstrate the effectiveness of MvSR using data generated from known expressions, as well as real-world data from 
    

