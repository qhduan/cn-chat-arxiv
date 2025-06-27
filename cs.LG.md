# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FOCIL: Finetune-and-Freeze for Online Class Incremental Learning by Training Randomly Pruned Sparse Experts](https://arxiv.org/abs/2403.14684) | FOCIL通过训练随机修剪稀疏子网络实现在线持续类递增学习，在避免存储重放数据的同时有效防止遗忘。 |
| [^2] | [NuGraph2: A Graph Neural Network for Neutrino Physics Event Reconstruction](https://arxiv.org/abs/2403.11872) | NuGraph2 是一种用于液氩时间投影室探测器中模拟中微子相互作用低级重建的图神经网络，通过多头注意力传递机制实现了高效的背景过滤和语义标记。 |
| [^3] | [Robustness, Efficiency, or Privacy: Pick Two in Machine Learning](https://arxiv.org/abs/2312.14712) | 该论文研究了在分布式机器学习架构中实现隐私和鲁棒性的成本，指出整合这两个目标会牺牲计算效率。 |
| [^4] | [A Novel Federated Learning-Based IDS for Enhancing UAVs Privacy and Security](https://arxiv.org/abs/2312.04135) | 本论文引入了基于联邦学习的入侵检测系统(FL-IDS)，旨在解决FANETs中集中式系统所遇到的挑战，降低了计算和存储成本，适合资源受限的无人机。 |
| [^5] | [PuriDefense: Randomized Local Implicit Adversarial Purification for Defending Black-box Query-based Attacks.](http://arxiv.org/abs/2401.10586) | PuriDefense是一种高效的防御机制，通过使用轻量级净化模型进行随机路径净化，减缓基于查询的攻击的收敛速度，并有效防御黑盒基于查询的攻击。 |
| [^6] | [Continual Learning as Computationally Constrained Reinforcement Learning.](http://arxiv.org/abs/2307.04345) | 本文研究了连续学习作为计算受限的强化学习的主题，提出了一个框架和一套工具来解决人工智能领域长期以来的挑战并促进进一步的研究。 |
| [^7] | [Variational quantum regression algorithm with encoded data structure.](http://arxiv.org/abs/2307.03334) | 本文介绍了一个具有编码数据结构的变分量子回归算法，在量子机器学习中具有模型解释性，并能有效地处理互连度较高的量子比特。算法通过压缩编码和数字-模拟门操作，大大提高了在噪声中尺度量子计算机上的运行时间复杂度。 |
| [^8] | [High-dimensional Contextual Bandit Problem without Sparsity.](http://arxiv.org/abs/2306.11017) | 本论文研究了高维情境赌博问题，无需施加稀疏性要求，并提出了一种探索-开发算法以解决此问题。研究表明，可以通过平衡探索和开发实现最优速率。同时，还介绍了一种自适应探索-开发算法来找到最优平衡点。 |
| [^9] | [StyleNAT: Giving Each Head a New Perspective.](http://arxiv.org/abs/2211.05770) | StyleNAT是一个新的基于transformer的图像生成框架，通过使用邻域注意力（NA）来捕捉局部和全局信息，能够高效灵活地适应不同的数据集，并在FFHQ-256上取得了新的最佳结果。 |

# 详细

[^1]: FOCIL: 通过训练随机修剪稀疏专家进行在线类递增学习的微调和冻结

    FOCIL: Finetune-and-Freeze for Online Class Incremental Learning by Training Randomly Pruned Sparse Experts

    [https://arxiv.org/abs/2403.14684](https://arxiv.org/abs/2403.14684)

    FOCIL通过训练随机修剪稀疏子网络实现在线持续类递增学习，在避免存储重放数据的同时有效防止遗忘。

    

    在线持续学习中的类递增学习（CIL）旨在从数据流中获取一系列新类的知识，仅使用每个数据点进行一次训练。与离线模式相比，这更加现实，离线模式假定所有新类的数据已经准备好。当前的在线CIL方法存储先前数据的子集，这会在内存和计算方面造成沉重的开销，还存在隐私问题。本文提出了一种名为FOCIL的新型在线CIL方法。它通过训练随机修剪稀疏子网络不断微调主体系结构，然后冻结训练连接以防止遗忘。FOCIL还自适应确定每个任务的稀疏度级别和学习速率，并确保（几乎）零遗忘跨所有任务，且不存储任何重放数据。

    arXiv:2403.14684v1 Announce Type: cross  Abstract: Class incremental learning (CIL) in an online continual learning setting strives to acquire knowledge on a series of novel classes from a data stream, using each data point only once for training. This is more realistic compared to offline modes, where it is assumed that all data from novel class(es) is readily available. Current online CIL approaches store a subset of the previous data which creates heavy overhead costs in terms of both memory and computation, as well as privacy issues. In this paper, we propose a new online CIL approach called FOCIL. It fine-tunes the main architecture continually by training a randomly pruned sparse subnetwork for each task. Then, it freezes the trained connections to prevent forgetting. FOCIL also determines the sparsity level and learning rate per task adaptively and ensures (almost) zero forgetting across all tasks without storing any replay data. Experimental results on 10-Task CIFAR100, 20-Task
    
[^2]: NuGraph2：用于中微子物理事件重建的图神经网络

    NuGraph2: A Graph Neural Network for Neutrino Physics Event Reconstruction

    [https://arxiv.org/abs/2403.11872](https://arxiv.org/abs/2403.11872)

    NuGraph2 是一种用于液氩时间投影室探测器中模拟中微子相互作用低级重建的图神经网络，通过多头注意力传递机制实现了高效的背景过滤和语义标记。

    

    arXiv:2403.11872v1 公告类型：跨领域 摘要：液氩时间投影室（LArTPC）探测器技术提供了丰富的高分辨率粒子相互作用信息，充分利用这些信息需要先进的自动重建技术。本文描述了NuGraph2，一种用于LArTPC探测器中模拟中微子相互作用低级重建的图神经网络（GNN）。MicroBooNE探测器几何形状中的模拟中微子相互作用被描述为异质图，每个探测器平面上的能量沉积形成平面子图上的节点。该网络利用多头注意力传递机制对这些图节点执行背景过滤和语义标记，以98.0\%的效率识别与主要物理相互作用相关联的节点，并以94.9\%的效率根据粒子类型将其标记。该网络直接在探测器可观察量上运行。

    arXiv:2403.11872v1 Announce Type: cross  Abstract: Liquid Argon Time Projection Chamber (LArTPC) detector technology offers a wealth of high-resolution information on particle interactions, and leveraging that information to its full potential requires sophisticated automated reconstruction techniques. This article describes NuGraph2, a Graph Neural Network (GNN) for low-level reconstruction of simulated neutrino interactions in a LArTPC detector. Simulated neutrino interactions in the MicroBooNE detector geometry are described as heterogeneous graphs, with energy depositions on each detector plane forming nodes on planar subgraphs. The network utilizes a multi-head attention message-passing mechanism to perform background filtering and semantic labelling on these graph nodes, identifying those associated with the primary physics interaction with 98.0\% efficiency and labelling them according to particle type with 94.9\% efficiency. The network operates directly on detector observables
    
[^3]: 机器学习中的鲁棒性、效率或隐私：只能选两样

    Robustness, Efficiency, or Privacy: Pick Two in Machine Learning

    [https://arxiv.org/abs/2312.14712](https://arxiv.org/abs/2312.14712)

    该论文研究了在分布式机器学习架构中实现隐私和鲁棒性的成本，指出整合这两个目标会牺牲计算效率。

    

    机器学习（ML）应用的成功依赖于庞大的数据集和分布式架构，随着它们的增长，这些架构带来了重大挑战。在真实世界的场景中，数据通常包含敏感信息，数据污染和硬件故障等问题很常见。确保隐私和鲁棒性对于ML在公共生活中的广泛应用至关重要。本文从理论和实证角度研究了在分布式ML架构中实现这些目标所带来的成本。我们概述了分布式ML中隐私和鲁棒性的含义，并阐明了如何单独高效实现它们。然而，我们认为整合这两个目标会在计算效率上有显著的折衷。简而言之，传统的噪声注入通过隐藏毒害输入来损害准确性，而加密方法与防毒防御相冲突，因为它们是非线性的。

    arXiv:2312.14712v2 Announce Type: replace  Abstract: The success of machine learning (ML) applications relies on vast datasets and distributed architectures which, as they grow, present major challenges. In real-world scenarios, where data often contains sensitive information, issues like data poisoning and hardware failures are common. Ensuring privacy and robustness is vital for the broad adoption of ML in public life. This paper examines the costs associated with achieving these objectives in distributed ML architectures, from both theoretical and empirical perspectives. We overview the meanings of privacy and robustness in distributed ML, and clarify how they can be achieved efficiently in isolation. However, we contend that the integration of these two objectives entails a notable compromise in computational efficiency. In short, traditional noise injection hurts accuracy by concealing poisoned inputs, while cryptographic methods clash with poisoning defenses due to their non-line
    
[^4]: 基于联邦学习的用于增强无人机隐私和安全的入侵检测系统

    A Novel Federated Learning-Based IDS for Enhancing UAVs Privacy and Security

    [https://arxiv.org/abs/2312.04135](https://arxiv.org/abs/2312.04135)

    本论文引入了基于联邦学习的入侵检测系统(FL-IDS)，旨在解决FANETs中集中式系统所遇到的挑战，降低了计算和存储成本，适合资源受限的无人机。

    

    无人机在飞行自组织网络(FANETs)中运行时会遇到安全挑战，因为这些网络具有动态和分布式的特性。先前的研究主要集中在集中式入侵检测上，假设一个中央实体负责存储和分析来自所有设备的数据。然而，这些方法面临计算和存储成本以及单点故障风险等挑战，威胁到数据隐私和可用性。数据在互连设备之间广泛分散的情况突显了去中心化方法的必要性。本文介绍了基于联邦学习的入侵检测系统(FL-IDS)，解决了FANETs中集中式系统遇到的挑战。FL-IDS在去中心化方式下运行，降低了客户端和中央服务器的计算和存储成本，这对于资源受限的无人机至关重要。

    arXiv:2312.04135v2 Announce Type: replace-cross  Abstract: Unmanned aerial vehicles (UAVs) operating within Flying Ad-hoc Networks (FANETs) encounter security challenges due to the dynamic and distributed nature of these networks. Previous studies predominantly focused on centralized intrusion detection, assuming a central entity responsible for storing and analyzing data from all devices.However, these approaches face challenges including computation and storage costs, along with a single point of failure risk, threatening data privacy and availability. The widespread dispersion of data across interconnected devices underscores the necessity for decentralized approaches. This paper introduces the Federated Learning-based Intrusion Detection System (FL-IDS), addressing challenges encountered by centralized systems in FANETs. FL-IDS reduces computation and storage costs for both clients and the central server, crucial for resource-constrained UAVs. Operating in a decentralized manner, F
    
[^5]: PuriDefense：用于防御黑盒基于查询的攻击的随机局部隐式对抗净化

    PuriDefense: Randomized Local Implicit Adversarial Purification for Defending Black-box Query-based Attacks. (arXiv:2401.10586v1 [cs.CR])

    [http://arxiv.org/abs/2401.10586](http://arxiv.org/abs/2401.10586)

    PuriDefense是一种高效的防御机制，通过使用轻量级净化模型进行随机路径净化，减缓基于查询的攻击的收敛速度，并有效防御黑盒基于查询的攻击。

    

    黑盒基于查询的攻击对机器学习作为服务系统构成重大威胁，因为它们可以生成对抗样本而不需要访问目标模型的架构和参数。传统的防御机制，如对抗训练、梯度掩盖和输入转换，要么带来巨大的计算成本，要么损害非对抗输入的测试准确性。为了应对这些挑战，我们提出了一种高效的防御机制PuriDefense，在低推理成本的级别上使用轻量级净化模型的随机路径净化。这些模型利用局部隐式函数并重建自然图像流形。我们的理论分析表明，这种方法通过将随机性纳入净化过程来减缓基于查询的攻击的收敛速度。对CIFAR-10和ImageNet的大量实验验证了我们提出的净化器防御的有效性。

    Black-box query-based attacks constitute significant threats to Machine Learning as a Service (MLaaS) systems since they can generate adversarial examples without accessing the target model's architecture and parameters. Traditional defense mechanisms, such as adversarial training, gradient masking, and input transformations, either impose substantial computational costs or compromise the test accuracy of non-adversarial inputs. To address these challenges, we propose an efficient defense mechanism, PuriDefense, that employs random patch-wise purifications with an ensemble of lightweight purification models at a low level of inference cost. These models leverage the local implicit function and rebuild the natural image manifold. Our theoretical analysis suggests that this approach slows down the convergence of query-based attacks by incorporating randomness into purifications. Extensive experiments on CIFAR-10 and ImageNet validate the effectiveness of our proposed purifier-based defen
    
[^6]: 连续学习作为计算受限的强化学习

    Continual Learning as Computationally Constrained Reinforcement Learning. (arXiv:2307.04345v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2307.04345](http://arxiv.org/abs/2307.04345)

    本文研究了连续学习作为计算受限的强化学习的主题，提出了一个框架和一套工具来解决人工智能领域长期以来的挑战并促进进一步的研究。

    

    一种能够在漫长的生命周期内高效积累知识并发展越来越复杂技能的智能体可以推动人工智能能力的前沿。连续学习这一长期以来一直是人工智能领域的挑战，本文介绍了关于连续学习的概念并提出了一个框架和一套工具，以促进进一步的研究。

    An agent that efficiently accumulates knowledge to develop increasingly sophisticated skills over a long lifetime could advance the frontier of artificial intelligence capabilities. The design of such agents, which remains a long-standing challenge of artificial intelligence, is addressed by the subject of continual learning. This monograph clarifies and formalizes concepts of continual learning, introducing a framework and set of tools to stimulate further research.
    
[^7]: 具有编码数据结构的变分量子回归算法

    Variational quantum regression algorithm with encoded data structure. (arXiv:2307.03334v1 [quant-ph])

    [http://arxiv.org/abs/2307.03334](http://arxiv.org/abs/2307.03334)

    本文介绍了一个具有编码数据结构的变分量子回归算法，在量子机器学习中具有模型解释性，并能有效地处理互连度较高的量子比特。算法通过压缩编码和数字-模拟门操作，大大提高了在噪声中尺度量子计算机上的运行时间复杂度。

    

    变分量子算法(VQAs)被广泛应用于解决实际问题，如组合优化、量子化学模拟、量子机器学习和噪声量子计算机上的量子错误纠正。对于变分量子机器学习，尚未开发出将模型解释性内嵌到算法中的变分算法。本文构建了一个量子回归算法，并确定了变分参数与学习回归系数之间的直接关系，同时采用了将数据直接编码为反映经典数据表结构的量子幅度的电路。该算法特别适用于互连度较高的量子比特。通过压缩编码和数字-模拟门操作，运行时间复杂度在数据输入量编码的情况下对数级更有优势，显著提升了噪声中尺度量子计算机的性能。

    Variational quantum algorithms (VQAs) prevail to solve practical problems such as combinatorial optimization, quantum chemistry simulation, quantum machine learning, and quantum error correction on noisy quantum computers. For variational quantum machine learning, a variational algorithm with model interpretability built into the algorithm is yet to be exploited. In this paper, we construct a quantum regression algorithm and identify the direct relation of variational parameters to learned regression coefficients, while employing a circuit that directly encodes the data in quantum amplitudes reflecting the structure of the classical data table. The algorithm is particularly suitable for well-connected qubits. With compressed encoding and digital-analog gate operation, the run time complexity is logarithmically more advantageous than that for digital 2-local gate native hardware with the number of data entries encoded, a decent improvement in noisy intermediate-scale quantum computers a
    
[^8]: 无稀疏性的高维情境赌博问题研究

    High-dimensional Contextual Bandit Problem without Sparsity. (arXiv:2306.11017v1 [stat.ML])

    [http://arxiv.org/abs/2306.11017](http://arxiv.org/abs/2306.11017)

    本论文研究了高维情境赌博问题，无需施加稀疏性要求，并提出了一种探索-开发算法以解决此问题。研究表明，可以通过平衡探索和开发实现最优速率。同时，还介绍了一种自适应探索-开发算法来找到最优平衡点。

    

    本研究探讨了高维线性情境赌博问题，其中特征数 $p$ 大于预算 $T$ 或甚至无限制。与此领域的大部分研究不同的是，我们不对回归系数施加稀疏性要求。相反，我们依靠最近关于过参数化模型的研究成果，从而能够在数据分布具有较小有效秩时分析最小范数插值估计器的性能。我们提出了一个探索-开发 (EtC) 算法来解决这个问题，并检验了它的性能。通过我们的分析，我们以 $T$ 为变量，导出了ETC算法的最优速率，并表明这个速率可以通过平衡探索和开发来实现。此外，我们介绍了一种自适应探索-开发 (AEtC)算法，它可以自适应地找到最优平衡点。我们通过一系列模拟评估了所提出算法的性能。

    In this research, we investigate the high-dimensional linear contextual bandit problem where the number of features $p$ is greater than the budget $T$, or it may even be infinite. Differing from the majority of previous works in this field, we do not impose sparsity on the regression coefficients. Instead, we rely on recent findings on overparameterized models, which enables us to analyze the performance the minimum-norm interpolating estimator when data distributions have small effective ranks. We propose an explore-then-commit (EtC) algorithm to address this problem and examine its performance. Through our analysis, we derive the optimal rate of the ETC algorithm in terms of $T$ and show that this rate can be achieved by balancing exploration and exploitation. Moreover, we introduce an adaptive explore-then-commit (AEtC) algorithm that adaptively finds the optimal balance. We assess the performance of the proposed algorithms through a series of simulations.
    
[^9]: StyleNAT：给每个头部一个新的视角

    StyleNAT: Giving Each Head a New Perspective. (arXiv:2211.05770v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2211.05770](http://arxiv.org/abs/2211.05770)

    StyleNAT是一个新的基于transformer的图像生成框架，通过使用邻域注意力（NA）来捕捉局部和全局信息，能够高效灵活地适应不同的数据集，并在FFHQ-256上取得了新的最佳结果。

    

    图像生成一直是一个既期望又具有挑战性的任务，以高效的方式执行生成任务同样困难。通常，研究人员试图创建一个“一刀切”的生成器，在参数空间中，即使是截然不同的数据集，也有很少的差异。在这里，我们提出了一种新的基于transformer的框架，称为StyleNAT，旨在实现高质量的图像生成，并具有卓越的效率和灵活性。在我们的模型核心是一个精心设计的框架，它将注意力头部划分为捕捉局部和全局信息的方式，这是通过使用邻域注意力（NA）实现的。由于不同的头部能够关注不同的感受野，模型能够更好地结合这些信息，并以高度灵活的方式适应手头的数据。StyleNAT在FFHQ-256上获得了新的SOTA FID得分2.046 ，击败了以卷积模型（如StyleGAN-XL）和transformer模型（如HIT）为基础的先前方法。

    Image generation has been a long sought-after but challenging task, and performing the generation task in an efficient manner is similarly difficult. Often researchers attempt to create a "one size fits all" generator, where there are few differences in the parameter space for drastically different datasets. Herein, we present a new transformer-based framework, dubbed StyleNAT, targeting high-quality image generation with superior efficiency and flexibility. At the core of our model, is a carefully designed framework that partitions attention heads to capture local and global information, which is achieved through using Neighborhood Attention (NA). With different heads able to pay attention to varying receptive fields, the model is able to better combine this information, and adapt, in a highly flexible manner, to the data at hand. StyleNAT attains a new SOTA FID score on FFHQ-256 with 2.046, beating prior arts with convolutional models such as StyleGAN-XL and transformers such as HIT 
    

