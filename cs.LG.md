# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Survey on Self-Supervised Pre-Training of Graph Foundation Models: A Knowledge-Based Perspective](https://arxiv.org/abs/2403.16137) | 该论文从基于知识的角度全面调查和分析了图基础模型的自监督预训练任务，涉及微观和宏观知识，包括9个知识类别、25个预训练任务以及各种下游任务适应策略。 |
| [^2] | [An Ordering of Divergences for Variational Inference with Factorized Gaussian Approximations](https://arxiv.org/abs/2403.13748) | 不同的散度排序可以通过它们的变分近似误估不确定性的各种度量，并且因子化近似无法同时匹配这些度量中的任意两个 |
| [^3] | [Imitation-regularized Optimal Transport on Networks: Provable Robustness and Application to Logistics Planning](https://arxiv.org/abs/2402.17967) | 本研究探讨了在网络上进行模仿正则化的最优输运（I-OT），通过模仿先验分布来提高网络系统的鲁棒性。 |
| [^4] | [Invertible Fourier Neural Operators for Tackling Both Forward and Inverse Problems](https://arxiv.org/abs/2402.11722) | 本文提出了可逆傅立叶神经算子(iFNO)，通过设计可逆傅立叶块和集成变分自动编码器，实现同时处理前向与反向问题的能力，为双向任务的学习提供有效的参数共享和信息交换，克服了不适定性、数据短缺和噪声等挑战。 |
| [^5] | [TinyCL: An Efficient Hardware Architecture for Continual Learning on Autonomous Systems](https://arxiv.org/abs/2402.09780) | TinyCL是一种用于自主系统持续学习的高效硬件架构，在CL中支持前向和反向传播，并通过滑动窗口的连续学习策略来减少内存访问。 |
| [^6] | [FreDF: Learning to Forecast in Frequency Domain](https://arxiv.org/abs/2402.02399) | FreDF是一种在频域中学习预测的方法，解决了时间序列建模中标签序列的自相关问题，相比现有方法有更好的性能表现，并且与各种预测模型兼容。 |
| [^7] | [A backdoor attack against link prediction tasks with graph neural networks.](http://arxiv.org/abs/2401.02663) | 本文研究了一种针对图神经网络链接预测任务的后门攻击方法，发现GNN模型容易受到后门攻击，提出了针对该任务的后门攻击方式。 |
| [^8] | [Revealing CNN Architectures via Side-Channel Analysis in Dataflow-based Inference Accelerators.](http://arxiv.org/abs/2311.00579) | 本文通过评估数据流加速器上的侧信道信息，提出了一种攻击方法来恢复CNN模型的架构。该攻击利用了数据流映射的数据重用以及架构线索，成功恢复了流行的CNN模型Lenet，Alexnet和VGGnet16的结构。 |
| [^9] | [Closed-Form Diffusion Models.](http://arxiv.org/abs/2310.12395) | 本研究提出了一种闭式扩散模型，通过显式平滑的闭式得分函数来生成新样本，无需训练，且在消费级CPU上能够实现与神经SGMs相竞争的采样速度。 |
| [^10] | [ParFam -- Symbolic Regression Based on Continuous Global Optimization.](http://arxiv.org/abs/2310.05537) | ParFam是一种新的符号回归方法，利用参数化的符号函数族将离散问题转化为连续问题，并结合全局优化器，能够有效解决符号回归问题。 |
| [^11] | [Breaking NoC Anonymity using Flow Correlation Attack.](http://arxiv.org/abs/2309.15687) | 本文研究了NoC架构中现有匿名路由协议的安全性，并展示了现有的匿名路由对基于机器学习的流相关攻击易受攻击。我们提出了一种轻量级的匿名路由，使用流量混淆技术，可以抵御基于机器学习的流相关攻击。 |
| [^12] | [A Unified Framework for Exploratory Learning-Aided Community Detection in Networks with Unknown Topology.](http://arxiv.org/abs/2304.04497) | META-CODE是一个统一的框架，通过探索学习和易于收集的节点元数据，在未知拓扑网络中检测重叠社区。实验结果证明了META-CODE的有效性和可扩展性。 |
| [^13] | [q-Learning in Continuous Time.](http://arxiv.org/abs/2207.00713) | 本文研究了连续时间下的q-Learning，通过引入小q函数作为一阶近似，研究了q-learning理论，应用于设计不同的演员-评论家算法。 |

# 详细

[^1]: 自监督预训练图基础模型的调查：基于知识的视角

    A Survey on Self-Supervised Pre-Training of Graph Foundation Models: A Knowledge-Based Perspective

    [https://arxiv.org/abs/2403.16137](https://arxiv.org/abs/2403.16137)

    该论文从基于知识的角度全面调查和分析了图基础模型的自监督预训练任务，涉及微观和宏观知识，包括9个知识类别、25个预训练任务以及各种下游任务适应策略。

    

    图自监督学习现在是预训练图基础模型的首选方法，包括图神经网络、图变换器，以及更近期的基于大型语言模型（LLM）的图模型。文章全面调查和分析了基于知识的视角下的图基础模型的预训练任务，包括微观（节点、链接等）和宏观知识（簇、全局结构等）。涵盖了共计9个知识类别和25个预训练任务，以及各种下游任务适应策略。

    arXiv:2403.16137v1 Announce Type: new  Abstract: Graph self-supervised learning is now a go-to method for pre-training graph foundation models, including graph neural networks, graph transformers, and more recent large language model (LLM)-based graph models. There is a wide variety of knowledge patterns embedded in the structure and properties of graphs which may be used for pre-training, but we lack a systematic overview of self-supervised pre-training tasks from the perspective of graph knowledge. In this paper, we comprehensively survey and analyze the pre-training tasks of graph foundation models from a knowledge-based perspective, consisting of microscopic (nodes, links, etc) and macroscopic knowledge (clusters, global structure, etc). It covers a total of 9 knowledge categories and 25 pre-training tasks, as well as various downstream task adaptation strategies. Furthermore, an extensive list of the related papers with detailed metadata is provided at https://github.com/Newiz430/
    
[^2]: 变分推断中因子化高斯近似的差异排序

    An Ordering of Divergences for Variational Inference with Factorized Gaussian Approximations

    [https://arxiv.org/abs/2403.13748](https://arxiv.org/abs/2403.13748)

    不同的散度排序可以通过它们的变分近似误估不确定性的各种度量，并且因子化近似无法同时匹配这些度量中的任意两个

    

    在变分推断（VI）中，给定一个难以处理的分布$p$，问题是从一些更易处理的族$\mathcal{Q}$中计算最佳近似$q$。通常情况下，这种近似是通过最小化Kullback-Leibler (KL)散度来找到的。然而，存在其他有效的散度选择，当$\mathcal{Q}$不包含$p$时，每个散度都支持不同的解决方案。我们分析了在高斯的密集协方差矩阵被对角协方差矩阵的高斯近似所影响的VI结果中，散度选择如何影响VI结果。在这种设置中，我们展示了不同的散度可以通过它们的变分近似误估不确定性的各种度量，如方差、精度和熵，进行\textit{排序}。我们还得出一个不可能定理，表明无法通过因子化近似同时匹配这些度量中的任意两个；因此

    arXiv:2403.13748v1 Announce Type: cross  Abstract: Given an intractable distribution $p$, the problem of variational inference (VI) is to compute the best approximation $q$ from some more tractable family $\mathcal{Q}$. Most commonly the approximation is found by minimizing a Kullback-Leibler (KL) divergence. However, there exist other valid choices of divergences, and when $\mathcal{Q}$ does not contain~$p$, each divergence champions a different solution. We analyze how the choice of divergence affects the outcome of VI when a Gaussian with a dense covariance matrix is approximated by a Gaussian with a diagonal covariance matrix. In this setting we show that different divergences can be \textit{ordered} by the amount that their variational approximations misestimate various measures of uncertainty, such as the variance, precision, and entropy. We also derive an impossibility theorem showing that no two of these measures can be simultaneously matched by a factorized approximation; henc
    
[^3]: 在网络上进行模仿正则化的最优输运：可证明的鲁棒性及其在物流规划中的应用

    Imitation-regularized Optimal Transport on Networks: Provable Robustness and Application to Logistics Planning

    [https://arxiv.org/abs/2402.17967](https://arxiv.org/abs/2402.17967)

    本研究探讨了在网络上进行模仿正则化的最优输运（I-OT），通过模仿先验分布来提高网络系统的鲁棒性。

    

    网络系统构成了现代社会的基础，在各种应用中起着至关重要的作用。然而，这些系统面临着由灾难等不可预见情况带来的重大风险。鉴于此，迫切需要研究加强网络系统的鲁棒性。最近在强化学习中，已经确定了获取鲁棒性和正则化熵之间的关系。此外，在这一框架内使用了模仿学习来反映专家的行为。然而，关于在网络上的最优输运中使用类似模仿框架的全面研究还没有。因此，在本研究中，研究了在网络上进行的模仿正则化的最优输运（I-OT）。它通过模仿给定的先验分布对网络的先验知识进行编码。I-OT解决方案在网络上定义的成本方面表现出了鲁棒性。

    arXiv:2402.17967v1 Announce Type: new  Abstract: Network systems form the foundation of modern society, playing a critical role in various applications. However, these systems are at significant risk of being adversely affected by unforeseen circumstances, such as disasters. Considering this, there is a pressing need for research to enhance the robustness of network systems. Recently, in reinforcement learning, the relationship between acquiring robustness and regularizing entropy has been identified. Additionally, imitation learning is used within this framework to reflect experts' behavior. However, there are no comprehensive studies on the use of a similar imitation framework for optimal transport on networks. Therefore, in this study, imitation-regularized optimal transport (I-OT) on networks was investigated. It encodes prior knowledge on the network by imitating a given prior distribution. The I-OT solution demonstrated robustness in terms of the cost defined on the network. More
    
[^4]: 可逆傅立叶神经算子处理前向和反问题

    Invertible Fourier Neural Operators for Tackling Both Forward and Inverse Problems

    [https://arxiv.org/abs/2402.11722](https://arxiv.org/abs/2402.11722)

    本文提出了可逆傅立叶神经算子(iFNO)，通过设计可逆傅立叶块和集成变分自动编码器，实现同时处理前向与反向问题的能力，为双向任务的学习提供有效的参数共享和信息交换，克服了不适定性、数据短缺和噪声等挑战。

    

    傅立叶神经算子（FNO）是一种流行的算子学习方法，已在许多任务中表现出色。然而，FNO主要用于前向预测，而许多应用程序依赖于解决反问题。在本文中，我们提出了一种可逆傅立叶神经算子（iFNO），旨在解决前向和反向问题。我们在潜在通道空间中设计了一系列可逆傅立叶块，以分享模型参数，有效交换信息，并相互正规化双向任务的学习。我们集成了变分自动编码器以捕获输入空间内在结构，并实现后验推断，以克服不适定性、数据短缺、噪声等挑战。我们开发了一个三步过程，用于预训练和微调以实现高效训练。对五个基准问题的评估已经证明...

    arXiv:2402.11722v1 Announce Type: new  Abstract: Fourier Neural Operator (FNO) is a popular operator learning method, which has demonstrated state-of-the-art performance across many tasks. However, FNO is mainly used in forward prediction, yet a large family of applications rely on solving inverse problems. In this paper, we propose an invertible Fourier Neural Operator (iFNO) that tackles both the forward and inverse problems. We designed a series of invertible Fourier blocks in the latent channel space to share the model parameters, efficiently exchange the information, and mutually regularize the learning for the bi-directional tasks. We integrated a variational auto-encoder to capture the intrinsic structures within the input space and to enable posterior inference so as to overcome challenges of illposedness, data shortage, noises, etc. We developed a three-step process for pre-training and fine tuning for efficient training. The evaluations on five benchmark problems have demonst
    
[^5]: TinyCL:一种用于自主系统持续学习的高效硬件架构

    TinyCL: An Efficient Hardware Architecture for Continual Learning on Autonomous Systems

    [https://arxiv.org/abs/2402.09780](https://arxiv.org/abs/2402.09780)

    TinyCL是一种用于自主系统持续学习的高效硬件架构，在CL中支持前向和反向传播，并通过滑动窗口的连续学习策略来减少内存访问。

    

    持续学习（CL）范式包括不断演化深度神经网络（DNN）模型的参数，以逐步学习执行新任务，而不降低先前任务的性能，即避免所谓的灾难性遗忘。然而，在基于CL的自主系统中，DNN参数更新对资源要求极高。现有的DNN加速器不能直接用于CL，因为它们只支持前向传播的执行。只有少数先前的架构执行反向传播和权重更新，但它们缺乏对CL的控制和管理。为此，我们设计了一个硬件架构TinyCL，用于在资源受限的自主系统上进行持续学习。它包括一个执行前向和反向传播的处理单元，以及一个管理基于内存的CL工作负载的控制单元。为了最小化内存访问，我们使用了滑动窗口的连续学习策略。

    arXiv:2402.09780v1 Announce Type: new  Abstract: The Continuous Learning (CL) paradigm consists of continuously evolving the parameters of the Deep Neural Network (DNN) model to progressively learn to perform new tasks without reducing the performance on previous tasks, i.e., avoiding the so-called catastrophic forgetting. However, the DNN parameter update in CL-based autonomous systems is extremely resource-hungry. The existing DNN accelerators cannot be directly employed in CL because they only support the execution of the forward propagation. Only a few prior architectures execute the backpropagation and weight update, but they lack the control and management for CL. Towards this, we design a hardware architecture, TinyCL, to perform CL on resource-constrained autonomous systems. It consists of a processing unit that executes both forward and backward propagation, and a control unit that manages memory-based CL workload. To minimize the memory accesses, the sliding window of the con
    
[^6]: FreDF: 在频域中学习预测

    FreDF: Learning to Forecast in Frequency Domain

    [https://arxiv.org/abs/2402.02399](https://arxiv.org/abs/2402.02399)

    FreDF是一种在频域中学习预测的方法，解决了时间序列建模中标签序列的自相关问题，相比现有方法有更好的性能表现，并且与各种预测模型兼容。

    

    时间序列建模在历史序列和标签序列中都面临自相关的挑战。当前的研究主要集中在处理历史序列中的自相关问题，但往往忽视了标签序列中的自相关存在。具体来说，新兴的预测模型主要遵循直接预测（DF）范式，在标签序列中假设条件独立性下生成多步预测。这种假设忽视了标签序列中固有的自相关性，从而限制了基于DF的模型的性能。针对这一问题，我们引入了频域增强直接预测（FreDF），通过在频域中学习预测来避免标签自相关的复杂性。我们的实验证明，FreDF在性能上大大超过了包括iTransformer在内的现有最先进方法，并且与各种预测模型兼容。

    Time series modeling is uniquely challenged by the presence of autocorrelation in both historical and label sequences. Current research predominantly focuses on handling autocorrelation within the historical sequence but often neglects its presence in the label sequence. Specifically, emerging forecast models mainly conform to the direct forecast (DF) paradigm, generating multi-step forecasts under the assumption of conditional independence within the label sequence. This assumption disregards the inherent autocorrelation in the label sequence, thereby limiting the performance of DF-based models. In response to this gap, we introduce the Frequency-enhanced Direct Forecast (FreDF), which bypasses the complexity of label autocorrelation by learning to forecast in the frequency domain. Our experiments demonstrate that FreDF substantially outperforms existing state-of-the-art methods including iTransformer and is compatible with a variety of forecast models.
    
[^7]: 用于图神经网络链接预测任务的后门攻击

    A backdoor attack against link prediction tasks with graph neural networks. (arXiv:2401.02663v1 [cs.LG])

    [http://arxiv.org/abs/2401.02663](http://arxiv.org/abs/2401.02663)

    本文研究了一种针对图神经网络链接预测任务的后门攻击方法，发现GNN模型容易受到后门攻击，提出了针对该任务的后门攻击方式。

    

    图神经网络（GNN）是一类能够处理图结构数据的深度学习模型，在各种实际应用中表现出显著的性能。最近的研究发现，GNN模型容易受到后门攻击。当具体的模式（称为后门触发器，例如子图、节点等）出现在输入数据中时，嵌入在GNN模型中的后门会被激活，将输入数据误分类为攻击者指定的目标类标签，而当输入中没有后门触发器时，嵌入在GNN模型中的后门不会被激活，模型正常工作。后门攻击具有极高的隐蔽性，给GNN模型带来严重的安全风险。目前，对GNN的后门攻击研究主要集中在图分类和节点分类等任务上，对链接预测任务的后门攻击研究较少。在本文中，我们提出一种后门攻击方法。

    Graph Neural Networks (GNNs) are a class of deep learning models capable of processing graph-structured data, and they have demonstrated significant performance in a variety of real-world applications. Recent studies have found that GNN models are vulnerable to backdoor attacks. When specific patterns (called backdoor triggers, e.g., subgraphs, nodes, etc.) appear in the input data, the backdoor embedded in the GNN models is activated, which misclassifies the input data into the target class label specified by the attacker, whereas when there are no backdoor triggers in the input, the backdoor embedded in the GNN models is not activated, and the models work normally. Backdoor attacks are highly stealthy and expose GNN models to serious security risks. Currently, research on backdoor attacks against GNNs mainly focus on tasks such as graph classification and node classification, and backdoor attacks against link prediction tasks are rarely studied. In this paper, we propose a backdoor a
    
[^8]: 通过数据流推理加速器中的侧信道分析揭示CNN架构

    Revealing CNN Architectures via Side-Channel Analysis in Dataflow-based Inference Accelerators. (arXiv:2311.00579v1 [cs.CR])

    [http://arxiv.org/abs/2311.00579](http://arxiv.org/abs/2311.00579)

    本文通过评估数据流加速器上的侧信道信息，提出了一种攻击方法来恢复CNN模型的架构。该攻击利用了数据流映射的数据重用以及架构线索，成功恢复了流行的CNN模型Lenet，Alexnet和VGGnet16的结构。

    

    卷积神经网络（CNN）广泛应用于各个领域。最近在基于数据流的CNN加速器的进展使得CNN推理可以在资源有限的边缘设备上进行。这些数据流加速器利用卷积层的固有数据重用来高效处理CNN模型。隐藏CNN模型的架构对于隐私和安全至关重要。本文评估了基于内存的侧信道信息，以从数据流加速器中恢复CNN架构。所提出的攻击利用了CNN加速器上数据流映射的空间和时间数据重用以及架构线索来恢复CNN模型的结构。实验结果表明，我们提出的侧信道攻击可以恢复流行的CNN模型Lenet，Alexnet和VGGnet16的结构。

    Convolution Neural Networks (CNNs) are widely used in various domains. Recent advances in dataflow-based CNN accelerators have enabled CNN inference in resource-constrained edge devices. These dataflow accelerators utilize inherent data reuse of convolution layers to process CNN models efficiently. Concealing the architecture of CNN models is critical for privacy and security. This paper evaluates memory-based side-channel information to recover CNN architectures from dataflow-based CNN inference accelerators. The proposed attack exploits spatial and temporal data reuse of the dataflow mapping on CNN accelerators and architectural hints to recover the structure of CNN models. Experimental results demonstrate that our proposed side-channel attack can recover the structures of popular CNN models, namely Lenet, Alexnet, and VGGnet16.
    
[^9]: 闭式扩散模型

    Closed-Form Diffusion Models. (arXiv:2310.12395v1 [cs.LG])

    [http://arxiv.org/abs/2310.12395](http://arxiv.org/abs/2310.12395)

    本研究提出了一种闭式扩散模型，通过显式平滑的闭式得分函数来生成新样本，无需训练，且在消费级CPU上能够实现与神经SGMs相竞争的采样速度。

    

    基于得分的生成模型(SGMs)通过迭代地使用扰动目标函数的得分函数来从目标分布中采样。对于任何有限的训练集，可以闭式地评估这个得分函数，但由此得到的SGMs会记忆其训练数据，不能生成新样本。在实践中，可以通过训练神经网络来近似得分函数，但这种近似的误差有助于推广，然而神经SGMs的训练和采样代价高，而且对于这种误差提供的有效正则化方法在理论上尚不清楚。因此，在这项工作中，我们采用显式平滑的闭式得分来获得一个生成新样本的SGMs，而无需训练。我们分析了我们的模型，并提出了一个基于最近邻的高效得分函数估计器。利用这个估计器，我们的方法在消费级CPU上运行时能够达到与神经SGMs相竞争的采样速度。

    Score-based generative models (SGMs) sample from a target distribution by iteratively transforming noise using the score function of the perturbed target. For any finite training set, this score function can be evaluated in closed form, but the resulting SGM memorizes its training data and does not generate novel samples. In practice, one approximates the score by training a neural network via score-matching. The error in this approximation promotes generalization, but neural SGMs are costly to train and sample, and the effective regularization this error provides is not well-understood theoretically. In this work, we instead explicitly smooth the closed-form score to obtain an SGM that generates novel samples without training. We analyze our model and propose an efficient nearest-neighbor-based estimator of its score function. Using this estimator, our method achieves sampling times competitive with neural SGMs while running on consumer-grade CPUs.
    
[^10]: ParFam - 基于连续全局优化的符号回归

    ParFam -- Symbolic Regression Based on Continuous Global Optimization. (arXiv:2310.05537v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2310.05537](http://arxiv.org/abs/2310.05537)

    ParFam是一种新的符号回归方法，利用参数化的符号函数族将离散问题转化为连续问题，并结合全局优化器，能够有效解决符号回归问题。

    

    符号回归（SR）问题在许多不同的应用中出现，比如从给定数据中识别物理定律或推导描述金融市场行为的数学方程。目前存在多种解决SR问题的方法，通常基于遗传编程。然而，这些方法通常非常复杂，需要大量超参数调整和计算资源。本文介绍了我们提出的新方法ParFam，它利用适合的符号函数的参数化族将离散的符号回归问题转化为连续问题，相比当前最先进的方法，这种方法的设置更加直观。结合强大的全局优化器，这种方法可以有效地解决SR问题。此外，它可以轻松扩展到更高级的算法，例如添加深度神经网络以找到适合的参数化族。我们证明了这种方法的性能。

    The problem of symbolic regression (SR) arises in many different applications, such as identifying physical laws or deriving mathematical equations describing the behavior of financial markets from given data. Various methods exist to address the problem of SR, often based on genetic programming. However, these methods are usually quite complicated and require a lot of hyperparameter tuning and computational resources. In this paper, we present our new method ParFam that utilizes parametric families of suitable symbolic functions to translate the discrete symbolic regression problem into a continuous one, resulting in a more straightforward setup compared to current state-of-the-art methods. In combination with a powerful global optimizer, this approach results in an effective method to tackle the problem of SR. Furthermore, it can be easily extended to more advanced algorithms, e.g., by adding a deep neural network to find good-fitting parametric families. We prove the performance of 
    
[^11]: 打破NoC匿名性使用流相关攻击

    Breaking NoC Anonymity using Flow Correlation Attack. (arXiv:2309.15687v1 [cs.CR])

    [http://arxiv.org/abs/2309.15687](http://arxiv.org/abs/2309.15687)

    本文研究了NoC架构中现有匿名路由协议的安全性，并展示了现有的匿名路由对基于机器学习的流相关攻击易受攻击。我们提出了一种轻量级的匿名路由，使用流量混淆技术，可以抵御基于机器学习的流相关攻击。

    

    网络片上互连（NoC）广泛用作当今多核片上系统（SoC）设计中的内部通信结构。片上通信的安全性至关重要，因为利用共享的NoC中的任何漏洞对攻击者来说都是一个富矿。NoC安全依赖于对各种攻击的有效防范措施。我们研究了NoC架构中现有匿名路由协议的安全性。具体而言，本文作出了两个重要贡献。我们展示了现有的匿名路由对基于机器学习（ML）的流相关攻击是易受攻击的。我们提出了一种轻量级的匿名路由，使用流量混淆技术，可以抵御基于ML的流相关攻击。使用实际和合成流量进行的实验研究表明，我们提出的攻击能够成功地对抗NoC架构中最先进的匿名路由，对于多种流量模式的分类准确率高达99％，同时。

    Network-on-Chip (NoC) is widely used as the internal communication fabric in today's multicore System-on-Chip (SoC) designs. Security of the on-chip communication is crucial because exploiting any vulnerability in shared NoC would be a goldmine for an attacker. NoC security relies on effective countermeasures against diverse attacks. We investigate the security strength of existing anonymous routing protocols in NoC architectures. Specifically, this paper makes two important contributions. We show that the existing anonymous routing is vulnerable to machine learning (ML) based flow correlation attacks on NoCs. We propose a lightweight anonymous routing that use traffic obfuscation techniques which can defend against ML-based flow correlation attacks. Experimental studies using both real and synthetic traffic reveal that our proposed attack is successful against state-of-the-art anonymous routing in NoC architectures with a high accuracy (up to 99%) for diverse traffic patterns, while o
    
[^12]: 未知拓扑网络中的探索学习辅助社区检测的统一框架

    A Unified Framework for Exploratory Learning-Aided Community Detection in Networks with Unknown Topology. (arXiv:2304.04497v2 [cs.SI] UPDATED)

    [http://arxiv.org/abs/2304.04497](http://arxiv.org/abs/2304.04497)

    META-CODE是一个统一的框架，通过探索学习和易于收集的节点元数据，在未知拓扑网络中检测重叠社区。实验结果证明了META-CODE的有效性和可扩展性。

    

    在社交网络中，发现社区结构作为各种网络分析任务中的一个基本问题受到了广泛关注。然而，由于隐私问题或访问限制，网络结构通常是未知的，这使得现有的社区检测方法在没有昂贵的网络拓扑获取的情况下无效。为了解决这个挑战，我们提出了 META-CODE，这是一个统一的框架，通过探索学习辅助易于收集的节点元数据，在未知拓扑网络中检测重叠社区。具体而言，META-CODE 除了初始的网络推理步骤外，还包括三个迭代步骤：1) 基于图神经网络（GNNs）的节点级社区归属嵌入，通过我们的新重构损失进行训练，2) 基于社区归属的节点查询进行网络探索，3) 使用探索网络中的基于边连接的连体神经网络模型进行网络推理。通过实验结果证明了 META-CODE 的有效性和可扩展性。

    In social networks, the discovery of community structures has received considerable attention as a fundamental problem in various network analysis tasks. However, due to privacy concerns or access restrictions, the network structure is often unknown, thereby rendering established community detection approaches ineffective without costly network topology acquisition. To tackle this challenge, we present META-CODE, a unified framework for detecting overlapping communities in networks with unknown topology via exploratory learning aided by easy-to-collect node metadata. Specifically, META-CODE consists of three iterative steps in addition to the initial network inference step: 1) node-level community-affiliation embeddings based on graph neural networks (GNNs) trained by our new reconstruction loss, 2) network exploration via community-affiliation-based node queries, and 3) network inference using an edge connectivity-based Siamese neural network model from the explored network. Through e
    
[^13]: 连续时间下的q-Learning

    q-Learning in Continuous Time. (arXiv:2207.00713v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2207.00713](http://arxiv.org/abs/2207.00713)

    本文研究了连续时间下的q-Learning，通过引入小q函数作为一阶近似，研究了q-learning理论，应用于设计不同的演员-评论家算法。

    

    我们研究了基于熵正则化的探索性扩散过程的Q-learning在连续时间下的应用。我们引入了“小q函数”作为大Q函数的一阶近似，研究了q函数的q-learning理论，并应用于设计不同的演员-评论家算法。

    We study the continuous-time counterpart of Q-learning for reinforcement learning (RL) under the entropy-regularized, exploratory diffusion process formulation introduced by Wang et al. (2020). As the conventional (big) Q-function collapses in continuous time, we consider its first-order approximation and coin the term ``(little) q-function". This function is related to the instantaneous advantage rate function as well as the Hamiltonian. We develop a ``q-learning" theory around the q-function that is independent of time discretization. Given a stochastic policy, we jointly characterize the associated q-function and value function by martingale conditions of certain stochastic processes, in both on-policy and off-policy settings. We then apply the theory to devise different actor-critic algorithms for solving underlying RL problems, depending on whether or not the density function of the Gibbs measure generated from the q-function can be computed explicitly. One of our algorithms inter
    

