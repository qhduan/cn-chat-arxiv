# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Taming Pre-trained LLMs for Generalised Time Series Forecasting via Cross-modal Knowledge Distillation](https://arxiv.org/abs/2403.07300) | 通过跨模态知识蒸馏和LLMs对齐框架，该方法利用静态和动态知识，充分释放LLMs在时间序列预测中的潜力 |
| [^2] | [SoK: Challenges and Opportunities in Federated Unlearning](https://arxiv.org/abs/2403.02437) | 联邦学习引入了新的隐私要求，促使研究开始关注适用于联邦学习环境的反学习机制。 |
| [^3] | [Ising on the Graph: Task-specific Graph Subsampling via the Ising Model](https://arxiv.org/abs/2402.10206) | 该论文提出了一种基于伊辛模型的图子抽样方法，可以针对特定任务在图结构上进行减小，并通过学习伊辛模型的外部磁场来实现。该方法的多功能性在图像分割、三维形状稀疏化和稀疏逼近矩阵求逆等应用中得到展示。 |
| [^4] | [Analysis of Linear Mode Connectivity via Permutation-Based Weight Matching](https://arxiv.org/abs/2402.04051) | 通过基于排列的权重匹配分析线性模式连接性，我们实验证明了通过权重匹配找到的排列可以改变权重矩阵奇异向量的方向，但不能改变奇异值。这一发现对于理解随机梯度下降的有效性及其在模型合并等领域的应用具有重要意义。 |
| [^5] | [TinyFormer: Efficient Transformer Design and Deployment on Tiny Devices.](http://arxiv.org/abs/2311.01759) | TinyFormer是一个具有SuperNAS、SparseNAS和SparseEngine组成的框架，专门用于在MCUs上开发和部署资源高效的transformer模型。其创新之处在于提出了SparseEngine，这是第一个可以在MCUs上执行稀疏模型的transformer推理的部署框架。 |
| [^6] | [Mitigating Communication Costs in Neural Networks: The Role of Dendritic Nonlinearity.](http://arxiv.org/abs/2306.11950) | 本研究发现，在神经网络中整合非线性树突结构可以显著提高模型的容量和性能，同时控制信号通信成本，这对于未来神经网络的发展具有重要的意义。 |
| [^7] | [Accelerating Generalized Random Forests with Fixed-Point Trees.](http://arxiv.org/abs/2306.11908) | 本文提出一种新的树生长规则，使广义随机森林在无梯度优化的情况下大大节省了时间。 |
| [^8] | [Relating Implicit Bias and Adversarial Attacks through Intrinsic Dimension.](http://arxiv.org/abs/2305.15203) | 本文通过研究神经网络的隐性偏差，着眼于其中涉及的傅里叶频率与图像分类和对抗性攻击之间的关系。研究提出了一种新方法，可以发现这些频率之间的非线性相关性。 |
| [^9] | [Self-Supervised Siamese Autoencoders.](http://arxiv.org/abs/2304.02549) | 本论文提出了一种新的自监督方法，名为SidAE，它结合了孪生架构和去噪自编码器的优点，可以更好地提取输入数据的特征，以在多个下游任务中获得更好的性能。 |
| [^10] | [Penalising the biases in norm regularisation enforces sparsity.](http://arxiv.org/abs/2303.01353) | 本研究表明，控制神经网络参数的范数可以获得良好的泛化性能。对神经网络中偏差项的范数进行惩罚可以实现稀疏估计量。 |
| [^11] | [Scalable Dynamic Mixture Model with Full Covariance for Probabilistic Traffic Forecasting.](http://arxiv.org/abs/2212.06653) | 本文提出了一种扩展性强、适用于复杂概率交通预测的动态混合模型，通过模拟复杂的时变分布以更准确预测交通情况，具有高效性、灵活性和可扩展性。 |

# 详细

[^1]: 通过跨模态知识蒸馏控制预训练LLMs进行广义时间序列预测

    Taming Pre-trained LLMs for Generalised Time Series Forecasting via Cross-modal Knowledge Distillation

    [https://arxiv.org/abs/2403.07300](https://arxiv.org/abs/2403.07300)

    通过跨模态知识蒸馏和LLMs对齐框架，该方法利用静态和动态知识，充分释放LLMs在时间序列预测中的潜力

    

    多变量时间序列预测最近随着深度学习模型的快速增长取得了巨大成功。然而，现有方法通常使用有限的时间数据从头开始训练模型，阻碍了它们的泛化。最近，随着大语言模型（LLMs）的激增，一些工作尝试将LLMs引入时间序列预测中。尽管取得了有希望的结果，但这些方法直接将时间序列作为LLMs的输入，忽略了时间和文本数据之间固有的模态差距。在这项工作中，我们提出了一个新颖的大语言模型和时间序列对齐框架，称为LLaTA，以充分发挥LLMs在时间序列预测挑战中的潜力。基于跨模态知识蒸馏，所提出的方法利用了预训练LLMs中的输入无关静态知识和输入相关动态知识。通过这种方式，该方法为预测模型赋能

    arXiv:2403.07300v1 Announce Type: cross  Abstract: Multivariate time series forecasting has recently gained great success with the rapid growth of deep learning models. However, existing approaches usually train models from scratch using limited temporal data, preventing their generalization. Recently, with the surge of the Large Language Models (LLMs), several works have attempted to introduce LLMs into time series forecasting. Despite promising results, these methods directly take time series as the input to LLMs, ignoring the inherent modality gap between temporal and text data. In this work, we propose a novel Large Language Models and time series alignment framework, dubbed LLaTA, to fully unleash the potentials of LLMs in the time series forecasting challenge. Based on cross-modal knowledge distillation, the proposed method exploits both input-agnostic static knowledge and input-dependent dynamic knowledge in pre-trained LLMs. In this way, it empowers the forecasting model with f
    
[^2]: SoK: 联邦反学习中的挑战与机遇

    SoK: Challenges and Opportunities in Federated Unlearning

    [https://arxiv.org/abs/2403.02437](https://arxiv.org/abs/2403.02437)

    联邦学习引入了新的隐私要求，促使研究开始关注适用于联邦学习环境的反学习机制。

    

    引入于2017年的联邦学习（FL）促进了不信任方之间的合作学习，无需各方明确共享其数据。这允许在尊重GDPR和CPRA等隐私规定的同时，在用户数据上训练模型。然而，新兴的隐私要求可能要求模型所有者能够“遗忘”一些已学习的数据，例如当数据所有者或执法机构要求时。这催生了一个名为“机器反学习”的活跃研究领域。在FL的背景下，许多为集中式环境开发的反学习技术并不容易应用！这是由于FL中集中式和分布式学习之间的独特差异，特别是互动性、随机性、异构性和有限可访问性。为应对这一挑战，最近的一系列研究工作聚焦于开发适用于FL的反学习机制。

    arXiv:2403.02437v1 Announce Type: cross  Abstract: Federated learning (FL), introduced in 2017, facilitates collaborative learning between non-trusting parties with no need for the parties to explicitly share their data among themselves. This allows training models on user data while respecting privacy regulations such as GDPR and CPRA. However, emerging privacy requirements may mandate model owners to be able to \emph{forget} some learned data, e.g., when requested by data owners or law enforcement. This has given birth to an active field of research called \emph{machine unlearning}. In the context of FL, many techniques developed for unlearning in centralized settings are not trivially applicable! This is due to the unique differences between centralized and distributed learning, in particular, interactivity, stochasticity, heterogeneity, and limited accessibility in FL. In response, a recent line of work has focused on developing unlearning mechanisms tailored to FL.   This SoK pape
    
[^3]: 异构图上基于伊辛模型的特定任务图子抽样

    Ising on the Graph: Task-specific Graph Subsampling via the Ising Model

    [https://arxiv.org/abs/2402.10206](https://arxiv.org/abs/2402.10206)

    该论文提出了一种基于伊辛模型的图子抽样方法，可以针对特定任务在图结构上进行减小，并通过学习伊辛模型的外部磁场来实现。该方法的多功能性在图像分割、三维形状稀疏化和稀疏逼近矩阵求逆等应用中得到展示。

    

    减少图的大小同时保持其整体结构是一个具有许多应用的重要问题。通常，减小图的方法要么删除边缘（稀疏化），要么合并节点（粗化），而没有特定的下游任务。在本文中，我们提出了一种使用在节点或边上定义的伊辛模型对图结构进行子抽样的方法，并使用图神经网络学习伊辛模型的外部磁场。我们的方法是任务特定的，因为它可以端到端地学习如何为特定的下游任务减小图的大小。所使用的任务损失函数甚至不需要可微分性。我们在三个不同的应用上展示了我们方法的多功能性：图像分割、三维形状稀疏化和稀疏逼近矩阵求逆。

    arXiv:2402.10206v1 Announce Type: cross  Abstract: Reducing a graph while preserving its overall structure is an important problem with many applications. Typically, the reduction approaches either remove edges (sparsification) or merge nodes (coarsening) in an unsupervised way with no specific downstream task in mind. In this paper, we present an approach for subsampling graph structures using an Ising model defined on either the nodes or edges and learning the external magnetic field of the Ising model using a graph neural network. Our approach is task-specific as it can learn how to reduce a graph for a specific downstream task in an end-to-end fashion. The utilized loss function of the task does not even have to be differentiable. We showcase the versatility of our approach on three distinct applications: image segmentation, 3D shape sparsification, and sparse approximate matrix inverse determination.
    
[^4]: 通过基于排列的权重匹配分析线性模式连接性

    Analysis of Linear Mode Connectivity via Permutation-Based Weight Matching

    [https://arxiv.org/abs/2402.04051](https://arxiv.org/abs/2402.04051)

    通过基于排列的权重匹配分析线性模式连接性，我们实验证明了通过权重匹配找到的排列可以改变权重矩阵奇异向量的方向，但不能改变奇异值。这一发现对于理解随机梯度下降的有效性及其在模型合并等领域的应用具有重要意义。

    

    最近，Ainsworth等人展示了使用权重匹配（WM）来最小化排列搜索模型参数中的$L_2$距离有效地识别满足线性模式连接性（LMC）的排列的方法，其中，在两个具有不同种子的独立训练模型之间的线性路径上的损失保持几乎恒定。本文通过WM提供了LMC的理论分析，这对于理解随机梯度下降的有效性及其在模型合并等领域的应用至关重要。我们首先通过实验和理论分析表明，WM找到的排列并不显着减少两个模型之间的$L_2$距离，而LMC的出现并不仅仅是由于WM本身的距离减小。然后，我们提供了理论洞见，表明排列可以改变每层权重矩阵的奇异向量的方向，但不能改变奇异值。这一发现表明，WM找到的排列主要改变了权重矩阵的方向，而不是奇异值。

    Recently, Ainsworth et al. showed that using weight matching (WM) to minimize the $L_2$ distance in a permutation search of model parameters effectively identifies permutations that satisfy linear mode connectivity (LMC), in which the loss along a linear path between two independently trained models with different seeds remains nearly constant. This paper provides a theoretical analysis of LMC using WM, which is crucial for understanding stochastic gradient descent's effectiveness and its application in areas like model merging. We first experimentally and theoretically show that permutations found by WM do not significantly reduce the $L_2$ distance between two models and the occurrence of LMC is not merely due to distance reduction by WM in itself. We then provide theoretical insights showing that permutations can change the directions of the singular vectors, but not the singular values, of the weight matrices in each layer. This finding shows that permutations found by WM mainly al
    
[^5]: TinyFormer: 高效的Transformer设计和在小型设备上的部署

    TinyFormer: Efficient Transformer Design and Deployment on Tiny Devices. (arXiv:2311.01759v1 [cs.LG])

    [http://arxiv.org/abs/2311.01759](http://arxiv.org/abs/2311.01759)

    TinyFormer是一个具有SuperNAS、SparseNAS和SparseEngine组成的框架，专门用于在MCUs上开发和部署资源高效的transformer模型。其创新之处在于提出了SparseEngine，这是第一个可以在MCUs上执行稀疏模型的transformer推理的部署框架。

    

    在各种嵌入式物联网应用中，以微控制器单元（MCUs）为代表的小型设备上开发深度学习模型引起了广泛关注。然而，由于严重的硬件资源限制，如何高效地设计和部署最新的先进模型（如transformer）在小型设备上是一项挑战。在这项工作中，我们提出了TinyFormer，这是一个特别设计用于在MCUs上开发和部署资源高效的transformer的框架。TinyFormer主要由SuperNAS、SparseNAS和SparseEngine组成。其中，SuperNAS旨在从广大的搜索空间中寻找适当的超网络。SparseNAS评估最佳的稀疏单路径模型，包括从已识别的超网络中提取的transformer架构。最后，SparseEngine将搜索到的稀疏模型高效地部署到MCUs上。据我们所知，SparseEngine是第一个能够在MCUs上执行稀疏模型的transformer推理的部署框架。在CIFAR-10数据集上的评估结果表明，TinyFormer在保持推理精度的同时，相比于传统的transformer模型，减少了大约78％的推理计算量和53％的模型大小。

    Developing deep learning models on tiny devices (e.g. Microcontroller units, MCUs) has attracted much attention in various embedded IoT applications. However, it is challenging to efficiently design and deploy recent advanced models (e.g. transformers) on tiny devices due to their severe hardware resource constraints. In this work, we propose TinyFormer, a framework specifically designed to develop and deploy resource-efficient transformers on MCUs. TinyFormer mainly consists of SuperNAS, SparseNAS and SparseEngine. Separately, SuperNAS aims to search for an appropriate supernet from a vast search space. SparseNAS evaluates the best sparse single-path model including transformer architecture from the identified supernet. Finally, SparseEngine efficiently deploys the searched sparse models onto MCUs. To the best of our knowledge, SparseEngine is the first deployment framework capable of performing inference of sparse models with transformer on MCUs. Evaluation results on the CIFAR-10 da
    
[^6]: 缓解神经网络中的通信成本：树突非线性的作用

    Mitigating Communication Costs in Neural Networks: The Role of Dendritic Nonlinearity. (arXiv:2306.11950v1 [cs.NE])

    [http://arxiv.org/abs/2306.11950](http://arxiv.org/abs/2306.11950)

    本研究发现，在神经网络中整合非线性树突结构可以显著提高模型的容量和性能，同时控制信号通信成本，这对于未来神经网络的发展具有重要的意义。

    

    生物神经网络的理解深刻地影响了人工神经网络（ANNs）的发展。然而，ANN中使用的神经元与其生物模型存在明显偏差，主要是由于缺少包含局部非线性的复杂树突。尽管存在这样的差异，先前的研究表明点神经元可以在执行计算任务方面在功能上替代树突神经元。在本研究中，我们审查了神经网络中非线性树突的重要性。通过使用机器学习方法，我们评估了树突结构非线性对神经网络性能的影响。我们的发现表明，整合树突结构可以在保持信号通信成本有效抑制的同时，显著增强模型容量和性能。这项研究提供了重要的见解，对未来神经网络的发展具有重要的意义。

    Our comprehension of biological neuronal networks has profoundly influenced the evolution of artificial neural networks (ANNs). However, the neurons employed in ANNs exhibit remarkable deviations from their biological analogs, mainly due to the absence of complex dendritic trees encompassing local nonlinearity. Despite such disparities, previous investigations have demonstrated that point neurons can functionally substitute dendritic neurons in executing computational tasks. In this study, we scrutinized the importance of nonlinear dendrites within neural networks. By employing machine-learning methodologies, we assessed the impact of dendritic structure nonlinearity on neural network performance. Our findings reveal that integrating dendritic structures can substantially enhance model capacity and performance while keeping signal communication costs effectively restrained. This investigation offers pivotal insights that hold considerable implications for the development of future neur
    
[^7]: 基于定点树的广义随机森林加速

    Accelerating Generalized Random Forests with Fixed-Point Trees. (arXiv:2306.11908v1 [stat.ML])

    [http://arxiv.org/abs/2306.11908](http://arxiv.org/abs/2306.11908)

    本文提出一种新的树生长规则，使广义随机森林在无梯度优化的情况下大大节省了时间。

    

    广义随机森林建立在传统随机森林的基础上，通过将其作为自适应核加权算法来构建估算器，并通过基于梯度的树生长过程来实现。我们提出了一种新的树生长规则，基于定点迭代近似表示梯度近似，实现了无梯度优化，并为此开发了渐近理论。这有效地节省了时间，尤其是在目标量的维度适中时。

    Generalized random forests arXiv:1610.01271 build upon the well-established success of conventional forests (Breiman, 2001) to offer a flexible and powerful non-parametric method for estimating local solutions of heterogeneous estimating equations. Estimators are constructed by leveraging random forests as an adaptive kernel weighting algorithm and implemented through a gradient-based tree-growing procedure. By expressing this gradient-based approximation as being induced from a single Newton-Raphson root-finding iteration, and drawing upon the connection between estimating equations and fixed-point problems arXiv:2110.11074, we propose a new tree-growing rule for generalized random forests induced from a fixed-point iteration type of approximation, enabling gradient-free optimization, and yielding substantial time savings for tasks involving even modest dimensionality of the target quantity (e.g. multiple/multi-level treatment effects). We develop an asymptotic theory for estimators o
    
[^8]: 通过内在维度将隐性偏见和对抗性攻击相关联

    Relating Implicit Bias and Adversarial Attacks through Intrinsic Dimension. (arXiv:2305.15203v1 [cs.LG])

    [http://arxiv.org/abs/2305.15203](http://arxiv.org/abs/2305.15203)

    本文通过研究神经网络的隐性偏差，着眼于其中涉及的傅里叶频率与图像分类和对抗性攻击之间的关系。研究提出了一种新方法，可以发现这些频率之间的非线性相关性。

    

    尽管神经网络在分类方面表现出色，但众所周知它们易受对抗性攻击的影响。这些攻击是针对模型的输入数据进行的小干扰，旨在欺骗模型。自然而然的问题是，模型的结构、设置或属性与攻击的性质之间可能存在潜在联系。在本文中，我们旨在通过关注神经网络的隐性偏差来解决这个问题，这指的是其固有倾向于支持特定模式或结果。具体而言，我们研究了隐性偏差的一个方面，其中包括进行准确图像分类所需的基本傅里叶频率。我们进行测试以评估这些频率与成功攻击所需的频率之间的统计关系。为了深入探讨这种关系，我们提出了一种新的方法，可以揭示坐标集之间的非线性相关性，在我们的情况下，这些坐标集就是前述的傅里叶频率。

    Despite their impressive performance in classification, neural networks are known to be vulnerable to adversarial attacks. These attacks are small perturbations of the input data designed to fool the model. Naturally, a question arises regarding the potential connection between the architecture, settings, or properties of the model and the nature of the attack. In this work, we aim to shed light on this problem by focusing on the implicit bias of the neural network, which refers to its inherent inclination to favor specific patterns or outcomes. Specifically, we investigate one aspect of the implicit bias, which involves the essential Fourier frequencies required for accurate image classification. We conduct tests to assess the statistical relationship between these frequencies and those necessary for a successful attack. To delve into this relationship, we propose a new method that can uncover non-linear correlations between sets of coordinates, which, in our case, are the aforementio
    
[^9]: 自监督的孪生自编码器

    Self-Supervised Siamese Autoencoders. (arXiv:2304.02549v1 [cs.LG])

    [http://arxiv.org/abs/2304.02549](http://arxiv.org/abs/2304.02549)

    本论文提出了一种新的自监督方法，名为SidAE，它结合了孪生架构和去噪自编码器的优点，可以更好地提取输入数据的特征，以在多个下游任务中获得更好的性能。

    

    完全监督的模型通常需要大量的标记训练数据，这往往是昂贵且难以获得的。相反，自监督表示学习减少了实现相同或更高下游性能所需的标记数据量。目标是在自监督任务上预先训练深度神经网络，以便网络能够从原始输入数据中提取有意义的特征。然后，将这些特征用作下游任务（如图像分类）中的输入。在先前的研究中，自编码器和孪生网络（如SimSiam）已成功应用于这些任务中。然而，仍然存在一些挑战，例如将特征的特性（例如，细节级别）与给定的任务和数据集匹配。在本文中，我们提出了一种结合了孪生架构和去噪自编码器优势的新自监督方法。我们展示了我们的模型，名为SidAE（孪生去噪自编码器），在多个下游任务上胜过了两个自监督最新基准。

    Fully supervised models often require large amounts of labeled training data, which tends to be costly and hard to acquire. In contrast, self-supervised representation learning reduces the amount of labeled data needed for achieving the same or even higher downstream performance. The goal is to pre-train deep neural networks on a self-supervised task such that afterwards the networks are able to extract meaningful features from raw input data. These features are then used as inputs in downstream tasks, such as image classification. Previously, autoencoders and Siamese networks such as SimSiam have been successfully employed in those tasks. Yet, challenges remain, such as matching characteristics of the features (e.g., level of detail) to the given task and data set. In this paper, we present a new self-supervised method that combines the benefits of Siamese architectures and denoising autoencoders. We show that our model, called SidAE (Siamese denoising autoencoder), outperforms two se
    
[^10]: 对正则化中的偏差进行惩罚将使稀疏化

    Penalising the biases in norm regularisation enforces sparsity. (arXiv:2303.01353v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2303.01353](http://arxiv.org/abs/2303.01353)

    本研究表明，控制神经网络参数的范数可以获得良好的泛化性能。对神经网络中偏差项的范数进行惩罚可以实现稀疏估计量。

    

    当训练神经网络时，通过控制参数的范数往往可以获得良好的泛化性能。然而，正则化参数的范数和所得估计量之间的关系在理论上尚未完全理解。本文针对具有单一隐藏层和一维数据的神经网络，展示了表示函数所需的参数范数由其二阶导数的总变差加权得到，其中所加权的因子为$\sqrt{1+x^2}$。值得注意的是，当不对偏差项的范数进行正则化时，这个加权因子会消失。这个额外的加权因子的存在非常重要，因为它被证明可以强制实现最小范数内插器的唯一性和稀疏性（在拐点数量上）。相反，省略偏差的范数则会导致非稀疏解。因此，在正则化中对偏差项进行惩罚，无论是显式还是隐式地，都会导致稀疏估计量。

    Controlling the parameters' norm often yields good generalisation when training neural networks. Beyond simple intuitions, the relation between regularising parameters' norm and obtained estimators remains theoretically misunderstood. For one hidden ReLU layer networks with unidimensional data, this work shows the parameters' norm required to represent a function is given by the total variation of its second derivative, weighted by a $\sqrt{1+x^2}$ factor. Notably, this weighting factor disappears when the norm of bias terms is not regularised. The presence of this additional weighting factor is of utmost significance as it is shown to enforce the uniqueness and sparsity (in the number of kinks) of the minimal norm interpolator. Conversely, omitting the bias' norm allows for non-sparse solutions. Penalising the bias terms in the regularisation, either explicitly or implicitly, thus leads to sparse estimators.
    
[^11]: 可扩展的完整协方差动态混合模型用于概率交通预测

    Scalable Dynamic Mixture Model with Full Covariance for Probabilistic Traffic Forecasting. (arXiv:2212.06653v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2212.06653](http://arxiv.org/abs/2212.06653)

    本文提出了一种扩展性强、适用于复杂概率交通预测的动态混合模型，通过模拟复杂的时变分布以更准确预测交通情况，具有高效性、灵活性和可扩展性。

    

    基于深度学习的多元多步骤交通预测模型通常在序列到序列的设置中使用均方误差（MSE）或平均绝对误差（MAE）作为损失函数，简单地假设误差遵循独立且各向同性的高斯或拉普拉斯分布。然而，在现实世界的交通预测任务中，这样的假设往往是不切实际的，因为时空预测的概率分布非常复杂，并且在时间上存在强烈的同时相关性，涉及传感器和预测时间跨度。在本文中，我们将矩阵变量误差过程的时变分布建模为零均值高斯分布的动态混合模型。为了实现高效性、灵活性和可扩展性，我们使用矩阵正态分布参数化每个混合成分，并允许混合权重随时间变化和可预测。所提出的方法可以无缝集成

    Deep learning-based multivariate and multistep-ahead traffic forecasting models are typically trained with the mean squared error (MSE) or mean absolute error (MAE) as the loss function in a sequence-to-sequence setting, simply assuming that the errors follow an independent and isotropic Gaussian or Laplacian distributions. However, such assumptions are often unrealistic for real-world traffic forecasting tasks, where the probabilistic distribution of spatiotemporal forecasting is very complex with strong concurrent correlations across both sensors and forecasting horizons in a time-varying manner. In this paper, we model the time-varying distribution for the matrix-variate error process as a dynamic mixture of zero-mean Gaussian distributions. To achieve efficiency, flexibility, and scalability, we parameterize each mixture component using a matrix normal distribution and allow the mixture weight to change and be predictable over time. The proposed method can be seamlessly integrated 
    

