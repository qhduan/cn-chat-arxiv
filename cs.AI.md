# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Continuous Spiking Graph Neural Networks](https://arxiv.org/abs/2404.01897) | COS-GNN将脉冲神经网络（SNNs）与连续图神经网络（CGNNs）结合在一起，以在每个时间步骤对图节点进行表示，并将其与时间一起集成到ODE过程中，以增强信息保存和解决在离散图神经网络中的问题。 |
| [^2] | [An In-depth Evaluation of GPT-4 in Sentence Simplification with Error-based Human Assessment](https://arxiv.org/abs/2403.04963) | 本文深入评估了GPT-4在句子简化中的表现，指出现有自动评估指标和人类评估方法对于大型语言模型的适用性仍有待进一步研究。 |
| [^3] | [Exploring the Potential of Large Language Models in Artistic Creation: Collaboration and Reflection on Creative Programming](https://arxiv.org/abs/2402.09750) | 这项研究探索了大型语言模型在艺术家与人工智能合作的创意编程中的艺术潜力，并比较了两种合作方式。研究发现反思类型与用户表现、用户满意度和主观体验相关。通过实验数据和定性访谈，我们从艺术家的角度提供了人工智能合作的批判性视角和设计建议。 |
| [^4] | [Dynamic Spiking Graph Neural Networks.](http://arxiv.org/abs/2401.05373) | 本文提出了一个名为"动态尖峰图神经网络"（DSGNN）的框架，它将尖峰神经网络（SNNs）与图神经网络（GNNs）结合起来，以解决动态图表示学习中的复杂性和内存开销问题。DSGNN通过动态调整尖峰神经元的状态和连接权重，在传播过程中保持图结构信息的完整性。 |
| [^5] | [Boosting for Bounding the Worst-class Error.](http://arxiv.org/abs/2310.14890) | 该论文提出了一种基于Boosting的算法，可以保证最差类别训练误差的上界，并降低了最差类别的测试误差率。 |
| [^6] | [CoCo: A Coupled Contrastive Framework for Unsupervised Domain Adaptive Graph Classification.](http://arxiv.org/abs/2306.04979) | CoCo是一种耦合对比图表示学习框架，其中包含一个图卷积网络和一个分层图内核网络，通过耦合对比学习减少领域差异，用于无监督领域自适应图分类。 |

# 详细

[^1]: 连续脉冲图神经网络

    Continuous Spiking Graph Neural Networks

    [https://arxiv.org/abs/2404.01897](https://arxiv.org/abs/2404.01897)

    COS-GNN将脉冲神经网络（SNNs）与连续图神经网络（CGNNs）结合在一起，以在每个时间步骤对图节点进行表示，并将其与时间一起集成到ODE过程中，以增强信息保存和解决在离散图神经网络中的问题。

    

    连续图神经网络（CGNNs）因引入连续动力学而引起了极大关注，能够推广现有的离散图神经网络（GNNs）。它们通常受扩散类方法启发，引入了一种新颖的传播方案，并使用常微分方程（ODE）进行分析。然而，CGNNs的实现需要大量计算能力，这使得它们难以部署在电池供电设备上。受最近脉冲神经网络（SNNs）的启发，SNNs模拟生物推理过程并提供一种节能的神经架构，我们将SNNs与CGNNs结合到一个统一框架中，命名为连续脉冲图神经网络（COS-GNN）。我们在每个时间步骤使用SNNs进行图节点表示，这些表示进一步与时间一起集成到ODE过程中，以增强信息保存和缓解...

    arXiv:2404.01897v1 Announce Type: cross  Abstract: Continuous graph neural networks (CGNNs) have garnered significant attention due to their ability to generalize existing discrete graph neural networks (GNNs) by introducing continuous dynamics. They typically draw inspiration from diffusion-based methods to introduce a novel propagation scheme, which is analyzed using ordinary differential equations (ODE). However, the implementation of CGNNs requires significant computational power, making them challenging to deploy on battery-powered devices. Inspired by recent spiking neural networks (SNNs), which emulate a biological inference process and provide an energy-efficient neural architecture, we incorporate the SNNs with CGNNs in a unified framework, named Continuous Spiking Graph Neural Networks (COS-GNN). We employ SNNs for graph node representation at each time step, which are further integrated into the ODE process along with time. To enhance information preservation and mitigate in
    
[^2]: 在基于错误的人类评估中深入评估GPT-4在句子简化中的表现

    An In-depth Evaluation of GPT-4 in Sentence Simplification with Error-based Human Assessment

    [https://arxiv.org/abs/2403.04963](https://arxiv.org/abs/2403.04963)

    本文深入评估了GPT-4在句子简化中的表现，指出现有自动评估指标和人类评估方法对于大型语言模型的适用性仍有待进一步研究。

    

    句子简化是一种重写句子以便更易阅读和理解的方法，对于帮助有各种阅读难题的人来说是一种有前途的技术。随着先进大型语言模型（LLMs）的兴起，评估它们在句子简化中的表现变得迫在眉睫。最近的研究利用自动评估指标和人类评估来评估LLMs的简化能力。然而，现有评估方法对LLMs在简化评估中的适用性仍然存在疑问。首先，现有自动指标在LLMs的简化评估中的适用性仍不确定。其次，当前在句子简化中的人类评估方法通常陷入两个极端：要么过于肤浅，无法清晰理解模型的表现，要么过于详细，使注释过程复杂且容易出现不一致性，从而影响评估的可靠性。

    arXiv:2403.04963v1 Announce Type: cross  Abstract: Sentence simplification, which rewrites a sentence to be easier to read and understand, is a promising technique to help people with various reading difficulties. With the rise of advanced large language models (LLMs), evaluating their performance in sentence simplification has become imperative. Recent studies have used both automatic metrics and human evaluations to assess the simplification abilities of LLMs. However, the suitability of existing evaluation methodologies for LLMs remains in question. First, the suitability of current automatic metrics on LLMs' simplification evaluation is still uncertain. Second, current human evaluation approaches in sentence simplification often fall into two extremes: they are either too superficial, failing to offer a clear understanding of the models' performance, or overly detailed, making the annotation process complex and prone to inconsistency, which in turn affects the evaluation's reliabil
    
[^3]: 探索大型语言模型在艺术创作中的潜力：艺术家与人工智能合作中的创意编程和反思

    Exploring the Potential of Large Language Models in Artistic Creation: Collaboration and Reflection on Creative Programming

    [https://arxiv.org/abs/2402.09750](https://arxiv.org/abs/2402.09750)

    这项研究探索了大型语言模型在艺术家与人工智能合作的创意编程中的艺术潜力，并比较了两种合作方式。研究发现反思类型与用户表现、用户满意度和主观体验相关。通过实验数据和定性访谈，我们从艺术家的角度提供了人工智能合作的批判性视角和设计建议。

    

    最近，大型语言模型（LLMs）在辅助编程方面的潜力被广泛使用。然而，当前的研究没有探索LLMs在艺术家与人工智能合作的创造性编程中的艺术潜力。我们的工作探索了在这种合作过程中艺术家的反思类型。我们比较了两种常见的合作方式：调用整个程序和多个子任务。我们的研究结果展示了艺术家在两种不同方法中不同的刺激性反思。我们的发现还显示了反思类型与用户表现、用户满意度和主观体验之间的相关性，通过进行两种方法，包括实验数据和定性访谈。在这个意义上，我们的工作揭示了LLM在创意编程中的艺术潜力。同时，我们从艺术家的角度提供了人工智能合作的批判性视角，并阐述了设计建议。

    arXiv:2402.09750v1 Announce Type: cross  Abstract: Recently, the potential of large language models (LLMs) has been widely used in assisting programming. However, current research does not explore the artist potential of LLMs in creative coding within artist and AI collaboration. Our work probes the reflection type of artists in the creation process with such collaboration. We compare two common collaboration approaches: invoking the entire program and multiple subtasks. Our findings exhibit artists' different stimulated reflections in two different methods. Our finding also shows the correlation of reflection type with user performance, user satisfaction, and subjective experience in two collaborations through conducting two methods, including experimental data and qualitative interviews. In this sense, our work reveals the artistic potential of LLM in creative coding. Meanwhile, we provide a critical lens of human-AI collaboration from the artists' perspective and expound design sugg
    
[^4]: 动态尖峰图神经网络

    Dynamic Spiking Graph Neural Networks. (arXiv:2401.05373v1 [cs.NE])

    [http://arxiv.org/abs/2401.05373](http://arxiv.org/abs/2401.05373)

    本文提出了一个名为"动态尖峰图神经网络"（DSGNN）的框架，它将尖峰神经网络（SNNs）与图神经网络（GNNs）结合起来，以解决动态图表示学习中的复杂性和内存开销问题。DSGNN通过动态调整尖峰神经元的状态和连接权重，在传播过程中保持图结构信息的完整性。

    

    将尖峰神经网络（SNNs）和图神经网络（GNNs）相结合渐渐引起了人们的关注，这是因为它在处理由图表示的非欧几里得数据时具有低功耗和高效率。然而，作为一个常见的问题，动态图表示学习面临着高复杂性和大内存开销的挑战。目前的工作通常通过使用二进制特征而不是连续特征的SNNs来替代循环神经网络（RNNs）进行高效训练，这会忽视图结构信息并在传播过程中导致细节的丢失。此外，优化动态尖峰模型通常需要在时间步之间传播信息，这增加了内存需求。为了解决这些挑战，我们提出了一个名为"动态尖峰图神经网络"（\method{}）的框架。为了减轻信息丢失问题，\method{} 在传播过程中引入了一种新的机制，它在每个时间步骤中动态地调整尖峰神经元的状态和连接权重，以保持图结构信息的完整性。

    The integration of Spiking Neural Networks (SNNs) and Graph Neural Networks (GNNs) is gradually attracting attention due to the low power consumption and high efficiency in processing the non-Euclidean data represented by graphs. However, as a common problem, dynamic graph representation learning faces challenges such as high complexity and large memory overheads. Current work often uses SNNs instead of Recurrent Neural Networks (RNNs) by using binary features instead of continuous ones for efficient training, which would overlooks graph structure information and leads to the loss of details during propagation. Additionally, optimizing dynamic spiking models typically requires propagation of information across time steps, which increases memory requirements. To address these challenges, we present a framework named \underline{Dy}namic \underline{S}p\underline{i}king \underline{G}raph \underline{N}eural Networks (\method{}). To mitigate the information loss problem, \method{} propagates
    
[^5]: Boosting用于界定最差分类误差

    Boosting for Bounding the Worst-class Error. (arXiv:2310.14890v1 [stat.ML])

    [http://arxiv.org/abs/2310.14890](http://arxiv.org/abs/2310.14890)

    该论文提出了一种基于Boosting的算法，可以保证最差类别训练误差的上界，并降低了最差类别的测试误差率。

    

    本文解决了最差类别误差率的问题，而不是针对所有类别的标准误差率的平均。例如，一个三类别分类任务，其中各类别的误差率分别为10％，10％和40％，其最差类别误差率为40％，而在类别平衡条件下的平均误差率为20％。最差类别错误在许多应用中很重要。例如，在医学图像分类任务中，对于恶性肿瘤类别具有40％的错误率而良性和健康类别具有10％的错误率是不能被接受的。我们提出了一种保证最差类别训练误差上界的提升算法，并推导出其泛化界。实验结果表明，该算法降低了最差类别的测试误差率，同时避免了对训练集的过拟合。

    This paper tackles the problem of the worst-class error rate, instead of the standard error rate averaged over all classes. For example, a three-class classification task with class-wise error rates of 10\%, 10\%, and 40\% has a worst-class error rate of 40\%, whereas the average is 20\% under the class-balanced condition. The worst-class error is important in many applications. For example, in a medical image classification task, it would not be acceptable for the malignant tumor class to have a 40\% error rate, while the benign and healthy classes have 10\% error rates.We propose a boosting algorithm that guarantees an upper bound of the worst-class training error and derive its generalization bound. Experimental results show that the algorithm lowers worst-class test error rates while avoiding overfitting to the training set.
    
[^6]: CoCo: 一种用于无监督领域自适应图分类的耦合对比框架

    CoCo: A Coupled Contrastive Framework for Unsupervised Domain Adaptive Graph Classification. (arXiv:2306.04979v1 [cs.LG])

    [http://arxiv.org/abs/2306.04979](http://arxiv.org/abs/2306.04979)

    CoCo是一种耦合对比图表示学习框架，其中包含一个图卷积网络和一个分层图内核网络，通过耦合对比学习减少领域差异，用于无监督领域自适应图分类。

    

    虽然图神经网络在图分类中取得了显著成果，但它们通常需要大量特定任务的标签，这可能需要极大的代价来获得。一种可靠的解决方案是探索其他标注图以增强目标域的无监督学习，但如何将图神经网络应用到领域适应中仍未解决，因为对图拓扑的不充分探索以及相当大的领域偏差。本文提出了一种称为CoCo（Coupled Contrastive Graph Representation Learning）方案，该方案从耦合学习分支中提取拓扑信息，并通过耦合对比学习减少领域差异。CoCo包含一个图卷积网络分支和分层图内核网络分支，分别用隐式和显式方式探索图拓扑。此外，我们将耦合分支结合到一个全面的多视角对比学习框架中，

    Although graph neural networks (GNNs) have achieved impressive achievements in graph classification, they often need abundant task-specific labels, which could be extensively costly to acquire. A credible solution is to explore additional labeled graphs to enhance unsupervised learning on the target domain. However, how to apply GNNs to domain adaptation remains unsolved owing to the insufficient exploration of graph topology and the significant domain discrepancy. In this paper, we propose \underline{Co}upled \underline{Co}ntrastive Graph Representation Learning (\method{}), which extracts the topological information from coupled learning branches and reduces the domain discrepancy with coupled contrastive learning. \method{} contains a graph convolutional network branch and a hierarchical graph kernel network branch, which explore graph topology in implicit and explicit manners. Besides, we incorporate coupled branches into a holistic multi-view contrastive learning framework, which 
    

