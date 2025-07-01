# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Deep Support Vectors](https://arxiv.org/abs/2403.17329) | 该论文探索了深度学习模型中的深度支持向量（DSVs）的概念，介绍了DeepKKT条件，通过实证研究发现DSVs与SVM中的支持向量类似，为解释模型决策标准提供了方法，同时证明了可以有效地使用DSVs重构模型。 |
| [^2] | [Probing the Robustness of Time-series Forecasting Models with CounterfacTS](https://arxiv.org/abs/2403.03508) | 提出了CounterfacTS工具，通过反事实探究深度学习模型在时间序列预测中的鲁棒性。 |
| [^3] | [An Autonomous Large Language Model Agent for Chemical Literature Data Mining](https://arxiv.org/abs/2402.12993) | 介绍了一个端到端的人工智能代理框架，利用大型语言模型实现从化学文献中高保真提取信息，充当化学助手的角色，自动化数据收集和分析，从而提高工作效率。 |
| [^4] | [Low-Rank Graph Contrastive Learning for Node Classification](https://arxiv.org/abs/2402.09600) | 本研究提出了一种新颖且鲁棒的低秩图对比学习（LR-GCL）算法，应用于转导节点分类任务。该算法通过低秩正规化的对比学习训练一个编码器，并使用生成的特征进行线性转导分类。 |
| [^5] | [Parameter-Efficient Fine-Tuning for Pre-Trained Vision Models: A Survey](https://arxiv.org/abs/2402.02242) | 本综述调研了面向预训练视觉模型的参数高效微调方法，通过最小参数修改超越全面微调的性能，提供了全面的概述和未来方向，并提供了丰富的资源收藏。 |
| [^6] | [Benchmarking Spiking Neural Network Learning Methods with Varying Locality](https://arxiv.org/abs/2402.01782) | 本研究使用不同局部性对脉冲神经网络学习方法进行基准测试，并发现这些方法在性能和生物学合理性之间存在权衡。此外，研究还探讨了SNN的隐式循环特性。 |
| [^7] | [CPT: Competence-progressive Training Strategy for Few-shot Node Classification](https://arxiv.org/abs/2402.00450) | CPT是一种新颖的两阶段课程学习方法，弥补了传统元学习方法在少样本节点分类上的困难。它使用能力递进的训练策略来提高元学习器的效果和稳定性。 |
| [^8] | [Improving Robustness and Reliability in Medical Image Classification with Latent-Guided Diffusion and Nested-Ensembles.](http://arxiv.org/abs/2310.15952) | 本文引入了一种新颖的三阶段方法，通过变换器和条件扩散模型来改善医学图像分类模型对实际应用中常见成像变异性的鲁棒性。 |
| [^9] | [Adversarial Robustness Unhardening via Backdoor Attacks in Federated Learning.](http://arxiv.org/abs/2310.11594) | 本文研究了联邦学习中对抗性训练和后门攻击的交叉点，引入了Adversarial Robustness Unhardening（ARU），通过有意介入分散式训练过程中破坏模型的鲁棒性，使模型更容易受到更广泛的逃避攻击。 |
| [^10] | [Rethinking Fairness for Human-AI Collaboration.](http://arxiv.org/abs/2310.03647) | 在人工智能与人类合作中，需要重新思考公平性，因为完全遵守算法决策很少是现实可行的，因此我们需要设计稳健公平的算法推荐来提升公平性。 |
| [^11] | [Value-Compressed Sparse Column (VCSC): Sparse Matrix Storage for Redundant Data.](http://arxiv.org/abs/2309.04355) | 值压缩的稀疏列（VCSC）是一种新的稀疏矩阵存储格式，能够利用高冗余性将数据进一步压缩，并在性能上没有显著的负面影响。通过增量编码和字节打包压缩索引数组，IVCSC实现了更大的存储空间节省。 |
| [^12] | [Scaling Data-Constrained Language Models.](http://arxiv.org/abs/2305.16264) | 研究人员研究了在数据受限制的情况下缩放语言模型，并提出了一个计算最优性的缩放定律，考虑到重复令牌和过量参数的价值递减。 |
| [^13] | [CBAGAN-RRT: Convolutional Block Attention Generative Adversarial Network for Sampling-Based Path Planning.](http://arxiv.org/abs/2305.10442) | 本文介绍了一种基于图像处理学习算法（CBAGAN-RRT）的路径规划方法，使用卷积块注意力生成对抗网络和一种新的损失函数，找到更优的最佳路径并提高算法的收敛速度，与先前的最先进算法相比，在图像质量生成指标和路径规划指标方面都表现更优。 |
| [^14] | [Bayes correlated equilibria and no-regret dynamics.](http://arxiv.org/abs/2304.05005) | 本文提出了贝叶斯相关均衡的一个概念，可以以分布式方式有效地计算和实现，并在广泛的博弈类别中达到近似最优的社会福利，在实验中验证了其有效性。 |

# 详细

[^1]: 深度支持向量

    Deep Support Vectors

    [https://arxiv.org/abs/2403.17329](https://arxiv.org/abs/2403.17329)

    该论文探索了深度学习模型中的深度支持向量（DSVs）的概念，介绍了DeepKKT条件，通过实证研究发现DSVs与SVM中的支持向量类似，为解释模型决策标准提供了方法，同时证明了可以有效地使用DSVs重构模型。

    

    尽管深度学习的成功通常被归因于其与支持向量机（SVM）在理论上的等价性，但这种关系的实际影响尚未得到全面探讨。本文在这一领域开展了一项探索，重点关注深度学习模型中深度支持向量（DSVs）的识别。我们引入了DeepKKT条件的概念，这是一种专为深度学习量身定制的传统Karush-Kuhn-Tucker（KKT）条件的调整版本。通过实证研究，我们阐明了DSVs与SVM中的支持向量之间存在相似性，提供了一种解释模型决策标准的切实方法。此外，我们的研究结果表明，可以有效地使用DSVs重构模型，类似于SVM中的过程。代码将会公开。

    arXiv:2403.17329v1 Announce Type: cross  Abstract: While the success of deep learning is commonly attributed to its theoretical equivalence with Support Vector Machines (SVM), the practical implications of this relationship have not been thoroughly explored. This paper pioneers an exploration in this domain, specifically focusing on the identification of Deep Support Vectors (DSVs) within deep learning models. We introduce the concept of DeepKKT conditions, an adaptation of the traditional Karush-Kuhn-Tucker (KKT) conditions tailored for deep learning. Through empirical investigations, we illustrate that DSVs exhibit similarities to support vectors in SVM, offering a tangible method to interpret the decision-making criteria of models. Additionally, our findings demonstrate that models can be effectively reconstructed using DSVs, resembling the process in SVM. The code will be available.
    
[^2]: 探究使用CounterfacTS探究时间序列预测模型的鲁棒性

    Probing the Robustness of Time-series Forecasting Models with CounterfacTS

    [https://arxiv.org/abs/2403.03508](https://arxiv.org/abs/2403.03508)

    提出了CounterfacTS工具，通过反事实探究深度学习模型在时间序列预测中的鲁棒性。

    

    机器学习模型应用于时间序列预测时面临的一个常见问题是数据分布的时间演化（即概念漂移）。由于大多数训练数据没有反映这些变化，模型在新的分布场景下表现出很差，因此，此类事件的影响事前无法可靠地预测。我们提出并公开发布CounterfacTS，这是一个通过反事实探究深度学习模型在时间序列预测任务中鲁棒性的工具。CounterfacTS具有用户友好的界面，允许用户可视化、比较和量化时间序列数据及其预测结果，适用于多个数据集和深度学习模型。此外，用户可以对时间序列应用各种变换，并以可解释的方式探索预测结果的变化。通过示例案例，我们演示了CounterfacTS如何用于：

    arXiv:2403.03508v1 Announce Type: new  Abstract: A common issue for machine learning models applied to time-series forecasting is the temporal evolution of the data distributions (i.e., concept drift). Because most of the training data does not reflect such changes, the models present poor performance on the new out-of-distribution scenarios and, therefore, the impact of such events cannot be reliably anticipated ahead of time. We present and publicly release CounterfacTS, a tool to probe the robustness of deep learning models in time-series forecasting tasks via counterfactuals. CounterfacTS has a user-friendly interface that allows the user to visualize, compare and quantify time series data and their forecasts, for a number of datasets and deep learning models. Furthermore, the user can apply various transformations to the time series and explore the resulting changes in the forecasts in an interpretable manner. Through example cases, we illustrate how CounterfacTS can be used to i)
    
[^3]: 用于化学文献数据挖掘的自主大型语言模型代理

    An Autonomous Large Language Model Agent for Chemical Literature Data Mining

    [https://arxiv.org/abs/2402.12993](https://arxiv.org/abs/2402.12993)

    介绍了一个端到端的人工智能代理框架，利用大型语言模型实现从化学文献中高保真提取信息，充当化学助手的角色，自动化数据收集和分析，从而提高工作效率。

    

    化学合成对于推动材料合成和药物发现至关重要，影响着包括环境科学和医疗保健在内的各个领域。化学领域的技术上升使得产生了大量的化学数据，挑战研究人员去识别模式并细化合成过程。人工智能通过分析数据来优化合成并提高产量。然而，人工智能在处理文献数据方面面临着挑战，因为化学文献的结构不规整，写作风格多样。为了克服这些困难，我们引入了一个端到端的人工智能代理框架，能够从广泛的化学文献中高保真地提取信息。这个人工智能代理采用大型语言模型（LLMs）进行快速生成和迭代优化。它充当化学助手的角色，自动化数据收集和分析，从而节省人力并提高性能。

    arXiv:2402.12993v1 Announce Type: cross  Abstract: Chemical synthesis, which is crucial for advancing material synthesis and drug discovery, impacts various sectors including environmental science and healthcare. The rise of technology in chemistry has generated extensive chemical data, challenging researchers to discern patterns and refine synthesis processes. Artificial intelligence (AI) helps by analyzing data to optimize synthesis and increase yields. However, AI faces challenges in processing literature data due to the unstructured format and diverse writing style of chemical literature. To overcome these difficulties, we introduce an end-to-end AI agent framework capable of high-fidelity extraction from extensive chemical literature. This AI agent employs large language models (LLMs) for prompt generation and iterative optimization. It functions as a chemistry assistant, automating data collection and analysis, thereby saving manpower and enhancing performance. Our framework's ef
    
[^4]: 低秩图对比学习用于节点分类

    Low-Rank Graph Contrastive Learning for Node Classification

    [https://arxiv.org/abs/2402.09600](https://arxiv.org/abs/2402.09600)

    本研究提出了一种新颖且鲁棒的低秩图对比学习（LR-GCL）算法，应用于转导节点分类任务。该算法通过低秩正规化的对比学习训练一个编码器，并使用生成的特征进行线性转导分类。

    

    图神经网络（GNNs）广泛应用于学习节点表示，并在节点分类等各种任务中表现出色。然而，最近的研究表明，在现实世界的图数据中不可避免地存在噪声，这会严重降低GNNs的性能。在本文中，我们提出了一种新颖且鲁棒的GNN编码器，即低秩图对比学习（LR-GCL）。我们的方法通过两个步骤进行转导节点分类。首先，通过低秩正常对比学习训练一个名为LR-GCL的低秩GCL编码器。然后，使用LR-GCL生成的特征，使用线性转导分类算法对图中的未标记节点进行分类。我们的LR-GCL受到图数据和其标签的低频性质的启示，并在理论上受到我们关于转导学习的尖锐泛化界限的推动。

    arXiv:2402.09600v1 Announce Type: new  Abstract: Graph Neural Networks (GNNs) have been widely used to learn node representations and with outstanding performance on various tasks such as node classification. However, noise, which inevitably exists in real-world graph data, would considerably degrade the performance of GNNs revealed by recent studies. In this work, we propose a novel and robust GNN encoder, Low-Rank Graph Contrastive Learning (LR-GCL). Our method performs transductive node classification in two steps. First, a low-rank GCL encoder named LR-GCL is trained by prototypical contrastive learning with low-rank regularization. Next, using the features produced by LR-GCL, a linear transductive classification algorithm is used to classify the unlabeled nodes in the graph. Our LR-GCL is inspired by the low frequency property of the graph data and its labels, and it is also theoretically motivated by our sharp generalization bound for transductive learning. To the best of our kno
    
[^5]: 面向预训练视觉模型的参数高效微调：一项综述

    Parameter-Efficient Fine-Tuning for Pre-Trained Vision Models: A Survey

    [https://arxiv.org/abs/2402.02242](https://arxiv.org/abs/2402.02242)

    本综述调研了面向预训练视觉模型的参数高效微调方法，通过最小参数修改超越全面微调的性能，提供了全面的概述和未来方向，并提供了丰富的资源收藏。

    

    大规模预训练的视觉模型（PVMs）展示了在各种下游视觉任务中的适应能力潜力。然而，随着最先进的PVMs达到数十亿甚至数万亿个参数，标准的全面微调范式由于高计算和存储需求变得不可持续。作为响应，研究人员正在探索参数高效微调（PEFT），旨在以最小参数修改超越全面微调的性能。本综述提供了视觉PEFT的全面概述和未来方向，对最新进展进行了系统审查。首先，我们提供了PEFT的正式定义，并讨论了模型预训练方法。然后，我们将现有方法分为三类：基于添加的、基于部分的和基于统一的。最后，我们介绍了常用的数据集和应用，并提出了潜在的未来研究挑战。该综述还提供了丰富的资源收藏。

    Large-scale pre-trained vision models (PVMs) have shown great potential for adaptability across various downstream vision tasks. However, with state-of-the-art PVMs growing to billions or even trillions of parameters, the standard full fine-tuning paradigm is becoming unsustainable due to high computational and storage demands. In response, researchers are exploring parameter-efficient fine-tuning (PEFT), which seeks to exceed the performance of full fine-tuning with minimal parameter modifications. This survey provides a comprehensive overview and future directions for visual PEFT, offering a systematic review of the latest advancements. First, we provide a formal definition of PEFT and discuss model pre-training methods. We then categorize existing methods into three categories: addition-based, partial-based, and unified-based. Finally, we introduce the commonly used datasets and applications and suggest potential future research challenges. A comprehensive collection of resources is
    
[^6]: 使用不同局部性对脉冲神经网络学习方法进行基准测试

    Benchmarking Spiking Neural Network Learning Methods with Varying Locality

    [https://arxiv.org/abs/2402.01782](https://arxiv.org/abs/2402.01782)

    本研究使用不同局部性对脉冲神经网络学习方法进行基准测试，并发现这些方法在性能和生物学合理性之间存在权衡。此外，研究还探讨了SNN的隐式循环特性。

    

    脉冲神经网络（SNN）提供更真实的神经动力学，在多个机器学习任务中已经显示出与人工神经网络（ANN）相当的性能。信息在SNN中以脉冲形式进行处理，采用事件驱动机制，显著降低了能源消耗。然而，由于脉冲机制的非可微性，训练SNN具有挑战性。传统方法如时间反向传播（BPTT）已经显示出一定的效果，但在计算和存储成本方面存在问题，并且在生物学上不可行。相反，最近的研究提出了具有不同局部性的替代学习方法，在分类任务中取得了成功。本文表明，这些方法在训练过程中有相似之处，同时在生物学合理性和性能之间存在权衡。此外，本研究还探讨了SNN的隐式循环特性，并进行了调查。

    Spiking Neural Networks (SNNs), providing more realistic neuronal dynamics, have shown to achieve performance comparable to Artificial Neural Networks (ANNs) in several machine learning tasks. Information is processed as spikes within SNNs in an event-based mechanism that significantly reduces energy consumption. However, training SNNs is challenging due to the non-differentiable nature of the spiking mechanism. Traditional approaches, such as Backpropagation Through Time (BPTT), have shown effectiveness but comes with additional computational and memory costs and are biologically implausible. In contrast, recent works propose alternative learning methods with varying degrees of locality, demonstrating success in classification tasks. In this work, we show that these methods share similarities during the training process, while they present a trade-off between biological plausibility and performance. Further, this research examines the implicitly recurrent nature of SNNs and investigat
    
[^7]: CPT: 应用于少样本节点分类的能 力递进式训练策略

    CPT: Competence-progressive Training Strategy for Few-shot Node Classification

    [https://arxiv.org/abs/2402.00450](https://arxiv.org/abs/2402.00450)

    CPT是一种新颖的两阶段课程学习方法，弥补了传统元学习方法在少样本节点分类上的困难。它使用能力递进的训练策略来提高元学习器的效果和稳定性。

    

    图神经网络（GNNs）在节点分类方面取得了显著的进展，但其成功仍然依赖于训练数据中每个类别有足够的标记节点。现实世界中的图数据通常呈现出长尾分布，标签稀疏，强调了GNN在少样本节点分类中的重要性，即使用有限的数据对节点进行分类。传统的情节元学习方法在这个领域显示出了潜力，但它们面临着固有的限制：随机和均匀任务分配可能导致模型收敛到次优解，忽视了任务的难度水平。这可能导致元学习器过早地面临复杂任务，阻碍了正常的学习。理想情况下，元学习器应该从简单概念开始，逐渐进入更复杂的概念，就像人类学习一样。因此，我们引入了CPT，一种新颖的两阶段课程学习方法，将任务难度与元学习器的递进能力相匹配，增强了元学习的效果和稳定性。

    Graph Neural Networks (GNNs) have made significant advancements in node classification, but their success relies on sufficient labeled nodes per class in the training data. Real-world graph data often exhibits a long-tail distribution with sparse labels, emphasizing the importance of GNNs' ability in few-shot node classification, which entails categorizing nodes with limited data. Traditional episodic meta-learning approaches have shown promise in this domain, but they face an inherent limitation: it might lead the model to converge to suboptimal solutions because of random and uniform task assignment, ignoring task difficulty levels. This could lead the meta-learner to face complex tasks too soon, hindering proper learning. Ideally, the meta-learner should start with simple concepts and advance to more complex ones, like human learning. So, we introduce CPT, a novel two-stage curriculum learning method that aligns task difficulty with the meta-learner's progressive competence, enhanci
    
[^8]: 通过潜在引导扩散和嵌套集成改进医学图像分类的鲁棒性和可靠性

    Improving Robustness and Reliability in Medical Image Classification with Latent-Guided Diffusion and Nested-Ensembles. (arXiv:2310.15952v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2310.15952](http://arxiv.org/abs/2310.15952)

    本文引入了一种新颖的三阶段方法，通过变换器和条件扩散模型来改善医学图像分类模型对实际应用中常见成像变异性的鲁棒性。

    

    尽管深度学习模型在各种医学图像分析任务中取得了显著的成功，但在真实临床环境中部署这些模型需要它们对所获取的图像的变异性具有鲁棒性。许多方法会对训练数据应用预定义的转换，以增强测试时的鲁棒性，但这些转换可能无法确保模型对患者图像中的多样性变异性具有鲁棒性。在本文中，我们提出了一种基于变换器和条件扩散模型的新型三阶段方法，旨在提高模型对实践中常见的成像变异性的鲁棒性，而无需预先确定的数据增强策略。为了实现这一目标，多个图像编码器首先学习分层特征表示来构建辨别潜在空间。接下来，一个由潜在代码引导的逆扩散过程作用于有信息先验，并提出预测候选。

    While deep learning models have achieved remarkable success across a range of medical image analysis tasks, deployment of these models in real clinical contexts requires that they be robust to variability in the acquired images. While many methods apply predefined transformations to augment the training data to enhance test-time robustness, these transformations may not ensure the model's robustness to the diverse variability seen in patient images. In this paper, we introduce a novel three-stage approach based on transformers coupled with conditional diffusion models, with the goal of improving model robustness to the kinds of imaging variability commonly encountered in practice without the need for pre-determined data augmentation strategies. To this end, multiple image encoders first learn hierarchical feature representations to build discriminative latent spaces. Next, a reverse diffusion process, guided by the latent code, acts on an informative prior and proposes prediction candi
    
[^9]: Adversarial Robustness Unhardening via Backdoor Attacks in Federated Learning. (arXiv:2310.11594v1 [cs.LG])

    Adversarial Robustness Unhardening via Backdoor Attacks in Federated Learning. (arXiv:2310.11594v1 [cs.LG])

    [http://arxiv.org/abs/2310.11594](http://arxiv.org/abs/2310.11594)

    本文研究了联邦学习中对抗性训练和后门攻击的交叉点，引入了Adversarial Robustness Unhardening（ARU），通过有意介入分散式训练过程中破坏模型的鲁棒性，使模型更容易受到更广泛的逃避攻击。

    

    在当今的数据驱动环境中，维护用户隐私和释放数据潜力之间微妙的平衡成为一个重要关注点。联邦学习是一种以隐私为中心的解决方案，它实现了协作模型训练而无需共享数据。这种分散式方法带来了安全挑战，特别是恶意实体注入损坏数据的中毒和后门攻击。我们的研究最初受到测试时间逃避攻击的启发，探讨了联邦学习中对抗性训练和后门攻击的交叉点，引入了Adversarial Robustness Unhardening（ARU）。ARU被一部分对手使用，以有意介入分散式训练过程中破坏模型的鲁棒性，使模型更容易受到更广泛的逃避攻击。我们进行了广泛的实证实验，评估了ARU对对抗性训练和现有的鲁棒聚合防御策略对中毒和后门攻击的影响。

    In today's data-driven landscape, the delicate equilibrium between safeguarding user privacy and unleashing data potential stands as a paramount concern. Federated learning, which enables collaborative model training without necessitating data sharing, has emerged as a privacy-centric solution. This decentralized approach brings forth security challenges, notably poisoning and backdoor attacks where malicious entities inject corrupted data. Our research, initially spurred by test-time evasion attacks, investigates the intersection of adversarial training and backdoor attacks within federated learning, introducing Adversarial Robustness Unhardening (ARU). ARU is employed by a subset of adversaries to intentionally undermine model robustness during decentralized training, rendering models susceptible to a broader range of evasion attacks. We present extensive empirical experiments evaluating ARU's impact on adversarial training and existing robust aggregation defenses against poisoning a
    
[^10]: 重新思考人工智能与人类合作的公平性

    Rethinking Fairness for Human-AI Collaboration. (arXiv:2310.03647v1 [cs.LG])

    [http://arxiv.org/abs/2310.03647](http://arxiv.org/abs/2310.03647)

    在人工智能与人类合作中，需要重新思考公平性，因为完全遵守算法决策很少是现实可行的，因此我们需要设计稳健公平的算法推荐来提升公平性。

    

    现有的算法公平性方法旨在确保人类决策者完全遵守算法决策时实现公平的结果。然而，在人工智能与人类合作中，完全遵守算法决策很少是现实或理想的结果。然而，最近的研究表明，对公平算法的选择性遵守会相对于人类以前的政策增加歧视。因此，确保公平结果需要基本不同的算法设计原则，以确保对决策者（事先不知道）的遵守模式具有稳健性。我们定义了一种遵守稳健公平的算法推荐，无论人类的遵守模式如何，它们都能确保在决策中改善公平性（弱形意义上）。我们提出了一种简单的优化策略来确定最佳的性能改进遵守稳健公平策略。然而，我们发现设计算法推荐可能是不可行的。

    Existing approaches to algorithmic fairness aim to ensure equitable outcomes if human decision-makers comply perfectly with algorithmic decisions. However, perfect compliance with the algorithm is rarely a reality or even a desirable outcome in human-AI collaboration. Yet, recent studies have shown that selective compliance with fair algorithms can amplify discrimination relative to the prior human policy. As a consequence, ensuring equitable outcomes requires fundamentally different algorithmic design principles that ensure robustness to the decision-maker's (a priori unknown) compliance pattern. We define the notion of compliance-robustly fair algorithmic recommendations that are guaranteed to (weakly) improve fairness in decisions, regardless of the human's compliance pattern. We propose a simple optimization strategy to identify the best performance-improving compliance-robustly fair policy. However, we show that it may be infeasible to design algorithmic recommendations that are s
    
[^11]: 值压缩的稀疏列（VCSC）：冗余数据的稀疏矩阵存储

    Value-Compressed Sparse Column (VCSC): Sparse Matrix Storage for Redundant Data. (arXiv:2309.04355v1 [cs.DS])

    [http://arxiv.org/abs/2309.04355](http://arxiv.org/abs/2309.04355)

    值压缩的稀疏列（VCSC）是一种新的稀疏矩阵存储格式，能够利用高冗余性将数据进一步压缩，并在性能上没有显著的负面影响。通过增量编码和字节打包压缩索引数组，IVCSC实现了更大的存储空间节省。

    

    压缩的稀疏列（CSC）和坐标（COO）是稀疏矩阵的常用压缩格式。然而，CSC和COO都是通用格式，不能利用除稀疏性以外的数据特性，如数据冗余性。高度冗余的稀疏数据在许多机器学习应用中很常见，例如基因组学，在传统的稀疏存储格式下，这些数据通常太大无法进行内存计算。本文中，我们提出了两个扩展的CSC格式：值压缩的稀疏列（VCSC）和索引和值压缩的稀疏列（IVCSC）。VCSC利用列内的高冗余性，将数据进一步压缩了3倍以上，相比COO压缩了2.25倍，而性能特征没有显著的负面影响。IVCSC通过增量编码和字节打包压缩索引数组，使内存使用量比COO减少了10倍，比CSC减少了7.5倍。

    Compressed Sparse Column (CSC) and Coordinate (COO) are popular compression formats for sparse matrices. However, both CSC and COO are general purpose and cannot take advantage of any of the properties of the data other than sparsity, such as data redundancy. Highly redundant sparse data is common in many machine learning applications, such as genomics, and is often too large for in-core computation using conventional sparse storage formats. In this paper, we present two extensions to CSC: (1) Value-Compressed Sparse Column (VCSC) and (2) Index- and Value-Compressed Sparse Column (IVCSC). VCSC takes advantage of high redundancy within a column to further compress data up to 3-fold over COO and 2.25-fold over CSC, without significant negative impact to performance characteristics. IVCSC extends VCSC by compressing index arrays through delta encoding and byte-packing, achieving a 10-fold decrease in memory usage over COO and 7.5-fold decrease over CSC. Our benchmarks on simulated and rea
    
[^12]: 缩放数据受限的语言模型

    Scaling Data-Constrained Language Models. (arXiv:2305.16264v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.16264](http://arxiv.org/abs/2305.16264)

    研究人员研究了在数据受限制的情况下缩放语言模型，并提出了一个计算最优性的缩放定律，考虑到重复令牌和过量参数的价值递减。

    

    现在扩展语言模型的趋势涉及增加参数计数和训练数据集大小。推断这个趋势表明，训练数据集大小可能很快就会受到互联网上可用文本数据的限制。出于此限制的动机，我们研究在数据受限制的情况下缩放语言模型。具体而言，我们运行了大量的实验，变化数据重复程度和计算预算，范围达到了9000亿个训练令牌和9亿参数模型。我们发现，在有限的数据的情况下，使用高达4次重复数据的训练与使用唯一数据相比对损失的贡献微不足道。然而，使用更多的重复数据，添加计算的价值最终会衰减为零。我们提出并经验证了一个计算最优性的缩放定律，考虑到重复令牌和过量参数的价值递减。最后，我们尝试了缓解数据稀缺的方法。

    The current trend of scaling language models involves increasing both parameter count and training dataset size. Extrapolating this trend suggests that training dataset size may soon be limited by the amount of text data available on the internet. Motivated by this limit, we investigate scaling language models in data-constrained regimes. Specifically, we run a large set of experiments varying the extent of data repetition and compute budget, ranging up to 900 billion training tokens and 9 billion parameter models. We find that with constrained data for a fixed compute budget, training with up to 4 epochs of repeated data yields negligible changes to loss compared to having unique data. However, with more repetition, the value of adding compute eventually decays to zero. We propose and empirically validate a scaling law for compute optimality that accounts for the decreasing value of repeated tokens and excess parameters. Finally, we experiment with approaches mitigating data scarcity,
    
[^13]: CBAGAN-RRT: 卷积块注意力生成对抗网络用于基于采样的路径规划

    CBAGAN-RRT: Convolutional Block Attention Generative Adversarial Network for Sampling-Based Path Planning. (arXiv:2305.10442v1 [cs.RO])

    [http://arxiv.org/abs/2305.10442](http://arxiv.org/abs/2305.10442)

    本文介绍了一种基于图像处理学习算法（CBAGAN-RRT）的路径规划方法，使用卷积块注意力生成对抗网络和一种新的损失函数，找到更优的最佳路径并提高算法的收敛速度，与先前的最先进算法相比，在图像质量生成指标和路径规划指标方面都表现更优。

    

    基于采样的路径规划算法在自主机器人中发挥着重要作用。但是，基于RRT算法的一个常见问题是生成的初始路径不是最优的，而且收敛速度过慢，无法应用于实际场景。本文提出了一种使用卷积块注意力生成对抗网络和一种新的损失函数的图像处理学习算法（CBAGAN-RRT），以设计启发式算法，找到更优的最佳路径，并提高算法的收敛速度。我们的GAN模型生成的路径概率分布用于引导RRT算法的采样过程。我们在由 \cite {zhang2021generative} 生成的数据集上进行了网络的训练和测试，并证明了我们的算法在图像质量生成指标（如IOU分数，Dice分数）和路径规划指标（如路径长度和成功率）方面均优于先前的最先进算法。

    Sampling-based path planning algorithms play an important role in autonomous robotics. However, a common problem among the RRT-based algorithms is that the initial path generated is not optimal and the convergence is too slow to be used in real-world applications. In this paper, we propose a novel image-based learning algorithm (CBAGAN-RRT) using a Convolutional Block Attention Generative Adversarial Network with a combination of spatial and channel attention and a novel loss function to design the heuristics, find a better optimal path, and improve the convergence of the algorithm both concerning time and speed. The probability distribution of the paths generated from our GAN model is used to guide the sampling process for the RRT algorithm. We train and test our network on the dataset generated by \cite{zhang2021generative} and demonstrate that our algorithm outperforms the previous state-of-the-art algorithms using both the image quality generation metrics like IOU Score, Dice Score
    
[^14]: 贝叶斯相关均衡和无悔动态

    Bayes correlated equilibria and no-regret dynamics. (arXiv:2304.05005v1 [cs.GT])

    [http://arxiv.org/abs/2304.05005](http://arxiv.org/abs/2304.05005)

    本文提出了贝叶斯相关均衡的一个概念，可以以分布式方式有效地计算和实现，并在广泛的博弈类别中达到近似最优的社会福利，在实验中验证了其有效性。

    

    本文研究了贝叶斯博弈的均衡概念，这是一种具有不完全信息的基本博弈模型。我们旨在实现三种理想的平衡性质。首先，通过在博弈中引入调解者来自然地实现均衡。其次，可以以分布式方式有效地计算均衡。第三，对于广泛的博弈类别，该类均衡近似地最大化社会福利，即通过灾变代价来度量。这三种属性允许玩家计算均衡，并通过调解者使其实现，从而在近乎最优的社会福利中达成稳定状态。我们的主要结果是存在一个均衡概念，满足这三种性质。为此，我们描述了各种（不等价的）相关均衡扩展，统称为贝叶斯相关均衡。特别是，我们关注促进玩家之间交流的沟通均衡（也称为协调机制）。然后，我们提出了一种无悔动态，以分布式方式收敛于贝叶斯相关均衡。最后，我们展示了所提出的平衡概念满足三种理想的平衡性质，并呈现了证明其有效性的实验结果。

    This paper explores equilibrium concepts for Bayesian games, which are fundamental models of games with incomplete information. We aim at three desirable properties of equilibria. First, equilibria can be naturally realized by introducing a mediator into games. Second, an equilibrium can be computed efficiently in a distributed fashion. Third, any equilibrium in that class approximately maximizes social welfare, as measured by the price of anarchy, for a broad class of games. These three properties allow players to compute an equilibrium and realize it via a mediator, thereby settling into a stable state with approximately optimal social welfare. Our main result is the existence of an equilibrium concept that satisfies these three properties.  Toward this goal, we characterize various (non-equivalent) extensions of correlated equilibria, collectively known as Bayes correlated equilibria. In particular, we focus on communication equilibria (also known as coordination mechanisms), which 
    

