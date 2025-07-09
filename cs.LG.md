# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [GreeDy and CoDy: Counterfactual Explainers for Dynamic Graphs](https://arxiv.org/abs/2403.16846) | 该论文介绍了两种针对动态图的新颖反事实解释方法：GreeDy和CoDy。实验证明，CoDy在寻找重要反事实输入方面表现优异，成功率高达59%。 |
| [^2] | [Optimal Transport for Domain Adaptation through Gaussian Mixture Models](https://arxiv.org/abs/2403.13847) | 通过高斯混合模型进行域自适应的最优输运，可以实现源域和目标域混合成分之间的匹配，从而在失效诊断中取得最先进的性能。 |
| [^3] | [Electrocardiogram Instruction Tuning for Report Generation](https://arxiv.org/abs/2403.04945) | 提出了Multimodal ECG Instruction Tuning（MEIT）框架，首次尝试使用LLMs和多模态指导解决ECG报告生成问题，并在两个大规模ECG数据集上进行了广泛的实验评估其优越性。 |
| [^4] | [Permutation invariant functions: statistical tests, dimension reduction in metric entropy and estimation](https://arxiv.org/abs/2403.01671) | 本文研究了如何在多元概率分布中测试排列不变性、估计排列不变密度以及分析排列不变函数类的度量熵，比较了它们与没有排列不变性的函数类的差异。 |
| [^5] | [Federated Complex Qeury Answering](https://arxiv.org/abs/2402.14609) | 研究了在多源知识图谱上回答复杂查询的联邦式方法，解决了知识图谱中的隐私保护和答案检索的挑战 |
| [^6] | [TopCoW: Benchmarking Topology-Aware Anatomical Segmentation of the Circle of Willis (CoW) for CTA and MRA.](http://arxiv.org/abs/2312.17670) | 这项研究提出了TopCoW挑战，通过发布具有13种血管组分注释的Willis循环（CoW）数据集，并使用虚拟现实（VR）技术进行拓扑感知解剖分割，解决了手动和耗时的CoW表征问题。 |
| [^7] | [Annotating 8,000 Abdominal CT Volumes for Multi-Organ Segmentation in Three Weeks.](http://arxiv.org/abs/2305.09666) | 本文提出了一种高效方法在短时间内标记8000个腹部CT扫描中的8个器官，建立了迄今为止最大的多器官数据集。 |
| [^8] | [Do deep neural networks have an inbuilt Occam's razor?.](http://arxiv.org/abs/2304.06670) | 该研究利用基于函数先验的贝叶斯视角来研究深度神经网络（DNNs）的表现来源，结果表明DNNs之所以成功，是因为它对于具有结构的数据，具备一种内在的奥卡姆剃刀式的归纳偏差，足以抵消函数数量及复杂度的指数级增长。 |

# 详细

[^1]: GreeDy和CoDy：动态图的反事实解释器

    GreeDy and CoDy: Counterfactual Explainers for Dynamic Graphs

    [https://arxiv.org/abs/2403.16846](https://arxiv.org/abs/2403.16846)

    该论文介绍了两种针对动态图的新颖反事实解释方法：GreeDy和CoDy。实验证明，CoDy在寻找重要反事实输入方面表现优异，成功率高达59%。

    

    时间图神经网络（TGNNs）对于建模具有时间变化交互的动态图至关重要，但由于其复杂的模型结构，在可解释性方面面临重大挑战。反事实解释对于理解模型决策至关重要，它研究输入图的变化如何影响结果。本文介绍了两种新颖的 TGNNs 反事实解释方法：GreeDy（动态图的贪心解释器）和 CoDy（动态图的反事实解释器）。它们将解释视为一个搜索问题，寻找改变模型预测的输入图修改。GreeDy 使用简单的贪心方法，而 CoDy 使用复杂的蒙特卡洛树搜索算法。实验证明，两种方法都能有效生成清晰的解释。值得注意的是，CoDy 的性能优于 GreeDy 和现有的事实方法，寻找到重要的反事实输入的成功率提高了高达 59\%。这突出了 CoDy 的优势。

    arXiv:2403.16846v1 Announce Type: cross  Abstract: Temporal Graph Neural Networks (TGNNs), crucial for modeling dynamic graphs with time-varying interactions, face a significant challenge in explainability due to their complex model structure. Counterfactual explanations, crucial for understanding model decisions, examine how input graph changes affect outcomes. This paper introduces two novel counterfactual explanation methods for TGNNs: GreeDy (Greedy Explainer for Dynamic Graphs) and CoDy (Counterfactual Explainer for Dynamic Graphs). They treat explanations as a search problem, seeking input graph alterations that alter model predictions. GreeDy uses a simple, greedy approach, while CoDy employs a sophisticated Monte Carlo Tree Search algorithm. Experiments show both methods effectively generate clear explanations. Notably, CoDy outperforms GreeDy and existing factual methods, with up to 59\% higher success rate in finding significant counterfactual inputs. This highlights CoDy's p
    
[^2]: 通过高斯混合模型进行域自适应的最优输运

    Optimal Transport for Domain Adaptation through Gaussian Mixture Models

    [https://arxiv.org/abs/2403.13847](https://arxiv.org/abs/2403.13847)

    通过高斯混合模型进行域自适应的最优输运，可以实现源域和目标域混合成分之间的匹配，从而在失效诊断中取得最先进的性能。

    

    在这篇论文中，我们探讨了通过最优输运进行域自适应的方法。我们提出了一种新颖的方法，即通过高斯混合模型对数据分布进行建模。这种策略使我们能够通过等价的离散问题解决连续最优输运。最优输运解决方案为我们提供了源域和目标域混合成分之间的匹配。通过这种匹配，我们可以在域之间映射数据点，或者将标签从源域组件转移到目标域。我们在失效诊断的两个域自适应基准测试中进行了实验，结果表明我们的方法具有最先进的性能。

    arXiv:2403.13847v1 Announce Type: cross  Abstract: In this paper we explore domain adaptation through optimal transport. We propose a novel approach, where we model the data distributions through Gaussian mixture models. This strategy allows us to solve continuous optimal transport through an equivalent discrete problem. The optimal transport solution gives us a matching between source and target domain mixture components. From this matching, we can map data points between domains, or transfer the labels from the source domain components towards the target domain. We experiment with 2 domain adaptation benchmarks in fault diagnosis, showing that our methods have state-of-the-art performance.
    
[^3]: 为报告生成调优心电图指导

    Electrocardiogram Instruction Tuning for Report Generation

    [https://arxiv.org/abs/2403.04945](https://arxiv.org/abs/2403.04945)

    提出了Multimodal ECG Instruction Tuning（MEIT）框架，首次尝试使用LLMs和多模态指导解决ECG报告生成问题，并在两个大规模ECG数据集上进行了广泛的实验评估其优越性。

    

    心电图（ECG）作为心脏病情监测的主要非侵入性诊断工具，对于协助临床医生至关重要。最近的研究集中在使用ECG数据对心脏病情进行分类，但忽略了ECG报告生成，这不仅耗时，而且需要临床专业知识。为了自动化ECG报告生成并确保其多功能性，我们提出了Multimodal ECG Instruction Tuning（MEIT）框架，这是\textit{首次}尝试使用LLMs和多模态指导来解决ECG报告生成问题。为了促进未来的研究，我们建立了一个基准来评估MEIT在两个大规模ECG数据集上使用各种LLM骨干的表现。我们的方法独特地对齐了ECG信号和报告的表示，并进行了大量实验来评估MEIT与九个开源LLMs，使用了超过80万个ECG报告。MEIT的结果凸显了其优越性。

    arXiv:2403.04945v1 Announce Type: new  Abstract: Electrocardiogram (ECG) serves as the primary non-invasive diagnostic tool for cardiac conditions monitoring, are crucial in assisting clinicians. Recent studies have concentrated on classifying cardiac conditions using ECG data but have overlooked ECG report generation, which is not only time-consuming but also requires clinical expertise. To automate ECG report generation and ensure its versatility, we propose the Multimodal ECG Instruction Tuning (MEIT) framework, the \textit{first} attempt to tackle ECG report generation with LLMs and multimodal instructions. To facilitate future research, we establish a benchmark to evaluate MEIT with various LLMs backbones across two large-scale ECG datasets. Our approach uniquely aligns the representations of the ECG signal and the report, and we conduct extensive experiments to benchmark MEIT with nine open source LLMs, using more than 800,000 ECG reports. MEIT's results underscore the superior p
    
[^4]: 排列不变函数：统计检验、度量熵中的降维和估计

    Permutation invariant functions: statistical tests, dimension reduction in metric entropy and estimation

    [https://arxiv.org/abs/2403.01671](https://arxiv.org/abs/2403.01671)

    本文研究了如何在多元概率分布中测试排列不变性、估计排列不变密度以及分析排列不变函数类的度量熵，比较了它们与没有排列不变性的函数类的差异。

    

    排列不变性是机器学习中可以利用来简化复杂问题的最常见的对称性之一。近年来关于构建排列不变的机器学习架构的研究活动激增。然而，在多元概率分布中的变量如何统计测试排列不变性却鲜有研究，其中样本量允许随着维数的增长。此外，在统计理论方面，关于排列不变性如何帮助估计中降维的知识甚少。本文通过研究几个基本问题，回顾并探讨这些问题：（i）测试多元分布排列不变性的假设；（ii）估计排列不变密度；（iii）分析光滑排列不变函数类的度量熵，并将其与未强加排列不变性的对应函数类进行比较。

    arXiv:2403.01671v1 Announce Type: new  Abstract: Permutation invariance is among the most common symmetry that can be exploited to simplify complex problems in machine learning (ML). There has been a tremendous surge of research activities in building permutation invariant ML architectures. However, less attention is given to how to statistically test for permutation invariance of variables in a multivariate probability distribution where the dimension is allowed to grow with the sample size. Also, in terms of a statistical theory, little is known about how permutation invariance helps with estimation in reducing dimensions. In this paper, we take a step back and examine these questions in several fundamental problems: (i) testing the assumption of permutation invariance of multivariate distributions; (ii) estimating permutation invariant densities; (iii) analyzing the metric entropy of smooth permutation invariant function classes and compare them with their counterparts without impos
    
[^5]: 联邦式复杂查询答案方法研究

    Federated Complex Qeury Answering

    [https://arxiv.org/abs/2402.14609](https://arxiv.org/abs/2402.14609)

    研究了在多源知识图谱上回答复杂查询的联邦式方法，解决了知识图谱中的隐私保护和答案检索的挑战

    

    知识图谱中的复杂逻辑查询答案是一个具有挑战性的任务，已经得到广泛研究。执行复杂逻辑推理的能力是必不可少的，并支持各种基于图推理的下游任务，比如搜索引擎。最近提出了一些方法，将知识图谱实体和逻辑查询表示为嵌入向量，并从知识图谱中找到逻辑查询的答案。然而，现有的方法主要集中在查询单个知识图谱上，并不能应用于多个图形。此外，直接共享带有敏感信息的知识图谱可能会带来隐私风险，使得共享和构建一个聚合知识图谱用于推理以检索查询答案是不切实际的。因此，目前仍然不清楚如何在多源知识图谱上回答查询。一个实体可能涉及到多个知识图谱，对多个知识图谱进行推理，并在多源知识图谱上回答复杂查询对于发现知识是重要的。

    arXiv:2402.14609v1 Announce Type: cross  Abstract: Complex logical query answering is a challenging task in knowledge graphs (KGs) that has been widely studied. The ability to perform complex logical reasoning is essential and supports various graph reasoning-based downstream tasks, such as search engines. Recent approaches are proposed to represent KG entities and logical queries into embedding vectors and find answers to logical queries from the KGs. However, existing proposed methods mainly focus on querying a single KG and cannot be applied to multiple graphs. In addition, directly sharing KGs with sensitive information may incur privacy risks, making it impractical to share and construct an aggregated KG for reasoning to retrieve query answers. Thus, it remains unknown how to answer queries on multi-source KGs. An entity can be involved in various knowledge graphs and reasoning on multiple KGs and answering complex queries on multi-source KGs is important in discovering knowledge 
    
[^6]: TopCoW：基于拓扑感知解剖分割的Willis循环（CoW）在CTA和MRA中的基准测试

    TopCoW: Benchmarking Topology-Aware Anatomical Segmentation of the Circle of Willis (CoW) for CTA and MRA. (arXiv:2312.17670v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2312.17670](http://arxiv.org/abs/2312.17670)

    这项研究提出了TopCoW挑战，通过发布具有13种血管组分注释的Willis循环（CoW）数据集，并使用虚拟现实（VR）技术进行拓扑感知解剖分割，解决了手动和耗时的CoW表征问题。

    

    Willis循环（CoW）是连接大脑主要循环的重要动脉网络。其血管结构被认为影响着严重神经血管疾病的风险、严重程度和临床结果。然而，对高度变化的CoW解剖进行表征仍然是一项需要手动和耗时的专家任务。CoW通常通过两种血管造影成像模式进行成像，即磁共振血管成像（MRA）和计算机断层血管造影（CTA），但是关于CTA的CoW解剖的公共数据集及其注释非常有限。因此，我们在2023年组织了TopCoW挑战赛，并发布了一个带有注释的CoW数据集。TopCoW数据集是第一个具有13种可能的CoW血管组分的体素级注释的公共数据集，通过虚拟现实（VR）技术实现。它也是第一个带有来自同一患者的成对MRA和CTA的大型数据集。TopCoW挑战将CoW表征问题形式化为多类问题。

    The Circle of Willis (CoW) is an important network of arteries connecting major circulations of the brain. Its vascular architecture is believed to affect the risk, severity, and clinical outcome of serious neuro-vascular diseases. However, characterizing the highly variable CoW anatomy is still a manual and time-consuming expert task. The CoW is usually imaged by two angiographic imaging modalities, magnetic resonance angiography (MRA) and computed tomography angiography (CTA), but there exist limited public datasets with annotations on CoW anatomy, especially for CTA. Therefore we organized the TopCoW Challenge in 2023 with the release of an annotated CoW dataset. The TopCoW dataset was the first public dataset with voxel-level annotations for thirteen possible CoW vessel components, enabled by virtual-reality (VR) technology. It was also the first large dataset with paired MRA and CTA from the same patients. TopCoW challenge formalized the CoW characterization problem as a multiclas
    
[^7]: 在三周内为8,000个腹部CT扫描标注多器官分割

    Annotating 8,000 Abdominal CT Volumes for Multi-Organ Segmentation in Three Weeks. (arXiv:2305.09666v1 [eess.IV])

    [http://arxiv.org/abs/2305.09666](http://arxiv.org/abs/2305.09666)

    本文提出了一种高效方法在短时间内标记8000个腹部CT扫描中的8个器官，建立了迄今为止最大的多器官数据集。

    

    医学影像标注，特别是器官分割，是费时费力的。本文提出了一种系统高效的方法来加速器官分割的标注过程。我们标注了8,448个腹部CT扫描，标记了脾脏、肝脏、肾脏、胃、胆囊、胰腺、主动脉和下腔静脉。传统的标注方法需要一位经验丰富的标注员1600周，而我们的标注方法仅用了三周。

    Annotating medical images, particularly for organ segmentation, is laborious and time-consuming. For example, annotating an abdominal organ requires an estimated rate of 30-60 minutes per CT volume based on the expertise of an annotator and the size, visibility, and complexity of the organ. Therefore, publicly available datasets for multi-organ segmentation are often limited in data size and organ diversity. This paper proposes a systematic and efficient method to expedite the annotation process for organ segmentation. We have created the largest multi-organ dataset (by far) with the spleen, liver, kidneys, stomach, gallbladder, pancreas, aorta, and IVC annotated in 8,448 CT volumes, equating to 3.2 million slices. The conventional annotation methods would take an experienced annotator up to 1,600 weeks (or roughly 30.8 years) to complete this task. In contrast, our annotation method has accomplished this task in three weeks (based on an 8-hour workday, five days a week) while maintain
    
[^8]: 深度神经网络是否具备内置的奥卡姆剃刀？

    Do deep neural networks have an inbuilt Occam's razor?. (arXiv:2304.06670v1 [cs.LG])

    [http://arxiv.org/abs/2304.06670](http://arxiv.org/abs/2304.06670)

    该研究利用基于函数先验的贝叶斯视角来研究深度神经网络（DNNs）的表现来源，结果表明DNNs之所以成功，是因为它对于具有结构的数据，具备一种内在的奥卡姆剃刀式的归纳偏差，足以抵消函数数量及复杂度的指数级增长。

    

    超参数化深度神经网络（DNNs）的卓越性能必须源自于网络架构、训练算法和数据结构之间的相互作用。为了区分这三个部分，我们应用了基于DNN所表达的函数的贝叶斯视角来进行监督学习。经过网络确定的函数先验通过利用有序和混沌状态之间的转变而变化。对于布尔函数分类，我们利用函数的误差谱在数据上进行可能性的近似。当与先验相结合时，它可以精确地预测使用随机梯度下降训练的DNN的后验概率。该分析揭示了结构化数据，以及内在的奥卡姆剃刀式归纳偏差，即足以抵消复杂度随函数数量呈指数增长而产生的影响，是DNNs成功的关键。

    The remarkable performance of overparameterized deep neural networks (DNNs) must arise from an interplay between network architecture, training algorithms, and structure in the data. To disentangle these three components, we apply a Bayesian picture, based on the functions expressed by a DNN, to supervised learning. The prior over functions is determined by the network, and is varied by exploiting a transition between ordered and chaotic regimes. For Boolean function classification, we approximate the likelihood using the error spectrum of functions on data. When combined with the prior, this accurately predicts the posterior, measured for DNNs trained with stochastic gradient descent. This analysis reveals that structured data, combined with an intrinsic Occam's razor-like inductive bias towards (Kolmogorov) simple functions that is strong enough to counteract the exponential growth of the number of functions with complexity, is a key to the success of DNNs.
    

