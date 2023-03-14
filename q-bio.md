# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Prefix-tree Decoding for Predicting Mass Spectra from Molecules.](http://arxiv.org/abs/2303.06470) | 本文提出了一种基于前缀树的中间策略，通过将质谱视为化学公式的集合来预测分子的质谱，克服了化学子公式的组合可能性。 |
| [^2] | [Learning interpretable causal networks from very large datasets, application to 400,000 medical records of breast cancer patients.](http://arxiv.org/abs/2303.06423) | 本文提出了一种更可靠和可扩展的因果发现方法（iMIIC），并在来自美国监测、流行病学和终末结果计划的396,179名乳腺癌患者的医疗保健数据上展示了其独特能力。超过90％的预测因果效应是正确的，而其余的意外直接和间接因果效应可以解释为诊断程序、治疗时间、患者偏好或社会经济差距。 |
| [^3] | [Intelligent diagnostic scheme for lung cancer screening with Raman spectra data by tensor network machine learning.](http://arxiv.org/abs/2303.06340) | 本文提出了一种基于张量网络机器学习的方案，通过筛查呼出气中挥发性有机化合物（VOC）的Raman光谱数据，可可靠地预测肺癌患者及其阶段。 |
| [^4] | [Enhancing Protein Language Models with Structure-based Encoder and Pre-training.](http://arxiv.org/abs/2303.06275) | 本文提出了一种结合基于结构的编码器和预训练的蛋白质语言模型，以明确地编码蛋白质结构，获得更好的结构感知蛋白质表示，并在实验中验证了其有效性。 |
| [^5] | [Resource saving taxonomy classification with k-mer distributions and machine learning.](http://arxiv.org/abs/2303.06154) | 本文提出了一种基于k-mer分布和机器学习的分类法，可以节约资源并提高分类器的性能。 |
| [^6] | [Learning Topology-Specific Experts for Molecular Property Prediction.](http://arxiv.org/abs/2302.13693) | 本文提出了TopExpert，利用拓扑特定的预测模型（称为专家），每个专家负责每个共享类似拓扑语义的分子组，以提高分子属性预测的性能。 |
| [^7] | [Sources of Richness and Ineffability for Phenomenally Conscious States.](http://arxiv.org/abs/2302.06403) | 本文提供了一个信息论动力系统的视角，来解释意识的丰富性和难以言说性。在我们的框架中，意识体验的丰富性对应于意识状态中的信息量，而难以言说性则对应于不同处理阶段丢失的信息量。 |
| [^8] | [Towards NeuroAI: Introducing Neuronal Diversity into Artificial Neural Networks.](http://arxiv.org/abs/2301.09245) | 引入神经元多样性可以解决人工神经网络的基本问题，走向神经人工智能。 |
| [^9] | [A attention way in Explainable methods for infant brain.](http://arxiv.org/abs/2301.00815) | 本文提出了一种可解释的几何深度网络，通过端到端学习解释因素以增强区分性表示提取，以反向保证细粒度的可解释性，适用于神经影像和神经科学研究中的高维数据。 |

# 详细

[^1]: 基于前缀树的分子质谱预测

    Prefix-tree Decoding for Predicting Mass Spectra from Molecules. (arXiv:2303.06470v1 [q-bio.QM])

    [http://arxiv.org/abs/2303.06470](http://arxiv.org/abs/2303.06470)

    本文提出了一种基于前缀树的中间策略，通过将质谱视为化学公式的集合来预测分子的质谱，克服了化学子公式的组合可能性。

    This paper proposes an intermediate strategy for predicting mass spectra from molecules by treating mass spectra as sets of chemical formulae, which are themselves multisets of atoms, and decoding the formula set using a prefix tree structure, atom-type by atom-type, overcoming the combinatorial possibilities for chemical subformulae.

    计算预测分子的质谱已经实现了临床相关代谢物的发现。然而，这样的预测工具仍然存在局限性，因为它们占据了两个极端，要么通过过度刚性的约束和较差的时间复杂度组合分子来进行操作，要么通过解码有损和非物理离散化的光谱向量来进行操作。在这项工作中，我们介绍了一种新的中间策略，通过将质谱视为化学公式的集合来预测分子的质谱，这些化学公式本身是原子的多重集合。在首先对输入分子图进行编码后，我们解码一组化学子公式，每个化学子公式指定质谱中的一个预测峰，其强度由第二个模型预测。我们的关键洞察力是通过使用前缀树结构，逐个原子类型地解码公式集，克服了化学子公式的组合可能性。

    Computational predictions of mass spectra from molecules have enabled the discovery of clinically relevant metabolites. However, such predictive tools are still limited as they occupy one of two extremes, either operating (a) by fragmenting molecules combinatorially with overly rigid constraints on potential rearrangements and poor time complexity or (b) by decoding lossy and nonphysical discretized spectra vectors. In this work, we introduce a new intermediate strategy for predicting mass spectra from molecules by treating mass spectra as sets of chemical formulae, which are themselves multisets of atoms. After first encoding an input molecular graph, we decode a set of chemical subformulae, each of which specify a predicted peak in the mass spectra, the intensities of which are predicted by a second model. Our key insight is to overcome the combinatorial possibilities for chemical subformulae by decoding the formula set using a prefix tree structure, atom-type by atom-type, represent
    
[^2]: 从大型数据集中学习可解释的因果网络，以乳腺癌患者的40万份医疗记录为例

    Learning interpretable causal networks from very large datasets, application to 400,000 medical records of breast cancer patients. (arXiv:2303.06423v1 [q-bio.QM])

    [http://arxiv.org/abs/2303.06423](http://arxiv.org/abs/2303.06423)

    本文提出了一种更可靠和可扩展的因果发现方法（iMIIC），并在来自美国监测、流行病学和终末结果计划的396,179名乳腺癌患者的医疗保健数据上展示了其独特能力。超过90％的预测因果效应是正确的，而其余的意外直接和间接因果效应可以解释为诊断程序、治疗时间、患者偏好或社会经济差距。

    This paper proposes a more reliable and scalable causal discovery method (iMIIC) and showcases its unique capabilities on healthcare data from 396,179 breast cancer patients from the US Surveillance, Epidemiology, and End Results program. Over 90% of predicted causal effects appear correct, while the remaining unexpected direct and indirect causal effects can be interpreted in terms of diagnostic procedures, therapeutic timing, patient preference or socio-economic disparity.

    发现因果效应是科学研究的核心，但当只有观察数据可用时，这仍然具有挑战性。在实践中，因果网络难以学习和解释，并且仅限于相对较小的数据集。我们报告了一种更可靠和可扩展的因果发现方法（iMIIC），基于一般的互信息最大原则，它极大地提高了推断的因果关系的精度，同时区分了真正的原因和假定的和潜在的因果效应。我们展示了iMIIC在来自美国监测、流行病学和终末结果计划的396,179名乳腺癌患者的合成和现实医疗保健数据上的独特能力。超过90％的预测因果效应是正确的，而其余的意外直接和间接因果效应可以解释为诊断程序、治疗时间、患者偏好或社会经济差距。iMIIC的独特能力开辟了发现可靠和可解释的因果网络的新途径。

    Discovering causal effects is at the core of scientific investigation but remains challenging when only observational data is available. In practice, causal networks are difficult to learn and interpret, and limited to relatively small datasets. We report a more reliable and scalable causal discovery method (iMIIC), based on a general mutual information supremum principle, which greatly improves the precision of inferred causal relations while distinguishing genuine causes from putative and latent causal effects. We showcase iMIIC on synthetic and real-life healthcare data from 396,179 breast cancer patients from the US Surveillance, Epidemiology, and End Results program. More than 90\% of predicted causal effects appear correct, while the remaining unexpected direct and indirect causal effects can be interpreted in terms of diagnostic procedures, therapeutic timing, patient preference or socio-economic disparity. iMIIC's unique capabilities open up new avenues to discover reliable and
    
[^3]: 基于张量网络机器学习的Raman光谱数据肺癌智能诊断方案

    Intelligent diagnostic scheme for lung cancer screening with Raman spectra data by tensor network machine learning. (arXiv:2303.06340v1 [q-bio.QM])

    [http://arxiv.org/abs/2303.06340](http://arxiv.org/abs/2303.06340)

    本文提出了一种基于张量网络机器学习的方案，通过筛查呼出气中挥发性有机化合物（VOC）的Raman光谱数据，可可靠地预测肺癌患者及其阶段。

    This paper proposes a tensor-network machine learning method to reliably predict lung cancer patients and their stages via screening Raman spectra data of Volatile organic compounds (VOCs) in exhaled breath.

    人工智能（AI）已经在生物医学科学中带来了巨大的影响，从学术研究到临床应用，例如生物标志物的检测和诊断、治疗优化以及药物发现中新的治疗靶点的识别。然而，当代AI技术，特别是深度机器学习（ML），严重受到非可解释性的影响，这可能会不可控地导致错误的预测。对于ML的可解释性尤其重要，因为消费者必须从坚实的基础或令人信服的解释中获得必要的安全感和信任感。在这项工作中，我们提出了一种基于张量网络（TN）-ML方法的方案，通过筛查呼出气中挥发性有机化合物（VOC）的Raman光谱数据，可可靠地预测肺癌患者及其阶段，这些数据通常适用于生物标志物，并被认为是非侵入性肺癌筛查的理想方式。TN-ML的预测基于

    Artificial intelligence (AI) has brought tremendous impacts on biomedical sciences from academic researches to clinical applications, such as in biomarkers' detection and diagnosis, optimization of treatment, and identification of new therapeutic targets in drug discovery. However, the contemporary AI technologies, particularly deep machine learning (ML), severely suffer from non-interpretability, which might uncontrollably lead to incorrect predictions. Interpretability is particularly crucial to ML for clinical diagnosis as the consumers must gain necessary sense of security and trust from firm grounds or convincing interpretations. In this work, we propose a tensor-network (TN)-ML method to reliably predict lung cancer patients and their stages via screening Raman spectra data of Volatile organic compounds (VOCs) in exhaled breath, which are generally suitable as biomarkers and are considered to be an ideal way for non-invasive lung cancer screening. The prediction of TN-ML is based
    
[^4]: 结合基于结构的编码器和预训练的蛋白质语言模型的增强

    Enhancing Protein Language Models with Structure-based Encoder and Pre-training. (arXiv:2303.06275v1 [q-bio.QM])

    [http://arxiv.org/abs/2303.06275](http://arxiv.org/abs/2303.06275)

    本文提出了一种结合基于结构的编码器和预训练的蛋白质语言模型，以明确地编码蛋白质结构，获得更好的结构感知蛋白质表示，并在实验中验证了其有效性。

    This paper proposes enhancing protein language models with structure-based encoder and pre-training to explicitly encode protein structures for better structure-aware protein representations, and empirically verifies its effectiveness.

    在大规模蛋白质序列语料库上预训练的蛋白质语言模型（PLMs）在各种下游蛋白质理解任务中取得了令人印象深刻的表现。尽管能够隐式地捕获残基间的接触信息，但基于变压器的PLMs不能明确地编码蛋白质结构，以获得更好的结构感知蛋白质表示。此外，尽管结构对于确定功能很重要，但尚未探索在可用蛋白质结构上进行预训练以改进这些PLMs的能力。为了解决这些限制，我们在本文中使用基于结构的编码器和预训练来增强PLMs。

    Protein language models (PLMs) pre-trained on large-scale protein sequence corpora have achieved impressive performance on various downstream protein understanding tasks. Despite the ability to implicitly capture inter-residue contact information, transformer-based PLMs cannot encode protein structures explicitly for better structure-aware protein representations. Besides, the power of pre-training on available protein structures has not been explored for improving these PLMs, though structures are important to determine functions. To tackle these limitations, in this work, we enhance the PLMs with structure-based encoder and pre-training. We first explore feasible model architectures to combine the advantages of a state-of-the-art PLM (i.e., ESM-1b1) and a state-of-the-art protein structure encoder (i.e., GearNet). We empirically verify the ESM-GearNet that connects two encoders in a series way as the most effective combination model. To further improve the effectiveness of ESM-GearNe
    
[^5]: 基于k-mer分布和机器学习的资源节约分类法

    Resource saving taxonomy classification with k-mer distributions and machine learning. (arXiv:2303.06154v1 [q-bio.GN])

    [http://arxiv.org/abs/2303.06154](http://arxiv.org/abs/2303.06154)

    本文提出了一种基于k-mer分布和机器学习的分类法，可以节约资源并提高分类器的性能。

    This paper proposes a resource-saving classification method based on k-mer distributions and machine learning, which can improve the performance of classifiers and reduce the consumption of energy.

    现代高通量测序技术（如宏基因组测序）生成数百万个序列，需要根据它们的分类级别进行分类。现代方法要么应用本地比对和与现有数据集（如MMseqs2）的比较，要么使用深度神经网络（如DeepMicrobes和BERTax）。基于比对的方法在运行时间方面成本高，特别是由于数据库变得越来越大。对于基于深度学习的方法，需要专门的硬件进行计算，这消耗大量能源。在本文中，我们建议使用从DNA中获得的k-mer分布作为特征，使用机器学习方法（如子空间k最近邻算法、神经网络或袋装决策树）来分类其分类起源。此外，我们提出了一种特征空间数据集平衡方法，允许减少训练数据集并提高分类器的性能。通过比较性能，我们证明了我们的方法比现有方法更有效。

    Modern high throughput sequencing technologies like metagenomic sequencing generate millions of sequences which have to be classified based on their taxonomic rank. Modern approaches either apply local alignment and comparison to existing data sets like MMseqs2 or use deep neural networks as it is done in DeepMicrobes and BERTax. Alignment-based approaches are costly in terms of runtime, especially since databases get larger and larger. For the deep learning-based approaches, specialized hardware is necessary for a computation, which consumes large amounts of energy. In this paper, we propose to use $k$-mer distributions obtained from DNA as features to classify its taxonomic origin using machine learning approaches like the subspace $k$-nearest neighbors algorithm, neural networks or bagged decision trees. In addition, we propose a feature space data set balancing approach, which allows reducing the data set for training and improves the performance of the classifiers. By comparing pe
    
[^6]: 学习分子属性预测的拓扑专家

    Learning Topology-Specific Experts for Molecular Property Prediction. (arXiv:2302.13693v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.13693](http://arxiv.org/abs/2302.13693)

    本文提出了TopExpert，利用拓扑特定的预测模型（称为专家），每个专家负责每个共享类似拓扑语义的分子组，以提高分子属性预测的性能。

    This paper proposes TopExpert, which leverages topology-specific prediction models (referred to as experts) to improve the performance of molecular property prediction by assigning each expert to a molecular group sharing similar topological semantics.

    最近，图神经网络（GNN）已成功应用于预测分子属性，这是一种具有各种应用的最经典的化学信息学任务。尽管它们很有效，但我们经验性地观察到，为具有不同结构模式的多种分子训练单个GNN模型会限制其预测性能。因此，我们提出了TopExpert，利用拓扑特定的预测模型（称为专家），每个专家负责每个共享类似拓扑语义的分子组。也就是说，每个专家在与其相应的拓扑组一起训练时学习特定于拓扑的判别特征。为了解决按其拓扑模式对分子进行分组的关键挑战，我们引入了基于聚类的门控模块，将输入分子分配到其中一个聚类中，并使用两种不同类型的自监督进一步优化门控模块。

    Recently, graph neural networks (GNNs) have been successfully applied to predicting molecular properties, which is one of the most classical cheminformatics tasks with various applications. Despite their effectiveness, we empirically observe that training a single GNN model for diverse molecules with distinct structural patterns limits its prediction performance. In this paper, motivated by this observation, we propose TopExpert to leverage topology-specific prediction models (referred to as experts), each of which is responsible for each molecular group sharing similar topological semantics. That is, each expert learns topology-specific discriminative features while being trained with its corresponding topological group. To tackle the key challenge of grouping molecules by their topological patterns, we introduce a clustering-based gating module that assigns an input molecule into one of the clusters and further optimizes the gating module with two different types of self-supervision:
    
[^7]: 现象意识状态的丰富性和难以言说性的来源

    Sources of Richness and Ineffability for Phenomenally Conscious States. (arXiv:2302.06403v3 [q-bio.NC] UPDATED)

    [http://arxiv.org/abs/2302.06403](http://arxiv.org/abs/2302.06403)

    本文提供了一个信息论动力系统的视角，来解释意识的丰富性和难以言说性。在我们的框架中，意识体验的丰富性对应于意识状态中的信息量，而难以言说性则对应于不同处理阶段丢失的信息量。

    This paper provides an information theoretic dynamical systems perspective on the richness and ineffability of consciousness. In their framework, the richness of conscious experience corresponds to the amount of information in a conscious state and ineffability corresponds to the amount of information lost at different stages of processing.

    意识状态（即有某种感受的状态）似乎既丰富又充满细节，又难以完全描述或回忆。特别是难以言说性的问题是哲学上长期存在的问题，部分激发了解释鸿沟的信念：意识不能归结为基础物理过程。在这里，我们提供了一个信息论动力系统的视角，来解释意识的丰富性和难以言说性。在我们的框架中，意识体验的丰富性对应于意识状态中的信息量，而难以言说性则对应于不同处理阶段丢失的信息量。我们描述了工作记忆中的吸引子动力学如何导致我们原始体验的贫乏回忆，语言的离散符号性质不足以描述体验的丰富和高维结构，以及认知功能相似性如何影响体验的共享和交流。

    Conscious states (states that there is something it is like to be in) seem both rich or full of detail, and ineffable or hard to fully describe or recall. The problem of ineffability, in particular, is a longstanding issue in philosophy that partly motivates the explanatory gap: the belief that consciousness cannot be reduced to underlying physical processes. Here, we provide an information theoretic dynamical systems perspective on the richness and ineffability of consciousness. In our framework, the richness of conscious experience corresponds to the amount of information in a conscious state and ineffability corresponds to the amount of information lost at different stages of processing. We describe how attractor dynamics in working memory would induce impoverished recollections of our original experiences, how the discrete symbolic nature of language is insufficient for describing the rich and high-dimensional structure of experiences, and how similarity in the cognitive function o
    
[^8]: 走向神经人工智能：将神经元多样性引入人工神经网络

    Towards NeuroAI: Introducing Neuronal Diversity into Artificial Neural Networks. (arXiv:2301.09245v2 [cs.NE] UPDATED)

    [http://arxiv.org/abs/2301.09245](http://arxiv.org/abs/2301.09245)

    引入神经元多样性可以解决人工神经网络的基本问题，走向神经人工智能。

    Introducing neuronal diversity can solve the fundamental problems of artificial neural networks and lead to NeuroAI.

    在整个历史上，人工智能的发展，特别是人工神经网络，一直对越来越深入的大脑理解持开放态度并不断受到启发，例如卷积神经网络的开创性工作neocognitron的启发。根据新兴领域神经人工智能的动机，大量的神经科学知识可以通过赋予网络更强大的能力来催化下一代人工智能的发展。我们知道，人类大脑有许多形态和功能不同的神经元，而人工神经网络几乎完全建立在单一神经元类型上。在人类大脑中，神经元多样性是各种生物智能行为的一个启动因素。由于人工网络是人类大脑的缩影，引入神经元多样性应该有助于解决人工网络的诸如效率、解释性等基本问题。

    Throughout history, the development of artificial intelligence, particularly artificial neural networks, has been open to and constantly inspired by the increasingly deepened understanding of the brain, such as the inspiration of neocognitron, which is the pioneering work of convolutional neural networks. Per the motives of the emerging field: NeuroAI, a great amount of neuroscience knowledge can help catalyze the next generation of AI by endowing a network with more powerful capabilities. As we know, the human brain has numerous morphologically and functionally different neurons, while artificial neural networks are almost exclusively built on a single neuron type. In the human brain, neuronal diversity is an enabling factor for all kinds of biological intelligent behaviors. Since an artificial network is a miniature of the human brain, introducing neuronal diversity should be valuable in terms of addressing those essential problems of artificial networks such as efficiency, interpret
    
[^9]: 一种用于婴儿脑可解释方法的注意力机制

    A attention way in Explainable methods for infant brain. (arXiv:2301.00815v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.00815](http://arxiv.org/abs/2301.00815)

    本文提出了一种可解释的几何深度网络，通过端到端学习解释因素以增强区分性表示提取，以反向保证细粒度的可解释性，适用于神经影像和神经科学研究中的高维数据。

    This paper proposes an explainable geometric deep network that enhances discriminative representation extraction by end-to-end learning of explanation factors, which is a more intuitive strategy to inversely assure fine-grained explainability, suitable for high-dimensional data in neuroimaging and neuroscience studies containing noisy, redundant, and task-irrelevant information.

    在跨学科应用中部署可靠的深度学习技术需要学习模型输出准确且（更重要的是）可解释的预测。现有方法通常以事后方式解释网络输出，隐含地假设忠实的解释来自准确的预测/分类。我们提出相反的观点，即解释提升（甚至决定）分类。也就是说，端到端学习解释因素以增强区分性表示提取可能是一种更直观的策略，以反向保证细粒度的可解释性，例如在那些包含噪声，冗余和任务无关信息的高维数据的神经影像和神经科学研究中。在本文中，我们提出了一种可解释的几何深度网络。

    Deploying reliable deep learning techniques in interdisciplinary applications needs learned models to output accurate and ({even more importantly}) explainable predictions. Existing approaches typically explicate network outputs in a post-hoc fashion, under an implicit assumption that faithful explanations come from accurate predictions/classifications. We have an opposite claim that explanations boost (or even determine) classification. That is, end-to-end learning of explanation factors to augment discriminative representation extraction could be a more intuitive strategy to inversely assure fine-grained explainability, e.g., in those neuroimaging and neuroscience studies with high-dimensional data containing noisy, redundant, and task-irrelevant information. In this paper, we propose such an explainable geometric deep network dubbed.
    

