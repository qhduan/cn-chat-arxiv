# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Don't Half-listen: Capturing Key-part Information in Continual Instruction Tuning](https://arxiv.org/abs/2403.10056) | 提出了一种基于关键部分信息增益的新型连续指导调整方法，通过动态重放数据和优化训练目标，使LLMs能够捕捉任务感知信息和减轻过度拟合。 |
| [^2] | [RadCLIP: Enhancing Radiologic Image Analysis through Contrastive Language-Image Pre-training](https://arxiv.org/abs/2403.09948) | RadCLIP是一种创新的跨模态基础模型，利用对比语言图像预训练以改进放射学图像分析，包含针对体积图像分析定制的新颖3D切片池化机制，并使用丰富多样的放射学图像-文本对数据集进行训练。 |
| [^3] | [Soft Reasoning on Uncertain Knowledge Graphs](https://arxiv.org/abs/2403.01508) | 本文研究了在不确定知识图上进行软查询，并提出了一种基于机器学习的方法，可以有效回答大规模、不完整和不确定的知识图上的查询。 |
| [^4] | [Deep Learning for Multivariate Time Series Imputation: A Survey](https://arxiv.org/abs/2402.04059) | 本文调查了深度学习在多变量时间序列插补中的应用。通过综述不同的方法以及它们的优点和限制，研究了它们对下游任务性能的改进，并指出了未来研究的开放问题。 |
| [^5] | [Word4Per: Zero-shot Composed Person Retrieval](https://arxiv.org/abs/2311.16515) | 提出了一个新任务：组合人员检索（CPR），旨在联合利用图像和文本信息进行目标人员检索，引入零样本组合人员检索（ZS-CPR）解决了CPR问题，提出了一个两阶段学习框架Word4Per。 |
| [^6] | [Diversity-aware clustering: Computational Complexity and Approximation Algorithms.](http://arxiv.org/abs/2401.05502) | 本研究讨论了多样性感知聚类问题，在选择聚类中心时要考虑多个属性，同时最小化聚类目标。我们提出了针对不同聚类目标的参数化近似算法，这些算法在保证聚类质量的同时，具有紧确的近似比。 |
| [^7] | [Gradient Leakage Defense with Key-Lock Module for Federated Learning.](http://arxiv.org/abs/2305.04095) | 本研究提出了一种新的联邦学习梯度泄露防御技术，使用私钥锁模块保护任意模型体系结构，并可确保无法从共享的梯度中重建私有训练数据。 |

# 详细

[^1]: 不要半心半意：捕捉连续指导调整中的关键部分信息

    Don't Half-listen: Capturing Key-part Information in Continual Instruction Tuning

    [https://arxiv.org/abs/2403.10056](https://arxiv.org/abs/2403.10056)

    提出了一种基于关键部分信息增益的新型连续指导调整方法，通过动态重放数据和优化训练目标，使LLMs能够捕捉任务感知信息和减轻过度拟合。

    

    arXiv:2403.10056v1 公告类型: 跨领域 摘要：大型语言模型（LLMs）的指导调整可以驱使它们在特定下游任务中产生符合人类目标的结果。然而，LLMs的连续指导调整（CIT）过程可能会带来灾难性遗忘（CF）问题，导致先前学到的能力退化。最近的方法尝试通过修改模型或重放数据来缓解CF问题，但这可能只记住指令的表面模式并在留存任务上感到困惑。在本文中，我们提出了一种基于关键部分信息增益（KPIG）的新型连续指导调整方法。我们的方法计算掩盖部分的信息增益，动态重放数据并优化训练目标，从而使LLMs能够捕捉与正确响应相关的任务感知信息，并减轻对指导中通用描述的过度拟合。此外，我们提出了两个指标，P分和V分，

    arXiv:2403.10056v1 Announce Type: cross  Abstract: Instruction tuning for large language models (LLMs) can drive them to produce results consistent with human goals in specific downstream tasks. However, the process of continual instruction tuning (CIT) for LLMs may bring about the catastrophic forgetting (CF) problem, where previously learned abilities are degraded. Recent methods try to alleviate the CF problem by modifying models or replaying data, which may only remember the surface-level pattern of instructions and get confused on held-out tasks. In this paper, we propose a novel continual instruction tuning method based on Key-part Information Gain (KPIG). Our method computes the information gain on masked parts to dynamically replay data and refine the training objective, which enables LLMs to capture task-aware information relevant to the correct response and alleviate overfitting to general descriptions in instructions. In addition, we propose two metrics, P-score and V-score,
    
[^2]: RadCLIP: 通过对比语言图像预训练增强放射学图像分析

    RadCLIP: Enhancing Radiologic Image Analysis through Contrastive Language-Image Pre-training

    [https://arxiv.org/abs/2403.09948](https://arxiv.org/abs/2403.09948)

    RadCLIP是一种创新的跨模态基础模型，利用对比语言图像预训练以改进放射学图像分析，包含针对体积图像分析定制的新颖3D切片池化机制，并使用丰富多样的放射学图像-文本对数据集进行训练。

    

    arXiv:2403.09948v1 公告类型: 跨领域  摘要: 人工智能（AI）与放射学的整合标志着医学诊断领域的变革时代。视觉基础模型已被采用来增强放射学图像分析。然而，放射学图像的独特复杂性，包括对2D和3D放射学数据的解读，带来了现有模型无法充分应对的挑战，因为这些模型是在通用非医学图像上训练的。为了弥合这一差距，并充分利用医学成像所需的诊断精度，我们引入了RadCLIP：一种开创性的跨模态基础模型，利用对比语言图像预训练（CLIP）来改进放射学图像分析。RadCLIP包含一种新颖的3D切片池化机制，专为体积图像分析定制，使用了丰富多样的放射学图像-文本对数据集进行训练。我们的评估表明，RadCLIP能有效地对齐放射学图像

    arXiv:2403.09948v1 Announce Type: cross  Abstract: The integration of artificial intelligence (AI) with radiology has marked a transformative era in medical diagnostics. Vision foundation models have been adopted to enhance radiologic imaging analysis. However, the distinct complexities of radiological imaging, including the interpretation of 2D and 3D radiological data, pose unique challenges that existing models, trained on general non-medical images, fail to address adequately. To bridge this gap and capitalize on the diagnostic precision required in medical imaging, we introduce RadCLIP: a pioneering cross-modal foundational model that harnesses Contrastive Language-Image Pre-training (CLIP) to refine radiologic image analysis. RadCLIP incorporates a novel 3D slice pooling mechanism tailored for volumetric image analysis and is trained using a comprehensive and diverse dataset of radiologic image-text pairs. Our evaluations demonstrate that RadCLIP effectively aligns radiological i
    
[^3]: 不确定知识图上的软推理

    Soft Reasoning on Uncertain Knowledge Graphs

    [https://arxiv.org/abs/2403.01508](https://arxiv.org/abs/2403.01508)

    本文研究了在不确定知识图上进行软查询，并提出了一种基于机器学习的方法，可以有效回答大规模、不完整和不确定的知识图上的查询。

    

    通过考虑知识中的不确定性，这项研究进一步推动了基于机器学习的逻辑查询回答的研究。该论文研究了不确定知识上的软查询设置，受软约束编程的建立启发。我们提出了一种基于机器学习的方法，既具有前向推理又具有后向校准，用于回答大规模、不完整和不确定的知识图上的软查询。

    arXiv:2403.01508v1 Announce Type: new  Abstract: The study of machine learning-based logical query-answering enables reasoning with large-scale and incomplete knowledge graphs. This paper further advances this line of research by considering the uncertainty in the knowledge. The uncertain nature of knowledge is widely observed in the real world, but \textit{does not} align seamlessly with the first-order logic underpinning existing studies. To bridge this gap, we study the setting of soft queries on uncertain knowledge, which is motivated by the establishment of soft constraint programming. We further propose an ML-based approach with both forward inference and backward calibration to answer soft queries on large-scale, incomplete, and uncertain knowledge graphs. Theoretical discussions present that our methods share the same complexity as state-of-the-art inference algorithms for first-order queries. Empirical results justify the superior performance of our approach against previous M
    
[^4]: 深度学习在多变量时间序列插补中的应用：一项调查

    Deep Learning for Multivariate Time Series Imputation: A Survey

    [https://arxiv.org/abs/2402.04059](https://arxiv.org/abs/2402.04059)

    本文调查了深度学习在多变量时间序列插补中的应用。通过综述不同的方法以及它们的优点和限制，研究了它们对下游任务性能的改进，并指出了未来研究的开放问题。

    

    普遍存在的缺失值导致多变量时间序列数据部分观测，破坏了时间序列的完整性，阻碍了有效的时间序列数据分析。最近，深度学习插补方法在提高损坏的时间序列数据质量方面取得了显著的成功，进而提高了下游任务的性能。本文对最近提出的深度学习插补方法进行了全面的调查。首先，我们提出了对这些方法进行分类的方法，并通过强调它们的优点和限制来进行了结构化的综述。我们还进行了实证实验，研究了不同方法，并比较了它们对下游任务的改进。最后，我们指出了多变量时间序列插补未来研究的开放问题。本文的所有代码和配置，包括定期维护的多变量时间序列插补论文列表，可以在以下位置找到。

    The ubiquitous missing values cause the multivariate time series data to be partially observed, destroying the integrity of time series and hindering the effective time series data analysis. Recently deep learning imputation methods have demonstrated remarkable success in elevating the quality of corrupted time series data, subsequently enhancing performance in downstream tasks. In this paper, we conduct a comprehensive survey on the recently proposed deep learning imputation methods. First, we propose a taxonomy for the reviewed methods, and then provide a structured review of these methods by highlighting their strengths and limitations. We also conduct empirical experiments to study different methods and compare their enhancement for downstream tasks. Finally, the open issues for future research on multivariate time series imputation are pointed out. All code and configurations of this work, including a regularly maintained multivariate time series imputation paper list, can be foun
    
[^5]: Word4Per: Zero-shot组合人员检索

    Word4Per: Zero-shot Composed Person Retrieval

    [https://arxiv.org/abs/2311.16515](https://arxiv.org/abs/2311.16515)

    提出了一个新任务：组合人员检索（CPR），旨在联合利用图像和文本信息进行目标人员检索，引入零样本组合人员检索（ZS-CPR）解决了CPR问题，提出了一个两阶段学习框架Word4Per。

    

    寻找特定人员具有极大的社会效益和安全价值，通常涉及视觉和文本信息的结合。本文提出了一个全新的任务，称为组合人员检索（CPR），旨在联合利用图像和文本信息进行目标人员检索。然而，监督CPR需要昂贵的手动注释数据集，而目前没有可用资源。为了解决这个问题，我们首先引入了零样本组合人员检索（ZS-CPR），利用现有的领域相关数据解决了CPR问题而不需要昂贵的注释。其次，为了学习ZS-CPR模型，我们提出了一个两阶段学习框架，即Word4Per，其中包含一个轻量级的文本反转网络。

    arXiv:2311.16515v2 Announce Type: replace-cross  Abstract: Searching for specific person has great social benefits and security value, and it often involves a combination of visual and textual information. Conventional person retrieval methods, whether image-based or text-based, usually fall short in effectively harnessing both types of information, leading to the loss of accuracy. In this paper, a whole new task called Composed Person Retrieval (CPR) is proposed to jointly utilize both image and text information for target person retrieval. However, the supervised CPR requires very costly manual annotation dataset, while there are currently no available resources. To mitigate this issue, we firstly introduce the Zero-shot Composed Person Retrieval (ZS-CPR), which leverages existing domain-related data to resolve the CPR problem without expensive annotations. Secondly, to learn ZS-CPR model, we propose a two-stage learning framework, Word4Per, where a lightweight Textual Inversion Netw
    
[^6]: 多样性感知聚类：计算复杂性和近似算法

    Diversity-aware clustering: Computational Complexity and Approximation Algorithms. (arXiv:2401.05502v1 [cs.DS])

    [http://arxiv.org/abs/2401.05502](http://arxiv.org/abs/2401.05502)

    本研究讨论了多样性感知聚类问题，在选择聚类中心时要考虑多个属性，同时最小化聚类目标。我们提出了针对不同聚类目标的参数化近似算法，这些算法在保证聚类质量的同时，具有紧确的近似比。

    

    在这项工作中，我们研究了多样性感知聚类问题，其中数据点与多个属性相关联，形成交叉的组。聚类解决方案需要确保从每个组中选择最少数量的聚类中心，同时最小化聚类目标，可以是$k$-中位数，$k$-均值或$k$-供应商。我们提出了参数化近似算法，近似比分别为$1+\frac{2}{e}$，$1+\frac{8}{e}$和$3$，用于多样性感知$k$-中位数，多样性感知$k$-均值和多样性感知$k$-供应商。这些近似比在假设Gap-ETH和FPT $\neq$ W[2]的情况下是紧确的。对于公平$k$-中位数和公平$k$-均值的不相交工厂组，我们提出了参数化近似算法，近似比分别为$1+\frac{2}{e}$和$1+\frac{8}{e}$。对于具有不相交工厂组的公平$k$-供应商，我们提出了一个多项式时间近似算法，因子为$3$。

    In this work, we study diversity-aware clustering problems where the data points are associated with multiple attributes resulting in intersecting groups. A clustering solution need to ensure that a minimum number of cluster centers are chosen from each group while simultaneously minimizing the clustering objective, which can be either $k$-median, $k$-means or $k$-supplier. We present parameterized approximation algorithms with approximation ratios $1+ \frac{2}{e}$, $1+\frac{8}{e}$ and $3$ for diversity-aware $k$-median, diversity-aware $k$-means and diversity-aware $k$-supplier, respectively. The approximation ratios are tight assuming Gap-ETH and FPT $\neq$ W[2]. For fair $k$-median and fair $k$-means with disjoint faicility groups, we present parameterized approximation algorithm with approximation ratios $1+\frac{2}{e}$ and $1+\frac{8}{e}$, respectively. For fair $k$-supplier with disjoint facility groups, we present a polynomial-time approximation algorithm with factor $3$, improv
    
[^7]: 基于密钥锁模块的联邦学习梯度泄露防御

    Gradient Leakage Defense with Key-Lock Module for Federated Learning. (arXiv:2305.04095v1 [cs.LG])

    [http://arxiv.org/abs/2305.04095](http://arxiv.org/abs/2305.04095)

    本研究提出了一种新的联邦学习梯度泄露防御技术，使用私钥锁模块保护任意模型体系结构，并可确保无法从共享的梯度中重建私有训练数据。

    

    联邦学习是一种广泛采用的隐私保护机器学习方法，其中私有数据保持本地，允许安全计算和本地模型梯度与第三方参数服务器之间的交换。然而，最近的研究发现，通过共享的梯度可能会危及隐私并恢复敏感信息。本研究提供了详细的分析和对梯度泄漏问题的新视角。这些理论工作导致了一种新的梯度泄露防御技术，使用私钥锁模块保护任意模型体系结构。只有锁定的梯度被传输到参数服务器进行全局模型聚合。我们提出的学习方法对梯度泄露攻击具有抵抗力，并且所设计和训练的密钥锁模块可以确保，没有密钥锁模块的私有信息：a) 无法从共享的梯度中重建私有训练数据。

    Federated Learning (FL) is a widely adopted privacy-preserving machine learning approach where private data remains local, enabling secure computations and the exchange of local model gradients between local clients and third-party parameter servers. However, recent findings reveal that privacy may be compromised and sensitive information potentially recovered from shared gradients. In this study, we offer detailed analysis and a novel perspective on understanding the gradient leakage problem. These theoretical works lead to a new gradient leakage defense technique that secures arbitrary model architectures using a private key-lock module. Only the locked gradient is transmitted to the parameter server for global model aggregation. Our proposed learning method is resistant to gradient leakage attacks, and the key-lock module is designed and trained to ensure that, without the private information of the key-lock module: a) reconstructing private training data from the shared gradient is
    

